import base64
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Flask, Response, jsonify, request

from lesson_014_multimodal.landing_page_server.build_landing_page import generate_landing_html


HERE = Path(__file__).resolve().parent
DEFAULT_IMAGES_DIR = HERE / "images"
DEFAULT_REFERENCE_DIR = HERE / "reference"
JOBS_DIR = Path(os.environ.get("LANDING_JOBS_DIR", "/tmp/landing_jobs")).resolve()


def _decode_data_url(data_url: str) -> Tuple[bytes, str]:
    """
    Decode data:image/...;base64,... -> (bytes, ext)
    """
    m = re.match(r"^data:(image\/[a-zA-Z0-9.+-]+);base64,(.*)$", data_url, flags=re.DOTALL)
    if not m:
        raise ValueError("Invalid data URL")
    mime = m.group(1).lower()
    b64 = m.group(2)
    raw = base64.b64decode(b64)
    ext = "png" if "png" in mime else ("jpg" if "jpeg" in mime else ("webp" if "webp" in mime else "bin"))
    return raw, ext


def _normalize_merge_input(obj: Any) -> Dict[str, Any]:
    """
    n8n sometimes sends arrays of 1 item; accept both.
    Also accept wrapper shapes { merge_json: {...} }.
    """
    if isinstance(obj, list):
        if not obj:
            return {}
        if len(obj) == 1 and isinstance(obj[0], dict):
            obj = obj[0]
        else:
            # best effort: pick first dict
            for it in obj:
                if isinstance(it, dict):
                    obj = it
                    break
    if isinstance(obj, dict) and "merge_json" in obj and isinstance(obj["merge_json"], dict):
        return obj["merge_json"]
    return obj if isinstance(obj, dict) else {}


def _materialize_images_if_present(merge: Dict[str, Any], images_dir: Path) -> Dict[str, str]:
    """
    If merge.images.* contains items with {src: "data:image/..."} we save them under images_dir and ensure fileName exists.
    Returns a map {fileName: data_url} for optional embedding.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    data_urls: Dict[str, str] = {}

    images = (merge.get("images") or {}) if isinstance(merge.get("images"), dict) else {}
    for bucket in ("hero", "gallery", "icons"):
        arr = images.get(bucket) or []
        if not isinstance(arr, list):
            continue
        for i, item in enumerate(arr):
            if not isinstance(item, dict):
                continue
            src = item.get("src")
            if not (isinstance(src, str) and src.startswith("data:image/")):
                continue
            try:
                raw, ext = _decode_data_url(src)
            except Exception:
                continue

            file_name = item.get("fileName")
            if not isinstance(file_name, str) or not file_name.strip():
                file_name = f"{bucket}_{i + 1}.{ext}"
                item["fileName"] = file_name

            out = images_dir / file_name
            try:
                out.write_bytes(raw)
                data_urls[file_name] = src
            except Exception:
                pass

    return data_urls


def _collect_data_urls_from_merge(merge: Dict[str, Any]) -> Dict[str, str]:
    """
    Collect {fileName: data_url} from merge.images.*[].src without writing anything to disk.
    """
    data_urls: Dict[str, str] = {}
    images = (merge.get("images") or {}) if isinstance(merge.get("images"), dict) else {}
    for bucket in ("hero", "gallery", "icons"):
        arr = images.get(bucket) or []
        if not isinstance(arr, list):
            continue
        for i, item in enumerate(arr):
            if not isinstance(item, dict):
                continue
            src = item.get("src")
            if not (isinstance(src, str) and src.startswith("data:image/")):
                continue

            file_name = item.get("fileName")
            if not isinstance(file_name, str) or not file_name.strip():
                # Best-effort name so we can replace placeholders later.
                try:
                    _, ext = _decode_data_url(src)
                except Exception:
                    ext = "png"
                file_name = f"{bucket}_{i + 1}.{ext}"
                item["fileName"] = file_name

            data_urls[file_name] = src

    return data_urls


def _strip_image_src_fields(merge: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a deep copy of merge and remove images.*[].src to keep the LLM prompt small.
    """
    copied: Dict[str, Any] = json.loads(json.dumps(merge, ensure_ascii=False))
    images = (copied.get("images") or {}) if isinstance(copied.get("images"), dict) else {}
    for bucket in ("hero", "gallery", "icons"):
        arr = images.get(bucket) or []
        if not isinstance(arr, list):
            continue
        for item in arr:
            if isinstance(item, dict):
                item.pop("src", None)
    return copied


def _embed_local_images(html: str, filename_to_dataurl: Dict[str, str]) -> str:
    """
    Replace src="./images/foo.png" / "images/foo.png" with data URLs (simple best effort).
    """
    if not filename_to_dataurl:
        return html

    for fn, data_url in filename_to_dataurl.items():
        # common variants
        html = html.replace(f'src="./images/{fn}"', f'src="{data_url}"')
        html = html.replace(f"src='./images/{fn}'", f"src='{data_url}'")
        html = html.replace(f'src="images/{fn}"', f'src="{data_url}"')
        html = html.replace(f"src='images/{fn}'", f"src='{data_url}'")
        # CSS url()
        html = html.replace(f'url("./images/{fn}")', f'url("{data_url}")')
        html = html.replace(f"url('./images/{fn}')", f'url("{data_url}")')
        html = html.replace(f'url("images/{fn}")', f'url("{data_url}")')
        html = html.replace(f"url('images/{fn}')", f'url("{data_url}")')

    return html


app = Flask(__name__)


@app.get("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@app.post("/build")
def build() -> Response:
    """
    Body: the Merge node JSON (object or [object]).
    Returns: HTML as downloadable attachment.
    """
    if not request.data:
        return jsonify({"error": "missing body"}), 400

    try:
        payload = request.get_json(force=True, silent=False)
    except Exception as e:
        return jsonify({"error": f"invalid json: {e}"}), 400

    merge = _normalize_merge_input(payload)
    # Keep src data-URLs for embedding AFTER generation.
    data_urls = _collect_data_urls_from_merge(merge)
    # Strip src fields before LLM to avoid context_length_exceeded.
    merge_for_llm = _strip_image_src_fields(merge)

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    reference_dir = job_dir / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    use_reference_dir = reference_dir if reference_dir.exists() else DEFAULT_REFERENCE_DIR

    try:
        # IMPORTANT:
        # - Always use the baked-in images dir for LLM visual inputs (or none), not the huge data-url payload.
        # - merge_for_llm keeps only fileName metadata, which is enough for HTML placeholders.
        html = generate_landing_html(merge=merge_for_llm, images_dir=DEFAULT_IMAGES_DIR, reference_dir=use_reference_dir)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # If we got data URLs, embed them so the downloaded HTML is self-contained.
    if data_urls:
        html = _embed_local_images(html, data_urls)

    return Response(
        html,
        mimetype="text/html; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="index.html"',
            "X-Job-Id": job_id,
        },
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)