import base64
import os
import uuid
from typing import Any, Dict, List, Optional

from flask import Flask, jsonify, request, send_file

from lesson_014_multimodal.images_and_styles_extractor.pdf_assets_core import export_images_to_disk, run_pipeline
from lesson_014_multimodal.images_and_styles_extractor.pdf_style_extract import extract_full_style

OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "/data")
EXPORT_BASE_DIR = os.getenv("EXPORT_BASE_DIR", "pdf_assets_files")

app = Flask(__name__)


def _strip_src(items: List[Dict[str, Any]]) -> None:
    for it in items:
        it.pop("src", None)


def _json_error(message: str, code: int = 400):
    return jsonify({"error": message}), code


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/process", methods=["POST"])
def process():
    """
    Accept PDF as multipart file (field name: `file`) or JSON with `pdf_base64`.
    Returns JSON: meta, images (hero/gallery/icons), styling (css + tokens), export (optional).
    """
    pdf_bytes: Optional[bytes] = None
    pdf_label = "upload.pdf"

    body: Dict[str, Any] = {}

    if request.content_type and "multipart/form-data" in request.content_type:
        f = request.files.get("file") or request.files.get("pdf")
        if not f or not f.filename:
            return _json_error("Missing file: use form field 'file' or 'pdf'")
        pdf_bytes = f.read()
        pdf_label = f.filename or pdf_label
    else:
        body = request.get_json(silent=True) or {}
        b64 = body.get("pdf_base64")
        path = body.get("pdf_path")
        if b64:
            try:
                pdf_bytes = base64.b64decode(b64)
            except Exception as e:
                return _json_error(f"Invalid pdf_base64: {e}")
            pdf_label = body.get("pdf_label", "input_pdf_base64")
        elif path and os.path.exists(path):
            with open(path, "rb") as fp:
                pdf_bytes = fp.read()
            pdf_label = os.path.basename(path)
        else:
            return _json_error("Send multipart file (field 'file') or JSON with pdf_base64 / pdf_path")

    use_openai = request.args.get("use_openai", "false").lower() in ("1", "true", "yes")
    export_icons = request.args.get("export_icons", "true").lower() in ("1", "true", "yes")
    save_files = request.args.get("save_files", "false").lower() in ("1", "true", "yes")
    include_src = request.args.get("include_src", "true").lower() in ("1", "true", "yes")
    include_icon_src = request.args.get("include_icon_src", "true").lower() in ("1", "true", "yes")
    openai_model = request.args.get("openai_model", "gpt-5.4")
    api_key = request.args.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

    if body:
        use_openai = bool(body.get("use_openai", use_openai))
        export_icons = bool(body.get("export_icons", export_icons))
        save_files = bool(body.get("save_files", save_files))
        include_src = bool(body.get("include_src", include_src))
        include_icon_src = bool(body.get("include_icon_src", include_icon_src))
        if body.get("openai_model"):
            openai_model = str(body["openai_model"])
        if body.get("openai_api_key"):
            api_key = str(body["openai_api_key"])

    result = run_pipeline(
        pdf_bytes=pdf_bytes,
        pdf_label=pdf_label,
        use_openai=use_openai,
        openai_api_key=api_key,
        openai_model=openai_model,
    )

    styling = extract_full_style(pdf_bytes)

    job_id = uuid.uuid4().hex
    export_info: Dict[str, Any] = {"job_id": job_id, "saved": False}

    files_base = request.url_root.rstrip("/")

    if save_files:
        out_dir = os.path.join(OUTPUT_ROOT, EXPORT_BASE_DIR, job_id)
        os.makedirs(out_dir, exist_ok=True)
        exported = export_images_to_disk(
            result=result, out_dir=out_dir, include_icons=export_icons
        )
        export_info.update(
            {
                "saved": True,
                "export_dir": os.path.join(EXPORT_BASE_DIR, job_id),
                "exported_files": exported,
                "icons_exported": export_icons,
            }
        )
        images = result.get("images", {}) or {}
        buckets = ("hero", "gallery", "icons") if export_icons else ("hero", "gallery")
        for bucket_name in buckets:
            for it in images.get(bucket_name, []) or []:
                bucket = it.get("bucket", bucket_name)
                fn = it.get("filename")
                if bucket and fn:
                    it["file_url"] = f"{files_base}/files/{job_id}/{bucket}/{fn}"

    hero = list((result.get("images", {}) or {}).get("hero", []) or [])
    gallery = list((result.get("images", {}) or {}).get("gallery", []) or [])
    icons = list((result.get("images", {}) or {}).get("icons", []) or [])

    if not include_src:
        _strip_src(hero)
        _strip_src(gallery)
    if not include_icon_src:
        _strip_src(icons)

    payload = {
        "meta": result.get("meta", {}),
        "images": {
            "hero": hero,
            "gallery": gallery,
            "icons": icons,
        },
        "styling": styling,
        "export": export_info,
    }
    return jsonify(payload)


@app.route("/files/<job_id>/<bucket>/<filename>", methods=["GET"])
def get_file(job_id: str, bucket: str, filename: str):
    if bucket not in ("hero", "gallery", "icons"):
        return _json_error("Invalid bucket", 400)
    if "/" in filename or "\\" in filename or ".." in filename:
        return _json_error("Invalid filename", 400)
    path = os.path.join(OUTPUT_ROOT, EXPORT_BASE_DIR, job_id, bucket, filename)
    if not os.path.exists(path):
        return _json_error("File not found", 404)
    return send_file(path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8000")))