import base64
import hashlib
import io
import os
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


# =========================
# Extraction thresholds
# =========================
MIN_EMBEDDED_IMAGE_WIDTH = 120
MIN_EMBEDDED_IMAGE_HEIGHT = 120

# Icon candidate cropping logic
PAGE_RENDER_DPI = 200
ICON_ROW_TOP_RATIO = 0.08
ICON_ROW_BOTTOM_RATIO = 0.42
ICON_MIN_CROP_WIDTH = 50
ICON_MIN_CROP_HEIGHT = 50

# Gallery limits
MAX_GALLERY_IMAGES = 6
MAX_ICON_IMAGES = 8


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clamp(n: int, low: int, high: int) -> int:
    return max(low, min(high, n))


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def to_data_url(image_bytes: bytes, fmt: str) -> str:
    fmt = fmt.lower().replace("jpeg", "jpg")
    mime = "jpeg" if fmt in ("jpg", "jpeg") else fmt
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


def decode_data_url(data_url: str) -> Tuple[bytes, str]:
    """
    Returns (bytes, ext). Supports data:image/png|jpeg|webp|gif;base64,...
    """
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise ValueError("Not a data:image/* URL")
    header, b64 = data_url.split(",", 1)
    mime = header.split(";")[0].split(":", 1)[1]  # image/png
    ext = mime.split("/", 1)[1].lower()
    if ext == "jpeg":
        ext = "jpg"
    return base64.b64decode(b64), ext


def extract_embedded_images(doc: fitz.Document) -> List[Dict[str, Any]]:
    seen_hashes = set()
    results: List[Dict[str, Any]] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)

        for img in image_list:
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                img_hash = sha256_bytes(image_bytes)
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)

                width = int(base_image.get("width", 0))
                height = int(base_image.get("height", 0))

                if width < MIN_EMBEDDED_IMAGE_WIDTH or height < MIN_EMBEDDED_IMAGE_HEIGHT:
                    continue

                # Re-encode through PIL into PNG for consistency
                with Image.open(io.BytesIO(image_bytes)) as pil_img:
                    pil_img = pil_img.convert("RGBA")
                    png_bytes = pil_to_bytes(pil_img, fmt="PNG")
                    width, height = pil_img.size

                results.append(
                    {
                        "asset_type": "embedded_image",
                        "page": page_index + 1,
                        "source": "pdf_embedded",
                        "width": width,
                        "height": height,
                        "area": width * height,
                        "hash": sha256_bytes(png_bytes),
                        "src": to_data_url(png_bytes, "png"),
                        "alt": f"Extracted image from page {page_index + 1}",
                    }
                )
            except Exception:
                continue

    return results


def render_page_to_pil(page: fitz.Page, dpi: int = PAGE_RENDER_DPI) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def trim_white_margins(img: Image.Image, threshold: int = 245) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")

    px = img.load()
    width, height = img.size

    def is_non_white(x: int, y: int) -> bool:
        r, g, b = px[x, y]
        return not (r >= threshold and g >= threshold and b >= threshold)

    min_x, min_y = width, height
    max_x, max_y = -1, -1

    for y in range(height):
        for x in range(width):
            if is_non_white(x, y):
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

    if max_x == -1 or max_y == -1:
        return img

    pad = 8
    min_x = clamp(min_x - pad, 0, width)
    min_y = clamp(min_y - pad, 0, height)
    max_x = clamp(max_x + pad, 0, width - 1)
    max_y = clamp(max_y + pad, 0, height - 1)

    return img.crop((min_x, min_y, max_x + 1, max_y + 1))


def extract_icon_candidates_from_page(page_img: Image.Image, page_number: int) -> List[Dict[str, Any]]:
    width, height = page_img.size

    top = int(height * ICON_ROW_TOP_RATIO)
    bottom = int(height * ICON_ROW_BOTTOM_RATIO)
    top = clamp(top, 0, height)
    bottom = clamp(bottom, top + 1, height)

    row_img = page_img.crop((0, top, width, bottom))
    row_width, row_height = row_img.size

    # Focus upper portion where icons usually sit (avoid grabbing the text label below).
    icon_top = int(row_height * 0.00)
    icon_bottom = int(row_height * 0.52)
    icon_band = row_img.crop((0, icon_top, row_width, icon_bottom))

    gray = icon_band.convert("L")
    max_w = 700
    scale = 1.0
    if gray.size[0] > max_w:
        scale = max_w / float(gray.size[0])
        gray_small = gray.resize((int(gray.size[0] * scale), int(gray.size[1] * scale)))
    else:
        gray_small = gray

    thresh = 235
    w, h = gray_small.size
    px = gray_small.load()

    visited = [[False] * w for _ in range(h)]
    bboxes: List[Tuple[int, int, int, int, int]] = []  # x1,y1,x2,y2,area

    def neighbors(x: int, y: int):
        if x > 0:
            yield x - 1, y
        if x + 1 < w:
            yield x + 1, y
        if y > 0:
            yield x, y - 1
        if y + 1 < h:
            yield x, y + 1

    for y in range(h):
        for x in range(w):
            if visited[y][x]:
                continue
            visited[y][x] = True
            if px[x, y] >= thresh:
                continue

            stack = [(x, y)]
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cx, cy = stack.pop()
                area += 1
                if cx < min_x:
                    min_x = cx
                if cy < min_y:
                    min_y = cy
                if cx > max_x:
                    max_x = cx
                if cy > max_y:
                    max_y = cy
                for nx, ny in neighbors(cx, cy):
                    if visited[ny][nx]:
                        continue
                    visited[ny][nx] = True
                    if px[nx, ny] < thresh:
                        stack.append((nx, ny))

            bw = max_x - min_x + 1
            bh = max_y - min_y + 1
            if area < 80:
                continue
            if bw < 12 or bh < 12:
                continue
            bboxes.append((min_x, min_y, max_x, max_y, area))

    bboxes.sort(key=lambda b: (b[0], -b[4]))

    items: List[Dict[str, Any]] = []
    for idx, (x1, y1, x2, y2, _area) in enumerate(bboxes[:20], start=1):
        inv = 1.0 / scale
        ox1 = int(x1 * inv)
        oy1 = int(y1 * inv)
        ox2 = int((x2 + 1) * inv)
        oy2 = int((y2 + 1) * inv)

        pad = 12
        ox1 = clamp(ox1 - pad, 0, icon_band.size[0])
        oy1 = clamp(oy1 - pad, 0, icon_band.size[1])
        ox2 = clamp(ox2 + pad, 0, icon_band.size[0])
        oy2 = clamp(oy2 + pad, 0, icon_band.size[1])
        if ox2 <= ox1 or oy2 <= oy1:
            continue

        crop = icon_band.crop((ox1, oy1, ox2, oy2))
        crop = trim_white_margins(crop)

        cw, ch = crop.size
        if cw < ICON_MIN_CROP_WIDTH or ch < ICON_MIN_CROP_HEIGHT:
            continue

        ar = cw / float(ch) if ch else 999.0
        if ar < 0.45 or ar > 2.2:
            continue

        png_bytes = pil_to_bytes(crop.convert("RGBA"), fmt="PNG")
        items.append(
            {
                "asset_type": "icon_candidate",
                "page": page_number,
                "source": "page_crop",
                "width": cw,
                "height": ch,
                "area": cw * ch,
                "hash": sha256_bytes(png_bytes),
                "src": to_data_url(png_bytes, "png"),
                "alt": f"Icon candidate from page {page_number}, component {idx}",
            }
        )

    items = sorted(items, key=lambda x: x.get("area", 0), reverse=True)[:MAX_ICON_IMAGES * 3]

    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        h = item.get("hash")
        if not h or h in seen:
            continue
        seen.add(h)
        out.append(item)
    return out


def heuristic_classification(assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    embedded = [a for a in assets if a.get("asset_type") == "embedded_image"]
    icons = [a for a in assets if a.get("asset_type") == "icon_candidate"]

    embedded_sorted = sorted(embedded, key=lambda x: x.get("area", 0), reverse=True)
    icons_sorted = sorted(icons, key=lambda x: x.get("area", 0), reverse=True)

    hero = embedded_sorted[:1]
    gallery = embedded_sorted[1 : 1 + MAX_GALLERY_IMAGES]
    icons_final = icons_sorted[:MAX_ICON_IMAGES]

    return {"hero": hero, "gallery": gallery, "icons": icons_final, "ignore": []}


def classify_assets_with_openai(
    assets: List[Dict[str, Any]],
    api_key: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    if not assets or OpenAI is None:
        return None

    client = OpenAI(api_key=api_key)
    candidates = assets[:20]

    input_items = []
    for idx, asset in enumerate(candidates):
        input_items.append(
            {
                "type": "input_text",
                "text": (
                    f"Asset index: {idx}\n"
                    f"asset_type: {asset.get('asset_type')}\n"
                    f"page: {asset.get('page')}\n"
                    f"width: {asset.get('width')}\n"
                    f"height: {asset.get('height')}\n"
                    f"alt: {asset.get('alt')}"
                ),
            }
        )
        input_items.append({"type": "input_image", "image_url": asset["src"]})

    prompt = """
You are selecting the best visual assets extracted from a PDF for a landing page.
You will be shown up to 20 candidates. Each candidate has an index (0..N-1) and an image.

### Output format (STRICT)
Return ONLY valid JSON (no markdown, no code fences, no commentary) with EXACTLY this shape:
{
  "hero": [indexes],
  "gallery": [indexes],
  "icons": [indexes],
  "ignore": [indexes],
  "alts": {
    "0": "alt text",
    "1": "alt text"
  }
}

### Selection rules
- Index lists MUST be disjoint (an index can appear in only one list).
- Use only integer indexes that exist (0..N-1).
- Prefer: hero <= 1, gallery <= 6, icons <= 8.
- If unsure about an asset's usefulness, put it in "ignore".

### What belongs where
- hero: ONE strongest, most representative, visually compelling image (usually photo/illustration). Avoid: tiny, blurry, mostly-text, logos, decorative fragments.
- gallery: supporting images that add context/variety. Prefer clear photos/illustrations; avoid near-duplicates and mostly-text slides.
- icons: small symbolic pictograms/marks intended as icons (simple shapes, flat symbols). Avoid cropped text, small photos, or complex scenes.
- ignore: everything else (repeats, low quality, text-heavy, partial crops, logos/wordmarks, UI fragments).

### Tie-breakers (in order)
1) clarity/sharpness and recognizability at small size
2) meaningful content over decorative filler
3) uniqueness (avoid near-duplicates)
4) good composition (not awkwardly cropped)

### Alt text
- Provide "alts" ONLY for indexes you selected into hero/gallery/icons.
- Alt text should be short, concrete, and not include "image of".
- If it's mostly text or unreadable, do NOT select it; if you must, alt text should not attempt to transcribe.
""".strip()

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
                {"role": "user", "content": input_items},
            ],
        )
        text = getattr(response, "output_text", "") or ""
        if not text:
            return None
        return json_loads_strict(text)
    except Exception:
        return None


def json_loads_strict(text: str) -> Dict[str, Any]:
    # Keep a separate function to make it easy to swap in custom parsing later.
    import json

    return json.loads(text)


def apply_openai_result(assets: List[Dict[str, Any]], openai_result: Dict[str, Any]) -> Dict[str, Any]:
    hero_indexes = set(openai_result.get("hero", []))
    gallery_indexes = set(openai_result.get("gallery", []))
    icon_indexes = set(openai_result.get("icons", []))
    ignore_indexes = set(openai_result.get("ignore", []))
    alts = openai_result.get("alts", {}) or {}

    hero: List[Dict[str, Any]] = []
    gallery: List[Dict[str, Any]] = []
    icons: List[Dict[str, Any]] = []
    ignore: List[Dict[str, Any]] = []

    for idx, asset in enumerate(assets[:20]):
        new_asset = dict(asset)
        alt_override = alts.get(str(idx))
        if alt_override:
            new_asset["alt"] = normalize_text(alt_override)

        if idx in hero_indexes:
            hero.append(new_asset)
        elif idx in gallery_indexes:
            gallery.append(new_asset)
        elif idx in icon_indexes:
            icons.append(new_asset)
        elif idx in ignore_indexes:
            ignore.append(new_asset)

    hero = hero[:1]
    gallery = gallery[:MAX_GALLERY_IMAGES]
    icons = icons[:MAX_ICON_IMAGES]

    return {"hero": hero, "gallery": gallery, "icons": icons, "ignore": ignore}


def run_pipeline(
    pdf_bytes: bytes,
    pdf_label: str,
    use_openai: bool,
    openai_api_key: Optional[str],
    openai_model: str,
) -> Dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        embedded_images = extract_embedded_images(doc)

        icon_candidates: List[Dict[str, Any]] = []
        for page_index in range(len(doc)):
            try:
                page = doc[page_index]
                page_img = render_page_to_pil(page, dpi=PAGE_RENDER_DPI)
                icon_candidates.extend(extract_icon_candidates_from_page(page_img, page_number=page_index + 1))
            except Exception:
                continue

        # Deduplicate icons by hash
        seen = set()
        unique_icons: List[Dict[str, Any]] = []
        for a in icon_candidates:
            h = a.get("hash")
            if not h or h in seen:
                continue
            seen.add(h)
            unique_icons.append(a)

        all_assets = embedded_images + unique_icons

        if use_openai and openai_api_key:
            openai_result = classify_assets_with_openai(
                all_assets,
                api_key=openai_api_key,
                model=openai_model,
            )
            if openai_result:
                classified = apply_openai_result(all_assets, openai_result)
                classification_mode = "openai"
            else:
                classified = heuristic_classification(all_assets)
                classification_mode = "heuristic"
        else:
            classified = heuristic_classification(all_assets)
            classification_mode = "heuristic"

        return {
            "meta": {
                "pdf_label": pdf_label,
                "pages": len(doc),
                "embedded_images_found": len(embedded_images),
                "icon_candidates_found": len(unique_icons),
                "classification_mode": classification_mode,
            },
            "images": {
                "hero": classified.get("hero", []),
                "gallery": classified.get("gallery", []),
                "icons": classified.get("icons", []),
            },
            "debug": {"ignored_assets": classified.get("ignore", [])},
        }
    finally:
        try:
            doc.close()
        except Exception:
            pass


def export_hero_gallery_images_to_disk(
    result: Dict[str, Any],
    out_dir: str,
) -> List[str]:
    """
    Saves ONLY hero + gallery images to disk.
    Icons are NOT written to files.

    Mutates items in `result["images"]` by adding:
    - file_path (relative to out_dir parent)
    - filename
    - bucket
    """
    ensure_dir(out_dir)
    images = (result or {}).get("images", {}) or {}

    exported: List[str] = []

    for bucket_name in ("hero", "gallery"):
        bucket = images.get(bucket_name, []) or []
        bucket_dir = os.path.join(out_dir, bucket_name)
        ensure_dir(bucket_dir)

        for i, item in enumerate(bucket, start=1):
            src = item.get("src", "")
            if not isinstance(src, str) or not src.startswith("data:image/"):
                continue

            data, ext = decode_data_url(src)

            h = (item.get("hash") or "")[:16] or f"{i:02d}"
            page = item.get("page", "x")
            filename = f"{i:02d}_page_{page}_{bucket_name}_{h}.{ext}"
            path = os.path.join(bucket_dir, filename)

            with open(path, "wb") as f:
                f.write(data)

            item["filename"] = filename
            item["bucket"] = bucket_name
            item["file_path"] = os.path.join(bucket_name, filename).replace("\\", "/")
            exported.append(path.replace("\\", "/"))

    return exported


def export_images_to_disk(
    result: Dict[str, Any],
    out_dir: str,
    include_icons: bool,
) -> List[str]:
    """
    Saves hero + gallery images to disk, and optionally icons.

    Mutates items in `result["images"]` by adding:
    - file_path (relative to out_dir)
    - filename
    - bucket
    """
    ensure_dir(out_dir)
    images = (result or {}).get("images", {}) or {}

    exported: List[str] = []

    buckets = ["hero", "gallery"]
    if include_icons:
        buckets.append("icons")

    for bucket_name in buckets:
        bucket = images.get(bucket_name, []) or []
        bucket_dir = os.path.join(out_dir, bucket_name)
        ensure_dir(bucket_dir)

        for i, item in enumerate(bucket, start=1):
            src = item.get("src", "")
            if not isinstance(src, str) or not src.startswith("data:image/"):
                continue

            data, ext = decode_data_url(src)

            h = (item.get("hash") or "")[:16] or f"{i:02d}"
            page = item.get("page", "x")
            filename = f"{i:02d}_page_{page}_{bucket_name}_{h}.{ext}"
            path = os.path.join(bucket_dir, filename)

            with open(path, "wb") as f:
                f.write(data)

            item["filename"] = filename
            item["bucket"] = bucket_name
            item["file_path"] = os.path.join(bucket_name, filename).replace("\\", "/")
            exported.append(path.replace("\\", "/"))

    return exported