import argparse
import base64
import json
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

try:
    from PIL import Image
except Exception:
    Image = None


MODEL = "gpt-5.4"

HERE = Path(__file__).resolve().parent
IMAGES_DIR = HERE / "images"
REFERENCE_DIR = HERE / "reference"

OUT_HTML = HERE / "index.html"


SYSTEM_PROMPT = """
You are a senior front-end engineer + Hebrew conversion copywriter.
You generate production-ready landing pages from structured JSON.

### Core objective
Create a premium, conversion-focused Hebrew (RTL) landing page with clean typography and clear hierarchy.
The output should feel like a real marketing page: confident, precise, readable, and persuasive.

### Source of truth & safety
- The user provides merge_json with:
  - content blocks (hero / benefits / course facts / career paths / CTAs)
  - styling tokens (palette + css string + font hints)
  - images list (hero/gallery/icons file names that exist under ./images)
- You MAY enrich missing/empty copy for clarity, but MUST NOT invent hard facts:
  - no made-up certifications, salaries, guarantees, dates, hours, eligibility, institutions, authorities, or outcomes
  - if a claim is not supported, phrase it generically (e.g., “יתרון בשוק” is ok; “הסמכה רשמית” is not unless present)

### Design requirements (make it look "finished")
- Typography: Hebrew-friendly font stack (system fallbacks). Do NOT rely on PDF-only fonts like "FbEleganti-*".
- Clear spacing rhythm: consistent section padding, headings, and card spacing.
- Strong contrast & readability: avoid low-contrast text on gradients.
- Use a design system with CSS variables: colors, spacing scale, radius, shadows.
- Make the page feel richer and more alive: layered gradients, subtle depth, and interesting full-width composition
  (not only a centered narrow column). Use decorative side visuals (pure CSS or the provided images with low opacity)
  but keep readability.
- Responsive: great on mobile, solid on desktop.

### Layout template (use this structure)
1) Sticky header with brand + CTA button
2) Hero: eyebrow + H1 + highlight line + short-but-rich description + 2 CTAs + 3 bullets
3) Proof/Context section: “future_of_field” + role explanation
4) Benefits grid: 4 cards (why_learn_here.cards)
5) Gallery: 2–3 images from gallery
6) Course facts: audience + requirements + bonus
7) Careers: cards grid
8) Lead form: name/phone/email + privacy note + submit success state
9) Closing CTA
10) Footer

### Images & icons
- Only use images that exist under ./images and are listed in available_images.
- Use hero image in hero.
- Use 2–3 gallery images as a gallery section.
- Icons logic (IMPORTANT):
  - Prefer extracted icons ONLY if they clearly look like real icons (simple glyph/shape), not text fragments, not cropped UI, not blurry.
  - If any extracted icon is questionable, IGNORE it and use an inline SVG icon instead.
  - Keep a consistent icon style set (same stroke weight / corner radius / fill style).
- In the careers section, include small visuals per role (either good extracted icons OR inline SVG badges).
- Add an infographic/graph section using inline SVG or CSS:
  - If salary ranges are not present in merge_json, you MAY provide a conservative estimate based on general market knowledge,
    but you MUST label it clearly as an estimate ("טווחים משוערים") and add a disclaimer ("עשוי להשתנות לפי ניסיון/אזור/מעסיק").
  - Prefer 3 clean bars (Junior / Mid / Senior) with readable labels.
  - NEVER draw a weird scribbly polyline graph. Use bars or a clean SVG chart.
- If any image seems like a text crop / low quality, mark it ignore in image_plan and do not place it.
- Do not crop important imagery. Avoid unexpected cut-offs.

### Animations (subtle)
- IntersectionObserver “reveal on scroll”
- Card hover micro-interactions
- Sticky header shadow on scroll
- Add a bit more motion: parallax-lite background accents or animated gradient/noise (respect reduced motion)
- Respect prefers-reduced-motion

### Output format (STRICT)
Return ONLY valid JSON (no markdown/fences/comments) with EXACTLY this shape:
{
  "files": {
    "index.html": "<single self-contained html>"
  },
  "image_plan": [
    {
      "bucket": "hero|gallery|icons",
      "fileName": "string",
      "use": true,
      "placement": "hero|gallery|icons|background|ignore",
      "reason": "short Hebrew reason"
    }
  ]
}

### HTML/CSS/JS constraints
- Everything must be in ONE HTML file:
  - Include ALL CSS in a single <style> tag in the <head>.
  - Include ALL JS in a single <script> tag before </body>.
- Do NOT reference external CSS/JS files.
- Use <html lang="he" dir="rtl">.
- No external JS/CSS frameworks, no remote fonts, no remote images.
- Use semantic HTML + accessibility: headings, labels, aria-live for form status, alt text.
- Form: validate client-side, show success message, no network request needed.

### Critical layout constraints (must pass)
- No text should overlap images. Use CSS grid/flex with gaps; avoid absolute-positioned text over images.
- Ensure text blocks are never behind images (z-index/background). Text containers should have readable background.
- Gallery alignment: all gallery cards must be the same visual size (same aspect ratio + same height).
  Use a fixed aspect-ratio container with object-fit: cover; keep important subject centered.
- Gallery images must not be cut off unexpectedly; do not use overflow hidden on the outer grid.
- Footer must be full-width and visually complete (no dead empty zones).
- Salary chart labels must always be readable (high-contrast text + optional text-shadow; never white-on-light).
""".strip()


FIXUP_PROMPT = """
You are fixing a generated Hebrew RTL landing page (single HTML file with inline CSS/JS).

You will receive:
- merge_json (source of truth)
- available_images list (hero/gallery images)
- the current generated HTML (single file)
- a list of issues to fix

Return JSON ONLY with EXACT schema:
{ "files": { "index.html": "<fixed html>" }, "image_plan": [...] }

Rules:
- Keep everything in one HTML file; no external assets.
- Preserve factual safety: do not invent hard facts. If salary ranges were not in merge_json, label as estimate + disclaimer.
- Fix the listed issues precisely without introducing new layout bugs.
""".strip()


def _find_merge_json(cli_path: Optional[str]) -> Path:
    """
    Resolve merge.json path robustly:
    - Use --merge if provided (absolute or relative to landing_page/)
    - Otherwise look in landing_page/ then project root
    """
    if cli_path:
        p = Path(cli_path).expanduser()
        if not p.is_absolute():
            p = (HERE / p).resolve()
        return p

    for p in [HERE / "merge.json", HERE.parent / "merge.json"]:
        if p.exists():
            return p

    return HERE / "merge.json"


def _load_merge_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"merge.json not found at: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Your n8n merge output is typically a list with a single item.
    if isinstance(data, list):
        if not data:
            return {}
        if len(data) == 1 and isinstance(data[0], dict):
            return data[0]
    return data if isinstance(data, dict) else {}


def _list_images() -> list[str]:
    if not IMAGES_DIR.exists():
        return []
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    files: list[str] = []
    for p in sorted(IMAGES_DIR.iterdir()):
        if p.is_file() and p.suffix.lower() in allowed:
            files.append(p.name)
    return files


def _encode_image_for_prompt(path: Path, max_side: int = 720) -> Optional[str]:
    """
    Returns a data URL for an image, downsampled for token efficiency.
    Requires Pillow; if Pillow is missing, returns None.
    """
    if Image is None or not path.exists():
        return None
    suf = path.suffix.lower()
    if suf not in {".png", ".jpg", ".jpeg", ".webp"}:
        return None

    mime = "image/png" if suf == ".png" else ("image/webp" if suf == ".webp" else "image/jpeg")
    with Image.open(path) as im:
        if mime == "image/png":
            im = im.convert("RGBA")
        else:
            im = im.convert("RGB")
        w, h = im.size
        scale = min(1.0, max_side / float(max(w, h) or 1))
        if scale < 1.0:
            im = im.resize((int(w * scale), int(h * scale)))

        buf = BytesIO()
        if mime == "image/png":
            im.save(buf, format="PNG")
        elif mime == "image/webp":
            im.save(buf, format="WEBP", quality=85)
        else:
            im.save(buf, format="JPEG", quality=85)

        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{b64}"


def _build_visual_inputs(
    merge: dict,
    available_images: List[str],
    images_dir: Path = IMAGES_DIR,
    reference_dir: Path = REFERENCE_DIR,
) -> List[Dict[str, Any]]:
    """
    Provide a curated set of images so the model can:
    - detect bad icon crops (text fragments)
    - imitate the reference screenshots (if provided)
    """
    items: List[Dict[str, Any]] = []

    def add(label: str, filename: str, max_side: int = 720):
        if filename not in available_images:
            return
        data_url = _encode_image_for_prompt(images_dir / filename, max_side=max_side)
        if not data_url:
            return
        items.append({"type": "input_text", "text": f"{label}: {filename}"})
        items.append({"type": "input_image", "image_url": data_url})

    images = (merge or {}).get("images", {}) or {}
    for h in (images.get("hero") or [])[:1]:
        fn = h.get("fileName")
        if isinstance(fn, str):
            add("HERO_CANDIDATE", fn, max_side=900)
    for i, g in enumerate((images.get("gallery") or [])[:3], start=1):
        fn = g.get("fileName")
        if isinstance(fn, str):
            add(f"GALLERY_CANDIDATE_{i}", fn, max_side=900)
    for i, ic in enumerate((images.get("icons") or [])[:8], start=1):
        fn = ic.get("fileName")
        if isinstance(fn, str):
            add(f"ICON_CANDIDATE_{i}", fn, max_side=420)

    # Optional reference screenshots: user can drop images into landing_page/reference/
    if reference_dir.exists():
        for p in sorted(reference_dir.iterdir()):
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            data_url = _encode_image_for_prompt(p, max_side=1100)
            if not data_url:
                continue
            items.append({"type": "input_text", "text": f"REFERENCE_SCREENSHOT (style target): {p.name}"})
            items.append({"type": "input_image", "image_url": data_url})
            if len(items) > 40:  # cap
                break

    return items


def _ensure_reference_dir_hint() -> None:
    """
    Create an empty reference/ folder (optional) to encourage the workflow.
    The generator will use any screenshots dropped there to match style more closely.
    """
    try:
        REFERENCE_DIR.mkdir(exist_ok=True)
    except Exception:
        pass


def _call_llm(payload: dict, visual_inputs: List[Dict[str, Any]]) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is missing.")
    client = OpenAI(api_key=api_key)

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": (
                    [
                        {"type": "input_text", "text": "INPUT_JSON (source of truth):"},
                        {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)},
                    ]
                    + (
                        [{"type": "input_text", "text": "VISUAL_INPUTS (judge icons/style; ignore bad crops):"}]
                        + visual_inputs
                        if visual_inputs
                        else []
                    )
                ),
            },
        ],
    )

    text = getattr(resp, "output_text", None) or ""
    if not text.strip():
        raise RuntimeError("Model returned empty output_text.")

    # Must be JSON-only; sanitize in case of fences.
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse model JSON. Error: {e}\n---\n{text[:2000]}") from e


def _call_llm_fixup(
    payload: dict,
    visual_inputs: List[Dict[str, Any]],
    current_html: str,
    issues: List[str],
) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var is missing.")
    client = OpenAI(api_key=api_key)

    content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": "ISSUES TO FIX (address all):\n- " + "\n- ".join(issues)},
        {"type": "input_text", "text": "INPUT_JSON (source of truth):"},
        {"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)},
        {"type": "input_text", "text": "CURRENT_HTML (fix this single file):"},
        {"type": "input_text", "text": current_html},
    ]
    if visual_inputs:
        content.append({"type": "input_text", "text": "VISUAL_INPUTS (icons/style reference):"})
        content.extend(visual_inputs)

    resp = client.responses.create(
        model=MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": FIXUP_PROMPT}]},
            {"role": "user", "content": content},
        ],
    )

    text = getattr(resp, "output_text", None) or ""
    if not text.strip():
        raise RuntimeError("Model returned empty output_text (fixup).")

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse fixup JSON. Error: {e}\n---\n{text[:2000]}") from e


def _write_files(result: dict) -> None:
    files = result.get("files") or {}
    html = files.get("index.html")

    if not isinstance(html, str):
        raise RuntimeError("Model output missing files.index.html as a string.")

    OUT_HTML.write_text(html, encoding="utf-8")


def generate_landing_result(
    merge: dict,
    images_dir: Path = IMAGES_DIR,
    reference_dir: Path = REFERENCE_DIR,
) -> dict:
    """
    Programmatic API: build the model output dict:
    { files: { index.html }, image_plan: [...] }

    - merge: the merged JSON object coming from n8n Merge node.
    - images_dir: directory containing image files referenced by merge.images.*.fileName
    - reference_dir: optional screenshots to guide style matching
    """
    available_images = _list_images() if images_dir == IMAGES_DIR else _list_images_from(images_dir)
    try:
        reference_dir.mkdir(exist_ok=True, parents=True)
    except Exception:
        pass

    payload = {
        "merge_json": merge,
        "available_images": available_images,
        "images_path": "./images/",
        "output_files": ["index.html"],
        "notes": [
            "Prefer using ./images for hero/gallery/icons if visually useful.",
            "If an image seems like a text fragment or low-quality crop, mark it ignore.",
            "Keep Hebrew RTL. Enrich missing copy without inventing factual claims.",
            "Keep the structure conversion-focused: hero → benefits → proof → course facts → lead form → closing CTA.",
        ],
        "quality_checklist": [
            "Hebrew RTL typography feels premium and readable",
            "No invented factual claims (dates/certs/salary/guarantees)",
            "Hero copy is concise but rich, with clear CTAs",
            "Consistent spacing and hierarchy across sections",
            "Icons are used only if they look like icons",
            "Responsive layout works on mobile and desktop",
            "Animations are subtle and respect prefers-reduced-motion",
        ],
    }

    visual_inputs = _build_visual_inputs(merge, available_images, images_dir=images_dir, reference_dir=reference_dir)
    result = _call_llm(payload, visual_inputs)

    # Second pass: enforce a targeted fix list based on known failure modes.
    try:
        html = (result.get("files") or {}).get("index.html")
        if isinstance(html, str) and html.strip():
            issues = [
                "Restore icon selection logic: use extracted icon PNGs ONLY if they truly look like icons; otherwise ignore and use inline SVG. Keep icon style consistent.",
                "Add a clean salary potential chart. If not in merge_json, label 'טווחים משוערים' + disclaimer about variability.",
                "Fix any overlap where a text box appears behind/under an image (ensure proper layout, z-index, background).",
                "Remove/replace any weird scribbly graph line; prefer simple bar chart with clean labels.",
                "Make gallery images aligned to same size: consistent card dimensions/aspect ratio; avoid uneven heights.",
                "Fix any images being cut off on the sides; ensure no unexpected overflow clipping; keep safe padding.",
                "Ensure salary numbers/labels are high-contrast and readable (not white on light background).",
                "Eliminate dead empty areas by adding relevant content blocks (no invented hard facts).",
                "Footer must fill full width; remove dead zone and add meaningful footer content blocks.",
            ]
            result = _call_llm_fixup(payload, visual_inputs, html, issues)
    except Exception:
        # If fixup fails, keep first pass.
        pass

    return result


def generate_landing_html(
    merge: dict,
    images_dir: Path = IMAGES_DIR,
    reference_dir: Path = REFERENCE_DIR,
) -> str:
    result = generate_landing_result(merge=merge, images_dir=images_dir, reference_dir=reference_dir)
    html = (result.get("files") or {}).get("index.html")
    if not isinstance(html, str) or not html.strip():
        raise RuntimeError("Generator returned empty HTML.")
    return html


def _list_images_from(images_dir: Path) -> list[str]:
    if not images_dir.exists():
        return []
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
    files: list[str] = []
    for p in sorted(images_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in allowed:
            files.append(p.name)
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", help="Path to merge.json (optional)")
    args = parser.parse_args()

    merge_path = _find_merge_json(args.merge)
    merge = _load_merge_json(merge_path)
    _ensure_reference_dir_hint()

    result = generate_landing_result(merge=merge, images_dir=IMAGES_DIR, reference_dir=REFERENCE_DIR)
    _write_files(result)

    print(
        json.dumps(
            {
                "status": "ok",
                "written": [str(OUT_HTML)],
                "image_plan_count": len(result.get("image_plan") or []),
                "merge_json_used": str(merge_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
