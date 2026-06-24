"""
Derive landing-page–style hints from a PDF: color palette from rendered pages + text span colors/sizes.
Outputs a CSS string and structured tokens for JSON APIs.
"""
from __future__ import annotations

import io
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image


def _int_color_to_hex(c: int) -> str:
    """PyMuPDF span color is typically 0xRRGGBB."""
    if c < 0:
        c = 0
    r = (c >> 16) & 255
    g = (c >> 8) & 255
    b = c & 255
    return f"#{r:02x}{g:02x}{b:02x}"


def extract_text_style_hints(doc: fitz.Document, max_pages: int = 3) -> Dict[str, Any]:
    colors_hex: List[str] = []
    font_sizes: List[float] = []
    font_names: List[str] = []

    n = min(len(doc), max_pages)
    for i in range(n):
        page = doc[i]
        d = page.get_text("dict") or {}
        for block in d.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sz = span.get("size")
                    if isinstance(sz, (int, float)) and sz > 0:
                        font_sizes.append(float(sz))
                    fn = span.get("font")
                    if isinstance(fn, str) and fn.strip():
                        font_names.append(fn.strip())
                    c = span.get("color")
                    if c is not None:
                        try:
                            colors_hex.append(_int_color_to_hex(int(c)))
                        except (TypeError, ValueError):
                            pass

    # Most common body-ish size (median of sizes in a reasonable band)
    body_px = 16.0
    heading_px = 28.0
    if font_sizes:
        font_sizes.sort()
        mid = len(font_sizes) // 2
        body_px = round(font_sizes[mid] * 1.33, 1)  # PDF points → rough px
        heading_px = round(max(font_sizes) * 1.33, 1)

    color_counts = Counter(colors_hex)
    top_text_colors = [h for h, _ in color_counts.most_common(8)]

    font_counter = Counter(font_names)
    top_fonts = [f for f, _ in font_counter.most_common(3)]

    return {
        "text_colors_sample": top_text_colors,
        "font_sizes_pt_sample": sorted(set(round(s, 1) for s in font_sizes))[:12],
        "estimated_body_font_size_px": body_px,
        "estimated_heading_font_size_px": heading_px,
        "font_families_observed": top_fonts,
    }


def _dominant_colors_from_pil(img: Image.Image, n_colors: int = 8) -> List[str]:
    img = img.convert("RGB")
    img = img.resize((120, int(120 * img.size[1] / max(img.size[0], 1))), Image.Resampling.LANCZOS)
    q = img.quantize(colors=min(n_colors, 256), method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()
    if not pal:
        return []
    counts = q.getcolors()
    if not counts:
        return []
    counts.sort(reverse=True, key=lambda x: x[0])
    out: List[str] = []
    for count, idx in counts[:n_colors]:
        r, g, b = pal[idx * 3], pal[idx * 3 + 1], pal[idx * 3 + 2]
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def extract_visual_palette_from_pdf(
    pdf_bytes: bytes,
    dpi: int = 72,
    max_pages: int = 2,
) -> List[str]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        all_colors: List[str] = []
        n = min(len(doc), max_pages)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i in range(n):
            page = doc[i]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pil = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            all_colors.extend(_dominant_colors_from_pil(pil, n_colors=8))
        # Dedupe preserving order
        seen = set()
        uniq: List[str] = []
        for c in all_colors:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        return uniq[:12]
    finally:
        doc.close()


def build_css_from_tokens(
    palette: List[str],
    text_hints: Dict[str, Any],
) -> str:
    """Single CSS block with :root variables suitable for a landing page shell."""
    primary = palette[0] if palette else "#1a1a1a"
    secondary = palette[1] if len(palette) > 1 else "#f5f5f5"
    accent = palette[2] if len(palette) > 2 else "#2563eb"
    body = text_hints.get("estimated_body_font_size_px") or 16
    heading = text_hints.get("estimated_heading_font_size_px") or 28
    text_cols = text_hints.get("text_colors_sample") or []
    text_main = text_cols[0] if text_cols else "#111827"

    fonts = text_hints.get("font_families_observed") or []
    stack = ", ".join(f'"{f}"' for f in fonts[:2]) if fonts else ""
    if stack:
        stack += ", "
    stack += "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"

    lines = [
        ":root {",
        f"  --color-bg: {secondary};",
        f"  --color-surface: #ffffff;",
        f"  --color-primary: {primary};",
        f"  --color-accent: {accent};",
        f"  --color-text: {text_main};",
        f"  --font-family-base: {stack};",
        f"  --font-size-body: {body}px;",
        f"  --font-size-heading: {heading}px;",
        "}",
        "",
        "body {",
        "  margin: 0;",
        "  font-family: var(--font-family-base);",
        "  font-size: var(--font-size-body);",
        "  color: var(--color-text);",
        "  background: var(--color-bg);",
        "}",
        "",
        ".hero, .gallery img, .icons img { max-width: 100%; height: auto; }",
    ]
    return "\n".join(lines)


def extract_full_style(pdf_bytes: bytes) -> Dict[str, Any]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        text_hints = extract_text_style_hints(doc)
    finally:
        doc.close()

    palette = extract_visual_palette_from_pdf(pdf_bytes, dpi=72, max_pages=2)
    css = build_css_from_tokens(palette, text_hints)

    return {
        "palette": palette,
        "text": text_hints,
        "css": css,
    }