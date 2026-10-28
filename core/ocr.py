"""Turn an uploaded PDF or image into text.

Digital PDFs are read directly (fast, exact). Scanned PDFs and images go
through PaddleOCR (PP-OCRv6), which runs locally - nothing leaves the server.
The models run on ONNX Runtime (Baidu's official ONNX exports of PP-OCRv6), so
PaddleOCR works without the paddlepaddle framework, which has no build for
newer Pythons. PaddleOCR is optional: if it isn't installed, scanned input is
reported clearly.
"""
from __future__ import annotations

import io
import os

MAX_PAGES = 5
# Measured on the scanned Mata v. Avianca brief (5 pages, 2 CPU threads): the small tier reads it in
# ~23 s with a 420 MB peak, and its text matches the medium tier's to 99.9%; medium takes ~108 s.
# 1.5x rendering (108 dpi) reads as well as 2x. ONNX Runtime's memory arena roughly doubles peak
# memory, so it is off.
DEFAULT_TIER = "small"
RENDER_SCALE = 1.5
ORT_CONFIG = {"intra_op_num_threads": 2, "enable_cpu_mem_arena": False, "enable_mem_pattern": False}


def ocr_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        import paddleocr  # noqa: F401
        return True
    except Exception:
        return False


def load_ocr(tier: str = DEFAULT_TIER):
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")  # skip a slow startup ping
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{tier}_det",
        text_recognition_model_name=f"PP-OCRv6_{tier}_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="onnxruntime",
        device="cpu",
        engine_config=ORT_CONFIG,
    )


def _ocr_images(engine, images) -> list[str]:
    """One string of recognized text per image."""
    import numpy as np

    pages = []
    for img in images:
        arr = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR for OpenCV
        lines = []
        for res in engine.predict(arr):
            data = res.json.get("res", res.json) if hasattr(res, "json") else res
            lines.extend(data.get("rec_texts", []))
        pages.append("\n".join(lines))
    return pages


# A page with less text than this is treated as scanned. Court e-filing stamps a text header
# ("Case 1:22-cv-01461 Document 21 Filed 03/01/23 Page 1 of 10", ~70 chars) onto scanned pages,
# so "has any text layer" is not enough.
MIN_PAGE_CHARS = 200


def pdf_text(data: bytes) -> tuple[list[str], int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    n = len(reader.pages)
    return [(p.extract_text() or "") for p in reader.pages[:MAX_PAGES]], n


def pdf_images(data: bytes, pages: list[int], scale: float = RENDER_SCALE):
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(data)
    return [doc[i].render(scale=scale).to_pil() for i in pages]


def file_to_text(name: str, data: bytes, get_engine) -> tuple[str, str]:
    """Return (text, how) where how describes the method used."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        texts, n = pdf_text(data)
        scanned = [i for i, t in enumerate(texts) if len(t.strip()) < MIN_PAGE_CHARS]
        if not scanned:
            return "\n\n".join(texts), f"Read the PDF's text layer ({len(texts)} of {n} pages)."
        if not ocr_available():
            raise RuntimeError("This PDF is scanned, and OCR isn't installed on this server. "
                               "Paste the memo text instead.")
        for i, t in zip(scanned, _ocr_images(get_engine(), pdf_images(data, scanned))):
            texts[i] = t
        how = ("Scanned PDF: read with PP-OCRv6" if len(scanned) == len(texts)
               else f"Read the PDF's text layer, and {len(scanned)} scanned page(s) with PP-OCRv6")
        return "\n\n".join(texts), f"{how} (first {len(texts)} of {n} pages)."
    from PIL import Image

    if not ocr_available():
        raise RuntimeError("OCR isn't installed on this server. Paste the memo text instead.")
    img = Image.open(io.BytesIO(data))
    return _ocr_images(get_engine(), [img])[0], "Image: read with PP-OCRv6."
