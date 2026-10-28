"""Turn an uploaded PDF or image into text.

Digital PDFs are read directly (fast, exact). Scanned PDFs and images go
through PaddleOCR (PP-OCRv6), which runs locally - nothing leaves the server.
PaddleOCR is optional: if it isn't installed, scanned input is reported clearly.
"""
from __future__ import annotations

import io

MAX_PAGES = 5


def ocr_available() -> bool:
    try:
        import paddleocr  # noqa: F401
        return True
    except Exception:
        return False


def load_ocr(tier: str = "medium"):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name=f"PP-OCRv6_{tier}_det",
        text_recognition_model_name=f"PP-OCRv6_{tier}_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _ocr_images(engine, images) -> str:
    import numpy as np

    pages = []
    for img in images:
        arr = np.array(img.convert("RGB"))[:, :, ::-1]  # RGB -> BGR for OpenCV
        lines = []
        for res in engine.predict(arr):
            data = res.json.get("res", res.json) if hasattr(res, "json") else res
            lines.extend(data.get("rec_texts", []))
        pages.append("\n".join(lines))
    return "\n\n".join(pages)


def pdf_text(data: bytes) -> tuple[str, int]:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    n = len(reader.pages)
    text = "\n\n".join((p.extract_text() or "") for p in reader.pages[:MAX_PAGES])
    return text, n


def pdf_images(data: bytes, scale: float = 2.0):
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(data)
    return [doc[i].render(scale=scale).to_pil() for i in range(min(len(doc), MAX_PAGES))]


def file_to_text(name: str, data: bytes, get_engine) -> tuple[str, str]:
    """Return (text, how) where how describes the method used."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        text, n = pdf_text(data)
        if len(text.strip()) > 200:
            note = f"Read the PDF's text layer ({min(n, MAX_PAGES)} of {n} pages)."
            return text, note
        if not ocr_available():
            raise RuntimeError("This PDF is scanned, and OCR isn't installed on this server. "
                               "Paste the memo text instead.")
        text = _ocr_images(get_engine(), pdf_images(data))
        return text, f"Scanned PDF: read with PP-OCRv6 (first {min(n, MAX_PAGES)} pages)."
    from PIL import Image

    if not ocr_available():
        raise RuntimeError("OCR isn't installed on this server. Paste the memo text instead.")
    img = Image.open(io.BytesIO(data))
    return _ocr_images(get_engine(), [img]), "Image: read with PP-OCRv6."
