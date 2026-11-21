from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".markdown", ".docx", ".pptx", ".xlsx",
    ".csv", ".tsv", ".json", ".html", ".htm", ".rtf",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff",
}
_MAX_EXTRACTED_CHARS = 2_000_000
_OCR_ENGINE = None
_OCR_LOCK = threading.Lock()


class DocumentProcessingError(ValueError):
    """Raised when a document cannot be parsed or validated."""


@dataclass(frozen=True)
class ParsedDocument:
    source_id: str
    content: str
    metadata: dict[str, Any]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_source_id(filename: str) -> str:
    stem = Path(filename).stem.strip() or "document"
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", stem).strip(".-")
    return value[:160] or "document"


def _extract_pdf(data: bytes) -> tuple[str, dict[int, str]]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise DocumentProcessingError("PDF support requires the 'pypdf' package.") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        pages: dict[int, str] = {}
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages[index] = text
        marked = "\n\n".join(f"[PAGE {page}]\n{text}" for page, text in pages.items())
        return marked, pages
    except Exception as exc:
        raise DocumentProcessingError("Could not extract text from the PDF document.") from exc


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
    except Exception as exc:
        raise DocumentProcessingError("DOCX support requires the 'python-docx' package.") from exc

    try:
        document = Document(io.BytesIO(data))
        blocks = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
        for table in document.tables:
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                if any(cells):
                    blocks.append(" | ".join(cells))
        return "\n\n".join(blocks)
    except Exception as exc:
        raise DocumentProcessingError("Could not extract text from the DOCX document.") from exc


def _extract_pptx(data: bytes) -> str:
    try:
        from pptx import Presentation
    except Exception as exc:
        raise DocumentProcessingError("PowerPoint support requires python-pptx.") from exc
    try:
        presentation = Presentation(io.BytesIO(data))
        slides = []
        for number, slide in enumerate(presentation.slides, start=1):
            parts = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    text = shape.text.strip()
                    if text:
                        parts.append(text)
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        cells = [cell.text.strip() for cell in row.cells]
                        if any(cells):
                            parts.append(" | ".join(cells))
            if parts:
                slides.append(f"[SLIDE {number}]\n" + "\n".join(parts))
        return "\n\n".join(slides)
    except Exception as exc:
        raise DocumentProcessingError("Could not extract text from the PowerPoint presentation.") from exc


def _extract_xlsx(data: bytes) -> str:
    try:
        from openpyxl import load_workbook
    except Exception as exc:
        raise DocumentProcessingError("Excel support requires openpyxl.") from exc
    try:
        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        sheets = []
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                cells = [str(value).strip() if value is not None else "" for value in row]
                if any(cells):
                    rows.append(" | ".join(cells))
                if sum(map(len, rows)) > _MAX_EXTRACTED_CHARS:
                    raise DocumentProcessingError("The spreadsheet contains too much text to index.")
            if rows:
                sheets.append(f"[SHEET {sheet.title}]\n" + "\n".join(rows))
        workbook.close()
        return "\n\n".join(sheets)
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError("Could not extract text from the Excel workbook.") from exc


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag in {"br", "p", "div", "li", "tr", "h1", "h2", "h3"} and not self._skip_depth:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"p", "div", "li", "tr", "h1", "h2", "h3"} and not self._skip_depth:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def _extract_image(data: bytes) -> str:
    global _OCR_ENGINE
    try:
        import numpy as np
        from PIL import Image
        from rapidocr import RapidOCR
    except Exception as exc:
        raise DocumentProcessingError(
            "Image text extraction is unavailable. Install the OCR extra with `pip install -e '.[ocr]'`."
        ) from exc
    try:
        with Image.open(io.BytesIO(data)) as image:
            pixels = np.asarray(image.convert("RGB"))
        with _OCR_LOCK:
            if _OCR_ENGINE is None:
                _OCR_ENGINE = RapidOCR()
            result = _OCR_ENGINE(pixels)
        return "\n".join(text.strip() for text in (result.txts or ()) if text and text.strip())
    except DocumentProcessingError:
        raise
    except Exception as exc:
        raise DocumentProcessingError("Could not read text from this image. Try a clearer PNG or JPEG image.") from exc


def parse_document(filename: str, data: bytes, source_id: str | None = None) -> ParsedDocument:
    if not data:
        raise DocumentProcessingError("The uploaded document is empty.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentProcessingError(f"Unsupported file type '{suffix or 'unknown'}'. Supported: {supported}.")

    pages: dict[int, str] = {}
    if suffix == ".pdf":
        content, pages = _extract_pdf(data)
    elif suffix == ".docx":
        content = _extract_docx(data)
    elif suffix == ".pptx":
        content = _extract_pptx(data)
    elif suffix == ".xlsx":
        content = _extract_xlsx(data)
    elif suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        content = _extract_image(data)
    elif suffix in {".csv", ".tsv"}:
        try:
            rows = csv.reader(io.StringIO(data.decode("utf-8-sig")), delimiter="\t" if suffix == ".tsv" else ",")
            content = "\n".join(" | ".join(cell.strip() for cell in row) for row in rows)
        except (UnicodeDecodeError, csv.Error) as exc:
            raise DocumentProcessingError("CSV and TSV files must be valid UTF-8 text.") from exc
    elif suffix == ".json":
        try:
            content = json.dumps(json.loads(data.decode("utf-8-sig")), ensure_ascii=False, indent=2)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DocumentProcessingError("The JSON file is not valid UTF-8 JSON.") from exc
    elif suffix in {".html", ".htm"}:
        try:
            parser = _HTMLTextParser()
            parser.feed(data.decode("utf-8-sig"))
            content = " ".join(" ".join(parser.parts).split())
        except Exception as exc:
            raise DocumentProcessingError("Could not extract readable text from the HTML document.") from exc
    elif suffix == ".rtf":
        try:
            raw = data.decode("utf-8-sig")
            content = re.sub(r"\\'[0-9a-fA-F]{2}|\\[a-zA-Z]+-?\d* ?|[{}]", " ", raw)
        except UnicodeDecodeError as exc:
            raise DocumentProcessingError("The RTF document must be UTF-8 encoded.") from exc
    else:
        try:
            content = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentProcessingError("Text documents must be UTF-8 encoded.") from exc

    content = content.replace("\x00", " ").strip()
    if not content:
        raise DocumentProcessingError("No readable text was found in the document.")
    if len(content) > _MAX_EXTRACTED_CHARS:
        raise DocumentProcessingError("The document contains too much extracted text to index (2 million character limit).")

    digest = hashlib.sha256(data).hexdigest()
    document_id = source_id.strip() if source_id else f"{_safe_source_id(filename)}-{digest[:10]}"
    metadata: dict[str, Any] = {
        "filename": filename,
        "file_type": suffix.lstrip("."),
        "content_hash": digest,
        "ingested_at": _now(),
        "updated_at": _now(),
        "page_count": len(pages),
    }

    return ParsedDocument(source_id=document_id, content=content, metadata=metadata)
