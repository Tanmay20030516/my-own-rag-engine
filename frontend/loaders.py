"""
Turn an uploaded file into plain text for the RAG pipeline.

Plain text and markdown are read directly; PDF and DOCX go through pypdf
and python-docx respectively. Everything funnels through load_document(),
which dispatches on file extension and raises DocumentLoadError with a
message meant to be shown to the user rather than logged.
"""

import io
from pathlib import Path

SUPPORTED_EXTENSIONS = ["txt", "md", "markdown", "pdf", "docx"]


class DocumentLoadError(Exception):
    """Raised when a file can't be turned into text -- message is user-facing."""


def _load_text(data: bytes) -> str:
    """Decode as UTF-8, falling back to latin-1 so odd encodings degrade instead of failing."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def _load_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise DocumentLoadError("PDF support needs `pypdf` (pip install pypdf).") from exc

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise DocumentLoadError(f"Could not read this PDF: {exc}") from exc

    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise DocumentLoadError(
            "No selectable text found -- this looks like a scanned PDF, which needs OCR first."
        )
    return text


def _load_docx(data: bytes) -> str:
    try:
        import docx
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise DocumentLoadError("DOCX support needs `python-docx` (pip install python-docx).") from exc

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise DocumentLoadError(f"Could not read this DOCX: {exc}") from exc

    return "\n\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())


_LOADERS = {
    "txt": _load_text,
    "md": _load_text,
    "markdown": _load_text,
    "pdf": _load_pdf,
    "docx": _load_docx,
}


def load_document(filename: str, data: bytes) -> str:
    """Extract text from an uploaded file's bytes, dispatching on its extension."""
    suffix = Path(filename).suffix.lower().lstrip(".")
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise DocumentLoadError(
            f"Unsupported file type {suffix or '(none)'!r}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
        )

    text = loader(data).strip()
    if not text:
        raise DocumentLoadError(f"{filename} contains no extractable text.")
    return text
