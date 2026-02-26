"""
Game Maker Agent v1 — PDF Loader Service
Secure PDF ingestion — extracts full text from a PDF.
Each PDF represents one chapter (no heading detection needed).
Stateless. No side effects.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pymupdf  # PyMuPDF

from app.config import PDFSettings

logger = logging.getLogger(__name__)


class PDFSecurityError(Exception):
    """Raised when PDF access violates security constraints."""


class PDFExtractionError(Exception):
    """Raised when text extraction fails."""


def _validate_pdf_path(pdf_path: str, settings: PDFSettings) -> Path:
    """Validate PDF path against security constraints."""
    resolved = Path(pdf_path).resolve()
    allowed = settings.allowed_directory.resolve()

    if not str(resolved).startswith(str(allowed)):
        raise PDFSecurityError(
            f"Access denied: {resolved} is outside allowed directory {allowed}"
        )

    if not resolved.exists():
        raise PDFSecurityError(f"PDF not found: {resolved}")

    if not resolved.is_file():
        raise PDFSecurityError(f"Not a file: {resolved}")

    if not resolved.suffix.lower() == ".pdf":
        raise PDFSecurityError(f"Not a PDF file: {resolved}")

    size = resolved.stat().st_size
    if size > settings.max_size_bytes:
        raise PDFSecurityError(
            f"PDF too large: {size} bytes (max {settings.max_size_bytes})"
        )

    logger.info("PDF validated: %s (%d bytes)", resolved, size)
    return resolved


def _extract_full_text(pdf_path: Path) -> str:
    """Extract all text from the PDF.

    Each PDF = one chapter, so we extract everything.
    """
    doc = pymupdf.open(str(pdf_path))
    pages_text: list[str] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_text.append(text)

    doc.close()

    if not pages_text:
        raise PDFExtractionError("PDF contains no extractable text")

    raw_text = "\n".join(pages_text)
    cleaned = _clean_text(raw_text)

    if len(cleaned) < 50:
        raise PDFExtractionError(
            f"PDF text too short ({len(cleaned)} chars). "
            "Extraction may have failed."
        )

    logger.info(
        "Extracted full PDF: %d characters from %d pages",
        len(cleaned),
        len(pages_text),
    )
    return cleaned


def _clean_text(text: str) -> str:
    """Normalize extracted PDF text."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def load_pdf_text(
    pdf_path: str,
    settings: PDFSettings,
) -> str:
    """Public entry point: validate + extract full PDF text.

    Args:
        pdf_path: Path to PDF file.
        settings: Security/config settings.

    Returns:
        Cleaned full text string.

    Raises:
        PDFSecurityError: On access violations.
        PDFExtractionError: On extraction failures.
    """
    validated_path = _validate_pdf_path(pdf_path, settings)
    return _extract_full_text(validated_path)
