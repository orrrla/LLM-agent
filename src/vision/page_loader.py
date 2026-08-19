# -*- coding: utf-8 -*-
"""Render PDF pages to PNG with PyMuPDF.

PyMuPDF is imported lazily so importing this module does not require the native
library until rendering is actually requested.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


def _load_fitz():
    """Return PyMuPDF, with a clear error when it is not installed."""
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "page_loader requires PyMuPDF. Install dependencies from "
            "src/vision/requirements-vision.txt first."
        ) from exc
    return fitz


def _pixmap_is_blank(pix, threshold: int = 245) -> bool:
    """Detect an essentially empty page by sampling rendered bytes.

    Sampling keeps blank detection cheap even at high DPI. A threshold of 245
    is deliberately conservative: any dark text, line, or image pixel makes the
    page non-empty.
    """
    samples = pix.samples
    if not samples:
        return True

    channels = max(1, int(getattr(pix, "n", 1)))
    pixel_count = len(samples) // channels
    if pixel_count == 0:
        return True

    step = max(1, pixel_count // 200_000)
    for pixel_idx in range(0, pixel_count, step):
        start = pixel_idx * channels
        if any(samples[start + channel] < threshold for channel in range(channels)):
            return False
    return True


def _validate_page_bounds(page_num: int, page_count: int) -> None:
    if page_num < 1 or page_num > page_count:
        raise IndexError(
            f"page_num {page_num} is out of range for a PDF with {page_count} pages"
        )


def _render_doc_page(
    doc,
    page_num: int,
    output_dir: str,
    dpi: int,
    skip_empty: bool,
    overwrite: bool,
    fitz,
) -> Optional[str]:
    """Render one 1-based page from an already-open PDF document."""
    page_count = doc.page_count
    _validate_page_bounds(page_num, page_count)

    output_path = os.path.join(output_dir, f"page_{page_num:04d}.png")
    if not overwrite and os.path.exists(output_path):
        return output_path

    page = doc.load_page(page_num - 1)
    try:
        # PyMuPDF's native resolution unit is 72 DPI.
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    except Exception as exc:
        raise RuntimeError(f"failed to render page {page_num}: {exc}") from exc

    if skip_empty and _pixmap_is_blank(pix):
        logger.warning("Skipping blank page %d", page_num)
        return None

    pix.save(output_path)
    return output_path


def render_page(
    pdf_path: str,
    page_num: int,
    output_dir: str,
    dpi: int = 150,
    skip_empty: bool = True,
    overwrite: bool = False,
) -> Optional[str]:
    """Render a single 1-based PDF page and return its PNG path."""
    pdf_path = os.path.abspath(os.path.expanduser(pdf_path))
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer")

    fitz = _load_fitz()
    try:
        with fitz.open(pdf_path) as doc:
            return _render_doc_page(
                doc, page_num, output_dir, dpi, skip_empty, overwrite, fitz
            )
    except (IndexError, FileNotFoundError, ValueError, RuntimeError, ImportError):
        raise
    except Exception as exc:
        raise RuntimeError(f"failed to open or render PDF {pdf_path}: {exc}") from exc


def render_pages(
    pdf_path: str,
    output_dir: str,
    dpi: int = 150,
    start_page: Optional[int] = None,
    end_page: Optional[int] = None,
    skip_empty: bool = True,
    overwrite: bool = False,
) -> List[str]:
    """Render a range of pages and return only successfully rendered PNG paths.

    Pages are numbered from 1. Invalid pages are skipped and logged instead of
    aborting a large batch, while errors such as a missing PDF or bad DPI still
    raise.
    """
    pdf_path = os.path.abspath(os.path.expanduser(pdf_path))
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    os.makedirs(output_dir, exist_ok=True)
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer")

    fitz = _load_fitz()
    rendered: List[str] = []
    with fitz.open(pdf_path) as doc:
        page_count = doc.page_count
        first = max(1, start_page or 1)
        last = min(page_count, end_page or page_count)

        for page_num in range(first, last + 1):
            try:
                path = _render_doc_page(
                    doc,
                    page_num,
                    output_dir,
                    dpi,
                    skip_empty,
                    overwrite,
                    fitz,
                )
                if path is not None:
                    rendered.append(path)
            except (IndexError, RuntimeError) as exc:
                logger.warning("Skipping page %d: %s", page_num, exc)

    return rendered
