# -*- coding: utf-8 -*-
"""Multimodal PDF page retrieval data models.

These dataclasses intentionally stay small and dependency-free so they can be
used by the parser, retriever, and fusion layers without importing model or
database libraries at module load time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class VisualPage:
    """One rendered PDF page and its page-level vector metadata."""

    page_id: str
    page_num: int
    image_path: Optional[str] = None
    page_vector: Optional[List[float]] = None
    caption: Optional[str] = None
    tables_json: Optional[Any] = None
    regions_json: Optional[Any] = None
    section_title: Optional[str] = None
    score: float = 0.0


@dataclass
class VisualAnswerSegment:
    """A visually grounded answer segment returned after page-level matching."""

    page_num: int
    image_path: Optional[str] = None
    bbox: Optional[Any] = None
    caption: Optional[str] = None
    score: float = 0.0
