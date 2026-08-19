# -*- coding: utf-8 -*-
"""Multimodal PDF page retrieval module.

Heavy dependencies are imported lazily by the individual submodules, so
`import src.vision` itself does not load transformers, torch, PyMuPDF,
pymilvus, or OpenAI.
"""

from src.vision.schemas import VisualAnswerSegment, VisualPage
from src.vision.page_loader import render_page, render_pages
from src.vision.colpali_index import ColPaliIndexer
from src.vision.page_retriever import PageRetriever
from src.vision.vlm_caption import VLMCaptioner, extract_regions, generate_caption
from src.vision.fusion import fuse_results

__all__ = [
    "VisualPage",
    "VisualAnswerSegment",
    "render_page",
    "render_pages",
    "ColPaliIndexer",
    "PageRetriever",
    "VLMCaptioner",
    "generate_caption",
    "extract_regions",
    "fuse_results",
]
