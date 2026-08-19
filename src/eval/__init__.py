# -*- coding: utf-8 -*-
"""Evaluation primitives for the Tesla manual RAG system.

The module is intentionally decoupled from the inference path. A runner only
needs to expose a small dictionary contract, so it can wrap any current or
future inference implementation without importing ``src.agent``/``src.vision``.
"""

from .schemas import EvalCase, EvalResult
from .golden_loader import case_from_dict, load_golden_set
from .metrics import answer_score, citation_metrics, lexical_similarity, retrieval_metrics
from .run_eval import DEFAULT_THRESHOLDS, run_eval, run_single
from .synthetic_gen import generate_synthetic_cases

__all__ = [
    "EvalCase",
    "EvalResult",
    "case_from_dict",
    "load_golden_set",
    "answer_score",
    "citation_metrics",
    "lexical_similarity",
    "retrieval_metrics",
    "DEFAULT_THRESHOLDS",
    "run_eval",
    "run_single",
    "generate_synthetic_cases",
]
