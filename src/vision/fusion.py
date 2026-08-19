# -*- coding: utf-8 -*-
"""Fuse text and visual retrieval results into one ranked mixed list."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.vision.schemas import VisualPage


DEFAULT_WEIGHTS: Dict[str, float] = {"text": 0.6, "visual": 0.4}


def _get_text_score(doc: Any, rank: int) -> float:
    metadata = getattr(doc, "metadata", {}) or {}
    for key in ("score", "rerank_score", "_score", "similarity", "distance"):
        value = metadata.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return float(rank)


def _get_visual_score(page: VisualPage, rank: int) -> float:
    if isinstance(page.score, (int, float)) and not isinstance(page.score, bool):
        return float(page.score)
    return float(rank)


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def fuse_results(
    text_docs: Sequence[Any],
    visual_pages: Sequence[VisualPage],
    weights: Optional[Dict[str, float]] = None,
) -> List[Any]:
    """Merge text Documents and VisualPage objects into one ranked list.

    Scores are normalized independently per modality before applying weights,
    which avoids mixing retrieval scores that may have very different ranges.
    The returned list preserves the original object types.
    """
    text_items = list(text_docs)
    visual_items = list(visual_pages)

    weights = weights or DEFAULT_WEIGHTS
    try:
        text_weight = float(weights.get("text", DEFAULT_WEIGHTS["text"]))
        visual_weight = float(weights.get("visual", DEFAULT_WEIGHTS["visual"]))
    except (TypeError, ValueError) as exc:
        raise ValueError("weights['text'] and weights['visual'] must be numbers") from exc

    if text_weight < 0 or visual_weight < 0:
        raise ValueError("fusion weights must be non-negative")
    if text_weight + visual_weight <= 0:
        raise ValueError("at least one fusion weight must be positive")

    text_raw_scores = [
        _get_text_score(doc, len(text_items) - index)
        for index, doc in enumerate(text_items)
    ]
    visual_raw_scores = [
        _get_visual_score(page, len(visual_items) - index)
        for index, page in enumerate(visual_items)
    ]

    text_norm = _normalize_scores(text_raw_scores)
    visual_norm = _normalize_scores(visual_raw_scores)

    fused: List[Tuple[float, int, Any]] = []
    for index, (doc, norm_score) in enumerate(zip(text_items, text_norm)):
        fused.append((norm_score * text_weight, index, doc))
    for index, (page, norm_score) in enumerate(zip(visual_items, visual_norm)):
        fused.append((norm_score * visual_weight, index, page))

    fused.sort(key=lambda item: (-item[0], item[1]))
    return [item for _, _, item in fused]
