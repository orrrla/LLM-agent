# -*- coding: utf-8 -*-
from __future__ import annotations

from difflib import SequenceMatcher
from math import log2
from typing import Any, Callable, Dict, List, Optional, Sequence


NO_ANSWER_VALUES = {"", "无答案", "无", "none", "no answer", "n/a"}


def _to_id(value: Any) -> str:
    if hasattr(value, "metadata") and isinstance(getattr(value, "metadata"), dict):
        metadata = value.metadata
        for key in ("unique_id", "doc_id", "id"):
            if metadata.get(key) is not None:
                return str(metadata[key])
    return str(value)


def _as_ids(values: Optional[Sequence[Any]]) -> List[str]:
    if not values:
        return []
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        identifier = _to_id(value).strip()
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        result.append(identifier)
    return result


def _is_no_answer(value: Any) -> bool:
    return str(value or "").strip().lower() in NO_ANSWER_VALUES


def _normalize_text(value: Any) -> str:
    return "".join(str(value or "").strip().lower().split())


def lexical_similarity(left: str, right: str) -> float:
    """Dependency-free fallback similarity for short Chinese/English answers."""

    left_text = _normalize_text(left)
    right_text = _normalize_text(right)
    if not left_text or not right_text:
        return 1.0 if left_text == right_text else 0.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def retrieval_metrics(
    predicted_doc_ids: Optional[Sequence[Any]],
    gold_doc_ids: Optional[Sequence[Any]],
    k: int = 5,
) -> Dict[str, float]:
    """Compute recall@k, NDCG@k and hit-rate against a gold doc-id set."""

    if k < 1:
        raise ValueError("k must be >= 1")
    predicted = _as_ids(predicted_doc_ids)[:k]
    gold = set(_as_ids(gold_doc_ids))

    if not gold:
        score = 1.0 if not predicted else 0.0
        return {"recall_at_k": score, "ndcg_at_k": score, "hit_rate": score}

    hits = [1 if doc_id in gold else 0 for doc_id in predicted]
    hit_rate = 1.0 if any(hits) else 0.0
    recall = sum(hits) / len(gold)
    dcg = sum(gain / log2(index + 2) for index, gain in enumerate(hits))
    ideal_length = min(k, len(gold))
    idcg = sum(1.0 / log2(index + 2) for index in range(ideal_length))
    ndcg = dcg / idcg if idcg else 0.0
    return {"recall_at_k": recall, "ndcg_at_k": ndcg, "hit_rate": hit_rate}


def citation_metrics(
    pred_citations: Optional[Sequence[Any]],
    gold_citations: Optional[Sequence[Any]],
) -> Dict[str, float]:
    """Precision/recall/F1 for citation identifiers.

    The empty/empty case is treated as a perfect no-citation match. Producing
    unsupported citations or missing required citations scores zero.
    """

    pred = set(_as_ids(pred_citations))
    gold = set(_as_ids(gold_citations))

    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred or not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = len(pred & gold)
    precision = intersection / len(pred)
    recall = intersection / len(gold)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def answer_score(
    pred_answer: Any,
    gold_answer: Any,
    similarity_fn: Optional[Callable[[str, str], float]] = None,
) -> float:
    """Score an answer with an injectable similarity function.

    ``similarity_fn`` must accept two normalized strings and return a float.
    If omitted, :func:`lexical_similarity` is used and no model is imported.
    """

    pred_no_answer = _is_no_answer(pred_answer)
    gold_no_answer = _is_no_answer(gold_answer)
    if pred_no_answer and gold_no_answer:
        return 1.0
    if pred_no_answer or gold_no_answer:
        return 0.0

    scorer = similarity_fn or lexical_similarity
    score = float(scorer(_normalize_text(pred_answer), _normalize_text(gold_answer)))
    return max(0.0, min(1.0, score))
