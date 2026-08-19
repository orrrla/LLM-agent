# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .metrics import answer_score, citation_metrics, retrieval_metrics, _is_no_answer
from .schemas import EvalCase, EvalResult


logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "retrieval_recall_at_5": 0.5,
    "citation_precision": 0.5,
    "citation_recall": 0.5,
    "answer_score": 0.6,
    "faithfulness": 0.5,
}


def _config_value(config: Optional[Mapping[str, Any]], key: str, default: Any) -> Any:
    if not config:
        return default
    return config.get(key, default)


def _failure_result(case_id: str) -> EvalResult:
    return EvalResult(case_id=case_id)


def _default_faithfulness(
    case: EvalCase,
    pred_answer: Any,
    pred_citations: Sequence[Any],
    citation_precision: float,
    citation_recall: float,
) -> float:
    pred_no_answer = _is_no_answer(pred_answer)
    gold_no_answer = _is_no_answer(case.gold_answer)
    if gold_no_answer:
        return 1.0 if pred_no_answer else 0.0
    if pred_no_answer:
        return 0.0
    if case.must_cite or case.required_pages:
        return (citation_precision + citation_recall) / 2
    if pred_citations:
        return citation_precision
    return 1.0


def _passed(
    case: EvalCase,
    result: EvalResult,
    thresholds: Optional[Mapping[str, float]],
) -> bool:
    if not thresholds:
        return True

    checks: Dict[str, float] = {
        "retrieval_recall_at_5": result.retrieval_recall_at_5,
        "answer_score": result.answer_score,
        "faithfulness": result.faithfulness,
    }
    if case.must_cite or case.required_pages:
        checks["citation_precision"] = result.citation_precision
        checks["citation_recall"] = result.citation_recall

    for metric, minimum in thresholds.items():
        if metric in checks and checks[metric] < float(minimum):
            return False
    return True


def run_single(
    case: EvalCase,
    runner: Callable[[EvalCase], Mapping[str, Any]],
    metrics_config: Optional[Mapping[str, Any]] = None,
) -> EvalResult:
    """Run one golden case through a callable runner.

    ``runner(case)`` must return a mapping with:
    ``predicted_doc_ids``, ``pred_answer``, ``pred_citations``, ``latency_ms``,
    ``token_usage`` and optional ``trace``/``faithfulness``.
    """

    try:
        output = runner(case)
    except Exception:
        logger.exception("runner failed for case %s", case.case_id)
        return _failure_result(case.case_id)
    if not isinstance(output, Mapping):
        logger.warning("runner returned non-mapping output for case %s", case.case_id)
        return _failure_result(case.case_id)

    predicted_doc_ids = output.get("predicted_doc_ids", [])
    pred_answer = output.get("pred_answer", "")
    pred_citations = output.get("pred_citations", [])
    latency_ms = output.get("latency_ms", 0.0)
    token_usage = output.get("token_usage", 0)

    k = int(_config_value(metrics_config, "k", 5))
    retrieval = retrieval_metrics(predicted_doc_ids, case.gold_chunks, k=k)
    citations = citation_metrics(pred_citations, case.required_pages)
    similarity_fn = _config_value(metrics_config, "similarity_fn", None)
    answer = answer_score(pred_answer, case.gold_answer, similarity_fn=similarity_fn)

    supplied_faithfulness = output.get("faithfulness")
    if supplied_faithfulness is not None:
        try:
            faithfulness = max(0.0, min(1.0, float(supplied_faithfulness)))
        except (TypeError, ValueError):
            faithfulness = 0.0
    else:
        faithfulness_fn = _config_value(metrics_config, "faithfulness_fn", None)
        if faithfulness_fn is not None:
            try:
                faithfulness = max(
                    0.0,
                    min(
                        1.0,
                        float(
                            faithfulness_fn(
                                case,
                                output,
                                citations["precision"],
                                citations["recall"],
                            )
                        ),
                    ),
                )
            except Exception:
                logger.exception("faithfulness_fn failed for case %s", case.case_id)
                faithfulness = 0.0
        else:
            faithfulness = _default_faithfulness(
                case,
                pred_answer,
                pred_citations,
                citations["precision"],
                citations["recall"],
            )

    thresholds = _config_value(metrics_config, "thresholds", DEFAULT_THRESHOLDS)
    result = EvalResult(
        case_id=case.case_id,
        retrieval_recall_at_5=retrieval["recall_at_k"],
        citation_precision=citations["precision"],
        citation_recall=citations["recall"],
        answer_score=answer,
        faithfulness=faithfulness,
        latency_ms=float(latency_ms),
        token_usage=int(token_usage),
    )
    result.passed = _passed(case, result, thresholds)
    return result


def run_eval(
    cases: Sequence[EvalCase],
    runner: Callable[[EvalCase], Mapping[str, Any]],
    metrics_config: Optional[Mapping[str, Any]] = None,
) -> tuple[List[EvalResult], Dict[str, Any]]:
    """Evaluate all cases and return results plus a compact summary."""

    results = [run_single(case, runner, metrics_config) for case in cases]
    pass_count = sum(1 for result in results if result.passed)

    metric_fields = [
        "retrieval_recall_at_5",
        "citation_precision",
        "citation_recall",
        "answer_score",
        "faithfulness",
        "latency_ms",
        "token_usage",
    ]
    means: Dict[str, float] = {}
    for field_name in metric_fields:
        values = [float(getattr(result, field_name)) for result in results]
        means[field_name] = sum(values) / len(values) if values else 0.0

    summary: Dict[str, Any] = {
        "case_count": len(results),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(results) if results else 0.0,
        "mean": means,
        "failed_case_ids": [
            result.case_id for result in results if not result.passed
        ],
    }
    return results, summary
