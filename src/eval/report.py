# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .schemas import EvalResult


_METRIC_FIELDS = (
    "retrieval_recall_at_5",
    "citation_precision",
    "citation_recall",
    "answer_score",
    "faithfulness",
)

DEFAULT_REGRESSION_THRESHOLDS: Dict[str, float] = {
    "retrieval_recall_at_5": 0.05,
    "citation_precision": 0.05,
    "citation_recall": 0.05,
    "answer_score": 0.05,
    "faithfulness": 0.05,
    "latency_ms": 100.0,
}


def results_to_markdown(
    results: Sequence[EvalResult],
    summary: Optional[Mapping[str, Any]] = None,
    title: str = "Evaluation Results",
) -> str:
    lines = [
        f"## {title}",
        "",
        "| case_id | recall@5 | cit_precision | cit_recall | answer | faithfulness | latency_ms | tokens | passed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.1f} | {} | {} |".format(
                result.case_id,
                result.retrieval_recall_at_5,
                result.citation_precision,
                result.citation_recall,
                result.answer_score,
                result.faithfulness,
                result.latency_ms,
                result.token_usage,
                "PASS" if result.passed else "FAIL",
            )
        )
    if summary:
        lines.extend(["", "### Summary", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```"])
    return "\n".join(lines) + "\n"


def results_to_json(
    results: Sequence[EvalResult],
    summary: Optional[Mapping[str, Any]] = None,
    indent: int = 2,
) -> str:
    payload: Any
    if summary is None:
        payload = [result.to_dict() for result in results]
    else:
        payload = {
            "summary": summary,
            "results": [result.to_dict() for result in results],
        }
    return json.dumps(payload, ensure_ascii=False, indent=indent)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare_runs(
    baseline: Sequence[EvalResult],
    current: Sequence[EvalResult],
    regression_thresholds: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Compare two runs and list per-case regressions exceeding thresholds."""

    thresholds = dict(DEFAULT_REGRESSION_THRESHOLDS)
    if regression_thresholds:
        thresholds.update(regression_thresholds)

    baseline_by_id = {result.case_id: result for result in baseline}
    current_by_id = {result.case_id: result for result in current}
    common_ids = sorted(set(baseline_by_id) & set(current_by_id))

    regressions: List[Dict[str, Any]] = []
    for case_id in common_ids:
        old = baseline_by_id[case_id]
        new = current_by_id[case_id]
        for metric in (*_METRIC_FIELDS, "latency_ms"):
            old_value = float(getattr(old, metric))
            new_value = float(getattr(new, metric))
            delta = new_value - old_value
            threshold = float(thresholds.get(metric, 0.0))
            is_regression = delta > threshold if metric == "latency_ms" else delta < -threshold
            if is_regression:
                regressions.append(
                    {
                        "case_id": case_id,
                        "metric": metric,
                        "baseline": old_value,
                        "current": new_value,
                        "delta": delta,
                        "threshold": threshold,
                    }
                )

    summary_delta: Dict[str, Any] = {}
    for metric in (*_METRIC_FIELDS, "latency_ms", "token_usage"):
        old_mean = _mean([float(getattr(result, metric)) for result in baseline])
        new_mean = _mean([float(getattr(result, metric)) for result in current])
        summary_delta[metric] = {
            "baseline_mean": old_mean,
            "current_mean": new_mean,
            "delta": new_mean - old_mean,
        }
    summary_delta["pass_rate"] = {
        "baseline_mean": _mean([1.0 if result.passed else 0.0 for result in baseline]),
        "current_mean": _mean([1.0 if result.passed else 0.0 for result in current]),
    }
    summary_delta["pass_rate"]["delta"] = (
        summary_delta["pass_rate"]["current_mean"]
        - summary_delta["pass_rate"]["baseline_mean"]
    )

    return {
        "baseline_count": len(baseline),
        "current_count": len(current),
        "matched_count": len(common_ids),
        "missing_in_current": sorted(set(baseline_by_id) - set(current_by_id)),
        "new_in_current": sorted(set(current_by_id) - set(baseline_by_id)),
        "thresholds": thresholds,
        "summary_delta": summary_delta,
        "regressions": regressions,
    }


def regressions_to_markdown(comparison: Mapping[str, Any]) -> str:
    regressions = comparison.get("regressions", [])
    if not regressions:
        return "No regressions exceeded configured thresholds.\n"
    lines = [
        "## Regressions",
        "",
        "| case_id | metric | baseline | current | delta | threshold |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in regressions:
        lines.append(
            "| {} | {} | {:.4f} | {:.4f} | {:+.4f} | {:.4f} |".format(
                item["case_id"],
                item["metric"],
                item["baseline"],
                item["current"],
                item["delta"],
                item["threshold"],
            )
        )
    return "\n".join(lines) + "\n"


def write_report(
    results: Sequence[EvalResult],
    markdown_path: Optional[str | Path] = None,
    json_path: Optional[str | Path] = None,
    summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    written: Dict[str, str] = {}
    if markdown_path:
        target = Path(markdown_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(results_to_markdown(results, summary=summary), encoding="utf-8")
        written["markdown"] = str(target)
    if json_path:
        target = Path(json_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(results_to_json(results, summary=summary), encoding="utf-8")
        written["json"] = str(target)
    return written
