# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvalCase:
    """One golden evaluation case.

    ``gold_chunks`` are expected to be stable document identifiers (normally
    ``metadata["unique_id"]`` values). ``required_pages`` is used as the golden
    citation set by the default runner.
    """

    case_id: str
    query: str
    profile: Dict[str, Any] = field(default_factory=dict)
    expected_route: Optional[str] = None
    gold_answer: str = ""
    gold_chunks: List[str] = field(default_factory=list)
    required_pages: List[int] = field(default_factory=list)
    must_cite: bool = False
    allow_no_answer: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvalResult:
    case_id: str
    retrieval_recall_at_5: float = 0.0
    citation_precision: float = 0.0
    citation_recall: float = 0.0
    answer_score: float = 0.0
    faithfulness: float = 0.0
    latency_ms: float = 0.0
    token_usage: int = 0
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
