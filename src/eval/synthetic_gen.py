# -*- coding: utf-8 -*-
from __future__ import annotations

from itertools import cycle, islice
from typing import Callable, Iterable, List, Sequence, Union

from .golden_loader import case_from_dict
from .schemas import EvalCase


def generate_synthetic_cases(
    golden_cases: Sequence[EvalCase],
    generator_fn: Callable[[EvalCase], Iterable[EvalCase]],
    n: int = 20,
) -> List[EvalCase]:
    """Generate up to ``n`` synthetic cases by cycling through golden cases.

    ``generator_fn`` receives one golden :class:`EvalCase` and returns an
    iterable of new cases. This module deliberately performs no LLM call; the
    caller supplies the model adapter.
    """

    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if not golden_cases:
        return []

    generated: List[EvalCase] = []
    seen_ids: set[str] = set()
    max_attempts = max(len(golden_cases) * 2, n * 2, 2)

    for golden_case in islice(cycle(golden_cases), max_attempts):
        try:
            candidates = generator_fn(golden_case)
        except Exception:
            # A single bad generator call should not abort the whole run.
            continue
        if candidates is None:
            continue
        if isinstance(candidates, EvalCase):
            candidates = [candidates]
        for candidate in candidates:
            if isinstance(candidate, EvalCase):
                case = candidate
            else:
                case = case_from_dict(candidate)
            if case.case_id in seen_ids:
                continue
            seen_ids.add(case.case_id)
            generated.append(case)
            if len(generated) >= n:
                return generated
    return generated
