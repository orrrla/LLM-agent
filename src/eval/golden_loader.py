# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .schemas import EvalCase


_ALIASES: Dict[str, Sequence[str]] = {
    "case_id": ("case_id", "id", "case"),
    "query": ("query", "question"),
    "profile": ("profile",),
    "expected_route": ("expected_route", "route"),
    "gold_answer": ("gold_answer", "answer", "reference"),
    "gold_chunks": ("gold_chunks", "chunks", "gold_doc_ids"),
    "required_pages": ("required_pages", "pages", "cite_pages"),
    "must_cite": ("must_cite",),
    "allow_no_answer": ("allow_no_answer",),
}


def _first(data: Mapping[str, Any], key: str) -> Any:
    for candidate in _ALIASES[key]:
        if candidate in data:
            return data[candidate]
    return None


def _as_nonempty_str(value: Any, field_name: str) -> str:
    if value is None:
        raise ValueError(f"missing required field: {field_name}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"field must be non-empty: {field_name}")
    return text


def _as_bool(value: Any, field_name: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_string_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"field must be a list: {field_name}")
    return [str(item) for item in value]


def _as_int_list(value: Any, field_name: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (int, str)):
        try:
            return [int(value)]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"field must contain integers: {field_name}") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"field must be a list: {field_name}")
    result: List[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"field must contain integers: {field_name}") from exc
    return result


def case_from_dict(data: Mapping[str, Any]) -> EvalCase:
    """Build an :class:`EvalCase` from a mapping.

    Required fields are ``case_id`` and ``query``. Optional fields use the
    defaults declared on the dataclass. Common legacy aliases such as
    ``question``/``answer`` are accepted to ease migration from the existing
    QA-pair JSON files.
    """

    if not isinstance(data, Mapping):
        raise ValueError("each golden case must be a JSON object")

    profile = _first(data, "profile") or {}
    if not isinstance(profile, Mapping):
        raise ValueError("field must be an object: profile")

    raw_route = _first(data, "expected_route")
    expected_route = (
        _as_nonempty_str(raw_route, "expected_route")
        if raw_route is not None and str(raw_route).strip()
        else None
    )

    return EvalCase(
        case_id=_as_nonempty_str(_first(data, "case_id"), "case_id"),
        query=_as_nonempty_str(_first(data, "query"), "query"),
        profile=dict(profile),
        expected_route=expected_route,
        gold_answer=str(_first(data, "gold_answer") or "").strip(),
        gold_chunks=_as_string_list(_first(data, "gold_chunks"), "gold_chunks"),
        required_pages=_as_int_list(_first(data, "required_pages"), "required_pages"),
        must_cite=_as_bool(_first(data, "must_cite"), "must_cite"),
        allow_no_answer=_as_bool(_first(data, "allow_no_answer"), "allow_no_answer"),
    )


def _read_json_records(path: Path) -> Iterable[Any]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        return

    with path.open("r", encoding="utf-8") as handle:
        text = handle.read().lstrip()
    if not text:
        return
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # A .json file may actually contain one JSON object per line.
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
        return
    if isinstance(payload, list):
        yield from payload
    else:
        yield payload


def load_golden_set(path: str | Path) -> List[EvalCase]:
    """Load and validate golden cases from a JSON array or JSONL file."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"golden set not found: {source}")

    cases: List[EvalCase] = []
    for index, record in enumerate(_read_json_records(source), 1):
        try:
            cases.append(case_from_dict(record))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid golden case at index {index}: {exc}") from exc
    return cases
