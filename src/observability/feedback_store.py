# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union


@dataclass
class FeedbackRecord:
    user_id: str
    case_id: Optional[str] = None
    query: Optional[str] = None
    rating: Optional[float] = None
    reason: Optional[str] = None
    trace_id: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FeedbackStore:
    """Append-only feedback store with JSONL persistence."""

    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else None
        self._records: List[FeedbackRecord] = []
        if self.path and self.path.exists():
            self._load(self.path)

    def _load(self, path: Path) -> None:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = FeedbackRecord(**json.loads(line))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid feedback at {path}:{line_number}") from exc
                self._records.append(record)

    def _coerce(self, record: Union[FeedbackRecord, Mapping[str, Any]]) -> FeedbackRecord:
        if isinstance(record, FeedbackRecord):
            return record
        if not isinstance(record, Mapping):
            raise TypeError("feedback must be FeedbackRecord or mapping")
        data = dict(record)
        data.pop("metadata", None)
        return FeedbackRecord(**data)

    def append(self, record: Union[FeedbackRecord, Mapping[str, Any]]) -> FeedbackRecord:
        feedback = self._coerce(record)
        if not feedback.user_id.strip():
            raise ValueError("feedback.user_id is required")
        if not feedback.created_at:
            feedback.created_at = datetime.now(timezone.utc).isoformat()
        self._records.append(feedback)
        if self.path:
            self._append_line(self.path, feedback)
        return feedback

    def _append_line(self, path: Path, record: FeedbackRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def to_jsonl(self, path: Optional[str | Path] = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("no path specified")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return target

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> List[FeedbackRecord]:
        return list(self._records)
