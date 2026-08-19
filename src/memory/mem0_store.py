# -*- coding: utf-8 -*-
"""Mem0-backed long-term memory for the Agentic RAG pipeline.

The Mem0 client is imported lazily so the rest of the repository can be
imported without installing ``mem0ai``. Configure Mem0 through the
``MEM0_CONFIG_JSON`` environment variable or pass a ``config`` dict directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryRecord:
    """Normalized representation of one Mem0 memory."""

    id: Optional[str] = None
    memory: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class Mem0MemoryStore:
    """Small adapter around Mem0's ``Memory`` client.

    Mem0 performs its own extraction, deduplication and retrieval. This class
    only adds a stable project-facing contract and graceful failure handling.
    """

    def __init__(
        self,
        user_id: str = "default_user",
        config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.user_id = user_id or "default_user"
        self._api_key = api_key or os.getenv("MEM0_API_KEY")
        self._config = config or self._load_config()
        self._client: Any = None

    @staticmethod
    def _load_config() -> Dict[str, Any]:
        raw = os.getenv("MEM0_CONFIG_JSON", "").strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("MEM0_CONFIG_JSON must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("MEM0_CONFIG_JSON must be a JSON object")
        return payload

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        try:
            from mem0 import Memory  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Mem0 is not installed. Install src/memory/requirements-memory.txt "
                "or pip install mem0ai."
            ) from exc

        if self._api_key:
            os.environ.setdefault("MEM0_API_KEY", self._api_key)

        if self._config:
            self._client = Memory.from_config(self._config)
        else:
            self._client = Memory()
        return self._client

    def _normalize_results(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict) and "results" in payload:
            payload = payload.get("results", [])
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def add_turn(self, query: str, answer: str) -> List[Dict[str, Any]]:
        """Persist one user/assistant turn through Mem0 extraction."""

        if not query.strip() or not answer.strip():
            return []

        messages = [
            {"role": "user", "content": query.strip()},
            {"role": "assistant", "content": answer.strip()},
        ]
        try:
            result = self._get_client().add(messages, user_id=self.user_id)
        except TypeError:
            # Older Mem0 versions accepted a single string instead of messages.
            result = self._get_client().add(
                f"用户问题：{query.strip()}\n系统回答：{answer.strip()}",
                user_id=self.user_id,
            )
        return self._normalize_results(result)

    def search(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        if not query.strip():
            return []
        payload = self._get_client().search(
            query.strip(),
            user_id=self.user_id,
            limit=max(1, int(limit)),
        )
        records: List[MemoryRecord] = []
        for item in self._normalize_results(payload):
            metadata = item.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            records.append(
                MemoryRecord(
                    id=item.get("id"),
                    memory=str(item.get("memory", "")),
                    created_at=item.get("created_at"),
                    updated_at=item.get("updated_at"),
                    score=float(item.get("score", 0.0) or 0.0),
                    metadata=metadata,
                )
            )
        return records

    def get_all(self, limit: int = 50) -> List[MemoryRecord]:
        payload = self._get_client().get_all(
            user_id=self.user_id,
            limit=max(1, int(limit)),
        )
        records: List[MemoryRecord] = []
        for item in self._normalize_results(payload):
            records.append(
                MemoryRecord(
                    id=item.get("id"),
                    memory=str(item.get("memory", "")),
                    created_at=item.get("created_at"),
                    updated_at=item.get("updated_at"),
                    score=float(item.get("score", 0.0) or 0.0),
                    metadata=item.get("metadata") or {},
                )
            )
        return records

    def build_context(self, query: str, limit: int = 5) -> str:
        records = self.search(query, limit=limit)
        if not records:
            return "无"
        return "\n".join(
            f"- {record.memory}" for record in records if record.memory
        )
