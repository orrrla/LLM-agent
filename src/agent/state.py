# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatState:
    """Mutable state passed through the sequential agent graph."""

    user_id: str = "default_user"
    query: str = ""
    rewritten_query: str | None = None
    profile: dict[str, Any] = field(default_factory=dict)
    recent_turns: list[dict[str, Any]] = field(default_factory=list)
    route: str | None = None
    sub_queries: list[str] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    text_docs: list[Any] = field(default_factory=list)
    visual_docs: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    confidence: float = 1.0
    needs_clarification: bool = False
    trace: list[str] = field(default_factory=list)


@dataclass
class AnswerResult:
    """Final answer payload extracted from the generation node."""

    answer: str
    citations: list[str] = field(default_factory=list)
    images: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    needs_clarification: bool = False
