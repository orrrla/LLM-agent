# -*- coding: utf-8 -*-
from collections.abc import Callable
from typing import Any

from src.agent.state import ChatState


class AgentGraph:
    """Minimal sequential graph with no external orchestration dependency."""

    def __init__(self) -> None:
        self._nodes: list[tuple[str, Callable[[ChatState], dict[str, Any] | None]]] = []

    def add_node(self, name: str, func: Callable[[ChatState], dict[str, Any] | None]) -> None:
        if not name:
            raise ValueError("node name must be a non-empty string")
        if any(existing_name == name for existing_name, _ in self._nodes):
            raise ValueError(f"duplicate node name: {name}")
        self._nodes.append((name, func))

    def run(self, state: ChatState, run_trace: list[str] | None = None) -> ChatState:
        """Execute nodes in insertion order and return the mutated state.

        ``run_trace`` is optional. When omitted, ``state.trace`` is used. A node
        may return ``None`` (no update) or a ``dict`` whose keys correspond to
        fields on :class:`ChatState`.
        """

        trace = state.trace if run_trace is None else run_trace
        if trace is not state.trace:
            state.trace = trace

        for name, func in self._nodes:
            trace.append(name)
            updates = func(state)
            if updates is None:
                continue
            if not isinstance(updates, dict):
                raise TypeError(
                    f"node {name!r} returned {type(updates).__name__}; "
                    "expected dict[str, Any] or None"
                )
            for key, value in updates.items():
                if not hasattr(state, key):
                    raise AttributeError(
                        f"node {name!r} returned unknown ChatState field {key!r}"
                    )
                setattr(state, key, value)

        return state
