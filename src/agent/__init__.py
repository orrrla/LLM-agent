# -*- coding: utf-8 -*-
"""Agentic RAG orchestration package.

The package intentionally keeps heavy runtime imports out of module import time.
Retrievers, rerankers and LLM clients are imported lazily by nodes so importing
``src.agent`` remains cheap and can be done even when model dependencies are
not yet available on the host.
"""

from src.agent.graph import AgentGraph
from src.agent.router import Route, RouteDecision, route_query
from src.agent.state import AnswerResult, ChatState

__all__ = [
    "AgentGraph",
    "AnswerResult",
    "ChatState",
    "Route",
    "RouteDecision",
    "route_query",
]
