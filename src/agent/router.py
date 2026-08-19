# -*- coding: utf-8 -*-
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Route(str, Enum):
    """High-level routing labels for a user query."""

    OPERATION = "operation"
    FAQ = "faq"
    DIAGNOSTIC = "diagnostic"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    OUT_OF_SCOPE = "out_of_scope"
    CHITCHAT = "chitchat"


@dataclass
class RouteDecision:
    route: Route
    needs_clarification: bool
    tool_calls: list[str] = field(default_factory=list)
    reason: str = ""


# Keep the rule engine self-contained; it only depends on the public regex exposed
# by the existing profile module, which is itself lightweight.
from src.profile.context_engineering import PRONOUN_PATTERN  # noqa: E402


_CHITCHAT_PATTERNS = (
    "你好",
    "谢谢",
    "再见",
    "你是谁",
    "你能做什么",
    "你会什么",
    "介绍一下你",
)

_OUT_OF_SCOPE_PATTERNS = (
    "天气",
    "股票",
    "新闻",
    "电影",
    "美食",
    "音乐",
    "旅游",
    "房价",
    "足球",
    "篮球",
)

_CAR_TERM_PATTERNS = (
    "车",
    "model",
    "tesla",
    "特斯拉",
    "电池",
    "充电",
    "驾驶",
    "车窗",
    "车门",
    "空调",
    "屏幕",
    "钥匙",
    "轮胎",
    "软件",
    "更新",
)

_OPERATION_PATTERNS = (
    "怎么",
    "如何",
    "怎样",
    "步骤",
    "操作",
    "打开",
    "关闭",
    "设置",
    "调节",
    "开启",
    "启用",
    "使用",
    "设置",
    "重置",
    "解锁",
    "锁定",
)

_DIAGNOSTIC_PATTERNS = (
    "故障",
    "报错",
    "错误",
    "无法",
    "不能",
    "失灵",
    "不工作",
    "坏了",
    "异响",
    "警告",
    "异常",
    "怎么修",
    "解决",
    "失效",
    "没反应",
)

_COMPARISON_PATTERNS = (
    "对比",
    "区别",
    "差别",
    "哪个",
    "哪一种",
    "比较",
    "vs",
    "versus",
)

_MULTI_HOP_PATTERNS = (
    "然后",
    "接着",
    "之后",
    "同时",
    "以及",
    "另外",
    "并且",
)

_MODEL_SPECIFIC_ROUTES = {
    Route.OPERATION,
    Route.DIAGNOSTIC,
    Route.COMPARISON,
    Route.MULTI_HOP,
}

_MODEL_NAMES = ("model 3", "model3", "model y", "modely", "model s", "model x")


def _has_any(query_lower: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in query_lower for pattern in patterns)


def _detect_route(query: str) -> Route:
    query_lower = query.lower().strip()
    if not query_lower:
        return Route.FAQ

    if _has_any(query_lower, _CHITCHAT_PATTERNS):
        return Route.CHITCHAT

    has_car_term = _has_any(query_lower, _CAR_TERM_PATTERNS)
    if _has_any(query_lower, _OUT_OF_SCOPE_PATTERNS) and not has_car_term:
        return Route.OUT_OF_SCOPE

    if _has_any(query_lower, _COMPARISON_PATTERNS):
        return Route.COMPARISON

    if _has_any(query_lower, _DIAGNOSTIC_PATTERNS):
        return Route.DIAGNOSTIC

    if query.count("?") > 1 or query.count("？") > 1:
        return Route.MULTI_HOP
    if _has_any(query_lower, _MULTI_HOP_PATTERNS):
        return Route.MULTI_HOP

    if _has_any(query_lower, _OPERATION_PATTERNS):
        return Route.OPERATION

    return Route.FAQ


def _needs_clarification(query: str, profile: dict[str, Any], recent_turns: list[dict[str, Any]], route: Route) -> bool:
    query_lower = query.lower()
    pronoun_hit = bool(PRONOUN_PATTERN.search(query))
    if pronoun_hit and not recent_turns:
        return True

    model_cfg = (profile or {}).get("model_cfg", "").strip()
    model_mentions = [name for name in _MODEL_NAMES if name in query_lower]

    if model_mentions and model_cfg:
        primary_mention = model_mentions[0]
        if primary_mention not in model_cfg.lower():
            return True

    if route in _MODEL_SPECIFIC_ROUTES and not model_cfg and not model_mentions:
        return True

    return False


def route_query(
    query: str,
    profile: dict[str, Any] | None = None,
    recent_turns: list[dict[str, Any]] | None = None,
) -> RouteDecision:
    """Classify a query with deterministic, explainable rules.

    The default is FAQ. Pronouns without conversation history and model-sensitive
    questions without a stored ``model_cfg`` are flagged for clarification.
    """

    profile = profile or {}
    recent_turns = recent_turns or []
    route = _detect_route(query)
    needs_clarification = _needs_clarification(query, profile, recent_turns, route)

    if route in (Route.CHITCHAT, Route.OUT_OF_SCOPE):
        tool_calls: list[str] = []
        reason = "chitchat or out-of-scope input does not require retrieval"
    else:
        tool_calls = ["bm25", "milvus"]
        reason = "retrieve from lexical and hybrid manual indexes"

    if needs_clarification:
        tool_calls = ["clarify"]
        reason = "pronoun reference or model configuration requires clarification"

    return RouteDecision(
        route=route,
        needs_clarification=needs_clarification,
        tool_calls=tool_calls,
        reason=reason,
    )
