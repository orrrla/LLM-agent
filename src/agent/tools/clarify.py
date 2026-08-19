# -*- coding: utf-8 -*-
import re

from src.agent.state import ChatState


_PRONOUN_HINT_PATTERN = re.compile(r"(它|这个|该功能|这个功能|这项功能)")
_MODEL_SENSITIVE_ROUTES = {"operation", "diagnostic", "comparison", "multi_hop"}


def make_clarify_question(state: ChatState) -> str:
    """Return one concise clarification question for the current state.

    The rule engine first disambiguates pronouns, then asks for a missing model
    configuration, and finally falls back to a generic prompt.
    """

    query = state.query.strip()
    if not query:
        return "请问你想了解什么？"

    profile = state.profile or {}
    recent_turns = state.recent_turns or []
    has_pronoun = bool(_PRONOUN_HINT_PATTERN.search(query))

    if has_pronoun and not recent_turns:
        return "你指的是哪一个功能或操作？请补充具体的功能名称或场景。"

    model_cfg = profile.get("model_cfg", "").strip()
    if state.route in _MODEL_SENSITIVE_ROUTES and not model_cfg:
        return "请补充你的 Model 3 具体车型或软件版本，这样我能给出更准确的答案。"

    model_mentions = [name for name in ("model y", "model s", "model x", "model3", "model 3") if name in query.lower()]
    if model_mentions and model_cfg and model_mentions[0] not in model_cfg.lower():
        return f"你提到了 {model_mentions[0]}，但当前画像中的车型是 {model_cfg}；请确认以哪个车型为准。"

    return "这个问题还需要更多信息才能准确回答，请补充一下具体场景。"
