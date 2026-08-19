# -*- coding: utf-8 -*-
import re


PRONOUN_PATTERN = re.compile(r"(它|这个|该功能|这个功能|这项功能|怎么弄|如何操作|怎么开|怎么关)")


def build_profile_terms(profile):
    model_cfg = profile.get("model_cfg", "").strip()
    software_version = profile.get("software_version", "").strip()
    profile_terms = []
    if model_cfg:
        profile_terms.extend([t.strip() for t in re.split(r"[,，/\s]+", model_cfg) if t.strip()])
    if software_version:
        profile_terms.extend([software_version])
    return profile_terms


def rewrite_query_with_profile(query, profile, recent_turns):
    terms = build_profile_terms(profile)
    rewritten_query = query.strip()

    if terms and terms[0] not in rewritten_query:
        rewritten_query = f"{terms[0]} {rewritten_query}"

    if recent_turns and PRONOUN_PATTERN.search(query):
        last_query = recent_turns[-1].get("query", "").strip()
        if last_query:
            rewritten_query = f"{rewritten_query}（结合上个问题：{last_query}）"

    return rewritten_query


def _calc_profile_score(text, profile):
    text_lower = text.lower()
    terms = build_profile_terms(profile)
    if not terms:
        return 0.0

    score = 0.0
    for term in terms:
        if term.lower() in text_lower:
            score += 0.15
    score = min(score, 0.45)

    model_cfg = profile.get("model_cfg", "").lower()
    if "model 3" in model_cfg and "model y" in text_lower:
        score -= 0.1
    if "model y" in model_cfg and "model 3" in text_lower:
        score -= 0.1
    return score


def profile_soft_rerank(docs, profile):
    if not profile:
        return docs
    scored = []
    for idx, doc in enumerate(docs):
        profile_score = _calc_profile_score(doc.page_content, profile)
        scored.append((profile_score, -idx, doc))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored]

