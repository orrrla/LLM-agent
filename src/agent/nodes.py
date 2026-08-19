# -*- coding: utf-8 -*-
import hashlib
import os
import re
from functools import lru_cache
from typing import Any

from src.agent.router import Route, RouteDecision, route_query
from src.agent.state import AnswerResult, ChatState
from src.agent.tools.clarify import make_clarify_question


_RETRIEVAL_ROUTES = {
    Route.OPERATION,
    Route.FAQ,
    Route.DIAGNOSTIC,
    Route.COMPARISON,
    Route.MULTI_HOP,
}

_NO_ANSWER_MARKERS = ("无答案", "无法得到答案", "没有相关信息", "未找到", "找不到答案")


def _used_query(state: ChatState) -> str:
    return (state.rewritten_query or state.query).strip()


def _merge_tool_results(state: ChatState, **updates: Any) -> dict[str, Any]:
    merged = dict(state.tool_results)
    merged.update(updates)
    return merged


def rewrite_node(state: ChatState) -> dict[str, Any]:
    """Rewrite pronouns and enrich profile terms using existing profile logic."""

    try:
        from src.profile.context_engineering import rewrite_query_with_profile
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError("rewrite_node unavailable: profile context module failed to import") from exc

    query = state.query.strip()
    if not query:
        return {"rewritten_query": query, "tool_results": _merge_tool_results(state, rewrite={"skipped": True})}

    rewritten_query = rewrite_query_with_profile(
        query,
        state.profile or {},
        state.recent_turns or [],
    )
    return {
        "rewritten_query": rewritten_query or query,
        "tool_results": _merge_tool_results(state, rewrite={"query": rewritten_query or query}),
    }


def memory_node(state: ChatState) -> dict[str, Any]:
    """Load relevant long-term memories when Mem0 mode is enabled."""

    if os.environ.get("ENABLE_MEM0", "0") != "1":
        return {
            "tool_results": _merge_tool_results(state, memory={"skipped": True}),
        }

    try:
        from src.memory.mem0_store import Mem0MemoryStore
    except Exception as exc:  # pragma: no cover - depends on host environment
        raise RuntimeError("memory_node unavailable: Mem0 module failed to import") from exc

    store = Mem0MemoryStore(user_id=state.user_id)
    memories = store.search(state.query or state.rewritten_query or "", limit=5)
    memory_context = "\n".join(
        f"- {record.memory}" for record in memories if record.memory
    )
    recent_turns = list(state.recent_turns or [])
    recent_turns.extend(
        {"query": record.memory, "answer": ""} for record in memories if record.memory
    )

    return {
        "memory_context": memory_context or "无",
        "recent_turns": recent_turns[-5:],
        "tool_results": _merge_tool_results(
            state,
            memory={
                "enabled": True,
                "record_count": len(memories),
                "user_id": state.user_id,
            },
        ),
    }


def route_node(state: ChatState) -> dict[str, Any]:
    """Classify the query and attach an optional clarification question."""

    decision = route_query(
        state.query,
        profile=state.profile or {},
        recent_turns=state.recent_turns or [],
    )
    tool_results = _merge_tool_results(
        state,
        route={
            "route": decision.route.value,
            "reason": decision.reason,
            "tool_calls": decision.tool_calls,
        },
    )

    if decision.needs_clarification:
        clarify_state = ChatState(
            user_id=state.user_id,
            query=state.query,
            rewritten_query=state.rewritten_query,
            profile=state.profile,
            recent_turns=state.recent_turns,
            route=decision.route.value,
        )
        tool_results["clarify_question"] = make_clarify_question(clarify_state)

    return {
        "route": decision.route.value,
        "needs_clarification": decision.needs_clarification,
        "tool_results": tool_results,
    }


@lru_cache(maxsize=1)
def _get_bm25_retriever() -> Any:
    try:
        from src.retriever.bm25_retriever import BM25
        return BM25(docs=None, retrieve=True)
    except Exception as exc:
        raise RuntimeError("BM25 retriever unavailable; check BM25 index and jieba/stopwords data") from exc


@lru_cache(maxsize=1)
def _get_milvus_retriever() -> Any:
    try:
        from src.retriever.milvus_retriever import MilvusRetriever
        return MilvusRetriever(docs=None, retrieve=True)
    except Exception as exc:
        raise RuntimeError("Milvus retriever unavailable; check Milvus Lite index and Mongo connection") from exc


def _doc_key(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    unique_id = metadata.get("unique_id")
    if unique_id:
        return str(unique_id)
    content = getattr(doc, "page_content", "") or ""
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def _deduplicate_docs(docs: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for doc in docs:
        key = _doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


def _retrieve_with(query: str, retriever: Any, topk: int) -> list[Any]:
    try:
        return list(retriever.retrieve_topk(query, topk=topk))
    except Exception as exc:
        raise RuntimeError(f"retriever failed for query {query!r}: {exc}") from exc


def retrieve_node(state: ChatState) -> dict[str, Any]:
    """Call only BM25 and Milvus, then deduplicate and merge their results."""

    if state.route not in {route.value for route in _RETRIEVAL_ROUTES}:
        return {"text_docs": [], "tool_results": _merge_tool_results(state, retrieval={"skipped": True})}

    bm25 = _get_bm25_retriever()
    milvus = _get_milvus_retriever()
    queries = {state.query}
    used_query = _used_query(state)
    if used_query and used_query != state.query:
        queries.add(used_query)

    docs: list[Any] = []
    bm25_count = 0
    milvus_count = 0
    for query in sorted(queries):
        if not query:
            continue
        bm25_hits = _retrieve_with(query, bm25, topk=10)
        bm25_count += len(bm25_hits)
        docs.extend(bm25_hits)

        milvus_hits = _retrieve_with(query, milvus, topk=10)
        milvus_count += len(milvus_hits)
        docs.extend(milvus_hits)

    docs = _deduplicate_docs(docs)
    return {
        "text_docs": docs,
        "tool_results": _merge_tool_results(
            state,
            retrieval={
                "bm25_count": bm25_count,
                "milvus_count": milvus_count,
                "merged_count": len(docs),
            },
        ),
    }


@lru_cache(maxsize=1)
def _get_reranker() -> Any:
    try:
        from src.constant import bge_reranker_tuned_model_path
        from src.reranker.bge_m3_reranker import BGEM3ReRanker

        model_path = os.environ.get("AGENT_RERANKER_MODEL_PATH", bge_reranker_tuned_model_path)
        return BGEM3ReRanker(model_path=model_path)
    except Exception as exc:
        raise RuntimeError("BGEM3ReRanker unavailable; check reranker model path and CUDA environment") from exc


def rerank_node(state: ChatState) -> dict[str, Any]:
    """Re-rank retrieved docs with the existing BGE-M3 reranker."""

    if not state.text_docs:
        return {"tool_results": _merge_tool_results(state, rerank={"skipped": True})}

    reranker = _get_reranker()
    used_query = _used_query(state) or state.query
    topk = min(5, len(state.text_docs))
    try:
        ranked_docs = reranker.rank(used_query, state.text_docs, topk=topk)
    except Exception as exc:
        raise RuntimeError(f"rerank failed for query {used_query!r}: {exc}") from exc

    return {
        "text_docs": list(ranked_docs),
        "tool_results": _merge_tool_results(state, rerank={"topk": topk, "ranked_count": len(ranked_docs)}),
    }


def _build_context(docs: list[Any]) -> str:
    return "\n".join(
        f"【{idx + 1}】{getattr(doc, 'page_content', '')}" for idx, doc in enumerate(docs)
    )


def _parse_citation_numbers(raw_response: str) -> list[int]:
    citation_blocks = re.findall(r"【([^】]*)】", raw_response)
    numbers: list[int] = []
    seen: set[int] = set()
    for block in citation_blocks:
        for token in re.split(r"[，,、\s]+", block):
            token = token.strip().lstrip("#")
            if token.isdigit():
                number = int(token)
                if number > 0 and number not in seen:
                    seen.add(number)
                    numbers.append(number)
    return numbers


def _clean_answer(raw_response: str) -> str:
    answer = re.sub(r"【[^】]*】", "", raw_response)
    return answer.strip()


def _is_no_answer(answer: str) -> bool:
    return any(marker in answer for marker in _NO_ANSWER_MARKERS)


def _collect_images(docs: list[Any], citation_numbers: list[int]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for number in citation_numbers:
        if number > len(docs):
            continue
        metadata = getattr(docs[number - 1], "metadata", {}) or {}
        for image in metadata.get("images_info", []) or []:
            if not isinstance(image, dict):
                continue
            title = image.get("title", "")
            image_path = image.get("image_path", "")
            key = (title, image_path)
            if key in seen:
                continue
            seen.add(key)
            images.append(
                {
                    "title": title,
                    "image_path": image_path,
                    "page": image.get("page", metadata.get("page")),
                }
            )
    return images


def generate_node(state: ChatState) -> dict[str, Any]:
    """Generate the answer with the local Qwen3 vLLM OpenAI-compatible client."""

    used_query = state.query.strip() or _used_query(state)
    if not used_query:
        return {
            "answer": "",
            "citations": [],
            "confidence": 0.0,
            "tool_results": _merge_tool_results(state, generate={"skipped": True}),
        }

    try:
        from src.client.llm_local_client import request_chat
    except Exception as exc:
        raise RuntimeError("generate_node unavailable: llm_local_client failed to import") from exc

    context = _build_context(state.text_docs)
    try:
        raw_response = request_chat(
            used_query,
            context,
            stream=False,
            profile=state.profile or {},
            recent_turns=state.recent_turns or [],
            memory_context=state.memory_context,
        )
    except Exception as exc:
        raise RuntimeError(f"local LLM request failed for query {used_query!r}") from exc

    response_text = raw_response if isinstance(raw_response, str) else str(raw_response)
    citation_numbers = _parse_citation_numbers(response_text)
    answer = _clean_answer(response_text)
    images = _collect_images(state.text_docs, citation_numbers)

    if _is_no_answer(answer):
        confidence = 0.05
        citations: list[str] = []
        images = []
    else:
        citations = [str(number) for number in citation_numbers]
        confidence = 0.85 if citations else 0.25

    answer_result = AnswerResult(
        answer=answer,
        citations=citations,
        images=images,
        confidence=confidence,
        needs_clarification=state.needs_clarification,
    )
    tool_results = _merge_tool_results(state, generate={"answer_result": answer_result})
    return {
        "answer": answer,
        "citations": citations,
        "visual_docs": images,
        "confidence": confidence,
        "tool_results": tool_results,
    }


def guardrail_node(state: ChatState) -> dict[str, Any]:
    """Adjust confidence when an answer has no citations or says it cannot answer."""

    confidence = state.confidence

    if _is_no_answer(state.answer):
        confidence = min(confidence, 0.05)

    if not state.citations and state.route in {route.value for route in _RETRIEVAL_ROUTES}:
        confidence = min(confidence, 0.2)

    return {
        "confidence": max(0.0, confidence),
        "tool_results": _merge_tool_results(
            state,
            guardrail={
                "confidence": max(0.0, confidence),
                "needs_clarification": state.needs_clarification,
            },
        ),
    }
