#!/usr/bin/env python3
"""CLI entrypoint for the new Agentic RAG graph.

This is intentionally separate from ``infer.py`` so the original fixed pipeline
remains the default until the Agent graph is validated against the golden set.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from src.agent import AgentGraph, ChatState
from src.agent import nodes
from src.observability.tracer import Tracer


def build_graph(enable_mem0: bool = False) -> AgentGraph:
    graph = AgentGraph()
    if enable_mem0:
        graph.add_node("memory", nodes.memory_node)
    graph.add_node("rewrite", nodes.rewrite_node)
    graph.add_node("route", nodes.route_node)
    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("rerank", nodes.rerank_node)
    graph.add_node("generate", nodes.generate_node)
    graph.add_node("guardrail", nodes.guardrail_node)
    return graph


def load_profile(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("profile JSON must be an object")
    return payload


def route_only(query: str, profile: dict[str, Any]) -> None:
    from src.agent.router import route_query

    decision = route_query(query, profile=profile)
    print(json.dumps(
        {
            "query": query,
            "route": decision.route.value,
            "needs_clarification": decision.needs_clarification,
            "tool_calls": decision.tool_calls,
            "reason": decision.reason,
        },
        ensure_ascii=False,
        indent=2,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="run a single query and exit")
    parser.add_argument("--user-id", default="default_user")
    parser.add_argument("--profile-json", help="optional profile JSON file")
    parser.add_argument(
        "--memory-backend",
        choices=["none", "mem0"],
        default=os.environ.get("MEMORY_BACKEND", "none"),
        help="long-term memory backend",
    )
    parser.add_argument(
        "--route-only",
        action="store_true",
        help="only run the deterministic router, without loading model/database dependencies",
    )
    parser.add_argument("--trace-out", help="write the JSON trace to this path")
    args = parser.parse_args()

    profile = load_profile(args.profile_json)
    use_mem0 = args.memory_backend == "mem0" or os.environ.get("ENABLE_MEM0") == "1"
    if use_mem0:
        os.environ["ENABLE_MEM0"] = "1"

    if args.route_only:
        if not args.query:
            raise SystemExit("--route-only requires --query")
        route_only(args.query, profile)
        return

    graph = build_graph(enable_mem0=use_mem0)
    state = ChatState(
        user_id=args.user_id,
        query=args.query or "",
        profile=profile,
    )
    tracer = Tracer()

    if args.query:
        final_state = graph.run(state)
        tracer.log_event("run_finished", {"trace": final_state.trace})
        if args.trace_out:
            with open(args.trace_out, "w", encoding="utf-8") as handle:
                json.dump(tracer.to_dict(), handle, ensure_ascii=False, indent=2)
        if final_state.needs_clarification:
            clarify = (final_state.tool_results or {}).get("clarify_question", "请补充更多信息。")
            print(f"追问：{clarify}")
        else:
            print(f"答案：{final_state.answer}")
            print(f"引用：{final_state.citations}")
            print(f"置信度：{final_state.confidence}")
        return

    # Interactive mode mirrors infer.py's simple loop.
    while True:
        query = input("输入—>").strip()
        if query.lower() in {"exit", "quit"}:
            break
        state.query = query
        final_state = graph.run(state)
        if final_state.needs_clarification:
            clarify = (final_state.tool_results or {}).get("clarify_question", "请补充更多信息。")
            print(f"追问：{clarify}")
        else:
            print(f"答案：{final_state.answer}")
            print(f"引用：{final_state.citations}")
            print(f"置信度：{final_state.confidence}")
        print("=" * 100)


if __name__ == "__main__":
    main()
