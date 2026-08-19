# Agentic RAG skeleton

This directory contains a small, framework-free orchestration layer for the
existing Tesla Model 3 manual RAG pipeline. It does not modify or replace the
current modules in `src/retriever`, `src/reranker`, `src/client`, `src/profile`,
`src/vision`, or `src/eval`.

## Structure

- `state.py`: `ChatState` and `AnswerResult` dataclasses.
- `router.py`: deterministic `route_query` and `Route`/`RouteDecision`.
- `graph.py`: `AgentGraph` with `add_node(name, func)` and `run(state)`.
- `nodes.py`: the six graph nodes used by a full agent run.
- `tools/clarify.py`: rule-based clarification questions.
- `requirements-agent.txt`: no additional hard dependencies are needed.

## Node flow

`rewrite_node -> route_node -> retrieve_node -> rerank_node -> generate_node ->
guardrail_node`

The graph executes nodes in insertion order. Each node receives the current
`ChatState` and returns either `None` or a `dict` of `ChatState` field updates.
Every executed node name is appended to `state.trace` (or a caller-provided
`run_trace` list).

## Run

Run from the repository root so the existing `src.*` absolute imports resolve:

```bash
cd /path/to/LLM-agent-src
python - <<'PY'
from src.agent.graph import AgentGraph
from src.agent.nodes import (
    guardrail_node,
    generate_node,
    rerank_node,
    retrieve_node,
    rewrite_node,
    route_node,
)
from src.agent.state import ChatState

graph = AgentGraph()
for node in (rewrite_node, route_node, retrieve_node, rerank_node, generate_node, guardrail_node):
    graph.add_node(node.__name__, node)

state = ChatState(query="怎么打开车窗", profile={"model_cfg": "Model 3"})
state = graph.run(state)
print(state.answer)
print(state.citations)
print(state.trace)
PY
```

The retrieval, reranking, and generation nodes use lazy imports and one-time
module-level caches. Heavy models are only loaded when the corresponding node is
first executed. The vLLM endpoint defaults to `http://localhost:8000/v1`.

## Integration with `infer.py`

`infer.py` currently owns interactive profile setup, warm-starting, retrieval,
reranking, streaming generation, and post-processing. The agent layer is intended
to sit behind it without changing that file:

1. Build the same `ChatState` that `infer.py` already collects:
   `user_id`, `query`, `profile`, and `recent_turns`.
2. Register the six nodes in the order above.
3. Call `AgentGraph.run(state)` instead of the inline loop.
4. Persist the returned answer and `needs_clarification` flag with
   `UserProfileStore.append_turn` and `UserProfileStore.upsert_profile`, exactly
   as `infer.py` already does.

For a minimal adapter:

```python
active_profile = profile_store.get_profile(user_id)
recent_turns = profile_store.get_recent_turns(user_id)
state = ChatState(
    user_id=user_id,
    query=query,
    profile=active_profile,
    recent_turns=recent_turns,
)
state = graph.run(state)
```

The graph deliberately does not call `src.utils.post_processing` directly; the
generate/guardrail nodes extract `answer`, citation numbers, and image metadata
into `ChatState`/`AnswerResult` so callers can keep using or replace existing
post-processing behavior.
