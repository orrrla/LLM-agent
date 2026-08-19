# Parallel Module Integration

This branch adds three disjoint slices to the existing Tesla manual RAG project.
The original `infer.py` fixed pipeline remains untouched.

## Run The Agent Pipeline

```bash
python agent_infer.py --route-only --query "怎么打开离车后自动上锁"
python agent_infer.py --query "怎么打开离车后自动上锁" --profile-json data/profile.json
```

The first command validates routing without loading model or database
dependencies. The second runs the full graph and requires the existing BM25,
Milvus, BGE-M3 reranker, and local Qwen3 vLLM service.

## Module Contracts

- `src/agent/` owns routing, graph execution, retrieval/rerank/generation nodes.
- `src/vision/` owns PDF page rendering, ColPali indexing, visual retrieval, VLM
  captioning, and text/visual fusion.
- `src/eval/` owns golden-set loading, retrieval/citation/answer metrics,
  regression reporting, and synthetic case generation hooks.
- `src/observability/` owns in-process tracing, feedback persistence, and an
  optional Langfuse client.

The slices only import one another through stable interfaces. The final fusion
point is the `retrieve_node` in `src/agent/nodes.py`, which can later call
`src.vision.page_retriever.PageRetriever` and `src.vision.fusion.fuse_results`.

## Enable Mem0 Memory

Set `ENABLE_MEM0=1` and provide `MEM0_CONFIG_JSON`, then run:

```bash
export ENABLE_MEM0=1
export MEM0_API_KEY="your-api-key"
export MEM0_CONFIG_JSON='{"llm":{"provider":"openai","config":{"model":"gpt-4o-mini"}}}'

python agent_infer.py --memory-backend mem0 --query "怎么打开离车后自动上锁"
```

The `memory_node` is inserted before query rewriting and injects relevant
Mem0 memories into the LLM prompt through `llm_local_client.request_chat`.

## Validate The Slices

```bash
python scripts/validate_modules.py
```

The validation script checks expected files and compiles every source under
`src/agent`, `src/vision`, `src/eval`, and `src/observability`.
