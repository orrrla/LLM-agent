# Mem0 Long-Term Memory

This module wraps Mem0 behind a small `Mem0MemoryStore` adapter.

## Configure

Set a `MEM0_CONFIG_JSON` value with the Mem0 LLM and vector-store settings, or
provide the equivalent configuration to `Mem0MemoryStore(config=...)`.

Example:

```bash
export ENABLE_MEM0=1
export MEM0_API_KEY="your-api-key"
export MEM0_CONFIG_JSON='{"llm":{"provider":"openai","config":{"model":"gpt-4o-mini"}}}'
```

The exact vector-store and LLM configuration follows the Mem0 version being
used. For a fully local setup, configure Mem0 with a local Qdrant or Chroma
store and an OpenAI-compatible LLM endpoint.

## Use In Code

```python
from src.memory.mem0_store import Mem0MemoryStore

store = Mem0MemoryStore(user_id="default_user")
store.add_turn("怎么打开离车后自动上锁", "点击控制 > 车锁 > 离车后自动上锁。")
memories = store.search("离车后自动上锁", limit=5)
```
