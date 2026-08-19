# Observability

轻量可观测性模块，覆盖评测 trace、用户反馈和可选 Langfuse 接入。没有 OpenTelemetry 硬依赖，外部系统通过 exporter hook 或可替换的 `LangfuseClient` 接入。

## 安装

```bash
pip install -r src/eval/requirements-eval.txt
```

本模块不要求额外依赖；只有实际对接真实 Langfuse SDK 时才需要安装 `langfuse`。

## Tracer

```python
from src.observability import Tracer

tracer = Tracer(exporter=lambda record: print(record))
tracer.span("retrieval", {"query": "离车后自动上锁"}, {"doc_ids": ["chunk-123"]})
tracer.log_event("user_feedback", {"rating": 5})

with tracer.start_span("generation", {"context": "..."}) as span:
    span.set_outputs({"answer": "..."})

trace_payload = tracer.to_dict()
```

`span(name, inputs, outputs)` 立即记录一个已完成 span，`start_span` 提供计时和 context-manager 用法。`exporter` 接收每个 span/event 的 JSON-serializable record，可替换成日志、Kafka 或远程 trace 后端。

## FeedbackStore

```python
from src.observability import FeedbackRecord, FeedbackStore

store = FeedbackStore("data/feedback/feedback.jsonl")
store.append(FeedbackRecord(
    user_id="default_user",
    case_id="model3_locking_001",
    query="Model 3 离车后如何自动上锁？",
    rating=5,
    reason="引用正确",
    trace_id="abc",
))
store.to_jsonl()
```

`append` 接受 `FeedbackRecord` 或 mapping；传入 `path` 时会立即追加 JSONL，`to_jsonl()` 可显式落盘。`case_id` 与 `query` 至少保留其一，方便把反馈关联回 golden case 或现场 query。

## LangfuseClient

标准环境变量为 `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY`、`LANGFUSE_HOST`。未配置 key 时 `LangfuseClient.from_env()` 返回 `None`；`LANGFUSE_ENABLED=0` 时返回显式禁用实例。

```python
from src.observability import LangfuseClient

client = LangfuseClient.from_env()
if client:
    client.trace("eval_run", {"case_id": "model3_locking_001"})
```

这个实现是不带真实 SDK 的本地 stub，不写 key、不发送网络请求。生产环境可以保留相同接口，把类替换为真实 `Langfuse` 客户端。

## CI 门禁

```bash
python -m compileall src/observability
```
