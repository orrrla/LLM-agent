# Eval

持续评测模块，负责把手工维护的 golden set 变成可重复的回归指标、报告和 CI 门禁。模块不依赖具体检索器或模型权重；运行器只需要把一个 case 映射为一段评测结果。

## 安装

```bash
pip install -r src/eval/requirements-eval.txt
```

默认指标只使用 Python 标准库，不引入新的硬依赖。需要语义相似度时，把 `text2vec` 或 `sentence-transformers` 的相似度函数通过 `metrics_config["similarity_fn"]` 注入，而不是在本模块内部启动模型。

## Golden 数据格式

支持 `.json`（数组或单个对象）和 `.jsonl`/`.ndjson`。必填字段是 `case_id` 和 `query`；其余字段可省略。常用旧字段别名也会被接受：`question` -> `query`，`answer` -> `gold_answer`，`chunks` -> `gold_chunks`，`pages` -> `required_pages`。

```json
[
  {
    "case_id": "model3_locking_001",
    "query": "Model 3 离车后如何自动上锁？",
    "profile": {"model_cfg": "Model 3"},
    "expected_route": "retrieval",
    "gold_answer": "携带手机钥匙离开车辆后，车辆会自动上锁。",
    "gold_chunks": ["chunk-123", "chunk-124"],
    "required_pages": [7, 8],
    "must_cite": true,
    "allow_no_answer": false
  }
]
```

`gold_chunks` 建议填写文档稳定 ID（例如 `metadata["unique_id"]`），`required_pages` 作为 golden citation set。

## 运行方式

```python
from src.eval import load_golden_set, run_eval
from src.eval.report import write_report

def runner(case):
    return {
        "predicted_doc_ids": ["chunk-123"],
        "pred_answer": "携带手机钥匙离开车辆后，车辆会自动上锁。",
        "pred_citations": [7],
        "latency_ms": 124.0,
        "token_usage": 380,
        "trace": {"some": "trace"},
    }

cases = load_golden_set("data/eval/golden.json")
results, summary = run_eval(cases, runner)
write_report(results, "reports/eval.md", "reports/eval.json", summary=summary)
```

`runner(case)` 返回的映射需要包含 `predicted_doc_ids`、`pred_answer`、`pred_citations`、`latency_ms`、`token_usage`，可选 `trace` 和 `faithfulness`。

## CI 门禁

```bash
python -m compileall src/eval
python - <<'PY'
from src.eval import load_golden_set, run_eval
cases = load_golden_set("data/eval/golden.json")
results, summary = run_eval(cases, runner)
assert summary["pass_rate"] >= 0.8, summary
PY
```

默认通过阈值见 `run_eval.DEFAULT_THRESHOLDS`。无引用要求的 case 只检查检索、答案和 faithfulness，`must_cite=true` 或存在 `required_pages` 时额外检查 citation precision/recall。

## 结果与回归对比

`results_to_markdown`/`results_to_json` 输出当前 run；`compare_runs` 对比 baseline 与 current，并标记超过阈值的退化项。默认允许的指标回退阈值为 0.05，延迟回退阈值为 100ms。
