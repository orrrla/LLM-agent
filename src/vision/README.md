# 多模态 PDF 页面检索模块

本目录是页面级视觉检索模块，独立于现有 `src/parser`、`src/retriever`、`src/agent` 和 `src/eval`。它把 PDF 页面渲染为 PNG，使用 ColPali 生成页面向量并写入 Milvus，再通过 Milvus 返回页面，最后与文本检索结果做融合。

## 安装

```bash
pip install -r src/vision/requirements-vision.txt
```

不要修改仓库根目录的 `requirements.txt`。模型权重也由人工提前准备，本模块不会自动下载权重。

## 模型准备

`ColPaliIndexer` 使用本地模型目录，建议设置环境变量：

```bash
export COLPAI_MODEL_PATH=/root/autodl-tmp/RAG/models/colpali-v1.2
export COLPAI_DEVICE=cuda
export MILVUS_URI=./data/saved_index/vision_milvus.db
```

如果 `COLPAI_MODEL_PATH` 为空或目录不存在，`build_from_pages()` 和 `encode_query()` 会抛出明确错误，不会静默返回假向量。

## 渲染页面

```python
from src.vision import render_pages

paths = render_pages(
    pdf_path="data/Tesla_Manual.pdf",
    output_dir="data/rendered_pages",
    dpi=150,
    start_page=1,
    end_page=20,
)
```

空页默认跳过，单页异常会记录 warning 后继续处理剩余页面。

## 构建视觉索引

```python
from src.vision import ColPaliIndexer, VisualPage

pages = [
    VisualPage(page_id="page_0001", page_num=1, image_path="data/rendered_pages/page_0001.png"),
    # ... 其他页面
]

indexer = ColPaliIndexer(
    model_path="/path/to/colpali-v1.2",
    device="cuda",
    collection_name="visual_pages_colpali",
    milvus_uri="./data/saved_index/vision_milvus.db",
)
count = indexer.build_from_pages(pages, drop_if_exists=True)
```

Milvus schema 字段由 `ColPaliIndexer.get_schema_fields(dim)` 提供：`page_id`、`page_num`、`image_path`、`caption`、`tables_json`、`regions_json`、`section_title` 和 `page_vector`。

## 页面查询

```python
from src.vision import PageRetriever

retriever = PageRetriever(
    indexer=indexer,
    collection_name="visual_pages_colpali",
)
pages = retriever.search("如何更换空调滤芯", topk=5)
```

Milvus 未连接、集合不存在或模型未准备好时，会抛出带上下文说明的异常。

## 视觉描述与区域提取

```python
from src.vision import generate_caption, extract_regions

caption = generate_caption("data/rendered_pages/page_0012.png", "简要描述本页内容")
regions = extract_regions("data/rendered_pages/page_0012.png")
```

默认读取 `VLM_API_KEY`、`VLM_BASE_URL`、`VLM_MODEL`（也兼容 `OPENAI_API_KEY` 等变量）。API key 为空时直接报错，不返回伪结果。

## 与文本检索融合

```python
from src.vision import fuse_results

merged = fuse_results(text_docs, visual_pages, weights={"text": 0.6, "visual": 0.4})
```

`text_docs` 保持原有 `Document` 类型，`visual_pages` 保持 `VisualPage` 类型。融合只返回新列表，不修改输入对象。

## 集成到 src/agent/retrieve_node

当前仓库里没有 `src/agent/` 目录时，建议在调用检索节点的外层做组合，不改动已有文件：

1. 用现有文本检索器获取 `text_docs`。
2. 用 `PageRetriever.search()` 获取 `visual_pages`。
3. 调用 `fuse_results()` 得到混合结果，再传给 LLM 或后续节点。

如果目标仓库后续出现 `src/agent/retrieve_node.py`，可以把第 2、3 步作为该节点的可选视觉分支；新增分支通过 `Optional[List[VisualPage]]` 返回，避免改动原文本返回协议。
