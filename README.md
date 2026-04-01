# Tesla 知识引擎 · 车书问答系统

> 基于 RAG（检索增强生成）技术，针对 Tesla Model 3 用户手册构建的智能问答系统。
> 完整覆盖 **PDF 解析 → 文本清洗 → 索引构建 → 多路召回 → 重排序 → LLM 生成答案** 全链路。

---

## 系统架构

```
Tesla_Manual.pdf
       │
       ▼
 ┌─────────────┐
 │  PDF 解析   │  PyMuPDF 提取文本 + 图片
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  文本清洗   │  LLM（Doubao/DeepSeek）语义清洗
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  文本切分   │  语义切分服务（FastAPI）
 └──────┬──────┘
        │
     ┌──┴──────────────────┐
     │                     │
     ▼                     ▼
┌─────────┐         ┌────────────┐
│  BM25   │         │  Milvus DB │  BGE-M3 稠密+稀疏向量
│ 稀疏召回 │         │  稠密召回  │  RRF 初排融合
└────┬────┘         └─────┬──────┘
     │                    │
     └──────────┬──────────┘
                ▼
        ┌──────────────┐
        │  去重 & 合并  │
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │   精排模型   │  BGE-M3 Reranker（微调后）
        │              │  / Qwen3-Reranker-4B
        └──────┬───────┘
               ▼
        ┌──────────────┐
        │  LLM 生成    │  Qwen3-8B（LoRA SFT 微调 + INT4 量化）
        │              │  通过 vLLM 本地部署
        └──────┬───────┘
               ▼
           答案 + 引用编号
```

---

## 目录结构

```
车书问答系统/
├── data/
│   ├── Tesla_Manual.pdf          # Tesla Model 3 用户手册（原始数据）
│   ├── stopwords.txt             # 中文停用词表
│   ├── processed_docs/           # PDF 解析 & 清洗后的中间结果
│   ├── saved_index/              # 向量索引 & BM25 索引（运行后生成）
│   ├── qa_pairs/                 # 问答对数据集
│   ├── rerank_data/              # Reranker 训练数据
│   └── summary_data/             # SFT 微调训练数据
│
├── src/
│   ├── parser/                   # PDF 解析模块
│   ├── client/                   # LLM 调用客户端（Doubao / vLLM / HyDE）
│   ├── retriever/                # 检索模块（BM25 / TF-IDF / Faiss / Milvus）
│   ├── reranker/                 # 重排序模块（BGE-M3 / Qwen3-Reranker）
│   ├── server/                   # 语义切分服务（FastAPI）
│   ├── gen_qa/                   # 问答对生成模块
│   ├── fields/                   # 字段定义
│   ├── constant.py               # 全局路径 & 常量配置
│   └── utils.py                  # 工具函数
│
├── LLaMA-Factory-main/           # 模型微调框架（LLaMA-Factory）
│   └── data/                     # 微调训练数据
│
├── RAG-Retrieval/                # Reranker 微调框架（RAG-Retrieval）
│
├── models/                       # 本地模型权重（不含于仓库，需自行下载）
│   ├── BAAI/bge-m3
│   ├── BAAI/bge-reranker-v2-m3
│   ├── Qwen3-8B
│   ├── Qwen3-Reranker-4B
│   └── text2vec-base-chinese
│
├── build_index.py                # 索引构建入口
├── infer.py                      # 问答推理入口（交互式命令行）
├── generate_sft_data.py          # 生成 SFT & Reranker 训练数据
├── final_score.py                # 问答效果评估（语义相似度 + RAGas）
├── config.ini                    # 启动脚本（环境变量 & 服务启动）
├── requirements.txt              # Python 依赖
└── .gitignore
```

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+ |
| GPU | NVIDIA GPU，显存 ≥ 24GB（推荐 A100/4090） |
| CUDA | 12.x |
| MongoDB | 7.0+ |

---

## 快速开始

### 1. 安装依赖

```bash
conda create -n rag python=3.10
conda activate rag
pip install -r requirements.txt
```

### 2. 下载模型

需要手动下载以下模型到 `models/` 目录（可通过 ModelScope 或 HuggingFace 镜像下载）：

```
models/
├── BAAI/bge-m3
├── BAAI/bge-reranker-v2-m3
├── Qwen3-8B
├── Qwen3-Reranker-4B
└── text2vec-base-chinese
```

### 3. 配置环境变量

在 `config.ini` 中修改以下参数，或 `export` 到当前 shell：

```bash
export DOUBAO_API_KEY="your_api_key"
export DOUBAO_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export DOUBAO_MODEL_NAME="your_model_endpoint"
export PYTHONPATH=$PYTHONPATH:$PWD
```

### 4. 启动依赖服务

```bash
# 启动语义切分服务
nohup python src/server/semantic_chunk.py > log/semantic_chunk.log 2>&1 &

# 启动 LLM 推理服务（需先完成 LoRA SFT 微调或使用原始 Qwen3-8B）
nohup vllm serve LLaMA-Factory-main/output/qwen3_lora_sft_int4 \
    --max-model-len 8192 --gpu-memory-utilization 0.7 > log/qwen3-8b.log 2>&1 &

# 启动 MongoDB
mongodb-7.0.20/bin/mongod --port=27017 \
    --dbpath=data/mongodb/data \
    --logpath=data/mongodb/log/mongodb.log \
    --bind_ip=0.0.0.0 --fork
```

### 5. 构建索引

```bash
python build_index.py
```

该脚本依次完成：
- PDF 解析（PyMuPDF）
- LLM 文本清洗（去除页眉页脚、乱码等）
- 文本语义切分
- BM25 索引构建
- Milvus 向量索引构建（BGE-M3 编码）

### 6. 启动问答系统

```bash
python infer.py
```

进入交互式命令行，输入问题即可获得带引用编号的答案。

---

## 用户画像MVP优化（已实现）

本项目已实现一版不重建索引的用户画像增强能力，目标是提升多轮问答和车型相关问题的一致性。

### 能力概览

- **轻量画像存储**：新增 `src/profile/user_profile_store.py`，维护 `user_id -> profile + recent_turns`
  - `profile` 字段：`model_cfg`、`software_version`、`updated_at`
  - `recent_turns`：最近多轮问答窗口（默认 5 轮）
- **检索前 Query 改写**：新增 `src/profile/context_engineering.py`
  - 基于用户画像与最近一轮问题，对省略主语/代词问题做补全
  - 保留原始 query，执行“原 query + 改写 query”双路召回
- **画像软过滤重排**：在 `infer.py` 中，`merge_docs` 后、`reranker` 前加入 profile-aware 软打分
  - 命中画像关键词（车型/版本）轻量加权
  - 存在明显车型冲突词时轻量降权
- **生成阶段上下文注入**：扩展 `request_chat()`，将画像与最近对话注入提示词
  - 文件：`src/client/llm_local_client.py`
  - 规则：回答优先参考画像；画像缺失或冲突时先做简短澄清

### 开关与配置（环境变量）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ENABLE_USER_PROFILE` | `1` | 是否启用用户画像能力 |
| `ENABLE_PROFILE_SOFT_FILTER` | `1` | 是否启用画像软过滤重排 |
| `USER_PROFILE_STORE_PATH` | `./data/user_profile_store.json` | 画像与记忆持久化文件路径 |
| `USER_MEMORY_WINDOW` | `5` | 最近对话窗口大小 |

示例：

```bash
export ENABLE_USER_PROFILE=1
export ENABLE_PROFILE_SOFT_FILTER=1
export USER_PROFILE_STORE_PATH=./data/user_profile_store.json
export USER_MEMORY_WINDOW=5
python infer.py
```

### 交互变化（`infer.py`）

`infer.py` 启动后会先询问用户ID，并可选输入画像字段：

- 用户 ID（默认 `default_user`）
- 车型/版本画像（可留空）
- 软件版本画像（可留空）

随后进入问答循环；每轮回答后自动写入最近对话记忆。

### 回归验证

新增脚本：`scripts/profile_mvp_smoke_test.py`

```bash
python scripts/profile_mvp_smoke_test.py
```

该脚本用于快速验证：
- query 改写是否生效
- 画像软过滤排序是否符合预期

### 实现提交记录

用户画像MVP采用 3 次小提交，便于回滚和审计：

- `feat: add user profile and memory store`
- `feat: add profile-aware query rewrite and soft rerank`
- `feat: inject profile context into answer generation`

---

## 模型微调

### Reranker 微调（BGE-M3）

使用 RAG-Retrieval 框架对 BGE-M3 Reranker 进行微调：

```bash
cd RAG-Retrieval
# 参考 RAG-Retrieval/README.md 进行训练配置
```

训练数据由 `generate_sft_data.py` 自动生成，保存在 `data/rerank_data/`。

### LLM 微调（Qwen3-8B LoRA SFT）

使用 LLaMA-Factory 框架进行 LoRA 微调：

```bash
cd LLaMA-Factory-main
# 将 data/summary_data/train.json 转换为 LLaMA-Factory 数据格式
# 参考 LLaMA-Factory-main/README.md 进行训练配置
```

微调完成后使用 AutoAWQ 进行 INT4 量化，输出路径为 `LLaMA-Factory-main/output/qwen3_lora_sft_int4`。

---

## 效果评估

```bash
python final_score.py
```

评估指标：
- **语义相似度 + 关键词加权得分**：使用 text2vec 计算语义相似度，结合关键词 Jaccard 系数
- **RAGas 评估**：基于 LLM 的 Context Recall 和 Context Precision 指标

---

## 核心技术栈

| 模块 | 技术 |
|------|------|
| PDF 解析 | PyMuPDF |
| 文本切分 | 语义切分（FastAPI 服务） |
| 稀疏检索 | BM25（rank-bm25）、TF-IDF |
| 稠密检索 | BGE-M3（Milvus-Lite，稠密+稀疏+ColBERT 混合） |
| 初排融合 | RRF（Reciprocal Rank Fusion） |
| 精排 | BGE-M3 Reranker（微调）/ Qwen3-Reranker-4B |
| LLM | Qwen3-8B（LoRA SFT 微调 + INT4 量化，vLLM 部署） |
| 向量数据库 | Milvus-Lite / Faiss |
| 文档存储 | MongoDB 7.0 |
| LLM 服务 | vLLM |
| 评估框架 | RAGas |

---

## 数据流说明

```
用户提问
  → BM25 召回（Top-10）+ BGE-M3 向量召回（Top-10，RRF 融合）
  → 去重合并
  → BGE-M3 Reranker 精排（Top-5）
  → 构建 Prompt（含编号引用格式）
  → Qwen3-8B 生成答案
  → 后处理（提取答案 + 引用编号）
  → 输出答案【引用1, 引用2, ...】
```

---

## 注意事项

- `models/` 目录下的模型权重文件已在 `.gitignore` 中排除，需自行下载
- `data/saved_index/` 中的索引文件运行 `build_index.py` 后自动生成
- `config.ini` 中的 API Key 仅为示例，请替换为自己的密钥
- 本项目需要 GPU 环境运行，CPU 模式下推理速度极慢
