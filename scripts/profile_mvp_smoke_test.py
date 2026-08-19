# -*- coding: utf-8 -*-
from dataclasses import dataclass
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.profile.context_engineering import rewrite_query_with_profile, profile_soft_rerank


@dataclass
class SimpleDoc:
    page_content: str
    metadata: dict


def main():
    profile = {"model_cfg": "Model 3", "software_version": "2024.14"}
    recent_turns = [
        {"query": "离车后自动上锁怎么打开？", "answer": "可以在控制-车锁里开启。"},
        {"query": "那它关闭后有什么影响？", "answer": "需要手动锁车。"},
    ]
    raw_query = "这个功能怎么关？"
    rewritten = rewrite_query_with_profile(raw_query, profile, recent_turns)
    print("原始问题:", raw_query)
    print("改写问题:", rewritten)

    docs = [
        SimpleDoc(page_content="Model Y 可在车锁设置中调整。", metadata={"unique_id": "1"}),
        SimpleDoc(page_content="Model 3 支持离车后自动上锁。", metadata={"unique_id": "2"}),
        SimpleDoc(page_content="车辆锁定时可设置提示音。", metadata={"unique_id": "3"}),
    ]
    reranked = profile_soft_rerank(docs, profile)
    print("软过滤后Top顺序:")
    for idx, doc in enumerate(reranked, 1):
        print(f"{idx}. {doc.page_content}")


if __name__ == "__main__":
    main()

