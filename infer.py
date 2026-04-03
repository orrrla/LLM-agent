# -*- coding: utf-8 -*-

import os
import pickle
import time
from src.retriever.bm25_retriever import BM25
from src.retriever.tfidf_retriever import TFIDF
from src.retriever.faiss_retriever import FaissRetriever
from src.retriever.milvus_retriever import MilvusRetriever 
from src.client.llm_local_client import request_chat
from src.client.llm_hyde_client import request_hyde
from src.reranker.bge_m3_reranker import BGEM3ReRanker 
from src.constant import bge_reranker_tuned_model_path
from src.utils import merge_docs, post_processing
from src.profile.user_profile_store import UserProfileStore
from src.profile.context_engineering import rewrite_query_with_profile, profile_soft_rerank


ENABLE_USER_PROFILE = os.environ.get("ENABLE_USER_PROFILE", "1") == "1"
ENABLE_PROFILE_SOFT_FILTER = os.environ.get("ENABLE_PROFILE_SOFT_FILTER", "1") == "1"
USER_PROFILE_STORE_PATH = os.environ.get("USER_PROFILE_STORE_PATH", "./data/user_profile_store.json")
MEMORY_WINDOW = int(os.environ.get("USER_MEMORY_WINDOW", "5"))

# warmstart
bm25_retriever = BM25(docs=None, retrieve=True)
milvus_retriever = MilvusRetriever(docs=None, retrieve=True) 
bge_m3_reranker = BGEM3ReRanker(model_path=bge_reranker_tuned_model_path)
milvus_retriever.retrieve_topk("这是一条测试数据", topk=3)
profile_store = UserProfileStore(store_path=USER_PROFILE_STORE_PATH, memory_window=MEMORY_WINDOW)

user_id = input("用户ID（直接回车默认default_user）—>").strip() or "default_user"
if ENABLE_USER_PROFILE:
    saved_profile = profile_store.get_profile(user_id)
    print(f"当前画像: {saved_profile if saved_profile else '未设置'}")
    model_cfg = input("车型/版本画像（可留空）—>").strip()
    software_version = input("软件版本画像（可留空）—>").strip()
    if model_cfg or software_version:
        profile_store.upsert_profile(user_id, model_cfg=model_cfg, software_version=software_version)
        print("画像已更新")
    print("=" * 100)

while True:
    query = input("输入—>")
    if query.strip() in ["exit", "quit"]:
        break
    active_profile = profile_store.get_profile(user_id) if ENABLE_USER_PROFILE else {}
    recent_turns = profile_store.get_recent_turns(user_id) if ENABLE_USER_PROFILE else []
    rewritten_query = rewrite_query_with_profile(query, active_profile, recent_turns) if ENABLE_USER_PROFILE else query
    used_query = rewritten_query if rewritten_query else query
    print(f"检索Query: {used_query}")

    # 检索
    # BM25召回
    t1 = time.time()
    bm25_docs = bm25_retriever.retrieve_topk(query, topk=10)
    if rewritten_query != query:
        bm25_docs_rewritten = bm25_retriever.retrieve_topk(rewritten_query, topk=10)
        bm25_docs = merge_docs(bm25_docs, bm25_docs_rewritten)
    print("BM25召回样例:")
    print(bm25_docs)
    print("="*100)
    t2 = time.time()


    # BGE-M3稠密+稀疏召回+RRF初排
    milvus_docs = milvus_retriever.retrieve_topk(query, topk=10)
    if rewritten_query != query:
        milvus_docs_rewritten = milvus_retriever.retrieve_topk(rewritten_query, topk=10)
        milvus_docs = merge_docs(milvus_docs, milvus_docs_rewritten)
    print("BGE-M3召回样例:")
    print(milvus_docs)
    print("="*100)
    t3 = time.time()


    # 去重
    merged_docs = merge_docs(bm25_docs, milvus_docs)
    if ENABLE_USER_PROFILE and ENABLE_PROFILE_SOFT_FILTER:
        merged_docs = profile_soft_rerank(merged_docs, active_profile)
    print(merged_docs)
    print("="*100)


    # 精排 
    ranked_docs = bge_m3_reranker.rank(used_query, merged_docs, topk=5)
    print(ranked_docs)
    print("="*100)


    # 答案
    context = "\n".join(["【" + str(idx+1) + "】" + doc.page_content for idx, doc in enumerate(ranked_docs)])
    res_handler = request_chat(query, context, stream=True, profile=active_profile, recent_turns=recent_turns)
    response = ""
    for r in res_handler:
        uttr = r.choices[0].delta.content
        response += uttr 
        print(uttr, end='')
    print("\n" + "="*100)

    # 后处理
    answer = post_processing(response, ranked_docs)
    print("\n答案—>", answer)
    if ENABLE_USER_PROFILE:
        profile_store.append_turn(user_id, query, answer.get("answer", ""))

