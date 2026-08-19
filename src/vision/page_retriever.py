# -*- coding: utf-8 -*-
"""Retrieve VisualPage objects from a ColPali-backed Milvus collection."""

from __future__ import annotations

from typing import Any, List, Optional

from src.vision.colpali_index import ColPaliIndexer
from src.vision.schemas import VisualPage


class PageRetriever:
    """Query a page-level visual index and return ranked VisualPage results."""

    OUTPUT_FIELDS = [
        "page_id",
        "page_num",
        "image_path",
        "caption",
        "tables_json",
        "regions_json",
        "section_title",
    ]

    def __init__(
        self,
        indexer: Optional[ColPaliIndexer] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        collection_name: Optional[str] = None,
        milvus_uri: Optional[str] = None,
    ) -> None:
        self.indexer = indexer or ColPaliIndexer(
            model_path=model_path,
            device=device,
            collection_name=collection_name,
            milvus_uri=milvus_uri,
        )
        self.collection_name = collection_name or self.indexer.collection_name

    @staticmethod
    def _maybe_json(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            import json

            try:
                return json.loads(value)
            except (TypeError, ValueError):
                return value
        return value

    def search(self, query: str, topk: int = 5) -> List[VisualPage]:
        """Return the top-k visual pages for a text query.

        Raises RuntimeError with a clear message when Milvus is not connected or
        the collection cannot be queried.
        """
        if topk <= 0:
            raise ValueError("topk must be a positive integer")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        # Connection errors are surfaced before model encoding so callers can
        # distinguish an environment problem from a model problem.
        try:
            self.indexer._ensure_connection()
        except RuntimeError as exc:
            raise RuntimeError(
                f"Milvus is not connected for visual page retrieval: {exc}"
            ) from exc

        query_vector = self.indexer.encode_query(query)

        try:
            from pymilvus import Collection, utility  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "PageRetriever requires pymilvus. Install "
                "src/vision/requirements-vision.txt first."
            ) from exc

        try:
            has_collection = utility.has_collection(self.collection_name)
        except Exception as exc:
            raise RuntimeError(
                f"failed to check Milvus collection {self.collection_name!r}: {exc}"
            ) from exc

        if not has_collection:
            raise RuntimeError(
                f"Milvus collection {self.collection_name!r} does not exist. "
                "Build it first with ColPaliIndexer.build_from_pages()."
            )

        try:
            collection = Collection(
                self.collection_name,
                using=self.indexer._alias,
            )
            collection.load()
            results = collection.search(
                data=[query_vector],
                anns_field="page_vector",
                param={"metric_type": "IP", "params": {}},
                limit=topk,
                output_fields=self.OUTPUT_FIELDS,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to search Milvus collection {self.collection_name!r}: {exc}"
            ) from exc

        if not results:
            return []

        pages: List[VisualPage] = []
        for hit in results[0]:
            entity = getattr(hit, "entity", {}) or {}
            score = float(getattr(hit, "score", getattr(hit, "distance", 0.0) or 0.0))
            pages.append(
                VisualPage(
                    page_id=str(entity.get("page_id", "")),
                    page_num=int(entity.get("page_num", 0)),
                    image_path=entity.get("image_path"),
                    page_vector=None,
                    caption=entity.get("caption"),
                    tables_json=self._maybe_json(entity.get("tables_json")),
                    regions_json=self._maybe_json(entity.get("regions_json")),
                    section_title=entity.get("section_title"),
                    score=score,
                )
            )

        return pages
