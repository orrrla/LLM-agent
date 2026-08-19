# -*- coding: utf-8 -*-
"""ColPali page embedding and Milvus indexing.

The heavy dependencies (transformers, torch, PIL, pymilvus) are imported only
when an embedding or database operation is requested. This keeps the module
importable in environments where the full vision stack is not installed.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Sequence

from src.vision.schemas import VisualPage


class ColPaliIndexer:
    """Build a page-level ColPali vector index in Milvus.

    ColPali is a multi-vector model. For page-level retrieval this implementation
    mean-pools the last hidden state of each page/query and stores one L2
    normalized vector per page. This trades some late-interaction fidelity for a
    simple VisualPage-compatible single-vector schema.
    """

    DEFAULT_COLLECTION_NAME = "visual_pages_colpali"
    DEFAULT_MILVUS_URI = os.getenv("MILVUS_URI", "./data/saved_index/vision_milvus.db")

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        collection_name: Optional[str] = None,
        milvus_uri: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or os.getenv(
            "COLPAI_MODEL_PATH", ""
        )
        self.device = device or os.getenv("COLPAI_DEVICE")
        self.collection_name = (
            collection_name
            or os.getenv("COLPAI_COLLECTION_NAME")
            or self.DEFAULT_COLLECTION_NAME
        )
        self.milvus_uri = milvus_uri or self.DEFAULT_MILVUS_URI
        self._alias = f"vision_colpali_{id(self)}"
        self._connected = False
        self._processor = None
        self._model = None

    def _ensure_model_path(self) -> None:
        if not self.model_path:
            raise ValueError(
                "COLPAI_MODEL_PATH is empty. Configure a local ColPali model "
                "directory before encoding."
            )
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"ColPali model path does not exist: {self.model_path}. "
                "Download the weights outside this module first; this code does "
                "not download models."
            )

    def _get_model_stack(self):
        """Lazily load transformers/torch and the local model."""
        self._ensure_model_path()
        try:
            import torch  # type: ignore
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ColPali encoding requires torch and transformers. Install "
                "src/vision/requirements-vision.txt first."
            ) from exc

        if self._processor is None or self._model is None:
            device = self._resolve_device(torch)
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_path, local_files_only=True
            ).to(device)
            self._model.eval()

        return self._processor, self._model, torch, self._resolve_device(torch)

    def _resolve_device(self, torch) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _l2_normalize(vector: List[float]) -> List[float]:
        norm = math.sqrt(sum(float(x) * float(x) for x in vector))
        if norm == 0.0:
            raise ValueError("encoded vector has zero norm; cannot index it")
        return [float(x) / norm for x in vector]

    def _pool_last_hidden(self, hidden, attention_mask, torch) -> List[float]:
        if attention_mask is not None:
            mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = hidden.mean(dim=1)
        vector = pooled[0].detach().cpu().float().tolist()
        if not vector:
            raise RuntimeError("model returned an empty vector")
        return self._l2_normalize(vector)

    def _encode_page(self, page: VisualPage, processor, model, torch, device) -> List[float]:
        if page.page_vector is not None:
            if not page.page_vector:
                raise ValueError(f"page {page.page_id} has an empty page_vector")
            return self._l2_normalize(list(page.page_vector))

        if not page.image_path:
            raise ValueError(
                f"page {page.page_id} has no image_path and no page_vector"
            )
        if not os.path.isfile(page.image_path):
            raise FileNotFoundError(
                f"rendered page image not found: {page.image_path}"
            )

        try:
            from PIL import Image  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ColPali image encoding requires Pillow. Install "
                "src/vision/requirements-vision.txt first."
            ) from exc

        image = Image.open(page.image_path).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        return self._pool_last_hidden(
            outputs.last_hidden_state,
            inputs.get("attention_mask"),
            torch,
        )

    def encode_query(self, query: str) -> List[float]:
        """Encode a text query to the same single-vector space as pages."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        processor, model, torch, device = self._get_model_stack()
        inputs = processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        return self._pool_last_hidden(
            outputs.last_hidden_state,
            inputs.get("attention_mask"),
            torch,
        )

    @staticmethod
    def get_schema_fields(dim: int) -> List[Dict[str, Any]]:
        """Return the Milvus field definitions used by this module."""
        return [
            {
                "name": "page_id",
                "dtype": "VARCHAR",
                "is_primary": True,
                "max_length": 256,
            },
            {"name": "page_num", "dtype": "INT64"},
            {"name": "image_path", "dtype": "VARCHAR", "max_length": 2048},
            {"name": "caption", "dtype": "VARCHAR", "max_length": 8192},
            {"name": "tables_json", "dtype": "VARCHAR", "max_length": 16384},
            {"name": "regions_json", "dtype": "VARCHAR", "max_length": 16384},
            {"name": "section_title", "dtype": "VARCHAR", "max_length": 1024},
            {"name": "page_vector", "dtype": "FLOAT_VECTOR", "dim": dim},
        ]

    def _ensure_connection(self):
        try:
            from pymilvus import connections  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ColPali indexing requires pymilvus. Install "
                "src/vision/requirements-vision.txt first."
            ) from exc

        if self._connected:
            return connections
        try:
            connections.connect(alias=self._alias, uri=self.milvus_uri)
            self._connected = True
        except Exception as exc:
            raise RuntimeError(
                f"failed to connect to Milvus at {self.milvus_uri}: {exc}"
            ) from exc
        return connections

    def _create_collection(self, collection_name: str, dim: int):
        from pymilvus import (  # type: ignore
            Collection,
            CollectionSchema,
            DataType,
            FieldSchema,
            utility,
        )

        fields = []
        for spec in self.get_schema_fields(dim):
            kwargs = dict(spec)
            kwargs["dtype"] = getattr(DataType, kwargs["dtype"])
            if kwargs.get("is_primary", False):
                kwargs["is_primary"] = True
            fields.append(FieldSchema(**kwargs))

        schema = CollectionSchema(
            fields,
            description="Visual page vectors built with ColPali",
        )
        collection = Collection(
            name=collection_name,
            schema=schema,
            consistency_level="Strong",
        )
        collection.create_index(
            field_name="page_vector",
            index_params={"index_type": "AUTOINDEX", "metric_type": "IP", "params": {}},
        )
        collection.load()
        return collection

    def build_from_pages(
        self,
        pages: Sequence[VisualPage],
        collection_name: Optional[str] = None,
        drop_if_exists: bool = False,
        batch_size: int = 16,
    ) -> int:
        """Encode and index VisualPage objects, returning the inserted count."""
        pages = list(pages)
        if not pages:
            raise ValueError("pages must not be empty")

        target_collection = collection_name or self.collection_name
        if not target_collection:
            raise ValueError("collection_name must be provided")

        processor, model, torch, device = self._get_model_stack()
        self._ensure_connection()

        try:
            from pymilvus import utility  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ColPali indexing requires pymilvus. Install "
                "src/vision/requirements-vision.txt first."
            ) from exc

        if utility.has_collection(target_collection):
            if drop_if_exists:
                utility.drop_collection(target_collection)
            else:
                raise RuntimeError(
                    f"Milvus collection {target_collection} already exists. "
                    "Pass drop_if_exists=True to rebuild it."
                )

        first_vector = self._encode_page(pages[0], processor, model, torch, device)
        dim = len(first_vector)
        collection = self._create_collection(target_collection, dim)

        inserted = 0
        for start in range(0, len(pages), batch_size):
            batch = pages[start : start + batch_size]
            vectors = [
                self._encode_page(page, processor, model, torch, device)
                for page in batch
            ]
            for page, vector in zip(batch, vectors):
                page.page_vector = vector

            entities = [
                [page.page_id for page in batch],
                [int(page.page_num) for page in batch],
                [page.image_path or "" for page in batch],
                [page.caption or "" for page in batch],
                [self._json_value(page.tables_json) for page in batch],
                [self._json_value(page.regions_json) for page in batch],
                [page.section_title or "" for page in batch],
                vectors,
            ]
            collection.insert(entities)
            inserted += len(batch)

        collection.flush()
        return inserted

    @staticmethod
    def _json_value(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        import json

        return json.dumps(value, ensure_ascii=False)
