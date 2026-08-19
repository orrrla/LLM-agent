# -*- coding: utf-8 -*-
"""OpenAI-compatible vision-language model helpers.

No model is loaded at import time. The OpenAI client and image encoding are
created only when a caption/region request is made, and an empty API key is
rejected before any network call.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


DEFAULT_REGION_PROMPT = """
你是PDF页面版面分析助手。请识别当前页面中的表格、图片、标题和正文区域。
只返回一个JSON对象，包含字段：page_title、sections（数组，每项含title、bbox、type、summary）。
bbox使用[x0, y0, x1, y1]像素坐标。不要返回JSON以外的内容。
""".strip()


class VLMCaptioner:
    """Call an OpenAI-compatible vision model for captions and region extraction."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.base_url = base_url or os.getenv(
            "VLM_BASE_URL", os.getenv("OPENAI_BASE_URL")
        )
        self.model = model or os.getenv("VLM_MODEL", os.getenv("OPENAI_MODEL"))
        self.api_key = api_key or os.getenv(
            "VLM_API_KEY", os.getenv("OPENAI_API_KEY")
        )

        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "VLM_API_KEY is empty. Set VLM_API_KEY or OPENAI_API_KEY before "
                "calling vision caption APIs; this module does not return fake output."
            )
        if not self.model or not self.model.strip():
            raise ValueError("VLM_MODEL is empty. Configure a vision model name.")

    def _get_client(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "VLMCaptioner requires the openai package. Install "
                "src/vision/requirements-vision.txt first."
            ) from exc

        return OpenAI(api_key=self.api_key, base_url=self.base_url or None)

    @staticmethod
    def _encode_image(image_path: str) -> str:
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"image not found: {image_path}")
        data = path.read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        # The OpenAI-compatible API accepts a data URL regardless of the exact
        # image extension; PNG files are the primary input from page_loader.
        return f"data:image/png;base64,{encoded}"

    def _call_vision(self, page_image_path: str, prompt: str) -> str:
        client = self._get_client()
        image_url = self._encode_image(page_image_path)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise RuntimeError("vision model returned an empty response")
        return content.strip()

    def generate_caption(self, page_image_path: str, prompt: str) -> str:
        """Return a caption for one rendered page."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        return self._call_vision(page_image_path, prompt)

    def extract_regions(
        self,
        page_image_path: str,
        prompt: str = DEFAULT_REGION_PROMPT,
    ) -> Union[Dict[str, Any], List[Any], str]:
        """Return structured regions when possible, otherwise the raw model text."""
        raw = self._call_vision(page_image_path, prompt)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> Union[Dict[str, Any], List[Any], str]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return raw


def generate_caption(
    page_image_path: str,
    prompt: str,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Convenience wrapper around VLMCaptioner.generate_caption."""
    return VLMCaptioner(
        base_url=base_url, model=model, api_key=api_key
    ).generate_caption(page_image_path, prompt)


def extract_regions(
    page_image_path: str,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Union[Dict[str, Any], List[Any], str]:
    """Convenience wrapper around VLMCaptioner.extract_regions."""
    return VLMCaptioner(
        base_url=base_url, model=model, api_key=api_key
    ).extract_regions(page_image_path)
