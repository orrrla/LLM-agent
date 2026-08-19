# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Mapping, Optional


class LangfuseClient:
    """Optional, dependency-free Langfuse-compatible client stub.

    The real SDK is deliberately not imported. When credentials are absent,
    :meth:`from_env` returns ``None``; when ``LANGFUSE_ENABLED=0`` it returns an
    explicitly disabled instance. Secrets are never written to disk or included
    in serialized records.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: str = "https://cloud.langfuse.com",
        enabled: bool = True,
    ) -> None:
        self.host = host or "https://cloud.langfuse.com"
        self._public_key = public_key
        self._secret_key = secret_key
        self.enabled = bool(enabled and public_key and secret_key)
        self._records: list[Dict[str, Any]] = []

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> Optional["LangfuseClient"]:
        environment = os.environ if env is None else env
        enabled_flag = environment.get("LANGFUSE_ENABLED")
        if enabled_flag is not None and enabled_flag.strip().lower() in {"0", "false", "no", "off"}:
            return cls(enabled=False)

        public_key = _first_env(environment, "LANGFUSE_PUBLIC_KEY", "LANGfuse_PUBLIC_KEY")
        secret_key = _first_env(environment, "LANGFUSE_SECRET_KEY", "LANGfuse_SECRET_KEY")
        if not public_key or not secret_key:
            return None
        host = environment.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        return cls(public_key=public_key, secret_key=secret_key, host=host)

    def is_enabled(self) -> bool:
        return self.enabled

    def __bool__(self) -> bool:
        return self.enabled

    def trace(
        self,
        name: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        **attributes: Any,
    ) -> Optional[Dict[str, Any]]:
        return self._record(
            "trace",
            {
                "name": name,
                "input": input or {},
                "output": output or {},
                "attributes": attributes,
            },
        )

    def generation(
        self,
        name: str,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        **attributes: Any,
    ) -> Optional[Dict[str, Any]]:
        return self._record(
            "generation",
            {
                "name": name,
                "input": input or {},
                "output": output or {},
                "model": model,
                "usage": usage or {},
                "attributes": attributes,
            },
        )

    def _record(self, kind: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        record = {
            "id": uuid.uuid4().hex,
            "type": kind,
            "timestamp_ms": time.time() * 1000.0,
            "host": self.host,
            **payload,
        }
        self._records.append(record)
        return record

    def flush(self) -> None:
        # Kept as an explicit no-op so callers can share one interface with the
        # real SDK. A production adapter can replace this class with the SDK.
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "host": self.host,
            "record_count": len(self._records),
        }


def _first_env(env: Mapping[str, str], *names: str) -> Optional[str]:
    for name in names:
        value = env.get(name)
        if value and value.strip():
            return value.strip()
    return None


def langfuse_from_env(env: Optional[Mapping[str, str]] = None) -> Optional[LangfuseClient]:
    return LangfuseClient.from_env(env)
