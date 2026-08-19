# -*- coding: utf-8 -*-
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


class Tracer:
    """Small in-process tracer with an optional exporter hook.

    This class intentionally has no OpenTelemetry dependency. Set ``exporter``
    to a callable accepting one JSON-serializable record to forward spans and
    events to any backend.
    """

    def __init__(
        self,
        trace_id: Optional[str] = None,
        exporter: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex
        self._exporter = exporter
        self._spans: List[Dict[str, Any]] = []
        self._events: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _now_ms(self) -> float:
        return time.time() * 1000.0

    def _export(self, record: Dict[str, Any]) -> None:
        if not self._exporter:
            return
        try:
            self._exporter(record)
        except Exception:
            logger.exception("tracer exporter failed")

    def span(
        self,
        name: str,
        inputs: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a completed span and return its dictionary form."""

        started_at = self._now_ms()
        finished_at = self._now_ms()
        span: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "span_id": uuid.uuid4().hex,
            "name": name,
            "inputs": inputs or {},
            "outputs": outputs or {},
            "start_time_ms": started_at,
            "end_time_ms": finished_at,
            "duration_ms": max(0.0, finished_at - started_at),
        }
        with self._lock:
            self._spans.append(span)
        self._export({"type": "span", **span})
        return span

    def start_span(self, name: str, inputs: Optional[Dict[str, Any]] = None) -> _SpanContext:
        """Start a timed span. The returned object can be used as a context manager."""

        return _SpanContext(self, name, inputs or {})

    def log_event(
        self,
        name: str,
        payload: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ) -> Dict[str, Any]:
        event: Dict[str, Any] = {
            "trace_id": self.trace_id,
            "event_id": uuid.uuid4().hex,
            "name": name,
            "level": level,
            "timestamp_ms": self._now_ms(),
            "payload": payload or {},
        }
        with self._lock:
            self._events.append(event)
        self._export({"type": "event", **event})
        return event

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "trace_id": self.trace_id,
                "spans": list(self._spans),
                "events": list(self._events),
            }


class _SpanContext:
    def __init__(
        self,
        tracer: Tracer,
        name: str,
        inputs: Dict[str, Any],
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._inputs = inputs
        self._started_at = tracer._now_ms()
        self._outputs: Dict[str, Any] = {}
        self._span_id = uuid.uuid4().hex

    def __enter__(self) -> "_SpanContext":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.finish(self._outputs)

    def set_outputs(self, outputs: Dict[str, Any]) -> "_SpanContext":
        self._outputs = outputs or {}
        return self

    def finish(self, outputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if outputs is not None:
            self._outputs = outputs
        finished_at = self._tracer._now_ms()
        span: Dict[str, Any] = {
            "trace_id": self._tracer.trace_id,
            "span_id": self._span_id,
            "name": self._name,
            "inputs": self._inputs,
            "outputs": self._outputs,
            "start_time_ms": self._started_at,
            "end_time_ms": finished_at,
            "duration_ms": max(0.0, finished_at - self._started_at),
        }
        with self._tracer._lock:
            self._tracer._spans.append(span)
        self._tracer._export({"type": "span", **span})
        return span
