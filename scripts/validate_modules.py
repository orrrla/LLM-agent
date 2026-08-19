#!/usr/bin/env python3
"""Validate the three parallel module slices before integration."""

from __future__ import annotations

import pathlib
import py_compile
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]

EXPECTED_FILES = [
    "src/agent/__init__.py",
    "src/agent/state.py",
    "src/agent/router.py",
    "src/agent/graph.py",
    "src/agent/nodes.py",
    "src/agent/tools/__init__.py",
    "src/agent/tools/clarify.py",
    "src/vision/__init__.py",
    "src/vision/schemas.py",
    "src/vision/page_loader.py",
    "src/vision/colpali_index.py",
    "src/vision/page_retriever.py",
    "src/vision/vlm_caption.py",
    "src/vision/fusion.py",
    "src/memory/__init__.py",
    "src/memory/mem0_store.py",
    "src/eval/__init__.py",
    "src/eval/schemas.py",
    "src/eval/golden_loader.py",
    "src/eval/metrics.py",
    "src/eval/run_eval.py",
    "src/eval/report.py",
    "src/eval/synthetic_gen.py",
    "src/observability/__init__.py",
    "src/observability/tracer.py",
    "src/observability/feedback_store.py",
    "src/observability/langfuse_client.py",
]


def main() -> int:
    missing = [rel for rel in EXPECTED_FILES if not (ROOT / rel).exists()]
    if missing:
        print("MISSING FILES:")
        for rel in missing:
            print(f"  - {rel}")
        return 1

    package_dirs = [
        ROOT / "src" / "agent",
        ROOT / "src" / "vision",
        ROOT / "src" / "memory",
        ROOT / "src" / "eval",
        ROOT / "src" / "observability",
    ]
    failed = []
    for package_dir in package_dirs:
        for source in package_dir.rglob("*.py"):
            if source.name == "__pycache__":
                continue
            try:
                py_compile.compile(str(source), doraise=True)
            except py_compile.PyCompileError as exc:
                failed.append(str(source))
                print(f"COMPILE FAILED: {source}: {exc}")

    if failed:
        return 1
    print("OK: expected files present and all module sources compile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
