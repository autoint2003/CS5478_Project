#!/usr/bin/env python
"""Entry point: frozen K49 representability audit (quantize + physical token replay)."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "consolidation" / "audit_k49_representability.py"),
    run_name="__main__",
)
