#!/usr/bin/env python
"""Entry point: freeze / rebuild the unified Plan A/B/C expert dataset."""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "consolidation" / "freeze_experts.py"),
    run_name="__main__",
)
