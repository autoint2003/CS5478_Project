"""Paths and limits shared by the K48 training modules."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTED_LIMIT_S = 4.0
CKPT = ROOT / "results/archive/demo_results/diagnostics/raw/unified_k47/supervised/checkpoint_epoch_0040.pt"
DATA = ROOT / "results/archive/demo_results/diagnostics/raw/k48_handoff"
