"""Idempotent: ensure result-dir docs exist (already written as static files)."""
from experiments.precision.p53_independent_grpo64.paths import OUT, CKPT_DIR

OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
print(OUT)
