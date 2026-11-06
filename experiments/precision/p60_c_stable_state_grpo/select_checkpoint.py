"""Phase 1 — choose P59 init checkpoint from validation (not held-out)."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p60_c_stable_state_grpo.paths import (
    OUT,
    P59_PASS1,
    P59_PASS2,
    P59_VAL_PASS1,
    P59_VAL_PASS2,
)


def select_init_checkpoint() -> dict:
    """Both passes have C MSP=0.145; prefer pass1 for better ABD retention."""
    OUT.mkdir(parents=True, exist_ok=True)

    def _load(path: Path) -> dict:
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    v1 = _load(P59_VAL_PASS1)
    v2 = _load(P59_VAL_PASS2)
    # Prefer fullest blob that contains both if available
    if (v2.get("reports_c") or {}).get("P59_pass1") and (v2.get("reports_c") or {}).get("P59_pass2"):
        blob = v2
    else:
        blob = v2 or v1

    rc = blob.get("reports_c") or {}
    ra = blob.get("reports_abd") or {}
    m1 = (rc.get("P59_pass1") or {}).get("MSP_RELAXED")
    m2 = (rc.get("P59_pass2") or {}).get("MSP_RELAXED")
    a1 = (ra.get("P59_pass1") or {}).get("MSP_RELAXED")
    a2 = (ra.get("P59_pass2") or {}).get("MSP_RELAXED")

    # Default preference: pass1 (better ABD). Only take pass2 if C clearly better
    # without relying on held-out.
    choice = "P59_pass1"
    path = P59_PASS1
    rationale = [
        f"C val MSP pass1={m1}, pass2={m2} (tied/near-tied).",
        f"ABD val MSP pass1={a1}, pass2={a2}.",
        "Prefer P59_pass1 for better A/B/D retention with equal C validation MSP.",
        "Held-out results were not used for this selection.",
    ]
    if m1 is not None and m2 is not None and float(m2) > float(m1) + 1e-9:
        choice = "P59_pass2"
        path = P59_PASS2
        rationale = [
            f"C val MSP pass2={m2} > pass1={m1}; select pass2 on validation C only.",
            f"ABD val MSP pass1={a1}, pass2={a2}.",
            "Held-out results were not used for this selection.",
        ]
    elif a1 is not None and a2 is not None and float(a1) >= float(a2) - 1e-12:
        choice = "P59_pass1"
        path = P59_PASS1

    if not path.is_file():
        raise FileNotFoundError(path)

    out = {
        "selected": choice,
        "path": str(path),
        "c_val_msp": {"P59_pass1": m1, "P59_pass2": m2},
        "abd_val_msp": {"P59_pass1": a1, "P59_pass2": a2},
        "rationale": rationale,
    }
    (OUT / "CHECKPOINT_SELECTION.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "CHECKPOINT_SELECTION.md").write_text(
        "# Checkpoint selection\n\n"
        + f"**Selected: `{choice}`**\n\n"
        + "\n".join(f"- {r}" for r in rationale)
        + "\n",
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    print(json.dumps(select_init_checkpoint(), indent=2))
