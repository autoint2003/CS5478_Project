"""Phase 6 — Evaluate P64_step4, P65_step6, P66 A/B on frozen P61 + P65 panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p64_ppo_multiround.eval_p64 import eval_tag
from experiments.precision.p66_c_curriculum_ppo.paths import (
    CKPT_DIR,
    OUT,
    P64_STEP4_ACTOR,
    P65_STEP6_ACTOR,
)


def _slim(e: dict) -> dict:
    keys = (
        "tag",
        "MSP_RELAXED",
        "MSP",
        "zero_success_ics",
        "p_recatch",
        "p_stable_recatch",
        "p_handoff",
        "p_r3f_ready",
        "process_by_class",
        "by_class",
        "n_ics",
        "n_rollouts",
    )
    out = {k: e.get(k) for k in keys if k in e or True}
    # class MSP
    by = e.get("by_class") or e.get("msp_by_class") or {}
    out["msp_by_class"] = by
    out["process_by_class"] = e.get("process_by_class")
    return out


def run_evaluations(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpts = {
        "P64_step4": P64_STEP4_ACTOR,
        "P65_step6": P65_STEP6_ACTOR,
        "P66_A_control": CKPT_DIR / "P66_A_control_actor.pt",
        "P66_B_curriculum": CKPT_DIR / "P66_B_curriculum_actor.pt",
    }
    results = {}
    for tag, path in ckpts.items():
        path = Path(path)
        if not path.is_file():
            results[tag] = {"missing": True, "path": str(path)}
            print(f"[eval] MISSING {tag}: {path}", flush=True)
            continue
        print(f"[eval] {tag} ← {path.name}", flush=True)
        rep = eval_tag(tag, path, device=device)
        results[tag] = _slim(rep)
        (OUT / f"EVAL_{tag}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(
            f"  MSP_RELAXED={results[tag].get('MSP_RELAXED')} "
            f"by={results[tag].get('msp_by_class')}",
            flush=True,
        )

    # Paired C deltas vs P64
    base = None
    if "P64_step4" in results and not results["P64_step4"].get("missing"):
        base_full = json.loads((OUT / "EVAL_P64_step4.json").read_text(encoding="utf-8"))
        base = base_full
    paired = {}
    if base is not None:
        for tag in ("P66_A_control", "P66_B_curriculum", "P65_step6"):
            ef = OUT / f"EVAL_{tag}.json"
            if not ef.is_file():
                continue
            cur = json.loads(ef.read_text(encoding="utf-8"))
            # C-only stage-ish from process_by_class
            pb = (base.get("process_by_class") or {}).get("C") or {}
            pc = (cur.get("process_by_class") or {}).get("C") or {}
            def _class_msp(rep: dict, lab: str) -> float:
                by = rep.get("by_class") or {}
                v = by.get(lab)
                if isinstance(v, dict):
                    return float(v.get("MSP_RELAXED") or 0)
                if v is None:
                    pb2 = (rep.get("process_by_class") or {}).get(lab) or {}
                    return float(pb2.get("MSP_RELAXED") or 0)
                return float(v or 0)

            paired[tag] = {
                "delta_MSP_RELAXED": float(cur.get("MSP_RELAXED") or 0)
                - float(base.get("MSP_RELAXED") or 0),
                "delta_C_MSP": _class_msp(cur, "C") - _class_msp(base, "C"),
                "delta_C_recatch": float(pc.get("p_recatch") or 0) - float(pb.get("p_recatch") or 0),
                "delta_C_stable": float(pc.get("p_stable_recatch") or 0)
                - float(pb.get("p_stable_recatch") or 0),
                "delta_C_handoff": float(pc.get("p_handoff") or 0) - float(pb.get("p_handoff") or 0),
            }
    out = {"evals": results, "paired_vs_P64_step4": paired}
    (OUT / "EVAL_COMPARE.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    run_evaluations()
