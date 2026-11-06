"""Phase 5 — Evaluate P64_step4, P69 A/B on frozen P61 + P65 panels."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.precision.p64_ppo_multiround.eval_p64 import eval_tag
from experiments.precision.p69_skill_sequence.paths import (
    CKPT_DIR,
    OUT,
    P64_STEP4_ACTOR,
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
    out = {k: e.get(k) for k in keys}
    by = e.get("by_class") or e.get("msp_by_class") or {}
    out["msp_by_class"] = by
    out["process_by_class"] = e.get("process_by_class")
    return out


def _class_msp(rep: dict, lab: str) -> float:
    by = rep.get("by_class") or {}
    v = by.get(lab)
    if isinstance(v, dict):
        return float(v.get("MSP_RELAXED") or 0)
    if v is None:
        pb2 = (rep.get("process_by_class") or {}).get(lab) or {}
        return float(pb2.get("MSP_RELAXED") or 0)
    return float(v or 0)


def run_evaluations(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpts = {
        "P64_step4": P64_STEP4_ACTOR,
        "P69_A_control": CKPT_DIR / "P69_A_control_actor.pt",
        "P69_B_seq_pref": CKPT_DIR / "P69_B_seq_pref_actor.pt",
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

    base = None
    if (OUT / "EVAL_P64_step4.json").is_file():
        base = json.loads((OUT / "EVAL_P64_step4.json").read_text(encoding="utf-8"))
    paired = {}
    if base is not None:
        for tag in ("P69_A_control", "P69_B_seq_pref"):
            ef = OUT / f"EVAL_{tag}.json"
            if not ef.is_file():
                continue
            cur = json.loads(ef.read_text(encoding="utf-8"))
            pb = (base.get("process_by_class") or {}).get("C") or {}
            pc = (cur.get("process_by_class") or {}).get("C") or {}
            paired[tag] = {
                "delta_MSP_RELAXED": float(cur.get("MSP_RELAXED") or 0)
                - float(base.get("MSP_RELAXED") or 0),
                "delta_C_MSP": _class_msp(cur, "C") - _class_msp(base, "C"),
                "delta_A_MSP": _class_msp(cur, "A") - _class_msp(base, "A"),
                "delta_B_MSP": _class_msp(cur, "B") - _class_msp(base, "B"),
                "delta_D_MSP": _class_msp(cur, "D") - _class_msp(base, "D"),
                "delta_C_recatch": float(pc.get("p_recatch") or 0) - float(pb.get("p_recatch") or 0),
                "delta_C_stable": float(pc.get("p_stable_recatch") or 0)
                - float(pb.get("p_stable_recatch") or 0),
                "delta_C_handoff": float(pc.get("p_handoff") or 0) - float(pb.get("p_handoff") or 0),
                "delta_C_r3f": float(pc.get("p_r3f_ready") or 0) - float(pb.get("p_r3f_ready") or 0),
            }
    out = {"evals": results, "paired_vs_P64": paired}
    (OUT / "PPO_CONTROL_COMPARISON.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


if __name__ == "__main__":
    run_evaluations()
