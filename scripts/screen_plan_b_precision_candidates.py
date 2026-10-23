#!/usr/bin/env python
"""Screen expanded Plan B pools for PRECISION_CANDIDATE ICs (direct HANDOFF fails).

Does not freeze a new pilot manifest. Writes:
  results/data_generation/stage2_abc_pilot/plan_b_candidate_screen.json
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.stable_grasp import hold_and_confirm_stable  # noqa: E402
from data_generation.oracles.precision.unified_precision_oracle import (  # noqa: E402
    entry_gate,
    measure_errors,
    probe_direct_handoff,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
# Import helpers from the pilot generator without requiring scripts/ as a package.
import importlib.util  # noqa: E402

_gen_path = ROOT / "scripts" / "generate_stage2_abc_pilot.py"
_spec = importlib.util.spec_from_file_location("generate_stage2_abc_pilot", _gen_path)
_gen = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_gen)
capture_fp = _gen.capture_fp
extract_stable_from_tape = _gen.extract_stable_from_tape
load_ue_manifest = _gen.load_ue_manifest

OUT = ROOT / "results/data_generation/stage2_abc_pilot"
HARD_DIR = ROOT / "results/diagnostics/hard_ic_mining/snapshots"
STABLE_B = ROOT / "results/diagnostics/abc_precision_architecture/stable_states/plan_b__hard_910220__stable.pkl"
VR = ROOT / "data/vertical_recatch/train"
SNAP = ROOT / "data/snapshots/train"


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def classify_capture(env, capture, ref) -> dict:
    restore_capture(env, capture)
    gate = entry_gate(env)
    restore_capture(env, capture)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, capture)
    err = measure_errors(env.sim)
    restore_capture(env, capture)
    if not (gate.get("ok") and stab.get("stable")):
        return {
            "bin": "GATE_FAIL",
            "entry_gate_ok": bool(gate.get("ok")),
            "stable": bool(stab.get("stable")),
            "e_pos_mm": float(err["e_pos_mm"]),
            "dtheta_deg": float(err["dtheta_deg"]),
        }
    direct = probe_direct_handoff(env, capture, ref, stabilize_n=8)
    restore_capture(env, capture)
    bin_ = "DIRECT_SUCCESS" if direct.get("SUCCESS_NEW") else "PRECISION_CANDIDATE"
    return {
        "bin": bin_,
        "entry_gate_ok": True,
        "stable": True,
        "e_pos_mm": float(err["e_pos_mm"]),
        "dtheta_deg": float(err["dtheta_deg"]),
        "ex_mm": float(err["e_signed_mm"][0]),
        "ez_mm": float(err["e_signed_mm"][2]),
        "direct_SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
        "direct_reason": direct.get("reason"),
        "capture_fp": capture_fp(capture),
    }


def main():
    env_cls = type("E", (), {})
    env = env_cls()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)

    ue_b_ids = set()
    ue_all_ids: dict[str, str] = {}
    for r in load_ue_manifest():
        rid = r.get("physical_root_id") or r["expert_id"].split("__", 1)[-1]
        ue_all_ids[rid] = r["plan_family"]
        if r.get("plan_family") == "B":
            ue_b_ids.add(rid)

    rows = []

    # 1) Verified stable diagnostic
    if STABLE_B.exists():
        blob = pickle.loads(STABLE_B.read_bytes())
        c = classify_capture(env, blob["capture"], blob["ref"])
        rows.append(
            {
                "source_kind": "diagnostic_stable",
                "ic_id": "hard_910220",
                "path": str(STABLE_B.relative_to(ROOT)),
                **c,
            }
        )
        print("stable hard_910220", c["bin"], c.get("e_pos_mm"), flush=True)

    # 2) hard_ic_mining raw snapshots (may or may not be Stage-2 eligible)
    hard_paths = sorted(HARD_DIR.glob("hard_*.pkl")) if HARD_DIR.exists() else []
    # Bound: screen all — typically ~120; each is one restore+gate+direct
    for hp in hard_paths:
        try:
            blob = pickle.loads(hp.read_bytes())
            if "capture" not in blob or "ref" not in blob:
                rows.append({"source_kind": "hard_ic_mining", "ic_id": hp.stem, "bin": "BAD_SCHEMA"})
                continue
            c = classify_capture(env, blob["capture"], blob["ref"])
            rows.append({"source_kind": "hard_ic_mining", "ic_id": hp.stem, "path": str(hp.relative_to(ROOT)), **c})
            if c["bin"] == "PRECISION_CANDIDATE":
                print(f"  HARD {hp.stem} PRECISION_CANDIDATE e_pos={c['e_pos_mm']:.1f}", flush=True)
        except Exception as e:
            rows.append({"source_kind": "hard_ic_mining", "ic_id": hp.stem, "bin": f"ERROR:{e}"})

    # 3) follow_early roots not claimed by any UE plan (avoid relabeling Plan A as B)
    for npz in sorted(VR.glob("*__follow_early.npz")):
        rid = npz.name.replace("__follow_early.npz", "")
        if rid in ue_all_ids:
            rows.append(
                {
                    "source_kind": "follow_early_other_plan",
                    "ic_id": rid,
                    "ue_plan": ue_all_ids[rid],
                    "bin": "SKIP_OTHER_PLAN",
                    "path": str(npz.relative_to(ROOT)),
                }
            )
            continue
        snap = SNAP / f"{rid}.pkl"
        if not snap.exists():
            continue
        try:
            blob = extract_stable_from_tape(env, snap, npz, "B", rid, f"plan_b__{rid}")
            if blob is None:
                rows.append(
                    {
                        "source_kind": "follow_early_unassigned",
                        "ic_id": rid,
                        "bin": "UNSTABLE",
                        "path": str(npz.relative_to(ROOT)),
                    }
                )
                continue
            c = classify_capture(env, blob["capture"], blob["ref"])
            rows.append(
                {
                    "source_kind": "follow_early_unassigned",
                    "ic_id": rid,
                    "path": str(npz.relative_to(ROOT)),
                    "n_prefix": blob["n_prefix"],
                    **c,
                }
            )
            print(f"  UNASSIGNED {rid} {c['bin']} e_pos={c.get('e_pos_mm')}", flush=True)
        except Exception as e:
            rows.append({"source_kind": "follow_early_unassigned", "ic_id": rid, "bin": f"ERROR:{e}"})

    def count(kind=None, bin_=None):
        return sum(
            1
            for r in rows
            if (kind is None or r.get("source_kind") == kind)
            and (bin_ is None or r.get("bin") == bin_)
        )

    summary = {
        "n_rows": len(rows),
        "PRECISION_CANDIDATE_total": count(bin_="PRECISION_CANDIDATE"),
        "DIRECT_SUCCESS_total": count(bin_="DIRECT_SUCCESS"),
        "GATE_FAIL_total": count(bin_="GATE_FAIL"),
        "by_source": {
            sk: {
                "PRECISION_CANDIDATE": count(sk, "PRECISION_CANDIDATE"),
                "DIRECT_SUCCESS": count(sk, "DIRECT_SUCCESS"),
                "GATE_FAIL": count(sk, "GATE_FAIL"),
                "other": sum(1 for r in rows if r.get("source_kind") == sk and r.get("bin") not in ("PRECISION_CANDIDATE", "DIRECT_SUCCESS", "GATE_FAIL")),
            }
            for sk in sorted({r.get("source_kind") for r in rows})
        },
        "precision_candidate_ids": [r["ic_id"] for r in rows if r.get("bin") == "PRECISION_CANDIDATE"],
    }
    out = {"summary": summary, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plan_b_candidate_screen.json").write_text(json.dumps(jsafe(out), indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
