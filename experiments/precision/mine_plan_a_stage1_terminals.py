#!/usr/bin/env python
"""Mine post–Stage-1 terminals for all 71 unified Plan A roots.

Replays historical continuous Stage-1 action tapes onto the original snapshots,
then probes direct HANDOFF. Does NOT relabel K49 tokens as continuous Oracle demos.

Outputs:
  results/data_generation/plan_a_stage1_terminals_v1/{ics,manifest.jsonl,EVAL.json}
"""

from __future__ import annotations

import json
import pickle
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.entry_conditions import assess_need  # noqa: E402
from data_generation.oracles.precision.handoff import probe_direct_handoff  # noqa: E402
from data_generation.oracles.precision.residuals import measure_errors  # noqa: E402
from data_generation.oracles.precision.state import PrecisionConfig  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/data_generation/plan_a_stage1_terminals_v1"
ICS = OUT / "ics"
MANIFEST = OUT / "manifest.jsonl"
EVAL = OUT / "EVAL.json"
UNIFIED = ROOT / "data/unified_experts/manifest.jsonl"
FINAL_V2 = ROOT / "results/data_generation/stage2_abc_final_v2/manifest.jsonl"


def jsafe(o: Any) -> Any:
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def load_unified_a() -> list[dict]:
    rows = []
    for line in UNIFIED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("plan_family", "")).upper() != "A":
            continue
        rows.append(row)
    rows.sort(key=lambda r: str(r.get("physical_root_id") or r.get("expert_id")))
    return rows


def final_v2_a_roots() -> dict[str, str]:
    """Map physical root hint → final_v2 sample_id for the 20 PILOT_A ICs."""
    out: dict[str, str] = {}
    if not FINAL_V2.is_file():
        return out
    for line in FINAL_V2.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("plan_provenance") != "A":
            continue
        sid = str(row["sample_id"])
        # PILOT_A__ball_train_0017__hash → ball_train_0017
        parts = sid.split("__")
        if len(parts) >= 2:
            out[parts[1]] = sid
        out[sid] = sid
    return out


def replay_stage1(env, snap_blob: dict, actions: np.ndarray) -> dict[str, Any]:
    restore_capture(env, snap_blob["capture"])
    n = 0
    for a in np.asarray(actions, dtype=np.float32):
        step_legal(env.sim, a)
        n += 1
    cap = take_capture(env)
    err = measure_errors(env.sim)
    return {
        "capture": cap,
        "ref": snap_blob["ref"],
        "n_actions_replayed": n,
        "errors_terminal": err,
        "bilateral": bool(err.get("bilateral")),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ICS.mkdir(parents=True, exist_ok=True)
    env = make_env()
    cfg_need = PrecisionConfig()
    final_map = final_v2_a_roots()
    rows = load_unified_a()
    eval_rows: list[dict] = []
    manifest_lines: list[str] = []

    print(f"mining {len(rows)} unified Plan A roots...", flush=True)
    for i, row in enumerate(rows):
        root_id = str(row.get("physical_root_id") or row.get("expert_id"))
        expert_id = str(row.get("expert_id"))
        mech = str(row.get("contact_grasp_mechanism") or "")
        ref = row.get("initial_physical_state_ref") or {}
        snap_rel = ref.get("snapshot")
        npz_rel = ref.get("continuous_npz")
        rec: dict[str, Any] = {
            "physical_root_id": root_id,
            "unified_expert_id": expert_id,
            "unified_mechanism": mech,
            "unified_SUCCESS_NEW": bool(row.get("SUCCESS_NEW")),
            "in_final_v2_20": root_id in final_map or any(root_id in k for k in final_map),
            "final_v2_sample_id": final_map.get(root_id),
            "snapshot": snap_rel,
            "continuous_npz": npz_rel,
        }
        try:
            if not snap_rel or not npz_rel:
                raise FileNotFoundError("missing snapshot or continuous_npz")
            snap_path = ROOT / str(snap_rel)
            if not snap_path.is_file():
                # Unified metadata often points at deleted diagnostics/raw paths;
                # prefer the canonical relocated snapshot under data/snapshots/.
                alts = [
                    ROOT / "data" / "snapshots" / "train" / f"{root_id}.pkl",
                    ROOT / "data" / "snapshots" / f"{root_id}.pkl",
                ]
                hit = next((p for p in alts if p.is_file()), None)
                if hit is None:
                    raise FileNotFoundError(snap_path)
                snap_path = hit
                snap_rel = str(hit.relative_to(ROOT)).replace("\\", "/")
            npz_path = ROOT / str(npz_rel)
            if not npz_path.is_file():
                raise FileNotFoundError(npz_path)
            snap_blob = pickle.loads(snap_path.read_bytes())
            if "capture" not in snap_blob or "ref" not in snap_blob:
                raise KeyError("snapshot missing capture/ref")
            archive = np.load(npz_path)
            if "actions" in archive.files:
                actions = archive["actions"]
                term = replay_stage1(env, snap_blob, actions)
            else:
                # immediate_handoff tapes store only handoff_obs; snapshot is terminal.
                restore_capture(env, snap_blob["capture"])
                err = measure_errors(env.sim)
                term = {
                    "capture": snap_blob["capture"],
                    "ref": snap_blob["ref"],
                    "n_actions_replayed": 0,
                    "errors_terminal": err,
                    "bilateral": bool(err.get("bilateral")),
                    "immediate_handoff_snapshot_terminal": True,
                }
            if not term["bilateral"]:
                rec["stage1_terminal_ok"] = False
                rec["stage1_fail"] = "lost_bilateral_after_stage1_replay"
                eval_rows.append(rec)
                print(f"[{i+1}/{len(rows)}] {root_id}: bilateral lost", flush=True)
                continue
            restore_capture(env, term["capture"])
            need = assess_need(measure_errors(env.sim), cfg_need)
            direct = probe_direct_handoff(env, term["capture"], term["ref"])
            sample_id = f"A_STAGE1__{root_id}"
            ic_path = ICS / f"{sample_id}.pkl"
            ic_blob = {
                "capture": term["capture"],
                "ref": term["ref"],
                "plan": "A",
                "plan_provenance": "A_STAGE1",
                "physical_root_id": root_id,
                "unified_expert_id": expert_id,
                "unified_mechanism": mech,
                "source_snapshot": str(snap_rel).replace("\\", "/"),
                "source_continuous_npz": str(npz_rel).replace("\\", "/"),
                "n_stage1_actions": int(term["n_actions_replayed"]),
                "mined_utc": datetime.now(timezone.utc).isoformat(),
                "note": "Stage-1 continuous tape replay terminal; not a K49 relabel",
            }
            ic_path.write_bytes(pickle.dumps(ic_blob))
            man = {
                "sample_id": sample_id,
                "plan_provenance": "A_STAGE1",
                "physical_root_id": root_id,
                "unified_expert_id": expert_id,
                "unified_mechanism": mech,
                "final_category": "STAGE1_TERMINAL",
                "source_pool": "plan_a_stage1_terminals_v1",
                "direct": {
                    "SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
                    "SUCCESS_12S": bool(direct.get("SUCCESS_12S")),
                    "reason": direct.get("reason"),
                },
                "precision_entry": jsafe(need),
                "in_final_v2_20": bool(rec["final_v2_sample_id"]),
            }
            manifest_lines.append(json.dumps(man, ensure_ascii=False))
            rec.update(
                {
                    "stage1_terminal_ok": True,
                    "sample_id": sample_id,
                    "ic_path": str(ic_path.relative_to(ROOT)).replace("\\", "/"),
                    "direct_SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
                    "direct_SUCCESS_12S": bool(direct.get("SUCCESS_12S")),
                    "direct_reason": direct.get("reason"),
                    "precision_entry": jsafe(need),
                    "admit_a_direct_candidate": True,
                    "precision_recoverable_candidate": bool(
                        need.get("dtheta_needed")
                        or need.get("dx_needed")
                        or need.get("dz_needed")
                        or need.get("any_needed")
                    ),
                }
            )
            print(
                f"[{i+1}/{len(rows)}] {root_id}: direct_NEW={rec['direct_SUCCESS_NEW']} "
                f"need={need.get('any_needed')}",
                flush=True,
            )
        except Exception as exc:
            rec["stage1_terminal_ok"] = False
            rec["stage1_fail"] = f"{type(exc).__name__}: {exc}"
            rec["traceback"] = traceback.format_exc(limit=4)
            print(f"[{i+1}/{len(rows)}] {root_id}: FAIL {exc}", flush=True)
        eval_rows.append(rec)

    MANIFEST.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    n_ok = sum(1 for r in eval_rows if r.get("stage1_terminal_ok"))
    n_direct = sum(1 for r in eval_rows if r.get("direct_SUCCESS_NEW"))
    n_need = sum(1 for r in eval_rows if r.get("precision_recoverable_candidate"))
    summary = {
        "n_unified_a": len(rows),
        "n_terminal_ok": n_ok,
        "n_direct_SUCCESS_NEW": n_direct,
        "n_precision_entry_needed": n_need,
        "n_terminal_fail": len(rows) - n_ok,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": jsafe(eval_rows),
    }
    EVAL.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"done: terminal_ok={n_ok}/{len(rows)} direct_NEW={n_direct} precision_need={n_need}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
