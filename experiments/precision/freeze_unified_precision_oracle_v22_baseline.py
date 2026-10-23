#!/usr/bin/env python
"""Freeze v2.2 Precision Oracle golden traces before v2.3 refactor.

Writes to results/diagnostics/unified_precision_oracle_v23/baseline_v22/
Do NOT regenerate after implementation changes.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.unified_precision_oracle import (  # noqa: E402
    PrecisionConfig,
    correct_dz_inward,
    measure_errors,
    run_precision,
    search_p1_p2_horizontal,
)
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402

OUT = ROOT / "results/diagnostics/unified_precision_oracle_v23/baseline_v22"
STATES = ROOT / "results/diagnostics/abc_precision_architecture/stable_states"
SRC = ROOT / "data_generation/oracles/precision/unified_precision_oracle.py"
B_POST_LAT = ROOT / "results/diagnostics/axis_aware_precision/snapshots/B_post_lateral.pkl"
POST_P5 = ROOT / "results/diagnostics/handoff_state_optimization/snapshots/post_P5.pkl"
AXIS_PREFIX = ROOT / "results/diagnostics/axis_aware_precision/post_lateral_prefix_actions.npz"


def jsafe(o):
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


def load_stable(plan: str) -> dict:
    p = {
        "A": STATES / "plan_a__ball_train_v2_0070__stable.pkl",
        "B": STATES / "plan_b__hard_910220__stable.pkl",
        "C": STATES / "plan_c__gbi_010__stable.pkl",
    }[plan]
    blob = pickle.loads(p.read_bytes())
    blob["plan"] = plan
    return blob


def capture_fingerprint(cap: dict) -> dict:
    ph = cap.get("physics") or {}
    qpos = np.asarray(ph.get("qpos"), float)
    qvel = np.asarray(ph.get("qvel"), float)
    return {
        "time": float(ph.get("time", 0.0)),
        "qpos_sha16": hashlib.sha256(qpos.tobytes()).hexdigest()[:16],
        "qvel_sha16": hashlib.sha256(qvel.tobytes()).hexdigest()[:16],
        "qpos_norm": float(np.linalg.norm(qpos)),
        "qvel_norm": float(np.linalg.norm(qvel)),
    }


def actions_fingerprint(actions: list) -> dict:
    arr = np.asarray(actions, np.float32) if actions else np.zeros((0, 7), np.float32)
    return {
        "n": int(arr.shape[0]),
        "sha16": hashlib.sha256(arr.tobytes()).hexdigest()[:16] if arr.size else "empty",
        "sum": float(np.sum(arr)) if arr.size else 0.0,
        "abs_max": float(np.max(np.abs(arr))) if arr.size else 0.0,
    }


def save_case(case_dir: Path, *, meta: dict, start_cap, terminal_cap, actions: list):
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "meta.json").write_text(json.dumps(jsafe(meta), indent=2), encoding="utf-8")
    (case_dir / "start_capture.pkl").write_bytes(pickle.dumps(start_cap))
    if terminal_cap is not None:
        (case_dir / "terminal_capture.pkl").write_bytes(pickle.dumps(terminal_cap))
    np.savez_compressed(case_dir / "actions.npz", actions=np.asarray(actions or [], np.float32))


def slim_result(res) -> dict:
    logs = res.primitive_logs or {}
    return {
        "status": res.status,
        "plan_id": res.plan_id,
        "precision_skipped": res.precision_skipped,
        "precision_executed": res.precision_executed,
        "phases": list(res.phases or []),
        "failures": list(res.failures or []),
        "n_actions": len(res.actions or []),
        "actions_fp": actions_fingerprint(res.actions or []),
        "start_fp": capture_fingerprint(res.ref and {} or {}) if False else None,
        "errors_entry": res.errors_entry,
        "errors_exit": res.errors_exit,
        "handoff": {
            k: (res.handoff or {}).get(k)
            for k in (
                "SUCCESS_NEW",
                "SUCCESS_12S",
                "SUCCESS_AXIS_V1",
                "std_reason",
                "reason",
                "theta_deg",
                "e_pos_mm",
            )
        },
        "dx_mechanism": logs.get("dx_mechanism"),
        "dx_partial": logs.get("dx_partial"),
        "dx_horizontal_recoverability": logs.get("dx_horizontal_recoverability"),
        "dx_horizontal_rejected": logs.get("dx_horizontal_rejected"),
        "dz_stop_reason": (logs.get("dz") or {}).get("stop_reason"),
        "dz_disp_total_mm": (logs.get("dz") or {}).get("disp_total_mm"),
        "dx_verified_stop": (logs.get("dx_verified") or {}).get("stop_reason"),
        "dx_verified_disp": (logs.get("dx_verified") or {}).get("disp_total_mm"),
        "decision": res.decision,
        "note": res.note,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases").mkdir(exist_ok=True)
    src_bytes = SRC.read_bytes()
    try:
        rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        rev = "unknown"

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": rev,
        "source_path": str(SRC.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(src_bytes).hexdigest(),
        "source_nbytes": len(src_bytes),
        "label": "v2.2_frozen_functional_baseline",
        "cases": {},
    }
    # Preserve exact source snapshot
    (OUT / "unified_precision_oracle_v22_source.py").write_bytes(src_bytes)

    env = make_env()
    st_a, st_b, st_c = load_stable("A"), load_stable("B"), load_stable("C")
    t0 = time.time()

    # 1. Direct C
    print("=== freeze Direct_C ===", flush=True)
    cfg_c = PrecisionConfig(allow_horizontal_dx=False)
    res_c = run_precision(env, st_c, "C", cfg_c)
    meta_c = slim_result(res_c)
    meta_c["case_id"] = "Direct_C"
    meta_c["config"] = asdict_cfg(cfg_c)
    meta_c["start_fp"] = capture_fingerprint(st_c["capture"])
    meta_c["terminal_fp"] = capture_fingerprint(res_c.capture) if res_c.capture else None
    save_case(
        OUT / "cases" / "Direct_C",
        meta=meta_c,
        start_cap=st_c["capture"],
        terminal_cap=res_c.capture,
        actions=res_c.actions or [],
    )
    manifest["cases"]["Direct_C"] = {
        "status": meta_c["status"],
        "SUCCESS_NEW": (meta_c["handoff"] or {}).get("SUCCESS_NEW"),
        "n_actions": meta_c["n_actions"],
        "actions_fp": meta_c["actions_fp"],
    }
    print(f"  {meta_c['status']} NEW={meta_c['handoff'].get('SUCCESS_NEW')}", flush=True)

    # 2. Plan A
    print("=== freeze Plan_A ===", flush=True)
    cfg_a = PrecisionConfig(allow_horizontal_dx=True, use_p1_p2_search=False)
    res_a = run_precision(env, st_a, "A", cfg_a)
    meta_a = slim_result(res_a)
    meta_a["case_id"] = "Plan_A"
    meta_a["config"] = asdict_cfg(cfg_a)
    meta_a["start_fp"] = capture_fingerprint(st_a["capture"])
    meta_a["terminal_fp"] = capture_fingerprint(res_a.capture) if res_a.capture else None
    save_case(OUT / "cases" / "Plan_A", meta=meta_a, start_cap=st_a["capture"], terminal_cap=res_a.capture, actions=res_a.actions or [])
    # also dump full primitive logs (no actions/captures)
    (OUT / "cases" / "Plan_A" / "primitive_logs.json").write_text(
        json.dumps(jsafe(_strip_heavy(res_a.primitive_logs)), indent=2), encoding="utf-8"
    )
    manifest["cases"]["Plan_A"] = {
        "status": meta_a["status"],
        "SUCCESS_NEW": (meta_a["handoff"] or {}).get("SUCCESS_NEW"),
        "n_actions": meta_a["n_actions"],
        "actions_fp": meta_a["actions_fp"],
        "dx_mechanism": meta_a["dx_mechanism"],
        "phases": meta_a["phases"],
    }
    print(f"  {meta_a['status']} NEW={meta_a['handoff'].get('SUCCESS_NEW')} n={meta_a['n_actions']}", flush=True)

    # 3. Plan B verified path (horizontal allowed, no P1/P2 search — matches H_reg_B)
    print("=== freeze Plan_B ===", flush=True)
    cfg_b = PrecisionConfig(allow_horizontal_dx=True, use_p1_p2_search=False)
    res_b = run_precision(env, st_b, "B", cfg_b)
    meta_b = slim_result(res_b)
    meta_b["case_id"] = "Plan_B"
    meta_b["config"] = asdict_cfg(cfg_b)
    meta_b["start_fp"] = capture_fingerprint(st_b["capture"])
    meta_b["terminal_fp"] = capture_fingerprint(res_b.capture) if res_b.capture else None
    save_case(OUT / "cases" / "Plan_B", meta=meta_b, start_cap=st_b["capture"], terminal_cap=res_b.capture, actions=res_b.actions or [])
    (OUT / "cases" / "Plan_B" / "primitive_logs.json").write_text(
        json.dumps(jsafe(_strip_heavy(res_b.primitive_logs)), indent=2), encoding="utf-8"
    )
    manifest["cases"]["Plan_B"] = {
        "status": meta_b["status"],
        "SUCCESS_NEW": (meta_b["handoff"] or {}).get("SUCCESS_NEW"),
        "n_actions": meta_b["n_actions"],
        "actions_fp": meta_b["actions_fp"],
        "dx_mechanism": meta_b["dx_mechanism"],
        "dz_stop_reason": meta_b["dz_stop_reason"],
        "dz_disp_total_mm": meta_b["dz_disp_total_mm"],
        "phases": meta_b["phases"],
    }
    print(
        f"  {meta_b['status']} NEW={meta_b['handoff'].get('SUCCESS_NEW')} "
        f"mech={meta_b['dx_mechanism']} dz_stop={meta_b['dz_stop_reason']}",
        flush=True,
    )

    # 4. B_post_lateral inward
    print("=== freeze B_post_lateral ===", flush=True)
    post = pickle.loads(B_POST_LAT.read_bytes())
    st_bl = {
        "capture": post["capture"],
        "ref": post.get("ref") or st_b["ref"],
        "stabilization_prefix_tokens": st_b.get("stabilization_prefix_tokens"),
        "precision_prefix_actions": np.load(AXIS_PREFIX)["actions"].tolist() if AXIS_PREFIX.exists() else [],
    }
    cfg_bl = PrecisionConfig(allow_horizontal_dx=False, ez_tol_mm=8.0)
    restore_capture(env, st_bl["capture"])
    e0 = measure_errors(env.sim)
    rin = correct_dz_inward(env, {"capture": st_bl["capture"], "ref": st_bl["ref"]}, PrecisionConfig())
    res_bl = run_precision(env, st_bl, "B", cfg_bl)
    meta_bl = {
        "case_id": "B_post_lateral",
        "inward_primitive": {
            "ok": rin.get("ok"),
            "stop_reason": rin.get("stop_reason"),
            "disp_total_mm": rin.get("disp_total_mm"),
            "goal_reached": rin.get("goal_reached"),
            "contract": rin.get("contract"),
            "errors_entry": e0,
            "errors": rin.get("errors"),
            "n_actions": len(rin.get("actions") or []),
            "actions_fp": actions_fingerprint(rin.get("actions") or []),
        },
        "pipeline": slim_result(res_bl),
        "config": asdict_cfg(cfg_bl),
        "start_fp": capture_fingerprint(st_bl["capture"]),
        "terminal_fp": capture_fingerprint(res_bl.capture) if res_bl.capture else None,
    }
    meta_bl["pipeline"]["config"] = asdict_cfg(cfg_bl)
    save_case(
        OUT / "cases" / "B_post_lateral",
        meta=meta_bl,
        start_cap=st_bl["capture"],
        terminal_cap=res_bl.capture,
        actions=res_bl.actions or [],
    )
    np.savez_compressed(
        OUT / "cases" / "B_post_lateral" / "inward_actions.npz",
        actions=np.asarray(rin.get("actions") or [], np.float32),
    )
    if rin.get("capture") is not None:
        (OUT / "cases" / "B_post_lateral" / "inward_terminal_capture.pkl").write_bytes(pickle.dumps(rin["capture"]))
    manifest["cases"]["B_post_lateral"] = {
        "inward_ok": rin.get("ok"),
        "inward_stop": rin.get("stop_reason"),
        "inward_disp": rin.get("disp_total_mm"),
        "pipeline_SUCCESS_NEW": (res_bl.handoff or {}).get("SUCCESS_NEW"),
        "pipeline_n_actions": len(res_bl.actions or []),
        "pipeline_actions_fp": actions_fingerprint(res_bl.actions or []),
    }
    print(
        f"  inward stop={rin.get('stop_reason')} disp={rin.get('disp_total_mm')} "
        f"pipe NEW={(res_bl.handoff or {}).get('SUCCESS_NEW')}",
        flush=True,
    )

    # 5. Outward C (pipeline already Direct_C; also freeze outward primitive alone)
    print("=== freeze Outward_C ===", flush=True)
    from data_generation.oracles.precision.unified_precision_oracle import correct_dz_outward

    restore_capture(env, st_c["capture"])
    e_out0 = measure_errors(env.sim)
    rout = correct_dz_outward(env, {"capture": st_c["capture"], "ref": st_c["ref"]}, PrecisionConfig())
    meta_out = {
        "case_id": "Outward_C",
        "errors_entry": e_out0,
        "ok": rout.get("ok"),
        "stop_reason": rout.get("stop_reason"),
        "disp_total_mm": rout.get("disp_total_mm"),
        "goal_reached": rout.get("goal_reached"),
        "contract": rout.get("contract"),
        "skipped_or_zero_travel": float(rout.get("disp_total_mm") or 0.0) < 1e-9,
        "n_actions": len(rout.get("actions") or []),
        "actions_fp": actions_fingerprint(rout.get("actions") or []),
        "errors": rout.get("errors"),
        "start_fp": capture_fingerprint(st_c["capture"]),
        "terminal_fp": capture_fingerprint(rout["capture"]) if rout.get("capture") else None,
        "pipeline_direct_C_SUCCESS_NEW": (res_c.handoff or {}).get("SUCCESS_NEW"),
    }
    save_case(
        OUT / "cases" / "Outward_C",
        meta=meta_out,
        start_cap=st_c["capture"],
        terminal_cap=rout.get("capture"),
        actions=rout.get("actions") or [],
    )
    manifest["cases"]["Outward_C"] = {
        "ok": rout.get("ok"),
        "stop_reason": rout.get("stop_reason"),
        "disp_total_mm": rout.get("disp_total_mm"),
        "n_actions": len(rout.get("actions") or []),
        "actions_fp": actions_fingerprint(rout.get("actions") or []),
    }
    print(f"  stop={rout.get('stop_reason')} disp={rout.get('disp_total_mm')}", flush=True)

    # 6. post_P5 negative dx → skip
    print("=== freeze post_P5 ===", flush=True)
    p5 = pickle.loads(POST_P5.read_bytes())
    cfg_p5 = PrecisionConfig(allow_horizontal_dx=True, use_p1_p2_search=False, use_goal_driven_slip=True)
    res_p5 = run_precision(env, {"capture": p5["capture"], "ref": p5["ref"]}, "B", cfg_p5)
    meta_p5 = slim_result(res_p5)
    meta_p5["case_id"] = "post_P5"
    meta_p5["config"] = asdict_cfg(cfg_p5)
    meta_p5["start_fp"] = capture_fingerprint(p5["capture"])
    meta_p5["terminal_fp"] = capture_fingerprint(res_p5.capture) if res_p5.capture else None
    save_case(
        OUT / "cases" / "post_P5",
        meta=meta_p5,
        start_cap=p5["capture"],
        terminal_cap=res_p5.capture,
        actions=res_p5.actions or [],
    )
    manifest["cases"]["post_P5"] = {
        "status": meta_p5["status"],
        "SUCCESS_NEW": (meta_p5["handoff"] or {}).get("SUCCESS_NEW"),
        "n_actions": meta_p5["n_actions"],
        "precision_skipped": meta_p5["precision_skipped"],
        "actions_fp": meta_p5["actions_fp"],
    }
    print(f"  {meta_p5['status']} skip={meta_p5['precision_skipped']} NEW={meta_p5['handoff'].get('SUCCESS_NEW')}", flush=True)

    # 7. hard_910220 horizontal rejected → verified fallback (with P1/P2 search)
    print("=== freeze hard_910220_horizontal_reject ===", flush=True)
    blob_b = {"capture": st_b["capture"], "ref": st_b["ref"]}
    search = search_p1_p2_horizontal(env, blob_b)
    win = search.get("winner_params") if search.get("ok") else None
    cfg_h = PrecisionConfig(
        allow_horizontal_dx=True,
        use_p1_p2_search=True,
        p1_p2_winner=win,
        accept_partial_horizontal_dx=True,
        use_goal_driven_slip=True,
        require_horizontal_recoverability=True,
    )
    res_h = run_precision(env, st_b, "B", cfg_h)
    meta_h = slim_result(res_h)
    meta_h["case_id"] = "hard_910220_horizontal_reject"
    meta_h["config"] = asdict_cfg(cfg_h)
    meta_h["p1_p2_winner"] = win
    meta_h["p1_p2_search_ok"] = search.get("ok")
    meta_h["start_fp"] = capture_fingerprint(st_b["capture"])
    meta_h["terminal_fp"] = capture_fingerprint(res_h.capture) if res_h.capture else None
    save_case(
        OUT / "cases" / "hard_910220_horizontal_reject",
        meta=meta_h,
        start_cap=st_b["capture"],
        terminal_cap=res_h.capture,
        actions=res_h.actions or [],
    )
    (OUT / "cases" / "hard_910220_horizontal_reject" / "primitive_logs.json").write_text(
        json.dumps(jsafe(_strip_heavy(res_h.primitive_logs)), indent=2), encoding="utf-8"
    )
    manifest["cases"]["hard_910220_horizontal_reject"] = {
        "status": meta_h["status"],
        "SUCCESS_NEW": (meta_h["handoff"] or {}).get("SUCCESS_NEW"),
        "n_actions": meta_h["n_actions"],
        "actions_fp": meta_h["actions_fp"],
        "dx_mechanism": meta_h["dx_mechanism"],
        "recoverability": meta_h["dx_horizontal_recoverability"],
        "horizontal_rejected": meta_h["dx_horizontal_rejected"] is not None,
        "phases": meta_h["phases"],
    }
    print(
        f"  {meta_h['status']} NEW={meta_h['handoff'].get('SUCCESS_NEW')} "
        f"mech={meta_h['dx_mechanism']} rejected={meta_h['dx_horizontal_rejected'] is not None}",
        flush=True,
    )

    manifest["wall_s"] = round(time.time() - t0, 1)
    (OUT / "MANIFEST.json").write_text(json.dumps(jsafe(manifest), indent=2), encoding="utf-8")
    print(json.dumps({"wall_s": manifest["wall_s"], "n_cases": len(manifest["cases"])}, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)


def asdict_cfg(cfg: PrecisionConfig) -> dict:
    from dataclasses import asdict

    return asdict(cfg)


def _strip_heavy(logs: dict | None) -> dict:
    if not logs:
        return {}
    out = {}
    for k, v in logs.items():
        if k in ("actions", "capture", "series"):
            continue
        if isinstance(v, dict):
            out[k] = _strip_heavy(v)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            out[k] = [_strip_heavy(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    main()
