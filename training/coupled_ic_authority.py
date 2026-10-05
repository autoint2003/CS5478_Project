"""Visually audited recovery-authority experiment on coupled s=2.0 IC.

    python training/coupled_ic_authority.py --all
    python training/coupled_ic_authority.py --freeze
    python training/coupled_ic_authority.py --repeat
    python training/coupled_ic_authority.py --zero-story
    python training/coupled_ic_authority.py --primitives
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import ori_error_deg
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import physical_pack
from training.demo_teleport_recovery_state import (
    AUTH_DIR,
    G_HOLD,
    IC_NPZ,
    PAIR_MU,
    S_COUPLED,
    SEQ_JSON,
    apply_coupled_pose,
    apply_schedule,
    build_parent,
    coupled_pose_from_ref,
    contact_debug,
    contact_events,
    cyl_axis_hand,
    dump_branch_state,
    freeze_targets,
    load_canonical_ic,
    load_or_measure_ref,
    make_parent_sim,
    pack_canonical_ic,
    restore_canonical_ic,
    run_teleport_case,
    save_canonical_ic,
    slim,
    unsigned_axis_angle_deg,
    _cam,
    _row,
    zero_continue,
)
from training.impact_demo_core import save_rgb
from training.replay_core import freeze, restore_replay

SHOT = AUTH_DIR / "frames"


def _json(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def maxabs(a, b) -> float:
    a = np.asarray(a, float).reshape(-1)
    b = np.asarray(b, float).reshape(-1)
    if a.size == 0 and b.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b))) if a.size == b.size else float("nan")


def compare_branch(a: dict, b: dict) -> dict:
    keys = ("qpos", "qvel", "act", "ctrl", "p_des", "v_cmd")
    out = {f"maxabs_{k}": maxabs(a[k], b[k]) for k in keys if k in a and k in b}
    out["ori_target_deg"] = float(
        ori_error_deg(np.asarray(a["r_des"]), np.asarray(b["r_des"]))
    )
    out["dtime"] = abs(float(a["time"]) - float(b["time"]))
    return out


def freeze_ic() -> dict:
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    SHOT.mkdir(parents=True, exist_ok=True)
    ref = load_or_measure_ref()
    parent = build_parent()
    if not parent.get("ok"):
        raise RuntimeError(parent)
    sim = make_parent_sim()
    restore_replay(sim, parent["snap"])
    freeze(sim)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    save_rgb(SHOT / "00_parent_airborne.png", sim, _cam(sim))
    before = dump_branch_state(sim)
    before_rel = {
        "nL": int(physical_pack(sim)["nL"]),
        "nR": int(physical_pack(sim)["nR"]),
        "contacts": contact_debug(sim)["contacts"],
        "cyl_axis_h": cyl_axis_hand(sim).tolist(),
        "R_rel": np.array(physical_pack(sim)["R_rel"], float).tolist(),
        "p_rel_h": np.array(physical_pack(sim)["rh"], float).tolist(),
    }
    dp, Rd = coupled_pose_from_ref(ref, S_COUPLED, "coupled")
    applied = apply_coupled_pose(sim, dp, Rd)
    after = dump_branch_state(sim)
    after_pack = physical_pack(sim)
    after_dbg = contact_debug(sim)
    save_rgb(SHOT / "01_after_teleport.png", sim, _cam(sim))
    ic = pack_canonical_ic(sim)
    save_canonical_ic(IC_NPZ, ic)
    axis_pre = np.asarray(before_rel["cyl_axis_h"], float)
    axis_post = cyl_axis_hand(sim)
    w = np.asarray(ref["rotvec_h"], float)
    ang = float(np.linalg.norm(w) * S_COUPLED)
    # wrap display angle to [0,180] for SO(3) of this construction
    from training.demo_teleport_recovery_state import rotvec_from_rotmat

    w_s = rotvec_from_rotmat(Rd)
    audit = {
        "s": S_COUPLED,
        "delta_p_h_mm": (1e3 * dp).tolist(),
        "raw_so3_axis_h": (w_s / (np.linalg.norm(w_s) + 1e-15)).tolist(),
        "raw_so3_angle_deg": float(np.degrees(np.linalg.norm(w_s))),
        "cyl_axis_h_before": axis_pre.tolist(),
        "cyl_axis_h_after": axis_post.tolist(),
        "unsigned_cyl_axis_tilt_deg": unsigned_axis_angle_deg(axis_pre, axis_post),
        "signed_dot_axis": float(np.dot(axis_pre / np.linalg.norm(axis_pre), axis_post / np.linalg.norm(axis_post))),
        "nL_before": before_rel["nL"],
        "nR_before": before_rel["nR"],
        "nL_after": int(after_pack["nL"]),
        "nR_after": int(after_pack["nR"]),
        "Fn_L_after": float(after_pack["Fn_L"]),
        "Fn_R_after": float(after_pack["Fn_R"]),
        "contacts_before": before_rel["contacts"],
        "contacts_after": after_dbg["contacts"],
        "min_dist_after": after_dbg.get("min_dist"),
        "p_hand": after["p_hand"].tolist(),
        "p_des": after["p_des"].tolist(),
        "p_des_minus_p_hand_mm": (1e3 * (after["p_des"] - after["p_hand"])).tolist(),
        "ori_err_deg": after["ori_err_deg"],
        "v_cmd": after["v_cmd"].tolist(),
        "ctrl7": float(after["ctrl"][7]) if after["ctrl"].size > 7 else None,
        "ctrlrange7": [
            float(sim.ids.ctrl_low[7]),
            float(sim.ids.ctrl_high[7]),
        ],
        "maxabs_arm_qpos": maxabs(before["arm_qpos"], after["arm_qpos"]),
        "maxabs_arm_qvel": maxabs(before["arm_qvel"], after["arm_qvel"]),
        "maxabs_obj_qpos": maxabs(before["object_qpos"], after["object_qpos"]),
        "maxabs_obj_qvel": maxabs(before["object_qvel"], after["object_qvel"]),
        "maxabs_p_des": maxabs(before["p_des"], after["p_des"]),
        "maxabs_r_des_flat": maxabs(before["r_des"].reshape(-1), after["r_des"].reshape(-1)),
        "maxabs_v_cmd": maxabs(before["v_cmd"], after["v_cmd"]),
        "maxabs_ctrl": maxabs(before["ctrl"], after["ctrl"]),
        "object_linvel_before": before["object_qvel"][:3].tolist(),
        "object_linvel_after": after["object_qvel"][:3].tolist(),
        "object_angvel_before": before["object_qvel"][3:].tolist(),
        "object_angvel_after": after["object_qvel"][3:].tolist(),
        "ic_path": str(IC_NPZ),
        "restored_fields": [
            "qpos", "qvel", "act", "ctrl", "time", "qacc_warmstart",
            "p_des", "r_des", "v_cmd", "w_cmd", "v_des", "w_des",
            "phase", "t_phase", "t_stable", "t_held", "lift_started", "success",
            "fg_cmd", "grasp_offset", "grasp_xy", "grasp_z",
            "meter prel_ref r_rel_ref D in_recovery exit_count",
            "captured", "mass", "friction", "pair_mu via set_finger_object_sliding_mu",
        ],
        "construction": "object free-joint qpos only; object qvel copied; mj_forward; no settle tick",
        "applied_obj_linvel_keep": applied["v_keep"][:3],
        "applied_obj_angvel_keep": applied["v_keep"][3:],
        "mass": after["mass"],
        "pair_mu_object": after["pair_mu_object"],
        "obj_contype": after["obj_contype"],
        "table_contype": after["table_contype"],
    }
    (AUTH_DIR / "teleport_audit.json").write_text(
        json.dumps(audit, indent=2, default=_json), encoding="utf-8"
    )
    print(json.dumps({k: audit[k] for k in (
        "delta_p_h_mm", "raw_so3_angle_deg", "unsigned_cyl_axis_tilt_deg",
        "nL_after", "nR_after", "maxabs_arm_qpos", "maxabs_obj_qvel",
        "p_des_minus_p_hand_mm", "ctrlrange7",
    )}, indent=2))
    return audit


def run_zero_from_ic(ic, tag: str, shots=False) -> dict:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    restore_canonical_ic(sim, ic)
    st0 = dump_branch_state(sim)
    t0 = float(sim.data.time)
    log = [_row(sim, "restore")]
    if shots:
        save_rgb(SHOT / f"{tag}_t0.png", sim, _cam(sim))
    cont = zero_continue(sim, cfg, viewer=None, pacer=None, s=S_COUPLED, log=log, t_tele=t0, mode="ZERO")
    ev = contact_events(log, t0)
    if shots:
        save_rgb(SHOT / f"{tag}_end.png", sim, _cam(sim))
    out = {
        "tag": tag,
        "task_success": cont.get("task_success"),
        "drop": cont.get("drop"),
        "lift_completed": cont.get("lift_completed"),
        "hold_completed": cont.get("hold_completed"),
        "t_lift_done": cont.get("t_lift_done"),
        "t_hold0": cont.get("t_hold0"),
        **ev,
        "end_nL": cont.get("end_nL"),
        "end_nR": cont.get("end_nR"),
        "end_e_x_mm": cont.get("end_e_x_mm"),
        "init": {
            "nL": log[0]["nL"],
            "nR": log[0]["nR"],
            "e_x_mm": 1e3 * log[0]["e_x"],
            "min_dist": log[0]["min_dist"],
        },
        "st0": {k: _json(st0[k]) for k in ("time", "mass", "pair_mu_object", "obj_contype", "table_contype")},
        "log": log,
    }
    return out


def repeat_zero(n: int = 5) -> dict:
    ic = load_canonical_ic(IC_NPZ)
    rows = []
    refs = None
    diffs = []
    for i in range(n):
        print("ZERO restore", i + 1, flush=True)
        r = run_zero_from_ic(ic, f"repeat_{i}", shots=(i == 0))
        slim_r = {k: v for k, v in r.items() if k not in ("log", "st0")}
        rows.append(slim_r)
        sim = make_parent_sim()
        restore_canonical_ic(sim, ic)
        st = dump_branch_state(sim)
        if refs is None:
            refs = st
        else:
            diffs.append(compare_branch(refs, st))
        (AUTH_DIR / f"zero_repeat_{i}.json").write_text(
            json.dumps({"summary": slim_r, "log": r["log"]}, indent=2, default=_json),
            encoding="utf-8",
        )
    drops = [r.get("dt_drop") for r in rows]
    boths = [r.get("dt_both") for r in rows]
    report = {
        "n": n,
        "dt_both": boths,
        "dt_drop": drops,
        "dt_both_range": None if any(x is None for x in boths) else [min(boths), max(boths)],
        "dt_drop_range": None if any(x is None for x in drops) else [min(drops), max(drops)],
        "restore_diffs": diffs,
        "rows": rows,
    }
    (AUTH_DIR / "zero_repeatability.json").write_text(json.dumps(report, indent=2, default=_json), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("dt_both", "dt_drop", "restore_diffs")}, indent=2, default=_json))
    return report


def zero_full_story() -> dict:
    """Headless full FSM+teleport+ZERO with frames. Same path as the viewer."""
    AUTH_DIR.mkdir(parents=True, exist_ok=True)
    ref = load_or_measure_ref()
    print("ZERO full story from t=0 ...", flush=True)
    r = run_teleport_case(S_COUPLED, ref, "coupled", interactive=False, mode="ZERO")
    (AUTH_DIR / "zero_story.json").write_text(
        json.dumps({"summary": slim(r), "log": r.get("log", [])}, indent=2, default=_json),
        encoding="utf-8",
    )
    print(json.dumps(slim(r), indent=2, default=_json))
    return r


def run_primitive(name: str, schedule: list, ic, shots=True) -> dict:
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_parent_sim()
    restore_canonical_ic(sim, ic)
    st0 = dump_branch_state(sim)
    t0 = float(sim.data.time)
    log = [_row(sim, "restore")]
    if shots:
        save_rgb(SHOT / f"{name}_00_start.png", sim, _cam(sim))
    ev = apply_schedule(sim, cfg, schedule, viewer=None, pacer=None, s=S_COUPLED, log=log)
    if shots:
        save_rgb(SHOT / f"{name}_01_after_pulse.png", sim, _cam(sim))
    cont = zero_continue(sim, cfg, viewer=None, pacer=None, s=S_COUPLED, log=log, t_tele=t0, mode="RECOVERY")
    if shots:
        save_rgb(SHOT / f"{name}_02_end.png", sim, _cam(sim))
    cev = contact_events(log, t0)
    out = {
        "name": name,
        "schedule": schedule,
        "task_success": cont.get("task_success"),
        "drop": cont.get("drop"),
        "lift_completed": cont.get("lift_completed"),
        "hold_completed": cont.get("hold_completed"),
        "end_nL": cont.get("end_nL"),
        "end_nR": cont.get("end_nR"),
        "end_e_x_mm": cont.get("end_e_x_mm"),
        "recovery_events": ev,
        **cev,
        "init_nL": log[0]["nL"],
        "init_nR": log[0]["nR"],
        "p_des_minus_p_hand_mm_at_restore": (
            1e3 * (st0["p_des"] - st0["p_hand"])
        ).tolist(),
        "v_cmd_at_restore": st0["v_cmd"].tolist(),
        "ori_err_deg_at_restore": st0["ori_err_deg"],
        "log": log,
    }
    (AUTH_DIR / f"prim_{name}.json").write_text(
        json.dumps({k: v for k, v in out.items() if k != "log"} | {"log": log}, indent=2, default=_json),
        encoding="utf-8",
    )
    print(
        name,
        "success",
        out["task_success"],
        "drop",
        out["drop"],
        "dt_right",
        out.get("dt_right"),
        "dt_bilat",
        out.get("dt_bilat"),
        "dt_drop",
        out.get("dt_drop"),
        flush=True,
    )
    return out


def primitives() -> dict:
    ic = load_canonical_ic(IC_NPZ)
    sim = make_parent_sim()
    restore_canonical_ic(sim, ic)
    lo, hi = float(sim.ids.ctrl_low[7]), float(sim.ids.ctrl_high[7])
    stronger = G_HOLD
    if lo < G_HOLD - 1e-9:
        stronger = max(lo, G_HOLD - 2.0)  # modest, still legal
    families = {}

    # A grip
    families["A_grip_secure"] = run_primitive(
        "A_grip_secure",
        [{"name": "grip_secure", "tau": G_HOLD, "duration": 0.20, "freeze_targets": True}],
        ic,
    )
    if stronger < G_HOLD - 1e-9:
        families["A_grip_stronger"] = run_primitive(
            "A_grip_stronger",
            [{"name": "grip_stronger", "tau": stronger, "duration": 0.20, "freeze_targets": True}],
            ic,
        )

    # B hand-x representative both signs, then refine only if mixed
    families["B_hx_p04_100"] = run_primitive(
        "B_hx_p04_100",
        [
            {"name": "hx", "v_hx": 0.04, "tau": G_HOLD, "duration": 0.10},
            {"name": "stop", "duration": 0.0, "freeze_targets": True},
        ],
        ic,
    )
    families["B_hx_m04_100"] = run_primitive(
        "B_hx_m04_100",
        [
            {"name": "hx", "v_hx": -0.04, "tau": G_HOLD, "duration": 0.10},
            {"name": "stop", "duration": 0.0, "freeze_targets": True},
        ],
        ic,
    )
    winner = next((k for k, v in families.items() if v.get("task_success")), None)
    if winner:
        return _stop_success(families, winner, ic)

    # refine the better HX sign if either recaptures right contact
    def right_gain(v):
        dt = v.get("dt_right")
        return 1e9 if dt is None else dt

    better = min(("B_hx_p04_100", "B_hx_m04_100"), key=lambda k: right_gain(families[k]))
    sign = 1.0 if "p04" in better else -1.0
    for speed, dur in ((0.02, 0.10), (0.06, 0.10), (0.04, 0.20), (0.06, 0.20)):
        key = f"B_hx_{'p' if sign > 0 else 'm'}{int(abs(speed)*100)}_{int(dur*1000)}"
        families[key] = run_primitive(
            key,
            [
                {"name": "hx", "v_hx": sign * speed, "tau": G_HOLD, "duration": dur},
                {"name": "stop", "duration": 0.0, "freeze_targets": True},
            ],
            ic,
        )
        if families[key].get("task_success"):
            return _stop_success(families, key, ic)

    # C wrist both signs about r_des y = hand y at IC
    families["C_w_p15_100"] = run_primitive(
        "C_w_p15_100",
        [
            {"name": "wrist", "w_hy": 1.5, "tau": G_HOLD, "duration": 0.10},
            {"name": "stop", "duration": 0.0, "freeze_targets": True},
        ],
        ic,
    )
    families["C_w_m15_100"] = run_primitive(
        "C_w_m15_100",
        [
            {"name": "wrist", "w_hy": -1.5, "tau": G_HOLD, "duration": 0.10},
            {"name": "stop", "duration": 0.0, "freeze_targets": True},
        ],
        ic,
    )
    winner = next((k for k, v in families.items() if v.get("task_success")), None)
    if winner:
        return _stop_success(families, winner, ic)
    for w, dur in ((1.5, 0.20), (-1.5, 0.20), (3.0, 0.10), (-3.0, 0.10)):
        key = f"C_w_{'p' if w > 0 else 'm'}{int(abs(w)*10)}_{int(dur*1000)}"
        families[key] = run_primitive(
            key,
            [
                {"name": "wrist", "w_hy": w, "tau": G_HOLD, "duration": dur},
                {"name": "stop", "duration": 0.0, "freeze_targets": True},
            ],
            ic,
        )
        if families[key].get("task_success"):
            return _stop_success(families, key, ic)

    # D unload / open
    families["D_unload80_reclose"] = run_primitive(
        "D_unload80_reclose",
        [
            {"name": "unload", "tau": -2.0, "duration": 0.08, "freeze_targets": True},
            {"name": "reclose", "tau": G_HOLD, "duration": 0.12},
            {"name": "stop", "duration": 0.0, "freeze_targets": True},
        ],
        ic,
    )
    if families["D_unload80_reclose"].get("task_success"):
        return _stop_success(families, "D_unload80_reclose", ic)
    families["D_open40_reclose"] = run_primitive(
        "D_open40_reclose",
        [
            {"name": "open", "tau": -1.0, "duration": 0.04, "freeze_targets": True},
            {"name": "reclose", "tau": G_HOLD, "duration": 0.12},
            {"name": "stop", "duration": 0.0, "freeze_targets": True},
        ],
        ic,
    )
    if families["D_open40_reclose"].get("task_success"):
        return _stop_success(families, "D_open40_reclose", ic)

    # E combinations justified if HX or wrist changed right-contact timing
    hx_m = families["B_hx_m04_100"]
    hx_p = families["B_hx_p04_100"]
    rationale = []
    if (hx_m.get("dt_right") is not None) or (hx_p.get("dt_right") is not None):
        use_sign = -0.04 if right_gain(hx_m) <= right_gain(hx_p) else 0.04
        rationale.append(f"HX sign {use_sign} recaptured or delayed drop vs other; pair with unload")
        key = "E_unload_hx_reclose"
        families[key] = run_primitive(
            key,
            [
                {"name": "unload", "tau": -2.0, "duration": 0.08, "freeze_targets": True},
                {"name": "hx", "v_hx": use_sign, "tau": -2.0, "duration": 0.10},
                {"name": "reclose", "tau": G_HOLD, "duration": 0.12},
                {"name": "stop", "duration": 0.0, "freeze_targets": True},
            ],
            ic,
        )
        if families[key].get("task_success"):
            return _stop_success(families, key, ic, extra={"rationale": rationale})
        key = "E_hx_reclose"
        families[key] = run_primitive(
            key,
            [
                {"name": "hx", "v_hx": use_sign, "tau": G_HOLD, "duration": 0.10},
                {"name": "reclose", "tau": G_HOLD, "duration": 0.12},
                {"name": "stop", "duration": 0.0, "freeze_targets": True},
            ],
            ic,
        )
        if families[key].get("task_success"):
            return _stop_success(families, key, ic, extra={"rationale": rationale})

    wr_p, wr_m = families["C_w_p15_100"], families["C_w_m15_100"]
    if (wr_p.get("dt_right") is not None) or (wr_m.get("dt_bilat") is not None) or (wr_p.get("dt_bilat") is not None):
        wuse = 1.5 if right_gain(wr_p) <= right_gain(wr_m) else -1.5
        rationale.append(f"wrist {wuse} better right/bilat; pair with reclose")
        key = "E_wrist_reclose"
        families[key] = run_primitive(
            key,
            [
                {"name": "wrist", "w_hy": wuse, "tau": G_HOLD, "duration": 0.10},
                {"name": "reclose", "tau": G_HOLD, "duration": 0.12},
                {"name": "stop", "duration": 0.0, "freeze_targets": True},
            ],
            ic,
        )
        if families[key].get("task_success"):
            return _stop_success(families, key, ic, extra={"rationale": rationale})

    summary = {k: {kk: vv for kk, vv in v.items() if kk != "log"} for k, v in families.items()}
    (AUTH_DIR / "primitives_summary.json").write_text(
        json.dumps({"success": None, "ctrlrange7": [lo, hi], "families": summary, "rationale": rationale}, indent=2, default=_json),
        encoding="utf-8",
    )
    print("NO RECOVERY AUTHORITY DEMONSTRATED FOR THIS IC")
    return {"success": None, "families": summary}


def _stop_success(families, key, ic, extra=None) -> dict:
    sch = families[key]["schedule"]
    SEQ_JSON.write_text(
        json.dumps({"name": key, "schedule": sch, **(extra or {})}, indent=2),
        encoding="utf-8",
    )
    summary = {k: {kk: vv for kk, vv in v.items() if kk != "log"} for k, v in families.items()}
    payload = {"success": key, "schedule": sch, "families": summary, **(extra or {})}
    (AUTH_DIR / "primitives_summary.json").write_text(json.dumps(payload, indent=2, default=_json), encoding="utf-8")
    print("SUCCESS primitive", key)
    # translation-only diagnostic
    ref = load_or_measure_ref()
    parent = build_parent()
    sim = make_parent_sim()
    restore_replay(sim, parent["snap"])
    freeze(sim)
    dp, Rd = coupled_pose_from_ref(ref, S_COUPLED, "translation")
    apply_coupled_pose(sim, dp, Rd)
    tic = pack_canonical_ic(sim)
    trans = run_primitive("trans_same_seq", sch, tic, shots=True)
    payload["translation_only_same_seq"] = {k: v for k, v in trans.items() if k != "log"}
    (AUTH_DIR / "primitives_summary.json").write_text(json.dumps(payload, indent=2, default=_json), encoding="utf-8")
    return payload


def ab_compare() -> dict:
    ic = load_canonical_ic(IC_NPZ)
    z = make_parent_sim()
    restore_canonical_ic(z, ic)
    a = dump_branch_state(z)
    r = make_parent_sim()
    restore_canonical_ic(r, ic)
    b = dump_branch_state(r)
    d = compare_branch(a, b)
    d["same_mass"] = a["mass"] == b["mass"]
    d["same_pair_mu"] = a["pair_mu_object"] == b["pair_mu_object"]
    d["same_obj_contype"] = a["obj_contype"] == b["obj_contype"]
    d["same_table_contype"] = a["table_contype"] == b["table_contype"]
    (AUTH_DIR / "ab_equivalence.json").write_text(json.dumps(d, indent=2, default=_json), encoding="utf-8")
    print(json.dumps(d, indent=2, default=_json))
    return d


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true")
    p.add_argument("--freeze", action="store_true")
    p.add_argument("--repeat", action="store_true")
    p.add_argument("--zero-story", action="store_true")
    p.add_argument("--primitives", action="store_true")
    p.add_argument("--ab", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    do_all = args.all or not any([args.freeze, args.repeat, args.zero_story, args.primitives, args.ab])
    if do_all or args.freeze:
        freeze_ic()
    if do_all or args.repeat:
        repeat_zero(5)
    if do_all or args.zero_story:
        zero_full_story()
    if do_all or args.primitives:
        primitives()
    if do_all or args.ab:
        ab_compare()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
