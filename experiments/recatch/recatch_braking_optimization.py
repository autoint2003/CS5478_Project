"""Brake and lift after the existing airborne recontact.

The chase until the brake trigger is the interface-C law that produced
C_down15_k8_corr1_open18. This file does not change that law, the official
controller, K48, or TABLE_DROP. Diagnostic labels are not SUCCESS_12S.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_generation.strategies.recatch_oracle import _bottom, _prepare

from envs.airborne_obs import relative_state
from envs.deterioration import body_twist
from data_generation.strategies.capture_geometry import capture_geometry
from envs.ballistic_impact import assert_noslip, make_sim
from training.full_3d_airborne_recatch_dataset import capture_nominal, capture_pads, drop_supports
from experiments.recatch.horizontal_grasp_offset_freefall_recatch import DT
from experiments.recatch.recatch_interface_comparison import (
    _advance,
    _capture_window,
    _grip,
    _law,
    _limits,
    _mapped_velocity,
    _render,
    _step,
)
from training.recovery_runtime import gains_from_cfg, stamp
from training.replay_core import TABLE_TOP

OUT = ROOT / "results" / "diagnostics" / "recatch_braking_optimization"
BASE_NPZ = ROOT / "results" / "diagnostics" / "recatch_interface_comparison" / "traces" / "C_down15_k8_corr1_open18.npz"
BASE_SPEC = {
    "interface": "C",
    "aim": "below",
    "aim_down": 0.015,
    "k": 8.0,
    "v_clip": 4.5,
    "corr_clip": 1.0,
    "open_tau": 18.0,
}




def _pad_force(sim) -> tuple[float, float]:
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    obj = int(sim.ids.object_geom)
    fl = fr = 0.0
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if obj not in (g1, g2):
            continue
        other = g2 if g1 == obj else g1
        body = int(sim.model.geom_bodyid[other])
        if body not in (left, right):
            continue
        wr = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, i, wr)
        if body == left:
            fl += abs(float(wr[0]))
        else:
            fr += abs(float(wr[0]))
    return fl, fr


def _trigger(mode, geo, both_s, free_s, armed):
    both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
    one = (int(geo["nL"]) > 0) ^ (int(geo["nR"]) > 0)
    if not (armed and free_s >= 0.040):
        return False
    if mode == "both":
        return both
    if mode == "both40":
        return both and both_s >= 0.040
    if mode == "uni":
        return one or both
    if mode == "enter":
        return _capture_window(geo, free_s, armed) or both
    raise ValueError(mode)


def _pinch_target(sim, rh, p_z):
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    pinch = rd @ np.asarray(rh, float)
    return np.array([po[0] - pinch[0], po[1] - pinch[1], p_z], float)


def run(sim, snap, nominal, pads, pad_mu, brake, horizon=3.2, interface="C", drop_table=True):
    from experiments.recatch.multi_disturbance_airborne_feasibility import restore

    restore(sim, snap, nominal, pads, pad_mu)
    sim.model.opt.timestep = DT
    if drop_table:
        drop_supports(sim)
        sim._watch_drop = False
    else:
        sim._watch_drop = True
        from envs.table_drop import restore_state

        restore_state(sim, None)
    if hasattr(sim, "park_ball"):
        sim.park_ball()
    lim = _limits(sim)
    t0 = float(sim.data.time)
    rel0 = relative_state(sim.model, sim.data, sim.ids)
    rh = np.array(rel0["p_rel_h"], float)
    aim_down = float(BASE_SPEC["aim_down"])
    phase = "chase"
    free_s = 0.0
    both_s = 0.0
    armed = False
    seen_free = False
    v_cmd_z = None
    p_z = None
    brake_info = None
    stable_s = 0.0
    rows = []
    crossed = False
    step_i = 0
    held = None
    anchor = None
    p_state = np.array(sim.data.xpos[sim.ids.hand_body], float).copy()
    limit = int(horizon / DT)
    while step_i < limit:
        geo = capture_geometry(sim)
        decide = interface == "C" or step_i % 10 == 0
        if not decide and held is not None:
            phase, p_star, v_star, tau = held
        else:
            if phase in ("brake", "lift", "stable") and brake is not None:
                vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
                both = int(geo["nL"]) > 0 and int(geo["nR"]) > 0
                accel = float(brake["accel"])
                cmd_dt = DT if interface == "C" else 0.02
                if phase == "brake":
                    v_cmd_z = min(0.0, float(v_cmd_z) + accel * cmd_dt)
                    if both and v_cmd_z >= -0.02 and float(vo[2]) > -0.15:
                        phase = "lift"
                elif phase == "lift":
                    v_cmd_z = min(float(brake["lift"]), float(v_cmd_z) + accel * cmd_dt)
                    risen = float(sim.data.xpos[sim.ids.object_body][2]) >= float(brake_info["zo"]) + 0.03
                    if both and float(vo[2]) > -0.02 and risen:
                        phase = "stable"
                        stable_s = 0.0
                if phase == "stable":
                    v_cmd_z = 0.0
                p_z = float(p_z) + float(v_cmd_z) * cmd_dt
                p_star = _pinch_target(sim, rh, p_z)
                v_star = np.array([float(vo[0]), float(vo[1]), float(v_cmd_z)], float)
                tau = float(brake["grip"])
                if phase == "stable" and (not both or not geo["between"]):
                    phase = "lost"
            else:
                phase, free_s, both_s, armed = _advance(phase, geo, free_s, both_s, armed, DT if interface == "C" else 0.02)
                if free_s >= 0.040:
                    seen_free = True
                if brake is not None and _trigger(brake["trigger"], geo, both_s, free_s, armed):
                    vh = np.array(body_twist(sim.model, sim.data, sim.ids.hand_body)[0], float)
                    vo = np.array(body_twist(sim.model, sim.data, sim.ids.object_body)[0], float)
                    v_cmd_z = float(min(vh[2], 0.0))
                    p_z = float(sim.data.xpos[sim.ids.hand_body][2])
                    bottom = _bottom(sim)
                    voz = float(vo[2])
                    accel = float(brake["accel"])
                    t_pred = 0.0 if voz >= 0.0 else max(0.0, -voz / max(accel, 1e-6))
                    dist_pred = 0.0 if voz >= 0.0 else float(voz * voz / (2.0 * max(accel, 1e-6)))
                    brake_info = {
                        "t": float(sim.data.time) - t0,
                        "zo": float(sim.data.xpos[sim.ids.object_body][2]),
                        "bottom": bottom,
                        "voz": voz,
                        "vhz": float(vh[2]),
                        "t_pred": round(t_pred, 4),
                        "dist_pred": round(dist_pred, 4),
                        "bottom_pred": round(bottom - dist_pred, 4),
                        "feasible_if_object_follows": bool(bottom - dist_pred >= TABLE_TOP),
                    }
                    phase = "brake"
                    p_star = _pinch_target(sim, rh, p_z)
                    v_star = np.array([float(vo[0]), float(vo[1]), float(v_cmd_z)], float)
                    tau = float(brake["grip"])
                else:
                    hold = phase == "hold"
                    p_star, v_star, _err = _law(
                        sim, rh, aim_down, BASE_SPEC["k"], BASE_SPEC["v_clip"], hold, BASE_SPEC["corr_clip"],
                    )
                    tau, phase = _grip(phase, geo, BASE_SPEC["open_tau"])
            held = (phase, p_star, v_star, tau)
            if interface == "A":
                anchor = np.array(sim.data.xpos[sim.ids.hand_body], float)
        if interface == "C":
            ctrl = _step(sim, p_star, v_star, tau)
            logged_p = p_star
            logged_v = v_star
            logged_tau = tau
        else:
            _action, v_apply, tau_apply = _mapped_velocity(sim, v_star, tau)
            if interface == "A":
                logged_p = anchor
                ctrl = _step(sim, anchor, v_apply, tau_apply)
            else:
                p_state = p_state + v_apply * DT
                logged_p = p_state
                ctrl = _step(sim, p_state, v_apply, tau_apply)
            logged_v = v_apply
            logged_tau = tau_apply
        row = _row(sim, t0, phase, logged_tau, logged_v, logged_p, ctrl, lim)
        step_i += 1
        rows.append(row)
        if row["bottom"] < TABLE_TOP:
            crossed = True
            if phase in ("brake", "lift", "stable", "lost"):
                break
        if phase == "stable":
            stable_s += DT
            if stable_s >= 1.0:
                break
        if phase == "lost":
            break
        if row["zo"] < 0.05 and phase == "chase":
            break
    return _judge(rows, brake, brake_info, crossed)


def _row(sim, t0, phase, tau, v_cmd, p_des, ctrl, lim):
    geo = capture_geometry(sim)
    rel = relative_state(sim.model, sim.data, sim.ids)
    fl, fr = _pad_force(sim)
    sat = np.abs(np.asarray(ctrl[:7], float)) >= 0.95 * lim
    ph = np.array(rel["p_hand_w"], float)
    return {
        "t": float(sim.data.time) - t0,
        "phase": phase,
        "gap": float(geo["inner_gap_m"]),
        "between": int(bool(geo["between"])),
        "nL": int(geo["nL"]),
        "nR": int(geo["nR"]),
        "cx": float(geo["center_hand_m"][0]),
        "cz": float(geo["center_hand_m"][2]),
        "vh": np.array(rel["v_hand_w"], float),
        "vo": np.array(rel["v_obj_w"], float),
        "vrel_h": np.array(rel["v_rel_h"], float),
        "fn": np.array([fl, fr], float),
        "tau": float(tau),
        "v_cmd": np.asarray(v_cmd, float).copy(),
        "p_des": np.asarray(p_des, float).copy(),
        "p_hand": ph,
        "p_err": float(np.linalg.norm(np.asarray(p_des, float) - ph)),
        "zo": float(rel["p_obj_w"][2]),
        "xy": [float(rel["p_obj_w"][0]), float(rel["p_obj_w"][1])],
        "bottom": _bottom(sim),
        "sat": int(np.sum(sat)),
        "tau_j": np.asarray(ctrl[:7], float).copy(),
        "qpos": np.array(sim.data.qpos, float).copy(),
    }


def _judge(rows, brake, info, crossed):
    t = np.array([r["t"] for r in rows])
    nL = np.array([r["nL"] for r in rows])
    nR = np.array([r["nR"] for r in rows])
    both = (nL > 0) & (nR > 0)
    off = (nL == 0) & (nR == 0)
    voz = np.array([r["vo"][2] for r in rows])
    bottom = np.array([r["bottom"] for r in rows])
    zo = np.array([r["zo"] for r in rows])
    re_idx = None
    seen_off = False
    for i, flag in enumerate(off):
        if flag:
            seen_off = True
        elif seen_off and both[i] and re_idx is None:
            re_idx = i
    labels = []
    if re_idx is not None:
        labels.append("RECONTACT")
    braked = False
    lifted = False
    if re_idx is not None:
        after = slice(re_idx, None)
        if np.any(voz[after] > -0.05) and float(np.min(bottom[after])) >= TABLE_TOP:
            braked = True
            labels.append("BRAKED")
        zmin_i = re_idx + int(np.argmin(zo[after]))
        if float(np.max(zo[zmin_i:])) >= float(zo[zmin_i]) + 0.03 and float(np.min(bottom)) >= TABLE_TOP:
            lifted = True
            labels.append("LIFTED")
    stable_rows = [r for r in rows if r["phase"] == "stable"]
    stable = False
    if len(stable_rows) >= int(1.0 / DT) - 2:
        window = stable_rows[: int(1.0 / DT)]
        frac = float(np.mean([r["nL"] > 0 and r["nR"] > 0 for r in window]))
        end = window[-1]
        stable = bool(
            frac >= 0.95
            and end["bottom"] >= TABLE_TOP + 0.01
            and end["vo"][2] > -0.05
            and end["vh"][2] > -0.05
            and float(np.min([r["bottom"] for r in rows])) >= TABLE_TOP
            and braked
            and lifted
        )
    if stable:
        labels.append("STABLE_AIRBORNE_RECATCH")
    if crossed or float(np.min(bottom)) < TABLE_TOP:
        labels.append("CROSSED_TABLE")
    if not labels:
        labels.append("NO_RECONTACT" if re_idx is None else "CONTACT_ONLY")
    vhz = np.array([r["vh"][2] for r in rows])
    tau = np.stack([r["tau_j"] for r in rows])
    return {
        "brake": brake,
        "labels": labels,
        "stable": stable,
        "braked": braked,
        "lifted": lifted,
        "recontact_t": None if re_idx is None else round(float(t[re_idx]), 4),
        "min_bottom_m": round(float(np.min(bottom)), 4),
        "min_zo_m": round(float(np.min(zo)), 4),
        "end_zo_m": round(float(zo[-1]), 4),
        "end_bottom_m": round(float(bottom[-1]), 4),
        "end_voz": round(float(voz[-1]), 3),
        "end_vhz": round(float(vhz[-1]), 3),
        "end_n": [int(nL[-1]), int(nR[-1])],
        "max_gap_m": round(float(np.max([r["gap"] for r in rows])), 4),
        "sat_fraction": round(float(np.mean([r["sat"] > 0 for r in rows])), 3),
        "peak_tau_j2": round(float(np.max(np.abs(tau[:, 1]))), 2),
        "peak_tau_j4": round(float(np.max(np.abs(tau[:, 3]))), 2),
        "brake_info": info,
        "table_top_m": TABLE_TOP,
        "rows": rows,
    }


def _save(name, result):
    rows = result["rows"]
    path = OUT / "traces" / f"{name}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        t=np.array([r["t"] for r in rows], np.float32),
        phase=np.array([r["phase"] for r in rows]),
        gap=np.array([r["gap"] for r in rows], np.float32),
        between=np.array([r["between"] for r in rows], np.int8),
        nL=np.array([r["nL"] for r in rows], np.int16),
        nR=np.array([r["nR"] for r in rows], np.int16),
        cx=np.array([r["cx"] for r in rows], np.float32),
        cz=np.array([r["cz"] for r in rows], np.float32),
        vh=np.stack([r["vh"] for r in rows]).astype(np.float32),
        vo=np.stack([r["vo"] for r in rows]).astype(np.float32),
        vrel_h=np.stack([r["vrel_h"] for r in rows]).astype(np.float32),
        fn=np.stack([r["fn"] for r in rows]).astype(np.float32),
        tau=np.array([r["tau"] for r in rows], np.float32),
        v_cmd=np.stack([r["v_cmd"] for r in rows]).astype(np.float32),
        p_des=np.stack([r["p_des"] for r in rows]).astype(np.float32),
        p_hand=np.stack([r["p_hand"] for r in rows]).astype(np.float32),
        zo=np.array([r["zo"] for r in rows], np.float32),
        bottom=np.array([r["bottom"] for r in rows], np.float32),
        sat=np.array([r["sat"] for r in rows], np.int8),
        tau_j=np.stack([r["tau_j"] for r in rows]).astype(np.float32),
        qpos=np.array([r["qpos"] for r in rows]).astype(np.float32),
    )
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _plot(baseline, best, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    for result, name in ((baseline, "baseline follow"), (best, "best brake")):
        if result is None:
            continue
        rows = result["rows"]
        t = np.array([r["t"] for r in rows])
        axes[0].plot(t, [r["vh"][2] for r in rows], label=f"{name} hand")
        axes[0].plot(t, [r["vo"][2] for r in rows], linestyle="--", label=f"{name} cylinder")
        axes[1].plot(t, [r["vrel_h"][2] for r in rows], label=f"{name} finger-axis")
        axes[1].plot(t, [r["vrel_h"][0] for r in rows], linestyle="--", label=f"{name} pad-length")
        axes[2].plot(t, [r["bottom"] for r in rows], label=f"{name} bottom")
        axes[3].plot(t, [r["gap"] for r in rows], label=f"{name} gap")
        axes[3].plot(t, np.max(np.stack([r["tau_j"] for r in rows]), axis=1) / 87.0, label=f"{name} peak tau/87")
    axes[2].axhline(TABLE_TOP, color="k", linewidth=0.8, label="table top 0.40 m")
    axes[0].set_ylabel("world vz (m/s)")
    axes[1].set_ylabel("hand-frame slip (m/s)")
    axes[2].set_ylabel("cylinder bottom (m)")
    axes[3].set_ylabel("gap (m) / torque")
    axes[3].set_xlabel("time (s)")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def _public(result, trace):
    out = {k: v for k, v in result.items() if k not in ("rows", "branch")}
    out["trace"] = trace
    return out


def _stage(result):
    if result["stable"]:
        return "STABLE_AIRBORNE_RECATCH"
    if "CROSSED_TABLE" in result["labels"] and "RECONTACT" in result["labels"]:
        return "CROSSED_AFTER_RECONTACT"
    if result["lifted"]:
        return "LIFTED"
    if result["braked"]:
        return "BRAKED"
    if "RECONTACT" in result["labels"]:
        return "RECONTACT_ONLY"
    return "NO_RECONTACT"


def _grid():
    specs = []
    for trigger in ("both", "both40", "uni", "enter"):
        for accel in (6.0, 12.0, 20.0):
            for grip in (-18.0, -12.0):
                specs.append({
                    "trigger": trigger,
                    "accel": accel,
                    "grip": grip,
                    "lift": 0.20,
                })
    return specs




def _replay_error(rows):
    saved = np.load(BASE_NPZ)
    err = []
    j = 0
    for i, t in enumerate(saved["t"]):
        while j + 1 < len(rows) and abs(rows[j + 1]["t"] - float(t)) < abs(rows[j]["t"] - float(t)):
            j += 1
        if abs(rows[j]["t"] - float(t)) > 0.003:
            continue
        n = min(saved["qpos"].shape[1], rows[j]["qpos"].shape[0])
        err.append(float(np.max(np.abs(saved["qpos"][i, :n] - rows[j]["qpos"][:n]))))
    keys = list(saved.files)
    return {
        "npz_fields": keys,
        "has_qvel": "qvel" in keys,
        "has_ctrl": "ctrl" in keys,
        "has_fsm": "fsm" in keys,
        "matched_samples": len(err),
        "max_qpos_abs_error": None if not err else max(err),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)
    (OUT / "videos").mkdir(exist_ok=True)
    sim, snap, nominal, pads, pad_mu = _prepare()
    gid = int(sim.ids.object_geom)
    print(
        "GEOM",
        "radius", float(sim.model.geom_size[gid][0]),
        "half", float(sim.model.geom_size[gid][1]),
        "table", TABLE_TOP,
        flush=True,
    )
    baseline = run(sim, snap, nominal, pads, pad_mu, None, horizon=1.4)
    replay = _replay_error(baseline["rows"])
    print("REPLAY", json.dumps(replay), "labels", baseline["labels"], "min_bottom", baseline["min_bottom_m"], flush=True)
    base_trace = _save("baseline_follow", baseline)
    results = []
    best = None
    for spec in _grid():
        result = run(sim, snap, nominal, pads, pad_mu, spec)
        name = f"{spec['trigger']}_a{spec['accel']:.0f}_g{abs(spec['grip']):.0f}"
        pub = _public(result, None)
        pub["name"] = name
        results.append(pub)
        rank = (
            bool(result["stable"]),
            bool(result["lifted"]),
            bool(result["braked"]),
            result["min_bottom_m"],
        )
        print("CASE", name, result["labels"], "bottom", result["min_bottom_m"], "voz", result["end_voz"], "info", result["brake_info"], flush=True)
        if best is None or rank > best[0]:
            best = (rank, name, result)
        if result["stable"]:
            break
    assert best is not None
    best_name, best_result = best[1], best[2]
    if not best_result["stable"]:
        # One tighter pass around the highest bottom that still recontacted.
        seed = best_result["brake"]
        extra = []
        for accel in (seed["accel"] - 4.0, seed["accel"] + 8.0, 30.0):
            if accel < 2.0:
                continue
            extra.append({**seed, "accel": float(accel)})
        for spec in extra:
            result = run(sim, snap, nominal, pads, pad_mu, spec)
            name = f"refine_{spec['trigger']}_a{spec['accel']:.0f}_g{abs(spec['grip']):.0f}"
            pub = _public(result, None)
            pub["name"] = name
            results.append(pub)
            rank = (bool(result["stable"]), bool(result["lifted"]), bool(result["braked"]), result["min_bottom_m"])
            print("REFINE", name, result["labels"], "bottom", result["min_bottom_m"], flush=True)
            if rank > best[0]:
                best = (rank, name, result)
            if result["stable"]:
                break
        best_name, best_result = best[1], best[2]
    best_trace = _save(best_name, best_result)
    for row in results:
        if row["name"] == best_name:
            row["trace"] = best_trace
    _plot(baseline, best_result, OUT / "brake_curves.png")
    from experiments.recatch.recatch_interface_comparison import _render as render

    render(sim, baseline, OUT / "videos" / "baseline_follow.mp4", "baseline")
    render(sim, best_result, OUT / "videos" / f"{best_name}.mp4", best_name)
    summary = {
        "replay": replay,
        "baseline": _public(baseline, base_trace),
        "best": best_name,
        "candidates": results,
    }
    (OUT / "candidates.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("BEST", best_name, best_result["labels"], "bottom", best_result["min_bottom_m"], flush=True)


def migrate():
    """Replay the winning brake on interfaces C, B, and A from the same seated IC."""
    sim, snap, nominal, pads, pad_mu = _prepare()
    spec = {"trigger": "both", "accel": 6.0, "grip": -18.0, "lift": 0.20}
    out = []
    for interface in ("C", "B", "A"):
        result = run(sim, snap, nominal, pads, pad_mu, spec, interface=interface)
        name = f"win_{interface}"
        trace = _save(name, result)
        pub = _public(result, trace)
        pub["interface"] = interface
        pub["stage"] = _stage(result)
        out.append(pub)
        print("MIG", interface, pub["stage"], "bottom", pub["min_bottom_m"], "re", pub["recontact_t"], "endz", pub["end_zo_m"], flush=True)
    later = run(sim, snap, nominal, pads, pad_mu, {"trigger": "both40", "accel": 6.0, "grip": -18.0, "lift": 0.20})
    pub = _public(later, _save("both40_a6_g18", later))
    pub["interface"] = "C"
    pub["stage"] = _stage(later)
    out.append(pub)
    print("MIG both40", pub["stage"], "bottom", pub["min_bottom_m"], "info", later["brake_info"], flush=True)
    (OUT / "interfaces.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        migrate()
    else:
        main()
