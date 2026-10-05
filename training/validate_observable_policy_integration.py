"""Observable policy integration gate. No SAC. No held-out eval NPZ load."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.residual import (
    RECOVERY4D_TAU_OPEN,
    RECOVERY4D_TAU_SECURE,
    RECOVERY4D_V_HX_MAX,
    RECOVERY4D_V_Z_MAX,
)
from controllers.rule_based_recovery import RuleBasedRecovery
from envs.config_util import load_yaml, merge_sim_config
from envs.grasp_sim import GraspSim
from envs.observable_obs import OBS_DIM_OBSERVABLE, OBS_NAMES, OBS_SOURCES
from envs.observable_reward import T_PROGRESS_S
from envs.physical_recovery import physical_pack, recovered
from envs.recovery_env import RecoveryEnv
from training.offset_construct import apply_rel_pose, ic_ok, lift_to_airborne, settle_hold
from training.replay_core import (
    MASS,
    MU,
    disable_object_table,
    freeze,
    params_from_frozen,
)

OUT = ROOT / "results" / "diagnostics" / "observable_policy_integration"
EVAL_OFF = ROOT / "results" / "eval_sets" / "airborne_offset_eval.npz"
EVAL_IMP = ROOT / "results" / "eval_sets" / "impact_severity_four.npz"
SEED = 5478
N_IC = 48


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha_nl(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def make_ics(cfg, n=N_IC, seed=SEED):
    rng = np.random.default_rng(seed)
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    if not lift_to_airborne(sim, cfg):
        raise RuntimeError("lift failed")
    disable_object_table(sim)
    freeze(sim)
    base = sim.snapshot()
    snaps = []
    signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])
    for i in range(n):
        mag = float(rng.uniform(0.0035, 0.0075))
        e = float(signs[i] * mag)
        sim = GraspSim(cfg)
        sim.reset(MASS, MU, np.zeros(3))
        disable_object_table(sim)
        sim.load_snapshot(base)
        disable_object_table(sim)
        freeze(sim)
        ok = False
        for _ in range(8):
            apply_rel_pose(sim, e)
            settle_hold(sim, cfg, 0.25, -18.0)
            o = physical_pack(sim)
            if ic_ok(o, 0.0030, 0.0085) and np.sign(o["e_x"]) == np.sign(e):
                snap = sim.snapshot()
                snap["source"] = "policy_integration_sanity"
                snap["e_x"] = float(o["e_x"])
                snaps.append(snap)
                ok = True
                break
            freeze(sim)
            e = float(np.sign(e) * float(rng.uniform(0.0035, 0.0075)))
        if not ok:
            raise RuntimeError(f"IC {i} failed")
    return snaps


def a_slip():
    return 2.0 * ((-2.0) - (-1.0)) / ((-18.0) - (-1.0)) - 1.0


def rule_action(env, ctrl):
    o = physical_pack(env.sim)
    cmd = ctrl.step(o, env.sim.dt_policy)
    if cmd["freeze_p"]:
        freeze(env.sim)
    if cmd["r_des"] is not None:
        env.sim.fsm.r_des = np.asarray(cmd["r_des"], float).copy()
    rh = np.array(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
    v_h = rh.T @ np.asarray(cmd["v_world"], float)
    a1 = float(np.clip(v_h[0] / RECOVERY4D_V_HX_MAX, -1, 1))
    a2 = float(np.clip(cmd["v_world"][2] / RECOVERY4D_V_Z_MAX, -1, 1))
    tau = float(cmd["tau"])
    a3 = 2.0 * (tau - RECOVERY4D_TAU_OPEN) / (RECOVERY4D_TAU_SECURE - RECOVERY4D_TAU_OPEN) - 1.0
    return np.array([0.0, a1, a2, float(np.clip(a3, -1, 1))], np.float32)


def rollout(cfg, snap, policy, mode, max_s=4.0):
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d", reward_mode=mode)
    obs, info = env.reset(options={"snapshot": snap})
    assert obs.shape[0] == (27 if mode == "observable_tactile" else 24)
    params, _ = params_from_frozen()
    ctrl = RuleBasedRecovery(params)
    ctrl.reset(np.array(env.sim.fsm.r_des, float).reshape(3, 3))
    dt = env.sim.dt_policy
    n = int(round(max_s / dt))
    ret = 0.0
    valid_n = 0
    bilat_n = 0
    t_obs = None
    t_gt_hold = None
    t_gt_now = None
    obs_rows = []
    last_info = info
    finite = True
    for i in range(n):
        if policy == "ZERO":
            a = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
        elif policy == "OPEN_SHORT":
            a = np.array([0.0, 0.0, 0.0, -1.0 if i * dt < 0.08 else 1.0], np.float32)
        else:
            a = rule_action(env, ctrl)
        obs, r, term, trunc, info = env.step(a)
        if not np.all(np.isfinite(obs)) or not np.isfinite(r):
            finite = False
        ret += float(r)
        last_info = info
        valid_n += float(info["estimate_valid"])
        bilat_n += float(info.get("bilateral_valid", 0))
        t = (i + 1) * dt
        if t_obs is None and info["success_obs"]:
            t_obs = t
        if t_gt_now is None and info.get("recovered_GT_now"):
            t_gt_now = t
        if t_gt_hold is None and info.get("term_GT") == "recovered":
            t_gt_hold = t
        o = physical_pack(env.sim)
        obs_rows.append(obs.copy())
        if term or trunc:
            break
    nstep = max(len(obs_rows), 1)
    fp_detail = None
    if last_info["success_obs"] and not last_info.get("recovered_GT_now"):
        fp_detail = {
            "e_x_mm": 1e3 * float(last_info["e_x"]),
            "v_rel": float(last_info.get("v_rel", np.nan)),
            "w_rel": float(last_info.get("w_rel", np.nan)),
            "nL": last_info.get("nL"),
            "nR": last_info.get("nR"),
            "obj_z": last_info.get("obj_z"),
            "e_hat": last_info.get("e_hat_x"),
            "term": last_info.get("term"),
        }
    return {
        "init_e_x": float(snap["e_x"]),
        "final_e_x": float(last_info["e_x"]),
        "final_e_hat": last_info.get("e_hat_x") if last_info.get("estimate_valid") else None,
        "return": ret,
        "success_obs": bool(last_info["success_obs"]),
        "success_GT": bool(last_info.get("term_GT") == "recovered" or last_info.get("success_GT")),
        "failure_obs": bool(last_info["failure_obs"]),
        "failure_GT": bool(last_info.get("failure_GT")),
        "term": last_info.get("term"),
        "term_GT": last_info.get("term_GT"),
        "valid_frac": valid_n / nstep,
        "bilat_frac": bilat_n / nstep,
        "n": nstep,
        "t_success_obs": t_obs,
        "t_success_GT_hold": t_gt_hold,
        "t_recovered_GT_now": t_gt_now,
        "finite": finite,
        "obs_stack": np.stack(obs_rows) if obs_rows else np.zeros((1, obs.shape[0])),
        "fp_detail": fp_detail,
        "recovered_GT_now_final": bool(last_info.get("recovered_GT_now")),
    }


def conf(rows, ok, gk):
    tp = fp = tn = fn = 0
    fps = []
    for i, r in enumerate(rows):
        o, g = bool(r[ok]), bool(r[gk])
        if o and g:
            tp += 1
        elif o and not g:
            fp += 1
            fps.append(i)
        elif (not o) and (not g):
            tn += 1
        else:
            fn += 1
    n = max(tp + fp + tn + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / n,
        "fnr": fn / n,
        "fp_indices": fps,
    }


def pct(a, q):
    return float(np.quantile(a, q)) if len(a) else float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h0, h1 = _sha(EVAL_OFF), _sha(EVAL_IMP)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    print("build ICs", N_IC)
    snaps = make_ics(cfg)
    np.savez_compressed(OUT / "sanity_ics.npz", n=len(snaps), e_x=np.array([s["e_x"] for s in snaps]))
    zero_rows = []
    rule_rows = []
    stacks = []
    for i, snap in enumerate(snaps):
        print(f"ep {i} e={1e3*snap['e_x']:+.2f} mm")
        z = rollout(cfg, snap, "ZERO", "observable_tactile", 1.0)
        r = rollout(cfg, snap, "RULE", "observable_tactile", 4.0)
        zero_rows.append(z)
        rule_rows.append(r)
        stacks.append(z["obs_stack"])
        stacks.append(r["obs_stack"])
    # OPEN_SHORT on first IC
    op = rollout(cfg, snaps[0], "OPEN_SHORT", "observable_tactile", 1.0)

    all_obs = np.concatenate(stacks, axis=0)
    norm = []
    for d, name in enumerate(OBS_NAMES):
        col = all_obs[:, d]
        clip = float(np.mean((col <= -1.0 + 1e-8) | (col >= 1.0 - 1e-8)))
        norm.append(
            {
                "i": d,
                "name": name,
                "source": OBS_SOURCES[name],
                "min": float(col.min()),
                "max": float(col.max()),
                "p1": pct(col, 0.01),
                "p50": pct(col, 0.50),
                "p99": pct(col, 0.99),
                "clip_frac": clip,
            }
        )

    def pack(rows, policy):
        out = []
        for i, r in enumerate(rows):
            rec = {k: r[k] for k in r if k not in ("obs_stack",)}
            rec["i"] = i
            rec["policy"] = policy
            rec["sign"] = float(np.sign(r["init_e_x"]))
            rec["mag_mm"] = abs(1e3 * r["init_e_x"])
            out.append(rec)
        return out

    zpack = pack(zero_rows, "ZERO")
    rpack = pack(rule_rows, "RULE")
    cs = conf(rpack, "success_obs", "success_GT")
    cf = conf(rpack, "failure_obs", "failure_GT")
    # Dangerous FP: success_obs while not recovered_GT_now
    danger = []
    for i, r in enumerate(rule_rows):
        if r["success_obs"] and not r["recovered_GT_now_final"]:
            danger.append({"i": i, "init_e_x_mm": 1e3 * r["init_e_x"], "detail": r["fp_detail"], **{k: r[k] for k in ("final_e_x", "t_success_obs", "t_recovered_GT_now", "term_GT")}})

    dts = []
    for r in rule_rows:
        if r["t_success_obs"] is not None and r["t_success_GT_hold"] is not None:
            dts.append(r["t_success_obs"] - r["t_success_GT_hold"])
        elif r["t_success_obs"] is not None and r["t_success_GT_hold"] is None:
            dts.append(None)
    dts_f = [x for x in dts if x is not None]

    timing = {
        "n_both_hold": len(dts_f),
        "n_obs_without_gt_hold": sum(1 for x in dts if x is None),
        "dt_min": float(min(dts_f)) if dts_f else None,
        "dt_median": float(np.median(dts_f)) if dts_f else None,
        "dt_max": float(max(dts_f)) if dts_f else None,
        "note": "negative => obs earlier than GT hold complete",
    }

    by_sign = {}
    for sg, lab in ((1.0, "pos"), (-1.0, "neg")):
        sub = [r for r in rpack if r["sign"] == sg]
        by_sign[lab] = {
            "n": len(sub),
            "success_obs": sum(r["success_obs"] for r in sub),
            "success_GT": sum(r["success_GT"] for r in sub),
            "mean_return": float(np.mean([r["return"] for r in sub])),
            "mean_final_e_mm": float(np.mean([1e3 * r["final_e_x"] for r in sub])),
        }

    if _sha(EVAL_OFF) != h0 or _sha(EVAL_IMP) != h1:
        raise RuntimeError("eval npz changed")

    hashes = {
        "observable_obs.py": _sha_nl(ROOT / "envs" / "observable_obs.py"),
        "observable_reward.py": _sha_nl(ROOT / "envs" / "observable_reward.py"),
        "spatial_tactile.py": _sha_nl(ROOT / "sensors" / "spatial_tactile.py"),
        "ex_estimator.py": _sha_nl(ROOT / "sensors" / "ex_estimator.py"),
        "residual.py": _sha_nl(ROOT / "controllers" / "residual.py"),
        "recovery_env.py": _sha_nl(ROOT / "envs" / "recovery_env.py"),
        "physical_recovery.py": _sha_nl(ROOT / "envs" / "physical_recovery.py"),
    }

    blockers = []
    if any(not r["finite"] for r in rule_rows + zero_rows):
        blockers.append("NaN in obs or reward")
    if not op["finite"] or op["failure_obs"]:
        blockers.append("open-short premature failure or NaN")
    if danger:
        # inspect: if |e_x| > 3.5mm treat as blocker
        hard = [d for d in danger if abs(d["final_e_x"]) > 0.0035]
        if hard:
            blockers.append(f"false success with |e_x|>3.5mm n={len(hard)}")

    meta = {
        "used_final_eval_data": False,
        "eval_offset_sha256": h0,
        "eval_impact_sha256": h1,
        "seed": SEED,
        "n_ic": len(snaps),
        "obs_dim": OBS_DIM_OBSERVABLE,
        "obs_names": OBS_NAMES,
        "T_progress": T_PROGRESS_S,
        "open_short": {k: op[k] for k in op if k != "obs_stack"},
        "zero_mean_return": float(np.mean([r["return"] for r in zpack])),
        "rule_mean_return": float(np.mean([r["return"] for r in rpack])),
        "success_conf_RULE": cs,
        "failure_conf_RULE": cf,
        "dangerous_fp": danger,
        "timing": timing,
        "by_sign": by_sign,
        "hashes": hashes,
        "blockers": blockers,
        "open_short_failure_obs": op["failure_obs"],
        "open_short_valid_frac": op["valid_frac"],
    }
    (OUT / "provenance.json").write_text(json.dumps({k: meta[k] for k in ("used_final_eval_data", "eval_offset_sha256", "eval_impact_sha256", "seed", "n_ic")}, indent=2), encoding="utf-8")
    (OUT / "normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")
    (OUT / "rule_rollouts.json").write_text(json.dumps(rpack, indent=2), encoding="utf-8")
    (OUT / "zero_rollouts.json").write_text(json.dumps(zpack, indent=2), encoding="utf-8")
    (OUT / "gate.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print("conf S", cs)
    print("conf F", cf)
    print("danger", danger)
    print("timing", timing)
    print("ZERO ret", meta["zero_mean_return"], "RULE ret", meta["rule_mean_return"])
    print("blockers", blockers)
    print("open", op["failure_obs"], op["valid_frac"], op["finite"])


if __name__ == "__main__":
    main()
