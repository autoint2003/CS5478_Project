"""Scripted sanity checks for observable_tactile reward. No SAC. No eval npz load."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.residual import map_recovery4d
from controllers.rule_based_recovery import RuleBasedRecovery
from envs.config_util import load_yaml, merge_sim_config
from envs.grasp_sim import GraspSim
from envs.observable_reward import T_PROGRESS_PROVENANCE, T_PROGRESS_S
from envs.physical_recovery import fail_kind, physical_pack, recovered
from envs.recovery_env import RecoveryEnv
from training.offset_construct import apply_rel_pose, lift_to_airborne, settle_hold
from training.replay_core import (
    MASS,
    MU,
    disable_object_table,
    freeze,
    params_from_frozen,
)
from controllers.jacobian_controller import gains_from_cfg
from training.replay_core import tick_vw

OUT = ROOT / "results" / "diagnostics" / "observable_reward_design"
FIG = OUT / "figures"
EVAL_OFF = ROOT / "results" / "eval_sets" / "airborne_offset_eval.npz"
EVAL_IMP = ROOT / "results" / "eval_sets" / "impact_severity_four.npz"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_base_ic(cfg, e_x=0.006):
    sim = GraspSim(cfg)
    sim.reset(MASS, MU, np.zeros(3))
    if not lift_to_airborne(sim, cfg):
        raise RuntimeError("lift failed")
    disable_object_table(sim)
    freeze(sim)
    apply_rel_pose(sim, e_x)
    settle_hold(sim, cfg, 0.25, -18.0)
    snap = sim.snapshot()
    snap["e_x"] = float(physical_pack(sim)["e_x"])
    return snap


def a_slip():
    return 2.0 * ((-2.0) - (-1.0)) / ((-18.0) - (-1.0)) - 1.0


def policy_action(name, env, ctrl, t_s):
    if name == "ZERO":
        return np.array([0.0, 0.0, 0.0, 1.0], np.float32)
    if name == "WRIST":
        return np.array([0.30, 0.0, 0.0, 1.0], np.float32)
    if name == "SLIP":
        return np.array([0.0, 0.0, 0.0, a_slip()], np.float32)
    if name == "SLIP_BRAKE":
        if t_s < 0.20:
            return np.array([0.0, 0.0, 0.0, a_slip()], np.float32)
        return np.array([0.0, 0.0, 0.0, 1.0], np.float32)
    if name == "OPEN_SHORT":
        if t_s < 0.08:
            return np.array([0.0, 0.0, 0.0, -1.0], np.float32)
        return np.array([0.0, 0.0, 0.0, 1.0], np.float32)
    if name == "CONTACT_LOSS":
        return np.array([0.0, 0.0, 0.0, -1.0], np.float32)
    if name == "HX_OSC":
        s = 1.0 if int(t_s / 0.10) % 2 == 0 else -1.0
        return np.array([0.0, 0.5 * s, 0.0, 1.0], np.float32)
    if name == "RULE":
        o = physical_pack(env.sim)
        dt = env.sim.dt_policy
        cmd = ctrl.step(o, dt)
        if cmd["freeze_p"]:
            freeze(env.sim)
        if cmd["r_des"] is not None:
            env.sim.fsm.r_des = np.asarray(cmd["r_des"], float).copy()
        # invert map: RULE outputs v_world/tau; env.step wants recovery4d a
        rh = np.array(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
        v_h = rh.T @ np.asarray(cmd["v_world"], float)
        # w is 0 in audit; RULE uses r_des not w_world in env.step path
        from controllers.residual import RECOVERY4D_V_HX_MAX, RECOVERY4D_V_Z_MAX, RECOVERY4D_TAU_OPEN, RECOVERY4D_TAU_SECURE
        a1 = float(np.clip(v_h[0] / RECOVERY4D_V_HX_MAX, -1, 1))
        a2 = float(np.clip(cmd["v_world"][2] / RECOVERY4D_V_Z_MAX, -1, 1))
        tau = float(cmd["tau"])
        a3 = 2.0 * (tau - RECOVERY4D_TAU_OPEN) / (RECOVERY4D_TAU_SECURE - RECOVERY4D_TAU_OPEN) - 1.0
        a3 = float(np.clip(a3, -1, 1))
        # wrist via r_des already applied; a0=0 this step (orientation integrated outside)
        return np.array([0.0, a1, a2, a3], np.float32)
    raise KeyError(name)


def rollout(cfg, snap, name, reward_mode, max_s=2.0):
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d", reward_mode=reward_mode)
    env.reset(options={"snapshot": snap})
    params, _ = params_from_frozen()
    ctrl = RuleBasedRecovery(params)
    ctrl.reset(np.array(env.sim.fsm.r_des, float).reshape(3, 3))
    dt = env.sim.dt_policy
    n = int(round(max_s / dt))
    rows = []
    ret = 0.0
    prog_acc = 0.0
    for i in range(n):
        a = policy_action(name, env, ctrl, i * dt)
        obs, r, term, trunc, info = env.step(a)
        ret += float(r)
        prog_acc += float(info.get("r_progress", 0.0))
        ehat = info["e_hat_x"]
        rows.append(
            {
                "t": (i + 1) * dt,
                "r": float(r),
                "ret": ret,
                "r_progress": float(info["r_progress"]),
                "prog_acc": prog_acc,
                "e_x": float(info["e_x"]),
                "e_hat": float(ehat) if info["estimate_valid"] else float("nan"),
                "valid": float(info["estimate_valid"]),
                "bilat": float(info["bilateral_valid"]),
                "success_obs": float(info["success_obs"]),
                "failure_obs": float(info["failure_obs"]),
                "success_GT": float(info["success_GT"]),
                "failure_GT": float(info["failure_GT"]),
                "recovered_GT_now": float(info.get("recovered_GT_now", 0)),
                "term": info["term"],
            }
        )
        if term or trunc:
            break
    last = rows[-1]
    return {
        "policy": name,
        "reward_mode": reward_mode,
        "n": len(rows),
        "return": ret,
        "progress_return": prog_acc,
        "final_e_x_mm": 1e3 * last["e_x"],
        "final_e_hat_mm": 1e3 * last["e_hat"] if np.isfinite(last["e_hat"]) else None,
        "success_obs": bool(last["success_obs"]),
        "success_GT": bool(last["success_GT"]),
        "failure_obs": bool(last["failure_obs"]),
        "failure_GT": bool(last["failure_GT"]),
        "term": last["term"],
        "bilat_frac": float(np.mean([r["bilat"] for r in rows])),
        "valid_frac": float(np.mean([r["valid"] for r in rows])),
        "rows": rows,
    }


def confusion(events, obs_key, gt_key):
    tp = fp = tn = fn = 0
    for e in events:
        o = bool(e[obs_key])
        g = bool(e[gt_key])
        if o and g:
            tp += 1
        elif o and not g:
            fp += 1
        elif (not o) and (not g):
            tn += 1
        else:
            fn += 1
    n = max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": prec,
        "recall": rec,
        "false_success_rate": fp / n,
        "false_failure_rate": fn / n,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    h0, h1 = _sha(EVAL_OFF), _sha(EVAL_IMP)
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    snap = make_base_ic(cfg, 0.006)
    policies = ["ZERO", "RULE", "WRIST", "SLIP", "SLIP_BRAKE", "OPEN_SHORT", "CONTACT_LOSS", "HX_OSC"]
    table = []
    all_rows = []
    for name in policies:
        print("roll", name)
        obs = rollout(cfg, snap, name, "observable_tactile", 2.0)
        ora = rollout(cfg, snap, name, "oracle", 2.0)
        rec = {
            "policy": name,
            "obs_return": obs["return"],
            "oracle_return": ora["return"],
            "obs_progress": obs["progress_return"],
            "final_e_x_mm_obs": obs["final_e_x_mm"],
            "final_e_hat_mm_obs": obs["final_e_hat_mm"],
            "success_obs": obs["success_obs"],
            "success_GT_obs_env": obs["success_GT"],
            "failure_obs": obs["failure_obs"],
            "failure_GT_obs_env": obs["failure_GT"],
            "success_GT_oracle_env": ora["success_GT"],
            "failure_GT_oracle_env": ora["failure_GT"],
            "term_obs": obs["term"],
            "term_oracle": ora["term"],
            "bilat_frac": obs["bilat_frac"],
            "valid_frac": obs["valid_frac"],
            "n_obs": obs["n"],
            "n_oracle": ora["n"],
        }
        table.append(rec)
        for row in obs["rows"]:
            row["policy"] = name
            all_rows.append(row)
        # plots per policy
        ts = [r["t"] for r in obs["rows"]]
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.plot(ts, [r["prog_acc"] for r in obs["rows"]], label="cum r_progress")
        e0 = abs(obs["rows"][0]["e_hat"]) if np.isfinite(obs["rows"][0]["e_hat"]) else np.nan
        pot = []
        for r in obs["rows"]:
            if np.isfinite(r["e_hat"]) and np.isfinite(e0):
                pot.append((e0 - abs(r["e_hat"])) / 0.0075)
            else:
                pot.append(np.nan)
        ax.plot(ts, pot, label="(|e0|-|e_hat|)/0.0075")
        ax.set_title(f"{name} progress vs potential")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"progress_{name}.png", dpi=130)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.2, 3.2))
        ax.plot(ts, [r["valid"] for r in obs["rows"]], label="estimate_valid")
        ax.plot(ts, [r["bilat"] for r in obs["rows"]], label="bilateral")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{name} validity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / f"valid_{name}.png", dpi=130)
        plt.close(fig)

    ep_conf_s = confusion(table, "success_obs", "success_GT_oracle_env")
    ep_conf_f = confusion(table, "failure_obs", "failure_GT_oracle_env")
    step_s = confusion(all_rows, "success_obs", "recovered_GT_now")

    open_ok = next(t for t in table if t["policy"] == "OPEN_SHORT")
    zero = next(t for t in table if t["policy"] == "ZERO")
    rule = next(t for t in table if t["policy"] == "RULE")
    closs = next(t for t in table if t["policy"] == "CONTACT_LOSS")
    osc = next(t for t in table if t["policy"] == "HX_OSC")

    exploits = {
        "A_ZERO_high_return": zero["obs_return"] >= rule["obs_return"],
        "B_squeeze_vs_recenter": zero["obs_return"],
        "C_open_avoids_progress_penalty": open_ok["obs_return"] > zero["obs_return"] + 1.0,
        "D_contact_loss_avoids_neg": closs["obs_return"] > zero["obs_return"],
        "E_osc_farms_progress": osc["obs_progress"] > 0.5,
        "H_obs_success_gt_fail": bool(rule["success_obs"] and not rule["success_GT_oracle_env"]),
        "open_short_premature_fail": open_ok["failure_obs"] and open_ok["term_obs"] == "failure_obs",
    }

    timing = {
        "physics_dt": 0.002,
        "n_substeps": 10,
        "policy_dt": 0.020,
        "reward_eval": "once per RecoveryEnv.step = policy_dt",
        "T_progress_s": T_PROGRESS_S,
        "n_prog_policy_steps": 5,
        "T_progress_provenance": T_PROGRESS_PROVENANCE,
        "option": "A: emit r_progress only when interval completes",
        "lambda_t_per": "policy step, not physics step",
    }
    if _sha(EVAL_OFF) != h0 or _sha(EVAL_IMP) != h1:
        raise RuntimeError("held-out eval changed")
    prov = {
        "used_final_eval_data": False,
        "eval_offset_sha256": h0,
        "eval_impact_sha256": h1,
        "ic_e_x_m": float(snap["e_x"]),
        "timing": timing,
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (OUT / "rollout_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (OUT / "confusion.json").write_text(
        json.dumps({"episode_success": ep_conf_s, "episode_failure": ep_conf_f, "step_success_vs_recovered_now": step_s}, indent=2),
        encoding="utf-8",
    )
    (OUT / "exploits.json").write_text(json.dumps(exploits, indent=2), encoding="utf-8")
    (OUT / "timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    names = [t["policy"] for t in table]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [t["obs_return"] for t in table], 0.4, label="observable")
    ax.bar(x + 0.2, [t["oracle_return"] for t in table], 0.4, label="oracle")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25)
    ax.set_ylabel("return")
    ax.legend()
    ax.set_title("Scripted returns (2 s or terminal)")
    fig.tight_layout()
    fig.savefig(FIG / "returns_bar.png", dpi=140)
    plt.close(fig)

    print(json.dumps(table, indent=2))
    print("exploits", exploits)
    print("conf S", ep_conf_s)
    print("conf F", ep_conf_f)


if __name__ == "__main__":
    main()
