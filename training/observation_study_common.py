"""Shared 24-case rollout for observation robustness and ablation.

Perturbations touch only the vector passed into the policy. The simulator
state, the controller, and the t = 12 continuation stay on the balanced path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import training.natural_loss_recatch_data_scaling as scale
import training.natural_loss_recatch_transformer_v3 as v3
from envs.airborne_obs import (
    SCALE_PREL,
    SCALE_ROT,
    SCALE_VREL,
    SCALE_WREL,
    observe_airborne,
    rotvec,
)
from envs.observable_obs import SCALE_PAD_V, ObservableObsState
from envs.physical_recovery import SCALE_EX, SCALE_F, physical_pack
from training.balanced_unified_recovery import (
    CKPT_39,
    EVAL_NATURAL,
    RAW,
    TRAIN_NATURAL,
    longest,
    mechanism_of,
)
from training.clean_natural_loss_recatch import distal_geoms
from training.demo_ballistic_impact import assert_noslip
from training.full_3d_airborne_recatch_dataset import (
    TRACK,
    begin,
    carry_then_release,
    drop_supports,
    grasp_settled,
    hold_step,
    limits_of,
    restore_to,
)
from training.phase_decoupled_recovery_pilot import world
from training.recovery_policy_pretraining_preparation import schedules
from training.recovery_runtime import T_TASK, continue_long, physical_of, restore_dyn, run_impact
from training.replay_core import park_impact_ball

DT = 0.02
SLOW_TAU = 0.20


def load_balanced(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, proto, saved = v3.load_airborne_policy(CKPT_39, device)
    return model, proto, saved, device


def build_cases():
    """Same 24 cases as balanced_unified_recovery.main, without training."""
    sim, _g, nominal, pads, pad_mu, _pl, _pr = world()
    assert_noslip(sim)
    seat = grasp_settled(sim, TRACK)
    if seat is None or not begin(sim, seat, nominal, pads, pad_mu, {"dx": 0, "dy": 0, "dz": 0, "pitch": 0}):
        raise RuntimeError("seat failed")
    for _ in range(40):
        hold_step(sim, TRACK)
    pinch = np.array(
        __import__("envs.airborne_obs", fromlist=["relative_state"]).relative_state(
            sim.model, sim.data, sim.ids
        )["p_rel_h"],
        float,
    )
    distal = distal_geoms(sim)
    q_ref, _margin = limits_of(sim)
    drop_supports(sim)
    base = v3.stamp(sim)
    eval_natural = []
    for spec in EVAL_NATURAL:
        rec, meta = scale.collect(sim, base, spec, pinch, distal, nominal, pads, pad_mu, "hold")
        if rec is None:
            raise RuntimeError(f"held-out natural window missing: {spec['tag']} {meta}")
        window = longest(rec["windows"])
        window["pool"] = "natural"
        eval_natural.append(window)

    sim_i, gains_i, nominal_i, pads_i, pad_mu_i, pads_l, pads_r = world()
    assert_noslip(sim_i)
    sched = schedules()
    snaps = {
        "zm6": run_impact(sim_i, gains_i, 6.5, -0.006, 0.0, pads_l, pads_r, nominal_i, None, None)["snap"],
        "zp6": run_impact(sim_i, gains_i, 6.5, 0.006, 0.0, pads_l, pads_r, nominal_i, None, None)["snap"],
        "lat": run_impact(sim_i, gains_i, 6.5, 0.0, 90.0, pads_l, pads_r, nominal_i, None, None)["snap"],
        "z3": run_impact(sim_i, gains_i, 6.5, -0.003, 0.0, pads_l, pads_r, nominal_i, None, None)["snap"],
    }

    def suffix(actions, index):
        info = restore_dyn(sim_i, snaps["zm6"], nominal_i, pads_i, pad_mu_i, None, None)
        park_impact_ball(sim_i)
        if not info["pose_ok"]:
            return None
        for i, act in enumerate(actions):
            if i == index:
                return v3.stamp(sim_i)
            v3.step_v3(sim_i, v3.v1_normalized_to_v3(np.asarray(act, np.float32)))
            if float(physical_pack(sim_i)["obj_z"]) < 0.44:
                return None
        return None

    def shift(delta):
        import copy
        j = int(sim_i.ids.object_jnt)
        adr = int(sim_i.model.jnt_qposadr[j])
        out = copy.deepcopy(snaps["zm6"])
        q = np.array(out["qpos"], float)
        q[adr:adr + 3] = q[adr:adr + 3] + np.asarray(delta, float)
        out["qpos"] = q
        return out

    cases = [
        {"id": "impact_zm6", "family": "impact", "scene": "impact", "snap": snaps["zm6"], "n_steps": 56},
        {"id": "impact_zp6", "family": "impact", "scene": "impact", "snap": snaps["zp6"], "n_steps": 56},
        {"id": "impact_phi90", "family": "impact", "scene": "impact", "snap": snaps["lat"], "n_steps": 56},
        {"id": "impact_z3", "family": "impact", "scene": "impact", "snap": snaps["z3"], "n_steps": 56},
    ]
    for index in (20, 36):
        snap_i = suffix(sched["gravity_inward"], index)
        if snap_i is None:
            raise RuntimeError(f"slide suffix {index} missing")
        cases.append({
            "id": f"slide_suffix_{index}", "family": "sliding", "scene": "impact",
            "snap": snap_i, "n_steps": max(12, 56 - index),
        })
    cases.append({"id": "slide_dy1", "family": "sliding", "scene": "impact", "snap": shift((0, 0.001, 0)), "n_steps": 56})
    cases.append({"id": "slide_dz1", "family": "sliding", "scene": "impact", "snap": shift((0, 0, 0.001)), "n_steps": 56})
    for index in (3, 6, 9, 12):
        snap_i = suffix(sched["wrist_reseat"], index)
        if snap_i is None:
            raise RuntimeError(f"wrist suffix {index} missing")
        cases.append({
            "id": f"wrist_suffix_{index}", "family": "wrist", "scene": "impact",
            "snap": snap_i, "n_steps": max(12, 56 - index),
        })
    for index in (8, 16, 24, 28):
        snap_i = suffix(sched["recatch_open20"], index)
        if snap_i is None:
            raise RuntimeError(f"recatch suffix {index} missing")
        cases.append({
            "id": f"recatch_suffix_{index}", "family": "recatch", "scene": "impact",
            "snap": snap_i, "n_steps": max(12, 56 - index),
        })
    for tag, speed, sign in (
        ("air_lat_pos", 0.30, 1.0),
        ("air_lat_neg", 0.30, -1.0),
        ("air_fast_pos", 0.45, 1.0),
        ("air_slow_neg", 0.20, -1.0),
    ):
        made = carry_then_release(sim, base, nominal, pads, pad_mu, q_ref, speed, sign, 0.08)
        if made is None:
            raise RuntimeError(f"carry missing {tag}")
        cases.append({"id": tag, "family": "airborne", "scene": "natural", "snap": made["snap"], "n_steps": 120})
    for w in eval_natural:
        cases.append({
            "id": "nl_" + w["tag"], "family": "natural", "scene": "natural",
            "snap": w["snap"], "n_steps": 120,
        })
    if len(cases) != 24:
        raise RuntimeError(f"expected 24 cases, built {len(cases)}")
    ctx = {
        "sim": sim,
        "nominal": nominal,
        "pads": pads,
        "pad_mu": pad_mu,
        "sim_i": sim_i,
        "nominal_i": nominal_i,
        "pads_i": pads_i,
        "pad_mu_i": pad_mu_i,
        "cases": cases,
        "eval_natural": eval_natural,
        "base": base,
        "pinch": pinch,
        "distal": distal,
        "q_ref": q_ref,
    }
    return ctx


def collect_train(ctx):
    """Regenerate the balanced training trajectories on an existing seated world."""
    sim, base = ctx["sim"], ctx["base"]
    nominal, pads, pad_mu = ctx["nominal"], ctx["pads"], ctx["pad_mu"]
    natural = []
    for spec in TRAIN_NATURAL:
        rec, meta = scale.collect(sim, base, spec, ctx["pinch"], ctx["distal"], nominal, pads, pad_mu, "train")
        if rec is None:
            raise RuntimeError(f"train natural missing {spec['tag']} {meta}")
        for w in rec["windows"]:
            w["pool"] = "natural"
            natural.append(w)
    sim_i = ctx["sim_i"]
    sched = schedules()
    zm6 = next(c["snap"] for c in ctx["cases"] if c["id"] == "impact_zm6")
    local = []
    for skill, pool in (
        ("wrist_reseat", "wrist"),
        ("gravity_inward", "sliding"),
        ("recatch_open20", "recatch"),
    ):
        tr = v3.record_local(sim_i, skill, sched[skill], zm6, ctx["nominal_i"], ctx["pads_i"], ctx["pad_mu_i"], None)
        if tr is None:
            raise RuntimeError(f"local skill failed {skill}")
        tr["pool"] = pool
        local.append(tr)
    return natural + local


def _skew(a):
    x, y, z = a
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], float)


def rv_to_R(rv):
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(rv, float) / th
    K = _skew(k)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def decode_obj(obs):
    return {
        "p": np.array(obs[27:30], float) * SCALE_PREL,
        "v": np.array(obs[30:33], float) * SCALE_VREL,
        "rv": np.array(obs[33:36], float) * SCALE_ROT,
        "w": np.array(obs[36:39], float) * SCALE_WREL,
    }


def encode_obj(obs, st):
    obs[27:30] = np.clip(st["p"] / SCALE_PREL, -1.0, 1.0)
    obs[30:33] = np.clip(st["v"] / SCALE_VREL, -1.0, 1.0)
    obs[33:36] = np.clip(st["rv"] / SCALE_ROT, -1.0, 1.0)
    obs[36:39] = np.clip(st["w"] / SCALE_WREL, -1.0, 1.0)


class SensorNoise:
    """Sensing error on the observation only. Empty spec returns the input unchanged."""

    def __init__(self, spec, seed):
        self.spec = spec
        self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.filt = None
        self.buffer = []
        self.hold = None
        self.slow_v = np.zeros(3)
        self.slow_w = np.zeros(3)
        self.e_prev = None
        self.e_prev_valid = False
        tau = spec.get("tau")
        self.alpha = None if not tau else DT / (float(tau) + DT)
        self.delay = int(round(float(spec.get("delay_s") or 0.0) / DT))
        a = float(np.exp(-DT / SLOW_TAU))
        self.ar = a
        self.ar_in = float(np.sqrt(max(0.0, 1.0 - a * a)))

    def _lowpass(self, st):
        if self.alpha is None:
            return {k: np.array(v, float) for k, v in st.items()}
        if self.filt is None:
            self.filt = {k: np.array(v, float) for k, v in st.items()}
            return {k: np.array(v, float) for k, v in self.filt.items()}
        a = self.alpha
        out = {}
        for key in ("p", "v", "w"):
            out[key] = self.filt[key] + a * (st[key] - self.filt[key])
        R0 = rv_to_R(self.filt["rv"])
        inc = rotvec(R0.T @ rv_to_R(st["rv"]))
        out["rv"] = rotvec(R0 @ rv_to_R(a * inc))
        self.filt = {k: np.array(v, float) for k, v in out.items()}
        return {k: np.array(v, float) for k, v in self.filt.items()}

    def _rotate(self, rv, sigma_deg):
        axis = self.rng.normal(size=3)
        n = float(np.linalg.norm(axis))
        if n < 1e-12:
            return rv
        ang = float(self.rng.normal(0.0, np.deg2rad(sigma_deg)))
        R = rv_to_R((axis / n) * ang) @ rv_to_R(rv)
        return rotvec(R)

    def apply(self, obs):
        spec = self.spec
        if not spec:
            self.t += 1
            return obs
        out = np.array(obs, np.float32, copy=True)
        st = decode_obj(out)
        st = self._lowpass(st)
        if spec.get("pos_sigma"):
            st["p"] = st["p"] + self.rng.normal(0.0, float(spec["pos_sigma"]), 3)
        if spec.get("pos_bias") is not None:
            st["p"] = st["p"] + np.asarray(spec["pos_bias"], float)
        if spec.get("ori_sigma_deg"):
            st["rv"] = self._rotate(st["rv"], float(spec["ori_sigma_deg"]))
        if spec.get("v_sigma"):
            st["v"] = st["v"] + self.rng.normal(0.0, float(spec["v_sigma"]), 3)
        if spec.get("v_slow"):
            self.slow_v = self.ar * self.slow_v + self.ar_in * float(spec["v_slow"]) * self.rng.normal(size=3)
            st["v"] = st["v"] + self.slow_v
        if spec.get("w_sigma"):
            st["w"] = st["w"] + self.rng.normal(0.0, float(spec["w_sigma"]), 3)
        if spec.get("w_slow"):
            self.slow_w = self.ar * self.slow_w + self.ar_in * float(spec["w_slow"]) * self.rng.normal(size=3)
            st["w"] = st["w"] + self.slow_w
        st = {k: np.array(v, float) for k, v in st.items()}
        self.buffer.append(st)
        shown = self.buffer[max(0, self.t - self.delay)]
        drop = spec.get("dropout")
        if drop is not None:
            start, n = int(drop["start"]), int(drop["n"])
            if self.t == start - 1:
                self.hold = {k: np.array(v, float) for k, v in shown.items()}
            if start <= self.t < start + n and self.hold is not None:
                age = (self.t - start + 1) * DT
                if drop["mode"] == "hold":
                    shown = self.hold
                else:
                    shown = {
                        "p": self.hold["p"] + self.hold["v"] * age,
                        "v": self.hold["v"].copy(),
                        "w": self.hold["w"].copy(),
                        "rv": rotvec(rv_to_R(self.hold["rv"]) @ rv_to_R(self.hold["w"] * age)),
                    }
        encode_obj(out, shown)
        self._tactile(out)
        self.t += 1
        return out

    def _tactile(self, out):
        spec = self.spec
        valid = float(out[1]) > 0.5
        e_phys = float(out[0]) * SCALE_EX
        if spec.get("cop_sigma"):
            sig = float(spec["cop_sigma"])
            uL = float(out[6]) * SCALE_EX + float(self.rng.normal(0.0, sig))
            vL = float(out[7]) * SCALE_PAD_V + float(self.rng.normal(0.0, sig))
            uR = float(out[8]) * SCALE_EX + float(self.rng.normal(0.0, sig))
            vR = float(out[9]) * SCALE_PAD_V + float(self.rng.normal(0.0, sig))
            out[6] = np.clip(uL / SCALE_EX, -1.0, 1.0)
            out[7] = np.clip(vL / SCALE_PAD_V, -1.0, 1.0)
            out[8] = np.clip(uR / SCALE_EX, -1.0, 1.0)
            out[9] = np.clip(vR / SCALE_PAD_V, -1.0, 1.0)
            parts = []
            if float(out[4]) > 0.5:
                parts.append(uL)
            if float(out[5]) > 0.5:
                parts.append(-uR)
            if parts:
                e_phys = float(np.mean(parts))
        if spec.get("ehat_sigma"):
            e_phys = e_phys + float(self.rng.normal(0.0, float(spec["ehat_sigma"])))
        if spec.get("cop_sigma") or spec.get("ehat_sigma"):
            out[0] = np.clip(e_phys / SCALE_EX, -1.0, 1.0)
            if valid and self.e_prev_valid:
                edot = (e_phys - self.e_prev) / DT
            else:
                edot = 0.0
            out[26] = np.clip(edot / 0.080, -1.0, 1.0)
            self.e_prev = e_phys
            self.e_prev_valid = valid
        if spec.get("fn_sigma"):
            for idx in (10, 11):
                fn = float(out[idx]) * SCALE_F + float(self.rng.normal(0.0, float(spec["fn_sigma"])))
                out[idx] = np.clip(fn / SCALE_F, -1.0, 1.0)


def eval_policy(ctx, model, case, sensor=None, columns=None):
    """Same success rule as eval_case. sensor and columns affect only the network input."""
    sim = ctx["sim_i"] if case["scene"] == "impact" else ctx["sim"]
    nominal = ctx["nominal_i"] if case["scene"] == "impact" else ctx["nominal"]
    pads = ctx["pads_i"] if case["scene"] == "impact" else ctx["pads"]
    pad_mu = ctx["pad_mu_i"] if case["scene"] == "impact" else ctx["pad_mu"]
    if case["scene"] == "impact":
        info = restore_dyn(sim, case["snap"], nominal, pads, pad_mu, None, None)
        park_impact_ball(sim)
    else:
        info = restore_to(sim, case["snap"], nominal, pads, pad_mu)
        drop_supports(sim)
    if not info.get("pose_ok", False):
        return {"t12": "RESTORE_FAIL", "mechanism": "none", "kind": "RESTORE_FAIL"}
    hist = ObservableObsState(v3.DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    rows = []
    model.eval()
    device = next(model.parameters()).device
    dropped = False
    with torch.no_grad():
        for _k in range(int(case["n_steps"])):
            obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            if sensor is not None:
                obs = sensor.apply(obs)
            if columns is not None:
                obs = np.asarray(obs, np.float32)[list(columns)]
            obs_seq.append(np.asarray(obs, np.float32))
            prev_seq.append(prev.copy())
            pred = model(
                torch.tensor(np.stack(obs_seq), device=device),
                torch.tensor(np.stack(prev_seq), device=device),
            )[2][-1].float().cpu().numpy()
            act = np.clip(pred, -1.0, 1.0).astype(np.float32)
            v3.step_v3(sim, act)
            o = physical_pack(sim)
            rows.append({"act": act, "nL": int(o["nL"]), "nR": int(o["nR"])})
            prev = act
            if float(o["obj_z"]) < 0.44:
                dropped = True
                break
    if dropped:
        t12, kind_y = "DROP", "DROP"
        end_n = [0, 0]
    else:
        rec = continue_long(sim, TRACK, T_TASK, "policy")
        y, kind_y = physical_of(rec["end"], rec["t_drop"], rec["t_escape"])
        t12 = "RETAINED_TO_12" if y == 1 else kind_y
        end_n = [int(rec["end"]["nL"]), int(rec["end"]["nR"])]
    mech = mechanism_of(rows, t12)
    acts = np.stack([r["act"] for r in rows]) if rows else np.zeros((1, 7), np.float32)
    return {
        "t12": t12,
        "kind": kind_y,
        "mechanism": mech,
        "end_n": end_n,
        "policy_steps": len(rows),
        "max_vz": float(np.max(acts[:, 2])),
        "max_abs_wy": float(np.max(np.abs(acts[:, 4]))),
        "mean_abs_vx": float(np.mean(np.abs(acts[:, 0]))),
        "mean_abs_vy": float(np.mean(np.abs(acts[:, 1]))),
        "min_grip": float(np.min(acts[:, 6])),
        "mean_vz": float(np.mean(acts[:, 2])),
    }


def run_suite(ctx, model, spec, seed, columns=None):
    rows = []
    for case in ctx["cases"]:
        # A fresh sensor timeline per case. Dropout and delay are within the case.
        sensor = SensorNoise(spec, seed + 17 * (sum(ord(ch) for ch in case["id"])))
        got = eval_policy(ctx, model, case, sensor if spec else None, columns)
        rows.append({"id": case["id"], "family": case["family"], **got})
    ok = sum(r["t12"] == "RETAINED_TO_12" for r in rows)
    return {"seed": int(seed), "ok": int(ok), "n": len(rows), "rows": rows}


def family_counts(rows):
    out = {}
    for fam in ("impact", "sliding", "wrist", "recatch", "airborne", "natural"):
        sub = [r for r in rows if r["family"] == fam]
        out[fam] = sum(r["t12"] == "RETAINED_TO_12" for r in sub)
    return out


def clean_check(ctx, model):
    """One unmodified pass. Must match the published summary case by case."""
    import json
    published = {
        r["id"]: r["balanced"]["t12"]
        for r in json.loads((RAW / "summary.json").read_text(encoding="utf-8"))["rows"]
    }
    got = run_suite(ctx, model, {}, 0)
    bad = [r for r in got["rows"] if published.get(r["id"]) != r["t12"]]
    return got, bad, published
