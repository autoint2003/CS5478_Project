"""One-shot CPP contact verification for large-tilt pilot deliverables."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture
from envs.step_legal import N_SUB, TRACK
from evaluation.grasp_geometry import measure
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.recovery_episode import make_training_env
from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import AIRBORNE_V3, map_action
from envs.outcomes.table_drop import object_vz, observe
import mujoco

from experiments.sliding.pilot_large_tilt_eccentric_grasp_mining import (
    CODEBOOK,
    OUT,
    P_TGT,
    SFT,
    U_TGT,
    contact_summary,
    render_video,
    run_prefix,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def step_legal_traced(sim, action):
    """Same as step_legal but yield contact after each mj_step."""
    action = np.clip(np.asarray(action, np.float32).reshape(7), -1.0, 1.0)
    rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rd = np.asarray(sim.fsm.r_des, float).reshape(3, 3)
    cmd = map_action(action, rd, rh, AIRBORNE_V3)
    vel, ang, tau = cmd["v_world"], cmd["w_world"], float(cmd["tau"])
    tau = float(np.clip(tau, sim.ids.ctrl_low[7], sim.ids.ctrl_high[7]))
    sim.fsm.p_des = np.array(sim.data.xpos[sim.ids.hand_body], float)
    sim.fsm.v_cmd = vel.copy()
    sim.fsm.w_cmd = ang.copy()
    sim.fsm.v_des = vel.copy()
    sim.fsm.w_des = ang.copy()
    sim.fsm.fg_cmd = float(-tau)
    dt = float(sim.model.opt.timestep)
    rows = []
    for _ in range(N_SUB):
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, dt)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
        )
        vz_before = object_vz(sim)
        mujoco.mj_step(sim.model, sim.data)
        observe(sim, vz_before)
        cs = contact_summary(sim)
        m = measure(sim, U_TGT, P_TGT)
        rows.append(
            {
                "nL": int(cs["nL"]),
                "nR": int(cs["nR"]),
                "e_theta_deg": float(m["e_theta_deg"]),
                "e_pos_mm": float(m["e_pos_mm"]),
            }
        )
    return action, rows


def phase_trace(env, book, blob, tokens):
    restore_capture(env, blob["capture"])
    env._prev = np.asarray(blob.get("prev_action", np.zeros(7)), np.float32)
    m0 = measure(env.sim, U_TGT, P_TGT)
    rows = []
    for ti, tok in enumerate(tokens):
        _r, action = decode_action(book, int(tok), np.zeros(7, np.float32))
        action, subs = step_legal_traced(env.sim, action)
        env._prev = np.asarray(action, np.float32)
        for si, s in enumerate(subs):
            rows.append({"tok_i": ti, "tok": int(tok), "sub": si, **s})
    m_hand = measure(env.sim, U_TGT, P_TGT)
    return m0, m_hand, rows


def summarize_phases(rows, n_tilt, n_unload, n_brake):
    nt, nu, nb = n_tilt * N_SUB, n_unload * N_SUB, n_brake * N_SUB
    phases = [
        ("tilt", rows[0:nt]),
        ("unload", rows[nt : nt + nu]),
        ("brake", rows[nt + nu : nt + nu + nb]),
    ]
    out = {}
    for name, chunk in phases:
        if not chunk:
            out[name] = None
            continue
        bi = sum(1 for s in chunk if s["nL"] > 0 and s["nR"] > 0) / len(chunk)
        zero = sum(1 for s in chunk if s["nL"] + s["nR"] == 0)
        out[name] = {
            "n_sub": len(chunk),
            "frac_bilateral": bi,
            "zero_subs": zero,
            "e_theta_start": chunk[0]["e_theta_deg"],
            "e_theta_end": chunk[-1]["e_theta_deg"],
            "e_pos_start": chunk[0]["e_pos_mm"],
            "e_pos_end": chunk[-1]["e_pos_mm"],
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    book = load_codebook(CODEBOOK)
    model, _ = load_trained_k48(SFT, DEVICE)
    model.eval()
    env = make_training_env(DEVICE, model=model)

    # --- hard_910263 CPP ---
    snap = ROOT / "results/diagnostics/hard_ic_mining/snapshots/hard_910263.pkl"
    blob = pickle.loads(snap.read_bytes())
    tok = [4] * 20 + [12] * 5 + [0] * 3
    m0, m_hand, rows = phase_trace(env, book, blob, tok)
    phases = summarize_phases(rows, 20, 5, 3)
    res = run_prefix(env, book, blob, tok, log_sub=True)
    res_s = run_prefix(env, book, blob, [32, 32], log_sub=False)
    verify = {
        "ic": "hard_910263",
        "tokens": tok,
        "m0": m0,
        "m_handoff": m_hand,
        "phases": phases,
        "frac_bilateral_tokens": res.get("frac_bilateral_tokens"),
        "zero_token_steps": res.get("zero_token_steps"),
        "success_12s": res["success_12s"],
        "win_mean_theta": res["win_mean_theta"],
        "win_mean_pos": res["win_mean_pos"],
        "recovery_elapsed": res["recovery_elapsed"],
        "short_tokens": [32, 32],
        "short_win_theta": res_s["win_mean_theta"],
        "short_win_pos": res_s["win_mean_pos"],
        "short_s12": res_s["success_12s"],
        "theta_improve_vs_short": float(res_s["win_mean_theta"] - res["win_mean_theta"]),
    }
    (OUT / "RAW" / "hard_910263_cpp_verify.json").write_text(
        json.dumps(verify, indent=2, default=float), encoding="utf-8"
    )
    print("CPP", json.dumps({k: verify[k] for k in ("success_12s", "win_mean_theta", "frac_bilateral_tokens", "zero_token_steps", "theta_improve_vs_short", "phases")}, indent=2))

    # --- gen recatch ---
    snap2 = OUT / "snapshots/gen_hrv1_920031_1_74e897d20146.pkl"
    blob2 = pickle.loads(snap2.read_bytes())
    tok2 = [4] * 12 + [36] * 5 + [0] * 8
    res2 = run_prefix(env, book, blob2, tok2, log_sub=True)
    m02, m_hand2, rows2 = phase_trace(env, book, blob2, tok2)
    zero_subs = sum(1 for s in rows2 if s["nL"] + s["nR"] == 0)
    verify2 = {
        "ic": "gen_hrv1_920031_1_74e897d20146",
        "tokens": tok2,
        "m0": m02,
        "m_handoff": m_hand2,
        "zero_subs": zero_subs,
        "frac_bilateral_tokens": res2.get("frac_bilateral_tokens"),
        "zero_token_steps": res2.get("zero_token_steps"),
        "success_12s": res2["success_12s"],
        "win_mean_theta": res2["win_mean_theta"],
        "win_mean_pos": res2["win_mean_pos"],
    }
    (OUT / "RAW" / "gen_hrv1_920031_1_recatch_verify.json").write_text(
        json.dumps(verify2, indent=2, default=float), encoding="utf-8"
    )
    print("RECATCH", verify2["success_12s"], verify2["win_mean_theta"], "zero_subs", zero_subs)

    render_video(env, book, blob, [32, 32], OUT / "videos" / "hard_910263_short.mp4", "short")
    render_video(env, book, blob, tok, OUT / "videos" / "hard_910263_pivot.mp4", "pivot")
    print("VIDEOS_OK")


if __name__ == "__main__":
    main()
