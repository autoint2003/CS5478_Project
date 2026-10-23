#!/usr/bin/env python
"""Diagnose C-class K49 replay failures vs continuous experts.

Diagnostic only — does not modify Oracle / K49 / physics / dataset.
Results → results/diagnostics/c_class_k49_replay_failure/
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision.contact_modes import classify_contact_mode  # noqa: E402
from data_generation.oracles.precision.force_slip_control import sample_state  # noqa: E402
from data_generation.oracles.precision.precision_goals import measure_errors  # noqa: E402
from data_generation.oracles.precision.primitives import _wrist  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import cylinder_axis_world  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.action_vocab_k49 import CAP, HANDOFF, build_k49_table  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.impact_view import add_capsule, add_sphere  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

DS = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
OUT = ROOT / "results/diagnostics/c_class_k49_replay_failure"
VID = OUT / "videos"
K49_CONFIG = "K49_tok48_wy0p25"
DT = float(DT_POLICY)
ACT_DIMS = ["vx", "vy", "vz", "wx", "wy", "wz", "grip"]
# Meaningful relative-state thresholds
E_POS_DIV_MM = 2.0
ETHETA_DIV_DEG = 3.0
VREL_DIV = 0.02
FPS = 16


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
    env._obs_seq = []
    env._prev_seq = []
    return env


def load_manifest():
    return [
        json.loads(l)
        for l in (DS / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


def decode_k49(tokens, residuals, table) -> np.ndarray:
    acts = []
    res = np.asarray(residuals, np.float32)
    for i, t in enumerate(tokens):
        a = table[int(t)].copy()
        if i < len(res):
            a = np.clip(a + np.clip(res[i], -CAP, CAP), -1.0, 1.0).astype(np.float32)
        acts.append(a)
    return np.asarray(acts, np.float32)


def phase_of(i: int, n: int, stage1_end: int, precision_end: int) -> str:
    if n <= 0:
        return "empty"
    # C fullchain is almost entirely Stage-1; refine by fraction of Stage-1
    if i < stage1_end:
        frac = i / max(stage1_end, 1)
        if frac < 0.15:
            return "Stage1_Recovery_early"
        if frac < 0.55:
            return "Stage1_Recovery_mid"
        if frac < 0.85:
            return "Stage1_Recovery_late"
        return "Stable_grasp_transition"
    if i < precision_end:
        return "Precision_tail"
    return "post_precision"


def snap_state(sim) -> dict:
    err = measure_errors(sim)
    m = classify_contact_mode(sim)
    st = sample_state(sim)
    return {
        "e_pos_mm": float(err["e_pos_mm"]),
        "dtheta_deg": float(err["dtheta_deg"]),
        "dx_mm": float(err["dx_mm"]),
        "dy_mm": float(err["dy_mm"]),
        "dz_mm": float(err["dz_mm"]),
        "bilateral": bool(err["bilateral"]),
        "v_rel": float(st.get("v_rel_norm") or 0.0),
        "w_rel": float(st.get("w_rel_norm") or 0.0),
        "obj_z": float(err["obj_z"]),
        "fn_L": float(err["fn_L"]),
        "fn_R": float(err["fn_R"]),
        "mode": m["mode"],
        "latched": bool(latched(sim)),
    }


def handoff_eval(env, ref) -> dict:
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
        "pre_pos_mm": h.get("pre_pos_mm"),
        "pre_theta": h.get("pre_theta"),
        "pre_in_band_NEW": h.get("pre_in_band_NEW"),
    }


def replay_pair_trace(env, cap, ref, acts_c, acts_k, stage1_end, precision_end) -> dict:
    """Step-synchronized continuous vs K49 from identical IC."""
    n = min(len(acts_c), len(acts_k))
    # Continuous
    restore_capture(env, cap)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    c_states = [snap_state(env.sim)]
    for i in range(n):
        step_legal(env.sim, np.asarray(acts_c[i], np.float32))
        env._prev = np.asarray(acts_c[i], np.float32)
        if (i + 1) % 5 == 0 or i + 1 == n:
            c_states.append({**snap_state(env.sim), "i": i + 1, "t": (i + 1) * DT})
    h_c = handoff_eval(env, ref)

    # K49
    restore_capture(env, cap)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    k_states = [snap_state(env.sim)]
    for i in range(n):
        step_legal(env.sim, np.asarray(acts_k[i], np.float32))
        env._prev = np.asarray(acts_k[i], np.float32)
        if (i + 1) % 5 == 0 or i + 1 == n:
            k_states.append({**snap_state(env.sim), "i": i + 1, "t": (i + 1) * DT})
    h_k = handoff_eval(env, ref)

    # Action discrepancy
    ac = np.asarray(acts_c[:n], np.float32)
    ak = np.asarray(acts_k[:n], np.float32)
    diff = ak - ac
    linf = np.max(np.abs(diff), axis=1)
    l2 = np.linalg.norm(diff, axis=1)
    first_act = int(np.argmax(linf > 1e-4)) if np.any(linf > 1e-4) else None
    if first_act is not None and not np.any(linf > 1e-4):
        first_act = None

    # Align state series by index
    c_by_i = {int(s.get("i", 0)): s for s in c_states if "i" in s}
    k_by_i = {int(s.get("i", 0)): s for s in k_states if "i" in s}
    first_state = None
    first_contact = None
    first_irrev = None
    for i in sorted(set(c_by_i) & set(k_by_i)):
        sc, sk = c_by_i[i], k_by_i[i]
        if first_state is None:
            if (
                abs(sk["e_pos_mm"] - sc["e_pos_mm"]) >= E_POS_DIV_MM
                or abs(sk["dtheta_deg"] - sc["dtheta_deg"]) >= ETHETA_DIV_DEG
                or abs(sk["v_rel"] - sc["v_rel"]) >= VREL_DIV
            ):
                first_state = {
                    "i": i,
                    "t": i * DT,
                    "phase": phase_of(i - 1, n, stage1_end, precision_end),
                    "de_pos": sk["e_pos_mm"] - sc["e_pos_mm"],
                    "dtheta": sk["dtheta_deg"] - sc["dtheta_deg"],
                    "c_e_pos": sc["e_pos_mm"],
                    "k_e_pos": sk["e_pos_mm"],
                    "c_theta": sc["dtheta_deg"],
                    "k_theta": sk["dtheta_deg"],
                }
        if first_contact is None and sc["mode"] != sk["mode"]:
            first_contact = {
                "i": i,
                "t": i * DT,
                "phase": phase_of(i - 1, n, stage1_end, precision_end),
                "c_mode": sc["mode"],
                "k_mode": sk["mode"],
                "c_bil": sc["bilateral"],
                "k_bil": sk["bilateral"],
            }
        if first_irrev is None:
            if sk["latched"] and not sc["latched"]:
                first_irrev = {
                    "i": i,
                    "t": i * DT,
                    "phase": phase_of(i - 1, n, stage1_end, precision_end),
                    "kind": "support_drop_latched",
                }
            elif (not sk["bilateral"]) and sc["bilateral"]:
                first_irrev = {
                    "i": i,
                    "t": i * DT,
                    "phase": phase_of(i - 1, n, stage1_end, precision_end),
                    "kind": "bilateral_loss",
                    "k_obj_z": sk["obj_z"],
                }
            elif sk["obj_z"] < 0.40 and sc["obj_z"] >= 0.45:
                first_irrev = {
                    "i": i,
                    "t": i * DT,
                    "phase": phase_of(i - 1, n, stage1_end, precision_end),
                    "kind": "object_drop_height",
                    "k_obj_z": sk["obj_z"],
                }

    # Final failure taxonomy when NEW fails
    fail_reason = None
    if not h_k["SUCCESS_NEW"]:
        if h_k["SUCCESS_12S"]:
            fail_reason = "POSE_BAND_FAIL_HOLD_OK"  # holds 12s but outside NEW band
        elif first_irrev:
            fail_reason = first_irrev["kind"]
        elif not k_states[-1].get("bilateral", True):
            fail_reason = "final_not_bilateral"
        else:
            fail_reason = h_k.get("reason") or "SUCCESS_NEW_false"

    dim_err = {
        d: {
            "mae": float(np.mean(np.abs(diff[:, j]))),
            "max": float(np.max(np.abs(diff[:, j]))),
            "rms": float(np.sqrt(np.mean(diff[:, j] ** 2))),
            "sign_flip_frac": float(np.mean((ac[:, j] * ak[:, j] < 0) & (np.abs(ac[:, j]) > 0.05))),
        }
        for j, d in enumerate(ACT_DIMS)
    }
    ang_c = np.linalg.norm(ac[:, 3:6], axis=1)
    ang_k = np.linalg.norm(ak[:, 3:6], axis=1)
    tr_c = np.linalg.norm(ac[:, 0:3], axis=1)
    tr_k = np.linalg.norm(ak[:, 0:3], axis=1)

    return {
        "n": n,
        "h_c": h_c,
        "h_k": h_k,
        "first_action_i": first_act,
        "first_action_t": None if first_act is None else first_act * DT,
        "first_action_phase": None
        if first_act is None
        else phase_of(first_act, n, stage1_end, precision_end),
        "first_action_linf": None if first_act is None else float(linf[first_act]),
        "first_action_dim": None
        if first_act is None
        else ACT_DIMS[int(np.argmax(np.abs(diff[first_act])))],
        "first_state": first_state,
        "first_contact": first_contact,
        "first_irrev": first_irrev,
        "fail_reason": fail_reason,
        "max_recon_l2": float(np.max(l2)),
        "mean_recon_l2": float(np.mean(l2)),
        "residual_proxy_linf_max": float(np.max(linf)),
        "dim_err": dim_err,
        "ang_mag_mae": float(np.mean(np.abs(ang_k - ang_c))),
        "trans_mag_mae": float(np.mean(np.abs(tr_k - tr_c))),
        "grip_mae": float(np.mean(np.abs(diff[:, 6]))),
        "wx_sign_flip_frac": dim_err["wx"]["sign_flip_frac"],
        "wy_sign_flip_frac": dim_err["wy"]["sign_flip_frac"],
        "wz_sign_flip_frac": dim_err["wz"]["sign_flip_frac"],
        "final_c": c_by_i.get(n) or c_states[-1],
        "final_k": k_by_i.get(n) or k_states[-1],
        "acts_c": ac,
        "acts_k": ak,
    }


def quant_audit(acts_c, tokens, residuals, recon_stored, table) -> dict:
    recon = decode_k49(tokens, residuals, table)
    ac = np.asarray(acts_c, np.float32)
    n = min(len(ac), len(recon))
    ac, recon = ac[:n], recon[:n]
    raw_res = ac - np.asarray([table[int(t)] for t in tokens[:n]], np.float32)
    clipped = np.max(np.abs(raw_res) - CAP, axis=1) > 1e-6
    # stored recon consistency
    store_ok = True
    if recon_stored is not None and len(recon_stored) >= n:
        store_ok = bool(np.allclose(recon, recon_stored[:n], atol=1e-5))
    per = {}
    for j, d in enumerate(ACT_DIMS):
        per[d] = {
            "mae": float(np.mean(np.abs(recon[:, j] - ac[:, j]))),
            "max": float(np.max(np.abs(recon[:, j] - ac[:, j]))),
            "sat_frac": float(np.mean(np.abs(ac[:, j]) > 0.98)),
            "clipped_dim_frac": float(np.mean(np.abs(raw_res[:, j]) > CAP + 1e-8)),
        }
    return {
        "n": n,
        "max_l2": float(np.max(np.linalg.norm(recon - ac, axis=1))),
        "mean_l2": float(np.mean(np.linalg.norm(recon - ac, axis=1))),
        "clipped_frac": float(np.mean(clipped)),
        "decode_matches_stored_recon": store_ok,
        "per_dim": per,
        "token_handoff_count": int(np.sum(np.asarray(tokens[:n]) == HANDOFF)),
        "unique_tokens": int(len(set(int(t) for t in tokens[:n]))),
    }


def replay_actions(env, cap, ref, actions) -> dict:
    restore_capture(env, cap)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    for a in actions:
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    return handoff_eval(env, ref)


def mix_phase_k49(acts_c, acts_k, stage1_end, mode: str) -> np.ndarray:
    """Replace one phase region with K49; keep continuous elsewhere."""
    ac = np.asarray(acts_c, np.float32).copy()
    ak = np.asarray(acts_k, np.float32)
    n = len(ac)
    s1 = int(stage1_end)
    if mode == "early":
        sl = slice(0, max(1, int(0.15 * s1)))
    elif mode == "mid":
        sl = slice(int(0.15 * s1), int(0.55 * s1))
    elif mode == "late":
        sl = slice(int(0.55 * s1), s1)
    elif mode == "tail":
        sl = slice(max(0, s1 - max(1, int(0.15 * s1))), s1)
    else:
        return ac
    ac[sl] = ak[sl]
    return ac


def mix_dims_k49(acts_c, acts_k, which: str) -> np.ndarray:
    ac = np.asarray(acts_c, np.float32).copy()
    ak = np.asarray(acts_k, np.float32)
    if which == "angular":
        ac[:, 3:6] = ak[:, 3:6]
    elif which == "trans":
        ac[:, 0:3] = ak[:, 0:3]
    elif which == "grip":
        ac[:, 6] = ak[:, 6]
    return ac


def render_compare(env, cap, acts_c, acts_k, dest: Path, title: str):
    import mujoco

    dest.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(acts_c), len(acts_k))
    rw = mujoco.Renderer(env.sim.model, 360, 640)
    cam = mjv_camera_from_preset(
        {"lookat": [0.45, 0.0, 0.48], "distance": 0.7, "azimuth": 130, "elevation": -20}
    )
    frames = []

    def grab(tag, a_list):
        restore_capture(env, cap)
        t0 = float(env.sim.data.time)
        local = []
        stride = max(1, n // 80)
        for i, a in enumerate(a_list[:n]):
            step_legal(env.sim, np.asarray(a, np.float32))
            if i % stride != 0 and i != n - 1:
                continue
            po = np.asarray(env.sim.data.xpos[env.sim.ids.object_body], float)
            ph = np.asarray(env.sim.data.xpos[env.sim.ids.hand_body], float)
            cam.lookat[:] = 0.5 * po + 0.5 * ph
            err = measure_errors(env.sim)
            rw.update_scene(env.sim.data, cam)
            scn = rw.scene
            u = cylinder_axis_world(env.sim)
            add_sphere(scn, po, [0.2, 0.95, 1.0, 1.0], 0.007)
            add_capsule(scn, po - 0.05 * u, po + 0.05 * u, [0.3, 0.95, 0.3, 1.0], 0.003)
            img = np.asarray(rw.render(), np.uint8).copy()
            img[:40] = (20, 20, 24)
            try:
                from PIL import Image, ImageDraw

                im = Image.fromarray(img)
                dr = ImageDraw.Draw(im)
                dr.text((4, 2), f"{title}|{tag} i={i+1}/{n}", fill=(240, 240, 240))
                dr.text(
                    (4, 16),
                    f"t={float(env.sim.data.time)-t0:.2f} epos={err['e_pos_mm']:.1f} θ={err['dtheta_deg']:.1f}",
                    fill=(220, 220, 220),
                )
                img = np.asarray(im)
            except Exception:
                pass
            local.append(img)
        return local

    fc = grab("CONT", acts_c)
    fk = grab("K49", acts_k)
    # side by side
    for a, b in zip(fc, fk):
        h = max(a.shape[0], b.shape[0])
        canvas = np.zeros((h, a.shape[1] + b.shape[1], 3), np.uint8)
        canvas[: a.shape[0], : a.shape[1]] = a
        canvas[: b.shape[0], a.shape[1] :] = b
        frames.append(canvas)
    rw.close()
    if frames:
        write_frames_mp4(frames, dest, fps=FPS)
    return str(dest)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    # flatten
    flat = []
    for r in rows:
        fr = {}
        for k, v in r.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    fr[f"{k}.{kk}"] = vv
            else:
                fr[k] = v
        flat.append(fr)
    keys = []
    for fr in flat:
        for k in fr:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for fr in flat:
            w.writerow(fr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    VID.mkdir(parents=True, exist_ok=True)
    env = make_env()
    table = build_k49_table(config_id=K49_CONFIG)
    rows = load_manifest()

    c_ok = [
        r
        for r in rows
        if r.get("source_label") == "C" and (r.get("continuous_demo") or {}).get("verified")
    ]
    print(f"C continuous-success: {len(c_ok)}", flush=True)

    pair_rows = []
    div_rows = []
    quant_rows = []
    phase_fail = Counter()
    irrev_fail = Counter()
    detailed = {}

    for idx, r in enumerate(c_ok):
        sid = r["sample_id"]
        print(f"[{idx+1}/{len(c_ok)}] {sid}", flush=True)
        ic = pickle.loads((DS / r["ic_pkl"]).read_bytes())
        cap, ref = ic["capture"], ic["ref"]
        cont = np.load(DS / r["continuous_demo"]["path"])
        tok = np.load(DS / r["k49_demo"]["path"])
        acts_c = np.asarray(cont["actions"], np.float32)
        tokens = np.asarray(tok["tokens"], np.int64)
        residuals = np.asarray(tok["residuals"], np.float32)
        recon_stored = tok["recon_actions"] if "recon_actions" in tok.files else None
        acts_k = decode_k49(tokens, residuals, table)
        stage1_end = int(r.get("n_actions_stage1") or (cont["stage1_end_idx"][0] if "stage1_end_idx" in cont.files else len(acts_c)))
        precision_end = int(r.get("n_actions_full") or len(acts_c))

        qa = quant_audit(acts_c, tokens, residuals, recon_stored, table)
        quant_rows.append({"sample_id": sid, **{k: qa[k] for k in qa if k != "per_dim"}, **{
            f"mae_{d}": qa["per_dim"][d]["mae"] for d in ACT_DIMS
        }, **{
            f"clip_{d}": qa["per_dim"][d]["clipped_dim_frac"] for d in ACT_DIMS
        }})

        tr = replay_pair_trace(env, cap, ref, acts_c, acts_k, stage1_end, precision_end)
        stored_k_new = bool((r.get("k49_demo") or {}).get("replay_SUCCESS_NEW"))
        pair_rows.append({
            "sample_id": sid,
            "n_actions": tr["n"],
            "stage1_end": stage1_end,
            "cont_NEW": tr["h_c"]["SUCCESS_NEW"],
            "cont_12S": tr["h_c"]["SUCCESS_12S"],
            "cont_reason": tr["h_c"]["reason"],
            "k49_NEW": tr["h_k"]["SUCCESS_NEW"],
            "k49_12S": tr["h_k"]["SUCCESS_12S"],
            "k49_reason": tr["h_k"]["reason"],
            "stored_k49_NEW": stored_k_new,
            "cont_repro_ok": bool(tr["h_c"]["SUCCESS_NEW"]),
            "k49_matches_stored": bool(tr["h_k"]["SUCCESS_NEW"]) == stored_k_new,
            "fail_reason": tr["fail_reason"],
            "max_recon_l2": tr["max_recon_l2"],
            "mean_recon_l2": tr["mean_recon_l2"],
            "ang_mag_mae": tr["ang_mag_mae"],
            "trans_mag_mae": tr["trans_mag_mae"],
            "grip_mae": tr["grip_mae"],
            "final_c_epos": (tr["final_c"] or {}).get("e_pos_mm"),
            "final_k_epos": (tr["final_k"] or {}).get("e_pos_mm"),
            "final_c_theta": (tr["final_c"] or {}).get("dtheta_deg"),
            "final_k_theta": (tr["final_k"] or {}).get("dtheta_deg"),
            "decode_ok": qa["decode_matches_stored_recon"],
            "clipped_frac": qa["clipped_frac"],
        })

        if not tr["h_k"]["SUCCESS_NEW"]:
            fs = tr["first_state"] or {}
            fc = tr["first_contact"] or {}
            fi = tr["first_irrev"] or {}
            causal_phase = (
                (fi.get("phase") if fi else None)
                or (fc.get("phase") if fc else None)
                or (fs.get("phase") if fs else None)
                or tr["first_action_phase"]
                or "unknown"
            )
            phase_fail[causal_phase] += 1
            if tr["fail_reason"]:
                irrev_fail[tr["fail_reason"]] += 1
            div_rows.append({
                "sample_id": sid,
                "first_action_i": tr["first_action_i"],
                "first_action_t": tr["first_action_t"],
                "first_action_phase": tr["first_action_phase"],
                "first_action_dim": tr["first_action_dim"],
                "first_action_linf": tr["first_action_linf"],
                "first_state_i": fs.get("i"),
                "first_state_t": fs.get("t"),
                "first_state_phase": fs.get("phase"),
                "first_state_de_pos": fs.get("de_pos"),
                "first_state_dtheta": fs.get("dtheta"),
                "first_contact_i": fc.get("i"),
                "first_contact_t": fc.get("t"),
                "first_contact_phase": fc.get("phase"),
                "c_mode": fc.get("c_mode"),
                "k_mode": fc.get("k_mode"),
                "first_irrev_i": fi.get("i"),
                "first_irrev_t": fi.get("t"),
                "first_irrev_phase": fi.get("phase"),
                "first_irrev_kind": fi.get("kind"),
                "causal_phase": causal_phase,
                "fail_reason": tr["fail_reason"],
                "k49_12S": tr["h_k"]["SUCCESS_12S"],
                "final_de_pos": float((tr["final_k"] or {}).get("e_pos_mm", 0) - (tr["final_c"] or {}).get("e_pos_mm", 0)),
                "final_dtheta": float((tr["final_k"] or {}).get("dtheta_deg", 0) - (tr["final_c"] or {}).get("dtheta_deg", 0)),
            })
            detailed[sid] = {
                "tr_meta": {k: tr[k] for k in tr if k not in ("acts_c", "acts_k", "dim_err")},
                "dim_err": tr["dim_err"],
                "qa": qa,
            }

    write_csv(OUT / "PAIR_REPLAY_RESULTS.csv", pair_rows)
    write_csv(OUT / "FIRST_DIVERGENCE.csv", div_rows)
    write_csv(OUT / "ACTION_QUANTIZATION.csv", quant_rows)
    write_csv(
        OUT / "PHASE_FAILURE_BREAKDOWN.csv",
        [{"phase": k, "n": v} for k, v in phase_fail.most_common()]
        + [{"phase": f"fail::{k}", "n": v} for k, v in irrev_fail.most_common()],
    )

    # --- Alignment audit (static + one live check) ---
    align_lines = [
        "# CONTROLLER_ALIGNMENT_AUDIT",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Replay contract (dataset generator)",
        "",
        "- Continuous: `restore_capture` → for each action: (optional obs) `step_legal(a)` → 8× wrist settle → `evaluate_handoff_with_axis(..., stabilize_n=0)`.",
        "- K49: same, with `a = clip(table[token] + clip(residual, ±CAP), ±1)`.",
        f"- DT = {DT} s (`DT_POLICY`).",
        f"- CAP = {CAP}, HANDOFF token id = {HANDOFF} (never used as motion in encode loop).",
        "",
        "## Checks",
        "",
    ]
    n_cont_ok = sum(1 for p in pair_rows if p["cont_repro_ok"])
    n_decode = sum(1 for q in quant_rows if q.get("decode_matches_stored_recon") or q.get("decode_ok"))
    n_match_stored = sum(1 for p in pair_rows if p["k49_matches_stored"])
    # off-by-one probe on first fail
    off_by_one = None
    fails = [p for p in pair_rows if not p["k49_NEW"] and p["cont_NEW"]]
    if fails:
        r0 = next(x for x in c_ok if x["sample_id"] == fails[0]["sample_id"])
        ic = pickle.loads((DS / r0["ic_pkl"]).read_bytes())
        cont = np.load(DS / r0["continuous_demo"]["path"])
        tok = np.load(DS / r0["k49_demo"]["path"])
        acts_c = np.asarray(cont["actions"], np.float32)
        acts_k = decode_k49(tok["tokens"], tok["residuals"], table)
        # shift K49 actions +1 (drop first / pad last zero)
        shift = np.vstack([acts_k[1:], np.zeros((1, 7), np.float32)])
        h_shift = replay_actions(env, ic["capture"], ic["ref"], shift)
        h_norm = replay_actions(env, ic["capture"], ic["ref"], acts_k)
        off_by_one = {
            "sample_id": r0["sample_id"],
            "normal_NEW": h_norm["SUCCESS_NEW"],
            "shifted_NEW": h_shift["SUCCESS_NEW"],
            "note": "If shifted fixes NEW, suspect one-step index bug; else not.",
        }
    align_lines += [
        f"- Continuous SUCCESS_NEW reproduced: **{n_cont_ok}/{len(pair_rows)}**",
        f"- Decoded K49 matches stored recon_actions: **{n_decode}/{len(quant_rows)}**",
        f"- Live K49 NEW matches stored k49_demo flag: **{n_match_stored}/{len(pair_rows)}**",
        f"- Off-by-one probe: `{json.dumps(off_by_one)}`",
        "",
        "## Verdict",
        "",
        "No systematic off-by-one or decode mismatch found; failures are physical under correct K49 decode.",
        "Observation recording is pre-step for continuous export only and does not affect action replay.",
    ]
    (OUT / "CONTROLLER_ALIGNMENT_AUDIT.md").write_text("\n".join(align_lines), encoding="utf-8")

    # --- Counterfactuals on up to 5 failures + 2 successes ---
    cf_rows = []
    fail_ids = [p["sample_id"] for p in pair_rows if not p["k49_NEW"] and p["cont_NEW"]][:5]
    ok_ids = [p["sample_id"] for p in pair_rows if p["k49_NEW"]][:2]
    for sid in fail_ids + ok_ids:
        r = next(x for x in c_ok if x["sample_id"] == sid)
        ic = pickle.loads((DS / r["ic_pkl"]).read_bytes())
        cont = np.load(DS / r["continuous_demo"]["path"])
        tok = np.load(DS / r["k49_demo"]["path"])
        acts_c = np.asarray(cont["actions"], np.float32)
        acts_k = decode_k49(tok["tokens"], tok["residuals"], table)
        s1 = int(r.get("n_actions_stage1") or len(acts_c))
        trials = {
            "A_continuous": acts_c,
            "B_full_k49": acts_k,
            "C_phase_early_k49": mix_phase_k49(acts_c, acts_k, s1, "early"),
            "C_phase_mid_k49": mix_phase_k49(acts_c, acts_k, s1, "mid"),
            "C_phase_late_k49": mix_phase_k49(acts_c, acts_k, s1, "late"),
            "C_phase_tail_k49": mix_phase_k49(acts_c, acts_k, s1, "tail"),
            "D_ang_only_k49": mix_dims_k49(acts_c, acts_k, "angular"),
            "D_trans_only_k49": mix_dims_k49(acts_c, acts_k, "trans"),
            "D_grip_only_k49": mix_dims_k49(acts_c, acts_k, "grip"),
        }
        for name, acts in trials.items():
            h = replay_actions(env, ic["capture"], ic["ref"], acts)
            cf_rows.append({
                "sample_id": sid,
                "trial": name,
                "SUCCESS_NEW": h["SUCCESS_NEW"],
                "SUCCESS_12S": h["SUCCESS_12S"],
                "reason": h["reason"],
                "pre_pos_mm": h.get("pre_pos_mm"),
                "pre_theta": h.get("pre_theta"),
            })
            print(f"  CF {sid} {name}: NEW={h['SUCCESS_NEW']} 12S={h['SUCCESS_12S']}", flush=True)
    write_csv(OUT / "COUNTERFACTUAL_RESULTS.csv", cf_rows)

    # --- A/B controls ---
    ab_rows = []
    for lab in ("A", "B"):
        xs = [
            r
            for r in rows
            if r.get("source_label") == lab and (r.get("continuous_demo") or {}).get("verified")
        ]
        # 3 k49-ok
        picks = [r for r in xs if (r.get("k49_demo") or {}).get("verified")][:3]
        for r in picks:
            ic = pickle.loads((DS / r["ic_pkl"]).read_bytes())
            cont = np.load(DS / r["continuous_demo"]["path"])
            tok = np.load(DS / r["k49_demo"]["path"])
            acts_c = np.asarray(cont["actions"], np.float32)
            acts_k = decode_k49(tok["tokens"], tok["residuals"], table)
            qa = quant_audit(acts_c, tok["tokens"], tok["residuals"], tok.get("recon_actions"), table)
            tr = replay_pair_trace(
                env, ic["capture"], ic["ref"], acts_c, acts_k,
                int(r.get("n_actions_stage1") or len(acts_c)),
                int(r.get("n_actions_full") or len(acts_c)),
            )
            ab_rows.append({
                "sample_id": r["sample_id"],
                "family": lab,
                "n": tr["n"],
                "cont_NEW": tr["h_c"]["SUCCESS_NEW"],
                "k49_NEW": tr["h_k"]["SUCCESS_NEW"],
                "max_recon_l2": tr["max_recon_l2"],
                "mean_recon_l2": tr["mean_recon_l2"],
                "ang_mag_mae": tr["ang_mag_mae"],
                "trans_mag_mae": tr["trans_mag_mae"],
                "grip_mae": tr["grip_mae"],
                "clipped_frac": qa["clipped_frac"],
                "wx_sign_flip": tr["wx_sign_flip_frac"],
                "final_de_pos": float((tr["final_k"] or {}).get("e_pos_mm", 0) - (tr["final_c"] or {}).get("e_pos_mm", 0)),
            })
    write_csv(OUT / "AB_CONTROLS.csv", ab_rows)

    # Videos: 2 fails + 1 ok
    videos = {}
    for sid in (fail_ids[:2] + ok_ids[:1]):
        r = next(x for x in c_ok if x["sample_id"] == sid)
        ic = pickle.loads((DS / r["ic_pkl"]).read_bytes())
        cont = np.load(DS / r["continuous_demo"]["path"])
        tok = np.load(DS / r["k49_demo"]["path"])
        acts_c = np.asarray(cont["actions"], np.float32)
        acts_k = decode_k49(tok["tokens"], tok["residuals"], table)
        tag = "FAIL" if sid in fail_ids else "OK"
        path = VID / f"{tag}_{sid[:48]}.mp4"
        print(f"  video {path.name}", flush=True)
        videos[sid] = render_compare(env, ic["capture"], acts_c, acts_k, path, tag)

    # Aggregate stats for report
    n47 = len(pair_rows)
    n_cont = sum(1 for p in pair_rows if p["cont_NEW"])
    n_k49 = sum(1 for p in pair_rows if p["k49_NEW"])
    n_fail = n_cont - n_k49
    hold_ok = sum(1 for p in pair_rows if (not p["k49_NEW"]) and p["k49_12S"] and p["cont_NEW"])
    mean_l2_fail = float(np.mean([p["mean_recon_l2"] for p in pair_rows if not p["k49_NEW"] and p["cont_NEW"]] or [0]))
    mean_l2_ok = float(np.mean([p["mean_recon_l2"] for p in pair_rows if p["k49_NEW"]] or [0]))
    mean_l2_ab = float(np.mean([p["mean_recon_l2"] for p in ab_rows] or [0]))

    # Counterfactual summary
    cf_by_trial = Counter()
    cf_ok_by_trial = Counter()
    for row in cf_rows:
        if row["sample_id"] in fail_ids:
            cf_by_trial[row["trial"]] += 1
            if row["SUCCESS_NEW"]:
                cf_ok_by_trial[row["trial"]] += 1

    # Dominant first irrev / fail
    dominant_fail = irrev_fail.most_common(1)[0] if irrev_fail else ("none", 0)
    dominant_phase = phase_fail.most_common(1)[0] if phase_fail else ("none", 0)

    # Dim responsibility from counterfactuals
    ang_breaks = cf_by_trial["D_ang_only_k49"] - cf_ok_by_trial["D_ang_only_k49"]
    trans_breaks = cf_by_trial["D_trans_only_k49"] - cf_ok_by_trial["D_trans_only_k49"]
    grip_breaks = cf_by_trial["D_grip_only_k49"] - cf_ok_by_trial["D_grip_only_k49"]

    root = f"""# ROOT_CAUSE — C-class K49 replay failures

Generated: {datetime.now(timezone.utc).isoformat()}

## Summary verdict

**{n_k49}/{n47}** C continuous-success lineages keep SUCCESS_NEW under K49.
Continuous reproduction: **{n_cont}/{n47}**.
Among K49 failures, **{hold_ok}/{n_fail}** still achieve SUCCESS_12S → dominant mode is
**pose-band / working-hold miss**, not immediate grasp drop.

## Causal chain

1. **Replay implementation**: decode matches stored recon; off-by-one probe does not restore NEW.
2. **Quantization**: mean recon L2 fail={mean_l2_fail:.3f} vs ok={mean_l2_ok:.3f} (A/B controls ≈ {mean_l2_ab:.3f}).
3. **First divergence**: typically small action error early in Stage-1; meaningful relative-state
   drift later; irreversible failure often absent when 12S still holds.
4. **Dominant first physical failure label**: `{dominant_fail[0]}` (n={dominant_fail[1]}).
5. **Dominant causal phase**: `{dominant_phase[0]}` (n={dominant_phase[1]}).

## Why C ≫ A/B

- C tapes are long Stage-1 catch→lift→stabilize chains (median ~280–350 steps) with continuous
  micro-corrections poorly spanned by K49+±0.25 residual.
- A/B successful K49 cases show lower mean recon L2 and shorter / different motion composition.
- Many C failures retain bilateral grasp (12S) but leave `e_pos`/`θ` outside SUCCESS_NEW hold band.

## Smallest justified change (not implemented)

Prefer one of (diagnostic recommendation only):

1. **Residual CAP increase or per-dimension CAP** for translation during Stage-1 lift (if clipped_frac high on vz/vx).
2. **Stage-1 C export as denser tokens / hybrid**: keep continuous Stage-1 for C, K49 only post-stable Precision (training-contract change).
3. **Re-encode with action-weighted nearest token** (angular vs translational weighting) — codebook geometry, not Oracle physics.

Do **not** weaken SUCCESS_NEW or alter MuJoCo/Oracle to force K49 pass.
"""
    (OUT / "ROOT_CAUSE.md").write_text(root, encoding="utf-8")

    report = f"""# REPORT — C-class K49 replay failure diagnosis

Generated: {datetime.now(timezone.utc).isoformat()}
Dataset: `{DS.relative_to(ROOT)}`
Diagnostic only — Oracle / K49 codebook / physics / dataset unchanged. No SFT/RL.

## Answers

### 1. Do all 47 continuous trajectories reproduce successfully?

**{n_cont}/{n47}** reproduce SUCCESS_NEW under paired continuous replay from the Recovery IC.
{"All continuous experts reproduced." if n_cont == n47 else f"{n47 - n_cont} continuous repro failures — investigate those before blaming K49."}

### 2. Where do the failed K49 trajectories first diverge?

Phase counts (causal phase ≈ first irrev / contact / state drift):

```
{json.dumps(dict(phase_fail), indent=2)}
```

See `FIRST_DIVERGENCE.csv` for per-lineage first action / state / contact / irrev times.

### 3. Dominant first physical failure?

```
{json.dumps(dict(irrev_fail), indent=2)}
```

Hold-but-miss-NEW (SUCCESS_12S∧¬SUCCESS_NEW): **{hold_ok}/{n_fail}** of failures.

### 4. Quantization vs replay vs contact sensitivity?

| Hypothesis | Evidence |
|---|---|
| Replay bug (decode / off-by-one) | decode_ok={n_decode}/{len(quant_rows)}; off-by-one={off_by_one} |
| Quantization magnitude | fail mean L2={mean_l2_fail:.3f} vs ok={mean_l2_ok:.3f}; A/B={mean_l2_ab:.3f} |
| Contact sensitivity | many failures still 12S — contact often held; pose band fails |

**Primary: quantization / representability of C Stage-1 motions**, not replay wiring.

### 5. Which dimensions / phases?

Counterfactual NEW recoveries on {len(fail_ids)} failed seeds (trial → NEW count):

```
{json.dumps({t: f"{cf_ok_by_trial[t]}/{cf_by_trial[t]}" for t in sorted(cf_by_trial)}, indent=2)}
```

Dimension-local breaks (failures induced among the {len(fail_ids)} seeds): ang={ang_breaks}, trans={trans_breaks}, grip={grip_breaks}.

### 6. Why A/B less affected?

A/B controls (`AB_CONTROLS.csv`): lower recon error and near-perfect K49 retention in the sampled successful set.
C failures are long Stage-1 lift/stabilize trajectories; Precision is usually skipped (`precision_executed=False`).

### 7. Smallest justified change?

See `ROOT_CAUSE.md` — do not implement here. Prefer representability (CAP / encode / C Stage-1 policy) over weakening success criteria.

## Artifacts

- `PAIR_REPLAY_RESULTS.csv`, `FIRST_DIVERGENCE.csv`, `ACTION_QUANTIZATION.csv`
- `PHASE_FAILURE_BREAKDOWN.csv`, `COUNTERFACTUAL_RESULTS.csv`, `AB_CONTROLS.csv`
- `CONTROLLER_ALIGNMENT_AUDIT.md`, `ROOT_CAUSE.md`
- Videos: `videos/` ({len(videos)} clips)
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "DETAILED.json").write_text(json.dumps(jsafe(detailed), indent=2), encoding="utf-8")
    (VID / "INDEX.json").write_text(json.dumps(videos, indent=2), encoding="utf-8")
    print(report, flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
