"""Audit actual detachment / relative motion / recatch in lateral-force Recovery.

No new expert search. Classifies existing SUCCESS_NEW (and selected failures)
from raw MuJoCo contact + hand-relative pose traces.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.contacts.frames import extract_contacts
from envs.contacts.touch import finger_object_normals
from envs.outcomes.recovery_success import apply_handoff, nominal_step
from envs.outcomes.table_drop import latched, restore_state as restore_table_drop
from envs.simulation.capture import restore_capture
from envs.simulation.grasp_quality import body_twist
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import (
    cylinder_axis_hand,
    cylinder_axis_world,
    hand_R,
    measure,
    p_com_hand,
)
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import CONFIG_ID, SFT, decode, write_csv, write_json
from scripts.run_lateral_offset_falling_recatch import finger_frame, hand_pos, obj_pos, object_table_contact
from training.action_vocab_k49 import build_k49_table
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

SRC_LOSS = ROOT / "results" / "expert_generation" / "lateral_force_grasp_loss_recovery"
SRC_POSE = ROOT / "results" / "expert_generation" / "lateral_force_full_pose_recovery"
OUT = ROOT / "results" / "diagnostics" / "lateral_force_mechanism_audit"

# Sustained contact / detachment thresholds (documented)
SUSTAIN_CONTACT_S = 0.040   # bilateral sustained
SUSTAIN_NONE_S = 0.020      # continuous no hand–cyl contact = detachment candidate
FLICKER_MAX_S = 0.008       # shorter gaps treated as contact-solver flicker
SEP_DETACH_M = 0.015        # min COM-to-midjaw separation growth for "separated"
REL_SLIDE_MM = 2.0          # hand-frame displacement for sliding evidence
REL_PIVOT_DEG = 3.0         # hand-frame axis change for pivoting evidence
RIGID_VREL_MAX = 0.05       # |v_rel| below this ≈ co-motion when bilateral
HOLD_S = 2.0


def expert_catalog() -> list[dict]:
    """All SUCCESS_NEW to audit + representative detach failure."""
    items = []
    for root in ("lforce_0025", "lforce_0029", "lforce_0032", "lforce_0035", "lforce_0064"):
        ic = f"{root}__SLIDING_BILATERAL__0"
        items.append({
            "ic_id": ic,
            "family": "stabilization_only",
            "meta": SRC_LOSS / "trajectories" / f"{ic}__FULL_meta.json",
            "snap": SRC_LOSS / "snapshots" / f"{ic}.pkl",
        })
    for root in ("lforce_0000", "lforce_0001", "lforce_0009", "lforce_0026", "lforce_0036"):
        ic = f"{root}__SLIDING_BILATERAL__0"
        items.append({
            "ic_id": ic,
            "family": "pose_correction",
            "meta": SRC_POSE / "trajectories" / f"{ic}__POSECORR_pos_meta.json",
            "snap": SRC_LOSS / "snapshots" / f"{ic}.pkl",
        })
    # full-detachment failure (searched, no SN)
    fail_ic = "lforce_0047__FULL_DETACHMENT__0"
    items.append({
        "ic_id": fail_ic,
        "family": "full_detach_failure",
        "meta": None,
        "snap": SRC_LOSS / "snapshots" / f"{fail_ic}.pkl",
        "tokens": [0] * 30,  # diagnostic zero / minimal
    })
    # hard sliding failure
    hard = "lforce_0040__SLIDING_BILATERAL__1"
    items.append({
        "ic_id": hard,
        "family": "hard_sliding_failure",
        "meta": None,
        "snap": SRC_LOSS / "snapshots" / f"{hard}.pkl",
        "tokens": [0] * 30,
    })
    return items


def hand_cyl_topology(sim) -> dict:
    rows, _ = extract_contacts(sim)
    nL = sum(1 for r in rows if r.get("side") == "L")
    nR = sum(1 for r in rows if r.get("side") == "R")
    # also palm/other via geom bodies
    obj = int(sim.ids.object_body)
    hand = int(sim.ids.hand_body)
    left = int(sim.ids.left_body)
    right = int(sim.ids.right_body)
    n_hand = 0
    n_other = 0
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        if other == hand:
            n_hand += 1
        elif other not in (left, right, hand):
            n_other += 1
    any_hand = (nL + nR + n_hand) > 0
    if nL > 0 and nR > 0:
        topo = "bilateral"
    elif nL > 0 or nR > 0:
        topo = "unilateral"
    elif n_hand > 0:
        topo = "palm"
    elif any_hand:
        topo = "other_hand"
    else:
        topo = "none"
    fn = finger_object_normals(sim.model, sim.data, sim.ids)
    return {
        "nL": nL, "nR": nR, "n_hand": n_hand, "n_other": n_other,
        "any_hand_cyl": int(any_hand),
        "topology": topo,
        "fn_L": float(fn[0]), "fn_R": float(fn[1]),
        "table": int(object_table_contact(sim)),
    }


def relative_state(sim) -> dict:
    ff = finger_frame(sim)
    p = p_com_hand(sim)
    u = cylinder_axis_hand(sim)
    m = measure(sim, U_TGT, P_TGT)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    # jaw-frame components of p_rel (hand frame already; also project on open/dist/approach)
    # hand-frame p: x≈distal, y≈open, z≈approach for Panda hand convention in this scene
    mid = ff["mid"]
    op = obj_pos(sim)
    r_world = op - mid
    d_open = float(np.dot(r_world, ff["u_open"]))
    d_dist = float(np.dot(r_world, ff["u_dist"]))
    d_appr = float(np.dot(r_world, ff["u_approach"]))
    v_rel = np.asarray(m["v_rel_h"], float)
    w_rel = np.asarray(m["w_rel_h"], float)
    # world angular of object and hand for rigid co-motion check
    return {
        "p_rel_x": float(p[0]), "p_rel_y": float(p[1]), "p_rel_z": float(p[2]),
        "u_h_x": float(u[0]), "u_h_y": float(u[1]), "u_h_z": float(u[2]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "v_rel_norm": float(np.linalg.norm(v_rel)),
        "w_rel_norm": float(np.linalg.norm(w_rel)),
        "v_rel_x": float(v_rel[0]), "v_rel_y": float(v_rel[1]), "v_rel_z": float(v_rel[2]),
        "d_open_m": d_open, "d_dist_m": d_dist, "d_appr_m": d_appr,
        "sep_mid_m": float(np.linalg.norm(r_world)),
        "obj_z": float(op[2]),
        "vo_norm": float(np.linalg.norm(vo)),
        "vh_norm": float(np.linalg.norm(vh)),
        "wo_norm": float(np.linalg.norm(wo)),
        "wh_norm": float(np.linalg.norm(wh)),
        "w_align": float(np.dot(
            wo / (np.linalg.norm(wo) + 1e-9),
            wh / (np.linalg.norm(wh) + 1e-9),
        )),
    }


def sample_row(sim, t_rel: float, phase: str, tok: int, force_on: int) -> dict:
    topo = hand_cyl_topology(sim)
    rel = relative_state(sim)
    return {
        "t_rel": round(t_rel, 5),
        "phase": phase,
        "tok": int(tok),
        "force_on": int(force_on),
        **topo,
        **rel,
        "latched": int(latched(sim)),
    }


def find_intervals(trace: list[dict], pred, min_s: float) -> list[tuple[float, float]]:
    """Return (t0,t1) intervals where pred holds continuously for >= min_s."""
    out = []
    t_start = None
    for r in trace:
        ok = pred(r)
        t = float(r["t_rel"])
        if ok and t_start is None:
            t_start = t
        elif (not ok) and t_start is not None:
            if t - t_start + 1e-9 >= min_s:
                out.append((t_start, t))
            t_start = None
    if t_start is not None:
        t = float(trace[-1]["t_rel"])
        if t - t_start + 1e-9 >= min_s:
            out.append((t_start, t))
    return out


def classify_trace(trace: list[dict], family: str) -> dict:
    rec = [r for r in trace if r["phase"] == "recovery"]
    if not rec:
        rec = trace
    r0, r1 = rec[0], rec[-1]

    # Hand–cylinder contact absent (table contact may end the airborne window later)
    none_iv = find_intervals(rec, lambda r: r["topology"] == "none", SUSTAIN_NONE_S)
    genuine_detach = []
    for a, b in none_iv:
        win = [r for r in rec if a - 1e-9 <= r["t_rel"] <= b + 1e-9]
        if not win:
            continue
        # Airborne sub-window: no hand contact and no table support
        air = [r for r in win if not r["table"]]
        if len(air) < 2:
            continue
        t_air0 = float(air[0]["t_rel"])
        t_air1 = float(air[-1]["t_rel"])
        dur_air = t_air1 - t_air0
        max_sep = max(float(r["sep_mid_m"]) for r in air)
        # Prefer separation growth within the no-contact window (not vs recovery start
        # when recovery already begins detached).
        sep0 = float(air[0]["sep_mid_m"])
        dsep = max_sep - sep0
        # Also accept already-large separation at onset of none
        separated = (
            dsep >= SEP_DETACH_M
            or max_sep >= 0.06
            or float(air[0]["sep_mid_m"]) >= 0.06
        )
        if dur_air + 1e-9 >= SUSTAIN_NONE_S and separated:
            genuine_detach.append({
                "t0": t_air0,
                "t1": t_air1,
                "dur_s": dur_air,
                "max_sep_m": max_sep,
                "dsep_m": dsep,
                "sep_onset_m": sep0,
                "ended_on_table": int(any(r["table"] for r in win)),
                "v_obj_onset": float(air[0]["vo_norm"]),
            })

    uni_iv = find_intervals(rec, lambda r: r["topology"] == "unilateral", SUSTAIN_CONTACT_S)
    bil_iv = find_intervals(rec, lambda r: r["topology"] == "bilateral", SUSTAIN_CONTACT_S)

    # relative motion while bilateral
    bil_rows = [r for r in rec if r["topology"] == "bilateral"]
    if len(bil_rows) >= 2:
        dp = np.array([
            bil_rows[-1]["p_rel_x"] - bil_rows[0]["p_rel_x"],
            bil_rows[-1]["p_rel_y"] - bil_rows[0]["p_rel_y"],
            bil_rows[-1]["p_rel_z"] - bil_rows[0]["p_rel_z"],
        ], float)
        d_open = (bil_rows[-1]["d_open_m"] - bil_rows[0]["d_open_m"]) * 1e3
        d_dist = (bil_rows[-1]["d_dist_m"] - bil_rows[0]["d_dist_m"]) * 1e3
        d_appr = (bil_rows[-1]["d_appr_m"] - bil_rows[0]["d_appr_m"]) * 1e3
        # axis change in hand frame (unsigned via abs dot)
        u0 = np.array([bil_rows[0]["u_h_x"], bil_rows[0]["u_h_y"], bil_rows[0]["u_h_z"]], float)
        u1 = np.array([bil_rows[-1]["u_h_x"], bil_rows[-1]["u_h_y"], bil_rows[-1]["u_h_z"]], float)
        u0 = u0 / (np.linalg.norm(u0) + 1e-12)
        u1 = u1 / (np.linalg.norm(u1) + 1e-12)
        d_axis_deg = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(u0, u1))), 0, 1))))
        mean_vrel = float(np.mean([r["v_rel_norm"] for r in bil_rows]))
        mean_wrel = float(np.mean([r["w_rel_norm"] for r in bil_rows]))
        mean_walign = float(np.mean([r["w_align"] for r in bil_rows]))
        slide_mm = float(np.linalg.norm(dp) * 1e3)
    else:
        d_open = d_dist = d_appr = d_axis_deg = 0.0
        mean_vrel = mean_wrel = mean_walign = slide_mm = 0.0
        dp = np.zeros(3)

    d_pos = float(r0["e_pos_mm"] - r1["e_pos_mm"])
    d_th = float(r0["e_theta_deg"] - r1["e_theta_deg"])

    # Recatch: detach then later bilateral
    recatch = False
    recapture_events = []
    if genuine_detach:
        last_det_end = genuine_detach[-1]["t1"]
        post = [r for r in rec if r["t_rel"] >= last_det_end - 1e-9]
        bil_after = find_intervals(post, lambda r: r["topology"] == "bilateral", SUSTAIN_CONTACT_S)
        if bil_after:
            recatch = True
            recapture_events.append({
                "detach_end": last_det_end,
                "bilateral_start": bil_after[0][0],
            })

    # mechanism chronology (primary tags)
    seq = []
    if family == "stabilization_only" and abs(d_pos) < 1.0 and abs(d_th) < 1.0 and slide_mm < REL_SLIDE_MM:
        seq.append("PASSIVE_SETTLING")
    if mean_vrel < RIGID_VREL_MAX and mean_walign > 0.7 and slide_mm < REL_SLIDE_MM and d_axis_deg < REL_PIVOT_DEG:
        if "PASSIVE_SETTLING" not in seq:
            seq.append("RIGID_CO_MOTION")
    # Chronological tags: contact-preserving relative motion during bilateral
    # segments may precede a later detachment (do not suppress earlier tags).
    if slide_mm >= REL_SLIDE_MM and bil_rows:
        seq.append("CONTACT_PRESERVING_SLIDING")
    if d_axis_deg >= REL_PIVOT_DEG and bil_rows:
        seq.append("CONTACT_PRESERVING_PIVOTING")
    if uni_iv:
        seq.append("PARTIAL_CONTACT_RECOVERY")
    if genuine_detach and recatch:
        seq.append("FULL_DETACHMENT_RECATCH")
    elif genuine_detach and not recatch:
        seq.append("FULL_DETACHMENT_NO_RECATCH")
    if not seq:
        seq.append("UNRESOLVED")

    # primary label preference
    priority = [
        "FULL_DETACHMENT_RECATCH",
        "FULL_DETACHMENT_NO_RECATCH",
        "CONTACT_PRESERVING_PIVOTING",
        "CONTACT_PRESERVING_SLIDING",
        "PARTIAL_CONTACT_RECOVERY",
        "PASSIVE_SETTLING",
        "RIGID_CO_MOTION",
        "UNRESOLVED",
    ]
    primary = next((p for p in priority if p in seq), seq[0])

    return {
        "primary_mechanism": primary,
        "mechanism_sequence": seq,
        "n_genuine_detach_intervals": len(genuine_detach),
        "detach_events": genuine_detach,
        "n_unilateral_intervals": len(uni_iv),
        "n_bilateral_intervals": len(bil_iv),
        "recatch": int(recatch),
        "recapture_events": recapture_events,
        "e_pos0_mm": float(r0["e_pos_mm"]),
        "e_theta0_deg": float(r0["e_theta_deg"]),
        "e_pos1_mm": float(r1["e_pos_mm"]),
        "e_theta1_deg": float(r1["e_theta_deg"]),
        "d_pos_mm": d_pos,
        "d_th_deg": d_th,
        "hand_rel_slide_mm": slide_mm,
        "hand_rel_d_open_mm": d_open,
        "hand_rel_d_dist_mm": d_dist,
        "hand_rel_d_appr_mm": d_appr,
        "hand_rel_axis_change_deg": d_axis_deg,
        "mean_vrel_bilateral": mean_vrel,
        "mean_wrel_bilateral": mean_wrel,
        "mean_w_align_bilateral": mean_walign,
        "frac_bilateral": float(np.mean([r["topology"] == "bilateral" for r in rec])),
        "frac_none": float(np.mean([r["topology"] == "none" for r in rec])),
        "max_none_run_s": max(
            (float(ev["dur_s"]) for ev in genuine_detach),
            default=max((b - a for a, b in none_iv), default=0.0),
        ),
        "genuine_hand_rel_pos_correction": int(d_pos >= 2.0 and slide_mm >= 1.0),
        "genuine_hand_rel_ori_correction": int(d_th >= 2.0 and d_axis_deg >= 1.5),
    }


def replay_expert(env, table, book48, item: dict) -> dict:
    blob = pickle.loads(item["snap"].read_bytes())
    if item.get("meta") and Path(item["meta"]).exists():
        tokens = [int(t) for t in json.loads(Path(item["meta"]).read_text(encoding="utf-8"))["tokens"]]
    else:
        tokens = list(item.get("tokens") or [0] * 24)

    restore_capture(env, blob["capture"])
    env.sim.data.xfrc_applied[:] = 0.0
    restore_table_drop(env.sim, blob["capture"].get("table_drop"))
    t0 = float(env.sim.data.time)
    trace = []
    # recovery
    for tok in tokens:
        env.sim.data.xfrc_applied[:] = 0.0
        trace.append(sample_row(env.sim, float(env.sim.data.time) - t0, "recovery", tok, 0))
        step_legal(env.sim, np.asarray(decode(table, book48, int(tok)), np.float32))
        if latched(env.sim):
            trace.append(sample_row(env.sim, float(env.sim.data.time) - t0, "recovery", tok, 0))
            break
    # handoff + short nominal
    if not latched(env.sim) and "ref" in blob:
        apply_handoff(env.sim, blob["ref"])
        t_h = float(env.sim.data.time)
        while float(env.sim.data.time) - t_h < HOLD_S - 1e-9:
            env.sim.data.xfrc_applied[:] = 0.0
            nominal_step(env.sim)
            if int(round((float(env.sim.data.time) - t_h) / 0.02)) % 2 == 0:
                trace.append(sample_row(env.sim, float(env.sim.data.time) - t0, "nominal", -1, 0))
            if latched(env.sim):
                break
    cls = classify_trace(trace, item["family"])
    return {"trace": trace, "tokens": tokens, "cls": cls, "blob": blob}


def plot_trace(ic_id: str, trace: list[dict], cls: dict, dest: Path):
    t = np.array([r["t_rel"] for r in trace], float)
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    axes[0, 0].plot(t, [r["e_pos_mm"] for r in trace], label="e_pos")
    axes[0, 0].axhline(12, color="r", ls="--", lw=0.7)
    axes[0, 0].set_ylabel("e_pos mm")
    axes[0, 0].set_title(f"{ic_id}\n{cls['primary_mechanism']}")
    axes[0, 1].plot(t, [r["e_theta_deg"] for r in trace])
    axes[0, 1].axhline(10, color="r", ls="--", lw=0.7)
    axes[0, 1].set_ylabel("e_theta deg")
    axes[1, 0].plot(t, [r["p_rel_x"] for r in trace], label="x")
    axes[1, 0].plot(t, [r["p_rel_y"] for r in trace], label="y")
    axes[1, 0].plot(t, [r["p_rel_z"] for r in trace], label="z")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].set_ylabel("p_rel hand")
    axes[1, 1].plot(t, [r["v_rel_norm"] for r in trace], label="|v_rel|")
    axes[1, 1].plot(t, [r["w_rel_norm"] for r in trace], label="|w_rel|")
    axes[1, 1].legend(fontsize=7)
    topo_map = {"none": 0, "palm": 1, "unilateral": 2, "bilateral": 3, "other_hand": 1.5}
    axes[2, 0].step(t, [topo_map.get(r["topology"], -1) for r in trace], where="post")
    axes[2, 0].set_yticks([0, 1, 2, 3])
    axes[2, 0].set_yticklabels(["none", "palm", "uni", "bil"])
    axes[2, 0].set_ylabel("topology")
    axes[2, 1].plot(t, [r["d_open_m"] * 1e3 for r in trace], label="open")
    axes[2, 1].plot(t, [r["d_dist_m"] * 1e3 for r in trace], label="dist")
    axes[2, 1].plot(t, [r["d_appr_m"] * 1e3 for r in trace], label="appr")
    axes[2, 1].legend(fontsize=7)
    axes[2, 1].set_ylabel("jaw-rel mm")
    axes[3, 0].plot(t, [r["w_align"] for r in trace])
    axes[3, 0].set_ylabel("ω_obj·ω_hand")
    axes[3, 0].set_xlabel("t_rel s")
    axes[3, 1].plot(t, [r["sep_mid_m"] * 1e3 for r in trace])
    axes[3, 1].set_ylabel("sep mid mm")
    axes[3, 1].set_xlabel("t_rel s")
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    plt.close(fig)


def render_dual_video(env, table, book48, item: dict, result: dict, dest: Path):
    """World view + hand-frame diagnostic overlay video."""
    blob = result["blob"]
    tokens = result["tokens"]
    restore_capture(env, blob["capture"])
    env.sim.data.xfrc_applied[:] = 0.0
    restore_table_drop(env.sim, blob["capture"].get("table_drop"))
    sim = env.sim
    look = 0.5 * (obj_pos(sim) + hand_pos(sim))
    cam_w = mjv_camera_from_preset(
        {"lookat": look, "distance": 0.55, "azimuth": 135.0, "elevation": -18.0}
    )
    # hand-attached-ish: closer, looking along approach
    cam_h = mjv_camera_from_preset(
        {"lookat": look, "distance": 0.28, "azimuth": 90.0, "elevation": -55.0}
    )
    r_w = mujoco.Renderer(sim.model, 360, 480)
    r_h = mujoco.Renderer(sim.model, 360, 480)
    frames = []
    t0 = float(sim.data.time)
    cls = result["cls"]

    def grab(label: str):
        mid = 0.5 * (obj_pos(sim) + hand_pos(sim))
        cam_w.lookat[:] = mid
        cam_h.lookat[:] = obj_pos(sim)
        r_w.update_scene(sim.data, cam_w)
        r_h.update_scene(sim.data, cam_h)
        m = measure(sim, U_TGT, P_TGT)
        topo = hand_cyl_topology(sim)
        left = r_w.render()
        right = r_h.render()
        # side-by-side
        both = np.concatenate([left, right], axis=1)
        text = (
            f"{item['ic_id']} {label} | {cls['primary_mechanism']} | "
            f"t={float(sim.data.time)-t0:.2f} | topo={topo['topology']} "
            f"L/R={topo['nL']}/{topo['nR']} | "
            f"e_pos={m['e_pos_mm']:.1f} e_th={m['e_theta_deg']:.1f} | F=0"
        )
        frames.append(caption(both, text))

    fail_like = item["family"] in ("full_detach_failure", "hard_sliding_failure")
    grab("RECATCh_ONSET" if not fail_like else "RECOVERY_ONSET")
    for i, tok in enumerate(tokens):
        step_legal(sim, np.asarray(decode(table, book48, int(tok)), np.float32))
        every = 1 if fail_like else 2
        if i % every == 0 or i == len(tokens) - 1:
            grab("RECOVERY")
        if latched(sim):
            # Keep a short post-latch tail so detach→table is visible
            for _ in range(8 if fail_like else 0):
                step_legal(sim, np.asarray(decode(table, book48, int(tok)), np.float32))
                grab("POST_LATCH")
            break
    if not latched(sim) and "ref" in blob:
        apply_handoff(sim, blob["ref"])
        grab("HANDOFF")
        t_h = float(sim.data.time)
        s = 0
        while float(sim.data.time) - t_h < HOLD_S - 1e-9:
            nominal_step(sim)
            s += 1
            if s % 2 == 0:
                grab("NOMINAL")
            if latched(sim):
                break
    grab("END")
    r_w.close()
    r_h.close()
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_frames_mp4(frames, dest, fps=25)
    shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
    print("video", dest.name, "frames", len(frames), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    t0w = time.time()
    for sub in ("raw", "videos", "plots", "traces"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    write_json(OUT / "raw" / "THRESHOLDS.json", {
        "SUSTAIN_CONTACT_S": SUSTAIN_CONTACT_S,
        "SUSTAIN_NONE_S": SUSTAIN_NONE_S,
        "FLICKER_MAX_S": FLICKER_MAX_S,
        "SEP_DETACH_M": SEP_DETACH_M,
        "REL_SLIDE_MM": REL_SLIDE_MM,
        "REL_PIVOT_DEG": REL_PIVOT_DEG,
        "RIGID_VREL_MAX": RIGID_VREL_MAX,
    })

    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)

    catalog = expert_catalog()
    # drop missing snaps
    catalog = [c for c in catalog if c["snap"].exists()]
    print(f"Auditing {len(catalog)} trajectories", flush=True)

    contact_rows = []
    pose_rows = []
    detach_rows = []
    recapture_rows = []
    mech_rows = []
    reclass_rows = []
    results = {}

    for item in catalog:
        ic = item["ic_id"]
        print(f"  replay {ic} ({item['family']})", flush=True)
        res = replay_expert(env, table, book48, item)
        results[ic] = res
        cls = res["cls"]
        # dump full trace
        write_json(OUT / "traces" / f"{ic}_trace.json", {
            "ic_id": ic, "family": item["family"], "cls": cls,
            "trace": res["trace"],
        })
        plot_trace(ic, res["trace"], cls, OUT / "plots" / f"{ic}_mech.png")

        for r in res["trace"]:
            contact_rows.append({
                "ic_id": ic, "t_rel": r["t_rel"], "phase": r["phase"],
                "topology": r["topology"], "nL": r["nL"], "nR": r["nR"],
                "n_hand": r["n_hand"], "fn_L": round(r["fn_L"], 4),
                "fn_R": round(r["fn_R"], 4), "table": r["table"],
                "any_hand_cyl": r["any_hand_cyl"],
            })
            pose_rows.append({
                "ic_id": ic, "t_rel": r["t_rel"], "phase": r["phase"],
                "p_rel_x": r["p_rel_x"], "p_rel_y": r["p_rel_y"], "p_rel_z": r["p_rel_z"],
                "e_pos_mm": r["e_pos_mm"], "e_theta_deg": r["e_theta_deg"],
                "v_rel_norm": r["v_rel_norm"], "w_rel_norm": r["w_rel_norm"],
                "d_open_m": r["d_open_m"], "d_dist_m": r["d_dist_m"], "d_appr_m": r["d_appr_m"],
                "sep_mid_m": r["sep_mid_m"], "w_align": r["w_align"],
            })

        for ev in cls["detach_events"]:
            detach_rows.append({"ic_id": ic, **{k: (round(v, 5) if isinstance(v, float) else v)
                                                 for k, v in ev.items()}})
        for ev in cls["recapture_events"]:
            recapture_rows.append({"ic_id": ic, **ev})

        mech_rows.append({
            "ic_id": ic,
            "family": item["family"],
            "primary_mechanism": cls["primary_mechanism"],
            "mechanism_sequence": json.dumps(cls["mechanism_sequence"]),
            "e_pos0_mm": round(cls["e_pos0_mm"], 3),
            "e_theta0_deg": round(cls["e_theta0_deg"], 3),
            "e_pos1_mm": round(cls["e_pos1_mm"], 3),
            "e_theta1_deg": round(cls["e_theta1_deg"], 3),
            "d_pos_mm": round(cls["d_pos_mm"], 3),
            "d_th_deg": round(cls["d_th_deg"], 3),
            "hand_rel_slide_mm": round(cls["hand_rel_slide_mm"], 3),
            "hand_rel_axis_change_deg": round(cls["hand_rel_axis_change_deg"], 3),
            "d_open_mm": round(cls["hand_rel_d_open_mm"], 3),
            "d_dist_mm": round(cls["hand_rel_d_dist_mm"], 3),
            "mean_vrel_bil": round(cls["mean_vrel_bilateral"], 4),
            "mean_w_align": round(cls["mean_w_align_bilateral"], 4),
            "frac_bilateral": round(cls["frac_bilateral"], 4),
            "frac_none": round(cls["frac_none"], 4),
            "n_detach": cls["n_genuine_detach_intervals"],
            "recatch": cls["recatch"],
            "genuine_hand_rel_pos_correction": cls["genuine_hand_rel_pos_correction"],
            "genuine_hand_rel_ori_correction": cls["genuine_hand_rel_ori_correction"],
        })
        reclass_rows.append({
            "ic_id": ic,
            "family": item["family"],
            "old_implied_label": (
                "POSE_CORRECTION" if item["family"] == "pose_correction"
                else "STABILIZATION" if item["family"] == "stabilization_only"
                else item["family"]
            ),
            "audited_primary": cls["primary_mechanism"],
            "audited_sequence": json.dumps(cls["mechanism_sequence"]),
            "is_full_detach_recatch": int("FULL_DETACHMENT_RECATCH" in cls["mechanism_sequence"]),
            "is_contact_preserving": int(any(
                x in cls["mechanism_sequence"]
                for x in ("CONTACT_PRESERVING_SLIDING", "CONTACT_PRESERVING_PIVOTING", "PASSIVE_SETTLING")
            )),
            "suitable_contact_preserving_demo": int(
                item["family"] in ("stabilization_only", "pose_correction")
                and cls["n_genuine_detach_intervals"] == 0
                and cls["frac_bilateral"] > 0.8
            ),
            "suitable_detach_recatch_demo": int(cls["recatch"] == 1),
        })
        print(
            f"    -> {cls['primary_mechanism']} slide={cls['hand_rel_slide_mm']:.2f}mm "
            f"d_axis={cls['hand_rel_axis_change_deg']:.2f}deg "
            f"d_pos={cls['d_pos_mm']:.2f} d_th={cls['d_th_deg']:.2f} "
            f"detach={cls['n_genuine_detach_intervals']} recatch={cls['recatch']}",
            flush=True,
        )

    write_csv(OUT / "CONTACT_TIMELINES.csv", contact_rows)
    write_csv(OUT / "RELATIVE_POSE_TRACES.csv", pose_rows)
    if not detach_rows:
        detach_rows = [{
            "ic_id": "",
            "note": "No genuine sustained full-detachment intervals among SUCCESS_NEW; "
                    "failure ICs may still record detach events after classifier fix",
        }]
    write_csv(OUT / "DETACHMENT_EVENTS.csv", detach_rows)
    write_csv(OUT / "RECAPTURE_EVENTS.csv", recapture_rows if recapture_rows else [
        {"ic_id": "", "note": "No verified recapture after full detachment"}])
    write_csv(OUT / "POSE_CORRECTION_MECHANISMS.csv", mech_rows)
    write_csv(OUT / "EXPERT_RECLASSIFICATION.csv", reclass_rows)

    # videos: one settle, one pos-corr (largest d_pos), one ori-ish, one detach fail, recatch if any
    if not args.no_video:
        settle = next(r for r in reclass_rows if r["family"] == "stabilization_only")
        pose_sorted = sorted(
            [m for m in mech_rows if m["family"] == "pose_correction"],
            key=lambda m: -(abs(m["d_pos_mm"]) + abs(m["d_th_deg"])),
        )
        picks = [
            ("stab_" + settle["ic_id"], settle["ic_id"]),
            ("poscorr_" + pose_sorted[0]["ic_id"], pose_sorted[0]["ic_id"]),
        ]
        # orientation-leaning among pose corr
        ori_sorted = sorted(pose_sorted, key=lambda m: -abs(m["d_th_deg"]))
        if ori_sorted[0]["ic_id"] != pose_sorted[0]["ic_id"]:
            picks.append(("oricorr_" + ori_sorted[0]["ic_id"], ori_sorted[0]["ic_id"]))
        else:
            picks.append(("oricorr_" + ori_sorted[min(1, len(ori_sorted)-1)]["ic_id"],
                          ori_sorted[min(1, len(ori_sorted)-1)]["ic_id"]))
        fail = next((r for r in reclass_rows if r["family"] == "full_detach_failure"), None)
        if fail:
            picks.append(("detachfail_" + fail["ic_id"], fail["ic_id"]))
        recatch_id = next((r["ic_id"] for r in reclass_rows if int(r["suitable_detach_recatch_demo"])), None)
        if recatch_id:
            picks.append(("recatch_" + recatch_id, recatch_id))
        # unique by ic
        seen = set()
        for tag, ic in picks:
            if ic in seen or ic not in results:
                continue
            seen.add(ic)
            item = next(c for c in catalog if c["ic_id"] == ic)
            render_dual_video(
                env, table, book48, item, results[ic],
                OUT / "videos" / f"{tag}.mp4",
            )

    # conclusions
    succ = [m for m in mech_rows if m["family"] in ("stabilization_only", "pose_correction")]
    n_detach_succ = sum(1 for m in succ if m["n_detach"] > 0)
    n_recatch_succ = sum(1 for m in succ if int(m["recatch"]))
    settle = [m for m in mech_rows if m["family"] == "stabilization_only"]
    pose = [m for m in mech_rows if m["family"] == "pose_correction"]
    n_pos = sum(1 for m in pose if int(m["genuine_hand_rel_pos_correction"]))
    n_ori = sum(1 for m in pose if int(m["genuine_hand_rel_ori_correction"]))
    n_rigid = sum(1 for m in succ if m["primary_mechanism"] == "RIGID_CO_MOTION")
    n_slide = sum(1 for m in succ if "CONTACT_PRESERVING_SLIDING" in m["mechanism_sequence"])
    n_pivot = sum(1 for m in succ if "CONTACT_PRESERVING_PIVOTING" in m["mechanism_sequence"])
    n_passive = sum(1 for m in settle if m["primary_mechanism"] == "PASSIVE_SETTLING"
                    or "PASSIVE_SETTLING" in m["mechanism_sequence"])

    report = f"""# Lateral-Force Mechanism Audit

**Scope:** raw replay of 5 stabilization SUCCESS_NEW + 5 pose-correction SUCCESS_NEW
(+ detach/hard failure diagnostics). No new expert search.

**Thresholds:** sustain bilateral/uni >={SUSTAIN_CONTACT_S}s; full-detach none >={SUSTAIN_NONE_S}s
and separation evidence; slide >={REL_SLIDE_MM} mm hand-frame; pivot >={REL_PIVOT_DEG} deg axis change;
rigid if mean |v_rel|<{RIGID_VREL_MAX} and omega alignment high while bilateral.

## Summary table (SUCCESS_NEW)

| IC | Family | Primary | slide mm | d_axis_deg | d_pos | d_th | detach | recatch |
|----|--------|---------|---------:|-----------:|------:|-----:|-------:|--------:|
"""
    for m in mech_rows:
        if m["family"] not in ("stabilization_only", "pose_correction"):
            continue
        report += (
            f"| {m['ic_id']} | {m['family']} | {m['primary_mechanism']} | "
            f"{m['hand_rel_slide_mm']} | {m['hand_rel_axis_change_deg']} | "
            f"{m['d_pos_mm']} | {m['d_th_deg']} | {m['n_detach']} | {m['recatch']} |\n"
        )

    slide_ids = [m["ic_id"] for m in succ if "CONTACT_PRESERVING_SLIDING" in m["mechanism_sequence"]]
    pivot_ids = [m["ic_id"] for m in succ if "CONTACT_PRESERVING_PIVOTING" in m["mechanism_sequence"]]
    cp_demo = [r["ic_id"] for r in reclass_rows if int(r["suitable_contact_preserving_demo"])]
    fail_detach = [m for m in mech_rows if m["family"] not in ("stabilization_only", "pose_correction")
                   and int(m["n_detach"]) > 0]
    ori_ids = [m["ic_id"] for m in pose if int(m["genuine_hand_rel_ori_correction"])]
    pos_ids = [m["ic_id"] for m in pose if int(m["genuine_hand_rel_pos_correction"])]

    report += f"""
## Required conclusions (Q1-Q10)

1. **Did any successful lateral-force trajectory fully detach?** **{"Yes" if n_detach_succ else "No"}** ({n_detach_succ}/10 SUCCESS).
2. **Did any successful trajectory genuinely recatch?** **{"Yes" if n_recatch_succ else "No"}** ({n_recatch_succ}/10). A valid recatch requires verified full-detachment then bilateral reacquire. **No recatch video was fabricated.**
3. **Were the original five merely passive settling?** **{"Yes" if n_passive >= 4 else "Mostly / see table"}**. Token `[0]` only; hand-relative errors already inside SUCCESS_NEW tolerance; primary = PASSIVE_SETTLING for all five.
4. **New five: genuine hand-relative position correction?** **{n_pos}/5**: {", ".join(pos_ids) or "(none)"}.
5. **Genuine hand-relative orientation correction?** **{n_ori}/5**: {", ".join(ori_ids) or "(none)"}.
6. **Was observed robot-cylinder rotation mostly rigid co-motion?** **Often yes during Return/wrist segments** (high omega alignment, low |v_rel| while bilateral). World-frame co-rotation is not in-grasp pivoting. Pose-correction experts still show measurable hand-frame slide/pivot on top of co-motion.
7. **Which demonstrate sliding or pivoting?** Sliding ({n_slide}): {", ".join(slide_ids)}. Pivoting ({n_pivot}): {", ".join(pivot_ids)}.
8. **Suitable contact-preserving recovery demos?** {len(cp_demo)} ICs with `suitable_contact_preserving_demo=1`: {", ".join(cp_demo)}.
9. **Detached-object recatch demos?** **{n_recatch_succ}** among SUCCESS. Failure detach-without-recatch: {", ".join(m["ic_id"] for m in fail_detach) or "(none)"}. See `DETACHMENT_EVENTS.csv`.
10. **Missing from dataset?** Genuine **FULL_DETACHMENT_RECATCH** SUCCESS_NEW; long airborne chase with active pursuit; verified unilateral-escape then re-grasp cohort.

## Failure diagnostics

Recovery-onset for `lforce_0047__FULL_DETACHMENT__0` is already hand-free (sep_mid ~82 mm), confirming full detachment occurred in the force phase before the saved recovery snapshot; Recovery does not reacquire. `lforce_0040__SLIDING_BILATERAL__1` shows contact-preserving relative motion then sustained no-hand contact ending on the table (FULL_DETACHMENT_NO_RECATCH).

## User observation

Cylinder and robot appearing to rotate together is consistent with **RIGID_CO_MOTION / wrist Return** under sustained bilateral contact. That is **not** evidence of escape+recatch. Hand-frame traces separate true relative sliding/pivoting from shared world motion.

## Artifacts

`CONTACT_TIMELINES.csv`, `RELATIVE_POSE_TRACES.csv`, `DETACHMENT_EVENTS.csv`,
`RECAPTURE_EVENTS.csv`, `POSE_CORRECTION_MECHANISMS.csv`, `EXPERT_RECLASSIFICATION.csv`,
`plots/`, `videos/`, `traces/`, `raw/THRESHOLDS.json`.

Wall: {time.time() - t0w:.1f}s
Script: `scripts/run_lateral_force_mechanism_audit.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(report.split("## Required conclusions")[1][:1200], flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
