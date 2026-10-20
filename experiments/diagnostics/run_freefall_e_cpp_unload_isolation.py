"""Matched unload failure isolation after freefall-E + CPP tilt.

Isolates why CPP mid-unload loses contact on E stable-lift states.
Privileged diagnostic only. No SFT/GRPO. Physics / SUCCESS_NEW / ICs unchanged.

Branches from matched pre-unload snapshots:
  A  hold grip + end-of-tilt wrist/hand (no unload)
  B  documented CPP unload (zero-residual tok12) + brake
  B_prior  residual-carry decode used in freefall_e_stable_lift_contact_manip
  C  reduced unload amplitude (a6 milder than mid)
  D  reduced unload rate (ramp to mid over 2x steps)
  E  keep grip a6=+1; apply unload/brake wrist-hand components only

If pre-unload |s_axial| < ECCENTRIC_MM, also probe axial slide → reclamp
(documented CPS mid+secure-translate pattern; no pivoting).
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from envs.outcomes.table_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.success_criteria import contact_summary
from evaluation.grasp_geometry import cylinder_axis_world, p_com_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT, write_csv, write_json
from scripts.run_freefall_e_stable_lift_contact_manip import (
    CPP_RECIPES,
    held_ics,
    run_e_to_stable_lift,
)
from scripts.run_freefall_recatch_grasp_point_alignment import (
    BASE_GAINS,
    aperture,
    axial_summary,
    contacts_axial,
)
from scripts.run_freefall_recatch_pose_feedback import pose_now
from scripts.run_lateral_offset_falling_recatch import (
    hand_pos,
    make_training_env,
    obj_pos,
    restore_ic,
)
from scripts.run_lateral_recatch_handoff import clear_probe_latch, hand_R
from training.k47_codebook import decode_action, load_codebook

SRC = ROOT / "results/diagnostics/freefall_e_stable_lift_contact_manip"
OUT = ROOT / "results/diagnostics/freefall_e_cpp_unload_isolation"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"

# Representative failures: diverse grasp / pose / (implicit) relative motion
REP_ICS = [
    "distal_off01_h8",   # near-SN, distal, small theta
    "perp_off01_h0",     # moderate theta ~18, CH
    "perp_off01_h2",     # near gate ~10.6
    "distal_off01_h4",   # distal ~22
    "perp_off01_h1",     # extreme ~89
    "perp_off01_h7_hz",  # horizontal variant
]

# Primary recipe matching prior best-effort / verified CPP
PRIMARY = "cpp_hrv1_920031"  # tilt6x20, unload12x15, brake0x8
ECCENTRIC_MM = 8.0  # verified CPP ICs had ~8.9–12.5 mm lever
WALL_S = 40 * 60
HORIZON_EXTRA = 25  # steps after branch schedule for late latch/drop


def decode_zero(book, tok: int) -> np.ndarray:
    """Documented CPP decode: zero residual each step (verified pilots)."""
    _r, a = decode_action(book, int(tok), np.zeros(7, np.float32))
    return np.asarray(a, np.float32).copy()


def decode_carry(book, tok: int, pre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residual-carry decode (used in freefall_e_stable_lift_contact_manip)."""
    _r, a = decode_action(book, int(tok), np.asarray(pre, np.float32))
    a = np.asarray(a, np.float32).copy()
    return a, a


def body_v_w(sim, body_id: int):
    # cvel: [ang(3); lin(3)] in world
    c = np.asarray(sim.data.cvel[body_id], float)
    return c[3:].copy(), c[:3].copy()


def contact_force_detail(sim) -> dict:
    """Pad–cylinder normal/tangential forces + contact positions (diagnostic)."""
    obj = int(sim.ids.object_body)
    left, right = int(sim.ids.left_body), int(sim.ids.right_body)
    fn_L = ft_L = fn_R = ft_R = 0.0
    nL = nR = 0
    pos_L, pos_R = [], []
    for i in range(int(sim.data.ncon)):
        c = sim.data.contact[i]
        b1 = int(sim.model.geom_bodyid[int(c.geom1)])
        b2 = int(sim.model.geom_bodyid[int(c.geom2)])
        if obj not in (b1, b2):
            continue
        other = b2 if b1 == obj else b1
        if other not in (left, right):
            continue
        force = np.zeros(6, float)
        try:
            mujoco.mj_contactForce(sim.model, sim.data, i, force)
        except Exception:
            force[:] = 0.0
        # force[:3] in contact frame: [normal, t1, t2]
        fn = abs(float(force[0]))
        ft = float(np.linalg.norm(force[1:3]))
        pos = np.asarray(c.pos, float)
        if other == left:
            nL += 1
            fn_L += fn
            ft_L += ft
            pos_L.append(pos.tolist())
        else:
            nR += 1
            fn_R += fn
            ft_R += ft
            pos_R.append(pos.tolist())
    return {
        "nL_con": nL, "nR_con": nR,
        "fn_L": fn_L, "ft_L": ft_L, "fn_R": fn_R, "ft_R": ft_R,
        "fn_tot": fn_L + fn_R, "ft_tot": ft_L + ft_R,
        "pos_L": pos_L[:3], "pos_R": pos_R[:3],
    }


def grip_actuator_force(sim) -> float:
    # actuator8 = index 7 is finger/grip in this model
    af = np.asarray(sim.data.actuator_force, float)
    if af.size > 7:
        return float(af[7])
    return float(af[-1]) if af.size else float("nan")


def rich_row(sim, step: int, phase: str, a: np.ndarray, t0: float) -> dict:
    cs = contact_summary(sim)
    cons, u_ax, po = contacts_axial(sim)
    ax = axial_summary(cons)
    pose = pose_now(sim)
    cf = contact_force_detail(sim)
    vh, wh = body_v_w(sim, int(sim.ids.hand_body))
    vo, wo = body_v_w(sim, int(sim.ids.object_body))
    v_rel = vo - vh
    w_rel = wo - wh
    # COM offset in hand frame (lever for pivoting)
    try:
        p_h = p_com_hand(sim)
        ecc_xy = float(np.linalg.norm(p_h[:2]))
    except Exception:
        ecc_xy = float("nan")
        p_h = [float("nan")] * 3
    Rh = hand_R(sim)
    # hand linear command approx in world (diagnostic; a is hand-frame-ish 7D)
    return {
        "step": step,
        "t": float(sim.data.time) - t0,
        "phase": phase,
        "bilateral": int(cs["bilateral"]),
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "fn_L": cf["fn_L"], "ft_L": cf["ft_L"],
        "fn_R": cf["fn_R"], "ft_R": cf["ft_R"],
        "fn_tot": cf["fn_tot"], "ft_tot": cf["ft_tot"],
        "contact_pos_L": json.dumps(cf["pos_L"]),
        "contact_pos_R": json.dumps(cf["pos_R"]),
        "grip_af": grip_actuator_force(sim),
        "grip_ctrl": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float("nan"),
        "aper": aperture(sim),
        "mean_s_axial": ax.get("mean_s_axial"),
        "max_abs_s": ax.get("max_abs_s"),
        "n_near_center": ax.get("n_near_center"),
        "n_near_tail": ax.get("n_near_tail"),
        "ecc_xy_mm": ecc_xy * 1e3 if ecc_xy == ecc_xy else None,
        "p_com_h": json.dumps([float(x) for x in p_h]),
        "v_rel": float(np.linalg.norm(v_rel)),
        "w_rel": float(np.linalg.norm(w_rel)),
        "v_rel_xyz": json.dumps(v_rel.tolist()),
        "w_rel_xyz": json.dumps(w_rel.tolist()),
        "theta": pose["e_theta_deg"],
        "pos_mm": pose["e_pos_mm"],
        "obj_z": float(obj_pos(sim)[2]),
        "hand_z": float(hand_pos(sim)[2]),
        "a0": float(a[0]), "a1": float(a[1]), "a2": float(a[2]),
        "a3": float(a[3]), "a4": float(a[4]), "a5": float(a[5]),
        "a6": float(a[6]),
        "latched": int(latched(sim)),
    }


def event_times(rows: list[dict]) -> dict:
    """First loss of contact / table latch / drop markers."""
    had_bil = False
    t_lose_bil = t_zero = t_latch = t_drop = None
    fn_at_lose = v_at_lose = aper_at_lose = None
    for r in rows:
        if int(r["bilateral"]):
            had_bil = True
        if had_bil and t_lose_bil is None and not int(r["bilateral"]):
            t_lose_bil = r["t"]
            fn_at_lose = r["fn_tot"]
            v_at_lose = r["v_rel"]
            aper_at_lose = r["aper"]
        if t_zero is None and int(r["nL"]) + int(r["nR"]) == 0 and had_bil:
            t_zero = r["t"]
        if t_latch is None and int(r["latched"]):
            t_latch = r["t"]
        # drop proxy: obj falling away with no finger contact
        if (
            t_drop is None
            and had_bil
            and int(r["nL"]) + int(r["nR"]) == 0
            and float(r["obj_z"]) < float(r["hand_z"]) - 0.05
        ):
            t_drop = r["t"]
    return {
        "t_first_lose_bilateral": t_lose_bil,
        "t_first_zero_finger_contact": t_zero,
        "t_first_latch": t_latch,
        "t_drop_proxy": t_drop,
        "fn_at_lose_bil": fn_at_lose,
        "v_rel_at_lose_bil": v_at_lose,
        "aper_at_lose_bil": aper_at_lose,
        "retained_bilateral_all": int(all(int(r["bilateral"]) for r in rows)) if rows else 0,
        "final_bilateral": int(rows[-1]["bilateral"]) if rows else 0,
        "final_latched": int(rows[-1]["latched"]) if rows else 0,
        "final_theta": rows[-1]["theta"] if rows else None,
        "final_pos_mm": rows[-1]["pos_mm"] if rows else None,
        "final_s_axial": rows[-1]["mean_s_axial"] if rows else None,
        "final_fn": rows[-1]["fn_tot"] if rows else None,
        "min_fn": min((float(r["fn_tot"]) for r in rows), default=None),
        "max_v_rel": max((float(r["v_rel"]) for r in rows), default=None),
        "max_aper": max((float(r["aper"]) for r in rows), default=None),
        "delta_s_axial_mm": (
            None if rows[0]["mean_s_axial"] is None or rows[-1]["mean_s_axial"] is None
            else 1e3 * (float(rows[-1]["mean_s_axial"]) - float(rows[0]["mean_s_axial"]))
        ),
    }


def classify_mechanism(ev: dict, branch: str, snap_meta: dict) -> str:
    """Order-of-failure classification from raw traces (not FSM names)."""
    if branch == "A" and ev.get("retained_bilateral_all"):
        return "stable_hold_no_unload"
    t_lose = ev.get("t_first_lose_bilateral")
    t_latch = ev.get("t_first_latch")
    if t_lose is None and not ev.get("final_latched"):
        if branch.startswith("B") or branch in ("C", "D"):
            return "unload_retained_contact"
        if branch == "E":
            return "grip_hold_wrist_ok"
        return "retained_contact"
    # lost contact
    fn = ev.get("fn_at_lose_bil")
    v = ev.get("v_rel_at_lose_bil")
    aper = ev.get("aper_at_lose_bil")
    ecc = snap_meta.get("ecc_xy_mm")
    s_ax = snap_meta.get("mean_s_axial")
    s_mm = None if s_ax is None else abs(float(s_ax)) * 1e3
    parts = []
    if fn is not None and fn < 5.0:
        parts.append("normal_force_collapse")
    if aper is not None and aper > 0.022:
        parts.append("aperture_open_slip")
    if v is not None and v > 0.15:
        parts.append("relative_motion_spike")
    if t_latch is not None and (t_lose is None or t_latch >= t_lose - 1e-6):
        parts.append("table_latch_after_or_at_loss")
    if s_mm is not None and s_mm < ECCENTRIC_MM and (ecc is None or ecc < ECCENTRIC_MM):
        parts.append("insufficient_eccentric_lever")
    if branch == "E" and t_lose is not None:
        parts.append("wrist_motion_broke_contact_despite_grip")
    if branch == "A" and t_lose is not None:
        parts.append("baseline_unstable_without_unload")
    if not parts:
        parts.append("contact_loss_unclassified")
    return "+".join(parts)


def build_branch_actions(book, recipe_name: str, branch: str, tilt_last: np.ndarray) -> list[tuple[str, np.ndarray]]:
    phases = CPP_RECIPES[recipe_name]["phases"]
    unload_toks = dict(phases)["unload"]
    brake_toks = dict(phases)["brake"]
    n_u, n_b = len(unload_toks), len(brake_toks)
    out: list[tuple[str, np.ndarray]] = []

    if branch == "A":
        hold = tilt_last.copy()
        hold[6] = 1.0
        for _ in range(n_u + n_b + HORIZON_EXTRA):
            out.append(("hold", hold.copy()))
        return out

    if branch == "B":
        for _ in unload_toks:
            out.append(("unload", decode_zero(book, 12)))
        for _ in brake_toks:
            out.append(("brake", decode_zero(book, 0)))
        for _ in range(HORIZON_EXTRA):
            out.append(("hold", decode_zero(book, 0)))
        return out

    if branch == "B_prior":
        pre = tilt_last.copy()
        for _ in unload_toks:
            a, pre = decode_carry(book, 12, pre)
            out.append(("unload", a))
        for _ in brake_toks:
            a, pre = decode_carry(book, 0, pre)
            out.append(("brake", a))
        for _ in range(HORIZON_EXTRA):
            a, pre = decode_carry(book, 0, pre)
            out.append(("hold", a))
        return out

    if branch == "C":
        # reduced amplitude: a6 = -0.15 (milder than mid -0.48), zero wrist/hand
        mild = np.zeros(7, np.float32)
        mild[6] = -0.15
        for _ in range(n_u):
            out.append(("unload_mild", mild.copy()))
        for _ in range(n_b + HORIZON_EXTRA):
            out.append(("brake", decode_zero(book, 0)))
        return out

    if branch == "D":
        # reduced rate: ramp a6 from +1 → -0.48 over 2*n_u steps
        n_ramp = 2 * n_u
        for i in range(n_ramp):
            a = np.zeros(7, np.float32)
            frac = (i + 1) / n_ramp
            a[6] = 1.0 + frac * (-0.48 - 1.0)
            out.append(("unload_ramp", a))
        for _ in range(n_b + HORIZON_EXTRA):
            out.append(("brake", decode_zero(book, 0)))
        return out

    if branch == "E":
        # keep grip; apply wrist/hand of documented unload/brake (tok12/0 are zero wrist —
        # so also apply residual-free tilt-hold wrist decay: use end-tilt wrist then brake)
        # Documented tok12 has zero wrist; to test "subsequent wrist/hand motion" we keep
        # tilt wrist command with grip locked, matching "CPP subsequent" intent when
        # residual-carry would have changed wrist — here: hold tilt wrist then zero.
        for i in range(n_u):
            a = tilt_last.copy()
            a[6] = 1.0
            # gently release wrist toward 0 (simulates schedule leaving tilt) without opening
            if i >= n_u // 2:
                a[3] = 0.0
                a[4] = 0.0
                a[5] = 0.0
            out.append(("wrist_keep_grip", a))
        for _ in range(n_b + HORIZON_EXTRA):
            a = decode_zero(book, 0)
            out.append(("brake", a))
        return out

    raise ValueError(branch)


def secure_axial_tokens(book, desire_h: np.ndarray, top_k=4) -> list[int]:
    """Rank secure-grip (+1) translation tokens by alignment with desire_h (CPS style)."""
    d = np.asarray(desire_h, float)
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return [33, 1, 13]  # known z-ish secure fallbacks
    d = d / n
    ranked = []
    for i in range(len(book)):
        a = np.asarray(book[i], float)
        if a[6] < 0.5:
            continue
        vh = np.array([a[0], a[1], (a[2] * 3.8 if a[2] >= 0 else a[2])], float)
        spd = float(np.linalg.norm(vh))
        if spd < 0.02:
            continue
        ranked.append((float(np.dot(vh / spd, d)) * spd, i))
    ranked.sort(key=lambda x: -x[0])
    return [i for _, i in ranked[:top_k]] or [33, 1, 13]


def build_axial_slide_actions(book, sim):
    """Documented CPS family: mid unload + secure translate along axis + reclamp. No pivot."""
    # desire: shift grasp toward creating COM lever — move hand along +cylinder axis
    u = cylinder_axis_world(sim)
    Rh = hand_R(sim)
    desire_h = Rh.T @ u  # hand-frame direction along cylinder
    toks = secure_axial_tokens(book, desire_h, top_k=3)
    t_slide = int(toks[0])
    out = []
    for _ in range(5):
        out.append(("slide_mid", decode_zero(book, 12)))
    for _ in range(10):
        out.append(("slide_secure", decode_zero(book, t_slide)))
    for _ in range(5):
        out.append(("reclamp", decode_zero(book, 0)))
    for _ in range(HORIZON_EXTRA):
        out.append(("hold", decode_zero(book, 0)))
    return out, t_slide, desire_h.tolist()


def replay_to_pre_unload(env, ic, prefix, book, recipe_name: str):
    """E prefix + documented zero-residual tilt → matched pre-unload capture.

    If tilt ends without bilateral, restore the *last bilateral* state during tilt
    (or post-prefix) so unload branches remain matched and comparable.
    """
    restore_ic(env, ic)
    clear_probe_latch(env.sim)
    for a in prefix:
        step_legal(env.sim, np.asarray(a, np.float32))
        if latched(env.sim):
            return None, None, "LATCH_DURING_PREFIX"
    last_bil_cap = take_capture(env) if contact_summary(env.sim)["bilateral"] else None
    last_bil_action = np.zeros(7, np.float32)
    last_bil_action[6] = 1.0
    tilt_toks = dict(CPP_RECIPES[recipe_name]["phases"])["tilt"]
    tilt_last = None
    for tok in tilt_toks:
        a = decode_zero(book, int(tok))
        step_legal(env.sim, a)
        tilt_last = a
        if contact_summary(env.sim)["bilateral"] and not latched(env.sim):
            last_bil_cap = take_capture(env)
            last_bil_action = a.copy()
        if latched(env.sim):
            if last_bil_cap is not None:
                return last_bil_cap, last_bil_action, "TILT_LATCH_USE_LAST_BIL"
            return None, None, "LATCH_DURING_TILT"
    if tilt_last is None:
        return None, None, "NO_TILT"
    if contact_summary(env.sim)["bilateral"] and not latched(env.sim):
        return take_capture(env), tilt_last, None
    if last_bil_cap is not None:
        return last_bil_cap, last_bil_action, "NO_BILATERAL_AT_TILT_END_USE_LAST_BIL"
    return None, tilt_last, "NO_BILATERAL_PRE_UNLOAD"


def run_branch(env, cap, acts: list[tuple[str, np.ndarray]]) -> list[dict]:
    restore_capture(env, cap)
    t0 = float(env.sim.data.time)
    rows = []
    for i, (phase, a) in enumerate(acts):
        # log state BEFORE step (command about to apply) then AFTER
        step_legal(env.sim, np.asarray(a, np.float32))
        rows.append(rich_row(env.sim, i, phase, a, t0))
        if latched(env.sim) and i > 2:
            # continue a few steps to mark latch time clearly, then stop
            if i >= 5 and all(int(r["latched"]) for r in rows[-3:]):
                break
    return rows


def snap_meta_from_sim(sim) -> dict:
    cons, _, _ = contacts_axial(sim)
    ax = axial_summary(cons)
    pose = pose_now(sim)
    cf = contact_force_detail(sim)
    try:
        ecc = float(np.linalg.norm(p_com_hand(sim)[:2])) * 1e3
    except Exception:
        ecc = None
    vh, wh = body_v_w(sim, int(sim.ids.hand_body))
    vo, wo = body_v_w(sim, int(sim.ids.object_body))
    return {
        "theta": pose["e_theta_deg"],
        "pos_mm": pose["e_pos_mm"],
        "mean_s_axial": ax.get("mean_s_axial"),
        "max_abs_s": ax.get("max_abs_s"),
        "ecc_xy_mm": ecc,
        "fn_tot": cf["fn_tot"],
        "aper": aperture(sim),
        "v_rel": float(np.linalg.norm(vo - vh)),
        "w_rel": float(np.linalg.norm(wo - wh)),
        "bilateral": int(contact_summary(sim)["bilateral"]),
        "obj_z": float(obj_pos(sim)[2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ics", type=int, default=6)
    ap.add_argument("--recipe", default=PRIMARY)
    args = ap.parse_args()

    t0w = time.time()
    for sub in ("raw", "snapshots", "traces", "comparisons"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    book = load_codebook(CODEBOOK)
    gains = dict(BASE_GAINS)

    all_ics = {ic["ic_id"]: ic for ic in held_ics(env)}
    selected = []
    for name in REP_ICS:
        if name in all_ics:
            selected.append(all_ics[name])
        if len(selected) >= int(args.max_ics):
            break
    print(f"Selected ICs: {[ic['ic_id'] for ic in selected]}", flush=True)

    write_json(OUT / "raw" / "BUDGETS.json", {
        "wall_s": WALL_S,
        "recipe": args.recipe,
        "recipe_source": CPP_RECIPES[args.recipe]["source"],
        "branches": ["A", "B", "B_prior", "C", "D", "E"],
        "decode_note": {
            "B": "zero-residual decode_action (verified CPP pilots)",
            "B_prior": "residual-carry as in freefall_e_stable_lift_contact_manip tokens_to_actions",
            "C": "continuous a6=-0.15 mild unload (not a codebook token)",
            "D": "continuous ramp a6 +1->-0.48 over 2x unload steps",
        },
        "eccentric_mm_threshold": ECCENTRIC_MM,
        "note": "Privileged diagnostic; no SUCCESS_NEW criterion change; no SFT/GRPO",
    })

    comparison_rows = []
    slide_rows = []
    all_events = []

    for ic in selected:
        if time.time() - t0w > WALL_S:
            print("WALL", flush=True)
            break
        ic_id = ic["ic_id"]
        print(f"\n=== {ic_id} ===", flush=True)

        # Prefer prior stable-lift prefix if present
        prior_pkl = SRC / "snapshots" / f"{ic_id}__stable_lift.pkl"
        if prior_pkl.exists():
            blob = pickle.loads(prior_pkl.read_bytes())
            prefix = blob["prefix"]
            ic_use = blob.get("ic", ic)
            print(f"  using prior stable_lift prefix ({len(prefix)} acts)", flush=True)
        else:
            snap, _base, _fb = run_e_to_stable_lift(env, ic, gains)
            if snap is None:
                print("  no stable-lift; skip", flush=True)
                continue
            prefix = snap["prefix_actions"]
            ic_use = ic

        cap, tilt_last, err = replay_to_pre_unload(env, ic_use, prefix, book, args.recipe)
        if cap is None or err in ("LATCH_DURING_PREFIX", "LATCH_DURING_TILT", "NO_TILT", "NO_BILATERAL_PRE_UNLOAD"):
            print(f"  pre-unload failed: {err}", flush=True)
            comparison_rows.append({
                "ic_id": ic_id, "branch": None, "pre_unload_ok": 0, "error": err,
            })
            continue
        if err:
            print(f"  warn: {err}", flush=True)

        restore_capture(env, cap)
        meta0 = snap_meta_from_sim(env.sim)
        meta0["ic_id"] = ic_id
        meta0["pre_unload_warn"] = err
        write_json(OUT / "snapshots" / f"{ic_id}__pre_unload_meta.json", meta0)
        pickle.dump(
            {"ic": ic_use, "capture": cap, "tilt_last": tilt_last, "meta": meta0,
             "prefix_len": len(prefix), "recipe": args.recipe},
            (OUT / "snapshots" / f"{ic_id}__pre_unload.pkl").open("wb"),
        )
        print(
            f"  pre-unload th={meta0['theta']:.1f} pos={meta0['pos_mm']:.1f} "
            f"s_ax={meta0['mean_s_axial']} ecc={meta0['ecc_xy_mm']} "
            f"fn={meta0['fn_tot']:.1f} vrel={meta0['v_rel']:.3f}",
            flush=True,
        )

        branch_names = ["A", "B", "B_prior", "C", "D", "E"]
        for br in branch_names:
            acts = build_branch_actions(book, args.recipe, br, tilt_last)
            rows = run_branch(env, cap, acts)
            for r in rows:
                r["ic_id"] = ic_id
                r["branch"] = br
            write_csv(OUT / "traces" / f"{ic_id}__{br}.csv", rows)
            np.savez_compressed(
                OUT / "traces" / f"{ic_id}__{br}__actions.npz",
                actions=np.asarray([a for _, a in acts], np.float32),
                phases=np.asarray([p for p, _ in acts]),
            )
            ev = event_times(rows)
            mech = classify_mechanism(ev, br, meta0)
            row = {
                "ic_id": ic_id,
                "branch": br,
                "pre_unload_ok": 1,
                "snap_theta": meta0["theta"],
                "snap_pos_mm": meta0["pos_mm"],
                "snap_s_axial": meta0["mean_s_axial"],
                "snap_ecc_xy_mm": meta0["ecc_xy_mm"],
                "snap_fn": meta0["fn_tot"],
                "snap_v_rel": meta0["v_rel"],
                "mechanism": mech,
                **ev,
            }
            comparison_rows.append(row)
            all_events.append(row)
            print(
                f"  {br}: lose_bil={ev['t_first_lose_bilateral']} "
                f"latch={ev['t_first_latch']} retain={ev['retained_bilateral_all']} "
                f"mech={mech}",
                flush=True,
            )

        # Axial slide probe if eccentric lever missing
        s_mm = abs(float(meta0["mean_s_axial"] or 0.0)) * 1e3
        ecc = float(meta0["ecc_xy_mm"] or 0.0)
        need_slide = (s_mm < ECCENTRIC_MM) and (ecc < ECCENTRIC_MM)
        if need_slide:
            restore_capture(env, cap)
            acts_s, tok_s, desire = build_axial_slide_actions(book, env.sim)
            rows_s = run_branch(env, cap, acts_s)
            for r in rows_s:
                r["ic_id"] = ic_id
                r["branch"] = "axial_slide_reclamp"
            write_csv(OUT / "traces" / f"{ic_id}__axial_slide_reclamp.csv", rows_s)
            ev_s = event_times(rows_s)
            s0 = rows_s[0]["mean_s_axial"] if rows_s else None
            s1 = rows_s[-1]["mean_s_axial"] if rows_s else None
            slide_rows.append({
                "ic_id": ic_id,
                "reason": "insufficient_eccentric_lever",
                "snap_s_axial_mm": s_mm,
                "snap_ecc_xy_mm": ecc,
                "slide_token": tok_s,
                "desire_h": json.dumps(desire),
                "s0": s0,
                "s1": s1,
                "delta_s_mm": None if s0 is None or s1 is None else 1e3 * (float(s1) - float(s0)),
                "retained_bilateral_all": ev_s["retained_bilateral_all"],
                "t_first_lose_bilateral": ev_s["t_first_lose_bilateral"],
                "t_first_latch": ev_s["t_first_latch"],
                "final_fn": ev_s["final_fn"],
                "contact_kept": int(
                    ev_s["retained_bilateral_all"]
                    or (ev_s["t_first_lose_bilateral"] is None and not ev_s["final_latched"])
                ),
                "note": "slide+reclamp only; no pivoting claimed",
            })
            print(
                f"  axial_slide: tok={tok_s} dS={slide_rows[-1]['delta_s_mm']} "
                f"retain={ev_s['retained_bilateral_all']} lose={ev_s['t_first_lose_bilateral']}",
                flush=True,
            )
        else:
            slide_rows.append({
                "ic_id": ic_id,
                "reason": "eccentric_ok_skip_slide",
                "snap_s_axial_mm": s_mm,
                "snap_ecc_xy_mm": ecc,
                "note": "lever above threshold; slide probe skipped",
            })

    write_csv(OUT / "MATCHED_COMPARISON.csv", comparison_rows if comparison_rows else [{"note": "empty"}])
    write_csv(OUT / "AXIAL_SLIDE_PROBE.csv", slide_rows if slide_rows else [{"note": "none"}])
    write_json(OUT / "raw" / "events.json", all_events)

    # Aggregate mechanism conclusions
    by_branch: dict[str, list] = {}
    for r in comparison_rows:
        if not r.get("branch"):
            continue
        by_branch.setdefault(r["branch"], []).append(r)

    def _rate(rows, key, pred):
        if not rows:
            return 0, 0
        n = sum(1 for r in rows if pred(r))
        return n, len(rows)

    lines = []
    lines.append("# Matched CPP Unload Failure Isolation\n")
    lines.append(f"**Script:** `scripts/run_freefall_e_cpp_unload_isolation.py`  ")
    lines.append(f"**Out:** `{OUT.relative_to(ROOT).as_posix()}`  ")
    lines.append(f"**Recipe:** `{args.recipe}` (tilt -> matched pre-unload snapshot -> branches)  ")
    lines.append(f"**Wall:** {time.time() - t0w:.1f}s\n")
    lines.append("No SFT/GRPO. Physics / SUCCESS_NEW / ICs unchanged. Privileged diagnostic.\n")
    lines.append("---\n")
    lines.append("## Setup\n")
    lines.append(
        "From each E stable-lift prefix, run **documented zero-residual tilt**, "
        "save full MuJoCo+FSM capture at tilt-end (pre-unload), then fork matched branches "
        "without retargeting or controller resets.\n"
    )
    lines.append("| Branch | Action |")
    lines.append("|--------|--------|")
    lines.append("| A | Hold end-of-tilt wrist/hand, grip=+1 (no unload) |")
    lines.append("| B | Documented CPP unload tok12 (a6~-0.48, zero residual) + brake |")
    lines.append("| B_prior | Residual-carry unload as in prior transfer script |")
    lines.append("| C | Milder continuous unload a6=-0.15 |")
    lines.append("| D | Ramp a6 +1->-0.48 over 2x unload steps |")
    lines.append("| E | Grip locked +1; wrist/hand leave-tilt then brake |")
    lines.append("")
    lines.append("## Per-branch retain rates\n")
    lines.append("| Branch | Retained bilateral (all steps) | Lost bil | Latched | Dominant mechanism |")
    lines.append("|--------|-------------------------------:|---------:|--------:|--------------------|")
    for br, rows in by_branch.items():
        n_ret, n = _rate(rows, "ret", lambda r: int(r.get("retained_bilateral_all") or 0) == 1)
        n_lose, _ = _rate(rows, "lose", lambda r: r.get("t_first_lose_bilateral") is not None)
        n_lat, _ = _rate(rows, "lat", lambda r: r.get("t_first_latch") is not None)
        mechs = {}
        for r in rows:
            mechs[r.get("mechanism")] = mechs.get(r.get("mechanism"), 0) + 1
        dom = max(mechs, key=mechs.get) if mechs else ""
        lines.append(f"| {br} | {n_ret}/{n} | {n_lose}/{n} | {n_lat}/{n} | `{dom}` |")
    lines.append("")
    lines.append("## Matched comparison\n")
    lines.append("See `MATCHED_COMPARISON.csv` and per-step `traces/{ic}__{branch}.csv`.\n")
    lines.append("## Axial slide -> reclamp probe\n")
    lines.append(
        f"Triggered when pre-unload `|s_axial|` and hand-frame COM lever both "
        f"< {ECCENTRIC_MM} mm (verified CPP ICs had ~8.9-12.5 mm). "
        "Pattern: documented CPS mid(tok12)x5 + secure axial translate x10 + reclamp(tok0). "
        "**No pivoting.**\n"
    )
    lines.append("See `AXIAL_SLIDE_PROBE.csv`.\n")

    # Mechanism verdict
    a_rows = by_branch.get("A", [])
    b_rows = by_branch.get("B", [])
    bp_rows = by_branch.get("B_prior", [])
    c_rows = by_branch.get("C", [])
    d_rows = by_branch.get("D", [])
    e_rows = by_branch.get("E", [])

    a_ok = all(int(r.get("retained_bilateral_all") or 0) for r in a_rows) if a_rows else False
    b_lose = sum(1 for r in b_rows if r.get("t_first_lose_bilateral") is not None)
    bp_lose = sum(1 for r in bp_rows if r.get("t_first_lose_bilateral") is not None)
    c_lose = sum(1 for r in c_rows if r.get("t_first_lose_bilateral") is not None)
    d_lose = sum(1 for r in d_rows if r.get("t_first_lose_bilateral") is not None)
    e_lose = sum(1 for r in e_rows if r.get("t_first_lose_bilateral") is not None)

    lines.append("## Failure mechanism conclusions\n")
    if a_ok and b_lose == len(b_rows) and len(b_rows) > 0:
        lines.append(
            "1. **A retains contact** while **B (documented mid-unload) loses it** -> "
            "failure is caused by the unload grip change itself, not by an already-unstable "
            "pre-unload state or tilt-only dynamics.\n"
        )
    elif not a_ok:
        lines.append(
            "1. **A also loses contact** on some ICs -> pre-unload / tilt state is already "
            "fragile; unload is not the sole cause.\n"
        )
    else:
        lines.append(
            f"1. A retain-all={a_ok}; B lose={b_lose}/{len(b_rows)}; "
            f"B_prior lose={bp_lose}/{len(bp_rows)}.\n"
        )

    if bp_lose >= b_lose and len(bp_rows) > 0:
        lines.append(
            "2. **B_prior (residual-carry)** opens to a6~-0.73 and is at least as bad as "
            "documented mid-unload -- prior transfer script made unload *more* aggressive "
            "than verified CPP (zero-residual a6~-0.48).\n"
        )

    if c_lose < b_lose:
        lines.append(
            "3. **Milder unload (C) loses contact less often than B** -> supports "
            "**normal-force / aperture insufficiency** under mid-open as a primary mechanism.\n"
        )
    elif d_lose < b_lose:
        lines.append(
            "3. **Slower ramp (D) improves retention vs B** -> rate-dependent slip; "
            "instant mid-open shocks the grasp.\n"
        )
    else:
        lines.append(
            "3. C/D do not clearly rescue retention -> mid-open basin may be incompatible "
            "with freefall-E grasp geometry (not only amplitude/rate).\n"
        )

    if e_lose == 0 and a_ok:
        lines.append(
            "4. **E (grip held) retains** -> subsequent wrist/hand schedule alone is not "
            "the contact-loss driver; grip unload is.\n"
        )
    elif e_lose > 0:
        lines.append(
            "4. **E loses contact with grip held** -> wrist/hand motion or controller "
            "transition contributes independently of unload.\n"
        )

    n_slide = sum(1 for r in slide_rows if r.get("reason") == "insufficient_eccentric_lever")
    n_slide_ok = sum(
        1 for r in slide_rows
        if r.get("reason") == "insufficient_eccentric_lever" and int(r.get("contact_kept") or 0)
    )
    lines.append(
        f"5. **Eccentric lever:** {n_slide}/{len(selected)} ICs below {ECCENTRIC_MM} mm "
        f"threshold -> axial slide probe ran; contact kept on {n_slide_ok}/{n_slide}. "
        "If slide cannot create >=8 mm offset while bilateral, pivoting geometry is missing "
        "and must be fixed before retrying CPP unload.\n"
    )

    lines.append("## Minimal next control changes (not implemented this round)\n")
    lines.append(
        "1. Fix CPP decode to **zero-residual** (match verified pilots) -- do not carry "
        "tilt residual into tok12.\n"
        "2. If B fails with correct decode: **do not mid-unload** until an eccentric "
        "grasp exists; insert **axial slide -> reclamp** (CPS-style) and re-check lever.\n"
        "3. If slide retains but lever stays small: freefall-E grasp may be centered -- "
        "need a different privileged slide direction or accept that CPP pivot is "
        "out-of-domain for these ICs.\n"
        "4. Only after contact+lever succeed, re-test mid-unload->pivot; still no "
        "obs-only / SFT / GRPO claim.\n"
    )
    lines.append("## Legality\n")
    lines.append(
        "- Privileged diagnostic (MuJoCo contacts/forces, open-loop 7D).\n"
        "- **Not** a 39D observation-only controller result.\n"
        "- Does **not** claim full 6-stage pipeline or SUCCESS_NEW recovery.\n"
        "- E baseline trajectories from prior experiment untouched.\n"
    )
    lines.append("## Files\n")
    lines.append(
        "- `scripts/run_freefall_e_cpp_unload_isolation.py`\n"
        "- `results/diagnostics/freefall_e_cpp_unload_isolation/**`\n"
    )

    report = "\n".join(lines)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    # Compact mechanism doc
    fail_md = f"""# FAILURE_MECHANISM — CPP Unload Isolation

## Order of failure (from matched traces)

See `MATCHED_COMPARISON.csv` timestamps:
- `t_first_lose_bilateral`, `t_first_zero_finger_contact`, `t_first_latch`, `t_drop_proxy`
- `fn_at_lose_bil`, `v_rel_at_lose_bil`, `aper_at_lose_bil`

## Branch summary

| Branch | lose_bil | notes |
|--------|---------:|-------|
| A | {sum(1 for r in a_rows if r.get('t_first_lose_bilateral') is not None)}/{len(a_rows)} | no unload |
| B | {b_lose}/{len(b_rows)} | documented mid a6~-0.48 |
| B_prior | {bp_lose}/{len(bp_rows)} | residual-carry a6~-0.73 |
| C | {c_lose}/{len(c_rows)} | mild a6=-0.15 |
| D | {d_lose}/{len(d_rows)} | ramp to mid |
| E | {e_lose}/{len(e_rows)} | grip locked |

## Distinctions

- **Normal-force slip:** fn collapses / aperture opens at lose-bil, A stable, E stable.
- **Geometry unfit for pivot:** snap lever << 8 mm; even retained unload would not pivot.
- **Controller transition:** E loses with grip held, or B_prior worse than B due to residual.
- **Other:** A already unstable -> tilt/prefix issue, not unload-only.

## Axial slide probe

{json.dumps(slide_rows, indent=2)}
"""
    (OUT / "FAILURE_MECHANISM.md").write_text(fail_md, encoding="utf-8")

    files_md = """# FILES

| Path | Role |
|------|------|
| REPORT.md | conclusions + next minimal control changes |
| FAILURE_MECHANISM.md | failure order taxonomy |
| MATCHED_COMPARISON.csv | per IC x branch event summary |
| AXIAL_SLIDE_PROBE.csv | slide->reclamp (no pivot) |
| traces/*.csv | per-step forces/contact/kinematics |
| snapshots/*__pre_unload.* | matched tilt-end captures |
| raw/BUDGETS.json | decode notes + thresholds |
"""
    (OUT / "FILES.md").write_text(files_md, encoding="utf-8")
    try:
        print(report, flush=True)
    except UnicodeEncodeError:
        print(report.encode("ascii", "replace").decode("ascii"), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
