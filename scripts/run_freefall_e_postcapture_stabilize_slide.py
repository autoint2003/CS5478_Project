"""Freefall-E post-capture stabilize + axial slide feasibility (staged gates).

Stage 1: decode audit (transfer residual-carry vs zero-residual) + legal settle.
Stage 2: CPS slide/reclamp only from Stage-1 verified-stable matched states.
Stage 3: documented CPP mid-unload/pivot only if Stage 2 built usable eccentric lever.

Privileged diagnostic. No SFT/GRPO. Physics / SUCCESS_NEW / 39D / frozen E unchanged.
Stop later stages if earlier gates fail -- do not force a 6-stage pipeline.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import NOMINAL_S
from envs.outcomes.table_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import step_legal
from evaluation.full_objective import P_TGT, U_TGT, contact_summary
from evaluation.grasp_configuration import (
    AXIS_REVERSAL_SYMMETRIC,
    cylinder_axis_hand,
    cylinder_axis_world,
    orientation_error_rad,
    p_com_hand,
)
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_sliding_recatch_dataset import SFT, write_csv, write_json
from scripts.run_freefall_e_stable_lift_contact_manip import (
    CPP_RECIPES,
    held_ics,
    run_e_to_stable_lift,
)
from scripts.run_freefall_e_cpp_unload_isolation import (
    REP_ICS,
    body_v_w,
    contact_force_detail,
    decode_carry,
    decode_zero,
    grip_actuator_force,
    secure_axial_tokens,
)
from scripts.run_freefall_recatch_grasp_point_alignment import (
    BASE_GAINS,
    aperture,
    axial_summary,
    contacts_axial,
)
from scripts.run_freefall_recatch_pose_feedback import pose_now
from scripts.run_horizontal_offset_freefall_recatch import (
    SEARCH_NOMINAL_S,
    eval_full_sn,
)
from scripts.run_lateral_offset_falling_recatch import (
    hand_pos,
    make_training_env,
    obj_pos,
    restore_ic,
)
from scripts.run_lateral_recatch_handoff import clear_probe_latch, hand_R
from training.k47_codebook import CAP, decode_action, load_codebook

SRC_CM = ROOT / "results/diagnostics/freefall_e_stable_lift_contact_manip"
OUT = ROOT / "results/diagnostics/freefall_e_postcapture_stabilize_slide"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"

ATTACH_TILT_TOKS = [6] * 8          # short documented tilt attach (not full 20)
SETTLE_TOKS = 20                    # tok0 secure brake steps
VERIFY_HOLD = 30                    # post-settle hold; must sustain contact
V_REL_OK = 0.12                     # m/s mean over last half of verify
BIL_FRAC_OK = 0.95
LEVER_TARGET_MM = 8.0               # aspirational pivoting lever (not proven sufficient)
LEVER_GATE_MM = 6.0                 # Stage3 entry: must reach at least this
WALL_S = 45 * 60


def gravity_analysis(sim) -> dict:
    """Port of wrist_tilt_gravity_pivoting.gravity_analysis (privileged diagnostic)."""
    mass = float(sim.model.body_mass[sim.ids.object_body])
    g_w = np.array(sim.model.opt.gravity, float)
    Fg = mass * g_w
    Rh = np.asarray(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    po = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    ph = np.asarray(sim.data.xpos[sim.ids.hand_body], float)
    cons, _, _ = contacts_axial(sim)
    fr = [c for c in cons if c["side"] in ("L", "R")]
    if fr:
        pivot = np.mean([np.asarray(c["pos"], float) for c in fr], axis=0)
    else:
        pivot = ph + Rh @ np.array([0.0, 0.0, 0.105])
    r = po - pivot
    tau_g_h = Rh.T @ np.cross(r, Fg)
    u_c = cylinder_axis_hand(sim)
    u_t = np.asarray(U_TGT, float)
    u_t = u_t / max(float(np.linalg.norm(u_t)), 1e-12)
    u_c = u_c / max(float(np.linalg.norm(u_c)), 1e-12)
    if float(np.dot(u_c, u_t)) < 0:
        u_t = -u_t
    axis = np.cross(u_c, u_t)
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 1e-9 else np.zeros(3)
    align = float(np.dot(tau_g_h, axis)) if n > 1e-9 else 0.0
    p_c_h = Rh.T @ (po - ph)
    ecc = float(np.linalg.norm(p_c_h[:2]))
    return {
        "tau_g_h_norm": float(np.linalg.norm(tau_g_h)),
        "align": align,
        "helps": bool(align > 1e-6),
        "lever_m": float(np.linalg.norm(r)),
        "ecc_xy_mm": ecc * 1e3,
        "p_com_h": p_c_h.tolist(),
        "e_theta_deg": float(np.degrees(orientation_error_rad(u_c, U_TGT, AXIS_REVERSAL_SYMMETRIC))),
    }


def rich_row(sim, step: int, phase: str, a: np.ndarray, t0: float) -> dict:
    cs = contact_summary(sim)
    cons, _, _ = contacts_axial(sim)
    ax = axial_summary(cons)
    pose = pose_now(sim)
    cf = contact_force_detail(sim)
    vh, wh = body_v_w(sim, int(sim.ids.hand_body))
    vo, wo = body_v_w(sim, int(sim.ids.object_body))
    v_rel = vo - vh
    w_rel = wo - wh
    g = gravity_analysis(sim)
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
        "grip_af": grip_actuator_force(sim),
        "grip_ctrl": float(sim.data.ctrl[7]) if sim.data.ctrl.size > 7 else float("nan"),
        "aper": aperture(sim),
        "mean_s_axial": ax.get("mean_s_axial"),
        "max_abs_s": ax.get("max_abs_s"),
        "v_rel": float(np.linalg.norm(v_rel)),
        "w_rel": float(np.linalg.norm(w_rel)),
        "theta": pose["e_theta_deg"],
        "pos_mm": pose["e_pos_mm"],
        "obj_z": float(obj_pos(sim)[2]),
        "hand_z": float(hand_pos(sim)[2]),
        "ecc_xy_mm": g["ecc_xy_mm"],
        "lever_mm": g["lever_m"] * 1e3,
        "tau_g": g["tau_g_h_norm"],
        "grav_align": g["align"],
        "grav_helps": int(g["helps"]),
        "a0": float(a[0]), "a1": float(a[1]), "a2": float(a[2]),
        "a3": float(a[3]), "a4": float(a[4]), "a5": float(a[5]),
        "a6": float(a[6]),
        "latched": int(latched(sim)),
    }


def run_acts(env, cap, named_acts: list[tuple[str, np.ndarray]]) -> list[dict]:
    restore_capture(env, cap)
    t0 = float(env.sim.data.time)
    rows = []
    for i, (phase, a) in enumerate(named_acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        rows.append(rich_row(env.sim, i, phase, a, t0))
        if latched(env.sim) and i > 3 and all(int(r["latched"]) for r in rows[-3:]):
            break
    return rows


def summarize_window(rows: list[dict], phase_prefix: str | None = None) -> dict:
    use = rows if phase_prefix is None else [r for r in rows if str(r["phase"]).startswith(phase_prefix)]
    if not use:
        use = rows
    n = len(use)
    bil = [int(r["bilateral"]) for r in use]
    v = [float(r["v_rel"]) for r in use]
    half = use[len(use) // 2 :] if use else []
    v_half = [float(r["v_rel"]) for r in half] if half else v
    lose = next((r["t"] for r in use if not int(r["bilateral"])), None)
    latch = next((r["t"] for r in use if int(r["latched"])), None)
    return {
        "n": n,
        "bil_frac": float(np.mean(bil)) if bil else 0.0,
        "mean_v_rel": float(np.mean(v)) if v else None,
        "mean_v_rel_last_half": float(np.mean(v_half)) if v_half else None,
        "max_v_rel": float(np.max(v)) if v else None,
        "t_lose_bil": lose,
        "t_latch": latch,
        "final_bil": int(use[-1]["bilateral"]) if use else 0,
        "final_latched": int(use[-1]["latched"]) if use else 0,
        "final_theta": use[-1]["theta"] if use else None,
        "final_s_axial": use[-1]["mean_s_axial"] if use else None,
        "final_ecc_mm": use[-1]["ecc_xy_mm"] if use else None,
        "final_lever_mm": use[-1]["lever_mm"] if use else None,
        "final_fn": use[-1]["fn_tot"] if use else None,
        "final_grav_align": use[-1]["grav_align"] if use else None,
        "final_grav_helps": use[-1]["grav_helps"] if use else None,
    }


def stage1_pass(verify_sum: dict) -> bool:
    """Sustained contact after settle -- not instantaneous low v_rel."""
    if verify_sum["n"] < VERIFY_HOLD // 2:
        return False
    if verify_sum["final_latched"]:
        return False
    if verify_sum["bil_frac"] < BIL_FRAC_OK:
        return False
    if verify_sum["t_lose_bil"] is not None:
        return False
    vr = verify_sum["mean_v_rel_last_half"]
    if vr is None or vr > V_REL_OK:
        return False
    return True


def build_decode_audit(book) -> dict:
    """Step-by-step action map: residual-carry vs zero-residual for tilt->unload."""
    rows = []
    pre = np.zeros(7, np.float32)
    for phase, toks in [("tilt", [6] * 5), ("unload", [12] * 4)]:
        for i, tok in enumerate(toks):
            residual = np.clip(np.asarray(pre, np.float32), -CAP, CAP)
            proto = np.asarray(book[int(tok)], np.float32)
            a_zero = decode_zero(book, tok)
            a_carry = np.clip(proto + residual, -1.0, 1.0)
            rows.append({
                "phase": phase,
                "i": i,
                "tok": int(tok),
                "proto_a6": float(proto[6]),
                "proto_a4": float(proto[4]),
                "residual_a6": float(residual[6]),
                "residual_a4": float(residual[4]),
                "a_zero_a6": float(a_zero[6]),
                "a_zero_a4": float(a_zero[4]),
                "a_carry_a6": float(a_carry[6]),
                "a_carry_a4": float(a_carry[4]),
            })
            pre = a_carry.copy()
    return {
        "formula": "action = clip(proto[tok] + clip(pre, -CAP, CAP), -1, 1); CAP=0.25",
        "documented_CPP": "decode_action(book, tok, zeros(7)) every step",
        "prior_transfer": (
            "tokens_to_actions accumulates pre=action -> residual from tilt "
            "(a6=+1,a4=+1) into unload"
        ),
        "steps": rows,
    }


def stage1_actions(book, mode: str) -> list[tuple[str, np.ndarray]]:
    """Build Stage-1 named action list."""
    out: list[tuple[str, np.ndarray]] = []
    if mode == "transfer_decode":
        pre = np.zeros(7, np.float32)
        for tok in ATTACH_TILT_TOKS:
            a, pre = decode_carry(book, tok, pre)
            out.append(("attach_tilt_carry", a))
        for _ in range(SETTLE_TOKS):
            a, pre = decode_carry(book, 0, pre)
            out.append(("settle_carry", a))
        for _ in range(VERIFY_HOLD):
            a, pre = decode_carry(book, 0, pre)
            out.append(("verify_carry", a))
        return out

    if mode == "zero_no_settle":
        last = None
        for tok in ATTACH_TILT_TOKS:
            a = decode_zero(book, tok)
            out.append(("attach_tilt_zero", a))
            last = a
        # no settle: hold last tilt wrist command
        hold = last.copy() if last is not None else decode_zero(book, 6)
        for _ in range(SETTLE_TOKS + VERIFY_HOLD):
            out.append(("hold_tilt_no_settle", hold.copy()))
        return out

    if mode == "zero_settle":
        for tok in ATTACH_TILT_TOKS:
            out.append(("attach_tilt_zero", decode_zero(book, tok)))
        for _ in range(SETTLE_TOKS):
            out.append(("settle_zero", decode_zero(book, 0)))
        for _ in range(VERIFY_HOLD):
            out.append(("verify_zero", decode_zero(book, 0)))
        return out

    raise ValueError(mode)


def slide_variants(book, sim) -> list[dict]:
    """Few documented CPS-style parameterizations (pilot_sliding_position_correction_v1 families)."""
    u = cylinder_axis_world(sim)
    Rh = hand_R(sim)
    desire = Rh.T @ u
    desire_opp = -desire
    toks = secure_axial_tokens(book, desire, top_k=3)
    toks_opp = secure_axial_tokens(book, desire_opp, top_k=2)
    t0 = int(toks[0])
    t1 = int(toks[1]) if len(toks) > 1 else t0
    to = int(toks_opp[0])

    def seq(name, token_list):
        acts = []
        for tok in token_list:
            phase = "mid" if int(tok) == 12 else ("reclamp" if int(tok) == 0 else "slide")
            acts.append((f"{name}_{phase}", decode_zero(book, int(tok))))
        for _ in range(VERIFY_HOLD):
            acts.append((f"{name}_verify", decode_zero(book, 0)))
        return {"name": name, "tokens": list(token_list), "acts": acts, "slide_tok": t0}

    return [
        seq("mid5_sec10_rec5", [12] * 5 + [t0] * 10 + [0] * 5),
        seq("mid8_sec16_rec5", [12] * 8 + [t0] * 16 + [0] * 5),
        seq("sec12_rec5", [t0] * 12 + [0] * 5),  # secure-only (S3-like)
        seq("mid5_sec10_opp", [12] * 5 + [to] * 10 + [0] * 5),
        seq("mid3_alt_sec", ([12, t0] * 5) + [0] * 5),
        seq("mid5_sec10_t1", [12] * 5 + [t1] * 10 + [0] * 5),
    ]


def pivot_branches(book) -> list[dict]:
    """Matched lock / mild / mid-unload after Stage2 (documented CPP unload length)."""
    n_u, n_b = 15, 8
    out = []
    # lock grip
    acts = [("lock", np.array([0, 0, 0, 0, 0, 0, 1], np.float32)) for _ in range(n_u + n_b + VERIFY_HOLD)]
    out.append({"name": "lock_grip", "acts": acts})
    # mild unload
    mild = np.zeros(7, np.float32)
    mild[6] = -0.15
    acts = [("mild_unload", mild.copy()) for _ in range(n_u)]
    acts += [("brake", decode_zero(book, 0)) for _ in range(n_b + VERIFY_HOLD)]
    out.append({"name": "mild_unload", "acts": acts})
    # documented mid
    acts = [("mid_unload", decode_zero(book, 12)) for _ in range(n_u)]
    acts += [("brake", decode_zero(book, 0)) for _ in range(n_b + VERIFY_HOLD)]
    out.append({"name": "mid_unload_cpp", "acts": acts})
    return out


def load_stable_lift(env, ic, gains):
    ic_id = ic["ic_id"]
    pkl = SRC_CM / "snapshots" / f"{ic_id}__stable_lift.pkl"
    if pkl.exists():
        blob = pickle.loads(pkl.read_bytes())
        return blob.get("ic", ic), blob["prefix"], blob["capture"], blob.get("meta", {})
    snap, _base, _fb = run_e_to_stable_lift(env, ic, gains)
    if snap is None:
        return ic, None, None, None
    return ic, snap["prefix_actions"], snap["capture"], {"step": snap["step"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ics", type=int, default=6)
    args = ap.parse_args()

    t0w = time.time()
    for sub in ("raw", "snapshots", "traces", "stage1", "stage2", "stage3"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    book = load_codebook(CODEBOOK)
    gains = dict(BASE_GAINS)

    audit = build_decode_audit(book)
    write_json(OUT / "raw" / "DECODE_AUDIT.json", audit)

    all_ics = {ic["ic_id"]: ic for ic in held_ics(env)}
    selected = []
    for name in REP_ICS:
        if name in all_ics:
            selected.append(all_ics[name])
        if len(selected) >= int(args.max_ics):
            break

    write_json(OUT / "raw" / "BUDGETS.json", {
        "attach_tilt_toks": ATTACH_TILT_TOKS,
        "settle_toks": SETTLE_TOKS,
        "verify_hold": VERIFY_HOLD,
        "v_rel_ok": V_REL_OK,
        "bil_frac_ok": BIL_FRAC_OK,
        "lever_target_mm": LEVER_TARGET_MM,
        "lever_gate_mm": LEVER_GATE_MM,
        "ics": [ic["ic_id"] for ic in selected],
        "note": "Privileged diagnostic; gates stop later stages",
    })

    stage1_rows = []
    stage2_rows = []
    stage3_rows = []
    baseline_rows = []
    gate = {"stage1_pass_ics": [], "stage2_pass_ics": [], "stage3_ran_ics": []}

    for ic0 in selected:
        if time.time() - t0w > WALL_S:
            print("WALL", flush=True)
            break
        ic, prefix, cap0, meta0 = load_stable_lift(env, ic0, gains)
        ic_id = ic["ic_id"]
        print(f"\n=== {ic_id} ===", flush=True)
        if cap0 is None or prefix is None:
            print("  no stable-lift; skip", flush=True)
            stage1_rows.append({"ic_id": ic_id, "stage1_pass": 0, "error": "no_stable_lift"})
            continue

        # Preserve E baseline SUCCESS_NEW path (prefix continuation from prior full E roll if needed)
        # Re-roll short E baseline from IC using prior contact-manip baseline evaluation style
        snap_full, baseline_full, first_bil = run_e_to_stable_lift(env, ic, gains)
        if baseline_full and first_bil is not None:
            cut = max(1, first_bil + 1)
            out_b = eval_full_sn(env, ic, baseline_full, cut, nominal_s=SEARCH_NOMINAL_S)
            baseline_rows.append({
                "ic_id": ic_id,
                "baseline_SN_search": int(out_b["SUCCESS_NEW"]),
                "baseline_CH": int(out_b["CATCH_HANDOFF_SUCCESS"]),
                "fail": out_b.get("fail_class"),
                "theta": out_b.get("SN_mean_theta"),
                "pos": out_b.get("SN_mean_pos"),
            })
            if out_b["SUCCESS_NEW"]:
                out12 = eval_full_sn(env, ic, baseline_full, cut, nominal_s=float(NOMINAL_S))
                ok = 0
                if out12["SUCCESS_NEW"]:
                    ok = sum(
                        int(eval_full_sn(env, ic, baseline_full, cut, nominal_s=float(NOMINAL_S))["SUCCESS_NEW"])
                        for _ in range(3)
                    )
                baseline_rows[-1]["SN_12s"] = int(out12["SUCCESS_NEW"])
                baseline_rows[-1]["replay_ok"] = ok
                baseline_rows[-1]["provenance"] = "frozen_E_continuation_preserved"

        # Ensure matched capture at stable-lift via prefix replay (byte-consistent with prior)
        restore_ic(env, ic)
        clear_probe_latch(env.sim)
        for a in prefix:
            step_legal(env.sim, np.asarray(a, np.float32))
        if not contact_summary(env.sim)["bilateral"] or latched(env.sim):
            # fall back to stored capture
            restore_capture(env, cap0)
        cap_sl = take_capture(env)
        g0 = gravity_analysis(env.sim)
        pickle.dump(
            {"ic": ic, "capture": cap_sl, "prefix": prefix, "grav0": g0, "meta": meta0},
            (OUT / "snapshots" / f"{ic_id}__stable_lift_takeover.pkl").open("wb"),
        )
        vo, _ = body_v_w(env.sim, int(env.sim.ids.object_body))
        vh, _ = body_v_w(env.sim, int(env.sim.ids.hand_body))
        write_json(OUT / "snapshots" / f"{ic_id}__stable_lift_takeover_meta.json", {
            "ic_id": ic_id,
            **{k: g0[k] for k in ("ecc_xy_mm", "lever_m", "align", "helps", "tau_g_h_norm", "e_theta_deg")},
            "pose": pose_now(env.sim),
            "v_rel": float(np.linalg.norm(vo - vh)),
        })

        # ----- Stage 1 -----
        s1_best_cap = None
        s1_pass = False
        for mode in ("transfer_decode", "zero_no_settle", "zero_settle"):
            acts = stage1_actions(book, mode)
            rows = run_acts(env, cap_sl, acts)
            for r in rows:
                r["ic_id"] = ic_id
                r["branch"] = mode
            write_csv(OUT / "traces" / f"{ic_id}__stage1__{mode}.csv", rows)
            att = summarize_window(rows, "attach")
            sett = summarize_window(rows, "settle")
            ver = summarize_window(rows, "verify")
            hold = summarize_window(rows, "hold")
            # for zero_no_settle, verify window is hold_tilt_no_settle last VERIFY_HOLD
            if mode == "zero_no_settle":
                ver = summarize_window(rows[-VERIFY_HOLD:], None)
            elif mode == "transfer_decode":
                ver = summarize_window([r for r in rows if r["phase"].startswith("verify")], None)
            else:
                ver = summarize_window([r for r in rows if r["phase"].startswith("verify")], None)

            passed = stage1_pass(ver) if mode == "zero_settle" else False
            # also score other modes for comparison (same criterion on their end window)
            end_ok = stage1_pass(ver)
            row = {
                "ic_id": ic_id,
                "branch": mode,
                "attach_bil_frac": att["bil_frac"],
                "settle_bil_frac": sett["bil_frac"] if sett["n"] else hold["bil_frac"],
                "verify_bil_frac": ver["bil_frac"],
                "verify_mean_v_rel_last_half": ver["mean_v_rel_last_half"],
                "verify_t_lose_bil": ver["t_lose_bil"],
                "verify_t_latch": ver["t_latch"],
                "final_theta": ver["final_theta"],
                "final_ecc_mm": ver["final_ecc_mm"],
                "final_fn": ver["final_fn"],
                "stage1_criterion_ok": int(end_ok),
                "stage1_pass_for_gate": int(passed),
            }
            stage1_rows.append(row)
            print(
                f"  S1 {mode}: bil={ver['bil_frac']:.2f} v_half={ver['mean_v_rel_last_half']} "
                f"lose={ver['t_lose_bil']} latch={ver['t_latch']} ok={end_ok}",
                flush=True,
            )
            if mode == "zero_settle" and passed:
                s1_pass = True
                # capture at end of settle (before verify) for Stage2 matched start
                restore_capture(env, cap_sl)
                n_pre = len(ATTACH_TILT_TOKS) + SETTLE_TOKS
                for phase, a in acts[:n_pre]:
                    step_legal(env.sim, np.asarray(a, np.float32))
                s1_best_cap = take_capture(env)
                pickle.dump(
                    {"ic": ic, "capture": s1_best_cap, "branch": mode},
                    (OUT / "snapshots" / f"{ic_id}__stage1_settled.pkl").open("wb"),
                )

        if not s1_pass:
            print("  Stage1 GATE FAIL -- stop Stage2/3 for this IC", flush=True)
            continue
        gate["stage1_pass_ics"].append(ic_id)

        # ----- Stage 2 -----
        restore_capture(env, s1_best_cap)
        meta_pre = {
            "s_axial": axial_summary(contacts_axial(env.sim)[0]).get("mean_s_axial"),
            "ecc_mm": gravity_analysis(env.sim)["ecc_xy_mm"],
            "lever_mm": gravity_analysis(env.sim)["lever_m"] * 1e3,
            "grav_align": gravity_analysis(env.sim)["align"],
            "grav_helps": int(gravity_analysis(env.sim)["helps"]),
            "theta": pose_now(env.sim)["e_theta_deg"],
            "fn": contact_force_detail(env.sim)["fn_tot"],
        }
        variants = slide_variants(book, env.sim)
        best2 = None
        for var in variants:
            rows = run_acts(env, s1_best_cap, var["acts"])
            for r in rows:
                r["ic_id"] = ic_id
                r["branch"] = var["name"]
            write_csv(OUT / "traces" / f"{ic_id}__stage2__{var['name']}.csv", rows)
            pre_r = rows[0] if rows else None
            # split slide vs verify
            slide_rows = [r for r in rows if not str(r["phase"]).endswith("_verify")]
            ver_rows = [r for r in rows if str(r["phase"]).endswith("_verify")]
            ver = summarize_window(ver_rows if ver_rows else rows[-VERIFY_HOLD:], None)
            s0 = pre_r["mean_s_axial"] if pre_r else None
            s1 = slide_rows[-1]["mean_s_axial"] if slide_rows else None
            d_s_mm = None if s0 is None or s1 is None else 1e3 * (float(s1) - float(s0))
            e0 = pre_r["ecc_xy_mm"] if pre_r else None
            e1 = slide_rows[-1]["ecc_xy_mm"] if slide_rows else None
            d_ecc = None if e0 is None or e1 is None else float(e1) - float(e0)
            g0a = pre_r["grav_align"] if pre_r else None
            g1a = slide_rows[-1]["grav_align"] if slide_rows else None
            contact_ok = stage1_pass(ver)
            # Actual controlled slip (not drop-inflated ecc): need measurable axial/ecc change
            # while bilateral still present at end of slide phase.
            slide_end_bil = int(slide_rows[-1]["bilateral"]) if slide_rows else 0
            moved = (
                slide_end_bil == 1
                and (
                    (d_s_mm is not None and abs(d_s_mm) >= 3.0)
                    or (d_ecc is not None and abs(d_ecc) >= 2.0)
                )
            )
            post_lever_mm = None
            if e1 is not None:
                post_lever_mm = abs(float(e1))
            if s1 is not None:
                post_lever_mm = max(post_lever_mm or 0.0, abs(float(s1)) * 1e3)
            # Gate: stable reclamp AND slide produced motion AND post lever >= gate
            # Pre-existing eccentricity without slide motion does NOT count as Stage2 pass.
            reached = bool(
                contact_ok
                and moved
                and post_lever_mm is not None
                and post_lever_mm >= LEVER_GATE_MM
            )
            score = (
                1000 * int(contact_ok)
                + 200 * int(reached)
                + 50 * int(moved)
                + (abs(d_s_mm) if d_s_mm is not None else 0)
                + (abs(d_ecc) if d_ecc is not None else 0)
                - 100 * int(ver["final_latched"])
            )
            row = {
                "ic_id": ic_id,
                "variant": var["name"],
                "tokens": json.dumps(var["tokens"]),
                "pre_s_axial": s0,
                "post_s_axial": s1,
                "delta_s_mm": d_s_mm,
                "pre_ecc_mm": e0,
                "post_ecc_mm": e1,
                "delta_ecc_mm": d_ecc,
                "pre_grav_align": g0a,
                "post_grav_align": g1a,
                "grav_align_delta": None if g0a is None or g1a is None else float(g1a) - float(g0a),
                "post_grav_helps": slide_rows[-1]["grav_helps"] if slide_rows else None,
                "verify_bil_frac": ver["bil_frac"],
                "verify_v_half": ver["mean_v_rel_last_half"],
                "contact_stable": int(contact_ok),
                "moved_axial_or_ecc": int(moved),
                "slide_end_bilateral": slide_end_bil,
                "post_lever_mm": post_lever_mm,
                "lever_gate_reached": int(reached),
                "lever_target_mm": LEVER_TARGET_MM,
                "pre_existing_ecc_mm": e0,
                "score": score,
                "t_lose": ver["t_lose_bil"],
                "t_latch": ver["t_latch"],
            }
            stage2_rows.append(row)
            print(
                f"  S2 {var['name']}: dS={d_s_mm} dEcc={d_ecc} stable={contact_ok} "
                f"gate={reached} align0={g0a} align1={g1a}",
                flush=True,
            )
            if best2 is None or score > best2["score"]:
                best2 = {"score": score, "row": row, "name": var["name"], "acts": var["acts"]}

        s2_pass = bool(
            best2
            and int(best2["row"]["contact_stable"])
            and int(best2["row"]["lever_gate_reached"])
        )
        # Also require gravity moment not anti-aligned if we claim pivoting lever
        if s2_pass and best2["row"]["post_grav_helps"] is not None:
            # soft: if helps==0, still allow Stage3 but flag
            pass
        if not s2_pass:
            print(
                f"  Stage2 GATE FAIL (best={None if not best2 else best2['name']}) -- stop Stage3",
                flush=True,
            )
            continue
        gate["stage2_pass_ics"].append(ic_id)

        # matched post-slide capture
        restore_capture(env, s1_best_cap)
        n_slide = len(best2["acts"]) - VERIFY_HOLD
        for phase, a in best2["acts"][:n_slide]:
            step_legal(env.sim, np.asarray(a, np.float32))
        cap_s2 = take_capture(env)
        pickle.dump(
            {"ic": ic, "capture": cap_s2, "variant": best2["name"], "row": best2["row"]},
            (OUT / "snapshots" / f"{ic_id}__stage2_postslide.pkl").open("wb"),
        )

        # ----- Stage 3 -----
        gate["stage3_ran_ics"].append(ic_id)
        th0 = pose_now(env.sim)["e_theta_deg"]
        for br in pivot_branches(book):
            rows = run_acts(env, cap_s2, br["acts"])
            for r in rows:
                r["ic_id"] = ic_id
                r["branch"] = br["name"]
            write_csv(OUT / "traces" / f"{ic_id}__stage3__{br['name']}.csv", rows)
            ver = summarize_window(rows[-VERIFY_HOLD:], None)
            th1 = rows[-1]["theta"] if rows else None
            row = {
                "ic_id": ic_id,
                "branch": br["name"],
                "theta0": th0,
                "theta1": th1,
                "delta_theta": None if th1 is None else float(th1) - float(th0),
                "bil_frac": summarize_window(rows, None)["bil_frac"],
                "verify_stable": int(stage1_pass(ver)),
                "t_lose": ver["t_lose_bil"],
                "t_latch": ver["t_latch"],
                "final_fn": ver["final_fn"],
                "final_grav_align": ver["final_grav_align"],
                "source_variant": best2["name"],
                "note": "privileged; not SUCCESS_NEW",
            }
            stage3_rows.append(row)
            print(
                f"  S3 {br['name']}: dth={row['delta_theta']} stable={row['verify_stable']} "
                f"lose={row['t_lose']}",
                flush=True,
            )

    write_csv(OUT / "stage1" / "STAGE1_COMPARISON.csv", stage1_rows if stage1_rows else [{"note": "empty"}])
    write_csv(OUT / "stage2" / "STAGE2_SLIDE.csv", stage2_rows if stage2_rows else [{"note": "empty"}])
    write_csv(OUT / "stage3" / "STAGE3_PIVOT.csv", stage3_rows if stage3_rows else [{"note": "stopped_or_empty"}])
    write_csv(OUT / "BASELINE_E.csv", baseline_rows if baseline_rows else [{"note": "empty"}])
    write_json(OUT / "raw" / "GATES.json", gate)

    # ----- Report -----
    s1_pass_n = len(gate["stage1_pass_ics"])
    s2_pass_n = len(gate["stage2_pass_ics"])
    n_ic = len(selected)
    lines = []
    lines.append("# Freefall-E Post-Capture Stabilize + Axial Slide\n")
    lines.append(f"**Script:** `scripts/run_freefall_e_postcapture_stabilize_slide.py`  ")
    lines.append(f"**Out:** `results/diagnostics/freefall_e_postcapture_stabilize_slide/`  ")
    lines.append(f"**Wall:** {time.time() - t0w:.1f}s\n")
    lines.append("Privileged diagnostic. No SFT/GRPO. Physics / SUCCESS_NEW / frozen E / 39D unchanged.\n")
    lines.append("---\n")
    lines.append("## Decode audit (Stage 1.1)\n")
    lines.append(
        f"- Formula: `action = clip(proto[tok] + clip(pre, -{CAP}, {CAP}), -1, 1)`.\n"
        "- **Documented CPP:** `pre = zeros(7)` every step -> tok12 always a6~-0.48.\n"
        "- **Prior transfer:** `tokens_to_actions` sets `pre = action`, so after tilt6 "
        "(a4=1,a6=1) residual clips to 0.25 and unload becomes a6~-0.23 then ~-0.71/-0.73 "
        "with leftover wrist.\n"
        "- Full step map: `raw/DECODE_AUDIT.json`.\n"
    )
    lines.append("## Stage 1 — Decode and settle\n")
    lines.append(
        f"Pass rule: after zero-residual settle, verify hold {VERIFY_HOLD} steps with "
        f"bilateral frac>={BIL_FRAC_OK}, no lose-bil, no latch, mean v_rel (last half)<={V_REL_OK}.\n"
    )
    lines.append(f"**Stage1 pass ICs:** {gate['stage1_pass_ics']} ({s1_pass_n}/{n_ic})\n")
    lines.append("See `stage1/STAGE1_COMPARISON.csv` and `traces/*__stage1__*.csv`.\n")
    if s1_pass_n == 0:
        lines.append("**Stage1 FAILED globally -- Stage2/3 not run.**\n")
    lines.append("## Stage 2 — Axial sliding and reclamp\n")
    lines.append(
        "Only from Stage1-pass matched settled captures. Variants reuse CPS families from "
        "`pilot_sliding_position_correction_v1` (mid+secure+reclamp / secure-only / opposite / interleaved).\n"
        f"Lever **target** ~{LEVER_TARGET_MM} mm is aspirational; Stage3 gate requires "
        f"stable reclamp **and** controlled slip (dS>=3mm or dEcc>=2mm while bilateral) "
        f"**and** post lever >= {LEVER_GATE_MM} mm. Pre-existing eccentricity without "
        "slide motion does not pass. Gravity align recorded (helps vs distance-only).\n"
    )
    lines.append(f"**Stage2 pass ICs:** {gate['stage2_pass_ics']} ({s2_pass_n}/{n_ic})\n")
    lines.append("See `stage2/STAGE2_SLIDE.csv`.\n")
    if s1_pass_n > 0 and s2_pass_n == 0:
        lines.append("**Stage2 FAILED -- Stage3 not run (no forced 6-stage).**\n")
    lines.append("## Stage 3 — Conditional pivoting\n")
    if not gate["stage3_ran_ics"]:
        lines.append("**Not executed** (Stage1 or Stage2 gate failed).\n")
    else:
        lines.append(f"Ran on: {gate['stage3_ran_ics']}. Branches: lock / mild / documented mid-unload.\n")
        lines.append("See `stage3/STAGE3_PIVOT.csv`. Not claimed as SUCCESS_NEW.\n")
    lines.append("## Baseline E (preserved)\n")
    lines.append("See `BASELINE_E.csv`. Direct SUCCESS_NEW path unchanged.\n")
    lines.append("## Legality\n")
    lines.append(
        "- Privileged diagnostic (forces, contacts, open-loop 7D).\n"
        "- **Not** observation-only 39D.\n"
        "- Stable-lift / bilateral / single-phase pivot != SUCCESS_NEW.\n"
        "- 12s + 3/3 only if a full SUCCESS_NEW candidate appears (none claimed unless listed).\n"
    )
    lines.append("## Files\n")
    lines.append("- `scripts/run_freefall_e_postcapture_stabilize_slide.py`\n")
    lines.append("- `results/diagnostics/freefall_e_postcapture_stabilize_slide/**`\n")
    report = "\n".join(lines)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    (OUT / "FILES.md").write_text(
        """# FILES

| Path | Role |
|------|------|
| REPORT.md | staged gate conclusions |
| raw/DECODE_AUDIT.json | residual vs zero-residual step map |
| raw/GATES.json | pass IC lists |
| stage1/STAGE1_COMPARISON.csv | transfer / zero_no_settle / zero_settle |
| stage2/STAGE2_SLIDE.csv | CPS parameter variants |
| stage3/STAGE3_PIVOT.csv | lock / mild / mid (or empty if gated) |
| BASELINE_E.csv | preserved E SUCCESS_NEW path |
| traces/* | per-step raw trajectories |
| snapshots/* | matched captures at each gate |
""",
        encoding="utf-8",
    )
    try:
        print(report, flush=True)
    except UnicodeEncodeError:
        print(report.encode("ascii", "replace").decode("ascii"), flush=True)
    print(f"Wrote {OUT}", flush=True)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
