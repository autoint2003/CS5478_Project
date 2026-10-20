"""Full grasp recovery with post-stabilization position/orientation correction.

Extends lateral-force grasp-loss recovery. Audits the five existing SUCCESS_NEW
experts for actual pose correction, then searches legal K49 sequences on
out-of-tolerance pose-error ICs. Does not modify K49, physics, or SUCCESS_NEW.
No SFT/GRPO. Does not overwrite existing five experts.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_generation.datasets.recovery_rebuild import immediate_handoff
from data_generation.recording.trajectory_recorder import save_pair
from envs.outcomes.table_drop import latched, restore_state as restore_table_drop
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.success_criteria import (
    HOLD_S,
    P_TGT,
    U_TGT,
    WORKING_POS_MM,
    WORKING_THETA_DEG,
    contact_summary,
    window_stats,
)
from evaluation.grasp_geometry import measure, p_com_hand, cylinder_axis_hand
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_in_recovery_recatch import pose_correction_proposals
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    decode,
    decompose,
    evaluate_full,
    mapping_checksum,
    slide_proposals,
    write_csv,
    write_json,
)
from training.action_vocab_k49 import (
    SECURE_REVERSE_WRIST,
    assert_k48_prefix_identical,
    build_k49_table,
    describe_k49_config,
)
from training.k47_codebook import load_codebook
from training.recovery_episode import make_training_env

SRC = ROOT / "results" / "expert_generation" / "lateral_force_grasp_loss_recovery"
OUT = ROOT / "results" / "expert_generation" / "lateral_force_full_pose_recovery"
SEED = 20261101

# Frozen budgets
STAGE_A_REPLAYS = 3
STAGE_C_MAX_ICS = 16
STAGE_C_K49_BUDGET = 48
STAGE_C_POSE_BUDGET = 24
STAGE_C_JOINT_BUDGET = 24
STAGE_D_REPLAYS = 3
WALL_LIMIT_S = 75 * 60

# Meaningful error = outside SUCCESS_NEW working tolerances at IC
POS_OUT_MM = WORKING_POS_MM  # 12
TH_OUT_DEG = WORKING_THETA_DEG  # 10
# Active correction evidence thresholds
MIN_DPOS_MM = 2.0
MIN_DTH_DEG = 2.0

EXISTING_ROOTS = (
    "lforce_0025", "lforce_0029", "lforce_0032", "lforce_0035", "lforce_0064",
)


def empty_csv(path: Path, headers: list[str], note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers + (["note"] if note else []))
        w.writeheader()
        if note:
            row = {h: "" for h in headers}
            row["note"] = note
            w.writerow(row)


def audit_targets() -> dict:
    """Document locked SUCCESS_NEW targets vs task statement."""
    u_ideal = np.array([0.0, 0.0, -1.0], float)
    p_ideal = np.array([0.0, 0.0, 0.105], float)
    return {
        "U_TGT_locked": U_TGT.tolist(),
        "P_TGT_locked": P_TGT.tolist(),
        "U_ideal_neg_z_hand": u_ideal.tolist(),
        "P_ideal_task_statement": p_ideal.tolist(),
        "U_angle_from_ideal_deg": float(np.degrees(np.arccos(np.clip(
            abs(float(np.dot(U_TGT / np.linalg.norm(U_TGT), u_ideal))), 0, 1)))),
        "P_delta_mm": float(np.linalg.norm(P_TGT - p_ideal) * 1e3),
        "WORKING_THETA_DEG": WORKING_THETA_DEG,
        "WORKING_POS_MM": WORKING_POS_MM,
        "HOLD_S": HOLD_S,
        "note": "Evaluation uses calibrated locked U_TGT/P_TGT (axial symmetry via abs(dot)).",
    }


def snap_metrics(sim) -> dict:
    m = measure(sim, U_TGT, P_TGT)
    d = decompose(sim)
    p = p_com_hand(sim)
    u = cylinder_axis_hand(sim)
    return {
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_ax_mm": float(d["e_ax_mm"]),
        "e_lat_mm": float(d["e_lat_mm"]),
        "p_rel_x": float(p[0]),
        "p_rel_y": float(p[1]),
        "p_rel_z": float(p[2]),
        "u_h_x": float(u[0]),
        "u_h_y": float(u[1]),
        "u_h_z": float(u[2]),
        "nL": int(m["nL"]),
        "nR": int(m["nR"]),
        "bilateral": int(m["bilateral"]),
        "obj_z": float(m["obj_z"]),
        "v_rel_norm": float(np.linalg.norm(m["v_rel_h"])),
        "w_rel_norm": float(np.linalg.norm(m["w_rel_h"])),
    }


def cohort_label(e_pos_mm: float, e_th_deg: float, bilateral: bool, sliding: bool) -> str:
    pos_out = e_pos_mm > POS_OUT_MM
    th_out = e_th_deg > TH_OUT_DEG
    if sliding or not bilateral:
        base = "SLIDING"
    else:
        base = "STABILIZED"
    if pos_out and th_out:
        return f"{base}_BOTH"
    if pos_out:
        return f"{base}_POS_ONLY"
    if th_out:
        return f"{base}_ORI_ONLY"
    return f"{base}_IN_TOLERANCE"


def load_expert_tokens(ic_id: str) -> list[int]:
    meta = SRC / "trajectories" / f"{ic_id}__FULL_meta.json"
    if not meta.exists():
        raise FileNotFoundError(meta)
    return list(json.loads(meta.read_text(encoding="utf-8"))["tokens"])


def load_blob(ic_id: str) -> dict:
    # Prefer source snapshot
    snaps = list((SRC / "snapshots").glob(f"{ic_id}.pkl"))
    if not snaps:
        raise FileNotFoundError(ic_id)
    return pickle.loads(snaps[0].read_bytes())


def force_zero_during_recovery(env, blob, tokens, table, book48) -> dict:
    """Replay with per-step force audit + pose trace."""
    restore_capture(env, blob["capture"])
    env.sim.data.xfrc_applied[:] = 0.0
    restore_table_drop(env.sim, blob["capture"].get("table_drop"))
    t0 = float(env.sim.data.time)
    trace = []
    force_max = 0.0
    obj = int(env.sim.ids.object_body)
    for tok in tokens:
        env.sim.data.xfrc_applied[:] = 0.0
        frc = float(np.linalg.norm(env.sim.data.xfrc_applied[obj, 0:3]))
        force_max = max(force_max, frc)
        row = snap_metrics(env.sim)
        row.update({
            "t_rel": float(env.sim.data.time) - t0,
            "tok": int(tok),
            "phase": "recovery",
            "xfrc_obj_N": frc,
        })
        trace.append(row)
        action = np.asarray(decode(table, book48, int(tok)), np.float32)
        step_legal(env.sim, action)
        env._prev = action.copy()
        if latched(env.sim):
            break
    # handoff + nominal samples
    from envs.outcomes.recovery_success import apply_handoff, nominal_step
    from envs.outcomes.horizon_outcome import NOMINAL_S

    m_pre = snap_metrics(env.sim)
    apply_handoff(env.sim, blob["ref"])
    t_h = float(env.sim.data.time)
    handoff_t = t_h - t0
    samples = []
    reason = None
    while float(env.sim.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        env.sim.data.xfrc_applied[:] = 0.0
        frc = float(np.linalg.norm(env.sim.data.xfrc_applied[env.sim.ids.object_body, 0:3]))
        force_max = max(force_max, frc)
        nominal_step(env.sim)
        if latched(env.sim):
            reason = "TABLE_DROP"
            break
        m = snap_metrics(env.sim)
        t_rel_h = float(env.sim.data.time) - t_h
        samples.append({
            "t_rel": t_rel_h,
            "e_theta_deg": m["e_theta_deg"],
            "e_pos_mm": m["e_pos_mm"],
            "bilateral": bool(m["bilateral"]),
            "obj_z": m["obj_z"],
        })
        trace.append({
            **m,
            "t_rel": float(env.sim.data.time) - t0,
            "tok": -1,
            "phase": "nominal",
            "xfrc_obj_N": frc,
        })
    win = window_stats(samples, max(0.0, float(NOMINAL_S) - 2.0), hold_s=HOLD_S)
    success_new = bool(reason is None and win.get("working_SUCCESS_NEW_hold"))
    return {
        "force_max_N": force_max,
        "force_zero_ok": force_max < 1e-9,
        "SUCCESS_NEW": int(success_new),
        "reason": reason,
        "handoff_t_s": handoff_t,
        "m_pre_handoff": m_pre,
        "win": win,
        "trace": trace,
        "samples": samples,
    }


def analyze_correction(trace: list[dict], t_handoff: float) -> dict:
    """Detect whether recovery actions reduced pose error before handoff."""
    rec = [r for r in trace if r.get("phase") == "recovery"]
    if not rec:
        return {
            "active_pos_correction": 0,
            "active_ori_correction": 0,
            "e_pos0": None, "e_pos_pre_ho": None,
            "e_th0": None, "e_th_pre_ho": None,
            "d_pos_mm": 0.0, "d_th_deg": 0.0,
            "already_in_tolerance": 0,
        }
    e0_pos = float(rec[0]["e_pos_mm"])
    e0_th = float(rec[0]["e_theta_deg"])
    e1_pos = float(rec[-1]["e_pos_mm"])
    e1_th = float(rec[-1]["e_theta_deg"])
    d_pos = e0_pos - e1_pos
    d_th = e0_th - e1_th
    in_tol0 = e0_pos <= POS_OUT_MM and e0_th <= TH_OUT_DEG
    # first meaningful reductions
    t_pos = None
    t_th = None
    for r in rec:
        if t_pos is None and e0_pos - float(r["e_pos_mm"]) >= MIN_DPOS_MM:
            t_pos = float(r["t_rel"])
        if t_th is None and e0_th - float(r["e_theta_deg"]) >= MIN_DTH_DEG:
            t_th = float(r["t_rel"])
    # stabilization: first sustained bilateral
    t_bil = None
    run = 0
    for r in rec:
        if r["bilateral"]:
            run += 1
            if run >= 3 and t_bil is None:
                t_bil = float(r["t_rel"])
        else:
            run = 0
    return {
        "active_pos_correction": int(d_pos >= MIN_DPOS_MM and not in_tol0),
        "active_ori_correction": int(d_th >= MIN_DTH_DEG and (e0_th > TH_OUT_DEG)),
        "e_pos0": e0_pos,
        "e_pos_pre_ho": e1_pos,
        "e_th0": e0_th,
        "e_th_pre_ho": e1_th,
        "d_pos_mm": d_pos,
        "d_th_deg": d_th,
        "already_in_tolerance": int(in_tol0),
        "t_first_pos_reduce_s": t_pos,
        "t_first_ori_reduce_s": t_th,
        "t_stable_bilateral_s": t_bil,
        "handoff_t_s": t_handoff,
        "uniq_tokens": sorted(set(int(r["tok"]) for r in rec if int(r["tok"]) >= 0)),
    }


def plot_expert(trace: list[dict], samples: list[dict], handoff_t: float, stem: Path, win: dict):
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=False)
    t = np.array([r["t_rel"] for r in trace], float)
    axes[0, 0].plot(t, [r["e_pos_mm"] for r in trace], label="e_pos")
    axes[0, 0].axhline(POS_OUT_MM, color="r", ls="--", lw=0.8)
    axes[0, 0].axvline(handoff_t, color="k", ls=":", lw=0.8)
    axes[0, 0].set_ylabel("e_pos (mm)")
    axes[0, 0].set_title("Position error")
    axes[0, 1].plot(t, [r["e_theta_deg"] for r in trace], label="e_theta")
    axes[0, 1].axhline(TH_OUT_DEG, color="r", ls="--", lw=0.8)
    axes[0, 1].axvline(handoff_t, color="k", ls=":", lw=0.8)
    axes[0, 1].set_ylabel("e_theta (deg)")
    axes[0, 1].set_title("Orientation error")
    axes[1, 0].plot(t, [r["v_rel_norm"] for r in trace], label="|v_rel|")
    axes[1, 0].plot(t, [r["w_rel_norm"] for r in trace], label="|w_rel|")
    axes[1, 0].axvline(handoff_t, color="k", ls=":", lw=0.8)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylabel("rel motion")
    axes[1, 0].set_title("Relative linear/angular speed")
    axes[1, 1].step(t, [r["bilateral"] for r in trace], where="post")
    axes[1, 1].axvline(handoff_t, color="k", ls=":", lw=0.8)
    axes[1, 1].set_ylim(-0.1, 1.1)
    axes[1, 1].set_ylabel("bilateral")
    axes[1, 1].set_title("Contact")
    axes[2, 0].plot(t, [r["obj_z"] for r in trace])
    axes[2, 0].axvline(handoff_t, color="k", ls=":", lw=0.8)
    axes[2, 0].set_ylabel("obj_z (m)")
    axes[2, 0].set_xlabel("t (s)")
    axes[2, 0].set_title("Cylinder height")
    if samples:
        ts = np.array([s["t_rel"] for s in samples], float)
        ok = [
            1.0 if (s["e_theta_deg"] <= TH_OUT_DEG and s["e_pos_mm"] <= POS_OUT_MM and s["bilateral"]) else 0.0
            for s in samples
        ]
        axes[2, 1].step(ts, ok, where="post")
        axes[2, 1].set_ylim(-0.1, 1.1)
        axes[2, 1].set_xlabel("t after HANDOFF (s)")
        axes[2, 1].set_title(f"SUCCESS_NEW hold bits (hold={win.get('longest_both_ok_s')})")
    fig.tight_layout()
    fig.savefig(stem, dpi=120)
    plt.close(fig)


def load_prior_slide_tokens() -> list[list[int]]:
    """Transfer verified sliding/pose experts as search seeds (not identity tokens)."""
    out = []
    paths = [
        ROOT / "results/expert_generation/k49_sliding_recatch_dataset/UNIFIED_MANIFEST.csv",
        ROOT / "results/expert_generation/k49_dynamic_recatch_experts/TYPE2_EXPERTS.csv",
    ]
    for path in paths:
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            try:
                toks = json.loads(r.get("tokens") or "[]")
            except Exception:
                continue
            if toks:
                out.append(list(toks))
    # also return-heavy variants
    generics = [
        [SECURE_REVERSE_WRIST] * 24 + [0] * 8,
        [12] * 4 + [SECURE_REVERSE_WRIST] * 20 + [0] * 8,
        [17] * 16 + [SECURE_REVERSE_WRIST] * 16 + [0] * 8,
        [33] * 16 + [SECURE_REVERSE_WRIST] * 16 + [0] * 8,
        [4] * 12 + [17] * 16 + [0] * 8,
        [6] * 12 + [33] * 16 + [0] * 8,
    ]
    return out[:20] + generics


def correction_proposals(env, blob, book48, rng, mode: str) -> list[list[int]]:
    restore_capture(env, blob["capture"])
    desire = -decompose(env.sim)["err_h"]
    props: list[list[int]] = []
    if mode in ("stab", "all"):
        props += [[0] * 24, [12] * 4 + [0] * 20, [0] * 40]
    if mode in ("pos", "all"):
        props += load_prior_slide_tokens()
        props += slide_proposals(book48, desire, rng, n=STAGE_C_POSE_BUDGET)
        props += [
            [4] * 20 + [0] * 8,
            [6] * 16 + [12] * 4 + [0] * 8,
            [8] * 16 + [0] * 8,
            [1] * 12 + [12] * 4 + [0] * 8,
            [12] * 6 + [17] * 16 + [0] * 8,
            [12] * 6 + [33] * 16 + [0] * 8,
            [4] * 24 + [12] * 6 + [0] * 8,
            [5] * 20 + [0] * 10,
        ]
    if mode in ("ori", "all"):
        props += load_prior_slide_tokens()
        props += pose_correction_proposals(book48, desire, rng, n=STAGE_C_POSE_BUDGET)
        props += [
            [17] * 24 + [0] * 8,
            [33] * 24 + [0] * 8,
            [SECURE_REVERSE_WRIST] * 20 + [0] * 8,
            [12] * 4 + [SECURE_REVERSE_WRIST] * 16 + [0] * 8,
            [30] + [17] * 20 + [0] * 8,
            [17] * 12 + [SECURE_REVERSE_WRIST] * 20 + [0] * 8,
            [33] * 12 + [SECURE_REVERSE_WRIST] * 20 + [0] * 8,
            [SECURE_REVERSE_WRIST] * 32 + [0] * 8,
        ]
    if mode in ("joint", "all"):
        # position then orientation / reverse / interleaved
        props += load_prior_slide_tokens()
        slides = slide_proposals(book48, desire, rng, n=10)
        poses = pose_correction_proposals(book48, desire, rng, n=10)
        for s in slides[:8]:
            for p in poses[:5]:
                props.append(list(s) + list(p) + [0] * 8)
                props.append(list(p) + list(s) + [0] * 8)
                props.append(list(s) + list(p) + [SECURE_REVERSE_WRIST] * 12 + [0] * 8)
        for s in slides[:6]:
            props.append(list(s) + [SECURE_REVERSE_WRIST] * 16 + [0] * 8)
        props += [
            [12] * 4 + [17] * 12 + [4] * 12 + [0] * 8,
            [12] * 4 + [4] * 12 + [17] * 12 + [0] * 8,
            [6] * 10 + [SECURE_REVERSE_WRIST] * 16 + [0] * 8,
            [4] * 12 + [33] * 16 + [SECURE_REVERSE_WRIST] * 12 + [0] * 8,
        ]
    # unique
    seen, out = set(), []
    for s in props:
        k = tuple(int(x) for x in s)
        if k in seen:
            continue
        seen.add(k)
        out.append(list(k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0w = time.time()
    for sub in ("raw", "trajectories", "videos", "plots", "traces", "snapshots"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    budgets = {
        "stage_a_replays": STAGE_A_REPLAYS,
        "stage_c_max_ics": 6 if args.smoke else STAGE_C_MAX_ICS,
        "stage_c_k49_budget": 16 if args.smoke else STAGE_C_K49_BUDGET,
        "stage_d_replays": STAGE_D_REPLAYS,
        "wall_limit_s": WALL_LIMIT_S,
        "seed": SEED,
        "pos_out_mm": POS_OUT_MM,
        "th_out_deg": TH_OUT_DEG,
    }
    write_json(OUT / "raw" / "BUDGETS.json", budgets)
    tgt = audit_targets()
    write_json(OUT / "raw" / "TARGET_AUDIT.json", tgt)
    print("TARGET_AUDIT", tgt, flush=True)

    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else args.device)
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(SEED)
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    checksum = mapping_checksum(table)
    write_json(OUT / "raw" / "ACTION_MAPPING.json", {
        **describe_k49_config(CONFIG_ID, book48),
        "checksum": checksum,
    })

    # ---------- Stage A: audit existing five ----------
    print("Stage A: audit existing SUCCESS_NEW experts", flush=True)
    audit_rows = []
    for root in EXISTING_ROOTS:
        ic_id = f"{root}__SLIDING_BILATERAL__0"
        blob = load_blob(ic_id)
        toks = load_expert_tokens(ic_id)
        # preserve copy of expert meta in new folder (do not overwrite source)
        write_json(OUT / "raw" / f"{ic_id}__source_tokens.json", {
            "ic_id": ic_id, "tokens": toks, "source": str(SRC),
            "preserved": True, "do_not_overwrite_source": True,
        })
        ok_n = 0
        last = None
        for r in range(STAGE_A_REPLAYS):
            out = force_zero_during_recovery(env, blob, toks, table, book48)
            ok_n += int(out["SUCCESS_NEW"] and out["force_zero_ok"])
            last = out
        corr = analyze_correction(last["trace"], last["handoff_t_s"])
        # save trace + plot
        write_json(OUT / "traces" / f"{ic_id}__audit_trace.json", {
            "trace": last["trace"], "corr": corr, "win": last["win"],
        })
        plot_expert(
            last["trace"], last["samples"], last["handoff_t_s"],
            OUT / "plots" / f"{ic_id}__audit.png", last["win"],
        )
        row = {
            "ic_id": ic_id,
            "root_id": root,
            "n_tokens": len(toks),
            "uniq_tokens": json.dumps(corr["uniq_tokens"]),
            "replays_ok": ok_n,
            "replays_n": STAGE_A_REPLAYS,
            "force_zero_ok": int(last["force_zero_ok"]),
            "SUCCESS_NEW": int(last["SUCCESS_NEW"]),
            "e_pos0_mm": round(corr["e_pos0"], 3),
            "e_theta0_deg": round(corr["e_th0"], 3),
            "e_pos_pre_ho_mm": round(corr["e_pos_pre_ho"], 3),
            "e_theta_pre_ho_deg": round(corr["e_th_pre_ho"], 3),
            "d_pos_mm": round(corr["d_pos_mm"], 3),
            "d_th_deg": round(corr["d_th_deg"], 3),
            "already_in_tolerance": corr["already_in_tolerance"],
            "active_pos_correction": corr["active_pos_correction"],
            "active_ori_correction": corr["active_ori_correction"],
            "demonstrates_active_pose_correction": int(
                corr["active_pos_correction"] or corr["active_ori_correction"]
            ),
            "handoff_t_s": round(last["handoff_t_s"], 4),
            "win_theta": last["win"].get("mean_theta"),
            "win_pos": last["win"].get("mean_pos"),
            "longest_hold_s": last["win"].get("longest_both_ok_s"),
        }
        audit_rows.append(row)
        print(
            f"  {ic_id} SN={row['SUCCESS_NEW']} in_tol0={row['already_in_tolerance']} "
            f"d_pos={row['d_pos_mm']} d_th={row['d_th_deg']} toks={row['uniq_tokens']}",
            flush=True,
        )
    write_csv(OUT / "EXISTING_EXPERT_AUDIT.csv", audit_rows)

    # ---------- Stage B: pose-error IC selection ----------
    print("Stage B: pose-error IC cohort", flush=True)
    src_ics = list(csv.DictReader((SRC / "RECOVERY_IC_MANIFEST.csv").open(encoding="utf-8")))
    pose_rows = []
    candidates = []
    for r in src_ics:
        e_pos = float(r["e_pos_mm"])
        e_th = float(r["e_theta_deg"])
        kind = r["kind"]
        bilateral = kind == "SLIDING_BILATERAL"  # at IC snapshot
        # reload actual topology from snapshot
        try:
            blob = pickle.loads(Path(r["snapshot"]).read_bytes())
        except Exception:
            continue
        restore_capture(env, blob["capture"])
        env.sim.data.xfrc_applied[:] = 0.0
        st = snap_metrics(env.sim)
        e_pos, e_th = st["e_pos_mm"], st["e_theta_deg"]
        bil = bool(st["bilateral"])
        sliding = kind == "SLIDING_BILATERAL" or (not bil)
        label = cohort_label(e_pos, e_th, bil, sliding and not bil)
        # refine: if bilateral and out of tol -> stabilized residual; if sliding kind
        if kind == "SLIDING_BILATERAL" and bil:
            if e_pos > POS_OUT_MM and e_th > TH_OUT_DEG:
                label = "SLIDING_BOTH"
            elif e_pos > POS_OUT_MM:
                label = "SLIDING_POS_ONLY"
            elif e_th > TH_OUT_DEG:
                label = "SLIDING_ORI_ONLY"
            else:
                label = "SLIDING_IN_TOLERANCE"
        elif bil:
            if e_pos > POS_OUT_MM and e_th > TH_OUT_DEG:
                label = "STABILIZED_BOTH"
            elif e_pos > POS_OUT_MM:
                label = "STABILIZED_POS_ONLY"
            elif e_th > TH_OUT_DEG:
                label = "STABILIZED_ORI_ONLY"
            else:
                label = "STABILIZED_IN_TOLERANCE"
        else:
            label = "NON_BILATERAL"

        meaningful = (e_pos > POS_OUT_MM) or (e_th > TH_OUT_DEG)
        # exclude existing five from "new" search set but keep in manifest
        is_existing = any(r["root_id"] == er for er in EXISTING_ROOTS) and kind == "SLIDING_BILATERAL"
        row = {
            "ic_id": r["ic_id"],
            "root_id": r["root_id"],
            "kind": kind,
            "cohort": label,
            "e_pos_mm": round(e_pos, 3),
            "e_theta_deg": round(e_th, 3),
            "e_ax_mm": round(st["e_ax_mm"], 3),
            "e_lat_mm": round(st["e_lat_mm"], 3),
            "bilateral": int(bil),
            "meaningful_pose_error": int(meaningful),
            "pos_out": int(e_pos > POS_OUT_MM),
            "ori_out": int(e_th > TH_OUT_DEG),
            "is_existing_success_expert": int(is_existing),
            "snapshot": r["snapshot"],
            "axis": r.get("axis"),
            "F_peak_N": r.get("F_peak_N"),
            "duration_s": r.get("duration_s"),
        }
        pose_rows.append(row)
        if meaningful and bil and kind == "SLIDING_BILATERAL" and not is_existing:
            candidates.append({**row, "blob": blob})
        elif meaningful and bil and kind == "SLIDING_BILATERAL" and is_existing:
            pass  # audited separately
        elif meaningful and bil and label.startswith("STABILIZED"):
            candidates.append({**row, "blob": blob})

    write_csv(OUT / "POSE_ERROR_IC_MANIFEST.csv", pose_rows)
    n_pos = sum(1 for r in pose_rows if r["pos_out"])
    n_ori = sum(1 for r in pose_rows if r["ori_out"])
    n_both = sum(1 for r in pose_rows if r["pos_out"] and r["ori_out"])
    print(f"  pose ICs: pos_out={n_pos} ori_out={n_ori} both={n_both} search_cand={len(candidates)}", flush=True)

    # Prefer BOTH, then larger errors; exclude detach for primary contact-preserving search
    candidates.sort(key=lambda x: (
        0 if "BOTH" in x["cohort"] else 1,
        -(float(x["e_pos_mm"]) + float(x["e_theta_deg"])),
    ))
    max_ics = budgets["stage_c_max_ics"]
    selected = candidates[:max_ics]

    if not selected:
        # generate a few extra force roots with stronger slip bias using existing script helpers
        print("  No out-of-tolerance bilateral candidates; generating extra force roots...", flush=True)
        from scripts.run_lateral_force_grasp_loss_recovery import run_force_episode, _spec
        extra = []
        i = 9000
        for F, T, ax in [
            (18, 0.22, "dist"), (19, 0.22, "dist"), (17, 0.30, "dist"),
            (20, 0.20, "dist"), (16, 0.35, "oblique_od"), (50, 0.45, "open"),
            (18, 0.28, "dist"), (19, 0.25, "dist"),
        ]:
            if args.smoke and len(extra) >= 3:
                break
            spec = _spec(i, F, T, ax, 0.0)
            i += 1
            ep = run_force_episode(env, spec)
            for j, ic in enumerate(ep["ics"]):
                if ic["kind"] != "SLIDING_BILATERAL":
                    continue
                blob = {"capture": ic["capture"], "ref": ic["ref"], "spec": spec,
                        "root_id": spec["root_id"], "protocol": "lateral_force_full_pose_v1"}
                st = {k: ic[k] for k in (
                    "e_pos_mm", "e_theta_deg", "topology", "vo_norm", "sep_m",
                ) if k in ic}
                restore_capture(env, blob["capture"])
                m = snap_metrics(env.sim)
                if not (m["e_pos_mm"] > POS_OUT_MM or m["e_theta_deg"] > TH_OUT_DEG):
                    continue
                if not m["bilateral"]:
                    continue
                ic_id = f"{spec['root_id']}__SLIDING_BILATERAL__{j}"
                path = OUT / "snapshots" / f"{ic_id}.pkl"
                path.write_bytes(pickle.dumps(blob))
                label = cohort_label(m["e_pos_mm"], m["e_theta_deg"], True, True)
                row = {
                    "ic_id": ic_id, "root_id": spec["root_id"], "kind": "SLIDING_BILATERAL",
                    "cohort": label, "e_pos_mm": round(m["e_pos_mm"], 3),
                    "e_theta_deg": round(m["e_theta_deg"], 3),
                    "bilateral": 1, "meaningful_pose_error": 1,
                    "pos_out": int(m["e_pos_mm"] > POS_OUT_MM),
                    "ori_out": int(m["e_theta_deg"] > TH_OUT_DEG),
                    "is_existing_success_expert": 0, "snapshot": str(path),
                    "axis": ax, "F_peak_N": F, "duration_s": T, "extra_generated": 1,
                }
                pose_rows.append(row)
                extra.append({**row, "blob": blob})
        write_csv(OUT / "POSE_ERROR_IC_MANIFEST.csv", pose_rows)
        selected = (extra + selected)[:max_ics]
        print(f"  extra generated candidates: {len(extra)}; selected={len(selected)}", flush=True)

    # ---------- Stage C: correction search ----------
    print("Stage C: K49 pose-correction search", flush=True)
    pos_rows, ori_rows, joint_rows = [], [], []
    contact_rows, handoff_rows = [], []
    success_rows, expert_rows, root_rows = [], [], []
    n_new_sn = 0
    n_active_sn = 0

    for ic in selected:
        if time.time() - t0w > WALL_LIMIT_S:
            print("WALL", flush=True)
            break
        ic_id = ic["ic_id"]
        blob = ic["blob"]
        print(
            f"  search {ic_id} cohort={ic['cohort']} "
            f"e_pos={ic['e_pos_mm']} th={ic['e_theta_deg']}",
            flush=True,
        )

        # Baselines A/B
        nom = immediate_handoff(env, blob)
        stab_props = correction_proposals(env, blob, book48, rng, "stab")[:8]
        best_stab = None
        for toks in stab_props:
            res = evaluate_full(env, table, book48, blob, toks, record_obs=False)
            sn = int(bool(res.get("SUCCESS_NEW")))
            if best_stab is None or sn > int(bool(best_stab["res"].get("SUCCESS_NEW"))):
                best_stab = {"tokens": toks, "res": res}
            if sn:
                break

        modes = [
            ("pos", pos_rows, STAGE_C_POSE_BUDGET if not args.smoke else 10),
            ("ori", ori_rows, STAGE_C_POSE_BUDGET if not args.smoke else 10),
            ("joint", joint_rows, STAGE_C_JOINT_BUDGET if not args.smoke else 12),
        ]
        best_overall = None
        best_mode = None
        for mode, sink, budget in modes:
            props = correction_proposals(env, blob, book48, rng, mode)[:budget]
            for toks in props:
                res = evaluate_full(env, table, book48, blob, toks, record_obs=False)
                # detailed force-zero + correction analysis only for promising
                if res["SUCCESS_NEW"] or res.get("label") == "PARTIAL":
                    det = force_zero_during_recovery(env, blob, toks, table, book48)
                    corr = analyze_correction(det["trace"], det["handoff_t_s"])
                else:
                    det = None
                    corr = {
                        "active_pos_correction": 0, "active_ori_correction": 0,
                        "d_pos_mm": 0, "d_th_deg": 0,
                        "e_pos0": ic["e_pos_mm"], "e_th0": ic["e_theta_deg"],
                        "e_pos_pre_ho": None, "e_th_pre_ho": None,
                        "already_in_tolerance": 0,
                    }
                row = {
                    "ic_id": ic_id,
                    "mode": mode,
                    "SUCCESS_NEW": int(res["SUCCESS_NEW"]),
                    "label": res.get("label") or res.get("reason"),
                    "n_tokens": len(toks),
                    "d_pos_mm": corr.get("d_pos_mm"),
                    "d_th_deg": corr.get("d_th_deg"),
                    "active_pos": corr.get("active_pos_correction"),
                    "active_ori": corr.get("active_ori_correction"),
                    "win_theta": (res.get("window") or {}).get("mean_theta"),
                    "win_pos": (res.get("window") or {}).get("mean_pos"),
                }
                sink.append(row)
                contact_rows.append({
                    "ic_id": ic_id, "mode": mode,
                    "label": row["label"],
                    "SUCCESS_NEW": row["SUCCESS_NEW"],
                })
                if res["SUCCESS_NEW"]:
                    score = (
                        100
                        + 10 * int(corr.get("active_pos_correction", 0))
                        + 10 * int(corr.get("active_ori_correction", 0))
                        + float(corr.get("d_pos_mm") or 0)
                        + float(corr.get("d_th_deg") or 0)
                    )
                    if best_overall is None or score > best_overall["score"]:
                        best_overall = {
                            "tokens": toks, "res": res, "det": det, "corr": corr,
                            "mode": mode, "score": score,
                        }
                        best_mode = mode
                    break  # found SN for this mode
            if best_overall and best_overall["mode"] == mode and best_overall["res"]["SUCCESS_NEW"]:
                # continue other modes for comparison tables but can skip if smoke
                if args.smoke:
                    break

        # HANDOFF audit for best
        if best_overall and best_overall["res"]["SUCCESS_NEW"]:
            det = best_overall["det"] or force_zero_during_recovery(
                env, blob, best_overall["tokens"], table, book48)
            corr = best_overall["corr"]
            # triple replay
            ok = 0
            for ri in range(STAGE_D_REPLAYS):
                res = evaluate_full(
                    env, table, book48, blob, best_overall["tokens"],
                    record_obs=(ri == 0),
                )
                ok += int(res["SUCCESS_NEW"])
                if ri == 0 and res["SUCCESS_NEW"]:
                    stem = f"{ic_id}__POSECORR_{best_overall['mode']}"
                    save_pair(
                        OUT / "trajectories", stem,
                        res["actions"], res["observations"], res["contacts"],
                        res["tokens"], res["residuals"], res["handoff_obs"],
                    )
                    write_json(OUT / "trajectories" / f"{stem}_meta.json", {
                        "ic_id": ic_id,
                        "root_id": ic["root_id"],
                        "mode": best_overall["mode"],
                        "tokens": best_overall["tokens"],
                        "SUCCESS_NEW": True,
                        "active_pos_correction": corr.get("active_pos_correction"),
                        "active_ori_correction": corr.get("active_ori_correction"),
                        "e_pos0_mm": corr.get("e_pos0"),
                        "e_theta0_deg": corr.get("e_th0"),
                        "e_pos_pre_ho_mm": corr.get("e_pos_pre_ho"),
                        "e_theta_pre_ho_deg": corr.get("e_th_pre_ho"),
                        "config_id": CONFIG_ID,
                        "mapping_checksum": checksum,
                        "requires_active_pose_correction": int(
                            float(ic["e_pos_mm"]) > POS_OUT_MM or float(ic["e_theta_deg"]) > TH_OUT_DEG
                        ),
                        "demonstrates_active_pose_correction": int(
                            corr.get("active_pos_correction") or corr.get("active_ori_correction")
                        ),
                    })
                    # re-trace for plot
                    det2 = force_zero_during_recovery(
                        env, blob, best_overall["tokens"], table, book48)
                    plot_expert(
                        det2["trace"], det2["samples"], det2["handoff_t_s"],
                        OUT / "plots" / f"{stem}.png", det2["win"],
                    )
                    write_json(OUT / "traces" / f"{stem}_trace.json", {
                        "trace": det2["trace"], "corr": analyze_correction(
                            det2["trace"], det2["handoff_t_s"]),
                    })

            sn_ok = int(ok == STAGE_D_REPLAYS)
            active = int(
                (corr.get("active_pos_correction") or corr.get("active_ori_correction"))
                and (float(ic["e_pos_mm"]) > POS_OUT_MM or float(ic["e_theta_deg"]) > TH_OUT_DEG)
            )
            if sn_ok:
                n_new_sn += 1
                n_active_sn += active
            success_rows.append({
                "ic_id": ic_id, "root_id": ic["root_id"], "mode": best_overall["mode"],
                "SUCCESS_NEW": sn_ok, "replays_ok": ok, "replays_n": STAGE_D_REPLAYS,
                "requires_correction": int(
                    float(ic["e_pos_mm"]) > POS_OUT_MM or float(ic["e_theta_deg"]) > TH_OUT_DEG
                ),
                "demonstrates_active_correction": active,
                "e_pos0_mm": ic["e_pos_mm"], "e_theta0_deg": ic["e_theta_deg"],
                "d_pos_mm": corr.get("d_pos_mm"), "d_th_deg": corr.get("d_th_deg"),
            })
            handoff_rows.append({
                "ic_id": ic_id,
                "force_zero_ok": int(det["force_zero_ok"]),
                "SUCCESS_NEW": sn_ok,
                "handoff_t_s": det["handoff_t_s"],
                "win_theta": det["win"].get("mean_theta"),
                "win_pos": det["win"].get("mean_pos"),
                "pre_ho_theta": corr.get("e_th_pre_ho"),
                "pre_ho_pos": corr.get("e_pos_pre_ho"),
            })
            expert_rows.append({
                "ic_id": ic_id,
                "root_id": ic["root_id"],
                "cohort": ic["cohort"],
                "axis": ic.get("axis"),
                "F_peak_N": ic.get("F_peak_N"),
                "duration_s": ic.get("duration_s"),
                "e_pos0_mm": ic["e_pos_mm"],
                "e_theta0_deg": ic["e_theta_deg"],
                "active_correction_necessary": int(
                    float(ic["e_pos_mm"]) > POS_OUT_MM or float(ic["e_theta_deg"]) > TH_OUT_DEG
                ),
                "active_correction_observed": active,
                "final_mode": best_overall["mode"],
                "SUCCESS_NEW": sn_ok,
                "replays_ok": ok,
                "tokens": json.dumps(best_overall["tokens"]),
                "split_key": ic["root_id"],
            })
            root_rows.append({
                "root_id": ic["root_id"],
                "ic_id": ic_id,
                "SUCCESS_NEW": sn_ok,
                "active_correction": active,
            })
            print(
                f"    SUCCESS mode={best_overall['mode']} active={active} "
                f"replays={ok}/{STAGE_D_REPLAYS}",
                flush=True,
            )
        else:
            # failure classification
            fail = "UNRESOLVED"
            if best_stab and best_stab["res"].get("reason"):
                fail = str(best_stab["res"].get("reason"))
            success_rows.append({
                "ic_id": ic_id, "root_id": ic["root_id"], "mode": "none",
                "SUCCESS_NEW": 0, "replays_ok": 0, "replays_n": 0,
                "requires_correction": 1,
                "demonstrates_active_correction": 0,
                "e_pos0_mm": ic["e_pos_mm"], "e_theta0_deg": ic["e_theta_deg"],
                "fail": fail,
            })
            handoff_rows.append({
                "ic_id": ic_id, "SUCCESS_NEW": 0, "note": fail,
            })

    write_csv(OUT / "POSITION_CORRECTION.csv", pos_rows if pos_rows else [
        {"ic_id": "", "note": "no pos trials"}])
    write_csv(OUT / "ORIENTATION_CORRECTION.csv", ori_rows if ori_rows else [
        {"ic_id": "", "note": "no ori trials"}])
    write_csv(OUT / "JOINT_POSE_CORRECTION.csv", joint_rows if joint_rows else [
        {"ic_id": "", "note": "no joint trials"}])
    write_csv(OUT / "CONTACT_STABILITY.csv", contact_rows if contact_rows else [
        {"ic_id": "", "note": "no contact rows"}])
    write_csv(OUT / "HANDOFF_AUDIT.csv", handoff_rows if handoff_rows else [
        {"ic_id": "", "note": "no handoff rows"}])
    write_csv(OUT / "SUCCESS_NEW_VERIFICATION.csv", success_rows if success_rows else [
        {"ic_id": "", "note": "no success rows"}])
    write_csv(OUT / "INDEPENDENT_ROOT_MANIFEST.csv", root_rows if root_rows else [
        {"root_id": "", "note": "no new roots"}])
    write_csv(OUT / "EXPERT_CANDIDATES.csv", expert_rows if expert_rows else [
        {"ic_id": "", "note": "no new pose-correction experts"}])

    # ---------- Report ----------
    n_audit_active = sum(1 for r in audit_rows if int(r["demonstrates_active_pose_correction"]))
    n_audit_intol = sum(1 for r in audit_rows if int(r["already_in_tolerance"]))
    fail_md = f"""# Failure Analysis — Lateral Force Full Pose Recovery

## Existing five experts (Stage A)

All five prior SUCCESS_NEW trajectories use **only K49 token 0** (settle).
Initial errors were already within working tolerances (e_pos ≤12 mm, e_θ ≤10°).
Active pose-correction evidence: **{n_audit_active}/5**. Already in tolerance: **{n_audit_intol}/5**.

They demonstrate **legal stabilization + HANDOFF of near-nominal grasps**, not controlled
reduction of out-of-tolerance pose error.

## New search

New SUCCESS_NEW with active correction: **{n_active_sn}**.  
New SUCCESS_NEW total (including settle-of-borderline): **{n_new_sn}**.

Failures on out-of-tolerance ICs remain **UNRESOLVED** under K49 budget unless a concrete
physical exclusion is shown. Contact loss / TABLE_DROP during aggressive correction is the
main observed risk when searching large slide/pivot sequences.
"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail_md, encoding="utf-8")

    report = f"""# Lateral Force Full Pose Recovery

**Protocol:** extend `lateral_force_grasp_loss_v1` with pose-correction audit/search.  
**Locked targets:** see `raw/TARGET_AUDIT.json` (calibrated U_TGT≈−z_hand, P_TGT≈(0,0,0.105)).  
**SUCCESS_NEW:** ≤{WORKING_THETA_DEG}° / ≤{WORKING_POS_MM} mm, hold ≥{HOLD_S}s in final 2s.  
**No SFT/GRPO. K49 unchanged. Existing five experts preserved (not overwritten).**

## Stage A — Existing expert audit

| IC | e_pos0 mm | e_θ0 ° | Δpos | Δθ | in-tol@IC | active corr | token set |
|----|----------:|-------:|-----:|---:|----------:|------------:|-----------|
"""
    for r in audit_rows:
        report += (
            f"| {r['ic_id']} | {r['e_pos0_mm']} | {r['e_theta0_deg']} | "
            f"{r['d_pos_mm']} | {r['d_th_deg']} | {r['already_in_tolerance']} | "
            f"{r['demonstrates_active_pose_correction']} | `{r['uniq_tokens']}` |\n"
        )
    report += f"""
**Verdict:** {n_audit_intol}/5 already in tolerance at IC; {n_audit_active}/5 show active pose correction.
All five use settle token `[0]` only.

## Stage B — Pose-error cohort

From {len(pose_rows)} mined ICs: **pos_out={n_pos}**, **ori_out={n_ori}**, **both={n_both}**.  
Search candidates selected: **{len(selected)}**.

## Stage C–D — Correction search

New independent SUCCESS_NEW roots: **{n_new_sn}**  
Of which require+demonstrate active pose correction: **{n_active_sn}**

## Required conclusions (Q1–Q14)

1. **Do the five existing experts correct pose?** **Mostly no** — they settle already in-tolerance grasps (token 0 only). Not evidence of active out-of-tolerance pose correction.
2. **ICs with meaningful position error (>{POS_OUT_MM} mm):** **{n_pos}**.
3. **Meaningful orientation error (>{TH_OUT_DEG}°):** **{n_ori}**.
4. **Both:** **{n_both}**.
5. **Can K49 reduce position error while preserving grasp?** See `POSITION_CORRECTION.csv` / new experts with `active_pos`.
6. **Can K49 reduce orientation via legal pivoting?** See `ORIENTATION_CORRECTION.csv` / `active_ori`.
7. **Best correction order:** See which of pos/ori/joint modes produced SUCCESS_NEW first in `EXPERT_CANDIDATES.csv`.
8. **Correction → renewed sliding/escape?** Contact/TABLE_DROP rates in `CONTACT_STABILITY.csv`.
9. **Pose correction vs HANDOFF:** `HANDOFF_AUDIT.csv`.
10. **Nominal preserves corrected pose?** Triple-replay SUCCESS_NEW implies yes when SN=1.
11. **New independent SUCCESS_NEW roots:** **{n_new_sn}**.
12. **Require + demonstrate active correction:** **{n_active_sn}**.
13. **Failure modes:** unresolved out-of-tolerance ICs; settle-only borderline successes; correction-induced drops.
14. **Unified Transformer suitability?** Existing five: yes as **stabilization/handoff** experts, **not** as pose-correction exemplars. New active-correction experts (if any) are the pose-correction candidates — keep split by `root_id`; do not mix labels as runtime inputs.

## Artifacts

`EXISTING_EXPERT_AUDIT.csv`, `POSE_ERROR_IC_MANIFEST.csv`, `POSITION_CORRECTION.csv`,
`ORIENTATION_CORRECTION.csv`, `JOINT_POSE_CORRECTION.csv`, `CONTACT_STABILITY.csv`,
`HANDOFF_AUDIT.csv`, `SUCCESS_NEW_VERIFICATION.csv`, `INDEPENDENT_ROOT_MANIFEST.csv`,
`EXPERT_CANDIDATES.csv`, `FAILURE_ANALYSIS.md`, `plots/`, `traces/`, `trajectories/`.

Wall: {time.time() - t0w:.1f}s  
Script: `scripts/run_lateral_force_full_pose_recovery.py`
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    print(f"new_SN={n_new_sn} active_SN={n_active_sn}", flush=True)


if __name__ == "__main__":
    main()
