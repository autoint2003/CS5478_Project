"""P87 — K48→M111 B1 Actor skill retention (bounded BC; official R3F unchanged)."""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import finish_recovery, recovery_failed  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    OUT as FINAL_OUT,
    P64_ACTOR,
    P64_CRITIC,
    T_MAX,
)
from experiments.precision.p43_r3_working_hold_v2.r3_rules import DwellTracker  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import (  # noqa: E402
    StabilizeGrpoEnv,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import (  # noqa: E402
    rollout_seed,
    seed_all,
)
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (  # noqa: E402
    load_actor,
    load_critic,
    save_pair,
)
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value  # noqa: E402
from experiments.precision.p66_c_curriculum_ppo.expert_drive import (  # noqa: E402
    ensure_inner_capture_buffers,
)
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import (  # noqa: E402
    evaluate_predicates,
)
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.controller import (  # noqa: E402
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal  # noqa: E402
from experiments.precision.p82_global_failure_budget.collect import _earliest_decisive  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from models.k48_checkpoint import HANDOFF as K48_HANDOFF  # noqa: E402
from models.k48_checkpoint import load_trained_k48  # noqa: E402
from models.m111_checkpoint import HANDOFF as M111_HANDOFF  # noqa: E402
from models.m111_checkpoint import decode_m111_action  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import decode_action, load_codebook  # noqa: E402
from training.k48_config import SELECTED_LIMIT_S  # noqa: E402
from training.recovery_episode import make_training_env  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory  # noqa: E402

OUT = ROOT / "results/experiments/p87_k48_actor_retention"
TEACHER_DIR = OUT / "teacher"
CKPT_DIR = OUT / "checkpoints"
P86_ABL = ROOT / "results/experiments/p86_b1_gate_counterfactual/PAIRED_GATE_ABLATION.csv"
P86_ATTR = ROOT / "results/experiments/p86_b1_gate_counterfactual/ATTRIBUTION.csv"
P86_PANEL = ROOT / "results/experiments/p86_b1_gate_counterfactual/PANEL.json"
K48_TREE80 = ROOT / "results/grpo/tree80/checkpoint_epoch_0003.pt"

SPLIT = "p87_k48_actor_retention"
PANEL_SEED = 87
N_B1_VAL = 16
N_REHEARSE_PER = 8
MIN_R3F_COMPAT = 8
BC_UPDATES = 60
BC_LR = 1e-5
BC_BATCH = 8
MAX_BC_LEN = 48
KL_WARN = 0.15
# Command-space L2 threshold for "first significant action divergence" vs K48/M111 teacher.
DIV_L2 = 0.15
# B1 BC uses only R3F-compatible demos (Phase-2 gate demos), not all SUCCESS_12S motion.


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def exposed_b1() -> tuple[set[str], set[str]]:
    man = json.loads((FINAL_OUT / "FROZEN_TEST_MANIFEST.json").read_text(encoding="utf-8"))
    tids = {e["traj_id"] for e in man["episodes"] if e["skill"] == "B1"}
    rids = {e["root_id"] for e in man["episodes"] if e["skill"] == "B1"}
    return tids, rids


def blob_from_start(start) -> dict:
    if start.kind == "capture":
        return {"capture": start.capture, "ref": start.ref, "traj_id": start.traj_id}
    ep = start.ep or {}
    return {"capture": ep["capture"], "ref": ep["ref"], "traj_id": start.traj_id}


# ---------------------------------------------------------------------------
# Phase 0
# ---------------------------------------------------------------------------


def phase0_reconcile() -> None:
    rows = list(csv.DictReader(P86_ABL.open(encoding="utf-8")))
    attr = list(csv.DictReader(P86_ATTR.open(encoding="utf-8")))
    by: dict[str, dict[str, dict]] = {}
    for r in rows:
        by.setdefault(r["traj_id"], {})[r["variant"]] = r

    def transitions(src: str, dst: str) -> Counter:
        c: Counter = Counter()
        for tid, d in by.items():
            a = int(d[src]["SUCCESS_RELAXED_12S"])
            b = int(d[dst]["SUCCESS_RELAXED_12S"])
            c[(a, b)] += 1
        return c

    lines = [
        f"# P86_TRANSITION_RECONCILIATION\n\nGenerated: {_utc()}\n",
        "Source: frozen P86 CSVs (not re-run). Explains GATE_RESCUED=5 vs net +4 per variant.\n",
        "## Paired SUCCESS_RELAXED_12S matrices (G0 → Gx)\n",
        "| Transition | G0→G1 | G0→G2 | G0→G3 |",
        "|---|---:|---:|---:|",
    ]
    mats = {gx: transitions("G0", gx) for gx in ("G1", "G2", "G3")}
    for key, label in [((0, 0), "fail→fail"), ((1, 1), "succ→succ"), ((0, 1), "fail→succ (rescue)"), ((1, 0), "succ→fail (regress)")]:
        lines.append(
            f"| {label} | {mats['G1'][key]} | {mats['G2'][key]} | {mats['G3'][key]} |"
        )
    lines += [
        "",
        "| Variant | Rescues | Regressions | Net Δ vs G0 | G0 succ | Gx succ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for gx in ("G1", "G2", "G3"):
        m = mats[gx]
        g0 = m[(1, 1)] + m[(1, 0)]
        gx_s = m[(1, 1)] + m[(0, 1)]
        lines.append(
            f"| {gx} | {m[(0,1)]} | {m[(1,0)]} | {m[(0,1)]-m[(1,0)]} | {g0} | {gx_s} |"
        )

    rescued_any = [r for r in attr if r["attribution"] == "GATE_RESCUED"]
    lines += [
        "",
        "## GATE_RESCUED union (attribution label = rescued by **any** of G1/G2/G3)",
        "",
        f"Count: **{len(rescued_any)}** (not equal to per-variant rescues).",
        "",
        "| traj_id | G1 | G2 | G3 |",
        "|---|---:|---:|---:|",
    ]
    for r in rescued_any:
        lines.append(
            f"| `{r['traj_id']}` | {r['G1_SUCCESS_RELAXED_12S']} | {r['G2_SUCCESS_RELAXED_12S']} | {r['G3_SUCCESS_RELAXED_12S']} |"
        )

    # variant-specific
    lines += [
        "",
        "## Reconciliation",
        "",
        "- Per-variant **net** gain is **+4** for G1, G2, and G3 (7→11).",
        "- G1/G2: **4 rescues**, **0 regressions** → net +4.",
        "- G3: **5 rescues**, **1 regression** (`k48r_train_00150`) → net +4.",
        "- Attribution `GATE_RESCUED=5` is the **union** of ICs rescued by ≥1 variant; "
        "`k48r_train_00130` is G2/G3-only; `slide_extra_260064` is G1/G3-only (not G2).",
        "- No historical P86 files were modified.",
        "",
    ]
    _write(OUT / "P86_TRANSITION_RECONCILIATION.md", "\n".join(lines))


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------


def build_splits() -> dict[str, Any]:
    exp_t, exp_r = exposed_b1()
    sched = build_p54_dataset(seed=5478)["schedule"]
    by = {"A1": [], "B1": [], "C": [], "D": []}
    for s in sched:
        if s.traj_id in exp_t or root_id(s.traj_id) in exp_r:
            continue
        by[s.skill].append(s)
    for sk in by:
        by[sk].sort(key=lambda s: s.traj_id)

    rng = np.random.default_rng(PANEL_SEED)
    b1 = by["B1"]
    idx = sorted(int(i) for i in rng.choice(len(b1), size=min(N_B1_VAL, len(b1)), replace=False))
    val_set = {b1[i].traj_id for i in idx}
    b1_val = [s for s in b1 if s.traj_id in val_set]
    b1_train = [s for s in b1 if s.traj_id not in val_set]

    # P86 panel for primary reachability compare
    p86 = json.loads(P86_PANEL.read_text(encoding="utf-8")) if P86_PANEL.is_file() else {"b1_traj_ids": []}
    p86_ids = set(p86.get("b1_traj_ids") or [])
    lookup = {s.traj_id: s for s in b1}
    p86_starts = [lookup[t] for t in p86.get("b1_traj_ids") or [] if t in lookup]

    def take(skill: str, n: int) -> list:
        pool = by[skill]
        if len(pool) <= n:
            return list(pool)
        ii = sorted(int(i) for i in rng.choice(len(pool), size=n, replace=False))
        return [pool[i] for i in ii]

    splits = {
        "b1_teacher": b1_train,
        "b1_val": b1_val,
        "b1_p86_panel": p86_starts,
        "a1_x": take("A1", N_REHEARSE_PER),
        "c_x": take("C", N_REHEARSE_PER),
        "d_x": take("D", N_REHEARSE_PER),
        "a1_rehearse_pool": by["A1"],
        "c_rehearse_pool": by["C"],
        "d_rehearse_pool": by["D"],
    }
    meta = {
        "excluded_exposed_b1": sorted(exp_t),
        "n_b1_teacher": len(b1_train),
        "n_b1_val": len(b1_val),
        "n_b1_p86_panel": len(p86_starts),
        "b1_val_ids": [s.traj_id for s in b1_val],
        "b1_teacher_ids": [s.traj_id for s in b1_train],
        "note": "Exposed independent B1 ICs excluded from all P87 training/selection.",
    }
    (OUT / "SPLITS.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return splits


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------


def run_k48_teacher_episode(env, model, book_k48, blob, device) -> dict:
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    dwell = DwellTracker.from_rule("R3", dt=float(DT_POLICY))
    r3f = R3FOnline.create(t_max=10_000)
    r3f.tracker = dwell

    actions: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    tokens: list[int] = []
    obs_list: list[np.ndarray] = []
    prev_list: list[np.ndarray] = []
    lost_low = 0
    how = None
    ever_frame9 = False
    ever_frame12 = False
    min_e_bil = 1e9
    cap = int(round(float(SELECTED_LIMIT_S) / float(DT_POLICY)))

    for i in range(cap):
        if float(env.sim.data.time) - t0 >= float(SELECTED_LIMIT_S) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        obs_list.append(obs.copy())
        prev_list.append(env._prev.copy())
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _ = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        tokens.append(token)
        if token == int(K48_HANDOFF):
            how = "early"
            break
        residual, action = decode_action(book_k48, token, mean[-1].detach().float().cpu().numpy())
        residuals.append(np.asarray(residual, np.float32))
        actions.append(np.asarray(action, np.float32))
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        st = r3f.observe(env.sim, i + 1, tape_ended=False)
        g = relative_geometry(env.sim)
        e_pos = float(g["e_pos_mm"])
        theta = float(g["theta_deg"])
        bil = bool(g["bilateral"])
        pad4 = int(g.get("min_pad") or 0) >= 4
        if bil and pad4:
            min_e_bil = min(min_e_bil, e_pos)
        wk = bool(st.get("working_ok"))
        if bil and pad4 and wk and e_pos <= 9 and theta <= 8:
            ever_frame9 = True
        if bil and pad4 and wk and e_pos <= 12 and theta <= 8:
            ever_frame12 = True
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "success_12s": 0,
                "reason": "FAILED_DURING_RECOVERY",
                "actions": np.zeros((0, 7), np.float32),
                "accepted": 0,
            }

    if how is None:
        how = "forced"
    handoff_preds = evaluate_predicates(env.sim, ok_s=float(dwell.ok_s))
    r3f_ready = bool(handoff_preds.get("frame_ok") and handoff_preds.get("dwell_complete"))
    out = finish_recovery(env, blob["ref"], t0, how)
    acts = np.asarray(actions, np.float32) if actions else np.zeros((0, 7), np.float32)
    return {
        "success_12s": int(bool(out.get("success"))),
        "reason": out.get("reason"),
        "handoff_mode": how,
        "actions": acts,
        "residuals_k48": np.asarray(residuals, np.float32) if residuals else np.zeros((0, 7), np.float32),
        "tokens_k48": tokens,
        "obs": np.stack(obs_list, axis=0) if obs_list else np.zeros((0, 39), np.float32),
        "prev": np.stack(prev_list, axis=0) if prev_list else np.zeros((0, 7), np.float32),
        "n_motion": int(len(acts)),
        "ever_frame9": int(ever_frame9),
        "ever_frame12": int(ever_frame12),
        "r3f_ready_at_k48_handoff": int(r3f_ready),
        "handoff_e_pos": handoff_preds.get("e_pos"),
        "handoff_theta": handoff_preds.get("theta"),
        "handoff_frame_ok": int(bool(handoff_preds.get("frame_ok"))),
        "min_e_pos_bil_pad": float(min_e_bil if min_e_bil < 1e8 else -1),
        "recovery_elapsed": out.get("recovery_elapsed"),
    }


def m111_replay_ok(env, blob, actions, book_m111, rho, handoff_how: str) -> dict:
    """Quantize K48 actions → M111, physically replay, and re-record (obs, prev) on the replay path."""
    acts = np.asarray(actions, np.float32).reshape(-1, 7)
    empty = {
        "success_12s": 0,
        "clipped_frac": 1.0,
        "tokens": [],
        "residuals": np.zeros((0, 7), np.float32),
        "recon": acts,
        "obs": np.zeros((0, 39), np.float32),
        "prev": np.zeros((0, 7), np.float32),
        "ever_frame9": 0,
        "ever_frame12": 0,
        "ever_bilateral": 0,
        "bilateral_lost_after": 0,
        "dwell_complete_ever": 0,
        "first_cmd_divergence_step": None,
    }
    if len(acts) == 0:
        return empty
    q = quantize_trajectory(book_m111, acts, rho, w=DEFAULT_WEIGHTS)
    recon = np.asarray(q["recon_actions"], np.float32)
    err_l2 = np.linalg.norm(recon - acts, axis=1)
    first_cmd = None
    for i, e in enumerate(err_l2):
        if float(e) > DIV_L2:
            first_cmd = int(i)
            break

    restore_capture(env, blob["capture"])
    # Match teacher loop: clear / seed sequence buffers if present.
    if hasattr(env, "_obs_seq"):
        env._obs_seq = []
    if hasattr(env, "_prev_seq"):
        env._prev_seq = []
    if not hasattr(env, "_prev") or env._prev is None:
        env._prev = np.zeros(7, np.float32)

    t0 = float(env.sim.data.time)
    dwell = DwellTracker.from_rule("R3", dt=float(DT_POLICY))
    r3f = R3FOnline.create(t_max=10_000)
    r3f.tracker = dwell
    lost_low = 0
    obs_list: list[np.ndarray] = []
    prev_list: list[np.ndarray] = []
    ever_frame9 = ever_frame12 = ever_bil = False
    bil_lost_after = False
    saw_bil = False
    dwell_ever = False

    for i, a in enumerate(recon):
        obs = np.asarray(env.observation(), np.float32)
        prev = np.asarray(env._prev, np.float32).copy()
        obs_list.append(obs)
        prev_list.append(prev)
        if hasattr(env, "_obs_seq"):
            env._obs_seq.append(obs)
        if hasattr(env, "_prev_seq"):
            env._prev_seq.append(prev)
        step_legal(env.sim, a)
        env._prev = np.asarray(a, np.float32)
        st = r3f.observe(env.sim, i + 1, tape_ended=False)
        g = relative_geometry(env.sim)
        e_pos = float(g["e_pos_mm"])
        theta = float(g["theta_deg"])
        bil = bool(g["bilateral"])
        pad4 = int(g.get("min_pad") or 0) >= 4
        wk = bool(st.get("working_ok"))
        if bil:
            ever_bil = True
            saw_bil = True
        elif saw_bil:
            bil_lost_after = True
        if bil and pad4 and wk and e_pos <= 9 + 1e-9 and theta <= 8 + 1e-9:
            ever_frame9 = True
        if bil and pad4 and wk and e_pos <= 12 + 1e-9 and theta <= 8 + 1e-9:
            ever_frame12 = True
        preds = evaluate_predicates(env.sim, ok_s=float(dwell.ok_s))
        if preds.get("dwell_complete") and preds.get("frame_ok"):
            dwell_ever = True
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            return {
                "success_12s": 0,
                "reason": "FAILED_DURING_RECOVERY",
                "clipped_frac": float(q["clipped_frac"]),
                "tokens": q["tokens"].tolist(),
                "residuals": np.asarray(q["residuals"], np.float32),
                "recon": recon,
                "obs": np.stack(obs_list) if obs_list else empty["obs"],
                "prev": np.stack(prev_list) if prev_list else empty["prev"],
                "mean_recon_l2": float(np.mean(err_l2)),
                "max_recon_l2": float(np.max(err_l2)),
                "ever_frame9": int(ever_frame9),
                "ever_frame12": int(ever_frame12),
                "ever_bilateral": int(ever_bil),
                "bilateral_lost_after": int(bil_lost_after),
                "dwell_complete_ever": int(dwell_ever),
                "first_cmd_divergence_step": first_cmd,
            }
    out = finish_recovery(env, blob["ref"], t0, handoff_how)
    return {
        "success_12s": int(bool(out.get("success"))),
        "reason": out.get("reason"),
        "clipped_frac": float(q["clipped_frac"]),
        "tokens": q["tokens"].tolist(),
        "residuals": np.asarray(q["residuals"], np.float32),
        "recon": recon,
        "obs": np.stack(obs_list) if obs_list else empty["obs"],
        "prev": np.stack(prev_list) if prev_list else empty["prev"],
        "mean_recon_l2": float(np.mean(err_l2)),
        "max_recon_l2": float(np.max(err_l2)),
        "ever_frame9": int(ever_frame9),
        "ever_frame12": int(ever_frame12),
        "ever_bilateral": int(ever_bil),
        "bilateral_lost_after": int(bil_lost_after),
        "dwell_complete_ever": int(dwell_ever),
        "first_cmd_divergence_step": first_cmd,
    }


def phase1_2_teacher(splits, device) -> dict:
    if TEACHER_DIR.exists():
        for p in TEACHER_DIR.glob("*.npz"):
            p.unlink()
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    model, book_k48 = load_trained_k48(K48_TREE80, device)
    model.eval()
    if book_k48 is None or len(book_k48) != 47:
        book_k48 = load_codebook()
    env = make_training_env(device, model=model)
    _, book_m111, rho, _ = load_actor(P64_ACTOR, device)

    attempted = []
    accepted = []
    for i, start in enumerate(splits["b1_teacher"]):
        blob = blob_from_start(start)
        print(f"[teacher {i+1}/{len(splits['b1_teacher'])}] {start.traj_id}", flush=True)
        rec = run_k48_teacher_episode(env, model, book_k48, blob, device)
        row = {
            "traj_id": start.traj_id,
            "root_id": root_id(start.traj_id),
            "k48_success_12s": rec.get("success_12s", 0),
            "k48_reason": rec.get("reason"),
            "n_motion": rec.get("n_motion", 0),
            "k48_ever_frame9": rec.get("ever_frame9", 0),
            "k48_ever_frame12": rec.get("ever_frame12", 0),
            "r3f_ready_at_k48_handoff": rec.get("r3f_ready_at_k48_handoff", 0),
            "handoff_frame_ok": rec.get("handoff_frame_ok", 0),
            "min_e_pos_bil_pad": rec.get("min_e_pos_bil_pad"),
            "accepted": 0,
            "m111_replay_success_12s": 0,
            "r3f_compatible_motion": 0,
            "m111_ever_frame9": 0,
            "m111_ever_frame12": 0,
            "m111_dwell_complete_ever": 0,
            "m111_bilateral_lost_after": 0,
            "first_cmd_divergence_step": "",
        }
        if int(rec.get("success_12s") or 0) != 1 or int(rec.get("n_motion") or 0) == 0:
            attempted.append(row)
            print(f"  reject k48 s12={row['k48_success_12s']} reason={row['k48_reason']}", flush=True)
            continue
        how = rec.get("handoff_mode") or "forced"
        m = m111_replay_ok(env, blob, rec["actions"], book_m111, rho, how if how in ("early", "forced") else "forced")
        row["m111_replay_success_12s"] = int(m.get("success_12s") or 0)
        row["m111_reason"] = m.get("reason")
        row["clipped_frac"] = m.get("clipped_frac")
        row["mean_recon_l2"] = m.get("mean_recon_l2")
        row["m111_ever_frame9"] = int(m.get("ever_frame9") or 0)
        row["m111_ever_frame12"] = int(m.get("ever_frame12") or 0)
        row["m111_dwell_complete_ever"] = int(m.get("dwell_complete_ever") or 0)
        row["m111_bilateral_lost_after"] = int(m.get("bilateral_lost_after") or 0)
        row["first_cmd_divergence_step"] = (
            "" if m.get("first_cmd_divergence_step") is None else int(m["first_cmd_divergence_step"])
        )
        # R3F-compatible = physically validated M111 replay ever enters official ≤9 mm candidate frame
        row["r3f_compatible_motion"] = int(m.get("ever_frame9") or 0)
        if int(m.get("success_12s") or 0) != 1:
            attempted.append(row)
            print(f"  reject m111 replay fail", flush=True)
            continue
        row["accepted"] = 1
        # BC targets from M111 replay path only (obs/prev re-recorded under quantized actions)
        toks = [int(t) for t in m["tokens"]]
        res = np.asarray(m["residuals"], np.float32)
        obs = np.asarray(m["obs"], np.float32)
        prev = np.asarray(m["prev"], np.float32)
        recon = np.asarray(m["recon"], np.float32)
        n = min(len(toks), len(obs), len(prev), len(res), len(recon))
        sample = {
            "traj_id": start.traj_id,
            "skill": "B1",
            "tokens": np.asarray(toks[:n], np.int64),
            "residuals": res[:n],
            "actions": recon[:n],
            "obs": obs[:n],
            "prev": prev[:n],
            "r3f_compatible_motion": int(row["r3f_compatible_motion"]),
            "ever_frame12": int(row["m111_ever_frame12"]),
            "r3f_ready_at_k48_handoff": int(rec.get("r3f_ready_at_k48_handoff") or 0),
            "dwell_complete_ever": int(row["m111_dwell_complete_ever"]),
            "bilateral_lost_after": int(row["m111_bilateral_lost_after"]),
            "first_cmd_divergence_step": -1
            if m.get("first_cmd_divergence_step") is None
            else int(m["first_cmd_divergence_step"]),
        }
        path = TEACHER_DIR / f"{start.traj_id.replace('/','_')}.npz"
        np.savez_compressed(path, **{k: v for k, v in sample.items() if k != "skill"})
        row["sample_path"] = str(path.relative_to(ROOT)).replace("\\", "/")
        attempted.append(row)
        accepted.append(row)
        print(
            f"  ACCEPT n={n} m111_frame9={row['r3f_compatible_motion']} "
            f"k48_frame9={row['k48_ever_frame9']} r3f_ready_handoff={row['r3f_ready_at_k48_handoff']}",
            flush=True,
        )

    _write_csv(OUT / "K48_M111_TRAIN_REPLAY.csv", attempted)
    n_r3 = sum(int(r["r3f_compatible_motion"]) for r in accepted)
    manifest = {
        "generated": _utc(),
        "k48_checkpoint": str(K48_TREE80.relative_to(ROOT)).replace("\\", "/"),
        "inference": "argmax_roll_policy_no_tree_search",
        "n_attempted": len(attempted),
        "n_accepted": len(accepted),
        "n_r3f_compatible_motion": n_r3,
        "min_r3f_compat_required": MIN_R3F_COMPAT,
        "accepted_traj_ids": [r["traj_id"] for r in accepted],
        "r3f_compatible_traj_ids": [r["traj_id"] for r in accepted if int(r["r3f_compatible_motion"])],
        "excluded_exposed_b1": sorted(exposed_b1()[0]),
    }
    (OUT / "TEACHER_DATA_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    compat_md = f"""# R3F_TEACHER_COMPATIBILITY

Generated: {_utc()}

## Separation of motion vs HANDOFF supervision

- Teacher: K48 tree80_e3 **argmax** (no tree search).
- Acceptance: K48 SUCCESS_12S **and** M111 quantized physical replay SUCCESS_12S.
- BC observation/action pairs: re-recorded on the **M111 physical replay** path (not K48 obs).
- BC targets: **motion tokens only** (K48 HANDOFF token not imitated).
- R3F-compatible demo: **M111 replay** ever enters official ≤9 mm candidate frame
  (WORKING ∧ bilateral ∧ min_pad≥4 ∧ θ≤8 ∧ e_pos≤9). K48-only frame hits do not count.
- Bounded BC uses **only** R3F-compatible accepted demos (non-compatible SUCCESS_12S motion retained in manifest but excluded from loss).

## Counts

| Item | n |
|---|---:|
| Attempted train B1 ICs | {len(attempted)} |
| Accepted motion demos | {len(accepted)} |
| R3F-compatible (M111 ever frame9) | {n_r3} |
| K48 ever frame9 (among accepted) | {sum(int(r.get('k48_ever_frame9') or 0) for r in accepted)} |
| R3F ready at K48 HANDOFF instant | {sum(int(r.get('r3f_ready_at_k48_handoff') or 0) for r in accepted)} |
| M111 dwell_complete ever (among accepted) | {sum(int(r.get('m111_dwell_complete_ever') or 0) for r in accepted)} |
| Minimum required for training | {MIN_R3F_COMPAT} |

## Decision gate

{"**PASS** — proceed to bounded BC on R3F-compatible demos only." if n_r3 >= MIN_R3F_COMPAT else "**STOP** — insufficient R3F-compatible demos (CASE D)."}
"""
    _write(OUT / "R3F_TEACHER_COMPATIBILITY.md", compat_md)
    return {"manifest": manifest, "accepted": accepted, "n_r3f": n_r3, "stop": n_r3 < MIN_R3F_COMPAT}


# ---------------------------------------------------------------------------
# Rehearsal from frozen P64 successes
# ---------------------------------------------------------------------------


def collect_rehearsal(starts, device, *, skill: str, max_keep: int) -> list[dict]:
    actor, book, rho, _ = load_actor(P64_ACTOR, device)
    critic = load_critic(P64_CRITIC, actor, device)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    out = []
    for start in starts:
        if len(out) >= max_keep:
            break
        seed = rollout_seed(split=SPLIT, traj_id=start.traj_id, rollout_i=0)
        seed_all(seed, deterministic_torch=True)
        env.reset(start)
        ensure_inner_capture_buffers(env)
        st = WrapState()
        v2 = v2_spec()
        done = False
        info = {}
        toks, acts, obs_l, prev_l = [], [], [], []
        while not done:
            obs_l.append(np.asarray(last_obs(env), np.float32))
            prev_l.append(np.asarray(env.prev_seq[-1] if env.prev_seq else np.zeros(7), np.float32))
            tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
            tok, st, _ = select_token(
                env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
            )
            # record pre-step command
            if int(tok) != M111_HANDOFF:
                action = decode_m111_action(book, int(tok), resid, rho=rho)
                toks.append(int(tok))
                acts.append(np.asarray(action, np.float32))
            _r, done, info = env.step(int(tok), resid)
            ensure_inner_capture_buffers(env)
        if int(info.get("SUCCESS_RELAXED_12S") or 0) != 1:
            continue
        n = min(len(toks), len(obs_l), len(prev_l), len(acts))
        if n < 4:
            continue
        sample = {
            "traj_id": start.traj_id,
            "skill": skill,
            "tokens": np.asarray(toks[:n], np.int64),
            "residuals": np.zeros((n, 7), np.float32),  # unused; CE on tokens
            "actions": np.stack(acts[:n]),
            "obs": np.stack(obs_l[:n]),
            "prev": np.stack(prev_l[:n]),
            "r3f_compatible_motion": 1,
        }
        path = TEACHER_DIR / f"rehearse_{skill}_{start.traj_id.replace('/','_')[:40]}.npz"
        np.savez_compressed(path, **{k: v for k, v in sample.items() if k != "skill"})
        sample["sample_path"] = str(path)
        out.append(sample)
        print(f"[rehearse {skill}] {start.traj_id} n={n}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def load_bc_samples(accepted_rows: list[dict], rehearse: list[dict]) -> list[dict]:
    samples = []
    for r in accepted_rows:
        # Only R3F-compatible B1 demos enter the BC loss.
        if int(r.get("r3f_compatible_motion") or 0) != 1:
            continue
        p = ROOT / r["sample_path"]
        z = np.load(p)
        samples.append(
            {
                "skill": "B1",
                "traj_id": r["traj_id"],
                "tokens": np.asarray(z["tokens"], np.int64),
                "actions": np.asarray(z["actions"], np.float32),
                "obs": np.asarray(z["obs"], np.float32),
                "prev": np.asarray(z["prev"], np.float32),
                "weight": 1.0,
            }
        )
    for s in rehearse:
        samples.append(
            {
                "skill": s["skill"],
                "traj_id": s["traj_id"],
                "tokens": s["tokens"],
                "actions": s["actions"],
                "obs": s["obs"],
                "prev": s["prev"],
                "weight": 0.5,
            }
        )
    return samples


def bc_batch_loss(actor, batch: list[dict], device) -> tuple[torch.Tensor, dict]:
    losses = []
    n_steps = 0
    for s in batch:
        toks = torch.as_tensor(s["tokens"], device=device, dtype=torch.long)
        obs = torch.as_tensor(s["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(s["prev"], device=device, dtype=torch.float32)
        T = min(int(len(toks)), int(len(obs)), int(len(prev)), MAX_BC_LEN)
        if T <= 0:
            continue
        # Drop any accidental HANDOFF targets
        motion = toks[:T] != M111_HANDOFF
        if not bool(motion.any()):
            continue
        logits, _res, _ = actor(obs[:T], prev[:T])
        logit = mask_handoff_logits(logits.float())
        loss = F.cross_entropy(logit[motion], toks[:T][motion])
        losses.append(loss * float(s.get("weight") or 1.0))
        n_steps += int(motion.sum())
    if not losses:
        return torch.zeros((), device=device, requires_grad=True), {"n_steps": 0}
    return torch.stack(losses).mean(), {"n_steps": n_steps}


@torch.no_grad()
def kl_to_ref(actor, ref, samples: list[dict], device, *, n: int = 6) -> float:
    if not samples:
        return 0.0
    rng = np.random.default_rng(0)
    idx = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    kls = []
    for i in idx:
        s = samples[int(i)]
        T = min(len(s["tokens"]), len(s["obs"]), MAX_BC_LEN, 24)
        if T <= 0:
            continue
        obs = torch.as_tensor(s["obs"][:T], device=device, dtype=torch.float32)
        prev = torch.as_tensor(s["prev"][:T], device=device, dtype=torch.float32)
        la, _, _ = actor(obs, prev)
        lr, _, _ = ref(obs, prev)
        pa = torch.log_softmax(mask_handoff_logits(la.float()), dim=-1)
        pr = torch.softmax(mask_handoff_logits(lr.float()), dim=-1)
        # KL(ref || actor) mean over steps
        kl = (pr * (torch.log(pr + 1e-8) - pa)).sum(dim=-1).mean()
        kls.append(float(kl))
    return float(np.mean(kls)) if kls else 0.0


def phase3_train(accepted_rows, rehearse, device) -> dict:
    samples = load_bc_samples(accepted_rows, rehearse)
    b1_n = sum(1 for s in samples if s["skill"] == "B1")
    protocol = f"""# TRAINING_PROTOCOL

Generated: {_utc()}

## Setup

| Item | Value |
|---|---|
| Init | Frozen P64 Actor (`{P64_ACTOR.name}`) — original file **untouched** |
| Output | `checkpoints/P87_actor.pt` (new path) |
| Critic | **Not trained / not modified** |
| Loss | Motion-token CE (HANDOFF masked out of targets + logits) |
| B1 demos | {b1_n} (R3F-compatible M111-replay obs only) |
| A1/C/D rehearsal | {len(samples)-b1_n} |
| Updates | {BC_UPDATES} |
| LR | {BC_LR} |
| Batch demos / update | {BC_BATCH} |
| Max steps / demo | {MAX_BC_LEN} |
| KL monitor | mean KL(ref\\|actor) warn if > {KL_WARN} |
| R3F | Official (unchanged) |
| Privileged class labels | Not used as runtime inputs |

## Constraints

- No reward / physics / M111 vocab / decoder / nominal changes
- No GRPO/PPO
- No hyperparameter sweep; single predefined budget
- Exposed B1 eval ICs never in training data
"""
    _write(OUT / "TRAINING_PROTOCOL.md", protocol)

    actor, book, rho, _ = load_actor(P64_ACTOR, device)
    ref, _, _, _ = load_actor(P64_ACTOR, device)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(actor.parameters(), lr=BC_LR)
    rng = np.random.default_rng(5478)
    log = []
    actor.train()
    for u in range(BC_UPDATES):
        idx = rng.choice(len(samples), size=min(BC_BATCH, len(samples)), replace=False)
        batch = [samples[int(i)] for i in idx]
        opt.zero_grad(set_to_none=True)
        loss, meta = bc_batch_loss(actor, batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt.step()
        kl = kl_to_ref(actor, ref, samples, device)
        row = {"update": u, "loss": float(loss.detach()), "n_steps": meta["n_steps"], "kl_ref": kl}
        log.append(row)
        if (u + 1) % 10 == 0 or u == 0:
            print(f"[train] u={u+1}/{BC_UPDATES} loss={row['loss']:.4f} kl={kl:.4f}", flush=True)
        if kl > KL_WARN * 3:
            print(f"[train] early stop: KL {kl:.4f} >> warn", flush=True)
            break

    out_actor = CKPT_DIR / "P87_actor.pt"
    # Save actor only (critic unused); reuse save_m111 via save_pair with dummy critic skip
    from models.m111_checkpoint import save_m111

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    save_m111(out_actor, actor, book, epoch=BC_UPDATES, rho=rho, extra={"p87": True, "init": "P64_step4_actor"})
    _write_csv(OUT / "TRAIN_LOG.csv", log)
    return {"actor_path": out_actor, "book": book, "rho": rho, "log": log, "n_samples": len(samples)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _teacher_actions_for(traj_id: str) -> np.ndarray | None:
    p = TEACHER_DIR / f"{traj_id.replace('/', '_')}.npz"
    if not p.is_file():
        return None
    z = np.load(p)
    if "actions" not in z:
        return None
    return np.asarray(z["actions"], np.float32)


def eval_actor_on_starts(actor, critic, book, rho, starts, device, *, variant: str = "G0") -> list[dict]:
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    rows = []
    for start in starts:
        seed = rollout_seed(split=SPLIT, traj_id=start.traj_id, rollout_i=0)
        seed_all(seed, deterministic_torch=True)
        env.reset(start)
        env.mon = make_monitor(variant, t_max=int(env.t_max))
        ensure_inner_capture_buffers(env)
        st = WrapState()
        v2 = v2_spec()
        done = False
        info = {}
        # frame9/frame12 = P86 candidate geometry; e9_raw/e12_raw = bare e_pos (diagnostic only)
        ever = {
            k: False
            for k in (
                "fc",
                "opp",
                "pad4",
                "e9_raw",
                "e12_raw",
                "frame9",
                "frame12",
                "frame",
                "r3",
                "bil",
                "sw",
                "dwell",
                "loss",
            )
        }
        min_e_bil = 1e9
        saw_bil = False
        bil_lost_after = False
        teacher_acts = _teacher_actions_for(start.traj_id)
        first_div = None
        motion_i = 0
        while not done:
            tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
            tok, st, _ = select_token(
                env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
            )
            if int(tok) != M111_HANDOFF and teacher_acts is not None and first_div is None:
                action = decode_m111_action(book, int(tok), resid, rho=rho)
                if motion_i < len(teacher_acts):
                    if float(np.linalg.norm(np.asarray(action, np.float32) - teacher_acts[motion_i])) > DIV_L2:
                        first_div = int(motion_i)
                motion_i += 1
            _r, done, info = env.step(int(tok), resid)
            ensure_inner_capture_buffers(env)
            g = relative_geometry(env.env.sim)
            tr = getattr(env.mon, "tracker", None)
            ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
            preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
            e_pos = float(g["e_pos_mm"])
            theta = float(g["theta_deg"])
            bil = bool(g["bilateral"])
            pad4 = int(g.get("min_pad") or 0) >= 4
            wk = bool(preds.get("working_ok"))
            if g["has_pad_contact"]:
                ever["fc"] = True
            if bil:
                ever["bil"] = True
                saw_bil = True
            elif saw_bil:
                bil_lost_after = True
                ever["loss"] = True
            if bil and g.get("correct_sidewall"):
                ever["opp"] = True
            if pad4:
                ever["pad4"] = True
            if e_pos <= 9:
                ever["e9_raw"] = True
            if e_pos <= 12:
                ever["e12_raw"] = True
            if bil and pad4:
                min_e_bil = min(min_e_bil, e_pos)
            # P86 candidate geometry (primary reachability)
            if bil and pad4 and wk and e_pos <= 9 + 1e-9 and theta <= 8 + 1e-9:
                ever["frame9"] = True
            if bil and pad4 and wk and e_pos <= 12 + 1e-9 and theta <= 8 + 1e-9:
                ever["frame12"] = True
            if preds.get("frame_ok"):
                ever["frame"] = True
            if preds.get("dwell_complete"):
                ever["dwell"] = True
            if (preds.get("dwell_complete") and preds.get("frame_ok")) or int(info.get("handoff") or 0):
                ever["r3"] = True
        term = _r3_terminal(env, info)
        rows.append(
            {
                "skill": start.skill,
                "traj_id": start.traj_id,
                "root_id": root_id(start.traj_id),
                "variant": variant,
                "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
                "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
                "handoff": int(info.get("handoff") or 0),
                "r3_ready": int(term.get("r3_ready") or 0),
                # Primary: P86 candidate geometry
                "reach_e12": int(ever["frame12"]),
                "reach_e9": int(ever["frame9"]),
                "reach_frame": int(ever["frame"]),
                "reach_frame9_candidate": int(ever["frame9"]),
                "reach_frame12_candidate": int(ever["frame12"]),
                "e_pos_le9_raw": int(ever["e9_raw"]),
                "e_pos_le12_raw": int(ever["e12_raw"]),
                "dwell_complete_ever": int(ever["dwell"]),
                "bilateral_ever": int(ever["bil"]),
                "bilateral_retained": int(ever["bil"] and not bil_lost_after),
                "bilateral_lost_after": int(bil_lost_after),
                "min_pad_ge4": int(ever["pad4"]),
                "opposing_ever": int(ever["opp"]),
                "min_e_pos_bil_pad": float(min_e_bil if min_e_bil < 1e8 else -1),
                "first_sig_action_div_vs_k48": "" if first_div is None else int(first_div),
                "has_k48_teacher_ref": int(teacher_acts is not None),
                "ep_len": int(env.t),
                "earliest_decisive_failure": _earliest_decisive(
                    {
                        "fc": ever["fc"],
                        "opp": ever["opp"],
                        "pad4": ever["pad4"],
                        "e9": ever["frame9"],
                        "frame": ever["frame"],
                        "r3": ever["r3"],
                        "loss": ever["loss"],
                    },
                    term,
                    bool(env.t >= env.max_steps - 1) and not term["SUCCESS_RELAXED"],
                )
                if not term.get("SUCCESS_RELAXED")
                else "success",
            }
        )
    return rows


def summarize_reach(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "reach_e12": sum(int(r["reach_e12"]) for r in rows) / n,
        "reach_e9": sum(int(r["reach_e9"]) for r in rows) / n,
        "reach_frame": sum(int(r["reach_frame"]) for r in rows) / n,
        "r3_ready": sum(int(r["r3_ready"]) for r in rows) / n,
        "dwell_complete_ever": sum(int(r.get("dwell_complete_ever") or 0) for r in rows) / n,
        "bilateral_retained": sum(int(r.get("bilateral_retained") or 0) for r in rows) / n,
        "SUCCESS_RELAXED_12S": sum(int(r["SUCCESS_RELAXED_12S"]) for r in rows) / n,
        "SUCCESS_12S": sum(int(r["SUCCESS_12S"]) for r in rows) / n,
        "n_relaxed": sum(int(r["SUCCESS_RELAXED_12S"]) for r in rows),
        "n_e12": sum(int(r["reach_e12"]) for r in rows),
        "n_e9": sum(int(r["reach_e9"]) for r in rows),
        "n_frame": sum(int(r["reach_frame"]) for r in rows),
        "n_r3": sum(int(r["r3_ready"]) for r in rows),
        "n_dwell": sum(int(r.get("dwell_complete_ever") or 0) for r in rows),
        "n_bil_retained": sum(int(r.get("bilateral_retained") or 0) for r in rows),
    }


def phase4_6_eval(p87_path: Path, splits, device) -> dict:
    # Load both actors; share critic (frozen)
    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    p87, book2, rho2, _ = load_actor(p87_path, device)
    critic = load_critic(P64_CRITIC, p64, device)
    for p in list(p64.parameters()) + list(p87.parameters()) + list(critic.parameters()):
        p.requires_grad_(False)

    panels = {
        "p86_b1": splits["b1_p86_panel"],
        "b1_val": splits["b1_val"],
        "a1": splits["a1_x"],
        "c": splits["c_x"],
        "d": splits["d_x"],
    }

    all_rows = []
    for name, starts in panels.items():
        if not starts:
            continue
        print(f"[eval P64 G0] {name} n={len(starts)}", flush=True)
        r64 = eval_actor_on_starts(p64, critic, book, rho, starts, device, variant="G0")
        for r in r64:
            r["policy"] = "P64"
            r["panel"] = name
        all_rows.extend(r64)
        print(f"[eval P87 G0] {name}", flush=True)
        r87 = eval_actor_on_starts(p87, critic, book2, rho2, starts, device, variant="G0")
        for r in r87:
            r["policy"] = "P87"
            r["panel"] = name
        all_rows.extend(r87)

    # Optional G1 on B1 panels if reachability improved
    p86_p64 = [r for r in all_rows if r["panel"] == "p86_b1" and r["policy"] == "P64"]
    p86_p87 = [r for r in all_rows if r["panel"] == "p86_b1" and r["policy"] == "P87"]
    s64 = summarize_reach(p86_p64)
    s87 = summarize_reach(p86_p87)
    reach_improved = (s87["reach_e12"] > s64["reach_e12"] + 1e-9) or (s87["reach_e9"] > s64["reach_e9"] + 1e-9)

    g1_rows = []
    if reach_improved and splits["b1_p86_panel"]:
        print("[eval P87 G1 diagnostic] p86_b1", flush=True)
        g1 = eval_actor_on_starts(p87, critic, book2, rho2, splits["b1_p86_panel"], device, variant="G1")
        for r in g1:
            r["policy"] = "P87"
            r["panel"] = "p86_b1"
            r["variant"] = "G1"
        g1_rows = g1
        all_rows.extend(g1)

    _write_csv(OUT / "REACHABILITY_COMPARISON.csv", all_rows)
    _write_csv(
        OUT / "OFFICIAL_SUCCESS_COMPARISON.csv",
        [
            {
                "panel": r["panel"],
                "policy": r["policy"],
                "variant": r["variant"],
                "traj_id": r["traj_id"],
                "SUCCESS_RELAXED_12S": r["SUCCESS_RELAXED_12S"],
                "SUCCESS_12S": r["SUCCESS_12S"],
                "reach_frame12_candidate": r["reach_e12"],
                "reach_frame9_candidate": r["reach_e9"],
                "r3_ready": r["r3_ready"],
                "dwell_complete_ever": r.get("dwell_complete_ever"),
                "bilateral_retained": r.get("bilateral_retained"),
                "first_sig_action_div_vs_k48": r.get("first_sig_action_div_vs_k48"),
            }
            for r in all_rows
        ],
    )

    # Cross-class IC-level
    x_lines = [
        f"# CROSS_CLASS_REGRESSION\n\nGenerated: {_utc()}\n",
        "Official R3F (G0). Paired seeds.\n",
    ]
    regressions = {}
    for skill, panel in (("A1", "a1"), ("C", "c"), ("D", "d")):
        a = {r["traj_id"]: r for r in all_rows if r["panel"] == panel and r["policy"] == "P64" and r["variant"] == "G0"}
        b = {r["traj_id"]: r for r in all_rows if r["panel"] == panel and r["policy"] == "P87" and r["variant"] == "G0"}
        lost = gained = 0
        for tid in a:
            if int(a[tid]["SUCCESS_RELAXED_12S"]) and not int(b[tid]["SUCCESS_RELAXED_12S"]):
                lost += 1
            if (not int(a[tid]["SUCCESS_RELAXED_12S"])) and int(b[tid]["SUCCESS_RELAXED_12S"]):
                gained += 1
        regressions[skill] = {
            "n": len(a),
            "P64": sum(int(x["SUCCESS_RELAXED_12S"]) for x in a.values()),
            "P87": sum(int(x["SUCCESS_RELAXED_12S"]) for x in b.values()),
            "lost": lost,
            "gained": gained,
        }
        x_lines.append(
            f"## {skill}\n\nP64 {regressions[skill]['P64']}/{regressions[skill]['n']} → "
            f"P87 {regressions[skill]['P87']}/{regressions[skill]['n']} "
            f"(lost {lost}, gained {gained})\n"
        )
    _write(OUT / "CROSS_CLASS_REGRESSION.md", "\n".join(x_lines))

    return {
        "p86_p64": s64,
        "p86_p87": s87,
        "reach_improved": reach_improved,
        "g1_summary": summarize_reach(g1_rows) if g1_rows else None,
        "regressions": regressions,
        "val64": summarize_reach([r for r in all_rows if r["panel"] == "b1_val" and r["policy"] == "P64" and r["variant"] == "G0"]),
        "val87": summarize_reach([r for r in all_rows if r["panel"] == "b1_val" and r["policy"] == "P87" and r["variant"] == "G0"]),
    }


def decide_case(eval_out: dict, teacher_stop: bool) -> tuple[str, str]:
    if teacher_stop:
        return "D", "K48-derived supervision cannot reliably reach R3F-compatible states"
    s64, s87 = eval_out["p86_p64"], eval_out["p86_p87"]
    reg = eval_out["regressions"]
    any_loss = any(reg[sk]["lost"] > 0 for sk in reg)
    reach_up = (s87["reach_e12"] > s64["reach_e12"] + 1e-9) or (s87["reach_e9"] > s64["reach_e9"] + 1e-9)
    succ_up = s87["n_relaxed"] > s64["n_relaxed"]
    succ_flat = s87["n_relaxed"] == s64["n_relaxed"]
    if reach_up and succ_up and not any_loss:
        return "A", "B1 reachability and official success improve without observed cross-class regression"
    if reach_up and (succ_flat or not succ_up) and not any_loss:
        return "B", "B1 reachability improves but official success does not"
    if (reach_up or succ_up) and any_loss:
        return "C", "B1 improves but A1/C/D regress"
    return "E", "No measurable improvement over frozen P64"


def write_final(case: str, title: str, eval_out: dict, manifest: dict) -> None:
    s64, s87 = eval_out["p86_p64"], eval_out["p86_p87"]
    text = f"""# FINAL_REPORT — P87 K48→M111 B1 Actor Skill Retention

Generated: {_utc()}

## CASE {case} — {title}

### Setup (honored)

- Init from frozen P64 Actor; new checkpoint at `checkpoints/P87_actor.pt` only
- Official R3F unchanged in primary comparison
- Exposed independent B1 ICs excluded from training/selection
- Independent P64 0/8 remains frozen

### Teacher data

| Item | n |
|---|---:|
| Accepted motion demos | {manifest.get('n_accepted')} |
| R3F-compatible (ever ≤9 mm frame) | {manifest.get('n_r3f_compatible_motion')} |

### Primary panel (P86 B1 train diagnostic, n={s64['n']}) — official R3F

Reachability uses **P86 candidate geometry** (bilateral ∧ pad≥4 ∧ θ≤8 ∧ WORKING ∧ e_pos).

| Metric | P64 | P87 |
|---|---:|---:|
| reach ≤12 mm candidate | {s64['n_e12']}/{s64['n']} | {s87['n_e12']}/{s87['n']} |
| reach ≤9 mm candidate | {s64['n_e9']}/{s64['n']} | {s87['n_e9']}/{s87['n']} |
| R3F ready | {s64['n_r3']}/{s64['n']} | {s87['n_r3']}/{s87['n']} |
| dwell_complete ever | {s64.get('n_dwell', 0)}/{s64['n']} | {s87.get('n_dwell', 0)}/{s87['n']} |
| bilateral retained | {s64.get('n_bil_retained', 0)}/{s64['n']} | {s87.get('n_bil_retained', 0)}/{s87['n']} |
| SUCCESS_RELAXED_12S | {s64['n_relaxed']}/{s64['n']} | {s87['n_relaxed']}/{s87['n']} |
| SUCCESS_12S | {int(s64['SUCCESS_12S']*s64['n'])}/{s64['n']} | {int(s87['SUCCESS_12S']*s87['n'])}/{s87['n']} |

Reserved B1 val: P64 {eval_out['val64']['n_relaxed']}/{eval_out['val64']['n']} → P87 {eval_out['val87']['n_relaxed']}/{eval_out['val87']['n']} RELAXED.

### Rerun notes (fixes vs first P87 pass)

- BC obs/prev from M111 physical replay (not K48 rollout obs)
- B1 BC restricted to R3F-compatible (M111 ever frame9) demos only
- Reach metrics aligned to P86 candidate frame9/frame12
- Added dwell / bilateral retention / first significant action divergence vs K48 teacher

### Cross-class (G0)

"""
    for sk, d in eval_out["regressions"].items():
        text += f"- **{sk}**: {d['P64']}/{d['n']} → {d['P87']}/{d['n']} (lost {d['lost']}, gained {d['gained']})\n"
    if eval_out.get("g1_summary"):
        g = eval_out["g1_summary"]
        text += f"\n### Optional G1 diagnostic (P87 only)\n\nRELAXED {g['n_relaxed']}/{g['n']}; reach_e12 {g['n_e12']}/{g['n']}. Not deployed.\n"
    text += """
### Non-actions

- Final frozen checkpoint **not** auto-replaced
- No gate deployment
- No training on exposed B1 eval ICs
"""
    _write(OUT / "FINAL_REPORT.md", text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    phase0_reconcile()
    print("[P87] Phase 0 done", flush=True)

    splits = build_splits()
    print(
        f"[P87] splits teacher={len(splits['b1_teacher'])} val={len(splits['b1_val'])} "
        f"p86={len(splits['b1_p86_panel'])}",
        flush=True,
    )

    teach = phase1_2_teacher(splits, device)
    if teach["stop"]:
        _write(
            OUT / "FINAL_REPORT.md",
            f"""# FINAL_REPORT — P87

Generated: {_utc()}

## CASE D — K48-derived supervision cannot reliably reach R3F-compatible states

Accepted demos: {teach['manifest']['n_accepted']}; R3F-compatible: {teach['n_r3f']} < required {MIN_R3F_COMPAT}.
Training was **not** started.
""",
        )
        print(json.dumps({"case": "D", "n_r3f": teach["n_r3f"]}, indent=2))
        return

    # Rehearsal
    rehearse = []
    rehearse += collect_rehearsal(splits["a1_rehearse_pool"], device, skill="A1", max_keep=N_REHEARSE_PER)
    rehearse += collect_rehearsal(splits["c_rehearse_pool"], device, skill="C", max_keep=N_REHEARSE_PER)
    rehearse += collect_rehearsal(splits["d_rehearse_pool"], device, skill="D", max_keep=N_REHEARSE_PER)

    n_bc_b1 = sum(1 for r in teach["accepted"] if int(r.get("r3f_compatible_motion") or 0) == 1)
    if n_bc_b1 < MIN_R3F_COMPAT:
        _write(
            OUT / "FINAL_REPORT.md",
            f"""# FINAL_REPORT — P87\n\nGenerated: {_utc()}\n\n## CASE D — insufficient R3F-compatible BC demos\n\nn={n_bc_b1} < {MIN_R3F_COMPAT}.\n""",
        )
        print(json.dumps({"case": "D", "n_bc_b1": n_bc_b1}, indent=2))
        return

    train_out = phase3_train(teach["accepted"], rehearse, device)
    eval_out = phase4_6_eval(train_out["actor_path"], splits, device)
    case, title = decide_case(eval_out, teacher_stop=False)
    write_final(case, title, eval_out, teach["manifest"])
    summary = {
        "case": case,
        "title": title,
        "teacher": teach["manifest"],
        "p86_panel": {"P64": eval_out["p86_p64"], "P87": eval_out["p86_p87"]},
        "regressions": eval_out["regressions"],
        "actor_path": str(train_out["actor_path"].relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
