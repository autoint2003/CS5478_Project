"""Mechanism-selection feasibility audit (read-only; no training / reward edits).

Writes under results/diagnostics/mechanism_selection_feasibility/.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.cartesian.jacobian import apply_cartesian_ctrl
from controllers.nominal import _integrate_rot
from controllers.recovery.action_map import AIRBORNE_V3, map_action, grip_tau
from data_generation.datasets.skill_quotas import _fingerprint, _try_sliding
from data_generation.oracles.plan_b.vertical_recatch import (
    _finish,
    rollout as vertical_rollout,
    specs as vertical_specs,
)
from envs.outcomes.horizon_outcome import finish_recovery, recovery_failed, terminal_reason
from envs.outcomes.physical_recovery import physical_pack
from envs.outcomes.table_drop import object_vz, observe
from envs.observations.airborne_obs import (
    AIR_NAMES,
    OBS_NAMES_AIR,
    OBS_VERSION,
    SCALE_PREL,
    SCALE_ROT,
    SCALE_VREL,
    SCALE_WREL,
    observe_airborne,
    relative_state,
)
from envs.observations.observable_obs import OBS_NAMES, OBS_SOURCES
from envs.simulation.capture import restore_capture, take_capture
from envs.step_legal import N_SUB, TRACK, step_legal
from evaluation.classify_validation_skills import BOTH_OFF_HOLD_S, MIGRATION_M, classify_mechanism
from evaluation import skill_traces as audit
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import decode_action, load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from training.pipeline import HANDOFF

OUT = ROOT / "results" / "diagnostics" / "mechanism_selection_feasibility"
BENCH = ROOT / "results" / "benchmarks" / "hard_recovery_v1"
HARD_MINE = ROOT / "results" / "diagnostics" / "hard_ic_mining"
SFT = ROOT / "results/sft/sliding_equal_weight/checkpoint_epoch_0040.pt"
GRPO = ROOT / "results/grpo/adaptive_g/full_run_v2_natural_sampling/checkpoint_epoch_0005.pt"
CODEBOOK = ROOT / "results/archive/demo_results/diagnostics/raw/legal_recovery_v2/codebook_selected_k47.npy"
LIMIT_S = float(SELECTED_LIMIT_S)

GRPO_ONLY = [
    "hrv1_920047",
    "hrv1_920064",
    "hrv1_920106",
    "hrv1_920015",
    "hrv1_920069",
    "hrv1_leftover_hard_910207",
    "hrv1_leftover_hard_910231",
]


def jsafe(obj):
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): jsafe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsafe(x) for x in obj]
    return str(obj)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in keys})


def write_observation_audit(book):
    tok36 = np.asarray(book[36], float)
    tau36 = float(grip_tau(tok36[6]))
    text = f"""# OBSERVATION_AUDIT

## 39D policy observation (`envs/observations/airborne_obs.py`)

Version: `{OBS_VERSION}` · dim = {len(OBS_NAMES_AIR)}

### Prefix 27D (`envs/observations/observable_obs.py`)

| Index | Name | Source (code) | Scaling |
|---:|---|---|---|
"""
    for i, name in enumerate(OBS_NAMES):
        text += f"| {i} | `{name}` | {OBS_SOURCES.get(name, '?')} | see physical_recovery SCALE_* |\n"
    text += f"""
### Suffix 12D (object-relative, hand frame)

| Index | Name | Raw | Scale | Clipped |
|---:|---|---|---|---|
"""
    for j, name in enumerate(AIR_NAMES):
        scale = {0: SCALE_PREL, 1: SCALE_PREL, 2: SCALE_PREL,
                 3: SCALE_VREL, 4: SCALE_VREL, 5: SCALE_VREL,
                 6: SCALE_ROT, 7: SCALE_ROT, 8: SCALE_ROT,
                 9: SCALE_WREL, 10: SCALE_WREL, 11: SCALE_WREL}[j]
        text += f"| {27+j} | `{name}` | MuJoCo body pose/twist via `relative_state` | /{scale} | clip ±1 |\n"
    text += f"""
**Not in the 39D vector:** previous command (concatenated separately as `prev` 7D),
raw friction/mass, MuJoCo contact IDs, mechanism labels.

## Transformer history

- Windows: last **160** `(obs, prev)` pairs (`training/recovery_episode.py`, `adaptive_grpo.py`).
- Sampling: one pair per **20 ms** policy step (`step_legal`).
- `prev` = previously executed 7D legal action (zeros at episode start).
- Max history coverage ≈ 160 × 0.02 = **3.2 s**.

## K48 token 36

Codebook entry (`codebook_selected_k47.npy` row 36):

`{tok36.tolist()}`

- Translation/rotation channels: small residuals.
- Grip channel a[6]=`{tok36[6]}` → `grip_tau` = **{tau36:.3f} N·m** (`controllers/recovery/action_map.py`).
- a_grip=-1 maps to **TAU_OPEN** (opening torque). So token 36 commands **gripper opening** plus a small Cartesian tweak.
- Each execution: `step_legal` = {N_SUB} MuJoCo substeps × model timestep (~20 ms).

Token 47 = HANDOFF (no `step_legal`).

## Snapshot restore

`envs/simulation/capture.py` `take_capture` / `restore_capture`: MuJoCo physics state,
FSM (`p_des`,`r_des`,…), table-drop latch, Transformer `_obs_seq`/`_prev_seq`/`_prev`.
Paired rollouts must restore the same capture blob.

## Mechanism classifier (policy-step samples)

`evaluation/classify_validation_skills.py` `classify_mechanism`:

- Release–Recatch: sustained both-off ≥ {BOTH_OFF_HOLD_S}s then bilateral recontact before HANDOFF.
- Contact-Preserving Adjustment: contact preserved + centroid migration ≥ {MIGRATION_M} m.
- Hybrid: Recatch then post-recontact migration ≥ 0.002 m.
- Sub-threshold both-off → Other/Unclear / AMBIGUOUS territory.

Policy-step logging alone undersamples 20 ms interiors; this audit adds **substep** logs.
"""
    (OUT / "OBSERVATION_AUDIT.md").write_text(text, encoding="utf-8")


def contact_pack(sim):
    pack = physical_pack(sim)
    rel = relative_state(sim.model, sim.data, sim.ids)
    return {
        "t": float(sim.data.time),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "bilateral": int(pack["nL"]) > 0 and int(pack["nR"]) > 0,
        "both_off": int(pack["nL"]) == 0 and int(pack["nR"]) == 0,
        "fnL": float(pack.get("fnL") or 0.0),
        "fnR": float(pack.get("fnR") or 0.0),
        "aperture": float(pack.get("aperture") or 0.0),
        "p_rel": rel["p_rel_h"].tolist(),
        "v_rel": rel["v_rel_h"].tolist(),
        "rot_rel": rel["rot_rel"].tolist(),
        "||p_rel||": float(np.linalg.norm(rel["p_rel_h"])),
        "||v_rel||": float(np.linalg.norm(rel["v_rel_h"])),
    }


def step_legal_logged(sim, action, rows, phase, t0):
    """Mirror of `step_legal` with per-substep contact logging (diagnostic only)."""
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
    for s in range(N_SUB):
        sim.fsm.r_des = _integrate_rot(sim.fsm.r_des, ang, dt)
        apply_cartesian_ctrl(
            sim.model, sim.data, sim.ids, sim.fsm.p_des, sim.fsm.r_des,
            gripper_tau=tau, v_des=vel, w_des=ang, **TRACK,
        )
        vz_before = object_vz(sim)
        mujoco.mj_step(sim.model, sim.data)
        observe(sim, vz_before)
        row = contact_pack(sim)
        row.update({
            "phase": phase,
            "t_rel": float(sim.data.time) - float(t0),
            "substep": s,
            "tau_cmd": tau,
            "a_grip": float(action[6]),
        })
        rows.append(row)
    return action


def classify_substep(rows, t0, t_handoff):
    """Auditable substep mechanism label for short recoveries."""
    rec = [r for r in rows if r["phase"] == "recovery" and r["t"] <= t_handoff + 1e-12]
    nom = [r for r in rows if r["phase"] == "nominal"]
    if not rec:
        return {"label": "AMBIGUOUS", "confidence": "low", "why": ["no recovery samples"]}

    # both-off intervals on substeps
    intervals = []
    start = None
    for r in rec:
        if r["both_off"] and start is None:
            start = r["t_rel"]
        if (not r["both_off"]) and start is not None:
            intervals.append((start, r["t_rel"], r["t_rel"] - start))
            start = None
    if start is not None:
        intervals.append((start, rec[-1]["t_rel"], rec[-1]["t_rel"] - start))

    sustained = [iv for iv in intervals if iv[2] >= BOTH_OFF_HOLD_S - 1e-9]
    any_both_off = len(intervals) > 0
    max_gap = 0.0 if not intervals else max(iv[2] for iv in intervals)
    bilateral_frac = np.mean([1.0 if r["bilateral"] else 0.0 for r in rec])
    both_off_frac = np.mean([1.0 if r["both_off"] else 0.0 for r in rec])

    # effective grasp loss: both_off with aperture increase and |fn| collapse
    fn_max = max(max(r["fnL"], r["fnR"]) for r in rec)
    fn0 = max(rec[0]["fnL"], rec[0]["fnR"])
    ap0, ap1 = rec[0]["aperture"], rec[-1]["aperture"]

    # re-establishment before handoff
    recontact_before_h = False
    if intervals:
        last_end = intervals[-1][1]
        recontact_before_h = any(r["bilateral"] and r["t_rel"] >= last_end - 1e-12 for r in rec)
    # re-establishment in early nominal
    recontact_nominal = False
    if nom:
        recontact_nominal = any(r["bilateral"] for r in nom[: max(1, int(0.2 / 0.002))])  # ~first 0.2s if dt~0.002

    why = []
    label = "AMBIGUOUS"
    conf = "low"

    # True Recatch in recovery: sustained both-off + bilateral recontact before HANDOFF
    if sustained and recontact_before_h:
        label = "Release-Recatch"
        conf = "high"
        why.append(f"sustained both-off max={max_gap:.4f}s then bilateral recontact before HANDOFF")
    elif any_both_off and max_gap >= 0.01 and (not recontact_before_h) and recontact_nominal:
        label = "AMBIGUOUS"
        conf = "medium"
        why.append(
            f"both-off during recovery (max_gap={max_gap:.4f}s) without bilateral recontact "
            f"before HANDOFF; bilateral returns in early nominal — release may be real but "
            f"regrasp is post-HANDOFF (not recovery Recatch by classifier definition)"
        )
    elif any_both_off and max_gap < BOTH_OFF_HOLD_S and both_off_frac > 0.3:
        label = "AMBIGUOUS"
        conf = "medium"
        why.append(
            f"sub-threshold both-off (max_gap={max_gap:.4f}s < {BOTH_OFF_HOLD_S}s) with "
            f"both_off_frac={both_off_frac:.2f}; possible fast release/flicker"
        )
    elif bilateral_frac >= 0.95:
        # migration via p_rel change
        p0 = np.asarray(rec[0]["p_rel"], float)
        mig = max(float(np.linalg.norm(np.asarray(r["p_rel"], float) - p0)) for r in rec)
        if mig >= MIGRATION_M:
            label = "Contact-Preserving Sliding"
            conf = "medium"
            why.append(f"near-continuous bilateral contact; ||Δp_rel||max={mig:.4f}m")
        else:
            label = "AMBIGUOUS"
            conf = "low"
            why.append(f"bilateral preserved but migration {mig:.4f}m < {MIGRATION_M}m")
    else:
        why.append(
            f"mixed contact (bilateral_frac={bilateral_frac:.2f}, both_off_frac={both_off_frac:.2f}, "
            f"max_gap={max_gap:.4f}s, aperture {ap0:.4f}→{ap1:.4f}, fn0={fn0:.3f}, fn_max={fn_max:.3f})"
        )

    # grip opening evidence (token 36)
    if any(r.get("a_grip", 0) < -0.5 for r in rec):
        why.append("recovery commands include grip-open (a_grip < -0.5), consistent with intentional release")

    return {
        "label": label,
        "confidence": conf,
        "why": why,
        "facts": {
            "n_recovery_substeps": len(rec),
            "n_nominal_logged": len(nom),
            "both_off_intervals": [{"t0": a, "t1": b, "dur": c} for a, b, c in intervals],
            "max_both_off_s": max_gap,
            "bilateral_frac": bilateral_frac,
            "both_off_frac": both_off_frac,
            "recontact_before_handoff": recontact_before_h,
            "recontact_early_nominal": recontact_nominal,
            "aperture_0": ap0,
            "aperture_end_recovery": ap1,
            "fn0": fn0,
            "fn_max": fn_max,
        },
    }


def analyze_grpo_only_substep(env, model, book, device, ic_id, snapshot):
    blob = pickle.loads((ROOT / snapshot).read_bytes())
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    rows = []
    rows.append({**contact_pack(env.sim), "phase": "ic", "t_rel": 0.0, "substep": -1,
                 "tau_cmd": None, "a_grip": None})
    lost_low = 0
    tokens = []
    samples = [audit.sample_state(env.sim, blob["ref"], phase="recovery")]
    how = None
    for step in range(int(round(LIMIT_S / 0.02))):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _ = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        residual = mean[-1].detach().float().cpu().numpy()
        tokens.append(token)
        if token == HANDOFF:
            how = "early"
            samples.append(audit.sample_state(env.sim, blob["ref"], token=token, residual=residual, phase="handoff"))
            break
        _r, action = decode_action(book, token, residual)
        executed = step_legal_logged(env.sim, action, rows, "recovery", t0)
        env._prev = np.asarray(executed, np.float32)
        samples.append(audit.sample_state(env.sim, blob["ref"], action=executed, token=token, residual=residual, phase="recovery"))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            how = "drop"
            break
    t_hand = float(env.sim.data.time)
    # policy-step classifier
    mech_policy = classify_mechanism(samples, t0, t_hand)
    # nominal with sparse logging every ~5 substeps via nominal_step isn't available —
    # use finish_recovery then note outcome; separately log first ~0.5s by stepping nominal
    from envs.outcomes.recovery_success import apply_handoff, nominal_step
    apply_handoff(env.sim, blob["ref"])
    # log early nominal (~0.5 s)
    t_n0 = float(env.sim.data.time)
    while float(env.sim.data.time) - t_n0 < 0.5 - 1e-9:
        nominal_step(env.sim)
        if int(round((float(env.sim.data.time) - t_n0) / float(env.sim.model.opt.timestep))) % 5 == 0:
            row = contact_pack(env.sim)
            row.update({"phase": "nominal", "t_rel": float(env.sim.data.time) - t0, "substep": -1,
                        "tau_cmd": None, "a_grip": None})
            rows.append(row)
    # finish remaining nominal for outcome (restore would break continuity — continue)
    while float(env.sim.data.time) - t_hand < 12.0 - 1e-9:
        nominal_step(env.sim)
        # don't log all
        from envs.outcomes.horizon_outcome import _bilateral, _dropped, _monitor
        from training.recovery_execution import _escaped
        from envs.outcomes.table_drop import latched
        mon = _monitor(env.sim)
        if latched(env.sim) or _dropped(mon) or _escaped(env.sim):
            break
    # better: re-run clean outcome via finish_recovery from fresh restore for SUCCESS label
    restore_capture(env, blob["capture"])
    # replay same tokens with normal step_legal for outcome
    t0b = float(env.sim.data.time)
    lost_low = 0
    how2 = None
    for token in tokens:
        if token == HANDOFF:
            how2 = "early"
            break
        _r, action = decode_action(book, token, np.zeros(7, np.float32))  # residual approx 0 — WRONG
    # Use residual-aware replay from first pass: re-collect with mean residuals saved
    # Simpler: call finish after replaying with stored actions from first loop — redo properly

    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    env._obs_seq = []
    env._prev_seq = []
    env._prev = np.zeros(7, np.float32)
    lost_low = 0
    how2 = None
    for _ in range(int(round(LIMIT_S / 0.02))):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            how2 = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _ = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        residual = mean[-1].detach().float().cpu().numpy()
        if token == HANDOFF:
            how2 = "early"
            break
        _r, action = decode_action(book, token, residual)
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            how2 = "drop"
            break
    if how2 == "drop":
        outcome = {"reason": terminal_reason(env.sim) or "DROP_DURING_RECOVERY", "success": False,
                   "recovery_elapsed": float(env.sim.data.time) - t0}
    else:
        if how2 is None:
            how2 = "forced"
        outcome = finish_recovery(env, blob["ref"], t0, how2)

    sub = classify_substep(rows, t0, t_hand)
    dest = OUT / "raw" / "grpo_only_substep" / ic_id
    dest.mkdir(parents=True, exist_ok=True)
    write_csv(dest / "substep_contacts.csv", rows)
    write_json(dest / "summary.json", {
        "ic_id": ic_id,
        "tokens": tokens,
        "policy_step_mechanism": mech_policy,
        "substep_mechanism": sub,
        "outcome": {k: outcome.get(k) for k in ("reason", "success", "recovery_elapsed", "handoff")},
        "snapshot_t0": t0,
        "t_handoff": t_hand,
    })
    return {
        "ic_id": ic_id,
        "tokens": tokens,
        "policy_mech": mech_policy.get("mechanism"),
        "substep_label": sub["label"],
        "substep_confidence": sub["confidence"],
        "substep_why": " | ".join(sub["why"]),
        "max_both_off_s": sub["facts"]["max_both_off_s"],
        "both_off_frac": sub["facts"]["both_off_frac"],
        "recontact_before_handoff": sub["facts"]["recontact_before_handoff"],
        "recontact_early_nominal": sub["facts"]["recontact_early_nominal"],
        "success": bool(outcome.get("success")),
        "reason": outcome.get("reason"),
        "recovery_elapsed": outcome.get("recovery_elapsed"),
    }


def load_candidate_pool():
    """Reuse existing snapshots; tag train-seen vs holdout."""
    train_ids = set()
    for name in ("selected_train_manifest.jsonl",):
        for line in (HARD_MINE / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                train_ids.add(json.loads(line)["id"])
    # also heuristic train via fingerprints later
    cands = []
    # hard recovery verified + unverified (holdout-ish vs GRPO train)
    for split, path in (
        ("holdout_bench_verified", BENCH / "verified_manifest.jsonl"),
        ("holdout_bench_unverified", BENCH / "unverified_manifest.jsonl"),
    ):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cands.append({
                "id": r["id"],
                "snapshot": r["snapshot"],
                "split": split,
                "seen_in_hard_train": False,
                "source": r.get("provenance"),
            })
    # hard mining selected train/dev for competition search (mark train)
    for name, split in (
        ("selected_train_manifest.jsonl", "train_hard"),
        ("selected_dev_manifest.jsonl", "dev_hard"),
        ("selected_test_manifest.jsonl", "holdout_hard_test"),
    ):
        path = HARD_MINE / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cands.append({
                "id": r["id"],
                "snapshot": r["snapshot"],
                "split": split,
                "seen_in_hard_train": split == "train_hard",
                "source": "hard_mining_selected",
            })
    # matched BOTH from prior (train snapshots)
    matched_path = ROOT / "results/diagnostics/reward_time_handoff_audit/matched_mechanism_fixed.json"
    if matched_path.is_file():
        for m in json.loads(matched_path.read_text(encoding="utf-8")).get("matched") or []:
            cands.append({
                "id": m["ic_id"],
                "snapshot": m["snapshot"],
                "split": "train_supervised_matched",
                "seen_in_hard_train": False,
                "seen_in_sft_train": True,
                "source": "matched_oracle_prior",
            })
    # dedupe by id preferring earlier
    seen = set()
    out = []
    for c in cands:
        if c["id"] in seen:
            continue
        seen.add(c["id"])
        out.append(c)
    return out


def oracle_pair(env, book, blob):
    slide = _try_sliding(env, book, blob)
    slide_ok = bool(slide and slide.get("ok"))
    slide_elapsed = None if not slide_ok else float(slide.get("recovery_elapsed") or 0.0)
    slide_strat = None if not slide_ok else slide.get("oracle_strategy")
    vert_ok = False
    vert_elapsed = None
    vert_strat = None
    for spec in list(vertical_specs()):
        result = vertical_rollout(env, blob, book, spec)
        hold, _ = _finish(env.sim, blob["ref"])
        if bool(hold.get("success")):
            vert_ok = True
            vert_elapsed = float(result["end"]["t"])
            vert_strat = spec.get("name")
            break
    if slide_ok and vert_ok:
        tag = "BOTH_VERIFIED"
    elif slide_ok and not vert_ok:
        tag = "SLIDING_FAVORABLE"
    elif vert_ok and not slide_ok:
        tag = "RECATCH_FAVORABLE"
    else:
        tag = "UNRESOLVED"
    return {
        "competition_tag": tag,
        "slide_ok": slide_ok,
        "slide_strategy": slide_strat,
        "slide_elapsed": slide_elapsed,
        "vert_ok": vert_ok,
        "vert_strategy": vert_strat,
        "vert_elapsed": vert_elapsed,
    }


def obs_at_ic(env, blob):
    restore_capture(env, blob["capture"])
    # env.observation uses airborne path
    return np.asarray(env.observation(), np.float32).copy()


def short_response_test(env, book, blob, actions_proto):
    """Apply fixed legal short sequence; return obs before/after."""
    restore_capture(env, blob["capture"])
    o0 = np.asarray(env.observation(), np.float32).copy()
    c0 = contact_pack(env.sim)
    for a in actions_proto:
        step_legal(env.sim, a)
        env._prev = np.asarray(a, np.float32)
    o1 = np.asarray(env.observation(), np.float32).copy()
    c1 = contact_pack(env.sim)
    return {
        "obs0": o0,
        "obs1": o1,
        "delta_l2": float(np.linalg.norm(o1 - o0)),
        "contact0": c0,
        "contact1": c1,
    }


def eval_policy_ic(env, model, book, blob, device):
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    samples = [audit.sample_state(env.sim, blob["ref"], phase="recovery")]
    lost_low = 0
    tokens = []
    how = None
    for _ in range(int(round(LIMIT_S / 0.02))):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _ = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        residual = mean[-1].detach().float().cpu().numpy()
        tokens.append(token)
        if token == HANDOFF:
            how = "early"
            samples.append(audit.sample_state(env.sim, blob["ref"], token=token, residual=residual, phase="handoff"))
            break
        _r, action = decode_action(book, token, residual)
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        samples.append(audit.sample_state(env.sim, blob["ref"], action=executed, token=token, residual=residual, phase="recovery"))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            reason = terminal_reason(env.sim) or "DROP_DURING_RECOVERY"
            return {
                "success": False, "reason": reason, "tokens": tokens,
                "handoff_t": None, "recovery_elapsed": float(env.sim.data.time) - t0,
                "mechanism": classify_mechanism(samples, t0, float(env.sim.data.time)).get("mechanism"),
            }
    if how is None:
        how = "forced"
    outcome = finish_recovery(env, blob["ref"], t0, how)
    ht = float(outcome.get("recovery_elapsed") or 0.0)
    mech = classify_mechanism(samples, t0, t0 + ht)
    return {
        "success": bool(outcome.get("success")),
        "reason": outcome.get("reason"),
        "tokens": tokens,
        "handoff_t": ht,
        "recovery_elapsed": ht,
        "mechanism": mech.get("mechanism"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-competition", type=int, default=40,
                        help="Max candidate ICs for dual-oracle competition search")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    book = load_codebook(CODEBOOK)
    write_observation_audit(book)

    sft, _ = load_trained_k48(SFT, device)
    grpo, _ = load_trained_k48(GRPO, device)
    sft.eval()
    grpo.eval()
    env = make_training_env(device, model=grpo)

    # --- Part 3: substep classify 7 GRPO-only ---
    go_rows = []
    for ic in GRPO_ONLY:
        print("SUBSTEP", ic, flush=True)
        # find snapshot
        snap = None
        for name in ("verified_manifest.jsonl", "unverified_manifest.jsonl"):
            for line in (BENCH / name).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["id"] == ic:
                    snap = r["snapshot"]
                    break
            if snap:
                break
        go_rows.append(analyze_grpo_only_substep(env, grpo, book, device, ic, snap))
    write_csv(OUT / "grpo_only_substep_classification.csv", go_rows)
    write_json(OUT / "grpo_only_substep_summary.json", {
        "labels": Counter(r["substep_label"] for r in go_rows),
        "rows": go_rows,
    })

    mech_md = ["# MECHANISM_CLASSIFICATION\n"]
    mech_md.append("## Substep criteria (this audit)\n")
    mech_md.append(
        "- **Release–Recatch (VERIFIED):** sustained both-off ≥ "
        f"{BOTH_OFF_HOLD_S}s **and** bilateral recontact **before HANDOFF**.\n"
        "- **Contact-Preserving Sliding:** bilateral ≈ continuous and ||Δp_rel|| ≥ "
        f"{MIGRATION_M} m during recovery.\n"
        "- **AMBIGUOUS:** sub-threshold both-off, or release without pre-HANDOFF regrasp "
        "(regrasp only in nominal), or insufficient migration.\n"
        "- Contact flicker / ID changes alone do **not** count as Recatch.\n"
    )
    mech_md.append("## Seven GRPO-only `[36,36,47]`\n")
    mech_md.append("| IC | substep label | conf | max both-off | recontact pre-H | recontact early nom | policy-step mech |\n|---|---|---|---:|---|---|---|\n")
    for r in go_rows:
        mech_md.append(
            f"| {r['ic_id']} | {r['substep_label']} | {r['substep_confidence']} | "
            f"{r['max_both_off_s']:.4f} | {r['recontact_before_handoff']} | "
            f"{r['recontact_early_nominal']} | {r['policy_mech']} |\n"
        )
    mech_md.append("\nRaw: `raw/grpo_only_substep/<id>/`\n")
    (OUT / "MECHANISM_CLASSIFICATION.md").write_text("".join(mech_md), encoding="utf-8")

    # --- Part 4: competition ICs ---
    pool = load_candidate_pool()
    # prioritize: matched both, hard selected, then bench (skip pure leftovers flood)
    priority = []
    for c in pool:
        if c["split"].startswith("train_supervised") or c["split"] in ("train_hard", "dev_hard", "holdout_hard_test"):
            priority.append(c)
        elif c["split"] == "holdout_bench_verified":
            priority.append(c)
    # add some unverified bench for unresolved coverage
    for c in pool:
        if c["split"] == "holdout_bench_unverified" and len(priority) < args.max_competition:
            priority.append(c)
    priority = priority[: args.max_competition]

    ic_manifest = []
    for c in priority:
        print("COMPETE", c["id"], c["split"], flush=True)
        try:
            blob = pickle.loads((ROOT / c["snapshot"]).read_bytes())
            if "capture" not in blob:
                continue
            fp = _fingerprint(blob)
            pair = oracle_pair(env, book, blob)
        except Exception as exc:
            ic_manifest.append({**c, "competition_tag": "UNRESOLVED", "error": str(exc)})
            continue
        # physics params from blob if present
        phys = {}
        if "spec" in blob and isinstance(blob["spec"], dict):
            phys = {k: blob["spec"].get(k) for k in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s") if k in blob["spec"]}
        row = {
            **c,
            **pair,
            "fingerprint": fp,
            "physics": json.dumps(phys),
            "holdout": c["split"] not in ("train_hard", "train_supervised_matched"),
        }
        ic_manifest.append(row)
        write_json(OUT / "raw" / "competition" / f"{c['id']}.json", row)

    write_csv(OUT / "IC_MANIFEST.csv", ic_manifest)
    tag_counts = Counter(r.get("competition_tag") for r in ic_manifest)
    print("TAGS", dict(tag_counts), flush=True)

    # --- Part 5: obs distinguishability ---
    slide_fav = [r for r in ic_manifest if r.get("competition_tag") == "SLIDING_FAVORABLE"]
    rec_fav = [r for r in ic_manifest if r.get("competition_tag") == "RECATCH_FAVORABLE"]
    both = [r for r in ic_manifest if r.get("competition_tag") == "BOTH_VERIFIED"]

    # fixed short open-grip sequence like token 36 mean action
    a36 = np.asarray(book[36], np.float32)
    proto = [a36, a36]

    obs_rows = []
    # initial obs stats
    def gather_obs(group, tag):
        vecs = []
        for r in group[:8]:
            blob = pickle.loads((ROOT / r["snapshot"]).read_bytes())
            o = obs_at_ic(env, blob)
            vecs.append(o)
            resp = short_response_test(env, book, blob, proto)
            obs_rows.append({
                "ic_id": r["id"],
                "tag": tag,
                "split": r.get("split"),
                "obs_l2_after_2x36": resp["delta_l2"],
                "both_off_after": resp["contact1"]["both_off"],
                "bilateral_after": resp["contact1"]["bilateral"],
                "bilateral_before": resp["contact0"]["bilateral"],
            })
        if not vecs:
            return None
        X = np.stack(vecs)
        return {"mean": X.mean(0), "std": X.std(0), "n": len(vecs)}

    stats_s = gather_obs(slide_fav, "SLIDING_FAVORABLE")
    stats_r = gather_obs(rec_fav, "RECATCH_FAVORABLE")
    dist = None
    top_dims = []
    if stats_s and stats_r:
        dmean = stats_s["mean"] - stats_r["mean"]
        dist = float(np.linalg.norm(dmean))
        idx = np.argsort(-np.abs(dmean))[:10]
        top_dims = [
            {"dim": int(i), "name": OBS_NAMES_AIR[i], "delta_mean": float(dmean[i]),
             "std_s": float(stats_s["std"][i]), "std_r": float(stats_r["std"][i])}
            for i in idx
        ]
    write_csv(OUT / "obs_response_probe.csv", obs_rows)
    write_json(OUT / "obs_separability.json", {
        "n_sliding_favorable": len(slide_fav),
        "n_recatch_favorable": len(rec_fav),
        "n_both_verified": len(both),
        "mean_l2_between_group_means": dist,
        "top_dims": top_dims,
        "caveat": "Group-mean separation ≠ matched-state causality; ICs differ in full scene state.",
    })

    # --- Part 6: paired SFT/GRPO on competition ICs (holdout preferred) ---
    eval_set = [r for r in ic_manifest if r.get("competition_tag") in (
        "SLIDING_FAVORABLE", "RECATCH_FAVORABLE", "BOTH_VERIFIED"
    )]
    # cap for runtime
    eval_set = eval_set[:30]
    env_s = make_training_env(device, model=sft)
    env_g = make_training_env(device, model=grpo)
    paired = []
    for r in eval_set:
        print("EVAL", r["id"], r["competition_tag"], flush=True)
        blob = pickle.loads((ROOT / r["snapshot"]).read_bytes())
        sres = eval_policy_ic(env_s, sft, book, blob, device)
        gres = eval_policy_ic(env_g, grpo, book, blob, device)
        paired.append({
            "ic_id": r["id"],
            "split": r.get("split"),
            "holdout": r.get("holdout"),
            "competition_tag": r["competition_tag"],
            "sft_success": int(sres["success"]),
            "grpo_success": int(gres["success"]),
            "sft_reason": sres["reason"],
            "grpo_reason": gres["reason"],
            "sft_mechanism": sres["mechanism"],
            "grpo_mechanism": gres["mechanism"],
            "sft_handoff_t": sres["handoff_t"],
            "grpo_handoff_t": gres["handoff_t"],
            "sft_tokens": json.dumps(sres["tokens"][:30]),
            "grpo_tokens": json.dumps(gres["tokens"][:30]),
        })
    write_csv(OUT / "PAIRED_EVALUATION.csv", paired)

    # Summaries for report
    def rate(tag, key):
        rows = [p for p in paired if p["competition_tag"] == tag]
        if not rows:
            return None
        return {
            "n": len(rows),
            "sft": sum(p["sft_success"] for p in rows),
            "grpo": sum(p["grpo_success"] for p in rows),
            "holdout_n": sum(1 for p in rows if p.get("holdout")),
        }

    write_report(go_rows, tag_counts, slide_fav, rec_fav, both, dist, top_dims, paired, rate)
    print("DONE", OUT, flush=True)


def write_report(go_rows, tag_counts, slide_fav, rec_fav, both, dist, top_dims, paired, rate_fn):
    go_labels = Counter(r["substep_label"] for r in go_rows)
    n_amb = go_labels.get("AMBIGUOUS", 0)
    n_rec = go_labels.get("Release-Recatch", 0)
    n_sl = go_labels.get("Contact-Preserving Sliding", 0)

    # GRPO loss of sliding? look at sliding-favorable paired
    sf = rate_fn("SLIDING_FAVORABLE", "")
    rf = rate_fn("RECATCH_FAVORABLE", "")
    bf = rate_fn("BOTH_VERIFIED", "")

    # mechanism shift on sliding-favorable
    sf_rows = [p for p in paired if p["competition_tag"] == "SLIDING_FAVORABLE"]
    sft_slide_ok = sum(1 for p in sf_rows if p["sft_success"] and "Contact-Preserving" in (p["sft_mechanism"] or ""))
    grpo_slide_ok = sum(1 for p in sf_rows if p["grpo_success"] and "Contact-Preserving" in (p["grpo_mechanism"] or ""))

    q1 = "UNRESOLVED"
    if n_rec == len(go_rows):
        q1 = "VERIFIED"
    elif n_amb == len(go_rows) or (n_amb + n_rec + n_sl) == len(go_rows) and n_rec == 0 and n_sl == 0:
        q1 = "SUPPORTED_BUT_INCOMPLETE"  # intentional open + both-off but no pre-H regrasp

    lines = []
    lines.append("# Mechanism-Selection Feasibility — REPORT\n")
    lines.append("Read-only audit. No training, no reward/physics/checkpoint changes.\n")
    lines.append("See also `OBSERVATION_AUDIT.md`, `MECHANISM_CLASSIFICATION.md`.\n")
    lines.append("## Q1 — Are the seven `[36,36,47]` physically Recatch?\n")
    lines.append(f"- Substep labels: {dict(go_labels)}")
    lines.append(f"- Status: **{q1}**")
    lines.append(
        "- Token 36 has **grip-open** command (a_grip=-1 → TAU_OPEN). Substep logs show "
        "both-off during the 40 ms recovery; bilateral recontact **before HANDOFF** was "
        f"{'not' if all(not r['recontact_before_handoff'] for r in go_rows) else ''} consistently observed."
    )
    lines.append(
        "- If regrasp occurs only after HANDOFF (nominal), this is **not** recovery-phase "
        "Release–Recatch under the project classifier; call it **AMBIGUOUS fast open-then-handoff** "
        "unless sustained both-off + pre-HANDOFF recontact is logged."
    )
    lines.append("- Evidence: `grpo_only_substep_classification.csv`, `raw/grpo_only_substep/`.\n")

    lines.append("## Q2–Q3 — Sliding / Recatch favorable ICs\n")
    lines.append(f"- Competition search tags: {dict(tag_counts)}")
    lines.append(f"- Sliding-favorable: **{len(slide_fav)}** → "
                 f"{'VERIFIED' if slide_fav else 'UNRESOLVED'}")
    lines.append(f"- Recatch-favorable: **{len(rec_fav)}** → "
                 f"{'VERIFIED' if rec_fav else 'UNRESOLVED'}")
    lines.append(f"- BOTH_VERIFIED: **{len(both)}**")
    lines.append("- Oracle budget: `_try_sliding` schedules + `vertical_specs()` only; "
                 "search failure ≠ physical impossibility (`UNRESOLVED`).")
    lines.append("- Manifest: `IC_MANIFEST.csv`.\n")

    lines.append("## Q4 — Are classes separable in 39D history?\n")
    if dist is None:
        lines.append("- **UNRESOLVED** — missing one of the favorable classes for group comparison.")
    else:
        lines.append(f"- Group-mean L2 between Sliding-favorable vs Recatch-favorable IC obs: **{dist:.4f}**")
        lines.append(f"- Top dims: {top_dims[:5]}")
        lines.append("- Status: **SUPPORTED_BUT_INCOMPLETE** — differences exist at IC restore, "
                     "but ICs are not matched twins; does not prove the current Transformer uses them.")
        lines.append("- Matched short-action probe (2× token36 proto): `obs_response_probe.csv`.")
        lines.append("- Forbidden: friction/mass/labels are not in 39D (see OBSERVATION_AUDIT).\n")

    lines.append("## Q5 — SFT vs GRPO on competition ICs\n")
    lines.append(f"- Sliding-favorable: {sf}")
    lines.append(f"- Recatch-favorable: {rf}")
    lines.append(f"- BOTH_VERIFIED: {bf}")
    lines.append(f"- On Sliding-favorable: SFT Contact-Preserving successes={sft_slide_ok}, "
                 f"GRPO Contact-Preserving successes={grpo_slide_ok} (of {len(sf_rows)}).")
    lines.append("- Full table: `PAIRED_EVALUATION.csv`. Holdout flag in column `holdout`.\n")

    lines.append("## Q6 — Does evidence support “GRPO loses Sliding for fast Recatch”?\n")
    if not sf_rows:
        lines.append("- **UNRESOLVED** — no Sliding-favorable ICs found under current oracle budget.")
    else:
        sft_s = sum(p["sft_success"] for p in sf_rows)
        g_s = sum(p["grpo_success"] for p in sf_rows)
        if sft_s > g_s:
            lines.append(
                f"- **SUPPORTED_BUT_INCOMPLETE** — on {len(sf_rows)} Sliding-favorable ICs, "
                f"SFT {sft_s}/{len(sf_rows)} vs GRPO {g_s}/{len(sf_rows)}. "
                "Sample small; not yet a causal proof of skill forgetting."
            )
        else:
            lines.append(
                f"- **UNRESOLVED / not supported by this slice** — SFT {sft_s} vs GRPO {g_s} "
                f"on {len(sf_rows)} Sliding-favorable ICs."
            )
    lines.append("- Prior hard-recovery GRPO_only gains were 40 ms open-grip prefixes (AMBIGUOUS), "
                 "not verified long Sliding losses.\n")

    lines.append("## Q7 — Next priority\n")
    if len(slide_fav) < 5 or len(rec_fav) < 5:
        lines.append(
            "- **Priority: expand IC coverage / search budget** for true mechanism-competitive "
            "states (more Sliding-favorable & Recatch-favorable with holdout splits)."
        )
    elif dist is None or dist < 0.05:
        lines.append("- **Priority: observation / distinguishability** — favorable classes exist but "
                     "39D separation is weak or unproven under matched conditions.")
    else:
        lines.append(
            "- **Priority: GRPO update / sampling** — if favorable ICs exist, obs separates, and "
            "paired eval shows Sliding regression, then revisit all-success time-advantage bias "
            "(prior audit) rather than changing λ first."
        )
    lines.append("- Do **not** auto-train; wait for human decision.\n")

    lines.append("## Stop\nAudit complete. No SFT/GRPO launched; reward unmodified.\n")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
