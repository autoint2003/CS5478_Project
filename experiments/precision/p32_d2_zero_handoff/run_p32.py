"""P32 — D2 zero-action HANDOFF and over-control diagnosis.

Diagnostic only: no training, no RL, no architecture/M111 changes, no relabeling.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from evaluation.grasp_geometry import measure  # noqa: E402
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import (  # noqa: E402
    make_env,
    replay_actions,
)
from experiments.precision.p27_history_multiskill_transformer.run_p27 import (  # noqa: E402
    load_d2_eval_eps,
    load_npz_flex,
)
from experiments.precision.p28_multiskill_distillation.harness import (  # noqa: E402
    policy_step_K,
    rollout_corrected,
)
from experiments.precision.p29_full_history_multiskill.harness import rollout as rollout_fh  # noqa: E402
from experiments.precision.p29_full_history_multiskill.run_p29 import load_fh_model  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import load_episode_npz  # noqa: E402

OUT = ROOT / "results/diagnostics/p32_d2_zero_handoff"
P22 = ROOT / "results/diagnostics/p22_full_horizon_sft"
P23 = ROOT / "results/diagnostics/p23_c_d2_sft"
P28 = ROOT / "results/diagnostics/p28_multiskill_distillation"
P29 = ROOT / "results/diagnostics/p29_full_history_multiskill"
P31 = ROOT / "results/diagnostics/p31_skill_interference"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"

CKPT_P22 = P22 / "checkpoints/A_natural/checkpoint_best.pt"
CKPT_P28 = P28 / "checkpoints/C_distill_corr_K32/checkpoint_best.pt"
CKPT_P29 = P29 / "checkpoints/ablation_L2_last/checkpoint_best.pt"
CKPT_P31 = P31 / "checkpoints/M2_pcgrad/checkpoint_best.pt"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _d2_start(env, ep) -> None:
    """Canonical D2 closed-loop start (matches P28 harness)."""
    restore_capture(env, ep["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)


def _initial_state_probe(env, ep) -> dict[str, Any]:
    _d2_start(env, ep)
    sim = env.sim
    m = measure(sim, U_TGT, P_TGT)
    cs = contact_summary(sim)
    obj = np.asarray(sim.data.xpos[sim.ids.object_body], float)
    oq = np.asarray(sim.data.xquat[sim.ids.object_body], float)
    # linear velocity of object free joint if present
    try:
        jnt = int(sim.ids.object_free_jnt) if hasattr(sim.ids, "object_free_jnt") else None
        ov = np.asarray(sim.data.qvel[jnt : jnt + 3], float) if jnt is not None else np.zeros(3)
    except Exception:
        ov = np.zeros(3)
    hand = np.asarray(m.get("p_c_h") or obj, float)
    bil = bool(cs.get("bilateral"))
    obs = np.asarray(
        observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, env._hist),
        np.float32,
    )
    return {
        "obj_pos_x": float(obj[0]),
        "obj_pos_y": float(obj[1]),
        "obj_pos_z": float(obj[2]),
        "obj_quat": "|".join(f"{x:.6f}" for x in oq),
        "obj_lin_vel_norm": float(np.linalg.norm(ov[:3])),
        "hand_obj_dist": float(np.linalg.norm(obj - np.asarray(hand[:3], float))),
        "e_pos_mm": float(m.get("e_pos_mm") or 0.0),
        "e_theta_deg": float(m.get("e_theta_deg") or 0.0),
        "bilateral": int(bil),
        "contact_left": cs.get("nL"),
        "contact_right": cs.get("nR"),
        "fsm_phase": str(sim.fsm.phase),
        "fg_cmd": float(sim.fsm.fg_cmd),
        "prev0_norm": float(np.linalg.norm(env._prev)),
        "obs0_norm": float(np.linalg.norm(obs)),
    }


def _drop_time_from_hand(hand: dict) -> float | None:
    reason = hand.get("std_reason") or hand.get("reason") or ""
    if "DROP" not in str(reason) and "LOST" not in str(reason) and "GROUND" not in str(reason):
        return None
    samples = hand.get("samples") or []
    for s in samples:
        if not s.get("bilateral", True):
            return float(s.get("t_rel", 0.0))
    if samples:
        return float(samples[-1].get("t_rel", 0.0))
    return None


def zero_handoff_once(env, ep) -> dict[str, Any]:
    """Legal step-0 HANDOFF: restore D2 start, no motion, evaluate_handoff_with_axis."""
    probe = _initial_state_probe(env, ep)
    # re-restore for clean eval (probe already mutated hist)
    _d2_start(env, ep)
    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=8)
    return {
        **probe,
        "SUCCESS_NEW": int(bool(hand.get("SUCCESS_NEW"))),
        "SUCCESS_12S": int(bool(hand.get("SUCCESS_12S"))),
        "eval_reason": hand.get("std_reason") or hand.get("reason"),
        "drop_time_s": _drop_time_from_hand(hand),
        "pre_bilateral": int(bool(hand.get("pre_bilateral"))),
        "n_motion": 0,
        "termination_kind": "policy_handoff",
        "SUCCESS_NEW_policy_handoff": int(bool(hand.get("SUCCESS_NEW"))),
        "immediate_handoff": 1,
    }


def expert_replay_once(env, ep) -> dict[str, Any]:
    acts = np.asarray(ep["actions"], np.float32).reshape(-1, 7)
    # Detect near-zero / HANDOFF-like leading actions
    norms = np.linalg.norm(acts, axis=1) if len(acts) else np.array([])
    n_nonzero = int(np.sum(norms > 1e-4)) if len(norms) else 0
    immediate_zero = int(len(acts) == 0 or (len(acts) > 0 and norms[0] <= 1e-4 and n_nonzero == 0))
    r = replay_actions(env, ep["capture"], ep["ref"], acts, stabilize_n=8)
    return {
        "n_expert_actions": int(len(acts)),
        "n_nonzero_actions": n_nonzero,
        "expert_immediate_zero_tape": immediate_zero,
        "expert_first_action_norm": float(norms[0]) if len(norms) else 0.0,
        "expert_mean_action_norm": float(np.mean(norms)) if len(norms) else 0.0,
        "SUCCESS_NEW": int(bool(r["SUCCESS_NEW"])),
        "SUCCESS_12S": int(bool(r["SUCCESS_12S"])),
        "eval_reason": r.get("reason"),
        "pre_pos_mm": r.get("pre_pos_mm"),
        "pre_theta": r.get("pre_theta"),
    }


def first_logits(model, book, rho, env, ep, device, *, mode: str, K: int) -> dict[str, Any]:
    _d2_start(env, ep)
    obs = np.asarray(
        observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
        np.float32,
    )
    prev = env._prev.copy()
    with torch.no_grad():
        if mode == "full":
            from experiments.precision.p29_full_history_multiskill.harness import policy_step_full

            tok, r, action, logit = policy_step_full(model, book, rho, [obs], [prev], device)
        else:
            tok, r, action, logit = policy_step_K(model, book, rho, [obs], [prev], device, K)
    logit_t = torch.as_tensor(logit)
    probs = F.softmax(logit_t, dim=-1).numpy()
    top5 = np.argsort(-probs)[:5]
    return {
        "first_token": int(tok),
        "first_is_handoff": int(tok == HANDOFF),
        "handoff_logit": float(logit[HANDOFF]),
        "handoff_prob": float(probs[HANDOFF]),
        "top5_tokens": "|".join(str(int(t)) for t in top5),
        "top5_probs": "|".join(f"{float(probs[t]):.4f}" for t in top5),
        "first_resid_norm": float(np.linalg.norm(r)),
        "first_action": None if action is None else "|".join(f"{x:.4f}" for x in action),
        "first_action_norm": 0.0 if action is None else float(np.linalg.norm(action)),
    }


def policy_rollout_detail(model, book, rho, env, ep, device, *, mode: str, K: int, max_steps: int = 800) -> dict:
    first = first_logits(model, book, rho, env, ep, device, mode=mode, K=K)
    if mode == "full":
        r = rollout_fh(env, model, book, rho, ep, device=device, mode="full", K=K, start_kind="d2", max_steps=max_steps)
    else:
        r = rollout_corrected(env, model, book, rho, ep, device=device, K=K, start_kind="d2", max_steps=max_steps)
    return {
        **first,
        "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
        "SUCCESS_12S": int(r["SUCCESS_12S"]),
        "SUCCESS_NEW_policy_handoff": int(r["SUCCESS_NEW_policy_handoff"]),
        "n_motion": int(r["n_motion"]),
        "termination_kind": r["termination_kind"],
        "explicit_handoff": int(r["explicit_handoff"]),
        "forced_timeout": int(r["forced_timeout"]),
        "immediate_handoff": int(r["immediate_handoff"]),
        "tokens_head": r.get("tokens_head"),
        "eval_reason": r.get("eval_reason"),
        "max_token_run": r.get("max_token_run"),
    }


def one_action_then_handoff(model, book, rho, env, ep, device, *, mode: str, K: int) -> dict:
    """Counterfactual: execute exactly one policy action (if motion), then HANDOFF eval."""
    _d2_start(env, ep)
    obs = np.asarray(
        observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
        np.float32,
    )
    prev = env._prev.copy()
    if mode == "full":
        from experiments.precision.p29_full_history_multiskill.harness import policy_step_full

        tok, r, action, _ = policy_step_full(model, book, rho, [obs], [prev], device)
    else:
        tok, r, action, _ = policy_step_K(model, book, rho, [obs], [prev], device, K)
    n_motion = 0
    if action is not None:
        step_legal(env.sim, action)
        env._prev = action
        n_motion = 1
    hand = evaluate_handoff_with_axis(env.sim, ep["ref"], stabilize_n=8)
    return {
        "first_token": int(tok),
        "n_motion": n_motion,
        "SUCCESS_NEW": int(bool(hand.get("SUCCESS_NEW"))),
        "SUCCESS_12S": int(bool(hand.get("SUCCESS_12S"))),
        "eval_reason": hand.get("std_reason") or hand.get("reason"),
        "drop_time_s": _drop_time_from_hand(hand),
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== P32 device={device} (diagnostic only) ===", flush=True)
    book, rho, _ = load_m111_book(M111_META)

    p23 = json.loads((P23 / "COMBINED_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    d2_rows = [r for r in p23["trajectories"] if r.get("family") == "D2"]
    d2_eps = load_d2_eval_eps({"d2": d2_rows})
    assert len(d2_eps) == 17, f"expected 17 D2 val ICs, got {len(d2_eps)}"
    print(f"Loaded {len(d2_eps)} D2 val ICs", flush=True)

    # ----- Phase 0 -----
    print("=== Phase 0: evaluator + restore audit ===", flush=True)
    env = make_env()
    restore_rows = []
    for ep in d2_eps[:5]:
        _d2_start(env, ep)
        q1 = np.asarray(env.sim.data.qpos, float).copy()
        v1 = np.asarray(env.sim.data.qvel, float).copy()
        obs1 = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
            np.float32,
        )
        prev1 = env._prev.copy()
        phase1 = str(env.sim.fsm.phase)
        _d2_start(env, ep)
        q2 = np.asarray(env.sim.data.qpos, float).copy()
        v2 = np.asarray(env.sim.data.qvel, float).copy()
        obs2 = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
            np.float32,
        )
        restore_rows.append(
            {
                "traj_id": ep["traj_id"],
                "qpos_max_abs_diff": float(np.max(np.abs(q1 - q2))),
                "qvel_max_abs_diff": float(np.max(np.abs(v1 - v2))),
                "obs_max_abs_diff": float(np.max(np.abs(obs1 - obs2))),
                "prev_norm": float(np.linalg.norm(prev1)),
                "fsm_phase": phase1,
                "deterministic": int(np.allclose(q1, q2) and np.allclose(v1, v2) and np.allclose(obs1, obs2)),
            }
        )
    # Also: double zero-HANDOFF on first IC
    ep0 = d2_eps[0]
    z1 = zero_handoff_once(env, ep0)
    z2 = zero_handoff_once(env, ep0)
    restore_rows.append(
        {
            "traj_id": ep0["traj_id"] + "__zero_handoff_repeat",
            "qpos_max_abs_diff": None,
            "qvel_max_abs_diff": None,
            "obs_max_abs_diff": None,
            "prev_norm": 0.0,
            "fsm_phase": "n/a",
            "deterministic": int(z1["SUCCESS_NEW"] == z2["SUCCESS_NEW"] and z1["eval_reason"] == z2["eval_reason"]),
            "z1_success": z1["SUCCESS_NEW"],
            "z2_success": z2["SUCCESS_NEW"],
            "z1_reason": z1["eval_reason"],
            "z2_reason": z2["eval_reason"],
        }
    )
    _write_csv(OUT / "SNAPSHOT_RESTORE_TEST.csv", restore_rows)
    n_det = sum(int(r["deterministic"]) for r in restore_rows)
    critical_bug = n_det < len(restore_rows)
    audit = f"""# EVALUATOR_AUDIT

Generated: {_utc()}

## D2 IC loading

- Manifest: `results/diagnostics/p23_c_d2_sft/COMBINED_DATASET_MANIFEST.json` (family=D2, split=val → 17).
- Loader: `load_d2_eval_eps` resolves IC pickles under p14/p15/p16 `ics/`.
- Each ep: `capture`, `ref`, continuous `actions` (expert).

## Snapshot restoration

- `restore_capture(env, capture)` restores physics + FSM + switch + drop latches.
- D2 closed-loop then **forces** `env._hist = ObservableObsState(DT_POLICY)` and `env._prev = 0`.
- This matches P28/P29/P31 D2 evaluation (capture prev/hist discarded for policy start).

## HANDOFF evaluator

- Token `HANDOFF=111` stops the policy loop (`termination_kind=policy_handoff`).
- Then `evaluate_handoff_with_axis(sim, ref, stabilize_n=8)`:
  - grip stabilize → `apply_handoff` → 12 s nominal → HOLD window.
  - `SUCCESS_NEW` ≡ historical STRICT (θ≤10°, pos≤12 mm hold).
- `SUCCESS_NEW_policy_handoff` requires **both** physical SUCCESS_NEW **and** explicit policy HANDOFF.
- Timeout physics success is **not** credited as policy HANDOFF.

## Step-0 HANDOFF legality

- **Allowed.** Emitting HANDOFF as the first decision (`n_motion=0`) is a legal policy action.
- Evaluator uses the **same** `evaluate_handoff_with_axis` path as a later HANDOFF.
- P32 zero-action baseline: restore D2 start → call `evaluate_handoff_with_axis` with no motion
  (equivalent to immediate legal HANDOFF; not a privileged oracle).

## Restore determinism

- Tested {len(restore_rows)} cases: deterministic={n_det}/{len(restore_rows)}.
- Critical bug found: **{"YES — STOP" if critical_bug else "NO"}**.

## Integrity constraints for P32

- No training / RL / architecture / M111 / dataset relabeling.
- Original MuJoCo physics, controller, HANDOFF semantics, 12 s STRICT preserved.
"""
    _write_md(OUT / "EVALUATOR_AUDIT.md", audit)
    if critical_bug:
        print("CRITICAL: restore non-deterministic — stopping", flush=True)
        return

    # ----- Phase 1+2: zero HANDOFF + expert -----
    print("=== Phase 1–2: zero HANDOFF + expert replay ===", flush=True)
    zero_rows = []
    expert_rows = []
    class_rows = []
    for ep in d2_eps:
        z = zero_handoff_once(env, ep)
        z_rep = zero_handoff_once(env, ep)
        zero_rows.append(
            {
                "traj_id": ep["traj_id"],
                "lineage_id": ep["lineage_id"],
                "ic_path": str(ep.get("ic_path", "")),
                **{k: z[k] for k in z if k not in ("obj_quat",)},
                "obj_quat": z["obj_quat"],
                "repeat_match": int(z["SUCCESS_NEW"] == z_rep["SUCCESS_NEW"] and z["eval_reason"] == z_rep["eval_reason"]),
            }
        )
        ex = expert_replay_once(env, ep)
        expert_rows.append({"traj_id": ep["traj_id"], "lineage_id": ep["lineage_id"], **ex})
        z_ok = bool(z["SUCCESS_NEW"])
        e_ok = bool(ex["SUCCESS_NEW"])
        if z_ok:
            cls = "H0"
            note = "immediate HANDOFF STRICT succeeds; correction not required under physical criterion"
        elif e_ok:
            cls = "H1"
            note = "immediate HANDOFF fails; expert correction succeeds — correction necessary vs zero baseline"
        elif ex["n_expert_actions"] == 0:
            cls = "HU"
            note = "immediate fails; expert tape empty/unavailable"
        else:
            cls = "H2"
            note = "immediate HANDOFF and expert both fail"
        class_rows.append(
            {
                "traj_id": ep["traj_id"],
                "lineage_id": ep["lineage_id"],
                "class": cls,
                "zero_SUCCESS_NEW": int(z_ok),
                "expert_SUCCESS_NEW": int(e_ok),
                "expert_n_actions": ex["n_expert_actions"],
                "expert_n_nonzero": ex["n_nonzero_actions"],
                "note": note,
            }
        )
        print(f"  {ep['traj_id'][:48]} class={cls} z={int(z_ok)} e={int(e_ok)}", flush=True)

    _write_csv(OUT / "D2_ZERO_HANDOFF.csv", zero_rows)
    _write_csv(OUT / "D2_EXPERT_REPLAY.csv", expert_rows)
    _write_csv(OUT / "D2_IC_CLASSIFICATION.csv", class_rows)
    by_cls = Counter(r["class"] for r in class_rows)
    both_ok = sum(1 for r in class_rows if r["zero_SUCCESS_NEW"] and r["expert_SUCCESS_NEW"])
    necessity = f"""# D2_ACTION_NECESSITY

Generated: {_utc()}

## Counts

| Class | n | Meaning |
|---|---:|---|
| H0 | {by_cls.get('H0',0)} | Immediate HANDOFF STRICT succeeds |
| H1 | {by_cls.get('H1',0)} | Needs correction vs immediate HANDOFF |
| H2 | {by_cls.get('H2',0)} | Immediate + expert both fail |
| HU | {by_cls.get('HU',0)} | Insufficient evidence |

Both immediate HANDOFF and expert succeed (multi-valid): **{both_ok}**.

## Interpretation

- Expert correction sequences are **not** required merely because they appear in the dataset.
- If H0 and expert both succeed, the IC admits multiple valid strategies (including zero correction).
- Historical expert trajectories were **not** modified.

## Expert behavior vs zero baseline

See `D2_EXPERT_REPLAY.csv` for per-IC action counts and STRICT.
Many D2 expert tapes may contain nonzero correction even when H0 holds — that is optional
over-control in the demonstration, not proof that correction is necessary.
"""
    _write_md(OUT / "D2_ACTION_NECESSITY.md", necessity)

    class_of = {r["traj_id"]: r["class"] for r in class_rows}
    zero_ok = {r["traj_id"]: bool(r["SUCCESS_NEW"]) for r in zero_rows}

    # ----- Phase 3: frozen policies -----
    print("=== Phase 3: frozen policy comparison ===", flush=True)
    policies = []
    # (tag, model, mode, K)
    m22, _, _ = load_m111(CKPT_P22, device)
    policies.append(("P22_A_natural", m22, "window", 160))
    m28, _, _ = load_m111(CKPT_P28, device)
    policies.append(("P28_C_K32", m28, "window", 32))
    m31, _, _ = load_m111(CKPT_P31, device)
    policies.append(("P31_M2_pcgrad", m31, "window", 160))
    try:
        m29 = load_fh_model(CKPT_P29, book, rho, device)
        policies.append(("P29_L2_last", m29, "full", 0))
    except Exception as exc:
        print("P29 load failed:", exc, flush=True)

    first_rows = []
    logit_rows = []
    roll_rows = []
    p29_imm = []
    for tag, model, mode, K in policies:
        model.eval()
        print(f"  eval {tag}", flush=True)
        for ep in d2_eps:
            det = policy_rollout_detail(model, book, rho, env, ep, device, mode=mode, K=K)
            row = {
                "condition": tag,
                "traj_id": ep["traj_id"],
                "lineage_id": ep["lineage_id"],
                "ic_class": class_of[ep["traj_id"]],
                "zero_handoff_ok": int(zero_ok[ep["traj_id"]]),
                **det,
            }
            roll_rows.append(row)
            first_rows.append(
                {
                    "condition": tag,
                    "traj_id": ep["traj_id"],
                    "ic_class": class_of[ep["traj_id"]],
                    "first_token": det["first_token"],
                    "first_is_handoff": det["first_is_handoff"],
                    "first_action_norm": det["first_action_norm"],
                    "first_action": det["first_action"],
                    "tokens_head": det["tokens_head"],
                }
            )
            logit_rows.append(
                {
                    "condition": tag,
                    "traj_id": ep["traj_id"],
                    "ic_class": class_of[ep["traj_id"]],
                    "handoff_logit": det["handoff_logit"],
                    "handoff_prob": det["handoff_prob"],
                    "top5_tokens": det["top5_tokens"],
                    "top5_probs": det["top5_probs"],
                    "first_token": det["first_token"],
                    "immediate_handoff": det["immediate_handoff"],
                    "SUCCESS_NEW_policy_handoff": det["SUCCESS_NEW_policy_handoff"],
                    "SUCCESS_NEW": det["SUCCESS_NEW"],
                }
            )
            if tag == "P29_L2_last" and det["immediate_handoff"]:
                p29_imm.append(row)

    _write_csv(OUT / "POLICY_FIRST_ACTIONS.csv", first_rows)
    _write_csv(OUT / "HANDOFF_LOGIT_ANALYSIS.csv", logit_rows)
    _write_csv(OUT / "D2_POLICY_ROLLOUTS.csv", roll_rows)

    # P29 audit
    p29_ok = [r for r in p29_imm if r["SUCCESS_NEW_policy_handoff"]]
    p29_fail = [r for r in p29_imm if not r["SUCCESS_NEW_policy_handoff"]]
    p29_md = f"""# P29_IMMEDIATE_HANDOFF_AUDIT

Generated: {_utc()}

## Claim

P29 `ablation_L2_last` was reported with D2 ≈ 9/17 via **immediate HANDOFF** (mean n_motion≈0).

## Re-evaluation on current 17 D2 val ICs

- Immediate HANDOFF events: **{len(p29_imm)}**
- Of those, explicit-HANDOFF STRICT success: **{len(p29_ok)}**
- Immediate HANDOFF but physical STRICT fail: **{len(p29_fail)}**

### Immediate HANDOFF ICs

| traj_id | class | HANDOFF executed | entered nominal (eval ran) | STRICT | reason |
|---|---|---:|---|---:|---|
"""
    for r in p29_imm:
        p29_md += (
            f"| {r['traj_id']} | {r['ic_class']} | 1 | yes (`evaluate_handoff_with_axis`) | "
            f"{r['SUCCESS_NEW_policy_handoff']} | {r['eval_reason']} |\n"
        )
    p29_md += f"""
## Classification of immediate-HANDOFF ICs

- H0 among immediate: {sum(1 for r in p29_imm if r['ic_class']=='H0')}
- H1 among immediate: {sum(1 for r in p29_imm if r['ic_class']=='H1')}
- H2 among immediate: {sum(1 for r in p29_imm if r['ic_class']=='H2')}

## Why failures (if any)

Immediate HANDOFF is **executed** and the evaluator **does** enter nominal 12 s control.
Failures are physical (DROP/LOST/geometry), not a metric that credits HANDOFF without STRICT.
Do not count immediate HANDOFF as success unless `SUCCESS_NEW_policy_handoff=1`.

Checkpoint: `{CKPT_P29.relative_to(ROOT).as_posix()}` (FullHistoryPolicy, sinusoidal, objective=L2_last).
"""
    _write_md(OUT / "P29_IMMEDIATE_HANDOFF_AUDIT.md", p29_md)

    # ----- Phase 4: over-control counterfactuals -----
    print("=== Phase 4: over-control counterfactuals ===", flush=True)
    over_rows = []
    first_eff = []
    # Focus policies that matter for diagnosis
    focus = [(t, m, mo, k) for t, m, mo, k in policies if t in ("P22_A_natural", "P31_M2_pcgrad", "P28_C_K32", "P29_L2_last")]
    for tag, model, mode, K in focus:
        for ep in d2_eps:
            cls = class_of[ep["traj_id"]]
            # Branch A always = zero HANDOFF (already measured)
            z = next(r for r in zero_rows if r["traj_id"] == ep["traj_id"])
            # Branch B = policy
            pol = next(r for r in roll_rows if r["condition"] == tag and r["traj_id"] == ep["traj_id"])
            avoidable = int(cls == "H0" and z["SUCCESS_NEW"] == 1 and pol["SUCCESS_NEW_policy_handoff"] == 0)
            over_rows.append(
                {
                    "condition": tag,
                    "traj_id": ep["traj_id"],
                    "ic_class": cls,
                    "branchA_zero_SUCCESS_NEW": z["SUCCESS_NEW"],
                    "branchA_reason": z["eval_reason"],
                    "branchB_pol_SUCCESS_NEW_handoff": pol["SUCCESS_NEW_policy_handoff"],
                    "branchB_pol_SUCCESS_NEW": pol["SUCCESS_NEW"],
                    "branchB_n_motion": pol["n_motion"],
                    "branchB_immediate_handoff": pol["immediate_handoff"],
                    "branchB_first_token": pol["first_token"],
                    "branchB_reason": pol["eval_reason"],
                    "avoidable_failure_by_overcontrol": avoidable,
                }
            )
            if cls in ("H0", "H1") and (avoidable or (cls == "H1" and not pol["SUCCESS_NEW_policy_handoff"])):
                fa = one_action_then_handoff(model, book, rho, env, ep, device, mode=mode, K=K)
                first_eff.append(
                    {
                        "condition": tag,
                        "traj_id": ep["traj_id"],
                        "ic_class": cls,
                        "zero_SUCCESS_NEW": z["SUCCESS_NEW"],
                        "one_action_then_handoff_SUCCESS_NEW": fa["SUCCESS_NEW"],
                        "one_action_token": fa["first_token"],
                        "one_action_n_motion": fa["n_motion"],
                        "one_action_reason": fa["eval_reason"],
                        "full_policy_SUCCESS_NEW_handoff": pol["SUCCESS_NEW_policy_handoff"],
                        "damaged_by_first_action": int(z["SUCCESS_NEW"] == 1 and fa["SUCCESS_NEW"] == 0),
                        "helped_by_first_action": int(z["SUCCESS_NEW"] == 0 and fa["SUCCESS_NEW"] == 1),
                    }
                )

    _write_csv(OUT / "OVERCONTROL_COUNTERFACTUAL.csv", over_rows)
    _write_csv(OUT / "FIRST_ACTION_EFFECT.csv", first_eff)

    def _avoid_count(tag):
        return sum(1 for r in over_rows if r["condition"] == tag and r["avoidable_failure_by_overcontrol"])

    over_md = f"""# OVERCONTROL_DIAGNOSIS

Generated: {_utc()}

## Method

For each IC and policy, compare independent restores:
- **Branch A:** immediate legal HANDOFF (zero motion).
- **Branch B:** frozen policy until HANDOFF/timeout.
Optional: one policy action then HANDOFF.

Avoidable over-control := IC is **H0** (Branch A STRICT) but Branch B fails explicit-HANDOFF STRICT.

## Avoidable failures (H0 destroyed by policy)

| Policy | avoidable / H0 |
|---|---|
| P22_A_natural | {_avoid_count('P22_A_natural')} / {by_cls.get('H0',0)} |
| P31_M2_pcgrad | {_avoid_count('P31_M2_pcgrad')} / {by_cls.get('H0',0)} |
| P28_C_K32 | {_avoid_count('P28_C_K32')} / {by_cls.get('H0',0)} |
| P29_L2_last | {_avoid_count('P29_L2_last')} / {by_cls.get('H0',0)} |

## First-action damage

Among probed cases, first-action damages H0: **{sum(1 for r in first_eff if r.get('damaged_by_first_action'))}**;
helps H1: **{sum(1 for r in first_eff if r.get('helped_by_first_action'))}**.

See `OVERCONTROL_COUNTERFACTUAL.csv`, `FIRST_ACTION_EFFECT.csv`.
"""
    _write_md(OUT / "OVERCONTROL_DIAGNOSIS.md", over_md)

    # ----- Phase 5: dataset supervision audit -----
    print("=== Phase 5: D2 HANDOFF supervision audit ===", flush=True)
    # Map traj_id -> class for train+val using zero HANDOFF on all D2 with IC
    all_d2_eps = load_d2_eval_eps({"d2": d2_rows})  # val only from loader
    # Also attempt train ICs: temporarily poke loader by marking split
    train_rows_m = [r for r in d2_rows if r["split"] == "train"]
    # reuse loader logic by faking split
    fake = [{**r, "split": "val"} for r in train_rows_m]
    train_eps = load_d2_eval_eps({"d2": fake})
    # classify train via zero+expert quickly
    train_class = {}
    for ep in train_eps:
        z = zero_handoff_once(env, ep)
        ex = expert_replay_once(env, ep)
        if z["SUCCESS_NEW"]:
            train_class[ep["traj_id"]] = "H0"
        elif ex["SUCCESS_NEW"]:
            train_class[ep["traj_id"]] = "H1"
        else:
            train_class[ep["traj_id"]] = "H2"

    # NPZ supervision stats
    sup_rows = []
    for split_name, rows in (("train", [r for r in d2_rows if r["split"] == "train"]), ("val", [r for r in d2_rows if r["split"] == "val"])):
        for r in rows:
            try:
                epn = load_npz_flex(P23, r["npz_relpath"])
            except Exception:
                continue
            toks = np.asarray(epn["tokens"])
            n = len(toks)
            n_hand = int(np.sum(toks == HANDOFF))
            t_hand = int(np.where(toks == HANDOFF)[0][0]) if n_hand else -1
            step0_hand = int(n > 0 and int(toks[0]) == HANDOFF)
            cls = class_of.get(r["traj_id"]) or train_class.get(r["traj_id"], "UNK")
            unnecessary = int(cls == "H0" and not step0_hand and n > 1)
            sup_rows.append(
                {
                    "traj_id": r["traj_id"],
                    "split": split_name,
                    "ic_class_est": cls,
                    "seq_len": n,
                    "handoff_tokens": n_hand,
                    "handoff_frac": n_hand / max(1, n),
                    "time_to_handoff": t_hand,
                    "step0_handoff_label": step0_hand,
                    "motion_before_handoff": max(0, t_hand) if t_hand >= 0 else n,
                    "h0_but_starts_with_correction": unnecessary,
                    "prev0_norm": float(np.linalg.norm(epn["prev_actions"][0])),
                }
            )
    _write_csv(OUT / "D2_HANDOFF_SUPERVISION_AUDIT.csv", sup_rows)

    # Compare with A/B/C sample from P21
    p21 = json.loads((P21 / "SFT_FULL_HORIZON_MANIFEST.json").read_text(encoding="utf-8"))
    cmp = []
    for fam in ("A", "B", "C", "D2"):
        if fam == "D2":
            xs = [r for r in sup_rows if r["split"] == "train"]
        else:
            xs = []
            for r in p21["trajectories"]:
                if r["family"] != fam or r["split"] != "train":
                    continue
                try:
                    epn = load_npz_flex(P21, r["npz_relpath"])
                except Exception:
                    continue
                toks = np.asarray(epn["tokens"])
                t_hand = int(np.where(toks == HANDOFF)[0][0]) if np.any(toks == HANDOFF) else -1
                xs.append({"seq_len": len(toks), "step0_handoff_label": int(len(toks) > 0 and int(toks[0]) == HANDOFF), "time_to_handoff": t_hand})
                if len(xs) >= 40:
                    break
        if not xs:
            continue
        cmp.append(
            {
                "family": fam,
                "n": len(xs),
                "mean_seq_len": float(np.mean([x["seq_len"] for x in xs])),
                "frac_step0_handoff": float(np.mean([x["step0_handoff_label"] for x in xs])),
                "mean_time_to_handoff": float(np.mean([x["time_to_handoff"] for x in xs if x["time_to_handoff"] >= 0])),
            }
        )

    n_h0_val = by_cls.get("H0", 0)
    n_h0_train = sum(1 for v in train_class.values() if v == "H0")
    n_h0_corr_demo = sum(1 for r in sup_rows if r["h0_but_starts_with_correction"])
    rec = f"""# D2_DATASET_RECOMMENDATIONS

Generated: {_utc()}

## Facts (no relabeling performed)

- Val H0 (immediate HANDOFF STRICT): **{n_h0_val}/17**
- Train ICs classified H0 via same physical probe: **{n_h0_train}**
- Demonstrations that are H0 but still begin with correction (not step-0 HANDOFF label): **{n_h0_corr_demo}**
- Family comparison (train sample): {cmp}

## Does the dataset penalize valid early HANDOFF?

If many H0 ICs are supervised with long correction→late HANDOFF, CE training **penalizes** a physically
valid immediate HANDOFF relative to the recorded expert tape. P32 does not relabel; it only proposes:

## Proposed future manifest classes (for P33+)

1. **H0_verified** — physically verified immediate-HANDOFF success; prefer step-0 HANDOFF labels
   (or multi-valid: both immediate HANDOFF tape and optional short correction).
2. **H1_verified** — immediate HANDOFF fails; expert (or verified) correction succeeds.
3. **multi_valid** — both strategies succeed; either label is physically OK.
4. **invalid_or_unresolved** — H2/HU; exclude from primary SFT or quarantine.

Do **not** introduce a skill selector / HANDOFF classifier head.
"""
    _write_md(OUT / "D2_DATASET_RECOMMENDATIONS.md", rec)

    # ----- Phase 6: FINAL REPORT -----
    print("=== Phase 6: FINAL_REPORT ===", flush=True)

    def pol_sum(tag):
        xs = [r for r in roll_rows if r["condition"] == tag]
        return {
            "n": len(xs),
            "handoff_strict": sum(int(r["SUCCESS_NEW_policy_handoff"]) for r in xs),
            "imm": sum(int(r["immediate_handoff"]) for r in xs),
            "imm_ok": sum(int(r["immediate_handoff"] and r["SUCCESS_NEW_policy_handoff"]) for r in xs),
            "mean_n_motion": float(np.mean([r["n_motion"] for r in xs])) if xs else 0.0,
            "h0_ok": sum(1 for r in xs if r["ic_class"] == "H0" and r["SUCCESS_NEW_policy_handoff"]),
            "h1_ok": sum(1 for r in xs if r["ic_class"] == "H1" and r["SUCCESS_NEW_policy_handoff"]),
            "h0_n": sum(1 for r in xs if r["ic_class"] == "H0"),
            "h1_n": sum(1 for r in xs if r["ic_class"] == "H1"),
        }

    s22, s28, s29, s31 = pol_sum("P22_A_natural"), pol_sum("P28_C_K32"), pol_sum("P29_L2_last"), pol_sum("P31_M2_pcgrad")
    n_h0, n_h1, n_h2 = by_cls.get("H0", 0), by_cls.get("H1", 0), by_cls.get("H2", 0)
    expert_all = sum(1 for r in expert_rows if r["SUCCESS_NEW"])
    if n_h0 >= 9:
        case = "A"
        case_txt = "Most D2 ICs are H0 — prioritize verified immediate-HANDOFF supervision / avoid over-control"
    elif n_h1 >= 9:
        case = "B"
        case_txt = "Most D2 ICs are H1 — prioritize correction learning"
    elif n_h0 > 0 and n_h1 > 0:
        case = "C"
        case_txt = "Mixed H0/H1 — learn both when to act and when to HANDOFF from verified examples"
    else:
        case = "D"
        case_txt = "Evaluator/IC validity concerns dominate"

    p28_imm_frac = s28["imm"] / max(1, s28["n"])
    bottleneck = []
    if n_h0 and _avoid_count("P22_A_natural") + _avoid_count("P31_M2_pcgrad") > 0:
        bottleneck.append("over-control / delayed HANDOFF on H0")
    if n_h1 and s31["h1_ok"] + s22["h1_ok"] == 0:
        bottleneck.append("failed Precision correction on H1")
    if not bottleneck:
        bottleneck.append("mixed / unclear")

    final = f"""# FINAL_REPORT — P32 D2 Zero-Action HANDOFF Diagnosis

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Setup

- Diagnostic only: **no training**, no RL, no architecture/M111/relabeling changes.
- 17 D2 val ICs from P23 combined manifest; physical STRICT via `evaluate_handoff_with_axis`.
- Step-0 HANDOFF is **legal** and uses the same 12 s physical protocol as later HANDOFF.

## Verified physical facts

| Fact | Value |
|---|---|
| Immediate HANDOFF STRICT (H0) | **{n_h0}/17** |
| Require correction vs immediate (H1) | **{n_h1}/17** |
| Both fail (H2) | **{n_h2}/17** |
| Expert STRICT on 17 | **{expert_all}/17** |
| Restore determinism | {n_det}/{len(restore_rows)} |

### Policy D2 explicit-HANDOFF STRICT

| Policy | STRICT | immediate HANDOFF events | mean n_motion | H0 solved | H1 solved |
|---|---:|---:|---:|---:|---:|
| P22_A_natural | {s22['handoff_strict']}/17 | {s22['imm']} | {s22['mean_n_motion']:.1f} | {s22['h0_ok']}/{s22['h0_n']} | {s22['h1_ok']}/{s22['h1_n']} |
| P28_C_K32 | {s28['handoff_strict']}/17 | {s28['imm']} | {s28['mean_n_motion']:.1f} | {s28['h0_ok']}/{s28['h0_n']} | {s28['h1_ok']}/{s28['h1_n']} |
| P29_L2_last | {s29['handoff_strict']}/17 | {s29['imm']} | {s29['mean_n_motion']:.1f} | {s29['h0_ok']}/{s29['h0_n']} | {s29['h1_ok']}/{s29['h1_n']} |
| P31_M2_pcgrad | {s31['handoff_strict']}/17 | {s31['imm']} | {s31['mean_n_motion']:.1f} | {s31['h0_ok']}/{s31['h0_n']} | {s31['h1_ok']}/{s31['h1_n']} |

Avoidable over-control (H0 BranchA ok, BranchB fail): P22={_avoid_count('P22_A_natural')}, P31={_avoid_count('P31_M2_pcgrad')}, P28={_avoid_count('P28_C_K32')}, P29={_avoid_count('P29_L2_last')}.

## Answers

1. **How many of 17 pass immediate HANDOFF STRICT?** {n_h0}/17 (H0).
2. **How many genuinely require correction?** {n_h1}/17 (H1).
3. **Does original D2 expert succeed on all 17?** {expert_all}/17 — {"YES" if expert_all==17 else "NO"}.
4. **Does P22 over-control initially stable grasps?** {"YES" if _avoid_count('P22_A_natural') else "NO clear avoidable H0 failures"} ({_avoid_count('P22_A_natural')} avoidable).
5. **Does P31 M2_pcgrad over-control them?** {"YES" if _avoid_count('P31_M2_pcgrad') else "NO clear avoidable H0 failures"} ({_avoid_count('P31_M2_pcgrad')} avoidable).
6. **Which 9 ICs triggered P29 L2_last immediate HANDOFF, and why fail?**  
   Re-eval immediate events={len(p29_imm)}; STRICT among them={len(p29_ok)}. Details in `P29_IMMEDIATE_HANDOFF_AUDIT.md`. Failures (if any) are physical after legal HANDOFF, not silent metric bugs.
7. **Does P28 C_K32 succeed mainly through correction or immediate HANDOFF?**  
   immediate events={s28['imm']}/17, mean n_motion={s28['mean_n_motion']:.1f} → {"mostly immediate HANDOFF" if p28_imm_frac>=0.5 else "mostly nonzero correction / delayed HANDOFF"}.
8. **Bottleneck: motion, HANDOFF recognition, or both?** {"; ".join(bottleneck)}.
9. **Does D2 dataset penalize valid early HANDOFF?** H0 demos that still start with correction: **{n_h0_corr_demo}** (see supervision audit) — {"YES, CE can punish valid step-0 HANDOFF" if n_h0_corr_demo else "limited evidence in this corpus"}.
10. **Minimum necessary P33 intervention?** Case **{case}**: {case_txt}.

## Decision case

**Case {case}.** {case_txt}

### Smallest justified P33 training intervention (hypothesis → proposal)

- Prefer adding **physically verified H0 step-0 HANDOFF** examples (and/or multi-valid labels) rather than more complex Precision motion SFT alone, if Case A/C.
- Keep skill-balanced multi-skill training; do **not** add a HANDOFF classifier / skill selector.
- For H1, retain verified correction tapes only where immediate HANDOFF fails.
- Quarantine H2 until recoverability is established.

## Hypotheses (not facts)

- P31’s D2=0/17 may be dominated by refusing immediate HANDOFF on H0 and/or failing H1 correction under mild FT — test via H0/H1 split metrics in P33.
- GradNorm down-weighting D2 (P31) may have blocked learning of both HANDOFF timing and correction.

## Integrity

- Checkpoints / ICs / experts / STRICT criteria unchanged.
- Artifacts: `results/diagnostics/p32_d2_zero_handoff/`
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p32_summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "H0": n_h0,
                "H1": n_h1,
                "H2": n_h2,
                "expert_ok": expert_all,
                "case": case,
                "p22": s22,
                "p28": s28,
                "p29": s29,
                "p31": s31,
                "p29_immediate": len(p29_imm),
                "avoidable": {
                    "P22": _avoid_count("P22_A_natural"),
                    "P31": _avoid_count("P31_M2_pcgrad"),
                    "P28": _avoid_count("P28_C_K32"),
                    "P29": _avoid_count("P29_L2_last"),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== P32 done ===", f"H0={n_h0} H1={n_h1} H2={n_h2} case={case}", flush=True)


if __name__ == "__main__":
    main()
