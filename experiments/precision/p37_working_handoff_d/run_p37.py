"""P37 — Frozen P36 policy: AXIS vs WORKING-aligned HANDOFF paired ablation.

Does not retrain. Does not modify P36 artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p28_multiskill_distillation.harness import policy_step_K  # noqa: E402
from experiments.precision.p37_working_handoff_d.handoff_rules import (  # noqa: E402
    AXIS_DX_MM,
    AXIS_DZ_MM,
    AXIS_THETA_DEG,
    WORK_POS_MM,
    WORK_THETA_DEG,
    handoff_ready,
    init_labels,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW  # noqa: E402

OUT = ROOT / "results/diagnostics/p37_working_handoff_d"
P35_MAN = ROOT / "results/diagnostics/p35_precision_sft/PACKAGED_MANIFEST.csv"
P36_CKPT = ROOT / (
    "results/diagnostics/p36_rule_handoff_d_sft/checkpoints/"
    "D_motion_rule_handoff/checkpoint_epoch_0010.pt"
)
M111_META = ROOT / (
    "results/diagnostics/skill_aware_action_vocab_design/artifacts/"
    "SELECTED_skill_aware_hybrid_meta.json"
)
EXPECTED_SHA256 = "f170b635a89b618f0e60cfb30dd778ddf240e1e8f4073bcbd5e069bc7c57230d"
SEED = 37
MAX_STEPS = 2500


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    def cell(v):
        if isinstance(v, (list, dict, tuple)):
            return json.dumps(v)
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def abs_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def load_p1() -> list[dict]:
    return [r for r in csv.DictReader(P35_MAN.open(encoding="utf-8")) if r.get("class") == "P1"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_failure(
    *,
    strict_ok: bool,
    drop: bool,
    intended_handoff: bool,
    reached_working: bool,
    s12: bool,
    s_new: bool,
    timeout: bool,
    reason: str | None,
) -> str:
    """P37 taxonomy (spec §4). SUCCESS_12S=1 & SUCCESS_NEW=0 is not a physical drop."""
    if strict_ok:
        return "SUCCESS"
    # 1 — never entered WORKING Euclidean ball
    if not reached_working:
        if intended_handoff and s12 and not s_new:
            # AXIS handoff while e_pos still > WORKING (metric mismatch; grasp held)
            return "1_NEVER_WORKING_HELD_AFTER_HANDOFF"
        if drop:
            return "1_NEVER_WORKING_THEN_DROP"
        if timeout:
            return "4_TIMEOUT_NO_DROP" if not drop else "1_NEVER_WORKING_THEN_DROP"
        return "1_NEVER_REACHES_WORKING"
    # 2 — reached WORKING then lost grasp
    if drop or (intended_handoff and not s12 and reason and "DROP" in str(reason).upper()):
        return "2_REACHED_WORKING_THEN_DROP"
    if intended_handoff and not s12:
        return f"2_POST_HANDOFF_FAIL:{reason}"
    # 3 — reached WORKING, remained grasped, failed sustained WORKING hold
    if intended_handoff and s12 and not s_new:
        return "3_WORKING_GRASPED_BUT_HOLD_FAIL"
    # 4 — timeout without drop (reached WORKING mid-way but never triggered? rare)
    if timeout and not drop:
        return "4_TIMEOUT_NO_DROP"
    return f"5_OTHER:{reason or 'unknown'}"


def rollout(
    env,
    model,
    book,
    rho,
    row: dict,
    *,
    device,
    rule: str,
    max_steps: int = MAX_STEPS,
    K: int = WINDOW,
) -> dict[str, Any]:
    """Closed-loop with rule-based HANDOFF only (learned HANDOFF masked)."""
    blob = pickle.loads(abs_path(row["ic_path"]).read_bytes())
    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    ref = blob["ref"]
    hist = env._hist

    labels = init_labels(env.sim)
    obs_seq: list = []
    prev_seq: list = []
    tokens: list[int] = []
    n_motion = 0
    termination = "timeout"
    handoff_step = None
    drop = False
    reached_working_step = None
    ever_working = False
    ever_axis = False
    handoff_geom = None
    pre_hand_p_des = None
    pre_hand_r_err = None

    for t in range(max_steps):
        if latched(env.sim):
            drop = True
            termination = "drop"
            break

        ready, geo = handoff_ready(env.sim, rule)
        if geo["working"]["working_ok"]:
            ever_working = True
            if reached_working_step is None:
                reached_working_step = t
        if geo["axis"]["axis_ok"]:
            ever_axis = True

        if ready:
            handoff_step = t
            handoff_geom = geo
            termination = "rule_handoff"
            pre_hand_p_des = np.asarray(env.sim.fsm.p_des, float).copy()
            break

        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action, _ = policy_step_K(model, book, rho, obs_seq, prev_seq, device, K)
        tokens.append(tok)

        if action is None:
            # Mask learned HANDOFF — motion-only continuation
            with torch.no_grad():
                logits, resid, _ = model(
                    torch.as_tensor(np.stack(obs_seq[-K:]), device=device),
                    torch.as_tensor(np.stack(prev_seq[-K:]), device=device),
                )
                logits = logits[-1].clone()
                logits[HANDOFF] = -1e9
                tok = int(torch.argmax(logits).item())
                rr = resid[-1].float().cpu().numpy().astype(np.float32)
                action = decode_m111_action(book, tok, rr, rho=rho)
                tokens[-1] = tok

        step_legal(env.sim, action)
        env._prev = action
        n_motion += 1

    # Geometry at stop
    _, geo_f = handoff_ready(env.sim, rule)
    if geo_f["working"]["working_ok"]:
        ever_working = True
        if reached_working_step is None:
            reached_working_step = n_motion

    intended = termination == "rule_handoff"
    # Official STRICT evaluator (unchanged)
    p_des_before_eval = np.asarray(env.sim.fsm.p_des, float).copy()
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    # Post-eval: did apply_handoff leave physics unedited? (contract audit flag)
    win = hand.get("window_SUCCESS_NEW") or {}
    s12 = bool(hand.get("SUCCESS_12S"))
    s_new = bool(hand.get("SUCCESS_NEW"))
    strict_ok = bool(s_new and intended)

    fail_cat = classify_failure(
        strict_ok=strict_ok,
        drop=drop,
        intended_handoff=intended,
        reached_working=ever_working,
        s12=s12,
        s_new=s_new,
        timeout=termination == "timeout",
        reason=hand.get("reason"),
    )

    hg = handoff_geom or geo_f
    ax_h = hg["axis"]
    wk_h = hg["working"]

    # Nominal drift proxy: change in e_pos / theta from handoff instant to window mean
    # (only meaningful when handoff happened)
    drift_pos = None
    drift_theta = None
    if intended and win.get("mean_pos") is not None:
        drift_pos = float(win["mean_pos"]) - float(wk_h["e_pos_mm"])
        drift_theta = float(win["mean_theta"]) - float(wk_h["dtheta_deg"])

    return {
        "traj_id": row["traj_id"],
        "lineage_id": row.get("lineage_id"),
        "split": row.get("split"),
        "rule": rule,
        **labels,
        "n_motion": n_motion,
        "termination": termination,
        "handoff_triggered": int(intended),
        "handoff_step": handoff_step,
        "ever_axis": int(ever_axis),
        "ever_working": int(ever_working),
        "reached_working_step": reached_working_step,
        "SUCCESS_12S": int(s12) if intended else 0,
        "SUCCESS_12S_raw": int(s12),
        "SUCCESS_NEW": int(strict_ok),
        "SUCCESS_NEW_raw": int(s_new),
        "eval_reason": hand.get("reason"),
        "timeout": int(termination == "timeout"),
        "drop": int(drop),
        "fail_category": fail_cat,
        # At HANDOFF (or final if timeout)
        "hand_dtheta_deg": wk_h["dtheta_deg"],
        "hand_e_pos_mm": wk_h["e_pos_mm"],
        "hand_dx_mm": ax_h["dx_mm"],
        "hand_dy_mm": ax_h["dy_mm"],
        "hand_dz_mm": ax_h["dz_mm"],
        "hand_axis_ok": int(ax_h["axis_ok"]),
        "hand_working_ok": int(wk_h["working_ok"]),
        "hand_bilateral": int(wk_h["bilateral"]),
        "hand_e_pos_match": int(wk_h.get("e_pos_match", False)),
        # Final window (STRICT)
        "win_mean_theta": win.get("mean_theta"),
        "win_max_theta": win.get("max_theta"),
        "win_mean_pos": win.get("mean_pos"),
        "win_max_pos": win.get("max_pos"),
        "win_frac_bilateral": win.get("frac_bilateral"),
        "win_working_hold": int(bool(win.get("working_SUCCESS_NEW_hold"))) if intended else 0,
        "nominal_drift_pos_mm": drift_pos,
        "nominal_drift_theta_deg": drift_theta,
        "p_des_before_eval": p_des_before_eval.tolist(),
        "tokens_head": tokens[:12],
    }


def eval_rule(label, model, book, rho, rows, device, env, rule: str) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        row = rollout(env, model, book, rho, r, device=device, rule=rule)
        row["condition"] = label
        out.append(row)
        print(
            f"  [{label}] {i+1}/{len(rows)} {r['traj_id'][:42]} "
            f"ax0={row['init_axis_satisfied']} wk0={row['init_working_satisfied']} "
            f"hand={row['handoff_triggered']} NEW={row['SUCCESS_NEW']} "
            f"12S={row['SUCCESS_12S']} cat={row['fail_category']}",
            flush=True,
        )
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "handoff": sum(int(r["handoff_triggered"]) for r in rows),
        "ever_working": sum(int(r["ever_working"]) for r in rows),
        "SUCCESS_12S": sum(int(r["SUCCESS_12S"]) for r in rows),
        "SUCCESS_NEW": sum(int(r["SUCCESS_NEW"]) for r in rows),
        "timeout": sum(int(r["timeout"]) for r in rows),
        "drop": sum(int(r["drop"]) for r in rows),
        "fail_cats": dict(Counter(r["fail_category"] for r in rows)),
        "init_axis": sum(int(r["init_axis_satisfied"]) for r in rows),
        "init_working": sum(int(r["init_working_satisfied"]) for r in rows),
        "init_axis_only": sum(int(r["init_axis_only"]) for r in rows),
    }


def pair_rows(axis_rows: list[dict], work_rows: list[dict]) -> list[dict]:
    by_a = {r["traj_id"]: r for r in axis_rows}
    by_w = {r["traj_id"]: r for r in work_rows}
    ids = sorted(set(by_a) | set(by_w))
    paired = []
    for tid in ids:
        a, w = by_a[tid], by_w[tid]
        a_ok = int(a["SUCCESS_NEW"])
        w_ok = int(w["SUCCESS_NEW"])
        if a_ok and w_ok:
            outcome = "BOTH_SUCCESS"
        elif a_ok and not w_ok:
            outcome = "AXIS_ONLY"
        elif (not a_ok) and w_ok:
            outcome = "WORKING_ONLY"
        else:
            outcome = "NEITHER"
        paired.append(
            {
                "traj_id": tid,
                "split": a["split"],
                "init_axis_satisfied": a["init_axis_satisfied"],
                "init_working_satisfied": a["init_working_satisfied"],
                "init_axis_only": a["init_axis_only"],
                "init_nontrivial_axis": a["init_nontrivial_axis"],
                "init_e_pos_mm": a["init_e_pos_mm"],
                "init_dtheta_deg": a["init_dtheta_deg"],
                "init_dx_mm": a["init_dx_mm"],
                "init_dz_mm": a["init_dz_mm"],
                "axis_SUCCESS_NEW": a_ok,
                "work_SUCCESS_NEW": w_ok,
                "axis_SUCCESS_12S": a["SUCCESS_12S"],
                "work_SUCCESS_12S": w["SUCCESS_12S"],
                "axis_handoff": a["handoff_triggered"],
                "work_handoff": w["handoff_triggered"],
                "axis_ever_working": a["ever_working"],
                "work_ever_working": w["ever_working"],
                "axis_timeout": a["timeout"],
                "work_timeout": w["timeout"],
                "axis_drop": a["drop"],
                "work_drop": w["drop"],
                "axis_fail": a["fail_category"],
                "work_fail": w["fail_category"],
                "axis_hand_e_pos": a["hand_e_pos_mm"],
                "work_hand_e_pos": w["hand_e_pos_mm"],
                "axis_win_mean_pos": a["win_mean_pos"],
                "work_win_mean_pos": w["win_mean_pos"],
                "axis_drift_pos": a["nominal_drift_pos_mm"],
                "work_drift_pos": w["nominal_drift_pos_mm"],
                "paired_outcome": outcome,
            }
        )
    return paired


def integrity_checks(device) -> dict:
    assert P36_CKPT.is_file(), P36_CKPT
    digest = sha256_file(P36_CKPT)
    ok_hash = digest == EXPECTED_SHA256
    book, rho, _ = load_m111_book(M111_META)
    model, meta, _ = load_m111(P36_CKPT, device)
    # Spot-check: no teleport in apply_handoff
    from envs.outcomes.recovery_success import apply_handoff

    env = make_env()
    rows = load_p1()
    blob = pickle.loads(abs_path(rows[0]["ic_path"]).read_bytes())
    restore_capture(env, blob["capture"])
    q0 = np.array(env.sim.data.qpos, float).copy()
    audit = apply_handoff(env.sim, blob["ref"])
    q1 = np.array(env.sim.data.qpos, float)
    checks = {
        "checkpoint": str(P36_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "sha256": digest,
        "sha256_matches_p36_record": ok_hash,
        "expected_sha256": EXPECTED_SHA256,
        "ckpt_meta_keys": list((meta or {}).keys())[:20] if isinstance(meta, dict) else str(type(meta)),
        "apply_handoff_physics_edited": audit.get("physics_edited"),
        "apply_handoff_qpos_max_abs": float(np.max(np.abs(q1 - q0))),
        "apply_handoff_contract": audit.get("contract"),
        "HANDOFF_only_experimental_diff": True,
        "WORKING_THETA_DEG": WORK_THETA_DEG,
        "WORKING_POS_MM": WORK_POS_MM,
        "AXIS_THETA_DEG": AXIS_THETA_DEG,
        "AXIS_DX_MM": AXIS_DX_MM,
        "AXIS_DZ_MM": AXIS_DZ_MM,
        "evaluator": "evaluation.axis_aware_success.evaluate_handoff_with_axis",
        "utc": utc(),
    }
    atomic_json(OUT / "INTEGRITY.json", checks)
    if not ok_hash:
        print(f"WARNING: checkpoint hash mismatch: {digest}", flush=True)
    else:
        print(f"Integrity OK sha256={digest[:16]}…", flush=True)
    del model
    return {"book": book, "rho": rho, "checks": checks, "env": env}


def write_final_clean(paired, axis_rows, work_rows, checks) -> None:
    def cohort(rows, pred):
        return summarize([r for r in rows if pred(r)])

    def nt(r):
        return int(r["init_nontrivial_axis"]) == 1

    def sat_axis(r):
        return int(r["init_axis_satisfied"]) == 1

    def sat_work(r):
        return int(r["init_working_satisfied"]) == 1

    def axis_only_init(r):
        return int(r["init_axis_only"]) == 1

    splits = {
        "train_nt": lambda r: r["split"] == "train" and nt(r),
        "val_nt": lambda r: r["split"] == "val" and nt(r),
        "test_nt": lambda r: r["split"] == "test" and nt(r),
        "test_axis_sat": lambda r: r["split"] == "test" and sat_axis(r),
        "test_work_sat": lambda r: r["split"] == "test" and sat_work(r),
        "test_axis_only": lambda r: r["split"] == "test" and axis_only_init(r),
    }
    agg = []
    for name, pred in splits.items():
        for rule, rows in (("axis", axis_rows), ("working", work_rows)):
            agg.append({"cohort": name, "rule": rule, **cohort(rows, pred)})
    write_csv(OUT / "AGGREGATE.csv", agg)

    pair_sum = []
    for name, pred in splits.items():
        sub = [p for p in paired if pred(p)]
        c = Counter(p["paired_outcome"] for p in sub)
        pair_sum.append(
            {
                "cohort": name,
                "n": len(sub),
                "BOTH_SUCCESS": c.get("BOTH_SUCCESS", 0),
                "AXIS_ONLY": c.get("AXIS_ONLY", 0),
                "WORKING_ONLY": c.get("WORKING_ONLY", 0),
                "NEITHER": c.get("NEITHER", 0),
            }
        )
    write_csv(OUT / "PAIRED_SUMMARY.csv", pair_sum)
    write_csv(
        OUT / "CHANGED_OUTCOMES.csv",
        [p for p in paired if p["paired_outcome"] in ("AXIS_ONLY", "WORKING_ONLY")],
    )

    # Failure breakdown
    fb = []
    for rule, rows in (("axis", axis_rows), ("working", work_rows)):
        for name, pred in splits.items():
            sub = [r for r in rows if pred(r)]
            for cat, cnt in Counter(r["fail_category"] for r in sub).items():
                fb.append({"rule": rule, "cohort": name, "fail_category": cat, "count": cnt, "n": len(sub)})
    write_csv(OUT / "FAILURE_BREAKDOWN.csv", fb)

    te_a, te_w = cohort(axis_rows, splits["test_nt"]), cohort(work_rows, splits["test_nt"])
    tr_a, tr_w = cohort(axis_rows, splits["train_nt"]), cohort(work_rows, splits["train_nt"])
    va_a, va_w = cohort(axis_rows, splits["val_nt"]), cohort(work_rows, splits["val_nt"])
    ts_a, ts_w = cohort(axis_rows, splits["test_axis_sat"]), cohort(work_rows, splits["test_axis_sat"])
    tao, two = cohort(axis_rows, splits["test_axis_only"]), cohort(work_rows, splits["test_axis_only"])
    tws_a, tws_w = cohort(axis_rows, splits["test_work_sat"]), cohort(work_rows, splits["test_work_sat"])
    ps_te = next(x for x in pair_sum if x["cohort"] == "test_nt")
    ps_ts = next(x for x in pair_sum if x["cohort"] == "test_axis_sat")

    work_nt_test = [r for r in work_rows if splits["test_nt"](r)]
    gap = sum(
        1
        for r in work_nt_test
        if int(r["ever_working"]) and int(r["handoff_triggered"]) and not int(r["SUCCESS_NEW"])
    )
    reach = sum(int(r["ever_working"]) for r in work_nt_test)
    te_pair = [p for p in paired if p["split"] == "test" and int(p["init_nontrivial_axis"]) == 1]
    extra_to = sum(1 for p in te_pair if int(p["work_timeout"]) > int(p["axis_timeout"]))
    extra_drop = sum(1 for p in te_pair if int(p["work_drop"]) > int(p["axis_drop"]))

    if te_w["SUCCESS_NEW"] > te_a["SUCCESS_NEW"]:
        verdict = (
            "WORKING-aligned HANDOFF **improves** held-out nontrivial STRICT vs AXIS "
            "on the same frozen P36 policy."
        )
    elif te_w["SUCCESS_NEW"] == te_a["SUCCESS_NEW"]:
        verdict = (
            "WORKING-aligned HANDOFF **matches** AXIS on held-out nontrivial STRICT; "
            "main effect is metric consistency on AXIS-only init-sat controls."
        )
    else:
        verdict = (
            "WORKING-aligned HANDOFF **does not improve** held-out nontrivial STRICT "
            "(tighter stop can add timeouts before Euclidean entry)."
        )

    if gap > 0:
        bottleneck = (
            f"Post-HANDOFF sustained-hold gap: {gap}/{max(reach,1)} test-NT reached WORKING "
            "but failed SUCCESS_NEW."
        )
    elif te_w["timeout"] > 0:
        bottleneck = "Motion-policy: timeouts without reaching WORKING under the tighter rule."
    else:
        bottleneck = "See failure categories."

    text = f"""# FINAL_REPORT — P37 WORKING-Aligned HANDOFF Ablation

Generated: {utc()}

## Integrity

- Checkpoint: `{checks['checkpoint']}`
- SHA256: `{checks['sha256']}`
- Matches P36 record: **{checks['sha256_matches_p36_record']}**
- apply_handoff: physics_edited={checks['apply_handoff_physics_edited']}, qpos_max_abs={checks['apply_handoff_qpos_max_abs']}, contract={checks['apply_handoff_contract']}
- Only experimental difference: HANDOFF trigger (AXIS vs WORKING)
- Evaluator: unchanged `evaluate_handoff_with_axis`
- WORKING trigger: `measure(U_TGT,P_TGT)` + WORKING_THETA/POS (single step, **no** 0.5s dwell)

## HANDOFF rules

| Rule | Condition |
|---|---|
| A AXIS_V1 | \|θ\|≤{AXIS_THETA_DEG}° ∧ \|dx\|≤{AXIS_DX_MM} mm ∧ \|dz\|≤{AXIS_DZ_MM} mm |
| B WORKING | θ≤{WORK_THETA_DEG}° ∧ e_pos≤{WORK_POS_MM} mm (Euclidean; STRICT instant region) |

## Nontrivial (not initially AXIS-satisfied)

| Split | Rule | n | Ever WORKING | HANDOFF | SUCCESS_12S | SUCCESS_NEW | Timeout | Drop |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Train | AXIS | {tr_a['n']} | {tr_a['ever_working']} | {tr_a['handoff']} | {tr_a['SUCCESS_12S']} | **{tr_a['SUCCESS_NEW']}** | {tr_a['timeout']} | {tr_a['drop']} |
| Train | WORKING | {tr_w['n']} | {tr_w['ever_working']} | {tr_w['handoff']} | {tr_w['SUCCESS_12S']} | **{tr_w['SUCCESS_NEW']}** | {tr_w['timeout']} | {tr_w['drop']} |
| Val | AXIS | {va_a['n']} | {va_a['ever_working']} | {va_a['handoff']} | {va_a['SUCCESS_12S']} | **{va_a['SUCCESS_NEW']}** | {va_a['timeout']} | {va_a['drop']} |
| Val | WORKING | {va_w['n']} | {va_w['ever_working']} | {va_w['handoff']} | {va_w['SUCCESS_12S']} | **{va_w['SUCCESS_NEW']}** | {va_w['timeout']} | {va_w['drop']} |
| Test | AXIS | {te_a['n']} | {te_a['ever_working']} | {te_a['handoff']} | {te_a['SUCCESS_12S']} | **{te_a['SUCCESS_NEW']}** | {te_a['timeout']} | {te_a['drop']} |
| Test | WORKING | {te_w['n']} | {te_w['ever_working']} | {te_w['handoff']} | {te_w['SUCCESS_12S']} | **{te_w['SUCCESS_NEW']}** | {te_w['timeout']} | {te_w['drop']} |

- P36 test NT (AXIS) was **8/11**. P37 AXIS re-eval **{te_a['SUCCESS_NEW']}/{te_a['n']}**; WORKING **{te_w['SUCCESS_NEW']}/{te_w['n']}**.
- Retains ≥8/11 under WORKING? **{te_w['SUCCESS_NEW'] >= 8 and te_w['n'] == 11}**
- Test NT fail cats AXIS: `{te_a['fail_cats']}`
- Test NT fail cats WORKING: `{te_w['fail_cats']}`

### Paired test nontrivial (n={ps_te['n']})

BOTH={ps_te['BOTH_SUCCESS']} · AXIS_ONLY={ps_te['AXIS_ONLY']} · WORKING_ONLY={ps_te['WORKING_ONLY']} · NEITHER={ps_te['NEITHER']}

Extra timeouts under WORKING: **{extra_to}**; extra drops: **{extra_drop}**.
Reach-WORKING→STRICT gap (WORKING rule, test NT): **{gap}/{reach}**.

## Test AXIS-init-sat controls

| Subset | n | AXIS NEW | WORKING NEW |
|---|---:|---:|---:|
| All AXIS-init-sat | {ts_a['n']} | {ts_a['SUCCESS_NEW']}/{ts_a['n']} | {ts_w['SUCCESS_NEW']}/{ts_w['n']} |
| Also WORKING-init-sat | {tws_a['n']} | {tws_a['SUCCESS_NEW']}/{tws_a['n']} | {tws_w['SUCCESS_NEW']}/{tws_w['n']} |
| AXIS-only (≠ WORKING) | {tao['n']} | {tao['SUCCESS_NEW']}/{tao['n']} | {two['SUCCESS_NEW']}/{two['n']} |

AXIS-only inits are **not** already recovered under WORKING. Paired axis-sat: BOTH={ps_ts['BOTH_SUCCESS']} AXIS_ONLY={ps_ts['AXIS_ONLY']} WORKING_ONLY={ps_ts['WORKING_ONLY']} NEITHER={ps_ts['NEITHER']}.
Controls improve? **{ts_w['SUCCESS_NEW'] > ts_a['SUCCESS_NEW']}**.

Fail cats — AXIS rule: `{ts_a['fail_cats']}`  
Fail cats — WORKING rule: `{ts_w['fail_cats']}`

## Conclusion

{verdict}

**Remaining bottleneck:** {bottleneck}

**Skill D solved?** **NO** — consistency of HANDOFF with SUCCESS_NEW is not a solve; test NT STRICT under WORKING = {te_w['SUCCESS_NEW']}/{te_w['n']}.

## Artifacts

`INTEGRITY.json`, `ROLLOUTS_*.csv`, `PAIRED_PER_ROOT.csv`, `PAIRED_SUMMARY.csv`, `AGGREGATE.csv`, `CHANGED_OUTCOMES.csv`, `FAILURE_BREAKDOWN.csv`, `FINAL_REPORT.md`
"""
    write_md(OUT / "FINAL_REPORT.md", text)
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--axis-only", action="store_true")
    ap.add_argument("--working-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    info = integrity_checks(device)
    book, rho, env = info["book"], info["rho"], info["env"]
    model, _, _ = load_m111(P36_CKPT, device)

    p1 = load_p1()
    # Annotate init labels once
    annotated = []
    for r in p1:
        blob = pickle.loads(abs_path(r["ic_path"]).read_bytes())
        restore_capture(env, blob["capture"])
        lab = init_labels(env.sim)
        annotated.append({**r, **lab})
    write_csv(OUT / "IC_LABELS.csv", annotated)
    print(
        f"IC: n={len(annotated)} axis_sat={sum(r['init_axis_satisfied'] for r in annotated)} "
        f"work_sat={sum(r['init_working_satisfied'] for r in annotated)} "
        f"axis_only={sum(r['init_axis_only'] for r in annotated)} "
        f"nt={sum(r['init_nontrivial_axis'] for r in annotated)}",
        flush=True,
    )

    # Required cohorts: all P1 (so train/val/test NT + test axis-sat are covered)
    rows = annotated
    if args.limit > 0:
        rows = rows[: args.limit]

    axis_rows: list[dict] = []
    work_rows: list[dict] = []
    if not args.working_only:
        print(f"=== AXIS rule n={len(rows)} ===", flush=True)
        axis_rows = eval_rule("AXIS", model, book, rho, rows, device, env, rule="axis")
        write_csv(OUT / "ROLLOUTS_AXIS.csv", axis_rows)
    if not args.axis_only:
        print(f"=== WORKING rule n={len(rows)} ===", flush=True)
        work_rows = eval_rule("WORKING", model, book, rho, rows, device, env, rule="working")
        write_csv(OUT / "ROLLOUTS_WORKING.csv", work_rows)

    if args.axis_only or args.working_only:
        print("Partial run; skip paired final.", flush=True)
        return

    paired = pair_rows(axis_rows, work_rows)
    write_csv(OUT / "PAIRED_PER_ROOT.csv", paired)
    write_final_clean(paired, axis_rows, work_rows, info["checks"])
    # Remove broken dual writer if present in namespace — only write_final_clean used
    del model


if __name__ == "__main__":
    main()
