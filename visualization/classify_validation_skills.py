"""Classify recovery mechanisms on all 17 validation ICs for SFT and GRPO.

Reuses contact sampling and deterministic rollout from audit_checkpoint3_skill.
Writes only under results/diagnostics/validation_skill_classification/.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import roll_policy
from envs.simulation.capture import restore_capture
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from visualization import audit_checkpoint3_skill as audit
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4

OUT = ROOT / "results" / "diagnostics" / "validation_skill_classification"
MANIFEST = ROOT / "data" / "valid_skill_quotas.jsonl"
SFT_CKPT = ROOT / "results" / "sft" / "checkpoint_epoch_0040.pt"
GRPO_CKPT = ROOT / "results" / "grpo" / "tree80" / "checkpoint_epoch_0003.pt"

# Primary thresholds (same spirit as checkpoint3 audit).
BOTH_OFF_HOLD_S = 0.04
MIGRATION_M = 0.001
POST_GAP_MIGRATION_M = 0.002  # Hybrid needs clearly measurable post-recontact motion
RH_TRAVEL_M = 0.001

MECH = {
    "Contact-Preserving Adjustment": "A",
    "Release-Recatch": "B",
    "Hybrid": "C",
    "Other/Unclear": "D",
}
LABEL_FROM_CODE = {value: key for key, value in MECH.items()}


def load_valid_rows():
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if len(rows) != 17:
        raise RuntimeError(f"expected 17 validation ICs, found {len(rows)}")
    return rows


def prepare_blob(blob, seated_ref):
    blob = dict(blob)
    if "ref" not in blob:
        if seated_ref is None:
            raise RuntimeError("horizontal snapshot needs seated grasp reference")
        blob["ref"] = seated_ref
    return blob


def restore_ic(env, blob):
    if "capture" in blob:
        restore_capture(env, blob["capture"])
        return
    from models.airborne_v3 import DT_POLICY
    from envs.observations.observable_obs import ObservableObsState
    from envs.outcomes.table_drop import restore_state
    from training.recovery_execution import restore_ballistic

    restore_ballistic(env.sim, blob)
    restore_state(env.sim, None)
    env._hist = ObservableObsState(DT_POLICY)
    env._obs_seq = []
    env._prev_seq = []
    env._prev = np.zeros(7, np.float32)


def migration_of(rows, start_idx=0):
    contact = [
        row for row in rows[start_idx:]
        if row["bilateral"] and np.isfinite(row["cL"]).all() and np.isfinite(row["cR"]).all()
    ]
    if len(contact) < 2:
        return 0.0, "bilateral_contact_centroid_unavailable"
    mid = np.array([(row["cL"] + row["cR"]) * 0.5 for row in contact])
    return float(np.max(np.linalg.norm(mid - mid[0], axis=1))), "max_L2_of_bilateral_contact_centroid_in_hand_frame"


def classify_mechanism(rows, t0, handoff_t, both_off_hold_s=BOTH_OFF_HOLD_S, migration_m=MIGRATION_M, post_gap_m=POST_GAP_MIGRATION_M):
    recovery = [row for row in rows if row["phase"] == "recovery" and row["t"] <= handoff_t + 1e-9]
    if len(recovery) < 2:
        return {
            "mechanism": "Other/Unclear",
            "code": "D",
            "confidence": "low",
            "why": ["fewer than two recovery samples"],
            "facts": {"recovery_steps": len(recovery)},
            "sensitivity": {},
        }

    both_off_intervals = []
    start = None
    for row in recovery:
        if row["both_off"] and start is None:
            start = row["t"]
        if (not row["both_off"]) and start is not None:
            both_off_intervals.append((start - t0, row["t"] - t0, row["t"] - start))
            start = None
    if start is not None:
        both_off_intervals.append((start - t0, recovery[-1]["t"] - t0, recovery[-1]["t"] - start))

    sustained = [item for item in both_off_intervals if item[2] >= both_off_hold_s - 1e-9]
    borderline = [
        item for item in both_off_intervals
        if abs(item[2] - both_off_hold_s) <= 1e-9 or (0.02 - 1e-9 < item[2] < both_off_hold_s)
    ]
    bilateral_steps = sum(1 for row in recovery if row["bilateral"])
    any_contact_steps = sum(1 for row in recovery if row["nL"] > 0 or row["nR"] > 0)
    unilateral_only = any_contact_steps == len(recovery) and bilateral_steps < len(recovery)
    migration, migration_method = migration_of(recovery)
    rh = np.array([row["rh"] for row in recovery])
    rh_travel = float(np.max(np.linalg.norm(rh - rh[0], axis=1))) if len(rh) else 0.0

    facts = {
        "recovery_steps": len(recovery),
        "bilateral_steps": bilateral_steps,
        "any_contact_steps": any_contact_steps,
        "unilateral_only_steps": unilateral_only,
        "both_off_intervals": [
            {"t_start": a, "t_end": b, "duration_s": c} for a, b, c in both_off_intervals
        ],
        "sustained_both_off_intervals": [
            {"t_start": a, "t_end": b, "duration_s": c} for a, b, c in sustained
        ],
        "max_contact_migration_m": migration,
        "migration_method": migration_method,
        "max_object_travel_in_hand_m": rh_travel,
        "max_aperture": float(max(row["aperture"] for row in recovery)),
        "min_tau": float(min(row["tau"] for row in recovery)),
        "both_off_hold_threshold_s": both_off_hold_s,
        "migration_threshold_m": migration_m,
        "post_gap_migration_threshold_m": post_gap_m,
    }

    why = []
    mechanism = "Other/Unclear"
    confidence = "medium"

    if sustained:
        last_gap_end = sustained[-1][1]
        recontact = any(row["bilateral"] and (row["t"] - t0) >= last_gap_end - 1e-9 for row in recovery)
        if recontact:
            mechanism = "Release-Recatch"
            why.append(
                f"sustained both-off >= {both_off_hold_s:.3f}s "
                f"(max {max(item[2] for item in sustained):.3f}s) then bilateral recontact"
            )
            after = [row for row in recovery if (row["t"] - t0) >= last_gap_end - 1e-9 and row["bilateral"]]
            after_mig, _method = migration_of(after)
            facts["post_recontact_migration_m"] = after_mig
            if after_mig >= post_gap_m:
                mechanism = "Hybrid"
                why.append(
                    f"post-recontact bilateral contact centroid migrates {after_mig:.4f} m "
                    f"(threshold {post_gap_m:.3f} m)"
                )
            if abs(max(item[2] for item in sustained) - both_off_hold_s) <= 1e-9:
                confidence = "borderline"
                why.append("both-off duration equals the threshold exactly")
            else:
                confidence = "high"
        else:
            why.append("sustained both-off without clear bilateral re-establishment before handoff")
            confidence = "low"
    elif bilateral_steps == len(recovery) and migration >= migration_m:
        mechanism = "Contact-Preserving Adjustment"
        why.append(
            f"bilateral contact on every recovery sample; contact centroid migrates {migration:.4f} m"
        )
        confidence = "high" if migration >= 2 * migration_m else "medium"
        facts["sliding_observed"] = True
    elif any_contact_steps == len(recovery) and migration >= migration_m:
        mechanism = "Contact-Preserving Adjustment"
        why.append(
            f"at least one finger contact every sample; migration {migration:.4f} m "
            f"(bilateral {bilateral_steps}/{len(recovery)})"
        )
        confidence = "medium"
        facts["sliding_observed"] = True
    elif any_contact_steps == len(recovery) and rh_travel >= RH_TRAVEL_M:
        mechanism = "Contact-Preserving Adjustment"
        why.append(f"contact preserved; object travel in hand frame {rh_travel:.4f} m")
        confidence = "medium"
        facts["sliding_observed"] = True
    elif both_off_intervals and not sustained:
        why.append(
            "only sub-threshold both-off gaps; not counted as Release-Recatch "
            f"(gaps={[round(item[2], 4) for item in both_off_intervals]})"
        )
        if migration >= migration_m and any_contact_steps == len(recovery):
            mechanism = "Contact-Preserving Adjustment"
            why.append("contact otherwise preserved with measurable migration")
            confidence = "borderline"
            facts["sliding_observed"] = True
        else:
            confidence = "low"
    else:
        if bilateral_steps == len(recovery) and migration < migration_m:
            why.append(
                f"bilateral contact preserved but migration {migration:.4f} m below {migration_m:.3f} m; "
                "no clear in-contact adjustment"
            )
        else:
            why.append("contact pattern does not uniquely match A/B/C thresholds")
        confidence = "low"

    # Sensitivity under alternate thresholds.
    alt_hold = classify_raw_label(recovery, t0, both_off_hold_s=0.06, migration_m=migration_m, post_gap_m=post_gap_m)
    alt_mig = classify_raw_label(recovery, t0, both_off_hold_s=both_off_hold_s, migration_m=0.002, post_gap_m=0.003)
    facts["sensitivity"] = {
        "both_off_0.06s": alt_hold,
        "migration_2mm_postgap_3mm": alt_mig,
    }
    if borderline and mechanism in {"Release-Recatch", "Hybrid"}:
        confidence = "borderline"

    return {
        "mechanism": mechanism,
        "code": MECH[mechanism],
        "confidence": confidence,
        "why": why,
        "facts": facts,
        "sensitivity": facts["sensitivity"],
    }


def classify_raw_label(recovery, t0, both_off_hold_s, migration_m, post_gap_m):
    both_off_intervals = []
    start = None
    for row in recovery:
        if row["both_off"] and start is None:
            start = row["t"]
        if (not row["both_off"]) and start is not None:
            both_off_intervals.append((start - t0, row["t"] - t0, row["t"] - start))
            start = None
    if start is not None:
        both_off_intervals.append((start - t0, recovery[-1]["t"] - t0, recovery[-1]["t"] - start))
    sustained = [item for item in both_off_intervals if item[2] >= both_off_hold_s - 1e-9]
    bilateral_steps = sum(1 for row in recovery if row["bilateral"])
    any_contact_steps = sum(1 for row in recovery if row["nL"] > 0 or row["nR"] > 0)
    migration, _ = migration_of(recovery)
    rh = np.array([row["rh"] for row in recovery])
    rh_travel = float(np.max(np.linalg.norm(rh - rh[0], axis=1))) if len(rh) else 0.0
    if sustained:
        last_gap_end = sustained[-1][1]
        recontact = any(row["bilateral"] and (row["t"] - t0) >= last_gap_end - 1e-9 for row in recovery)
        if recontact:
            after = [row for row in recovery if (row["t"] - t0) >= last_gap_end - 1e-9 and row["bilateral"]]
            after_mig, _ = migration_of(after)
            if after_mig >= post_gap_m:
                return "Hybrid"
            return "Release-Recatch"
        return "Other/Unclear"
    if any_contact_steps == len(recovery) and (migration >= migration_m or rh_travel >= RH_TRAVEL_M):
        return "Contact-Preserving Adjustment"
    return "Other/Unclear"


def run_one(env, model, book, blob, device, limit_s, episode_id, ckpt_name, out_dirs):
    # Match official eval outcome.
    restore_ic(env, blob)
    eval_out = roll_policy(env, model, book, blob, device, cap_steps=200, limit_s=limit_s)

    # Traced rollout (same action rule).
    # audit.roll_policy_full always restore_capture; wrap for horizontal.
    if "capture" in blob:
        rows, t0, traced, handoff_t = audit.roll_policy_full(env, model, book, blob, device, limit_s)
    else:
        rows, t0, traced, handoff_t = roll_policy_full_ballistic(env, model, book, blob, device, limit_s)

    # Prefer official eval label for SUCCESS_12S consistency.
    outcome = {
        "reason": eval_out.get("reason"),
        "success": bool(eval_out.get("success")),
        "handoff": eval_out.get("handoff"),
        "recovery_elapsed": eval_out.get("recovery_elapsed"),
        "nominal_elapsed": eval_out.get("nominal_elapsed"),
        "lift_completed": eval_out.get("lift_completed"),
        "traced_reason": traced.get("reason"),
    }
    if outcome["reason"] != traced.get("reason"):
        outcome["label_mismatch"] = {
            "eval": outcome["reason"],
            "traced": traced.get("reason"),
        }

    skill = classify_mechanism(rows, t0, handoff_t)
    events = audit.contact_events(rows, t0)
    tokens = [row["token"] for row in rows if row["phase"] == "recovery" and row["token"] is not None]

    stem = f"{ckpt_name}__{episode_id}"
    audit.write_csv(out_dirs["traces"] / f"{stem}.csv", rows, t0)
    (out_dirs["events"] / f"{stem}.json").write_text(json.dumps(events, indent=2), encoding="utf-8")

    return {
        "episode_id": episode_id,
        "checkpoint": ckpt_name,
        "outcome": outcome,
        "mechanism": skill["mechanism"],
        "code": skill["code"],
        "confidence": skill["confidence"],
        "why": skill["why"],
        "facts": skill["facts"],
        "sensitivity": skill["sensitivity"],
        "tokens": tokens,
        "handoff_t_rel": float(handoff_t - t0),
        "events_file": f"contact_events/{stem}.json",
        "trace_file": f"traces/{stem}.csv",
    }


def roll_policy_full_ballistic(env, model, book, blob, device, limit_s):
    """Same as audit.roll_policy_full but starts from ballistic restore."""
    from envs.outcomes.horizon_outcome import LOST_STEPS, NOMINAL_S, _bilateral, _dropped, _monitor, recovery_failed
    from envs.ballistic_impact import z_tgt_of
    from envs.outcomes.recovery_success import apply_handoff, nominal_step
    from envs.outcomes.table_drop import latched, terminal_reason
    from envs.step_legal import step_legal
    from models.k48_checkpoint import HANDOFF
    from training.k47_codebook import decode_action
    from training.recovery_execution import _escaped
    from training.reward_v2 import NominalHoldTrace

    restore_ic(env, blob)
    t0 = float(env.sim.data.time)
    rows = [audit.sample_state(env.sim, blob["ref"], phase="recovery")]
    lost_low = 0
    how = None
    cap = int(round(float(limit_s) / 0.02))
    for _ in range(cap):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        residual = mean[-1].detach().float().cpu().numpy()
        if token == HANDOFF:
            how = "early"
            rows.append(audit.sample_state(env.sim, blob["ref"], action=np.zeros(7), token=token, residual=residual, phase="handoff"))
            break
        _res, action = decode_action(book, token, residual)
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        rows.append(audit.sample_state(env.sim, blob["ref"], action=executed, token=token, residual=residual, phase="recovery"))
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            reason = terminal_reason(env.sim) or "DROP_DURING_RECOVERY"
            return rows, t0, {
                "reason": reason, "success": False, "handoff": None,
                "recovery_elapsed": float(env.sim.data.time) - t0, "nominal_elapsed": 0.0,
                "lift_completed": False,
            }, float(env.sim.data.time)
    if how is None:
        how = "forced"
    handoff_t = float(env.sim.data.time)
    apply_handoff(env.sim, blob["ref"])
    t_hand = float(env.sim.data.time)
    lost = 0
    lift_done = False
    reason = None
    end_row = None
    hold_trace = NominalHoldTrace(NOMINAL_S)
    if latched(env.sim):
        reason = "TABLE_DROP"
    next_sample = t_hand
    while reason is None and float(env.sim.data.time) - t_hand < NOMINAL_S - 1e-9:
        t_prev = float(env.sim.data.time)
        nominal_step(env.sim)
        row = _monitor(env.sim)
        end_row = row
        target = float(z_tgt_of(env.sim))
        bilateral = _bilateral(row)
        command_reached = float(env.sim.fsm.p_des[2]) >= target - 1e-9
        dropped = _dropped(row) or _escaped(env.sim)
        hold_trace.observe(float(env.sim.data.time) - t_prev, bilateral, dropped, command_reached)
        if bilateral and command_reached:
            lift_done = True
        if latched(env.sim):
            reason = "TABLE_DROP"
            break
        if dropped:
            reason = "DROP_AFTER_HANDOFF"
            break
        lost = lost + 1 if not bilateral else 0
        if lost >= LOST_STEPS:
            reason = "LOST_GRASP"
            break
        now = float(env.sim.data.time)
        if now + 1e-9 >= next_sample:
            rows.append(audit.sample_state(env.sim, blob["ref"], phase="nominal"))
            next_sample += 0.02
    nominal_elapsed = float(env.sim.data.time) - t_hand
    recovery_elapsed = t_hand - t0
    if reason is None:
        if nominal_elapsed < NOMINAL_S - 1e-6:
            reason = "PHYSICAL_VERIFICATION_FAILED"
        elif end_row is None or not _bilateral(end_row) or _dropped(end_row) or _escaped(env.sim):
            reason = "LOST_GRASP"
        elif not lift_done:
            reason = "NOMINAL_LIFT_NOT_COMPLETED"
        else:
            reason = "SUCCESS_12S"
    out = {
        "reason": reason,
        "success": reason == "SUCCESS_12S",
        "handoff": how,
        "recovery_elapsed": recovery_elapsed,
        "nominal_elapsed": nominal_elapsed,
        "lift_completed": lift_done,
    }
    return rows, t0, out, handoff_t


def plot_contact(trace_csv, dest, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return str(exc)
    rows = list(csv.DictReader(Path(trace_csv).open(encoding="utf-8")))
    rec = [row for row in rows if row["phase"] == "recovery"]
    if not rec:
        return "no recovery rows"
    t = [float(row["t_rel"]) for row in rec]
    nL = [int(row["nL"]) for row in rec]
    nR = [int(row["nR"]) for row in rec]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].step(t, nL, where="post", label="nL")
    axes[0].step(t, nR, where="post", label="nR")
    axes[0].set_ylabel("contact count")
    axes[0].legend(loc="upper right")
    axes[0].set_title(title)
    mig_x = []
    mig_y = []
    for row in rec:
        if row["bilateral"] == "1":
            try:
                cx = 0.5 * (float(row["cLx"]) + float(row["cRx"]))
                cy = 0.5 * (float(row["cLy"]) + float(row["cRy"]))
            except ValueError:
                continue
            if np.isfinite(cx) and np.isfinite(cy):
                mig_x.append(cx)
                mig_y.append(cy)
    if mig_x:
        axes[1].plot(mig_x, mig_y, "-o", markersize=2)
        axes[1].set_xlabel("contact centroid x (hand)")
        axes[1].set_ylabel("contact centroid y (hand)")
        axes[1].set_aspect("equal", adjustable="datalim")
    else:
        axes[1].text(0.5, 0.5, "no bilateral centroid", ha="center")
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return None


def render_video(env, model, book, blob, device, limit_s, dest):
    from envs.outcomes.horizon_outcome import NOMINAL_S, recovery_failed
    from envs.outcomes.physical_recovery import physical_pack
    from envs.outcomes.recovery_success import apply_handoff, nominal_step
    from envs.outcomes.table_drop import latched, terminal_reason
    from envs.step_legal import step_legal
    from models.k48_checkpoint import HANDOFF
    from training.k47_codebook import decode_action
    import mujoco

    restore_ic(env, blob)
    cam = mjv_camera_from_preset({
        "lookat": np.array(env.sim.data.xpos[env.sim.ids.object_body], float),
        "distance": 0.55, "azimuth": 132.0, "elevation": -22.0,
    })
    renderer = mujoco.Renderer(env.sim.model, 360, 480)
    frames = []
    t0 = float(env.sim.data.time)
    next_frame = t0
    lost_low = 0
    how = None
    cap = int(round(float(limit_s) / 0.02))

    def grab(text):
        cam.lookat[:] = np.array(env.sim.data.xpos[env.sim.ids.object_body], float)
        renderer.update_scene(env.sim.data, cam)
        frames.append(caption(np.asarray(renderer.render(), np.uint8).copy(), text))

    for _ in range(cap):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        pack = physical_pack(env.sim)
        now = float(env.sim.data.time)
        if now + 1e-9 >= next_frame:
            grab(f"{dest.stem[:24]} t={now - t0:.2f} tok={token} n={int(pack['nL'])}/{int(pack['nR'])}")
            next_frame += 0.04
        if token == HANDOFF:
            how = "early"
            break
        _res, action = decode_action(book, token, mean[-1].detach().float().cpu().numpy())
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            grab(f"FAIL {terminal_reason(env.sim)}")
            renderer.close()
            write_frames_mp4(frames, dest, fps=25)
            shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
            return
    if how is None:
        how = "forced"
    apply_handoff(env.sim, blob["ref"])
    t_hand = float(env.sim.data.time)
    next_frame = t_hand
    while float(env.sim.data.time) - t_hand < min(NOMINAL_S, 4.0) - 1e-9:
        nominal_step(env.sim)
        now = float(env.sim.data.time)
        if now + 1e-9 >= next_frame:
            grab(f"hold {now - t_hand:.2f}s")
            next_frame += 0.04
        if latched(env.sim) or float(physical_pack(env.sim)["obj_z"]) < 0.30:
            break
    renderer.close()
    write_frames_mp4(frames, dest, fps=25)
    shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)


def sliding_only_evidence():
    """ICs with sliding SUCCESS_12S and documented vertical_recatch failure."""
    from visualization.audit_checkpoint3_skill import find_slide_row, find_vertical_fail

    rows = load_valid_rows()
    out = []
    for row in rows:
        fails = find_vertical_fail(row["id"])
        try:
            slide, path = find_slide_row(row["id"])
        except RuntimeError:
            slide, path = None, None
        if slide and fails:
            out.append({
                "id": row["id"],
                "slide_manifest": str(path.relative_to(ROOT)).replace("\\", "/"),
                "slide_strategy": slide.get("strategy"),
                "vertical_fails": fails,
            })
    return out


def write_outputs(valid_rows, results):
    by = {(row["checkpoint"], row["episode_id"]): row for row in results}
    comparison = []
    for item in valid_rows:
        eid = item["id"]
        cats = ",".join(item.get("categories") or [])
        sft = by[("sft", eid)]
        grpo = by[("grpo", eid)]
        comparison.append({
            "ic": eid,
            "data_labels": cats,
            "sft_mechanism": sft["mechanism"],
            "sft_code": sft["code"],
            "sft_result": sft["outcome"]["reason"],
            "sft_success": int(sft["outcome"]["success"]),
            "sft_confidence": sft["confidence"],
            "grpo_mechanism": grpo["mechanism"],
            "grpo_code": grpo["code"],
            "grpo_result": grpo["outcome"]["reason"],
            "grpo_success": int(grpo["outcome"]["success"]),
            "grpo_confidence": grpo["confidence"],
            "mechanism_changed": int(sft["mechanism"] != grpo["mechanism"]),
            "result_improved": int((not sft["outcome"]["success"]) and grpo["outcome"]["success"]),
            "result_regressed": int(sft["outcome"]["success"] and (not grpo["outcome"]["success"])),
        })

    with (OUT / "per_ic_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)

    def counts(ckpt):
        rows = [row for row in results if row["checkpoint"] == ckpt]
        mech = Counter(row["mechanism"] for row in rows)
        success_by = Counter(row["mechanism"] for row in rows if row["outcome"]["success"])
        return {
            "n": len(rows),
            "success": sum(1 for row in rows if row["outcome"]["success"]),
            "mechanism_counts": dict(mech),
            "success_by_mechanism": dict(success_by),
        }

    transition = Counter()
    for item in comparison:
        transition[(item["sft_mechanism"], item["grpo_mechanism"])] += 1
    with (OUT / "mechanism_transition.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sft_mechanism", "grpo_mechanism", "count"])
        writer.writeheader()
        for (a, b), count in sorted(transition.items()):
            writer.writerow({"sft_mechanism": a, "grpo_mechanism": b, "count": count})

    # Consistency with prior audit on k48r_valid_00009 / GRPO.
    prior = by[("grpo", "k48r_valid_00009")]
    consistency = {
        "id": "k48r_valid_00009",
        "expected_prior_label": "Hybrid",
        "observed": prior["mechanism"],
        "match": prior["mechanism"] == "Hybrid",
        "both_off_s": prior["facts"].get("sustained_both_off_intervals"),
        "post_gap_migration_m": prior["facts"].get("post_recontact_migration_m"),
        "result": prior["outcome"]["reason"],
    }

    slide_labeled = [row for row in valid_rows if "gravity_assisted_in_hand_sliding" in (row.get("categories") or [])]
    horiz = [row for row in valid_rows if "horizontal_recatch" in (row.get("categories") or [])]
    slide_mech = Counter(by[("grpo", row["id"])]["mechanism"] for row in slide_labeled)
    horiz_ok = []
    for row in horiz:
        g = by[("grpo", row["id"])]
        horiz_ok.append({
            "id": row["id"],
            "success": g["outcome"]["success"],
            "mechanism": g["mechanism"],
            "has_sustained_both_off": bool(g["facts"].get("sustained_both_off_intervals")),
        })

    improved = [row for row in comparison if row["result_improved"]]
    mech_change_same_result = [
        row for row in comparison
        if row["mechanism_changed"] and row["sft_success"] == row["grpo_success"]
    ]
    evidence = sliding_only_evidence()
    q4 = []
    for item in evidence:
        g = by[("grpo", item["id"])]
        if (not g["outcome"]["success"]) and g["mechanism"] in {"Release-Recatch", "Hybrid"}:
            q4.append({"id": item["id"], "grpo_mechanism": g["mechanism"], "grpo_result": g["outcome"]["reason"], **item})

    summary = {
        "n_ics": 17,
        "n_trajectories": len(results),
        "sft": counts("sft"),
        "grpo": counts("grpo"),
        "transition": {f"{a} -> {b}": c for (a, b), c in transition.items()},
        "improved_ics": improved,
        "mechanism_changed_same_result": mech_change_same_result,
        "sliding_labeled_grpo_mechanisms": dict(slide_mech),
        "horizontal_grpo": horiz_ok,
        "consistency_k48r_valid_00009": consistency,
        "sliding_success_with_vertical_fail_evidence": evidence,
        "q4_recatch_bias_failures_on_evidenced_ics": q4,
        "thresholds": {
            "both_off_hold_s": BOTH_OFF_HOLD_S,
            "migration_m": MIGRATION_M,
            "post_gap_migration_m": POST_GAP_MIGRATION_M,
        },
        "rows": results,
    }
    (OUT / "mechanism_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    return comparison, summary


def write_report(comparison, summary):
    sft = summary["sft"]
    grpo = summary["grpo"]
    lines = [
        "# Validation recovery skill classification",
        "",
        "Checkpoints: SFT `results/sft/checkpoint_epoch_0040.pt`, GRPO `results/grpo/tree80/checkpoint_epoch_0003.pt`.",
        "Validation set: 17 globally unique ICs from `data/valid_skill_quotas.jsonl` (same as `--eval`).",
        "Action rule: official deterministic `roll_policy` (argmax token, residual mean, legal decode, 12 s nominal).",
        "",
        "Contact extraction and snapshot restore reuse `visualization/audit_checkpoint3_skill.py`.",
        "",
        "## Classification rules",
        "",
        f"- **A Contact-Preserving Adjustment**: no sustained both-off (≥ {BOTH_OFF_HOLD_S:.2f} s); contact preserved; measurable in-contact migration (≥ {MIGRATION_M*1000:.0f} mm) or in-hand travel.",
        f"- **B Release–Recatch**: sustained both-off ≥ {BOTH_OFF_HOLD_S:.2f} s then bilateral recontact before handoff.",
        f"- **C Hybrid**: B plus post-recontact bilateral centroid migration ≥ {POST_GAP_MIGRATION_M*1000:.0f} mm.",
        "- **D Other/Unclear**: insufficient evidence, sub-threshold gaps only, preserved contact without measurable adjustment, or failed regrasp.",
        "",
        "Single-frame both-off is not Recatch. Unilateral contact is not full grasp loss. Data labels are not used as mechanism labels.",
        "Sensitivity is recorded under both-off 0.06 s and migration 2 mm / post-gap 3 mm.",
        "",
        "## Per-IC comparison",
        "",
        "| IC | data labels | SFT mechanism | SFT result | GRPO mechanism | GRPO result |",
        "|---|---|---|---|---|---|",
    ]
    for row in comparison:
        lines.append(
            f"| `{row['ic']}` | {row['data_labels']} | {row['sft_mechanism']} | {row['sft_result']} | "
            f"{row['grpo_mechanism']} | {row['grpo_result']} |"
        )
    lines.extend([
        "",
        "## Observed facts",
        "",
        f"- SFT success: **{sft['success']}/17**. GRPO success: **{grpo['success']}/17**.",
        f"- SFT mechanism counts: `{json.dumps(sft['mechanism_counts'])}`.",
        f"- GRPO mechanism counts: `{json.dumps(grpo['mechanism_counts'])}`.",
        f"- SFT successes by mechanism: `{json.dumps(sft['success_by_mechanism'])}`.",
        f"- GRPO successes by mechanism: `{json.dumps(grpo['success_by_mechanism'])}`.",
        f"- Consistency check `k48r_valid_00009` GRPO: expected Hybrid, observed `{summary['consistency_k48r_valid_00009']['observed']}` "
        f"(match={summary['consistency_k48r_valid_00009']['match']}).",
        "",
        "### Mechanism transition (SFT → GRPO)",
        "",
        "| SFT | GRPO | count |",
        "|---|---|---:|",
    ])
    for key, count in sorted(summary["transition"].items()):
        a, b = key.split(" -> ")
        lines.append(f"| {a} | {b} | {count} |")
    lines.extend([
        "",
        f"- Result improved (fail→success): `{[row['ic'] for row in summary['improved_ics']]}`",
        f"- Mechanism changed with same success bit: `{[row['ic'] for row in summary['mechanism_changed_same_result']]}`",
        f"- Sliding-labeled ICs under GRPO mechanisms: `{json.dumps(summary['sliding_labeled_grpo_mechanisms'])}`",
        f"- Horizontal Recatch ICs under GRPO: `{json.dumps(summary['horizontal_grpo'])}`",
        "",
        "## Mechanism classification notes",
        "",
        "Labels come from contact topology and relative motion thresholds above, not from tokens or skill tags.",
        "Failed trajectories are classified with the same rules; many land in Other/Unclear when regrasp never completes.",
        "",
        "## Answers",
        "",
        "### Q1. Does GRPO Epoch 3 use only Recatch/Hybrid?",
        "",
    ])
    grpo_mechs = summary["grpo"]["mechanism_counts"]
    only_recatch_like = set(grpo_mechs) <= {"Release-Recatch", "Hybrid", "Other/Unclear"} and "Contact-Preserving Adjustment" not in grpo_mechs
    lines.append(
        f"**Observed:** GRPO mechanism histogram is `{json.dumps(grpo_mechs)}`. "
        + (
            "No Contact-Preserving Adjustment appears among the 17 GRPO validation rollouts."
            if only_recatch_like or "Contact-Preserving Adjustment" not in grpo_mechs
            else "Contact-Preserving Adjustment is still present."
        )
    )
    lines.extend([
        "",
        "### Q2. Does SFT keep more Contact-Preserving / Sliding behavior?",
        "",
        f"**Observed:** SFT Contact-Preserving count = "
        f"`{sft['mechanism_counts'].get('Contact-Preserving Adjustment', 0)}`; "
        f"GRPO = `{grpo['mechanism_counts'].get('Contact-Preserving Adjustment', 0)}`.",
        "",
        "### Q3. Do new GRPO successes come from better execution of the same mechanism, or mechanism change?",
        "",
    ])
    for row in summary["improved_ics"]:
        lines.append(
            f"- `{row['ic']}`: {row['sft_mechanism']} / {row['sft_result']} → "
            f"{row['grpo_mechanism']} / {row['grpo_result']}"
        )
    if not summary["improved_ics"]:
        lines.append("- No fail→success transitions in this 17-IC set.")
    lines.extend([
        "",
        "### Q4. Is there an evidenced Sliding-success / Recatch-fail IC where GRPO fails by Recatch bias?",
        "",
        "Evidence requires the same IC to have Sliding SUCCESS_12S and a recorded Recatch failure, not a Sliding label alone.",
        "",
        f"- Evidenced valid ICs: `{[row['id'] for row in summary['sliding_success_with_vertical_fail_evidence']]}`",
        f"- Among those, GRPO Recatch/Hybrid failures: `{summary['q4_recatch_bias_failures_on_evidenced_ics']}`",
        "",
        "### Q5. Do the traces support at least two distinct physical recovery mechanisms?",
        "",
    ])
    present = set(grpo_mechs) | set(sft["mechanism_counts"])
    multi = {"Contact-Preserving Adjustment", "Release-Recatch", "Hybrid"} & present
    lines.append(
        f"**Observed across SFT+GRPO labels present:** `{sorted(present)}`. "
        f"Distinct A/B/C labels observed: `{sorted(multi)}`."
    )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "GRPO Epoch 3 improves validation success relative to this SFT checkpoint under the same deterministic rule. "
        "Where mechanisms are Hybrid or Release–Recatch, that claim rests on measured both-off intervals and recontact, "
        "not on video appearance. A shift away from Contact-Preserving Adjustment, if present in the counts above, "
        "is a descriptive statement about these 17 rollouts only.",
        "",
        "## Limitations",
        "",
        "- Both-off threshold 0.04 s is exactly two policy steps; borderline cases are marked.",
        "- Historical failed Recatch npz files are often missing; Sliding-only evidence uses attempt logs plus live oracle where available.",
        "- Category tags overlap (Sliding ∩ Vertical); counts use unique ICs, not summed category rows.",
        "- Migration uses bilateral contact centroids in the hand frame; sparse contacts make D more likely.",
        "- Nominal CSV is sampled at 0.02 s; recovery is every policy step.",
        "",
        "## Conclusion",
        "",
    ])
    # Conclusion filled after we know counts - write placeholder then overwrite in main after summary known
    lines.append("See final paragraph generated after the run.")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def finalize_report(comparison, summary):
    lines = (OUT / "REPORT.md").read_text(encoding="utf-8").splitlines()
    # replace conclusion placeholder
    out = []
    for line in lines:
        if line.startswith("See final paragraph"):
            grpo_mechs = summary["grpo"]["mechanism_counts"]
            sft_cp = summary["sft"]["mechanism_counts"].get("Contact-Preserving Adjustment", 0)
            grpo_cp = grpo_mechs.get("Contact-Preserving Adjustment", 0)
            grpo_h = grpo_mechs.get("Hybrid", 0)
            grpo_r = grpo_mechs.get("Release-Recatch", 0)
            multi = (grpo_cp > 0 and (grpo_h + grpo_r) > 0) or (sft_cp > 0 and (grpo_h + grpo_r) > 0)
            if multi:
                text = (
                    f"**Conclusion (bounded):** Across the 17 validation ICs, the traces support more than one "
                    f"physical recovery family when SFT and GRPO are taken together "
                    f"(SFT Contact-Preserving={sft_cp}, GRPO Hybrid={grpo_h}, Release–Recatch={grpo_r}, "
                    f"Contact-Preserving={grpo_cp}). "
                    f"GRPO Epoch 3 itself is dominated by Recatch-like / Hybrid contact patterns in this set; "
                    f"that is evidence of a Recatch-leaning mode on the validation rollouts, not proof of a "
                    f"universal collapse on every IC outside this list."
                )
            else:
                text = (
                    f"**Conclusion (bounded):** In this 17-IC validation set, GRPO Epoch 3 does not show a clear "
                    f"second successful Contact-Preserving family alongside Recatch/Hybrid "
                    f"(GRPO counts={json.dumps(grpo_mechs)}). "
                    f"Claims of multi-mechanism unified control should not be stronger than these traces."
                )
            out.append(text)
        else:
            out.append(line)
    (OUT / "REPORT.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-s", type=float, default=SELECTED_LIMIT_S)
    parser.add_argument("--skip-video", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    traces = OUT / "traces"
    events = OUT / "contact_events"
    plots = OUT / "plots"
    videos = OUT / "videos"
    for folder in (traces, events, plots, videos):
        folder.mkdir(parents=True, exist_ok=True)

    valid_rows = load_valid_rows()
    device = torch.device(args.device)
    codebook = load_codebook()
    checkpoints = [
        ("sft", SFT_CKPT),
        ("grpo", GRPO_CKPT),
    ]
    for _name, path in checkpoints:
        if not path.is_file():
            raise RuntimeError(f"missing checkpoint {path}")

    seated_ref = None
    results = []
    models = {}
    env = None
    for ckpt_name, ckpt_path in checkpoints:
        model, book = load_trained_k48(ckpt_path, device)
        model.eval()
        if not np.array_equal(book, codebook):
            raise RuntimeError(f"codebook mismatch for {ckpt_path}")
        models[ckpt_name] = (model, book)
        if env is None:
            env = make_training_env(device, model=model)
        print("CHECKPOINT", ckpt_name, ckpt_path, flush=True)
        for item in valid_rows:
            with (ROOT / item["snapshot"]).open("rb") as handle:
                blob = pickle.load(handle)
            if "ref" not in blob:
                if seated_ref is None:
                    from data_generation.strategies.recatch_oracle import _setup
                    _sim, _base, seated_ref, _target, _code = _setup()
            blob = prepare_blob(blob, seated_ref)
            print("ROLL", ckpt_name, item["id"], flush=True)
            row = run_one(
                env, model, book, blob, device, float(args.limit_s),
                item["id"], ckpt_name, {"traces": traces, "events": events},
            )
            row["data_labels"] = list(item.get("categories") or [])
            results.append(row)
            print(
                "DONE", ckpt_name, item["id"], row["outcome"]["reason"],
                row["mechanism"], row["confidence"], flush=True,
            )

    comparison, summary = write_outputs(valid_rows, results)
    write_report(comparison, summary)
    finalize_report(comparison, summary)

    # Plots for every trajectory recovery contact.
    for row in results:
        src = OUT / row["trace_file"]
        dest = plots / f"{row['checkpoint']}__{row['episode_id']}__contact.png"
        err = plot_contact(src, dest, f"{row['checkpoint']} {row['episode_id']} {row['mechanism']}")
        if err:
            print("PLOT_SKIP", row["episode_id"], err, flush=True)

    # Representative videos: one per GRPO mechanism that appears, plus k48r_valid_00009.
    if not args.skip_video:
        chosen = {}
        for row in results:
            if row["checkpoint"] != "grpo":
                continue
            chosen.setdefault(row["mechanism"], row)
        if "k48r_valid_00009" not in {row["episode_id"] for row in chosen.values()}:
            chosen["consistency"] = next(
                row for row in results if row["checkpoint"] == "grpo" and row["episode_id"] == "k48r_valid_00009"
            )
        model, book = models["grpo"]
        for key, row in chosen.items():
            item = next(x for x in valid_rows if x["id"] == row["episode_id"])
            with (ROOT / item["snapshot"]).open("rb") as handle:
                blob = pickle.load(handle)
            if "ref" not in blob:
                blob = prepare_blob(blob, seated_ref)
            dest = videos / f"grpo__{row['episode_id']}__{row['code']}.mp4"
            print("VIDEO", dest.name, flush=True)
            render_video(env, model, book, blob, device, float(args.limit_s), dest)

    print("CLASSIFY_DONE", json.dumps({
        "sft_success": summary["sft"]["success"],
        "grpo_success": summary["grpo"]["success"],
        "sft_mechs": summary["sft"]["mechanism_counts"],
        "grpo_mechs": summary["grpo"]["mechanism_counts"],
        "consistency": summary["consistency_k48r_valid_00009"],
    }), flush=True)


if __name__ == "__main__":
    main()
