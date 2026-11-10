"""Render four verified P64 success videos (A1/B1/C/D) vs do-nothing drop.

Continuous dual-sim: seated → ball flight → impact on BOTH panels, then fork
P64 recovery (left) vs do-nothing drop (right). No mid-video restore/splice.

Read-only visualization: does not retrain, modify P64, or open sealed panels.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from controllers.cartesian.jacobian import apply_cartesian_ctrl  # noqa: E402
from controllers.cartesian.tracking import TRACK  # noqa: E402
from controllers.cartesian.velocity_step import tick_vw  # noqa: E402
from data_generation.datasets.ball_impact import aim_ball, cylinder_contacts  # noqa: E402
from data_generation.oracles.plan_c.future_trajectory import (  # noqa: E402
    ball_hits_cylinder,
    hold_action,
    launch_ball,
)
from data_generation.oracles.plan_c.interception import rebuild_settled_root  # noqa: E402
from envs.ballistic_impact import HAND_RISE_M, TAU_SEC  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.ground_drop import arm, latched, restore_state as restore_drop  # noqa: E402
from envs.outcomes.ground_drop import disarm  # noqa: E402
from envs.outcomes.horizon_outcome import NOMINAL_S  # noqa: E402
from envs.outcomes.physical_recovery import physical_pack  # noqa: E402
from envs.outcomes.recovery_success import apply_handoff, nominal_step  # noqa: E402
from envs.simulation.impact_ball import apply_ball_free_state  # noqa: E402
from envs.step_legal import N_SUB, step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p53_independent_grpo64.success_relaxed import (  # noqa: E402
    evaluate_success_relaxed_12s,
)
from experiments.precision.final_independent_success_rate.paths import P64_ACTOR  # noqa: E402
from experiments.precision.p50_ppo_reward_redesign.env_b_recovery import state_metrics  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import sample_step  # noqa: E402
from experiments.precision.p53_independent_grpo64.probabilistic_eval import (  # noqa: E402
    load_heldout_starts,
)
from experiments.precision.p57_post_recatch_stabilization.criteria import RecatchTracker  # noqa: E402
from experiments.precision.p61_reproducible_joint_eval.eval_core import (  # noqa: E402
    load_actor,
    make_eval_env,
    rollout_seeded,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import (  # noqa: E402
    rollout_seed,
    seed_all,
)
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
import models.transformer_policy as v3  # noqa: E402
from training.recovery_episode import fresh_nominal_state, nominal_substep  # noqa: E402
from training.recovery_execution import make_snap, restore_ballistic  # noqa: E402
from visualization.camera import mjv_camera_from_preset  # noqa: E402
from visualization.compare_strategy_nominal import load_blob, load_row  # noqa: E402
from visualization.overlays import caption  # noqa: E402
from visualization.video_recorder import write_frames_mp4  # noqa: E402

EXPECTED_SHA = "ec08f59816aa2fdd"
P64_JSON = ROOT / "results/experiments/p64_ppo_multiround/P64_step4.json"
P83_MD = ROOT / "results/experiments/p83_recoverable_failure_mass/BASELINE_PANEL_VALIDATION.md"
OUT_VID = ROOT / "videos"
CACHE = OUT_VID / "_cache"
SPLIT = "heldout"
FPS = 24
# MuJoCo default offscreen buffer is 640×480; render native then upscale to 1280×720.
W, H = 640, 480
OUT_W, OUT_H = 1280, 720
SAMPLE_S = 0.04
HOLD_KEEP_EVERY = 2  # mild visual speedup of uneventful 12s hold
CL_SEED_PROBE = 64
DROP_Z = 0.08

# Preferred high-yield ICs from P64_step4 (heldout / exposed-dev style panel)
PREFERRED = {
    "A1": "A__slide_quota_400660__ef7cfa9e9ca57a40",
    "B1": "plan_b__slide_extra_260006",
    "C": "plan_c__gbi_041_7f94ea38dd",
    # Upright fingers (world −Z); prior DTHETA IC looked sideways at start.
    "D": "p33_ball_train_0001_COMBINED_dx4_dz12",
}

# Flight-prefix lookup (physical episode id → sliding manifest)
FLIGHT_MANIFEST = {
    "A1": ("data/inward_sliding/manifest.jsonl", "slide_quota_400660", "inward_reseat"),
    "B1": ("data/controlled_sliding/manifest.jsonl", "slide_extra_260006", "controlled_slide"),
    "D": ("data/immediate_handoff/manifest.jsonl", "ball_train_0001", "immediate_handoff"),
}

# Mild in-hand slip does not drop under fixed-arm closed hold; slightly stronger impact does.
SLIDING_IMPACT_SPEED_SCALE = {
    "A1": 1.25,
    "B1": 1.0,
    "D": 1.15,
}

# Closed-grip nominal hold on the right (negative = close). Slightly lighter than
# TAU_SEC so post-impact slip is visible without commanding an open.
NOMINAL_GRIP_TAU = -6.0

# Presentation pacing: slow-mo ball flight / impact / recovery, then resume 1×.
SLOW_FLIGHT_X = 8  # each captured flight/impact frame repeated
SLOW_RECOVERY_X = 3  # each P64 recovery step frame repeated
# Sliding keeps dataset aim_ball (recoverability). Longer aim_s is gravity-
# compensated; B1 is fragile so it stays on the short dataset flight.
SLIDING_FLIGHT_AIM_S = {"A1": 0.35, "B1": 0.10, "D": 0.30}
# Prefer visibility pullback; render falls back to 0.0 if recoverability breaks.
C_BALL_PULLBACK_CANDIDATES_M = (0.08, 0.04, 0.0)
FLIGHT_AIM_S = 0.45  # used only by the unused D capture-ball helper


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def emit_frames(frames: list, img, times: int = 1) -> None:
    for _ in range(max(1, int(times))):
        frames.append(img)


def slow_tag(label: str, slow_x: int) -> str:
    if slow_x <= 1:
        return label
    return f"{label} [{slow_x}x SLOW]"


def aim_ball_long(sim, spec, *, speed_scale: float = 1.0, flight_s: float = FLIGHT_AIM_S) -> dict:
    """Aim like dataset ``aim_ball``, but with a longer approach for visibility."""
    speed = float(spec["speed_mps"]) * float(speed_scale)
    base = aim_ball(
        sim,
        spec["azimuth_deg"],
        speed,
        spec["z_off_m"],
        spec["phi_deg"],
    )
    approach = np.asarray(base["approach"], float)
    p_aim = np.asarray(base["p_aim"], float)
    # Same geometry as aim_ball, but replace the short FLIGHT_S with flight_s.
    p_launch = p_aim - approach * (speed * float(flight_s))
    g = np.array(sim.model.opt.gravity, float)
    v0 = (p_aim - p_launch) / float(flight_s) - 0.5 * g * float(flight_s)
    return {
        "p_launch": p_launch,
        "v0": v0,
        "p_aim": p_aim,
        "approach": approach,
        "flight_s": float(flight_s),
    }


def snap_hash(obj) -> str:
    return hashlib.sha256(pickle.dumps(obj, protocol=4)).hexdigest()[:16]


def pair(left, right, lt, rt):
    stacked = np.hstack([caption(left, lt), caption(right, rt)])
    return upscale_frame(stacked)


def upscale_frame(img, w=OUT_W, h=OUT_H):
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR), np.uint8)
    except Exception:
        return img


def resize(img, w=W, h=H):
    """Keep native MuJoCo resolution for intermediate grabs."""
    return np.asarray(img, np.uint8).copy()


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")).replace(",", ";") for k in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_counts() -> dict:
    d = json.loads(P64_JSON.read_text(encoding="utf-8"))
    by = d["eval"]["by_class"]
    # P64_step4: 4 ICs × 64 seeds per class (heldout)
    per = d["eval_full_per_ic"]
    # Count ICs with ≥1 success / 4 per class — NOT 6/16 style
    report = {
        "provisional_claimed": "6/16, 4/16, 3/16, 7/16",
        "provisional_valid": False,
        "official_p64_step4_msp_relaxed": {
            "A1": by["A"]["MSP_RELAXED"],
            "B1": by["B"]["MSP_RELAXED"],
            "C": by["C"]["MSP_RELAXED"],
            "D": by["D"]["MSP_RELAXED"],
            "overall": d["eval"]["MSP_RELAXED"],
            "panel": "heldout 4 IC × 64 seeds / class (P61/P62/P64 protocol)",
        },
        "p83_panel_success_counts": {
            "A1": "8/32",
            "B1": "6/32",
            "C": "4/32",
            "D": "17/32",
            "note": "P83 independent baseline panel; same P64 actor MD5",
        },
        "correction": (
            "Provisional 6/16·4/16·3/16·7/16 do NOT match any frozen P64 panel. "
            "Use P64_step4 MSP rates or P83 8/32·6/32·4/32·17/32 in slides."
        ),
        "per_ic": [
            {
                "skill": r["skill"],
                "traj_id": r["traj_id"],
                "n_success_relaxed": r["n_success_relaxed"],
                "n_rollouts": r["n_rollouts"],
                "provenance": "heldout_dev_style",
            }
            for r in per
        ],
    }
    (OUT_VID / "COUNT_CORRECTION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def enumerate_successes(actor, book, rho, device, starts_by_skill) -> list[dict]:
    rows = []
    for skill, tid in PREFERRED.items():
        start = next(s for s in starts_by_skill[skill] if s.traj_id == tid)
        ep = start.ep or {}
        sh = ""
        if start.kind == "capture" and start.capture is not None:
            sh = snap_hash(start.capture)
        elif ep.get("catch_snap") is not None:
            sh = snap_hash(ep["catch_snap"])
        elif ep.get("capture") is not None:
            sh = snap_hash(ep["capture"])
        env = make_eval_env(book, rho)
        successes = []
        for i in range(CL_SEED_PROBE):
            seed = rollout_seed(split=SPLIT, traj_id=tid, rollout_i=i)
            out = rollout_seeded(env, actor, start, device, seed=seed)
            if int(out["SUCCESS_RELAXED_12S"]) == 1:
                successes.append(
                    {
                        "rollout_i": i,
                        "seed": seed,
                        "n_steps": int(out["n_steps"]),
                        "recatch": int(out.get("recatch") or 0),
                        "stable_hold": int(out.get("stable_hold") or 0),
                        "handoff": int(out.get("handoff") or 0),
                    }
                )
            print(
                f"  probe {skill} i={i} succ={out['SUCCESS_RELAXED_12S']} "
                f"steps={out['n_steps']} recatch={out.get('recatch')}",
                flush=True,
            )
            # early stop if plenty
            if skill != "C" and len(successes) >= 12:
                break
        for s in successes:
            rows.append(
                {
                    "skill": skill,
                    "traj_id": tid,
                    "snapshot_hash": sh,
                    "seed": s["seed"],
                    "rollout_i": s["rollout_i"],
                    "n_steps": s["n_steps"],
                    "recatch": s["recatch"],
                    "stable_hold": s["stable_hold"],
                    "handoff": s["handoff"],
                    "SUCCESS_RELAXED_12S": 1,
                    "provenance": "heldout_p64_step4_panel",
                    "independent_label": 0,
                }
            )
    write_csv(OUT_VID / "SUCCESS_ENUMERATION.csv", rows)
    return rows


def select_representatives(rows: list[dict]) -> list[dict]:
    """Median successful recovery duration among nontrivial successes; tie-break traj_id."""
    selected = []
    for skill in ("A1", "B1", "C", "D"):
        sub = [r for r in rows if r["skill"] == skill]
        if not sub:
            raise RuntimeError(f"no SUCCESS_RELAXED_12S for {skill}")
        # nontrivial filters
        if skill in ("A1", "B1", "D"):
            nt = [r for r in sub if int(r["n_steps"]) >= 5]
            label = "corrective"
            if not nt:
                nt = sub
                label = "immediate_HANDOFF_fallback"
        else:  # C: require recatch
            nt = [r for r in sub if int(r["recatch"]) == 1 and int(r["n_steps"]) >= 8]
            label = "detach_recatch"
            if not nt:
                nt = [r for r in sub if int(r["n_steps"]) >= 8]
                label = "C_no_recatch_flag_fallback"
            if not nt:
                nt = sub
                label = "immediate_HANDOFF_fallback"
        nt = sorted(nt, key=lambda r: (int(r["n_steps"]), str(r["traj_id"]), int(r["seed"])))
        mid = nt[len(nt) // 2]
        mid = dict(mid)
        mid["selection_rule"] = (
            "median_n_steps_among_nontrivial_successes; "
            "tie-break full traj_id then seed; "
            f"behavior_label={label}"
        )
        mid["behavior_label"] = label
        selected.append(mid)
    write_csv(OUT_VID / "SELECTION_MANIFEST.csv", selected)
    return selected


def attach_live(env, start, *, defer_handoff_eval: bool = False) -> np.ndarray:
    """Prime GRPO buffers on the current physical state — no teleport."""
    env.start = start
    env.t = 0
    env.prev_tok = None
    env.comp_sums = {k: 0.0 for k in env.comp_sums}
    if hasattr(env, "recatch"):
        env.recatch = RecatchTracker()
        env.stab_sums = {k: 0.0 for k in env.stab_sums}
        env.diag = {
            "recatch": 0,
            "recatch_step": None,
            "stable_hold": 0,
            "lost_after_recatch": 0,
            "post_recatch_steps": 0,
        }
    env.env._hist = ObservableObsState(DT_POLICY)
    env.env._prev = np.zeros(7, np.float32)
    env.env.ref = start.ref if start.kind == "capture" else (start.ep or {}).get("ref")
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    env.obs_seq = []
    env.prev_seq = []
    # When True, handoff SUCCESS is scored on a clone and nominal return is filmed.
    env.defer_handoff_eval = bool(defer_handoff_eval)
    obs = env._observe()
    env.obs_seq.append(obs)
    env.prev_seq.append(env.env._prev.copy())
    env._prev_m = state_metrics(env.env.sim)
    return obs


def flight_hold_no_park(sim, gains, tau: float = TAU_SEC) -> None:
    """Cartesian hold that does NOT park the impact ball (unlike tick_vw)."""
    apply_cartesian_ctrl(
        sim.model,
        sim.data,
        sim.ids,
        sim.fsm.p_des,
        sim.fsm.r_des,
        gripper_tau=float(tau),
        v_des=np.zeros(3),
        w_des=np.zeros(3),
        **gains,
    )
    mujoco.mj_step(sim.model, sim.data)


# Locked presentation camera — never re-aim mid-clip (re-aim reads as a splice).
CAM_PRESET = {
    "lookat": [0.55, 0.0, 0.52],
    "distance": 1.10,
    "azimuth": 135,
    "elevation": -18,
}


def _open_renderers(sim_l, sim_r):
    cam = mjv_camera_from_preset(CAM_PRESET)
    renderer_l = mujoco.Renderer(sim_l.model, H, W)
    renderer_r = mujoco.Renderer(sim_r.model, H, W)
    return cam, renderer_l, renderer_r


def _grab(sim, cam, renderer, include_ball: bool = True):
    # ``include_ball`` kept for call-site compat; camera stays fixed.
    _ = include_ball
    renderer.update_scene(sim.data, cam)
    return resize(np.asarray(renderer.render(), np.uint8).copy())


def _bind_live_policy_env(book, rho, live_sim):
    """Wrap a live ballistic sim for P64 without teleporting into a fresh model."""
    env = make_eval_env(book, rho)
    env.env.sim = live_sim
    return env


def _mirror_sim(dst_sim, src_sim) -> None:
    """Copy full ballistic state so dual panels stay pixel-synced pre-fork."""
    restore_ballistic(dst_sim, make_snap(src_sim))
    if hasattr(src_sim, "captured"):
        dst_sim.captured = bool(src_sim.captured)
    if hasattr(src_sim, "fsm") and hasattr(dst_sim, "fsm"):
        dst_sim.fsm.phase = src_sim.fsm.phase
        if hasattr(src_sim.fsm, "lift_started"):
            dst_sim.fsm.lift_started = bool(src_sim.fsm.lift_started)


def _obj_z(sim) -> float:
    return float(sim.data.xpos[sim.ids.object_body][2])


def _right_dropped(sim) -> bool:
    return bool(latched(sim) or _obj_z(sim) < DROP_Z)


def _freeze_right_hold(sim_r) -> float:
    """Freeze world p_des/r_des at current pose; return hand_z0."""
    sim_r.fsm.p_des = np.array(sim_r.data.xpos[sim_r.ids.hand_body], float).copy()
    sim_r.fsm.r_des = np.array(sim_r.data.xmat[sim_r.ids.hand_body].reshape(3, 3), float).copy()
    sim_r.fsm.phase = "lift"
    sim_r.fsm.lift_started = True
    return float(sim_r.fsm.p_des[2])


def _step_right_hold(sim_r, *, right_done: bool) -> bool:
    if right_done:
        return True
    for _ in range(N_SUB):
        tick_vw(sim_r, np.zeros(3), np.zeros(3), NOMINAL_GRIP_TAU, TRACK)
    return _right_dropped(sim_r)


def _play_visible_handoff(
    *,
    book,
    rho,
    start,
    skill: str,
    sim_l,
    sim_r,
    cam,
    renderer_l,
    renderer_r,
    frames: list,
    t_fork: float,
    right_done: bool,
    right_steps: int,
    right_keep_every: int = 4,
) -> tuple[int, bool, int, dict]:
    """Score SUCCESS on a clone; film stabilize + orientation return on the live sim."""
    from data_generation.oracles.precision.corrections.contact_primitives import _wrist

    ref = start.ref if start.kind == "capture" else (start.ep or {}).get("ref")
    if ref is None:
        raise RuntimeError(f"{skill} missing ref for visible handoff")

    pre = make_snap(sim_l)
    eval_env = make_eval_env(book, rho)
    restore_ballistic(eval_env.env.sim, pre)
    restore_drop(eval_env.env.sim, None)
    arm(eval_env.env.sim)
    ev = evaluate_handoff_with_axis(eval_env.env.sim, ref, stabilize_n=8)
    rel = evaluate_success_relaxed_12s(ev)
    ok = int(rel["SUCCESS_RELAXED_12S"])

    hold = _wrist(0.0, 0.0, 1.0)
    for _ in range(8):
        step_legal(sim_l, hold)
        right_done = _step_right_hold(sim_r, right_done=right_done)
        right_steps += 1
        rt = float(sim_r.data.time) - t_fork
        fr_l = _grab(sim_l, cam, renderer_l)
        fr_r = _grab(sim_r, cam, renderer_r)
        emit_frames(
            frames,
            pair(
                fr_l,
                fr_r,
                slow_tag(f"P64|{skill} HANDOFF stabilize", SLOW_RECOVERY_X),
                f"NOMINAL|{skill} {'DROPPED' if right_done else 'fixed hold'} t={rt:.2f}s",
            ),
            SLOW_RECOVERY_X,
        )

    apply_handoff(sim_l, ref)
    t_h = float(sim_l.data.time)
    next_t = t_h
    # Dense sampling while orientation slews back to ref (归正).
    sample_s = 0.02
    print(
        f"  [{skill}] visible handoff start ok={ok} t={t_h:.2f}s",
        flush=True,
    )
    while float(sim_l.data.time) - t_h < float(NOMINAL_S) - 1e-9:
        nominal_step(sim_l)
        right_done = _step_right_hold(sim_r, right_done=right_done)
        right_steps += 1
        now = float(sim_l.data.time)
        t_rel = now - t_h
        if now + 1e-9 >= next_t:
            rt = float(sim_r.data.time) - t_fork
            # Always film the return — do not thin frames just because right dropped.
            fr_l = _grab(sim_l, cam, renderer_l)
            fr_r = _grab(sim_r, cam, renderer_r)
            slow_x = 3 if t_rel < 3.0 else 1
            l_label = slow_tag(
                (
                    f"P64|{skill} HANDOFF return t={t_rel:.2f}s"
                    if t_rel < 3.0 or not ok
                    else f"P64|{skill} SUCCESS_RELAXED_12S"
                ),
                slow_x,
            )
            emit_frames(
                frames,
                pair(
                    fr_l,
                    fr_r,
                    l_label,
                    f"NOMINAL|{skill} {'DROPPED' if right_done else 'fixed hold'} t={rt:.2f}s",
                ),
                slow_x,
            )
            next_t += sample_s
        # Keep ≥3s of 归正 on screen; then stop if right already dropped.
        if right_done and ok and t_rel > 5.0:
            break

    meta = {
        "SUCCESS_RELAXED_12S": ok,
        "SUCCESS_12S": int(rel["SUCCESS_12S"]),
        "SUCCESS_NEW": int(rel["SUCCESS_NEW"]),
        "nominal_reason": ev.get("reason"),
        "handoff": 1,
    }
    return ok, right_done, right_steps, meta


def _run_fork_p64_vs_drop(
    actor,
    start,
    book,
    rho,
    device,
    seed: int,
    skill: str,
    snap=None,
    *,
    frames: list,
    env_l=None,
    env_r=None,
    cam=None,
    renderer_l=None,
    renderer_r=None,
    close_renderers: bool = True,
    defer_handoff_eval: bool = False,
) -> dict:
    """Fork: left P64, right fixed-pose nominal hold until drop.

    Pass live ``env_l``/``env_r`` already at the impact state to keep the camera
    continuous (Plan-C). Otherwise restore ``snap`` into fresh eval envs.

    Right freezes world ``p_des``/``r_des`` and steps with ``tick_vw`` zeros +
    closed grip. Do **not** use ``step_legal`` — it re-snaps ``p_des`` each step
    so the arm sinks with the object.

    Set ``defer_handoff_eval=True`` for filmed clips so ~12s nominal return is
    visible (otherwise evaluate_handoff collapses it into one unfilmed step).
    """
    own_envs = env_l is None or env_r is None
    if own_envs:
        if snap is None:
            raise RuntimeError("fork requires snap when envs are not provided")
        env_l = make_eval_env(book, rho)
        env_r = make_eval_env(book, rho)
        restore_ballistic(env_l.env.sim, snap)
        restore_ballistic(env_r.env.sim, snap)
        restore_drop(env_l.env.sim, None)
        restore_drop(env_r.env.sim, None)
        arm(env_l.env.sim)
        arm(env_r.env.sim)

    seed_all(seed, deterministic_torch=True)
    attach_live(env_l, start, defer_handoff_eval=defer_handoff_eval)

    sim_l = env_l.env.sim
    sim_r = env_r.env.sim
    hand_z0 = _freeze_right_hold(sim_r)

    own_renderers = renderer_l is None or renderer_r is None or cam is None
    if own_renderers:
        cam, renderer_l, renderer_r = _open_renderers(sim_l, sim_r)
        close_renderers = True

    done = False
    info: dict = {}
    step_i = 0
    phase = "RECOVERY"
    right_done = False
    t_fork = float(sim_r.data.time)
    right_steps = 0
    right_keep_every = 4
    max_right_s = 12.0

    try:
        while True:
            rt = float(sim_r.data.time) - t_fork
            record = True
            if done and (not right_done) and (right_steps % right_keep_every != 0):
                record = False

            # Slow-mo through RECOVERY; resume 1× after handoff / episode end.
            if (not done) and phase == "RECOVERY":
                slow_x = SLOW_RECOVERY_X
            else:
                slow_x = 1

            if record:
                fr_l = _grab(sim_l, cam, renderer_l, include_ball=False)
                fr_r = _grab(sim_r, cam, renderer_r, include_ball=False)
                if right_done:
                    r_label = f"NOMINAL|{skill} DROPPED t={rt:.2f}s"
                elif done:
                    r_label = f"NOMINAL|{skill} fixed hold [{right_keep_every}x] t={rt:.1f}s"
                else:
                    r_label = slow_tag(
                        f"NOMINAL|{skill} fixed hold (no recovery) t={rt:.2f}s",
                        slow_x,
                    )
                if done:
                    ok_so_far = int(info.get("SUCCESS_RELAXED_12S") or 0)
                    l_label = (
                        f"P64|{skill} SUCCESS_RELAXED_12S"
                        if ok_so_far
                        else f"P64|{skill} {phase}"
                    )
                else:
                    l_label = slow_tag(
                        f"P64|{skill} {phase} t={float(sim_l.data.time):.2f}s",
                        slow_x,
                    )
                emit_frames(frames, pair(fr_l, fr_r, l_label, r_label), slow_x)

            if not done:
                tok, resid, _lp, _m = sample_step(actor, env_l.obs_seq, env_l.prev_seq, device)
                _r, done, info = env_l.step(int(tok), resid)
                step_i += 1
                if int(info.get("handoff") or 0) == 1:
                    phase = "HANDOFF"
                    deferred = bool((info.get("handoff_diag") or {}).get("deferred_handoff_eval"))
                    if deferred:
                        ok_h, right_done, right_steps, hand_meta = _play_visible_handoff(
                            book=book,
                            rho=rho,
                            start=start,
                            skill=skill,
                            sim_l=sim_l,
                            sim_r=sim_r,
                            cam=cam,
                            renderer_l=renderer_l,
                            renderer_r=renderer_r,
                            frames=frames,
                            t_fork=t_fork,
                            right_done=right_done,
                            right_steps=right_steps,
                            right_keep_every=right_keep_every,
                        )
                        info = dict(info)
                        info.update(hand_meta)
                        info["SUCCESS_RELAXED_12S"] = ok_h
                        # Right already advanced inside visible handoff.
                        if done and right_done:
                            break
                        continue

            if not right_done:
                # Keep p_des fixed — do not call step_legal (it follows a sinking hand).
                right_done = _step_right_hold(sim_r, right_done=right_done)
                right_steps += 1

            if done and right_done:
                break
            if (not right_done) and rt > max_right_s:
                raise RuntimeError(
                    f"{skill} nominal fixed-hold did not drop within {max_right_s}s "
                    f"(z={_obj_z(sim_r):.3f}, arm_dz={float(sim_r.data.xpos[sim_r.ids.hand_body][2]) - hand_z0:.3f})"
                )
            if step_i > int(env_l.max_steps) + 5 and not done:
                break

        ok = int(info.get("SUCCESS_RELAXED_12S") or 0)
        arm_dz = float(sim_r.data.xpos[sim_r.ids.hand_body][2]) - hand_z0
        for k in range(36):
            fr_l = _grab(sim_l, cam, renderer_l, include_ball=False)
            fr_r = _grab(sim_r, cam, renderer_r, include_ball=False)
            if k % HOLD_KEEP_EVERY == 0:
                emit_frames(
                    frames,
                    pair(
                        fr_l,
                        fr_r,
                        f"P64|{skill} {'SUCCESS_RELAXED_12S' if ok else phase}",
                        f"NOMINAL|{skill} DROPPED",
                    ),
                    1,
                )
    finally:
        if close_renderers:
            renderer_l.close()
            renderer_r.close()

    return {
        "SUCCESS_RELAXED_12S": ok,
        "n_steps": step_i,
        "handoff": int(info.get("handoff") or 0),
        "fail_reason": info.get("fail_reason"),
        "recatch": int((info.get("c_diag") or {}).get("recatch") or info.get("recatch") or 0),
        "sim_time": float(sim_l.data.time),
        "right_dropped": int(_right_dropped(sim_r)),
        "right_mode": "fixed_pose_closed_hold",
        "right_grip_tau": float(NOMINAL_GRIP_TAU),
        "right_drop_s": float(sim_r.data.time) - t_fork,
        "right_arm_dz": float(arm_dz),
        "continuous": True,
        "same_sim_fork": (not own_envs),
        "slow_recovery_x": int(SLOW_RECOVERY_X),
    }


def _pull_ball_back(sim, p_launch, v0, pullback_m: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Move launch point along −v; keep v0 so impact impulse stays the same."""
    p = np.asarray(p_launch, float).copy()
    v = np.asarray(v0, float).copy()
    speed = float(np.linalg.norm(v))
    info = {"pullback_m": 0.0, "flight_est_s": 0.0, "speed": speed}
    if speed >= 1e-6 and pullback_m > 0:
        p = p - (v / speed) * float(pullback_m)
        info["pullback_m"] = float(pullback_m)
        info["flight_est_s"] = float(pullback_m) / speed
    apply_ball_free_state(sim, p, v)
    mujoco.mj_forward(sim.model, sim.data)
    return p, v, info


def _launch_ball_pulled(sim, cfg: dict, pullback_m: float = 0.0) -> dict:
    """Cfg launch then pull the ball back along −v so the approach is visible."""
    info = launch_ball(
        sim,
        speed=float(cfg["speed"]),
        approach=str(cfg["approach"]),
        offset=float(cfg["offset"]),
        z_off=float(cfg.get("z_off", 0.0)),
    )
    p2, v2, extra = _pull_ball_back(sim, info["p_launch"], info["v0"], pullback_m)
    out = dict(info)
    out["p_launch"] = p2.tolist()
    out["v0"] = v2.tolist()
    out.update(extra)
    return out


def render_continuous_plan_c(actor, start, book, rho, device, seed: int) -> tuple[list, dict]:
    """Settled grasp → cfg ball launch → impact → same-sim P64 vs do-nothing drop.

    Uses eval envs for the whole clip so policy takeover does not teleport /
    rebuild, which previously made the left panel discontinuous at impact.
    """
    ep = start.ep or {}
    pkl_path = ep.get("raw_pkl")
    expert = pickle.loads(Path(pkl_path).read_bytes()) if pkl_path else {}
    cfg = expert.get("cfg") or {}
    if not cfg:
        raise RuntimeError("Plan-C cfg missing; cannot rebuild continuous ball incidence")

    # Same eval envs for shared flight AND post-impact fork (no restore jump).
    env_l = make_eval_env(book, rho)
    env_r = make_eval_env(book, rho)
    sim_l, sim_r = env_l.env.sim, env_r.env.sim
    pre_l = rebuild_settled_root(sim_l, cfg)
    pre_r = rebuild_settled_root(sim_r, cfg)
    if not (pre_l.get("bilateral") and pre_r.get("bilateral")):
        raise RuntimeError(f"Plan-C seated grasp failed: {pre_l} / {pre_r}")

    cam, renderer_l, renderer_r = _open_renderers(sim_l, sim_r)
    frames: list = []
    act = hold_action(1.0)

    # Static seated beat (no physics drift) so the grasp is readable before launch.
    fr_seat_l = _grab(sim_l, cam, renderer_l, include_ball=False)
    fr_seat_r = _grab(sim_r, cam, renderer_r, include_ball=False)
    seat_pair = pair(
        fr_seat_l,
        fr_seat_r,
        "P64|C SEATED grasp",
        "NOMINAL|C SEATED grasp",
    )
    for _ in range(int(0.7 * FPS)):
        frames.append(seat_pair)

    renderers_closed = False
    trial_seeds = [seed] + [
        rollout_seed(split=SPLIT, traj_id=start.traj_id, rollout_i=i)
        for i in range(CL_SEED_PROBE)
    ]
    trial_seeds = list(dict.fromkeys(int(s) for s in trial_seeds))

    try:
        last_err = None
        for pullback_m in C_BALL_PULLBACK_CANDIDATES_M:
            # Fresh seated install for each pullback candidate.
            pre_l = rebuild_settled_root(sim_l, cfg)
            pre_r = rebuild_settled_root(sim_r, cfg)
            if not (pre_l.get("bilateral") and pre_r.get("bilateral")):
                raise RuntimeError(f"Plan-C seated grasp failed: {pre_l} / {pre_r}")

            # Keep the seated beat; replace only the flight/impact frames.
            frames[:] = frames[: int(0.7 * FPS)]
            launch_info = None
            for sim in (sim_l, sim_r):
                restore_drop(sim, None)
                disarm(sim)
                launch_info = _launch_ball_pulled(sim, cfg, float(pullback_m))

            armed_l = armed_r = False
            impact_t = None
            fork_snap = None
            t0 = float(sim_l.data.time)
            max_s = 2.5

            while float(sim_l.data.time) - t0 < max_s - 1e-9:
                for sim, armed_flag in ((sim_l, "l"), (sim_r, "r")):
                    hit = ball_hits_cylinder(sim)
                    if hit:
                        if armed_flag == "l" and not armed_l:
                            arm(sim)
                            armed_l = True
                        if armed_flag == "r" and not armed_r:
                            arm(sim)
                            armed_r = True
                    step_legal(sim, act)

                now = float(sim_l.data.time)
                hit_now = ball_hits_cylinder(sim_l)
                if fork_snap is None and hit_now:
                    impact_t = now - t0
                    # First-contact IC; fork from THIS live state (no later rewind).
                    fork_snap = make_snap(sim_l)

                tag = "IMPACT" if impact_t is not None else "FLIGHT"
                fr_l = _grab(sim_l, cam, renderer_l)
                fr_r = _grab(sim_r, cam, renderer_r)
                emit_frames(
                    frames,
                    pair(
                        fr_l,
                        fr_r,
                        slow_tag(f"P64|C {tag} t={now - t0:.2f}s", SLOW_FLIGHT_X),
                        slow_tag(f"NOMINAL|C {tag} t={now - t0:.2f}s", SLOW_FLIGHT_X),
                    ),
                    SLOW_FLIGHT_X,
                )

                # Stop at first contact frame — continuing then restore looked like teleport.
                if fork_snap is not None:
                    break

            if fork_snap is None or impact_t is None:
                last_err = {"error": "no_impact", "pullback_m": pullback_m}
                print(f"  [C] pullback={pullback_m} no impact", flush=True)
                continue

            n_shared = len(frames)
            used_seed = None
            # Prefer short search per pullback; expand on the final (0.0) candidate.
            n_try = 24 if float(pullback_m) <= 1e-9 else 8
            for trial in trial_seeds[: min(n_try, len(trial_seeds))]:
                try:
                    probe = _run_fork_p64_vs_drop(
                        actor,
                        start,
                        book,
                        rho,
                        device,
                        int(trial),
                        "C",
                        snap=fork_snap,
                        frames=[],
                    )
                    last_err = probe
                    if int(probe.get("SUCCESS_RELAXED_12S") or 0) == 1 and int(
                        probe.get("right_dropped") or 0
                    ) == 1:
                        used_seed = int(trial)
                        print(
                            f"  [C] probe ok seed={used_seed} pullback={pullback_m}",
                            flush=True,
                        )
                        break
                except Exception as e:
                    last_err = {"error": str(e), "pullback_m": pullback_m}
                    print(f"  [C] probe seed={trial} pullback={pullback_m} err={e}", flush=True)

            if used_seed is None:
                print(f"  [C] pullback={pullback_m} no recoverable seed", flush=True)
                continue

            # Live state == fork_snap; do not restore_ballistic (that teleports the ball).
            restore_drop(sim_l, None)
            restore_drop(sim_r, None)
            arm(sim_l)
            arm(sim_r)

            meta = _run_fork_p64_vs_drop(
                actor,
                start,
                book,
                rho,
                device,
                used_seed,
                "C",
                snap=None,
                frames=frames,
                env_l=env_l,
                env_r=env_r,
                cam=cam,
                renderer_l=renderer_l,
                renderer_r=renderer_r,
                close_renderers=True,
                defer_handoff_eval=True,
            )
            renderers_closed = True
            meta["seed"] = used_seed
            meta["flight"] = {
                "mode": "same_sim_rebuild+pulled_cfg_launch",
                "impact_t": impact_t,
                "n_shared_frames": n_shared,
                "n_total_frames": len(frames),
                "cfg": cfg,
                "launch": launch_info,
                "pullback_m": float(pullback_m),
                "slow_flight_x": int(SLOW_FLIGHT_X),
                "slow_recovery_x": int(SLOW_RECOVERY_X),
                "seed_used": used_seed,
                "seed_requested": seed,
                "pre": {k: pre_l[k] for k in ("bilateral", "obj_z", "hand_z", "topo")},
            }
            return frames, meta

        raise RuntimeError(f"Plan-C fork probe failed: {last_err}")
    finally:
        if not renderers_closed:
            renderer_l.close()
            renderer_r.close()


def render_continuous_d_capture_ball(
    actor, start, book, rho, device, seed: int
) -> tuple[list, dict]:
    """D upright MixedStart capture → milder ball → same-cam P64 vs fixed-hold drop.

    Uses the preferred capture IC (fingers world −Z) so post-handoff 归正 has a
    correct ref orientation; sliding lift is not used for D presentation.
    """
    man, eid, strat = FLIGHT_MANIFEST["D"]
    row = load_row(ROOT / man, eid, strat)
    blob = load_blob(ROOT / row["snapshot"])
    spec = blob["spec"]
    # Strong enough to slip under fixed closed hold; 0.5 was too mild to drop.
    speed_scale = float(SLIDING_IMPACT_SPEED_SCALE.get("D", 1.15))

    env_l = make_eval_env(book, rho)
    env_r = make_eval_env(book, rho)
    env_l.reset(start)
    env_r.reset(start)
    sim_l, sim_r = env_l.env.sim, env_r.env.sim
    cam, renderer_l, renderer_r = _open_renderers(sim_l, sim_r)
    frames: list = []
    renderers_closed = False

    # Short seated beat so upright grasp is readable before the ball appears.
    fr0_l = _grab(sim_l, cam, renderer_l)
    fr0_r = _grab(sim_r, cam, renderer_r)
    seat = pair(fr0_l, fr0_r, "P64|D upright capture", "NOMINAL|D upright capture")
    for _ in range(int(0.5 * FPS)):
        frames.append(seat)

    launch = aim_ball_long(sim_l, spec, speed_scale=speed_scale, flight_s=FLIGHT_AIM_S)
    for sim in (sim_l, sim_r):
        sim.set_guide_weld(False)
        apply_ball_free_state(sim, launch["p_launch"], launch["v0"])

    act = hold_action(1.0)
    next_t = float(sim_l.data.time)
    t0 = float(sim_l.data.time)
    impact_t = None
    fork_snap = None
    sample_flight = max(0.02, SAMPLE_S / 2.0)

    try:
        while float(sim_l.data.time) - t0 < 2.5:
            for sim in (sim_l, sim_r):
                step_legal(sim, act)
            now = float(sim_l.data.time)
            if fork_snap is None and cylinder_contacts(sim_l)[0]:
                impact_t = now - t0
                fork_snap = make_snap(sim_l)

            if now + 1e-9 >= next_t:
                tag = "IMPACT" if impact_t is not None else "FLIGHT"
                fr_l = _grab(sim_l, cam, renderer_l)
                fr_r = _grab(sim_r, cam, renderer_r)
                emit_frames(
                    frames,
                    pair(
                        fr_l,
                        fr_r,
                        slow_tag(f"P64|D {tag} t={now - t0:.2f}s", SLOW_FLIGHT_X),
                        slow_tag(f"NOMINAL|D {tag} t={now - t0:.2f}s", SLOW_FLIGHT_X),
                    ),
                    SLOW_FLIGHT_X,
                )
                next_t += sample_flight

            # Fork at first contact — no bounce-then-restore.
            if fork_snap is not None:
                break

        if fork_snap is None:
            raise RuntimeError("D continuous capture+ball path never recorded impact")

        n_shared = len(frames)
        trial_seeds = [seed] + [
            rollout_seed(split=SPLIT, traj_id=start.traj_id, rollout_i=i)
            for i in range(CL_SEED_PROBE)
        ]
        trial_seeds = list(dict.fromkeys(int(s) for s in trial_seeds))
        used_seed = None
        last_err = None
        for trial in trial_seeds[: min(24, len(trial_seeds))]:
            try:
                probe = _run_fork_p64_vs_drop(
                    actor,
                    start,
                    book,
                    rho,
                    device,
                    int(trial),
                    "D",
                    snap=fork_snap,
                    frames=[],
                )
                last_err = probe
                if int(probe.get("SUCCESS_RELAXED_12S") or 0) == 1 and int(
                    probe.get("right_dropped") or 0
                ) == 1:
                    used_seed = int(trial)
                    print(f"  [D] probe ok seed={used_seed}", flush=True)
                    break
            except Exception as e:
                last_err = {"error": str(e)}
                print(f"  [D] probe seed={trial} err={e}", flush=True)
        if used_seed is None:
            raise RuntimeError(f"D fork probe failed: {last_err}")

        restore_ballistic(sim_l, fork_snap)
        restore_ballistic(sim_r, fork_snap)
        restore_drop(sim_l, None)
        restore_drop(sim_r, None)
        arm(sim_l)
        arm(sim_r)

        meta = _run_fork_p64_vs_drop(
            actor,
            start,
            book,
            rho,
            device,
            used_seed,
            "D",
            snap=None,
            frames=frames,
            env_l=env_l,
            env_r=env_r,
            cam=cam,
            renderer_l=renderer_l,
            renderer_r=renderer_r,
            close_renderers=True,
            defer_handoff_eval=True,
        )
        renderers_closed = True
        meta["flight"] = {
            "mode": "upright_capture+mild_spec_launch",
            "episode": eid,
            "impact_t": impact_t,
            "speed_scale": speed_scale,
            "n_shared_frames": n_shared,
            "n_total_frames": len(frames),
            "flight_aim_s": float(FLIGHT_AIM_S),
            "slow_flight_x": int(SLOW_FLIGHT_X),
            "spec": {
                k: spec[k] for k in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s")
            },
            "seed_used": used_seed,
            "seed_requested": seed,
        }
        meta["seed"] = used_seed
        return frames, meta
    finally:
        if not renderers_closed:
            renderer_l.close()
            renderer_r.close()


def render_continuous_sliding(
    actor, start, book, rho, device, seed: int, skill: str
) -> tuple[list, dict]:
    """Nominal lift → ball → impact → same-camera P64 vs fixed-hold drop.

    Dual live sims + one locked camera pair for the whole clip. Fork never
    closes renderers or teleports into a fresh model (that was the splice).
    """
    from data_generation.datasets.recovery_rebuild import _open_env

    man, eid, strat = FLIGHT_MANIFEST[skill]
    row = load_row(ROOT / man, eid, strat)
    blob = load_blob(ROOT / row["snapshot"])
    spec = blob["spec"]

    raw_l = _open_env()
    raw_r = _open_env()
    raw_l.begin_nominal()
    raw_r.begin_nominal()
    # Mirror so both panels share one physical timeline from frame 0.
    _mirror_sim(raw_r.sim, raw_l.sim)
    st = fresh_nominal_state()
    cam, renderer_l, renderer_r = _open_renderers(raw_l.sim, raw_r.sim)
    frames: list = []
    next_t = None
    launched = False
    impact_t = None
    catch_snap = None
    sample_flight = max(0.02, SAMPLE_S / 2.0)
    t_launch = None
    renderers_closed = False

    try:
        for _ in range(20000):
            if not launched:
                for _ in range(v3.N_SUB):
                    nominal_substep(raw_l.sim, raw_l.gains, st)
                    if st.get("z_tgt") is not None and float(st["z_tgt"]) < 0.55:
                        st["z_tgt"] = 0.55
                        st["holding"] = False
                _mirror_sim(raw_r.sim, raw_l.sim)
            else:
                for _ in range(v3.N_SUB):
                    flight_hold_no_park(raw_l.sim, raw_l.gains, tau=TAU_SEC)
                    hits, _other = cylinder_contacts(raw_l.sim)
                    if hits and impact_t is None:
                        impact_t = float(raw_l.sim.data.time)
                        # First-contact IC; fork from this live state (no rewind).
                        catch_snap = make_snap(raw_l.sim)
                        print(f"  [{skill}] IMPACT at t={impact_t:.2f}s", flush=True)
                        break
                _mirror_sim(raw_r.sim, raw_l.sim)

            now = float(raw_l.sim.data.time)
            pack = physical_pack(raw_l.sim)
            phase = str(raw_l.sim.fsm.phase)
            if phase in ("descend", "lift") and next_t is None:
                next_t = now
            if next_t is not None and now + 1e-9 >= next_t:
                tag = (
                    "IMPACT"
                    if impact_t is not None
                    else ("FLIGHT" if launched else "LIFT")
                )
                fr_l = _grab(raw_l.sim, cam, renderer_l)
                fr_r = _grab(raw_r.sim, cam, renderer_r)
                slow_x = SLOW_FLIGHT_X if launched else 1
                emit_frames(
                    frames,
                    pair(
                        fr_l,
                        fr_r,
                        slow_tag(f"P64|{skill} {tag} t={now:.2f}s", slow_x),
                        slow_tag(f"NOMINAL|{skill} {tag} t={now:.2f}s", slow_x),
                    ),
                    slow_x,
                )
                next_t += sample_flight if launched else SAMPLE_S

            if (
                (not launched)
                and bool(raw_l.sim.captured)
                and phase == "lift"
                and st.get("hand_z0") is not None
                and float(pack["obj_z"]) >= 0.43
            ):
                hz = float(raw_l.sim.data.xpos[raw_l.sim.ids.hand_body][2])
                if (hz - float(st["hand_z0"])) >= HAND_RISE_M:
                    speed_scale = float(SLIDING_IMPACT_SPEED_SCALE.get(skill, 1.0))
                    flight_s = float(SLIDING_FLIGHT_AIM_S.get(skill, 0.10))
                    if flight_s > 0.12:
                        launch = aim_ball_long(
                            raw_l.sim,
                            spec,
                            speed_scale=speed_scale,
                            flight_s=flight_s,
                        )
                    else:
                        launch = aim_ball(
                            raw_l.sim,
                            spec["azimuth_deg"],
                            float(spec["speed_mps"]) * speed_scale,
                            spec["z_off_m"],
                            spec["phi_deg"],
                        )
                    raw_l.sim.set_guide_weld(False)
                    apply_ball_free_state(raw_l.sim, launch["p_launch"], launch["v0"])
                    _mirror_sim(raw_r.sim, raw_l.sim)
                    launched = True
                    t_launch = now
                    next_t = now
                    print(
                        f"  [{skill}] ball launched at t={now:.2f}s obj_z={pack['obj_z']:.3f} "
                        f"speed_scale={speed_scale} flight_s={flight_s:.2f}",
                        flush=True,
                    )

            # Fork at the contact frame — do not keep stepping (ball flies away) then restore.
            if catch_snap is not None:
                break
            if launched and t_launch is not None and now > float(t_launch) + 2.5:
                break
            if (not launched) and now > 16.0:
                break

        if catch_snap is None:
            raise RuntimeError(
                f"{skill} continuous path never recorded ball impact (launched={launched})"
            )

        n_shared = len(frames)
        trial_seeds = [seed] + [
            rollout_seed(split=SPLIT, traj_id=start.traj_id, rollout_i=i)
            for i in range(CL_SEED_PROBE)
        ]
        trial_seeds = list(dict.fromkeys(int(s) for s in trial_seeds))

        # Probe on the SAME live sims as the final clip (eval-env probe lied for D).
        used_seed = None
        last_err = None
        for trial in trial_seeds[: min(24, len(trial_seeds))]:
            restore_ballistic(raw_l.sim, catch_snap)
            _mirror_sim(raw_r.sim, raw_l.sim)
            restore_drop(raw_l.sim, None)
            restore_drop(raw_r.sim, None)
            arm(raw_l.sim)
            arm(raw_r.sim)
            env_l = _bind_live_policy_env(book, rho, raw_l.sim)
            env_r = _bind_live_policy_env(book, rho, raw_r.sim)
            try:
                probe = _run_fork_p64_vs_drop(
                    actor,
                    start,
                    book,
                    rho,
                    device,
                    int(trial),
                    skill,
                    snap=None,
                    frames=[],
                    env_l=env_l,
                    env_r=env_r,
                    cam=cam,
                    renderer_l=renderer_l,
                    renderer_r=renderer_r,
                    close_renderers=False,
                )
                last_err = probe
                if int(probe.get("SUCCESS_RELAXED_12S") or 0) == 1 and int(
                    probe.get("right_dropped") or 0
                ) == 1:
                    used_seed = int(trial)
                    print(f"  [{skill}] probe ok seed={used_seed}", flush=True)
                    break
            except Exception as e:
                last_err = {"error": str(e)}
                print(f"  [{skill}] fork seed={trial} err={e}", flush=True)
        if used_seed is None:
            raise RuntimeError(f"{skill} fork probe failed: {last_err}")

        # Identity restore after probe mutated the live sims — still at catch contact,
        # so the ball does not jump from a bounced-away pose.
        restore_ballistic(raw_l.sim, catch_snap)
        _mirror_sim(raw_r.sim, raw_l.sim)
        restore_drop(raw_l.sim, None)
        restore_drop(raw_r.sim, None)
        arm(raw_l.sim)
        arm(raw_r.sim)
        env_l = _bind_live_policy_env(book, rho, raw_l.sim)
        env_r = _bind_live_policy_env(book, rho, raw_r.sim)

        meta = _run_fork_p64_vs_drop(
            actor,
            start,
            book,
            rho,
            device,
            used_seed,
            skill,
            snap=None,
            frames=frames,
            env_l=env_l,
            env_r=env_r,
            cam=cam,
            renderer_l=renderer_l,
            renderer_r=renderer_r,
            close_renderers=True,
            defer_handoff_eval=True,
        )
        renderers_closed = True
        meta["flight"] = {
            "mode": "same_cam_dual_sim_lift+aim",
            "episode": eid,
            "impact_t": impact_t,
            "launched": launched,
            "n_shared_frames": n_shared,
            "n_total_frames": len(frames),
            "flight_aim_s": float(SLIDING_FLIGHT_AIM_S.get(skill, 0.10)),
            "slow_flight_x": int(SLOW_FLIGHT_X),
            "slow_recovery_x": int(SLOW_RECOVERY_X),
            "spec": {
                k: spec[k] for k in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s")
            },
            "seed_used": used_seed,
            "seed_requested": seed,
        }
        meta["seed"] = used_seed
        return frames, meta
    finally:
        if not renderers_closed:
            renderer_l.close()
            renderer_r.close()


def save_poster(frames: list, path: Path):
    idx = min(len(frames) - 1, max(0, int(0.55 * len(frames))))
    img = frames[idx]
    try:
        from PIL import Image

        Image.fromarray(img).save(path)
    except Exception:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), img)


def main():
    OUT_VID.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    assert sha16(P64_ACTOR) == EXPECTED_SHA, sha16(P64_ACTOR)
    print("[P64-VID] counts…", flush=True)
    counts = verify_counts()
    print(counts["correction"], flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device)
    starts = load_heldout_starts(per_class=4)

    print("[P64-VID] enumerate successes…", flush=True)
    enum_path = OUT_VID / "SUCCESS_ENUMERATION.csv"
    import csv

    if enum_path.is_file() and enum_path.stat().st_size > 50:
        rows = list(csv.DictReader(enum_path.open(encoding="utf-8")))
        print(f"  resumed {len(rows)} success rows", flush=True)
    else:
        rows = enumerate_successes(actor, book, rho, device, starts)

    # If preferred IC set changed (e.g. upright D), refresh missing skills.
    have = {(r["skill"], r["traj_id"]) for r in rows}
    missing = [sk for sk, tid in PREFERRED.items() if (sk, tid) not in have]
    if missing:
        print(f"  re-probe missing preferred ICs: {missing}", flush=True)
        extra = enumerate_successes(actor, book, rho, device, starts)
        # enumerate_successes writes full PREFERRED; prefer its rows
        rows = extra

    print("[P64-VID] select representatives…", flush=True)
    selected = select_representatives(rows)
    validation = []

    for sel in selected:
        skill = sel["skill"]
        tid = sel["traj_id"]
        seed = int(sel["seed"])
        start = next(s for s in starts[skill] if s.traj_id == tid)
        print(f"[P64-VID] render CONTINUOUS {skill} {tid} seed={seed}", flush=True)
        if skill == "C":
            frames, meta = render_continuous_plan_c(actor, start, book, rho, device, seed)
        else:
            # A1/B1/D: continuous lift path. D MixedStart ref is still upright, so
            # visible handoff return shows 归正; capture+mild-ball path is too fragile.
            frames, meta = render_continuous_sliding(
                actor, start, book, rho, device, seed, skill
            )
        if int(meta["SUCCESS_RELAXED_12S"]) != 1:
            raise RuntimeError(f"continuous reproduction failed for {skill}: {meta}")
        if int(meta.get("right_dropped") or 0) != 1:
            raise RuntimeError(f"do-nothing side did not drop for {skill}: {meta}")

        target = int(14 * FPS)
        if len(frames) < target and frames:
            last = frames[-1]
            for _ in range(target - len(frames)):
                frames.append(last)
        mp4 = OUT_VID / f"P64_{skill}_success.mp4"
        wrote = write_frames_mp4(frames, mp4, fps=FPS)
        if not wrote:
            raise RuntimeError(f"ffmpeg encode failed for {mp4}")
        try:
            import imageio_ffmpeg
            import subprocess

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            tmp = mp4.with_suffix(".fast.mp4")
            r = subprocess.run(
                [exe, "-y", "-i", str(mp4), "-c", "copy", "-movflags", "+faststart", str(tmp)],
                capture_output=True,
            )
            if r.returncode == 0 and tmp.is_file():
                tmp.replace(mp4)
        except Exception:
            pass

        poster = OUT_VID / f"P64_{skill}_poster.png"
        save_poster(frames, poster)
        sh = sel.get("snapshot_hash") or ""
        validation.append(
            {
                "skill": skill,
                "traj_id": tid,
                "snapshot_hash": sh,
                "checkpoint": str(P64_ACTOR),
                "sha16": EXPECTED_SHA,
                "seed": int(meta.get("seed") or seed),
                "seed_requested": seed,
                "original_SUCCESS_RELAXED_12S": 1,
                "reproduced_SUCCESS_RELAXED_12S": meta["SUCCESS_RELAXED_12S"],
                "right_dropped": meta.get("right_dropped"),
                "continuous": True,
                "n_steps": meta["n_steps"],
                "recatch": meta.get("recatch"),
                "behavior_label": sel.get("behavior_label"),
                "selection_rule": sel.get("selection_rule"),
                "video_frames": len(frames),
                "video_duration_s": len(frames) / FPS,
                "sim_time_s": meta.get("sim_time"),
                "flight": meta.get("flight"),
                "mp4": str(mp4),
                "poster": str(poster),
                "command": "python experiments/precision/p64_presentation_videos/render_p64_family_videos.py",
            }
        )
        print(
            f"  WROTE {mp4} frames={len(frames)} ok={meta['SUCCESS_RELAXED_12S']} "
            f"drop={meta.get('right_dropped')}",
            flush=True,
        )

    (OUT_VID / "VIDEO_VALIDATION.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    md = ["# VIDEO_VALIDATION — P64 presentation clips", ""]
    md.append(f"- Checkpoint: `{P64_ACTOR}` SHA16=`{EXPECTED_SHA}`")
    md.append("- Official criterion: SUCCESS_RELAXED_12S (left) + do-nothing drop (right)")
    md.append("- Continuous dual-sim: seated/lift → ball flight → impact on BOTH panels, then fork")
    md.append(f"- Eval protocol seeds: split=`{SPLIT}`, temperature=1.0 (matches P64_step4)")
    md.append("")
    md.append("## Count correction")
    md.append(counts["correction"])
    md.append("")
    md.append("## Per-clip")
    for v in validation:
        md.append(f"### {v['skill']}")
        for k, val in v.items():
            if k == "flight":
                md.append(f"- flight: `{json.dumps(val)}`")
            else:
                md.append(f"- {k}: `{val}`")
        md.append("")
    md.append("## Dependencies")
    md.append("- mujoco Renderer, imageio / imageio-ffmpeg (libx264 yuv420p), numpy, torch, PIL")
    md.append("- No audio. Resolution 1280×720 @ 24fps (two 640×480 panels upscaled).")
    (OUT_VID / "VIDEO_VALIDATION.md").write_text("\n".join(md), encoding="utf-8")
    print("[P64-VID] DONE", flush=True)


if __name__ == "__main__":
    main()
