"""Phase 1 — Validate C sidewall / stage diagnostic labels on Expert + policy trajectories."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np
import torch

from data_generation.oracles.plan_c.contact_surface import classify_contacts
from envs.simulation.ground_scene import CYL_HALF, CYL_R
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import is_correct_sidewall
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import expert_actions, step_raw
from experiments.precision.p66_c_curriculum_ppo.paths import (
    GEOM_POLICY_ROLLOUTS,
    MAX_EP_STEPS,
    OUT,
    P64_STEP4_ACTOR,
    SEED,
    T_MAX,
)


def _r3f_ready(env: StabilizeGrpoEnv) -> bool:
    if env.mon is None or env.mon.tracker is None:
        return False
    t = env.mon.tracker
    return bool(t._active and float(t.ok_s) + 1e-12 >= float(t.dwell_s))


def _audit_sanity(sim, audit: dict) -> dict[str, Any]:
    """Check frames, cylinder axis, normals, opposing geometry for one audit."""
    issues = []
    ok_points = 0
    for c in audit.get("contacts") or []:
        rho = float(c.get("rho") or 0)
        abs_z = float(c.get("abs_z") or 0)
        surf = str(c.get("surface") or "")
        # Sidewall contacts should be near cylinder radius and not on end faces
        if surf == "sidewall":
            if abs(rho - CYL_R) > 0.01:
                issues.append(f"sidewall_rho_far:{rho:.4f}")
            if abs_z > CYL_HALF + 0.005:
                issues.append(f"sidewall_z_past_half:{abs_z:.4f}")
            ok_points += 1
        if surf == "end_face":
            if abs_z < CYL_HALF - 0.01:
                issues.append(f"face_z_low:{abs_z:.4f}")
    # Cylinder body orientation exists
    try:
        Ro = np.asarray(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
        axis = Ro[:, 2]
        if abs(float(np.linalg.norm(axis)) - 1.0) > 1e-3:
            issues.append("cyl_axis_not_unit")
    except Exception as e:  # noqa: BLE001
        issues.append(f"axis_err:{e}")

    oppose = int(audit.get("opposing_sidewall") or 0)
    nd = audit.get("normal_dot_LR")
    lat = audit.get("lateral_sep_dot")
    if oppose:
        if nd is None or float(nd) >= -0.2:
            issues.append(f"oppose_but_ndot={nd}")
        if lat is None or float(lat) >= 0.0:
            issues.append(f"oppose_but_lat={lat}")
        if not audit.get("bilateral"):
            issues.append("oppose_without_bilateral")
        if str(audit.get("grasp_kind")) != "opposing_sidewall":
            issues.append(f"oppose_kind={audit.get('grasp_kind')}")

    return {
        "n_sidewall_pts_checked": ok_points,
        "n_issues": len(issues),
        "issues": issues[:12],
        "grasp_kind": audit.get("grasp_kind"),
        "opposing_sidewall": oppose,
        "bilateral": int(bool(audit.get("bilateral"))),
        "is_correct_sidewall": int(is_correct_sidewall(audit)),
        "normal_dot_LR": None if nd is None else float(nd),
        "lateral_sep_dot": None if lat is None else float(lat),
        "n_end_face": int(audit.get("n_end_face") or 0),
        "maj_L": audit.get("maj_L"),
        "maj_R": audit.get("maj_R"),
    }


def _train_c_starts() -> list:
    blocked = blocked_eval_ids()
    ds = build_p54_dataset(seed=SEED)
    return [s for s in ds["schedule"] if s.skill == "C" and s.traj_id not in blocked]


def replay_expert_geometry(*, n_experts: int = 24) -> dict:
    """Replay successful C Expert trajectories; audit sidewall classifier at first contact."""
    starts = _train_c_starts()[:n_experts]
    device = torch.device("cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=max(MAX_EP_STEPS, 500), t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    del actor

    rows = []
    agree_sw_success = 0
    n_sw = 0
    n_terminal_succ = 0
    sanity_fail = 0

    for st in starts:
        ep = st.ep or {}
        acts = expert_actions(ep)
        # Extend horizon so Expert can finish
        env.max_steps = int(len(acts) + 5)
        env.reset(st)
        first_sw = None
        first_sw_audit = None
        first_bil = None
        first_stable = None
        first_r3f = None
        ever_sw = False
        handoff = False
        info: dict = {}
        done = False
        for i, a in enumerate(acts):
            if done:
                break
            audit = classify_contacts(env.env.sim)
            # snapshot BEFORE action is the state at which Actor would act;
            # classify after previous step — for first-contact we check after step
            _r, done, info = step_raw(env, a)
            audit = classify_contacts(env.env.sim)
            if first_bil is None and audit.get("bilateral"):
                first_bil = int(env.t)
            if is_correct_sidewall(audit):
                ever_sw = True
                if first_sw is None:
                    first_sw = int(env.t)
                    first_sw_audit = _audit_sanity(env.env.sim, audit)
            if env.diag.get("stable_hold") and first_stable is None:
                first_stable = int(env.t)
            if _r3f_ready(env) and first_r3f is None:
                first_r3f = int(env.t)
            if info.get("handoff"):
                handoff = True

        # If Expert actions exhausted without done, force evaluate remaining as truncated
        succ = int((info or {}).get("SUCCESS_RELAXED_12S") or 0)
        # Many experts terminate via handoff inside step_raw; if not, check SUCCESS via elev path
        if not done and int(ep.get("SUCCESS_NEW") or 0):
            # Expert pack claims success; count as expert-labeled success even if truncated horizon
            succ = 1

        if ever_sw:
            n_sw += 1
            if succ:
                agree_sw_success += 1
        if succ:
            n_terminal_succ += 1
        if first_sw_audit and first_sw_audit["n_issues"] > 0:
            sanity_fail += 1

        rows.append(
            {
                "traj_id": st.traj_id,
                "n_actions": int(len(acts)),
                "first_sidewall_step": first_sw,
                "first_bilateral_step": first_bil,
                "first_stable_step": first_stable,
                "first_r3f_ready_step": first_r3f,
                "ever_correct_sidewall": int(ever_sw),
                "handoff": int(handoff or int((info or {}).get("handoff") or 0)),
                "SUCCESS_RELAXED_12S": int(succ),
                "expert_SUCCESS_NEW": int(ep.get("SUCCESS_NEW") or 0),
                "opposing_sidewall_meta": int(ep.get("opposing_sidewall") or 0),
                "first_sw_sanity": first_sw_audit,
            }
        )
        print(
            f"[geom-expert] {st.traj_id[-36:]} sw={first_sw} bil={first_bil} "
            f"stab={first_stable} r3f={first_r3f} hand={handoff} succ={succ}",
            flush=True,
        )

    report = {
        "n_experts": len(rows),
        "n_ever_sidewall": n_sw,
        "n_terminal_succ_relaxed_or_label": n_terminal_succ,
        "p_sidewall_among_experts": n_sw / max(1, len(rows)),
        "p_succ_given_sidewall": agree_sw_success / max(1, n_sw),
        "n_first_sw_sanity_issues": sanity_fail,
        "cyl_r": float(CYL_R),
        "cyl_half": float(CYL_HALF),
        "definitions": {
            "correct_sidewall": "grasp_kind==opposing_sidewall AND opposing_sidewall AND bilateral AND n_end_face==0",
            "stable_recatch": "RecatchTracker.stable_hold (bilateral min_pad + low vrel hold)",
            "r3f_ready": "R3FOnline tracker ok_s >= dwell_s",
            "handoff": "env.mon.observe ready → info.handoff",
            "SUCCESS_RELAXED": "evaluate_success_relaxed_12s after handoff",
        },
        "rows": rows,
    }
    return report


def replay_policy_geometry(*, n_ics: int = 8, n_per: int = GEOM_POLICY_ROLLOUTS) -> dict:
    """Replay P64_step4 on train C ICs; compare sidewall vs bilateral vs success."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device)
    # critic not required for geometry
    from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_critic
    from experiments.precision.p66_c_curriculum_ppo.paths import P64_STEP4_CRITIC

    critic = load_critic(P64_STEP4_CRITIC, actor, device)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    starts = _train_c_starts()[:n_ics]
    rows = []
    counts = Counter()
    for st in starts:
        for i in range(n_per):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=1000 + i)
            seed_all(seed, deterministic_torch=True)
            env.reset(st)
            ever_sw = ever_bil = False
            first_sw = first_bil = None
            max_bil_run = cur = 0
            done = False
            info: dict = {}
            while not done:
                tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
                r, done, info = env.step(tok, resid)
                audit = classify_contacts(env.env.sim)
                if audit.get("bilateral"):
                    ever_bil = True
                    cur += 1
                    max_bil_run = max(max_bil_run, cur)
                    if first_bil is None:
                        first_bil = int(env.t)
                else:
                    cur = 0
                if is_correct_sidewall(audit):
                    ever_sw = True
                    if first_sw is None:
                        first_sw = int(env.t)
            cd = info.get("c_diag") or {}
            row = {
                "traj_id": st.traj_id,
                "ever_bilateral": int(ever_bil),
                "ever_correct_sidewall": int(ever_sw),
                "first_sw": first_sw,
                "first_bil": first_bil,
                "max_bil_run": max_bil_run,
                "recatch": int(cd.get("recatch") or 0),
                "stable_hold": int(cd.get("stable_hold") or 0),
                "handoff": int(info.get("handoff") or 0),
                "SUCCESS_RELAXED_12S": int(info.get("SUCCESS_RELAXED_12S") or 0),
                "false_bilateral_without_sidewall": int(ever_bil and not ever_sw),
            }
            rows.append(row)
            for k in (
                "ever_bilateral",
                "ever_correct_sidewall",
                "recatch",
                "stable_hold",
                "handoff",
                "SUCCESS_RELAXED_12S",
                "false_bilateral_without_sidewall",
            ):
                counts[k] += row[k]
            print(
                f"[geom-pol] {st.traj_id[-28:]} i={i} bil={ever_bil} sw={ever_sw} "
                f"rec={row['recatch']} stab={row['stable_hold']} hand={row['handoff']} succ={row['SUCCESS_RELAXED_12S']}",
                flush=True,
            )

    n = max(1, len(rows))
    rates = {k: counts[k] / n for k in counts}
    # Key diagnostic: previously reported ~15.6% sidewall vs ~81% bilateral
    report = {
        "policy": str(P64_STEP4_ACTOR),
        "n_episodes": len(rows),
        "n_ics": len(starts),
        "rollouts_per_ic": n_per,
        "rates": rates,
        "counts": dict(counts),
        "sidewall_rate_pct": 100.0 * rates.get("ever_correct_sidewall", 0.0),
        "bilateral_rate_pct": 100.0 * rates.get("ever_bilateral", 0.0),
        "note": (
            "Sidewall rate is fraction of episodes that EVER achieve opposing_sidewall "
            "correct grasp; bilateral alone is NOT correct C geometry."
        ),
        "rows": rows,
    }
    return report


def run_geometry_validation() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== P66 Phase 1a: Expert geometry replay ===", flush=True)
    expert = replay_expert_geometry(n_experts=24)
    print("=== P66 Phase 1b: Policy geometry replay ===", flush=True)
    policy = replay_policy_geometry(n_ics=8, n_per=GEOM_POLICY_ROLLOUTS)
    out = {"expert": expert, "policy": policy}
    (OUT / "C_GEOMETRY_RAW.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Write markdown summary
    md = []
    md.append("# C Geometry Validation (P66 Phase 1)\n")
    md.append("## Definitions (verified in code)\n")
    for k, v in expert["definitions"].items():
        md.append(f"- **{k}**: `{v}`")
    md.append("\n## Cylinder frame\n")
    md.append(f"- `CYL_R={expert['cyl_r']}`, `CYL_HALF={expert['cyl_half']}`")
    md.append("- Contact points mapped via `Ro.T @ (pos_w - po)`; normals likewise.")
    md.append(
        "- Opposing sidewall requires maj_L/R==sidewall, normal_dot_LR < -0.2, lateral_sep_dot < 0, n_end_face==0."
    )
    md.append("\n## Expert successful C trajectories (train ICs)\n")
    md.append(f"- N={expert['n_experts']}, ever correct sidewall={expert['n_ever_sidewall']} "
              f"({100*expert['p_sidewall_among_experts']:.1f}%)")
    md.append(f"- P(SUCCESS | sidewall) among these Experts ≈ {expert['p_succ_given_sidewall']:.3f} "
              f"(denominator={expert['n_ever_sidewall']})")
    md.append(f"- First-sidewall sanity issues: {expert['n_first_sw_sanity_issues']}")
    # timing
    sw_steps = [r["first_sidewall_step"] for r in expert["rows"] if r["first_sidewall_step"] is not None]
    if sw_steps:
        md.append(
            f"- First sidewall step: median={float(np.median(sw_steps)):.0f}, "
            f"p10={float(np.percentile(sw_steps,10)):.0f}, p90={float(np.percentile(sw_steps,90)):.0f}"
        )
    md.append("\n## P64_step4 policy on train C (geometry rates)\n")
    md.append(f"- Episodes: {policy['n_episodes']} ({policy['n_ics']} ICs × {policy['rollouts_per_ic']})")
    md.append(f"- Ever bilateral: **{policy['bilateral_rate_pct']:.1f}%**")
    md.append(f"- Ever correct sidewall: **{policy['sidewall_rate_pct']:.1f}%**")
    md.append(f"- False bilateral without sidewall: {100*policy['rates'].get('false_bilateral_without_sidewall',0):.1f}%")
    md.append(f"- Recatch (tracker): {100*policy['rates'].get('recatch',0):.1f}%")
    md.append(f"- Stable hold: {100*policy['rates'].get('stable_hold',0):.1f}%")
    md.append(f"- HANDOFF: {100*policy['rates'].get('handoff',0):.1f}%")
    md.append(f"- SUCCESS_RELAXED: {100*policy['rates'].get('SUCCESS_RELAXED_12S',0):.1f}%")
    md.append(
        "\n**Verdict on prior 15.6% claim:** "
        f"Measured sidewall rate under this protocol is **{policy['sidewall_rate_pct']:.1f}%**. "
        "Do not equate bilateral contact with correct C geometry.\n"
    )
    (OUT / "C_GEOMETRY_VALIDATION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print((OUT / "C_GEOMETRY_VALIDATION.md").read_text(encoding="utf-8")[:1200], flush=True)
    return out


if __name__ == "__main__":
    run_geometry_validation()
