"""Phase 0 — Validate pre-contact e_pos measurement protocol from P68."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from evaluation.success_criteria import P_TGT, U_TGT, WORKING_POS_MM
from evaluation.grasp_geometry import measure
from experiments.precision.p37_working_handoff_d.handoff_rules import working_geometry
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p69_skill_sequence.paths import (
    MAX_EP_STEPS,
    N_ICS_AUDIT,
    N_PER_IC_AUDIT,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    T_MAX,
)


def _train_c(n: int) -> list:
    blocked = blocked_eval_ids()
    return [
        s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    ][:n]


def e_pos_definitions(sim) -> dict[str, float]:
    """Compare measurement paths used by P68 / R3F."""
    wk = working_geometry(sim)  # P68 path
    m = measure(sim, U_TGT, P_TGT)  # R3 / SUCCESS path
    return {
        "working_geometry_e_pos_mm": float(wk["e_pos_mm"]),
        "measure_e_pos_mm": float(m["e_pos_mm"]),
        "working_theta_deg": float(wk["dtheta_deg"]),
        "measure_theta_deg": float(m["e_theta_deg"]),
        "abs_diff_e_pos": abs(float(wk["e_pos_mm"]) - float(m["e_pos_mm"])),
    }


def rollout_audit(env, actor, critic, start, device, *, seed: int) -> dict:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    def_t0 = e_pos_definitions(env.env.sim)
    geo0 = relative_geometry(env.env.sim)

    last_no_contact = None  # stage-matched pre-contact
    first_contact_t = None
    first_contact_geo = None
    rows_steps = []
    done = False
    info: dict[str, Any] = {}
    r3 = False

    while not done:
        t = int(env.t)
        geo = relative_geometry(env.env.sim)
        defs = e_pos_definitions(env.env.sim)
        if not geo["has_pad_contact"]:
            last_no_contact = {
                "t": t,
                "e_pos_mm": defs["working_geometry_e_pos_mm"],
                "measure_e_pos_mm": defs["measure_e_pos_mm"],
                "theta_deg": defs["working_theta_deg"],
                "remaining": int(env.max_steps - t),
                "rel_dist_m": geo["rel_dist_m"],
            }
        tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        r, done, info = env.step(tok, resid)
        geo_a = relative_geometry(env.env.sim)
        if first_contact_t is None and geo_a["has_pad_contact"]:
            first_contact_t = int(env.t)
            first_contact_geo = {
                "t": first_contact_t,
                "e_pos_mm": float(geo_a["e_pos_mm"]),
                "theta_deg": float(geo_a["theta_deg"]),
                "remaining": int(env.max_steps - env.t),
                "correct_sidewall": geo_a["correct_sidewall"],
            }
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if preds["dwell_complete"] and preds["frame_ok"]:
            r3 = True
        rows_steps.append({"t": t, "defs": defs, "has_contact": geo["has_pad_contact"]})

    succ = bool(int(info.get("SUCCESS_RELAXED_12S") or 0) or r3 or int(info.get("handoff") or 0))
    return {
        "traj_id": start.traj_id,
        "seed": seed,
        "success_or_r3": int(succ),
        "r3_ready": int(r3 or int(info.get("handoff") or 0)),
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "def_t0": def_t0,
        "geo0_e_pos": geo0["e_pos_mm"],
        "pre_contact_matched": last_no_contact,  # last no-contact frame (stage-matched)
        "first_contact": first_contact_geo,
        "first_contact_t": first_contact_t,
        "max_abs_def_diff": float(max(r["defs"]["abs_diff_e_pos"] for r in rows_steps)),
    }


def run_phase0(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    starts = _train_c(N_ICS_AUDIT)
    episodes = []
    for st in starts:
        for i in range(N_PER_IC_AUDIT):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=90000 + i)
            ep = rollout_audit(env, actor, critic, st, device_t, seed=seed)
            episodes.append(ep)
            pc = ep["pre_contact_matched"]
            print(
                f"[p0] {st.traj_id[-28:]} i={i} succ={ep['success_or_r3']} "
                f"pre_e={None if not pc else round(pc['e_pos_mm'],1)} "
                f"fc_t={ep['first_contact_t']} rem={None if not pc else pc['remaining']}",
                flush=True,
            )

    # Protocol audit notes
    protocol = {
        "e_pos_definition": (
            "Hand-frame Euclidean position error of object COM target vs grasp target P_TGT "
            "via evaluation.grasp_geometry.measure / working_geometry (same path as R3 margin)."
        ),
        "coordinate_frame": "Hand frame; P_TGT/U_TGT from evaluation.success_criteria",
        "WORKING_POS_MM": float(WORKING_POS_MM),
        "R3_MARGIN_POS_MM": 9.0,
        "threshold_25mm_status": "descriptive only — NOT a success criterion",
        "stage_matched_pre_contact": "last policy step with no pad contact before first contact",
        "future_leakage": "first_contact detection uses only current contacts; intervention timing from no-contact flag",
        "working_vs_measure_max_abs_diff_mm": float(max(e["max_abs_def_diff"] for e in episodes)),
    }

    by_ic = defaultdict(lambda: {"succ_pre": [], "fail_pre": [], "succ_t0": [], "fail_t0": [], "n": 0})
    for e in episodes:
        ic = e["traj_id"]
        by_ic[ic]["n"] += 1
        pc = e["pre_contact_matched"]
        epos = float(pc["e_pos_mm"]) if pc else float(e["geo0_e_pos"])
        rem = int(pc["remaining"]) if pc else MAX_EP_STEPS
        bucket = "succ_pre" if e["success_or_r3"] else "fail_pre"
        by_ic[ic][bucket].append({"e_pos_mm": epos, "remaining": rem, "t": None if not pc else pc["t"]})
        by_ic[ic]["succ_t0" if e["success_or_r3"] else "fail_t0"].append(float(e["geo0_e_pos"]))

    per_ic = []
    for tid, d in by_ic.items():
        def summ(xs):
            if not xs:
                return None
            a = np.asarray([x["e_pos_mm"] if isinstance(x, dict) else x for x in xs], float)
            return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}
        rem_s = [x["remaining"] for x in d["succ_pre"]]
        rem_f = [x["remaining"] for x in d["fail_pre"]]
        per_ic.append(
            {
                "traj_id": tid,
                "n": d["n"],
                "succ_pre_contact": summ(d["succ_pre"]),
                "fail_pre_contact": summ(d["fail_pre"]),
                "succ_t0": summ(d["succ_t0"]),
                "fail_t0": summ(d["fail_t0"]),
                "mean_remaining_succ": float(np.mean(rem_s)) if rem_s else None,
                "mean_remaining_fail": float(np.mean(rem_f)) if rem_f else None,
            }
        )

    succ_e = [
        float(e["pre_contact_matched"]["e_pos_mm"])
        for e in episodes
        if e["success_or_r3"] and e["pre_contact_matched"]
    ]
    fail_e = [
        float(e["pre_contact_matched"]["e_pos_mm"])
        for e in episodes
        if (not e["success_or_r3"]) and e["pre_contact_matched"]
    ]
    # Stage timing: contact time success vs fail
    fc_s = [e["first_contact_t"] for e in episodes if e["success_or_r3"] and e["first_contact_t"] is not None]
    fc_f = [e["first_contact_t"] for e in episodes if (not e["success_or_r3"]) and e["first_contact_t"] is not None]

    pooled = {
        "succ_pre_e_pos": {
            "n": len(succ_e),
            "mean": float(np.mean(succ_e)) if succ_e else None,
            "median": float(np.median(succ_e)) if succ_e else None,
        },
        "fail_pre_e_pos": {
            "n": len(fail_e),
            "mean": float(np.mean(fail_e)) if fail_e else None,
            "median": float(np.median(fail_e)) if fail_e else None,
        },
        "first_contact_t_succ_mean": float(np.mean(fc_s)) if fc_s else None,
        "first_contact_t_fail_mean": float(np.mean(fc_f)) if fc_f else None,
    }

    # Descriptive threshold check (not criterion)
    thr = 25.0
    low = [int(e["r3_ready"]) for e in episodes if e["pre_contact_matched"] and e["pre_contact_matched"]["e_pos_mm"] <= thr]
    high = [int(e["r3_ready"]) for e in episodes if e["pre_contact_matched"] and e["pre_contact_matched"]["e_pos_mm"] > thr]
    thr_scan = {
        "thr_mm": thr,
        "p_r3_low": float(np.mean(low)) if low else None,
        "p_r3_high": float(np.mean(high)) if high else None,
        "n_low": len(low),
        "n_high": len(high),
        "note": "descriptive association only",
    }

    # Interpretation: is gap due to timing?
    timing_confound = False
    if pooled["first_contact_t_succ_mean"] is not None and pooled["first_contact_t_fail_mean"] is not None:
        # If fail contacts much earlier while still far — still a real gap
        if abs(pooled["first_contact_t_succ_mean"] - pooled["first_contact_t_fail_mean"]) > 5:
            timing_confound = True

    report = {
        "protocol": protocol,
        "pooled": pooled,
        "per_ic": per_ic,
        "threshold_scan_25mm": thr_scan,
        "timing_confound_flag": timing_confound,
        "n_episodes": len(episodes),
        "interpretation": (
            "Stage-matched pre-contact = last no-pad-contact frame. "
            "If succ/fail means remain widely separated after matching, the P68 gap is robust. "
            "25 mm threshold is descriptive only."
        ),
    }
    (OUT / "PRECONTACT_GEOMETRY_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# Pre-Contact Geometry Audit (P69 Phase 0)\n"]
    md.append("## Protocol\n")
    for k, v in protocol.items():
        md.append(f"- **{k}**: {v}")
    md.append("\n## Pooled stage-matched pre-contact e_pos\n")
    md.append(f"- Success/R3: `{pooled['succ_pre_e_pos']}`")
    md.append(f"- Failure: `{pooled['fail_pre_e_pos']}`")
    md.append(f"- First-contact step mean succ/fail: {pooled['first_contact_t_succ_mean']} / {pooled['first_contact_t_fail_mean']}")
    md.append(f"- Timing confound flag: **{timing_confound}**")
    md.append(f"\n## Descriptive 25 mm association\n`{thr_scan}`\n")
    md.append("## Per-IC distributions\n")
    for row in per_ic:
        md.append(
            f"- `{row['traj_id'][-40:]}`: fail_pre={row['fail_pre_contact']}, "
            f"succ_pre={row['succ_pre_contact']}, rem_fail={row['mean_remaining_fail']}, rem_succ={row['mean_remaining_succ']}"
        )
    md.append(f"\n_{report['interpretation']}_\n")
    (OUT / "PRECONTACT_GEOMETRY_AUDIT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:40]), flush=True)
    return report


if __name__ == "__main__":
    run_phase0()
