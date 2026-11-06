"""Phase 4–5 — sensing feasibility catalog + diagnostic upper-bound collection & info gain."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs
from experiments.precision.p71_contact_safe_alignment.controllers import CtrlState, make_specs, select_token
from experiments.precision.p77_contact_observability.metrics import eval_binary_probe
from experiments.precision.p77_contact_observability.paths import (
    CACHE,
    FC_WINDOWS,
    N_SENSE_EP,
    OUT,
    P64_ACTOR,
    SEED,
    VAL_FRAC,
)


def sensing_catalog() -> dict:
    """Classify modalities A/B/C/D without claiming D is deployable."""
    return {
        "1_fingertip_tactile_contact_location": {
            "class": "A",
            "franka": "Requires tactile skin / DigiTac / similar — not stock Panda F/T.",
            "mujoco": "Pad-frame CoP (u,v) from contact.pos + Fn weights already in obs idx 6–9.",
            "in_39d": True,
            "freq": "control rate (decision steps)",
            "noise": "CoP sensitive to light contact / invalid mask",
            "latency": "same-step",
            "distinguish_sidewall": "Partial — location on pad, not cylinder surface class",
        },
        "2_contact_pressure_distribution": {
            "class": "C",
            "franka": "Needs tactile array; stock Panda lacks pad pressure map.",
            "mujoco": "Can approximate via multi-point contacts / touch sites; not full pressure image in current obs.",
            "in_39d": False,
            "note": "fn_L/R are scalar sums only (A-level partial).",
        },
        "3_finger_contact_force_direction": {
            "class": "B",
            "franka": "Not directly available without fingertip F/T or tactile.",
            "mujoco": "mj_contactForce wrench available offline (normal+tangential); NOT in current Actor obs.",
            "in_39d": False,
            "derivable": "Yes in sim via envs/contacts/frames.extract_contacts — logging only today.",
        },
        "4_slip_or_shear_force": {
            "class": "B",
            "franka": "Needs tactile shear or estimated slip; not stock.",
            "mujoco": "Ft1/Ft2 tangential components in contact wrench — sim upper-bound.",
            "in_39d": False,
        },
        "5_gripper_joint_actuator_force_residuals": {
            "class": "B",
            "franka": "Finger effort/current partially available via libfranka; not in current 39D.",
            "mujoco": "Actuator force / qfrc available; tau_from_secure is ctrl residual proxy only.",
            "in_39d": "partial (tau_from_secure)",
        },
        "6_wrist_force_torque": {
            "class": "C",
            "franka": "Panda has wrist F/T sensor — typically exposed in real stacks; NOT wired into this project's 39D obs.",
            "mujoco": "No <force>/<torque> sensors in panda_torque.xml currently; can be added as approx.",
            "in_39d": False,
        },
        "privileged_sidewall_surface_class": {
            "class": "D",
            "note": "classify_contacts / is_correct_sidewall — simulation-only surface labels. Never deploy as Actor input.",
        },
        "gt_airborne_27_38": {
            "class": "D",
            "note": "GT object-relative pose/twist in 39D suffix — privileged stand-in for vision; treat carefully in sensing claims.",
        },
    }


def _contact_wrench_summary(sim) -> dict:
    from envs.contacts.frames import extract_contacts

    contacts, _ = extract_contacts(sim)
    fn = ft = 0.0
    n = 0
    dirs = []
    for c in contacts:
        # Prefer explicit Fn/Ft fields (avoid double-counting Ft + Ft1 + Ft2)
        if c.get("Fn") is not None:
            fn += abs(float(c["Fn"]))
            ft += abs(float(c.get("Ft") or 0))
            n += 1
            continue
        w = c.get("wrench6") or c.get("wrench")
        if w is None:
            continue
        w = np.asarray(w, float).reshape(-1)
        fn += abs(float(w[0]))
        if w.size >= 3:
            ft += float(np.linalg.norm(w[1:3]))
        n += 1
    return {
        "n_contacts": n,
        "fn_sum": float(fn),
        "ft_sum": float(ft),
        "ft_over_fn": float(ft / max(fn, 1e-6)),
    }


def collect_sensing_diag(*, device: str = "cuda") -> list[dict]:
    """Small frozen-policy diagnostic rollouts with Ft/shear + privileged labels."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "sensing_diag_episodes.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    # critic unused — pass a dummy by loading critic? act_with_value needs critic
    from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_critic
    from experiments.precision.p75_critic_calibration.paths import P64_STEP4_CRITIC

    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=80, t_max=3100, weights=PoseRewardWeights(), use_stabilization=False
    )
    schedule = build_p54_dataset(seed=SEED)["schedule"]
    blocked = blocked_eval_ids()
    c_starts = [s for s in schedule if s.skill == "C" and s.traj_id not in blocked]
    rng = np.random.default_rng(SEED)
    picks = [c_starts[i] for i in rng.choice(len(c_starts), size=min(N_SENSE_EP // 8, len(c_starts)), replace=False)]
    # 8 rolls each ≈ N_SENSE_EP
    rolls = max(1, N_SENSE_EP // max(len(picks), 1))
    eps = []
    ep_i = 0
    v2 = make_specs([])["V2"]
    for st in picks:
        for j in range(rolls):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=77000 + ep_i)
            seed_all(seed, deterministic_torch=True)
            env.reset(st)
            ctrl = CtrlState()
            done = False
            steps = []
            saw_fc = ever_opp = lost = had = False
            while not done:
                obs = np.asarray(env.obs_seq[-1], np.float32).copy()
                # Geometry/wrench at the SAME decision state as obs (before action)
                g0 = relative_geometry(env.env.sim)
                wr0 = _contact_wrench_summary(env.env.sim)
                if not saw_fc:
                    window = "pre_fc"
                elif ever_opp:
                    window = "opposing_to_frame"
                elif lost:
                    window = "fc_to_contact_loss"
                else:
                    window = "fc_to_opposing"
                steps.append(
                    {
                        "obs39": obs.tolist(),
                        "obs27": obs[:27].tolist(),
                        "window": window,
                        "correct_sidewall": int(g0["correct_sidewall"]),
                        "bilateral": int(g0["bilateral"]),
                        "opposing": int(g0["bilateral"] and g0["correct_sidewall"]),
                        "has_pad_contact": int(g0["has_pad_contact"]),
                        "e_pos_mm": float(g0["e_pos_mm"]),
                        "nL": int(g0.get("nL") or 0),
                        "nR": int(g0.get("nR") or 0),
                        "min_pad": int(g0.get("min_pad") or 0),
                        "theta_deg": float(g0.get("theta_deg") or 0),
                        "fn_sum": wr0["fn_sum"],
                        "ft_sum": wr0["ft_sum"],
                        "ft_over_fn": wr0["ft_over_fn"],
                        "n_contacts": wr0["n_contacts"],
                        "u_L": float(obs[6]),
                        "v_L": float(obs[7]),
                        "u_R": float(obs[8]),
                        "v_R": float(obs[9]),
                    }
                )
                tok_a, resid, logp, val = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device_t)
                tok, ctrl, dbg = select_token(
                    env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=ctrl, spec=v2
                )
                steps[-1]["executed"] = int(tok)
                r, done, info = env.step(int(tok), resid)
                g1 = relative_geometry(env.env.sim)
                wr1 = _contact_wrench_summary(env.env.sim)
                # Post-step labels for "imminent sidewall" prediction (FC-window task)
                steps[-1]["correct_sidewall_next"] = int(g1["correct_sidewall"])
                steps[-1]["ft_sum_next"] = wr1["ft_sum"]
                if g1["has_pad_contact"]:
                    if not saw_fc:
                        saw_fc = True
                    had = True
                elif saw_fc and had:
                    lost = True
                if g1["bilateral"] and g1["correct_sidewall"]:
                    ever_opp = True

            eps.append(
                {
                    "traj_id": st.traj_id,
                    "seed": seed,
                    "steps": steps,
                    "reached_opposing": int(ever_opp),
                    "lost_contact_after_fc": int(lost),
                    "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
                }
            )
            ep_i += 1
            if ep_i % 16 == 0 or ep_i == 1:
                print(f"[sense-diag] {ep_i} {st.traj_id[-20:]} opp={ever_opp} loss={lost}", flush=True)
    path.write_bytes(pickle.dumps(eps, protocol=pickle.HIGHEST_PROTOCOL))
    return eps


def _ic_split(ids, val_frac, seed):
    rng = np.random.default_rng(seed)
    ids = sorted(set(ids))
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac))) if len(ids) > 3 else max(1, len(ids) // 4)
    return set(ids[n_val:]), set(ids[:n_val])


def incremental_info_gain(eps: list[dict]) -> dict:
    """Compare obs27 / eng / +Ft / +CoP-hist / +privileged-D (diagnostic only)."""
    rows = []
    for ep in eps:
        steps = ep["steps"]
        hist27 = []
        for ti, s in enumerate(steps):
            o27 = np.asarray(s["obs27"], float)
            hist27.append(o27)
            # Primary geometry task: FC-window only, predict imminent (post-step) sidewall.
            # Avoid opposing_to_frame where y_sw≈1 by stage definition (trivial).
            if s["window"] not in FC_WINDOWS:
                continue
            if "correct_sidewall_next" not in s:
                continue
            ft = np.asarray([s["ft_sum"], s["fn_sum"], s["ft_over_fn"], s["n_contacts"]], float)
            # short CoP history
            lo = max(0, ti + 1 - 3)
            cop_h = np.concatenate(
                [[steps[k]["u_L"], steps[k]["v_L"], steps[k]["u_R"], steps[k]["v_R"]] for k in range(lo, ti + 1)]
            )
            if cop_h.size < 12:
                cop_h = np.pad(cop_h, (12 - cop_h.size, 0))
            eng = np.asarray(
                [
                    o27[0],
                    o27[2],
                    o27[3],
                    float(np.mean([h[2] for h in hist27[-5:]])),
                    float(np.mean([h[3] for h in hist27[-5:]])),
                    o27[10] - o27[11],
                    o27[6] - o27[8],
                ],
                float,
            )
            # Privileged upper bound — MUST NOT include target label
            priv = np.asarray(
                [
                    s["e_pos_mm"],
                    float(s.get("nL") or 0),
                    float(s.get("nR") or 0),
                    float(s.get("min_pad") or 0),
                    float(s.get("theta_deg") or 0),
                    float(s["bilateral"]),
                ],
                float,
            )
            rows.append(
                {
                    "traj_id": ep["traj_id"],
                    "base": o27,
                    "eng": np.concatenate([o27, eng]),
                    "plus_ft": np.concatenate([o27, eng, ft]),
                    "plus_cop_hist": np.concatenate([o27, eng, cop_h]),
                    "plus_ft_cop": np.concatenate([o27, eng, ft, cop_h]),
                    "plus_priv_D": np.concatenate([o27, eng, priv]),
                    "y_sw": int(s["correct_sidewall_next"]),  # imminent sidewall after this decision
                    "y_loss": int(ep["lost_contact_after_fc"]),
                    "y_opp": int(ep["reached_opposing"]),
                    "t": ti,
                    "window": s["window"],
                }
            )

    # Stratify ICs by whether they ever show correct_sidewall (rare-positive)
    by_ic = {}
    for r in rows:
        by_ic.setdefault(r["traj_id"], False)
        by_ic[r["traj_id"]] = by_ic[r["traj_id"]] or bool(r["y_sw"])
    pos_ics = [i for i, v in by_ic.items() if v]
    neg_ics = [i for i, v in by_ic.items() if not v]
    rng = np.random.default_rng(SEED + 1)

    def split_list(lst):
        lst = list(lst)
        rng.shuffle(lst)
        n_val = max(1, int(round(len(lst) * VAL_FRAC))) if len(lst) > 3 else max(1, len(lst) // 4)
        # keep at least one positive IC in eval when possible
        return set(lst[n_val:]), set(lst[:n_val])

    fit_ics, eval_ics = set(), set()
    for pool in (pos_ics, neg_ics):
        if not pool:
            continue
        f, e = split_list(pool)
        fit_ics |= f
        eval_ics |= e
    if not any(by_ic[i] for i in eval_ics) and pos_ics:
        # force one positive IC into eval
        move = pos_ics[0]
        fit_ics.discard(move)
        eval_ics.add(move)
    fit = np.asarray([r["traj_id"] in fit_ics for r in rows])
    ev = np.asarray([r["traj_id"] in eval_ics for r in rows])

    report = {
        "n_steps": len(rows),
        "n_fit_ics": len(fit_ics),
        "n_eval_ics": len(eval_ics),
        "label": "sensing_upper_bound_study",
        "tasks": {},
    }
    for task, yk in (("correct_sidewall_now", "y_sw"), ("eventual_loss", "y_loss"), ("eventual_opposing", "y_opp")):
        report["tasks"][task] = {}
        for feat in ("base", "eng", "plus_ft", "plus_cop_hist", "plus_ft_cop", "plus_priv_D"):
            X = np.stack([r[feat] for r in rows])
            y = np.asarray([r[yk] for r in rows])
            blob = eval_binary_probe(X, y, fit, ev)
            if feat == "plus_priv_D":
                blob["class"] = "D_privileged_sim_only_not_deployable"
            elif feat in ("plus_ft", "plus_ft_cop"):
                blob["class"] = "B_sim_derivable_Ft_upper_bound"
            elif feat == "plus_cop_hist":
                blob["class"] = "A_or_B_existing_CoP_history"
            else:
                blob["class"] = "A_existing"
            report["tasks"][task][feat] = blob

    # Lead time: score sidewall with plus_ft on eval ICs — crude threshold
    # Fit scores on all fit, apply to eval episodes
    sw_rows = [r for r in rows if r["traj_id"] in eval_ics]
    if sw_rows:
        # use eng+ft scores from a quick refit
        X = np.stack([r["plus_ft"] for r in rows])
        y = np.asarray([r["y_sw"] for r in rows])
        mu, sd = X[fit].mean(0), X[fit].std(0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        Xz = (X - mu) / sd
        from experiments.precision.p77_contact_observability.metrics import ridge_scores

        scores_all, _ = ridge_scores(Xz, y, fit, np.ones(len(y), dtype=bool))
        # per eval episode first detect vs loss
        by_ep = {}
        for i, r in enumerate(rows):
            if r["traj_id"] not in eval_ics:
                continue
            by_ep.setdefault(r["traj_id"], []).append((r["t"], float(scores_all[i]), r["y_sw"], r["window"]))
        leads = []
        for tid, seq in by_ep.items():
            seq = sorted(seq, key=lambda z: z[0])
            # find loss window start
            t_loss = next((t for t, _, _, w in seq if w == "fc_to_contact_loss"), None)
            # detect sidewall score above median of positives on fit
            thr = float(np.median(scores_all[fit & (y == 1)])) if np.any(fit & (y == 1)) else 0.0
            t_det = next((t for t, sc, _, _ in seq if sc >= thr), None)
            if t_det is not None and t_loss is not None:
                leads.append(float(t_loss - t_det))
        report["detection_lead_before_loss_steps"] = {
            "n": len(leads),
            "mean_lead": float(np.mean(leads)) if leads else None,
            "frac_positive_lead": float(np.mean([l > 0 for l in leads])) if leads else None,
        }
    return report


def run_sensing(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = sensing_catalog()
    (OUT / "SENSING_FEASIBILITY.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    (OUT / "SENSING_FEASIBILITY.md").write_text(
        "# Sensing Feasibility (P77 Phase 4)\n\n"
        "Classes: **A** Actor-available · **B** derivable from causal sensors · "
        "**C** new hardware · **D** privileged sim-only (never silently deploy).\n\n```json\n"
        + json.dumps(catalog, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print("=== sensing diagnostic collect ===", flush=True)
    eps = collect_sensing_diag(device=device)
    print("=== incremental info gain ===", flush=True)
    gain = incremental_info_gain(eps)
    (OUT / "INCREMENTAL_INFORMATION_GAIN.json").write_text(json.dumps(gain, indent=2), encoding="utf-8")
    compact = {
        t: {f: v.get("auroc") for f, v in feats.items()} for t, feats in gain["tasks"].items()
    }
    compact["lead"] = gain.get("detection_lead_before_loss_steps")
    (OUT / "INCREMENTAL_INFORMATION_GAIN.md").write_text(
        "# Incremental Information Gain (P77 Phase 5)\n\n"
        "**Sensing upper-bound study** (sim Ft/shear logged; privileged D reported separately).\n\n```json\n"
        + json.dumps({"compact_auroc": compact, "full": gain}, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(compact, indent=2), flush=True)
    return {"catalog": catalog, "gain": gain, "n_diag_eps": len(eps)}


if __name__ == "__main__":
    run_sensing()
