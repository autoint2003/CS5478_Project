"""Phase 1–2: Trace pre-contact geometry and classify C failures."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates
from experiments.precision.p68_precontact_geometry.geometry import contact_points_summary, relative_geometry
from experiments.precision.p68_precontact_geometry.paths import (
    MAX_EP_STEPS,
    N_ICS,
    N_PER_IC,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    SEED,
    STATE_DIR,
    T_MAX,
    TRACE_DIR,
)
import pickle
from pathlib import Path


def _train_c(n: int) -> list:
    blocked = blocked_eval_ids()
    return [
        s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    ][:n]


def rollout_trace(env, actor, critic, start, device, *, seed: int) -> dict:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    ensure_inner_capture_buffers(env)

    geo0 = relative_geometry(env.env.sim)
    first_contact_t: int | None = None
    first_contact: dict | None = None
    pre_contact_last: dict | None = None
    bilateral_t: int | None = None
    stable_t: int | None = None
    near_t: int | None = None
    r3_t: int | None = None

    # Snapshots for later CF (pre-first-contact)
    snap_buf: dict[int, dict] = {}
    tokens_log = []
    epos_series = []
    theta_series = []
    contact_disp = None  # obj displacement at first contact
    obj_before = None

    ever_sw = False
    max_min_pad = 0
    max_ok_s = 0.0
    done = False
    info: dict[str, Any] = {}

    while not done:
        t_before = int(env.t)
        geo_before = relative_geometry(env.env.sim)
        epos_series.append(geo_before["e_pos_mm"])
        theta_series.append(geo_before["theta_deg"])

        # buffer only a few pre-contact frames (RAM)
        if t_before not in snap_buf and t_before <= 16 and len(snap_buf) < 10:
            ensure_inner_capture_buffers(env)
            snap_buf[t_before] = take_full_state(env, meta={"t": t_before, "traj_id": start.traj_id})

        tok, resid, logp, val = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tokens_log.append(
            {
                "t": t_before,
                "token": int(tok),
                "resid": np.asarray(resid, np.float32).tolist(),
                "logp": float(logp),
                "value": float(val),
                "e_pos_mm": geo_before["e_pos_mm"],
                "theta_deg": geo_before["theta_deg"],
                "has_pad_contact": geo_before["has_pad_contact"],
                "bilateral": geo_before["bilateral"],
                "min_pad": geo_before["min_pad"],
                "correct_sidewall": geo_before["correct_sidewall"],
            }
        )

        if not geo_before["has_pad_contact"]:
            pre_contact_last = {
                "t": t_before,
                "geo": geo_before,
                "token": int(tok),
                "resid": np.asarray(resid, np.float32).tolist(),
                "logp": float(logp),
            }
            obj_before = np.asarray(geo_before["obj_pos"], float)

        r, done, info = env.step(tok, resid)
        geo_after = relative_geometry(env.env.sim)
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        max_ok_s = max(max_ok_s, ok_s)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        max_min_pad = max(max_min_pad, int(geo_after["min_pad"]))
        if geo_after["correct_sidewall"]:
            ever_sw = True

        if first_contact_t is None and geo_after["has_pad_contact"]:
            first_contact_t = int(env.t)
            pts = contact_points_summary(env.env.sim)
            obj_after = np.asarray(geo_after["obj_pos"], float)
            if obj_before is not None:
                contact_disp = float(np.linalg.norm(obj_after - obj_before))
            first_contact = {
                "t": first_contact_t,
                "geo_before": pre_contact_last["geo"] if pre_contact_last else geo_before,
                "geo_after": geo_after,
                "token_at_contact": int(tok),
                "points": pts,
                "object_disp_m": contact_disp,
                "delta_e_pos": float(geo_after["e_pos_mm"] - (pre_contact_last["geo"]["e_pos_mm"] if pre_contact_last else geo_before["e_pos_mm"])),
                "delta_theta": float(geo_after["theta_deg"] - (pre_contact_last["geo"]["theta_deg"] if pre_contact_last else geo_before["theta_deg"])),
            }

        if bilateral_t is None and geo_after["bilateral"]:
            bilateral_t = int(env.t)
        if stable_t is None and env.diag.get("stable_hold"):
            stable_t = int(env.t)
        if near_t is None and preds["frame_ok"] and ok_s > 0:
            near_t = int(env.t)
        if r3_t is None and preds["dwell_complete"] and preds["frame_ok"]:
            r3_t = int(env.t)

    stages = {
        "ever_bilateral": int(bilateral_t is not None),
        "ever_correct_sidewall": int(ever_sw),
        "stable_hold": int(env.diag.get("stable_hold") or 0),
        "near_ready": int(near_t is not None),
        "r3_ready": int(r3_t is not None or int(info.get("handoff") or 0)),
        "handoff": int(info.get("handoff") or 0),
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "max_min_pad": max_min_pad,
        "max_ok_s": max_ok_s,
    }

    return {
        "traj_id": start.traj_id,
        "seed": seed,
        "geo0": geo0,
        "pre_contact_last": pre_contact_last,
        "first_contact": first_contact,
        "first_contact_t": first_contact_t,
        "bilateral_t": bilateral_t,
        "stable_t": stable_t,
        "near_t": near_t,
        "r3_t": r3_t,
        "stages": stages,
        "tokens": tokens_log,
        "epos_series": epos_series,
        "theta_series": theta_series,
        "snap_times": sorted(snap_buf.keys()),
        "_snaps": snap_buf,  # kept in memory for mining; stripped from JSON
        "terminal": {
            "handoff_reason": info.get("handoff_reason"),
            "fail_reason": info.get("fail_reason"),
            "drop": int(info.get("drop") or 0),
        },
    }


def classify_failure(ep: dict) -> list[str]:
    """Multi-label A–F classification (not exclusive)."""
    if ep["stages"]["SUCCESS_RELAXED"] or ep["stages"]["r3_ready"]:
        return ["SUCCESS"]
    cats = []
    fc = ep.get("first_contact")
    pre = ep.get("pre_contact_last")
    geo0 = ep.get("geo0") or {}

    pre_e = float((pre or {}).get("geo", geo0).get("e_pos_mm") or geo0.get("e_pos_mm") or 0)
    pre_th = float((pre or {}).get("geo", geo0).get("theta_deg") or geo0.get("theta_deg") or 0)

    # A: incorrect pre-contact alignment
    if pre_e > 25.0 or pre_th > 20.0:
        cats.append("A_incorrect_precontact_alignment")

    # B: correct-ish alignment but bad first-contact geometry
    if fc:
        after = fc["geo_after"]
        if pre_e <= 25.0 and pre_th <= 20.0:
            if not after.get("correct_sidewall") and after.get("grasp_kind") not in (
                "opposing_sidewall",
                "sidewall_partial",
            ):
                cats.append("B_incorrect_first_contact_geometry")
        # C: displacement by contact / subsequent — large delta e_pos or object disp
        if abs(float(fc.get("delta_e_pos") or 0)) > 8.0 or float(fc.get("object_disp_m") or 0) > 0.012:
            cats.append("C_displaced_by_contact_or_actions")
    else:
        cats.append("F_no_contact")

    # D: geometry ok-ish but insufficient pad count
    if ep["stages"]["ever_bilateral"] and int(ep["stages"].get("max_min_pad") or 0) < 4:
        if ep["stages"]["ever_correct_sidewall"] or (
            fc and fc["geo_after"].get("correct_sidewall")
        ):
            cats.append("D_insufficient_contact_count")
        elif ep["stages"]["ever_bilateral"]:
            cats.append("D_insufficient_contact_count")

    # E: had contact/stable but lost / never near-ready after bilateral
    if ep["stages"]["stable_hold"] and not ep["stages"]["near_ready"]:
        cats.append("E_later_stability_or_pose_loss")
    if ep["stages"]["ever_bilateral"] and not ep["stages"]["r3_ready"] and ep["stages"]["near_ready"]:
        # near but didn't finish — rare per P67
        cats.append("E_later_stability_or_pose_loss")

    if not cats:
        cats.append("F_other_ambiguous")
    return cats


def run_phase12(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    starts = _train_c(N_ICS)
    episodes = []
    snap_bank = []  # metadata only; states on disk
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    n_snaps_written = 0

    for st in starts:
        for i in range(N_PER_IC):
            seed = rollout_seed(split="train", traj_id=st.traj_id, rollout_i=80000 + i)
            ep = rollout_trace(env, actor, critic, st, device_t, seed=seed)
            cats = classify_failure(ep)
            ep["failure_classes"] = cats
            # harvest CF candidate snaps — write state blobs to disk (avoid RAM blowup)
            fc_t = ep.get("first_contact_t")
            snaps = ep.pop("_snaps", {})
            if fc_t is not None:
                for off, kind in (
                    (-4, "early_approach"),
                    (-2, "pre_contact"),
                    (-1, "immediate_pre_contact"),
                    (0, "at_first_contact"),
                    (2, "post_first_contact"),
                ):
                    target = int(fc_t) + off - 1
                    if target not in snaps or n_snaps_written >= 120:
                        continue
                    # Prefer failing episodes for CF bank
                    if (ep["stages"]["r3_ready"] or ep["stages"]["SUCCESS_RELAXED"]) and n_snaps_written > 8:
                        continue
                    sid = f"bank_{n_snaps_written:04d}_{kind}_t{target}"
                    path = STATE_DIR / f"{sid}.pkl"
                    path.write_bytes(pickle.dumps(snaps[target]))
                    snap_bank.append(
                        {
                            "traj_id": st.traj_id,
                            "seed": seed,
                            "kind": kind,
                            "t": target,
                            "first_contact_t": fc_t,
                            "failure_classes": cats,
                            "stages": ep["stages"],
                            "state_path": str(path),
                            "pre_e_pos": float(
                                ((ep.get("pre_contact_last") or {}).get("geo") or ep["geo0"]).get(
                                    "e_pos_mm"
                                )
                                or 0
                            ),
                        }
                    )
                    n_snaps_written += 1
            # free large snap dicts
            del snaps
            slim = {k: v for k, v in ep.items() if k != "tokens"}
            slim["n_tokens"] = len(ep["tokens"])
            # keep token head around contact
            fc = ep.get("first_contact_t")
            if fc is not None:
                slim["tokens_around_contact"] = [
                    t for t in ep["tokens"] if abs(t["t"] - fc) <= 5
                ]
            else:
                slim["tokens_around_contact"] = ep["tokens"][:8]
            episodes.append(slim)
            print(
                f"[trace] {st.traj_id[-28:]} i={i} fc={fc_t} sw={ep['stages']['ever_correct_sidewall']} "
                f"r3={ep['stages']['r3_ready']} cats={cats}",
                flush=True,
            )

    # Aggregate success vs fail pre-contact geometry
    succ = [e for e in episodes if e["stages"]["SUCCESS_RELAXED"] or e["stages"]["r3_ready"]]
    fail = [e for e in episodes if e not in succ]

    def _pre_stats(eps: list) -> dict:
        epos, th, disp, d_epos = [], [], [], []
        sw = 0
        for e in eps:
            pre = (e.get("pre_contact_last") or {}).get("geo") or e.get("geo0") or {}
            epos.append(float(pre.get("e_pos_mm") or 0))
            th.append(float(pre.get("theta_deg") or 0))
            fc = e.get("first_contact") or {}
            if fc.get("object_disp_m") is not None:
                disp.append(float(fc["object_disp_m"]))
            if fc.get("delta_e_pos") is not None:
                d_epos.append(float(fc["delta_e_pos"]))
            sw += int(e["stages"]["ever_correct_sidewall"])
        def summ(xs):
            if not xs:
                return None
            a = np.asarray(xs, float)
            return {"mean": float(a.mean()), "median": float(np.median(a)), "p90": float(np.percentile(a, 90))}
        return {
            "n": len(eps),
            "pre_e_pos": summ(epos),
            "pre_theta": summ(th),
            "first_contact_obj_disp_m": summ(disp),
            "delta_e_pos_at_contact": summ(d_epos),
            "correct_sidewall_rate": sw / max(1, len(eps)),
        }

    # Classification table
    class_rows = defaultdict(list)
    for e in episodes:
        for c in e["failure_classes"]:
            class_rows[c].append(e)

    class_summary = {}
    for c, eps in class_rows.items():
        class_summary[c] = {
            "n": len(eps),
            "mean_pre_e_pos": float(
                np.mean(
                    [
                        float(((e.get("pre_contact_last") or {}).get("geo") or e["geo0"]).get("e_pos_mm") or 0)
                        for e in eps
                    ]
                )
            ),
            "median_pre_e_pos": float(
                np.median(
                    [
                        float(((e.get("pre_contact_last") or {}).get("geo") or e["geo0"]).get("e_pos_mm") or 0)
                        for e in eps
                    ]
                )
            ),
            "correct_sidewall_rate": float(np.mean([e["stages"]["ever_correct_sidewall"] for e in eps])),
            "near_ready_rate": float(np.mean([e["stages"]["near_ready"] for e in eps])),
            "r3_ready_rate": float(np.mean([e["stages"]["r3_ready"] for e in eps])),
            "success_rate": float(np.mean([e["stages"]["SUCCESS_RELAXED"] for e in eps])),
        }

    # Predictor: correlate pre_e_pos with r3
    pre_e = [
        float(((e.get("pre_contact_last") or {}).get("geo") or e["geo0"]).get("e_pos_mm") or 0)
        for e in episodes
    ]
    r3 = [int(e["stages"]["r3_ready"]) for e in episodes]
    # simple threshold scan
    best_thr = None
    best_sep = -1
    for thr in (10, 15, 20, 25, 30, 40):
        low = [r for e, r in zip(pre_e, r3) if e <= thr]
        high = [r for e, r in zip(pre_e, r3) if e > thr]
        if low and high:
            sep = float(np.mean(low) - np.mean(high))
            if sep > best_sep:
                best_sep = sep
                best_thr = {"thr_mm": thr, "p_r3_low": float(np.mean(low)), "p_r3_high": float(np.mean(high)), "sep": sep}

    report = {
        "policy": str(P64_STEP4_ACTOR),
        "n_episodes": len(episodes),
        "n_ics": len(starts),
        "success_vs_fail": {"success": _pre_stats(succ), "fail": _pre_stats(fail)},
        "class_summary": class_summary,
        "pre_e_pos_predictor": best_thr,
        "stage_rates": {
            k: float(np.mean([e["stages"][k] for e in episodes]))
            for k in ("ever_bilateral", "ever_correct_sidewall", "stable_hold", "near_ready", "r3_ready", "SUCCESS_RELAXED")
        },
        "episodes": episodes,
    }
    # persist snap bank separately (pickle paths written in mine)
    # Drop heavy series from JSON if needed — keep compact
    for e in episodes:
        e.pop("epos_series", None)
        e.pop("theta_series", None)
        e.pop("snap_times", None)
    report["episodes"] = episodes
    (OUT / "TRACE_EPISODES.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "SNAP_BANK.json").write_text(json.dumps({"n": len(snap_bank), "snaps": snap_bank}, indent=2), encoding="utf-8")
    print(
        f"[p68] traced {len(episodes)} eps; succ={len(succ)} fail={len(fail)}; "
        f"snap_bank={len(snap_bank)}; pre_e predictor={best_thr}",
        flush=True,
    )
    return report


if __name__ == "__main__":
    run_phase12()
