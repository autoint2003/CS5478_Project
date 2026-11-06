"""Phase 3–4: on-policy post-contact counterfactuals + dominant mechanism."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs
from experiments.precision.p72_postcontact_reliability.paths import (
    MAX_EP_STEPS,
    N_CF_SEEDS,
    N_SNAP_PER_KIND,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_SPLIT,
    SEED,
    STATE_DIR,
    T_MAX,
)
from experiments.precision.p72_postcontact_reliability.v2_wrap import (
    PostRefinement,
    WrapState,
    select_token,
    v2_spec,
)


def _lookup():
    blocked = blocked_eval_ids()
    return {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }


def _mine_snapshots(env, actor, critic, book, device, lookup, funnel_rows: list) -> dict[str, list]:
    """Replay V2 trajectories and capture full states at post-contact events."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    kinds = {
        "after_first_contact": [],
        "first_bilateral": [],
        "first_sidewall": [],
        "e_le_12": [],
        "min_pad_ge4": [],
    }
    # Prefer diverse trajs from funnel
    seen = set()
    plans = []
    for r in funnel_rows:
        key = (r["traj_id"], r["seed"])
        if key in seen:
            continue
        seen.add(key)
        plans.append(r)
        if len(plans) >= 40:
            break

    v2 = v2_spec()
    for r in plans:
        if all(len(kinds[k]) >= N_SNAP_PER_KIND for k in kinds):
            break
        start = lookup.get(r["traj_id"])
        if start is None:
            continue
        seed_all(int(r["seed"]), deterministic_torch=True)
        env.reset(start)
        ensure_inner_capture_buffers(env)
        st = WrapState()
        done = False
        got = {k: False for k in kinds}
        while not done:
            tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
            tok, st, _dbg = select_token(
                env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, post=None
            )
            # capture BEFORE step if event about to... actually capture AFTER step when condition holds
            _r, done, _info = env.step(int(tok), resid)
            g = relative_geometry(env.env.sim)
            ensure_inner_capture_buffers(env)

            def maybe(kind, cond):
                if got[kind] or len(kinds[kind]) >= N_SNAP_PER_KIND or not cond:
                    return
                if env.t + 8 > env.max_steps:
                    return
                state = take_full_state(
                    env,
                    meta={"kind": kind, "traj_id": start.traj_id, "seed": r["seed"], "t": int(env.t), "succ": r["r3_ready"]},
                )
                sid = f"{kind}_{len(kinds[kind]):03d}_{start.traj_id[-16:]}_t{env.t}"
                path = STATE_DIR / f"{sid}.pkl"
                path.write_bytes(pickle.dumps(state))
                kinds[kind].append(
                    {
                        "id": sid,
                        "path": str(path),
                        "traj_id": start.traj_id,
                        "seed": int(r["seed"]),
                        "t": int(env.t),
                        "e_pos_mm": float(g["e_pos_mm"]),
                        "min_pad": int(g["min_pad"]),
                        "from_succ": int(r["r3_ready"]),
                    }
                )
                got[kind] = True

            maybe("after_first_contact", g["has_pad_contact"] and not got["after_first_contact"])
            # only mark first time
            maybe("first_bilateral", bool(g["bilateral"]))
            maybe("first_sidewall", bool(g["correct_sidewall"] and g["bilateral"]))
            maybe("e_le_12", float(g["e_pos_mm"]) <= 12.0 and g["has_pad_contact"])
            maybe("min_pad_ge4", int(g["min_pad"]) >= 4)

        print(
            f"[mine] {r['traj_id'][-22:]} bank="
            + ",".join(f"{k}:{len(v)}" for k, v in kinds.items()),
            flush=True,
        )
    return kinds


def _continue_from_state(env, actor, critic, book, start, state, device, *, post: PostRefinement | None, seed: int) -> dict:
    seed_all(seed, deterministic_torch=True)
    restore_full_state(env, state, start)
    ensure_inner_capture_buffers(env)
    # Finish episode with V2 wrap + optional post overlay from current contact state
    # Rebuild wrap: mark saw_contact True since these are post-contact snaps
    st = WrapState()
    st.saw_contact = True
    st.v2.phase = "done"  # pre-contact V2 finished
    st.v2.activated = True
    v2 = v2_spec()
    done = False
    info = {}
    e0 = float(relative_geometry(env.env.sim)["e_pos_mm"])
    pad0 = int(relative_geometry(env.env.sim)["min_pad"])
    bil0 = int(relative_geometry(env.env.sim)["bilateral"])
    ever_r3 = ever_pad4 = ever_e9 = contact_loss = False
    had_c = True
    while not done:
        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, _ = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, post=post
        )
        _r, done, info = env.step(int(tok), resid)
        g = relative_geometry(env.env.sim)
        if g["has_pad_contact"]:
            had_c = True
        if had_c and not g["has_pad_contact"]:
            contact_loss = True
        if g["min_pad"] >= 4:
            ever_pad4 = True
        if g["e_pos_mm"] <= 9:
            ever_e9 = True
        tr = env.mon.tracker
        ok_s = float(tr.ok_s) if tr else 0.0
        from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import evaluate_predicates

        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        if (preds["dwell_complete"] and preds["frame_ok"]) or int(info.get("handoff") or 0):
            ever_r3 = True
    return {
        "e0": e0,
        "pad0": pad0,
        "bil0": bil0,
        "r3_ready": int(ever_r3 or int(info.get("handoff") or 0)),
        "handoff": int(info.get("handoff") or 0),
        "SUCCESS_RELAXED": int(info.get("SUCCESS_RELAXED_12S") or 0),
        "ever_pad4": int(ever_pad4),
        "ever_e9": int(ever_e9),
        "contact_loss": int(contact_loss),
        "post_name": None if post is None else post.name,
    }


def _mode_seq(funnel_rows: list, *, succ_only: bool = True, L: int = 5) -> list[int]:
    cnt = Counter()
    for r in funnel_rows:
        if succ_only and not r.get("r3_ready"):
            continue
        seq = r.get("post_tokens") or []
        if len(seq) >= 2:
            cnt[tuple(seq[:L])] += 1
    if not cnt:
        return [82, 50, 71, 50, 71]  # GRIP then APPROACH-ish fallback
    return list(cnt.most_common(1)[0][0])


def run_phase34(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "FUNNEL_RAW.json"
    if not raw_path.exists():
        raise FileNotFoundError("Run Phase 1–2 first (FUNNEL_RAW.json missing)")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    funnel_rows = raw["rows"]
    report12 = {k: raw[k] for k in raw if k != "rows"}

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    lookup = _lookup()

    print("=== Mine post-contact snapshots ===", flush=True)
    banks = _mine_snapshots(env, actor, critic, book, device_t, lookup, funnel_rows)
    (OUT / "SNAP_BANK.json").write_text(
        json.dumps({k: [{kk: vv for kk, vv in s.items() if kk != "path"} | {"path": s["path"]} for s in v] for k, v in banks.items()}, indent=2),
        encoding="utf-8",
    )

    # Verify restoration equivalence on one snapshot
    restore_ok = None
    for kind in ("after_first_contact", "first_bilateral", "e_le_12", "min_pad_ge4"):
        snaps = banks.get(kind) or []
        if not snaps:
            continue
        sm = snaps[0]
        start = lookup[sm["traj_id"]]
        state = pickle.loads(Path(sm["path"]).read_bytes())
        restore_full_state(env, state, start)
        ensure_inner_capture_buffers(env)
        g1 = relative_geometry(env.env.sim)
        live = take_full_state(env, meta={"verify": True})
        restore_full_state(env, live, start)
        g2 = relative_geometry(env.env.sim)
        restore_ok = {
            "kind": kind,
            "e_pos_match": abs(float(g1["e_pos_mm"]) - float(g2["e_pos_mm"])) < 1e-6,
            "t_match": int(env.t) == int(state["policy_step"]),
            "min_pad_match": int(g1["min_pad"]) == int(g2["min_pad"]),
        }
        print(f"[restore_check] {restore_ok}", flush=True)
        break

    corr_seq = _mode_seq(funnel_rows, succ_only=True, L=5)
    # Contact-preserving: bias GRIP then soft approach
    preserve_seq = [82] + [t for t in corr_seq if t != 82][:4]
    if len(preserve_seq) < 3:
        preserve_seq = [82, 50, 71, 50]
    hold_seq = [82, 82, 71, 71, 50]  # near-ready stabilization candidates from M111

    posts = {
        "baseline": None,
        "A_continue": PostRefinement(name="A_continue", kind="continue_seq", sequence=corr_seq, max_steps=5),
        "B_preserve": PostRefinement(
            name="B_preserve", kind="contact_preserve", sequence=preserve_seq, max_steps=5, abort_on_contact_loss=True
        ),
        "C_nearhold": PostRefinement(
            name="C_nearhold",
            kind="near_ready_hold",
            sequence=hold_seq,
            max_steps=5,
            require_bilateral_obs=True,
            abort_on_contact_loss=True,
        ),
    }

    # Focus CF on kinds most relevant
    focus_kinds = ["after_first_contact", "first_bilateral", "e_le_12", "min_pad_ge4"]
    results = {}
    for kind in focus_kinds:
        snaps = banks.get(kind) or []
        results[kind] = {}
        for pname, post in posts.items():
            stats = []
            for sm in snaps:
                start = lookup[sm["traj_id"]]
                state = pickle.loads(Path(sm["path"]).read_bytes())
                for si in range(N_CF_SEEDS):
                    stats.append(
                        _continue_from_state(
                            env, actor, critic, book, start, state, device_t,
                            post=post, seed=SEED + 8000 + si + 17 * (sum(ord(c) for c in sm["id"]) % 10000),
                        )
                    )
            agg = {
                "n": len(stats),
                "r3_ready": float(np.mean([s["r3_ready"] for s in stats])) if stats else None,
                "SUCCESS_RELAXED": float(np.mean([s["SUCCESS_RELAXED"] for s in stats])) if stats else None,
                "ever_e9": float(np.mean([s["ever_e9"] for s in stats])) if stats else None,
                "ever_pad4": float(np.mean([s["ever_pad4"] for s in stats])) if stats else None,
                "contact_loss": float(np.mean([s["contact_loss"] for s in stats])) if stats else None,
            }
            results[kind][pname] = agg
            print(f"[cf] {kind}/{pname} r3={agg['r3_ready']} e9={agg['ever_e9']} pad4={agg['ever_pad4']} loss={agg['contact_loss']}", flush=True)

    # Dominant mechanism from Phase 1–2 + CF deltas vs baseline
    g_s, g_f = report12.get("geom_succ") or {}, report12.get("geom_fail") or {}
    funnel = (report12.get("funnel_all") or {}).get("first_passage_ever") or {}
    near_n = int(report12.get("near_fail_n") or 0)
    n_fail = max(int(report12.get("n_fail") or 1), 1)

    # CF uplifts vs baseline across stages (not only after_first_contact)
    uplifts = {}
    uplifts_by_kind = {}
    for pname in ("A_continue", "B_preserve", "C_nearhold"):
        per = {}
        for kind in focus_kinds:
            base = (results.get(kind) or {}).get("baseline") or {}
            cur = (results.get(kind) or {}).get(pname) or {}
            if base.get("r3_ready") is None or cur.get("r3_ready") is None:
                continue
            per[kind] = float(cur["r3_ready"]) - float(base["r3_ready"])
        uplifts_by_kind[pname] = per
        uplifts[pname] = float(per.get("after_first_contact") or 0)

    def _generalizes(pname: str) -> bool:
        """True if uplift is non-negative on ≥2 stages and does not crater any stage."""
        per = uplifts_by_kind.get(pname) or {}
        if not per:
            return False
        vals = list(per.values())
        if any(v < -0.05 for v in vals):
            return False
        return sum(1 for v in vals if v > 0.02) >= 2

    # Heuristic case — geometry/contact evidence first, then require CF generalization
    frac_fail_no_e9 = 1.0 - float(g_f.get("frac_reach_e9") or 0)
    frac_fail_no_pad = 1.0 - float(g_f.get("frac_pad4") or 0)
    switch_gap = float(g_f.get("mean_post_switches") or 0) - float(g_s.get("mean_post_switches") or 0)
    near_frac = near_n / n_fail

    if near_frac >= 0.35 and _generalizes("C_nearhold"):
        case = "D"
        chosen = "C_nearhold"
    elif frac_fail_no_pad >= 0.55 and (g_f.get("frac_reach_e9") or 0) >= 0.25 and _generalizes("B_preserve"):
        case = "B"
        chosen = "B_preserve"
    elif frac_fail_no_e9 >= 0.6 and _generalizes("A_continue"):
        case = "C"
        chosen = "A_continue"
    elif switch_gap > 0.5 and _generalizes("A_continue"):
        case = "A"
        chosen = "A_continue"
    elif max(uplifts.values()) <= 0.02 or not any(_generalizes(p) for p in uplifts):
        # Oracle/single-stage gains only → Case E
        case = "E"
        chosen = "baseline"
    else:
        chosen = max((p for p in uplifts if _generalizes(p)), key=lambda p: uplifts[p], default="baseline")
        case = {"A_continue": "A", "B_preserve": "B", "C_nearhold": "D", "baseline": "E"}.get(chosen, "E")
        if chosen == "baseline":
            case = "E"

    mechanism = {
        "case": case,
        "chosen_intervention": chosen,
        "uplifts_after_fc": uplifts,
        "uplifts_by_kind": uplifts_by_kind,
        "corr_seq": corr_seq,
        "preserve_seq": preserve_seq,
        "hold_seq": hold_seq,
        "evidence": {
            "frac_fail_no_e9": frac_fail_no_e9,
            "frac_fail_no_pad4": frac_fail_no_pad,
            "switch_gap_fail_minus_succ": switch_gap,
            "near_fail_frac": near_frac,
            "funnel_ever": funnel,
        },
        "interpretation": {
            "A": "Correction sequence interrupted — extend/maintain successful post-contact tokens",
            "B": "Insufficient contact support — contact-preserving GRIP transition",
            "C": "Geometry remains outside R3 — bounded post-contact sequence",
            "D": "Near-ready instability — short stabilization",
            "E": "No generalizable intervention — keep V2",
        }[case if case in "ABCDE" else "E"],
    }

    out = {
        "cf_results": results,
        "mechanism": mechanism,
        "restoration_check": restore_ok,
        "posts": {k: None if v is None else {"name": v.name, "kind": v.kind, "sequence": v.sequence, "max_steps": v.max_steps} for k, v in posts.items()},
    }
    (OUT / "MATCHED_COUNTERFACTUALS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / "DOMINANT_MECHANISM.json").write_text(json.dumps(mechanism, indent=2), encoding="utf-8")
    (OUT / "DOMINANT_MECHANISM.md").write_text(
        f"# Dominant Mechanism (P72 Phase 4)\n\n**Case {case}** — choose `{chosen}`\n\n"
        f"{mechanism['interpretation']}\n\n```json\n{json.dumps(mechanism, indent=2)}\n```\n",
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    run_phase34()
