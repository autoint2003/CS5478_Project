"""Phase 2 — expand independent FC snapshot coverage."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

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
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs, obs_features
from experiments.precision.p73_first_contact_bilateral.controller import WrapState, select_token, v2_spec
from experiments.precision.p79_matched_cf_coverage.paths import (
    CACHE,
    EVAL_FRAC,
    MAX_EP_STEPS,
    N_IC_TARGET,
    OUT,
    P64_ACTOR,
    P64_CRITIC,
    SEED,
    STATE_DIR,
    T_MAX,
)


def _ic_pools() -> dict:
    """Train/eval IC pools from P75 when available; else split C-skill schedule."""
    meta_path = OUT.parents[0] / "p75_critic_calibration" / "COLLECTION_META.json"
    blocked = blocked_eval_ids()
    schedule = [
        s for s in build_p54_dataset(seed=5478)["schedule"] if s.skill == "C" and s.traj_id not in blocked
    ]
    lookup = {s.traj_id: s for s in schedule}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        train_raw = list(meta.get("train_ics") or [])
        eval_raw = list(meta.get("eval_ics") or [])
        # Map P75 IC labels to traj_ids present in schedule
        def match(pool):
            out = []
            for ic in pool:
                if ic in lookup:
                    out.append(ic)
                    continue
                # P75 uses prefixed forms like C__gbi_...__hash or plan_c__gbi_...
                for tid in lookup:
                    if tid in ic or ic.endswith(tid) or tid.split("__")[-1] in ic:
                        out.append(tid)
                        break
            return sorted(set(out))

        train_ics = match(train_raw)
        eval_ics = match(eval_raw)
        # Drop overlap
        eval_ics = [t for t in eval_ics if t not in set(train_ics)]
    else:
        tids = sorted(lookup.keys())
        rng = np.random.default_rng(SEED)
        rng.shuffle(tids)
        n_ev = max(1, int(round(len(tids) * EVAL_FRAC)))
        eval_ics, train_ics = tids[:n_ev], tids[n_ev:]
    return {"lookup": lookup, "train_ics": train_ics, "eval_ics": eval_ics, "all_ics": sorted(lookup.keys())}


def _stratum(g: dict, feat: dict) -> str:
    """Privileged diagnostic strata (labels only)."""
    sw = bool(g["correct_sidewall"])
    bil = bool(g["bilateral"])
    opp = bil and sw
    contact = bool(g["has_pad_contact"])
    if not contact:
        return "no_contact"
    if opp:
        return "correct_opposing"
    if bil and not sw:
        return "incorrect_bilateral"
    if sw and not bil:
        return "correct_sidewall_unilateral"
    if not sw and contact:
        # ambiguous / incomplete
        if feat.get("contact_L") and feat.get("contact_R"):
            return "ambiguous_obs_bilateral_priv_wrong"
        return "incorrect_sidewall_unilateral"
    return "other"


def mine_snapshots(*, device: str = "cuda", max_ics: int | None = None, force: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    bank_path = OUT / "SNAP_BANK.json"
    if bank_path.exists() and not force:
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        print(f"[mine] cached bank n_snaps={sum(len(v) for v in bank.get('populations', {}).values() if isinstance(v, list))}", flush=True)
        return bank

    pools = _ic_pools()
    lookup = pools["lookup"]
    all_ics = pools["all_ics"]
    # Prefer covering train then eval; cap at available (≤68 typically)
    ordered = []
    for tid in pools["train_ics"] + pools["eval_ics"]:
        if tid not in ordered and tid in lookup:
            ordered.append(tid)
    for tid in all_ics:
        if tid not in ordered:
            ordered.append(tid)
    if max_ics is not None:
        ordered = ordered[:max_ics]
    else:
        ordered = ordered[: min(len(ordered), N_IC_TARGET)]

    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_ACTOR, device_t)
    critic = load_critic(P64_CRITIC, actor, device_t)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    v2 = v2_spec()

    natural: list[dict] = []
    by_ic: dict[str, list] = {}
    strata_counts: dict[str, int] = {}

    for ii, tid in enumerate(ordered):
        start = lookup[tid]
        seed = rollout_seed(split="train", traj_id=tid, rollout_i=79000 + ii)
        seed_all(seed, deterministic_torch=True)
        env.reset(start)
        ensure_inner_capture_buffers(env)
        st = WrapState()
        done = False
        fc_t = None
        captured_kinds = set()
        had_contact = False
        ever_sw = False
        while not done:
            tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device_t)
            tok, st, dbg = select_token(
                env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
            )
            _r, done, info = env.step(int(tok), resid)
            ensure_inner_capture_buffers(env)
            g = relative_geometry(env.env.sim)
            feat = obs_features(last_obs(env))
            t = int(env.t)
            if g["correct_sidewall"]:
                ever_sw = True
            if g["has_pad_contact"]:
                had_contact = True

            def capture(kind: str, *, population: str = "natural"):
                if kind in captured_kinds:
                    return
                if env.t + 4 > env.max_steps:
                    return
                # Actor ownership gate: after V2 pre-contact (saw_contact)
                if not st.saw_contact:
                    return
                state = take_full_state(
                    env,
                    meta={
                        "kind": kind,
                        "traj_id": tid,
                        "seed": seed,
                        "t": t,
                        "population": population,
                        "actor_owns": dbg.get("forced") is None,
                    },
                )
                sid = f"{kind}_{len(natural):04d}_{tid[-16:]}_t{t}"
                path = STATE_DIR / f"{sid}.pkl"
                path.write_bytes(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL))
                stratum = _stratum(g, feat)
                strata_counts[stratum] = strata_counts.get(stratum, 0) + 1
                meta = {
                    "id": sid,
                    "path": str(path),
                    "traj_id": tid,
                    "seed": int(seed),
                    "t": t,
                    "kind": kind,
                    "population": population,
                    "split": "eval" if tid in pools["eval_ics"] else "fit",
                    "e_pos_mm": float(g["e_pos_mm"]),
                    "correct_sidewall": int(g["correct_sidewall"]),
                    "bilateral": int(g["bilateral"]),
                    "opposing": int(bool(g["bilateral"] and g["correct_sidewall"])),
                    "min_pad": int(g["min_pad"]),
                    "has_pad_contact": int(g["has_pad_contact"]),
                    "nL": int(g["nL"]),
                    "nR": int(g["nR"]),
                    "stratum": stratum,
                    "obs_L": int(bool(feat.get("contact_L"))),
                    "obs_R": int(bool(feat.get("contact_R"))),
                    "actor_tok": int(tok_a),
                    "forced": dbg.get("forced"),
                    "phase": dbg.get("phase"),
                    "ever_sw_before": int(ever_sw),
                }
                natural.append(meta)
                by_ic.setdefault(tid, []).append(sid)
                captured_kinds.add(kind)

            if g["has_pad_contact"] and fc_t is None:
                fc_t = t
                capture("first_contact")
            # shortly after FC, still incomplete opposing
            if (
                fc_t is not None
                and "shortly_after_fc" not in captured_kinds
                and t >= fc_t + 2
                and not (g["bilateral"] and g["correct_sidewall"])
            ):
                capture("shortly_after_fc")
            # contact persistence incomplete opposing
            if (
                fc_t is not None
                and "persist_incomplete" not in captured_kinds
                and t >= fc_t + 4
                and g["has_pad_contact"]
                and not (g["bilateral"] and g["correct_sidewall"])
            ):
                capture("persist_incomplete")
            # approaching contact loss: had contact, now weak/unilateral with rising e_pos
            if (
                had_contact
                and "approach_loss" not in captured_kinds
                and g["has_pad_contact"]
                and g["min_pad"] <= 1
                and float(g["e_pos_mm"]) > 12.0
            ):
                capture("approach_loss")

        print(
            f"[mine] {ii+1}/{len(ordered)} {tid[-24:]} snaps={len(by_ic.get(tid, []))} contact={had_contact}",
            flush=True,
        )

    # Population A = all natural FC snaps (one primary per IC: first_contact preferred)
    primary = []
    for tid, sids in by_ic.items():
        metas = [m for m in natural if m["id"] in sids]
        fc = [m for m in metas if m["kind"] == "first_contact"]
        primary.append((fc or metas)[0])

    # Ambiguous subset for diagnostic enrichment *geometry*-based (not CF outcomes)
    ambiguous = [
        m
        for m in natural
        if m["stratum"]
        in ("ambiguous_obs_bilateral_priv_wrong", "incorrect_sidewall_unilateral", "incorrect_bilateral")
    ]

    bank = {
        "sampling": {
            "policy": "frozen P64 Actor + P70 V2 full-episode",
            "ic_source": "P54 C-skill schedule minus blocked_eval_ids; P75 train/eval overlap when available",
            "n_ics_available": len(all_ics),
            "n_ics_mined": len(by_ic),
            "n_train_pool": len(pools["train_ics"]),
            "n_eval_pool": len(pools["eval_ics"]),
            "target_was": N_IC_TARGET,
            "note": "Independent IC = unique traj_id. Multiple snaps per IC remain grouped.",
        },
        "populations": {
            "A_natural_fc_primary": primary,
            "A_natural_all": natural,
            "B_geometry_enriched_pre_cf": ambiguous,
        },
        "strata_counts": strata_counts,
        "by_ic": {k: v for k, v in by_ic.items()},
        "train_ics": pools["train_ics"],
        "eval_ics": pools["eval_ics"],
        "n_independent_ics": len(by_ic),
        "n_unique_snapshots": len(natural),
    }
    bank_path.write_text(json.dumps(bank, indent=2), encoding="utf-8")
    (OUT / "IC_AND_SNAPSHOT_COVERAGE.md").write_text(
        "# IC and Snapshot Coverage (P79 Phase 2)\n\n```json\n"
        + json.dumps(
            {
                "sampling": bank["sampling"],
                "n_independent_ics": bank["n_independent_ics"],
                "n_unique_snapshots": bank["n_unique_snapshots"],
                "strata_counts": strata_counts,
                "n_primary": len(primary),
                "n_geometry_enriched": len(ambiguous),
                "n_eval_ics_in_bank": len([t for t in by_ic if t in set(pools["eval_ics"])]),
                "n_fit_ics_in_bank": len([t for t in by_ic if t not in set(pools["eval_ics"])]),
            },
            indent=2,
        )
        + "\n```\n\n"
        "## Populations\n\n"
        "- **A (natural)**: first-contact / near-FC snaps from frozen P64+V2 rollouts; primary = one FC per IC.\n"
        "- **B (geometry-enriched, pre-CF)**: incorrect / ambiguous privileged geometry strata — "
        "selected **without** using counterfactual outcomes.\n"
        "- CF disagreement enrichment (if any) is reported separately after Phase 5 and is **not** "
        "the primary evaluation population.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "n_ics": bank["n_independent_ics"],
                "n_snaps": bank["n_unique_snapshots"],
                "strata": strata_counts,
            },
            indent=2,
        ),
        flush=True,
    )
    return bank


if __name__ == "__main__":
    mine_snapshots()
