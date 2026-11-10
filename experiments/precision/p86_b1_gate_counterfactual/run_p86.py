"""P86 — Frozen-Actor B1 HANDOFF gate counterfactual (diagnostic; no training)."""

from __future__ import annotations

import csv
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

from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    OUT as FINAL_OUT,
    P64_ACTOR,
    P64_CRITIC,
    T_MAX,
)
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import (  # noqa: E402
    StabilizeGrpoEnv,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import (  # noqa: E402
    rollout_seed,
    seed_all,
)
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (  # noqa: E402
    load_actor,
    load_critic,
)
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value  # noqa: E402
from experiments.precision.p66_c_curriculum_ppo.expert_drive import (  # noqa: E402
    ensure_inner_capture_buffers,
)
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import (  # noqa: E402
    evaluate_predicates,
)
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p69_skill_sequence.skill_families import family_of_token  # noqa: E402
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.controller import (  # noqa: E402
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal  # noqa: E402
from experiments.precision.p82_global_failure_budget.collect import _earliest_decisive  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import (  # noqa: E402
    GATE_SPECS,
    gate_constants,
    make_monitor,
)

OUT = ROOT / "results/experiments/p86_b1_gate_counterfactual"
PANEL_SEED = 86
N_B1 = 24
N_XCLASS = 8
SPLIT = "p86_train_gate_cf"
VARIANTS = ("G0", "G1", "G2", "G3")
XCLASS_VARIANTS = ("G0", "G3")  # baseline vs most-relaxed predefined


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def exposed_b1_ids() -> tuple[set[str], set[str]]:
    man = json.loads((FINAL_OUT / "FROZEN_TEST_MANIFEST.json").read_text(encoding="utf-8"))
    tids = {e["traj_id"] for e in man["episodes"] if e["skill"] == "B1"}
    rids = {e["root_id"] for e in man["episodes"] if e["skill"] == "B1"}
    return tids, rids


def select_panel() -> dict[str, list]:
    """Fixed training-split panel; excludes exposed independent B1 ICs."""
    sched = build_p54_dataset(seed=5478)["schedule"]
    exp_t, exp_r = exposed_b1_ids()
    by: dict[str, list] = {"A1": [], "B1": [], "C": [], "D": []}
    for s in sched:
        rid = root_id(s.traj_id)
        if s.traj_id in exp_t or rid in exp_r:
            continue
        by[s.skill].append(s)

    rng = np.random.default_rng(PANEL_SEED)

    def take(skill: str, n: int) -> list:
        pool = list(by[skill])
        # stable sort then sample without replacement
        pool.sort(key=lambda s: s.traj_id)
        if len(pool) <= n:
            return pool
        idx = rng.choice(len(pool), size=n, replace=False)
        idx = sorted(int(i) for i in idx)
        return [pool[i] for i in idx]

    panel = {
        "B1": take("B1", N_B1),
        "A1": take("A1", N_XCLASS),
        "C": take("C", N_XCLASS),
        "D": take("D", N_XCLASS),
    }
    meta = {
        "panel_seed": PANEL_SEED,
        "n_b1": len(panel["B1"]),
        "n_xclass": {k: len(panel[k]) for k in ("A1", "C", "D")},
        "excluded_exposed_b1": sorted(exp_t),
        "b1_traj_ids": [s.traj_id for s in panel["B1"]],
        "source": "p54 train schedule seed=5478",
        "note": "Diagnostic train-split panel only; not an independent final-test estimate.",
    }
    (OUT / "PANEL.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return panel


def ic_seed(traj_id: str) -> int:
    return rollout_seed(split=SPLIT, traj_id=traj_id, rollout_i=0)


def rollout_frozen(
    env: StabilizeGrpoEnv,
    actor,
    critic,
    book,
    start,
    device,
    *,
    seed: int,
    variant: str,
    record_reachability: bool = False,
) -> dict:
    seed_all(seed, deterministic_torch=True)
    env.reset(start)
    # Replace HANDOFF monitor AFTER reset (reset installs official R3F).
    env.mon = make_monitor(variant, t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)

    st = WrapState()
    v2 = v2_spec()
    done = False
    info: dict = {}
    ever = {
        k: False
        for k in (
            "fc",
            "sw",
            "bil",
            "opp",
            "pad4",
            "e9",
            "e12",
            "working",
            "frame",
            "frame9",
            "frame12",
            "r3",
            "loss",
        )
    }
    t_first = {
        "e9_bil_pad": None,
        "e12_bil_pad": None,
        "working_bil_pad": None,
        "g0_frame": None,
        "g1_frame": None,
        "g2_ready_instant": None,
        "g3_ready_instant": None,
    }
    min_e_pos = 1e9
    min_e_pos_with_bil_pad = 1e9
    had = False
    t_fc = t_opp = t_loss = None
    contact_loss_after_candidate = 0
    saw_candidate = False

    while not done:
        tok_a, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
        tok, st, _dbg = select_token(
            env_t=int(env.t), obs=last_obs(env), actor_tok=int(tok_a), st=st, v2=v2, rule=None
        )
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)

        g = relative_geometry(env.env.sim)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)

        e_pos = float(g["e_pos_mm"])
        theta = float(g["theta_deg"])
        bil = bool(g["bilateral"])
        pad4 = int(g.get("min_pad") or 0) >= 4
        min_e_pos = min(min_e_pos, e_pos)
        if bil and pad4:
            min_e_pos_with_bil_pad = min(min_e_pos_with_bil_pad, e_pos)

        if g["has_pad_contact"]:
            if not ever["fc"]:
                t_fc = int(env.t)
            ever["fc"] = True
            had = True
        elif had and not g["has_pad_contact"]:
            ever["loss"] = True
            if t_loss is None:
                t_loss = int(env.t)
            if saw_candidate:
                contact_loss_after_candidate = 1

        if g.get("correct_sidewall"):
            ever["sw"] = True
        if bil:
            ever["bil"] = True
        if bil and g.get("correct_sidewall"):
            if not ever["opp"]:
                t_opp = int(env.t)
            ever["opp"] = True
        if pad4:
            ever["pad4"] = True
        if e_pos <= 9:
            ever["e9"] = True
        if e_pos <= 12:
            ever["e12"] = True
        if preds.get("working_ok"):
            ever["working"] = True
        if preds.get("frame_ok"):
            ever["frame"] = True
        if (preds.get("dwell_complete") and preds.get("frame_ok")) or int(info.get("handoff") or 0):
            ever["r3"] = True

        # Candidate gate geometry (instantaneous; dwell separate)
        if bil and pad4 and e_pos <= 9 + 1e-9 and theta <= 8 + 1e-9 and preds.get("working_ok"):
            ever["frame9"] = True
            if t_first["g0_frame"] is None:
                t_first["g0_frame"] = int(env.t)
            if t_first["e9_bil_pad"] is None:
                t_first["e9_bil_pad"] = int(env.t)
            saw_candidate = True
        if bil and pad4 and e_pos <= 12 + 1e-9 and theta <= 8 + 1e-9 and preds.get("working_ok"):
            ever["frame12"] = True
            if t_first["g1_frame"] is None:
                t_first["g1_frame"] = int(env.t)
            if t_first["e12_bil_pad"] is None:
                t_first["e12_bil_pad"] = int(env.t)
            saw_candidate = True
        if bil and pad4 and preds.get("working_ok"):
            if t_first["working_bil_pad"] is None:
                t_first["working_bil_pad"] = int(env.t)
        if ever["frame9"] and t_first["g2_ready_instant"] is None:
            t_first["g2_ready_instant"] = t_first["g0_frame"]
        if ever["frame12"] and t_first["g3_ready_instant"] is None:
            t_first["g3_ready_instant"] = t_first["g1_frame"]

    term = _r3_terminal(env, info)
    timed_out = bool(int(env.t) >= int(env.max_steps) - 1) and not term["SUCCESS_RELAXED"]
    decisive = _earliest_decisive(ever, term, timed_out)

    out = {
        "skill": start.skill,
        "traj_id": start.traj_id,
        "root_id": root_id(start.traj_id),
        "seed": seed,
        "variant": variant,
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
        "SUCCESS_NEW": int(info.get("SUCCESS_NEW") or 0),
        "handoff": int(term.get("handoff") or info.get("handoff") or 0),
        "handoff_reason": info.get("handoff_reason"),
        "r3_ready": int(term.get("r3_ready") or 0),
        "ep_len": int(env.t),
        "earliest_decisive_failure": decisive if not term.get("SUCCESS_RELAXED") else "success",
        "first_contact": int(ever["fc"]),
        "opposing_ever": int(ever["opp"]),
        "bilateral_ever": int(ever["bil"]),
        "min_pad_ge4": int(ever["pad4"]),
        "e_pos_le9_ever": int(ever["e9"]),
        "e_pos_le12_ever": int(ever["e12"]),
        "working_ever": int(ever["working"]),
        "frame9_ever": int(ever["frame9"]),
        "frame12_ever": int(ever["frame12"]),
        "contact_loss_ever": int(ever["loss"]),
        "contact_loss_after_candidate": int(contact_loss_after_candidate),
        "min_e_pos_mm": float(min_e_pos if min_e_pos < 1e8 else -1),
        "min_e_pos_mm_bil_pad": float(min_e_pos_with_bil_pad if min_e_pos_with_bil_pad < 1e8 else -1),
        "t_fc": t_fc,
        "t_opp": t_opp,
        "t_loss": t_loss,
        "timeout": int(timed_out),
        "fail_reason": info.get("fail_reason"),
    }
    if record_reachability:
        out.update({f"t_first_{k}": v for k, v in t_first.items()})
        out["reachable_g0_geometry"] = int(ever["frame9"])
        out["reachable_g1_geometry"] = int(ever["frame12"])
        out["reachable_g3_geometry"] = int(ever["frame12"])  # same instant geom; dwell differs
        out["actor_never_bil_pad"] = int(not (ever["bil"] and ever["pad4"]))
    return out


def classify_attribution(g0: dict, by_var: dict[str, dict]) -> dict:
    """Paired IC attribution across G0–G3."""
    g0_ok = int(g0.get("SUCCESS_RELAXED_12S") or 0)
    if g0_ok:
        label = "ALREADY_SUCCESSFUL"
    else:
        rescued_by = [
            v
            for v in ("G1", "G2", "G3")
            if int(by_var[v].get("SUCCESS_RELAXED_12S") or 0) == 1
        ]
        reached = int(g0.get("frame12_ever") or g0.get("frame9_ever") or 0)
        # Also consider if a relaxed gate actually fired (handoff) but failed physically
        fired_fail = [
            v
            for v in ("G1", "G2", "G3")
            if int(by_var[v].get("handoff") or 0) == 1
            and int(by_var[v].get("SUCCESS_RELAXED_12S") or 0) == 0
            and int(by_var[v].get("SUCCESS_12S") or 0) == 0
        ]
        if rescued_by:
            label = "GATE_RESCUED"
        elif not reached and int(g0.get("min_e_pos_mm_bil_pad") or 1e9) > 12.0:
            label = "ACTOR_NOT_REACHABLE"
        elif fired_fail:
            label = "GATE_REACHED_BUT_FAILED"
        elif not reached:
            label = "ACTOR_NOT_REACHABLE"
        else:
            # Reached geometry under G0 path but no relaxed variant succeeded
            # (dwell never completed / timeout) → still actor timing or GATE_REACHED_BUT_FAILED
            if int(g0.get("frame9_ever") or 0) or int(g0.get("frame12_ever") or 0):
                label = "GATE_REACHED_BUT_FAILED"
            else:
                label = "INDETERMINATE"

    return {
        "traj_id": g0["traj_id"],
        "root_id": g0["root_id"],
        "attribution": label,
        "G0_SUCCESS_RELAXED_12S": int(g0.get("SUCCESS_RELAXED_12S") or 0),
        "G0_SUCCESS_12S": int(g0.get("SUCCESS_12S") or 0),
        "G1_SUCCESS_RELAXED_12S": int(by_var["G1"].get("SUCCESS_RELAXED_12S") or 0),
        "G2_SUCCESS_RELAXED_12S": int(by_var["G2"].get("SUCCESS_RELAXED_12S") or 0),
        "G3_SUCCESS_RELAXED_12S": int(by_var["G3"].get("SUCCESS_RELAXED_12S") or 0),
        "G1_SUCCESS_12S": int(by_var["G1"].get("SUCCESS_12S") or 0),
        "G2_SUCCESS_12S": int(by_var["G2"].get("SUCCESS_12S") or 0),
        "G3_SUCCESS_12S": int(by_var["G3"].get("SUCCESS_12S") or 0),
        "G0_handoff": int(g0.get("handoff") or 0),
        "G3_handoff": int(by_var["G3"].get("handoff") or 0),
        "reachable_frame9": int(g0.get("frame9_ever") or 0),
        "reachable_frame12": int(g0.get("frame12_ever") or 0),
        "min_e_pos_mm_bil_pad": g0.get("min_e_pos_mm_bil_pad"),
        "G0_fail": g0.get("earliest_decisive_failure"),
        "G3_fail": by_var["G3"].get("earliest_decisive_failure"),
    }


def write_protocol(panel: dict) -> None:
    gc = gate_constants()
    text = f"""# PROTOCOL — P86 Frozen-Actor B1 HANDOFF Gate Counterfactual

Generated: {_utc()}

## Objective

Causal attribution: overly restrictive HANDOFF gate vs Actor never reaching recoverable states.

## Fixed constraints

- Frozen P64 Actor/Critic: `{P64_ACTOR.name}` / `{P64_CRITIC.name}`
- No BC / GRPO / PPO / reward / overlays / physics / M111 / nominal changes
- Official R3F implementation **unchanged**; ablations use a parallel diagnostic monitor
- Training-split ICs only (P54 schedule); **excludes** eight exposed independent B1 ICs
- Matched inference seeds via `rollout_seed(split={SPLIT}, traj_id, 0)`
- Keep all failures; no cherry-picking

## Panel

| Skill | n | Role |
|---|---:|---|
| B1 | {len(panel['B1'])} | Gate reachability + G0–G3 ablation |
| A1 | {len(panel['A1'])} | Cross-class G0 vs G3 |
| C | {len(panel['C'])} | Cross-class G0 vs G3 |
| D | {len(panel['D'])} | Cross-class G0 vs G3 |

See `PANEL.json`.

## Gate variants

| Variant | Pos threshold | Dwell | Monitor |
|---|---|---|---|
| G0 | 9 mm (official R3) | HOLD_S={gc['HOLD_S']} | Official `R3FOnline` |
| G1 | 12 mm | HOLD_S | Diagnostic ablation |
| G2 | 9 mm | Removed (0) | Diagnostic ablation |
| G3 | 12 mm | Removed (0) | Diagnostic ablation |

Preserved on all variants: WORKING (θ≤{gc['WORKING_THETA_DEG']}°, e_pos≤{gc['WORKING_POS_MM']} mm), bilateral, min_pad≥{gc['MIN_PAD_CONTACTS']}, θ margin ≤{gc['R3_THETA_DEG']}°.

K48 HANDOFF states with e_pos>12 mm remain **excluded by G3** (reported separately from P85).

## Success metrics

Both **SUCCESS_12S** and **SUCCESS_RELAXED_12S** via unchanged env terminal evaluation after HANDOFF/timeout.
Reaching the gate alone is **not** success.

## Evidence class

All P86 scores are **train-split diagnostic** — not a replacement for the frozen independent P64 B1 0/8.
"""
    _write(OUT / "PROTOCOL.md", text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    panel = select_panel()
    write_protocol(panel)

    actor, book, rho, _ = load_actor(P64_ACTOR, device)
    critic = load_critic(P64_CRITIC, actor, device)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )

    # ----- Phase 1+2+3: B1 paired ablations -----
    reach_rows = []
    abl_rows = []
    by_ic: dict[str, dict[str, dict]] = {}

    for i, start in enumerate(panel["B1"]):
        seed = ic_seed(start.traj_id)
        by_ic[start.traj_id] = {}
        for variant in VARIANTS:
            print(f"[B1 {i+1}/{len(panel['B1'])}] {variant} {start.traj_id[-32:]}", flush=True)
            rec = rollout_frozen(
                env,
                actor,
                critic,
                book,
                start,
                device,
                seed=seed,
                variant=variant,
                record_reachability=(variant == "G0"),
            )
            by_ic[start.traj_id][variant] = rec
            abl_rows.append(rec)
            if variant == "G0":
                reach_rows.append(rec)
            print(
                f"  relaxed={rec['SUCCESS_RELAXED_12S']} s12={rec['SUCCESS_12S']} "
                f"hand={rec['handoff']} fail={rec['earliest_decisive_failure']} "
                f"min_e_bil={rec['min_e_pos_mm_bil_pad']:.2f} frame9={rec.get('frame9_ever')} "
                f"frame12={rec.get('frame12_ever')}",
                flush=True,
            )

    _write_csv(OUT / "GATE_REACHABILITY.csv", reach_rows)
    _write_csv(OUT / "PAIRED_GATE_ABLATION.csv", abl_rows)

    attr_rows = []
    for tid, d in by_ic.items():
        attr_rows.append(classify_attribution(d["G0"], d))
    _write_csv(OUT / "ATTRIBUTION.csv", attr_rows)

    # ----- Phase 5: cross-class -----
    x_rows = []
    for skill in ("A1", "C", "D"):
        for j, start in enumerate(panel[skill]):
            seed = ic_seed(start.traj_id)
            for variant in XCLASS_VARIANTS:
                print(f"[{skill} {j+1}/{len(panel[skill])}] {variant} {start.traj_id[-28:]}", flush=True)
                rec = rollout_frozen(
                    env, actor, critic, book, start, device, seed=seed, variant=variant, record_reachability=False
                )
                x_rows.append(rec)
                print(
                    f"  relaxed={rec['SUCCESS_RELAXED_12S']} s12={rec['SUCCESS_12S']} hand={rec['handoff']}",
                    flush=True,
                )
    _write_csv(OUT / "CROSS_CLASS_ABLATION.csv", x_rows)

    # ----- Reports -----
    attr_counts = Counter(r["attribution"] for r in attr_rows)
    n = len(attr_rows)
    g0_rel = sum(int(r["G0_SUCCESS_RELAXED_12S"]) for r in attr_rows)
    g3_rel = sum(int(r["G3_SUCCESS_RELAXED_12S"]) for r in attr_rows)
    g1_rel = sum(int(r["G1_SUCCESS_RELAXED_12S"]) for r in attr_rows)
    g2_rel = sum(int(r["G2_SUCCESS_RELAXED_12S"]) for r in attr_rows)
    g0_s12 = sum(int(r["G0_SUCCESS_12S"]) for r in attr_rows)
    g3_s12 = sum(int(r["G3_SUCCESS_12S"]) for r in attr_rows)
    rescued = attr_counts.get("GATE_RESCUED", 0)
    not_reach = attr_counts.get("ACTOR_NOT_REACHABLE", 0)
    reached_fail = attr_counts.get("GATE_REACHED_BUT_FAILED", 0)
    already = attr_counts.get("ALREADY_SUCCESSFUL", 0)

    reach_n9 = sum(int(r.get("frame9_ever") or 0) for r in reach_rows)
    reach_n12 = sum(int(r.get("frame12_ever") or 0) for r in reach_rows)
    never_bil = sum(int(r.get("actor_never_bil_pad") or 0) for r in reach_rows)

    # Cross-class deltas
    x_summary = {}
    for skill in ("A1", "C", "D"):
        g0s = [r for r in x_rows if r["skill"] == skill and r["variant"] == "G0"]
        g3s = [r for r in x_rows if r["skill"] == skill and r["variant"] == "G3"]
        by0 = {r["traj_id"]: r for r in g0s}
        lost = gained = 0
        for r3 in g3s:
            a = by0[r3["traj_id"]]
            if int(a["SUCCESS_RELAXED_12S"]) and not int(r3["SUCCESS_RELAXED_12S"]):
                lost += 1
            if (not int(a["SUCCESS_RELAXED_12S"])) and int(r3["SUCCESS_RELAXED_12S"]):
                gained += 1
        x_summary[skill] = {
            "n": len(g0s),
            "G0_relaxed": sum(int(r["SUCCESS_RELAXED_12S"]) for r in g0s),
            "G3_relaxed": sum(int(r["SUCCESS_RELAXED_12S"]) for r in g3s),
            "G0_s12": sum(int(r["SUCCESS_12S"]) for r in g0s),
            "G3_s12": sum(int(r["SUCCESS_12S"]) for r in g3s),
            "lost_success_G0_to_G3": lost,
            "gained_success_G0_to_G3": gained,
        }

    # K48 >12mm note from P85
    p85 = ROOT / "results/experiments/p85_k48_b1_skill_retention/trajectories"
    k48_above_12 = []
    if p85.is_dir():
        for p in p85.glob("*.json"):
            d = json.loads(p.read_text(encoding="utf-8"))
            e = (d.get("handoff_preds") or {}).get("e_pos")
            if e is not None and float(e) > 12.0 + 1e-9:
                k48_above_12.append({"traj_id": d.get("traj_id"), "e_pos": e})

    # CASE decision
    cross_regressed = any(x_summary[sk]["lost_success_G0_to_G3"] > 0 for sk in x_summary)
    gate_helps = rescued >= 1 and (g3_rel > g0_rel or g1_rel > g0_rel or g2_rel > g0_rel)
    actor_primary = not_reach >= max(1, n // 2) and rescued == 0
    both = rescued >= 1 and not_reach >= 1

    if gate_helps and not cross_regressed and not_reach == 0:
        case, title = "A", "Gate relaxation physically rescues B1 without observed cross-class regression"
    elif gate_helps and cross_regressed:
        case, title = "B", "Gate relaxation rescues B1 but causes cross-class regression"
    elif actor_primary:
        case, title = "C", "P64 rarely reaches suitable HANDOFF states; Actor is primary limitation"
    elif both or (gate_helps and not_reach >= 1) or (rescued >= 1 and reached_fail >= 1):
        case, title = "D", "Gate and Actor limitations both remain, with quantified contributions"
    elif rescued == 0 and not_reach >= 1:
        case, title = "C", "P64 rarely reaches suitable HANDOFF states; Actor is primary limitation"
    else:
        case, title = "E", "Inconclusive"

    # Prefer C if almost never reachable and no rescue
    if reach_n12 == 0 and rescued == 0:
        case, title = "C", "P64 rarely reaches suitable HANDOFF states; Actor is primary limitation"
    elif rescued >= 1 and not_reach >= 1:
        case, title = "D", "Gate and Actor limitations both remain, with quantified contributions"
    elif rescued >= 1 and cross_regressed:
        case, title = "B", "Gate relaxation rescues B1 but causes cross-class regression"
    elif rescued >= 1 and not cross_regressed and not_reach == 0:
        case, title = "A", "Gate relaxation physically rescues B1 without observed cross-class regression"

    attr_md = f"""# B1_CAUSAL_ATTRIBUTION

Generated: {_utc()}

## Verdict driver: **CASE {case}** — {title}

### Train-split B1 panel (n={n}; diagnostic only)

| Variant | SUCCESS_RELAXED_12S | SUCCESS_12S |
|---|---:|---:|
| G0 (official R3F) | {g0_rel}/{n} | {g0_s12}/{n} |
| G1 (12 mm + dwell) | {g1_rel}/{n} | {sum(int(r['G1_SUCCESS_12S']) for r in attr_rows)}/{n} |
| G2 (9 mm, no dwell) | {g2_rel}/{n} | {sum(int(r['G2_SUCCESS_12S']) for r in attr_rows)}/{n} |
| G3 (12 mm, no dwell) | {g3_rel}/{n} | {g3_s12}/{n} |

### Attribution counts

| Label | n |
|---|---:|
| GATE_RESCUED | {rescued} |
| GATE_REACHED_BUT_FAILED | {reached_fail} |
| ACTOR_NOT_REACHABLE | {not_reach} |
| ALREADY_SUCCESSFUL | {already} |
| INDETERMINATE | {attr_counts.get('INDETERMINATE', 0)} |

### Reachability under frozen Actor + G0 path

| Signal | n/{n} |
|---|---:|
| Ever R3-like frame (≤9 mm + bil + pad4 + WORKING + θ≤8) | {reach_n9} |
| Ever G1/G3 geometry (≤12 mm + …) | {reach_n12} |
| Never bilateral+min_pad≥4 | {never_bil} |

### K48 HANDOFF states still excluded by G3 (from P85; e_pos>12)

n={len(k48_above_12)}: {k48_above_12 or 'none among the eight diagnostic heldout ICs'}

### Per-IC

| traj_id | attribution | G0 rel/s12 | G3 rel/s12 | min_e_bil | frame9 | frame12 |
|---|---|---|---|---:|---:|---:|
"""
    for r in attr_rows:
        attr_md += (
            f"| `{r['traj_id']}` | {r['attribution']} | "
            f"{r['G0_SUCCESS_RELAXED_12S']}/{r['G0_SUCCESS_12S']} | "
            f"{r['G3_SUCCESS_RELAXED_12S']}/{r['G3_SUCCESS_12S']} | "
            f"{r['min_e_pos_mm_bil_pad']} | {r['reachable_frame9']} | {r['reachable_frame12']} |\n"
        )
    _write(OUT / "B1_CAUSAL_ATTRIBUTION.md", attr_md)

    x_md = f"""# CROSS_CLASS_REGRESSION

Generated: {_utc()}

Shared HANDOFF logic — G0 (official) vs G3 (most relaxed predefined) on training-split A1/C/D.

| Skill | n | G0 RELAXED | G3 RELAXED | G0 12S | G3 12S | Lost G0→G3 | Gained G0→G3 |
|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for sk, s in x_summary.items():
        x_md += (
            f"| {sk} | {s['n']} | {s['G0_relaxed']}/{s['n']} | {s['G3_relaxed']}/{s['n']} | "
            f"{s['G0_s12']}/{s['n']} | {s['G3_s12']}/{s['n']} | {s['lost_success_G0_to_G3']} | "
            f"{s['gained_success_G0_to_G3']} |\n"
        )
    x_md += f"""

Cross-class regression observed (lost successes): **{"yes" if cross_regressed else "no"}**.

No class-specific runtime routing was introduced.
"""
    _write(OUT / "CROSS_CLASS_REGRESSION.md", x_md)

    final = f"""# FINAL_REPORT — P86 B1 HANDOFF Gate Counterfactual

Generated: {_utc()}

## CASE {case} — {title}

### What was tested

Frozen P64 Actor/Critic on a **preselected training-split** panel ({n} B1 + {N_XCLASS}×A1/C/D), matched seeds, gate variants G0–G3. Official R3F code path used for G0; parallel diagnostic monitors for G1–G3. Exposed independent B1 ICs **excluded**. Independent frozen **0/8** unchanged.

### Key numbers (B1 train panel)

| | |
|---|---|
| G0 SUCCESS_RELAXED_12S | **{g0_rel}/{n}** |
| G3 SUCCESS_RELAXED_12S | **{g3_rel}/{n}** |
| GATE_RESCUED | **{rescued}** |
| ACTOR_NOT_REACHABLE | **{not_reach}** |
| Ever ≤12 mm candidate geometry under Actor | **{reach_n12}/{n}** |
| Ever ≤9 mm candidate geometry under Actor | **{reach_n9}/{n}** |
| Cross-class lost successes (G0→G3) | **{"yes" if cross_regressed else "no"}** |

SUCCESS_12S tracked in parallel (G0 {g0_s12}/{n}, G3 {g3_s12}/{n}); not treated as interchangeable with RELAXED.

### Interpretation

"""
    if case == "C":
        final += (
            "The frozen Actor rarely (or never) enters a state that any predefined gate can accept. "
            "Relaxing the HANDOFF threshold cannot rescue B1 if the policy does not reach bil+pad+WORKING "
            "pose bands comparable to K48’s successful handoffs. **Next work should target Actor behavior "
            "(e.g. train-split K48 teacher retention), not gate deployment.**\n"
        )
    elif case == "A":
        final += (
            "Relaxing the gate rescues some B1 episodes without observed A1/C/D success loss on this panel. "
            "Still do **not** auto-deploy; validate on a larger train-split panel and keep independent eval frozen.\n"
        )
    elif case == "B":
        final += (
            "Gate relaxation helps some B1 episodes but regresses other classes under shared HANDOFF logic. "
            "Do not deploy a global weaker gate; any follow-up must be class-agnostic and regression-aware.\n"
        )
    elif case == "D":
        final += (
            f"Both mechanisms appear: gate rescue on {rescued}/{n}, unreachable Actor states on {not_reach}/{n}. "
            "Quantified in `B1_CAUSAL_ATTRIBUTION.md`. A single global gate change is insufficient; "
            "Actor retention remains necessary for the unreachable subset.\n"
        )
    else:
        final += "Results do not support a clean causal claim; see CSVs for paired trajectories.\n"

    final += f"""
### Non-actions (honored)

- No BC / checkpoint swap / reward change / official R3F edit
- No training or selection on the eight exposed B1 eval ICs
- No automatic final-model modification

### Artifacts

- `PROTOCOL.md`
- `GATE_REACHABILITY.csv`
- `PAIRED_GATE_ABLATION.csv`
- `B1_CAUSAL_ATTRIBUTION.md`
- `CROSS_CLASS_REGRESSION.md`
- `FINAL_REPORT.md`
- `PANEL.json`, `ATTRIBUTION.csv`, `CROSS_CLASS_ABLATION.csv`
"""
    _write(OUT / "FINAL_REPORT.md", final)

    summary = {
        "case": case,
        "title": title,
        "b1_n": n,
        "G0_relaxed": g0_rel,
        "G3_relaxed": g3_rel,
        "attribution": dict(attr_counts),
        "reach_frame9": reach_n9,
        "reach_frame12": reach_n12,
        "cross_regressed": cross_regressed,
        "x_summary": x_summary,
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
