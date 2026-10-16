"""Policy-guided hard IC mining for unified recovery GRPO.

Does not train GRPO or modify SFT/physics/codebook/reward. Writes under
results/diagnostics/hard_ic_mining/.

Split rule is frozen at process start (hash of scenario fingerprint) before
any selection by SFT/oracle outcomes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.ball_impact_dataset import run_episode
from data_generation.expand_skill_quotas import _fingerprint, _try_sliding
from data_generation.rebuild_k48_recovery import (
    _known_specs,
    _reserved_keys,
    _scenario_key,
    immediate_handoff,
)
from data_generation.strategies.vertical_recatch_search import (
    _finish,
    _summary,
    rollout as vertical_rollout,
    specs as vertical_specs,
)
from data_generation.validation.impact_labels import cylinder_first, near
from envs.outcomes.horizon_outcome import roll_policy
from models.k48_checkpoint import load_trained_k48
from training.k47_codebook import load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from evaluation import skill_traces as audit
from evaluation.classify_validation_skills import classify_mechanism
from experiments.sliding.diagnose_sft_sliding_root_cause import free_rollout
from visualization.compare_policy_nominal import open_camera, pair, shoot, stitch
from visualization.video_recorder import write_frames_mp4

OUT = ROOT / "results" / "diagnostics" / "hard_ic_mining"
SFT_CKPT = ROOT / "results" / "sft" / "sliding_equal_weight" / "checkpoint_epoch_0040.pt"
HARD_DRAW_START = 910_000
STOCH_SEEDS = (11, 22, 33)
# Frozen before outcome-based selection (documented in split_rule.json).
SPLIT_TRAIN_MAX = 69   # hash%100 in [0, 69] -> train
SPLIT_DEV_MAX = 84     # [70, 84] -> dev; [85, 99] -> test


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=float) + "\n")


def write_split_rule() -> dict:
    rule = {
        "frozen_at": "process_start_before_selection",
        "method": "sha1(fingerprint)[:8] as int hex -> mod 100",
        "train": f"0..{SPLIT_TRAIN_MAX}",
        "dev": f"{SPLIT_TRAIN_MAX + 1}..{SPLIT_DEV_MAX}",
        "test": f"{SPLIT_DEV_MAX + 1}..99",
        "grouping": "same scenario_key always maps together via fingerprint of blob",
        "historical_valid_17": "data/valid_skill_quotas.jsonl kept as regression only; never mixed into new GRPO train",
        "note": "Outcomes do not reassign splits. Selection filters which ICs enter each manifest, but assigned split is fixed by hash.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "split_rule.json").write_text(json.dumps(rule, indent=2), encoding="utf-8")
    return rule


def assigned_split(fingerprint: str) -> str:
    bucket = int(hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket <= SPLIT_TRAIN_MAX:
        return "train"
    if bucket <= SPLIT_DEV_MAX:
        return "dev"
    return "test"


def draw_hard_spec(draw: int) -> dict:
    """Harder-biased launch than rebuild _draw_spec; still legal continuous ball impact."""
    rng = np.random.default_rng(HARD_DRAW_START + int(draw))
    roll = float(rng.random())
    if roll < 0.55:
        speed = float(rng.uniform(9.2, 10.2))
    elif roll < 0.85:
        speed = float(rng.uniform(8.4, 9.2))
    else:
        speed = float(rng.uniform(7.0, 8.4))
    if float(rng.random()) < 0.4:
        z_off = float(rng.uniform(-0.006, 0.006))
    else:
        z_off = float(rng.uniform(-0.022, 0.022))
    if float(rng.random()) < 0.45:
        phi = float(rng.uniform(-22.0, 22.0))
    else:
        phi = float(rng.choice([-1.0, 1.0]) * rng.uniform(22.0, 55.0))
    return {
        "azimuth_deg": float(rng.uniform(0.0, 360.0)),
        "speed_mps": speed,
        "z_off_m": z_off,
        "phi_deg": phi,
        "delay_s": float(rng.uniform(0.0, 0.45)),
        "direction": "continuous",
        "family": "ballistic_impact",
        "protocol": "k48_hard_ic_mining_v1",
        "draw": int(draw),
    }


def load_known_fingerprints() -> set[str]:
    known = set()
    for folder in (ROOT / "data" / "snapshots" / "train", ROOT / "data" / "snapshots" / "valid"):
        if not folder.is_dir():
            continue
        for path in folder.glob("*.pkl"):
            try:
                blob = pickle.loads(path.read_bytes())
                if "capture" in blob or "qpos" in blob:
                    known.add(_fingerprint(blob))
            except Exception:
                continue
    return known


def load_valid_ids() -> set[str]:
    path = ROOT / "data" / "valid_skill_quotas.jsonl"
    if not path.is_file():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def generate_candidates(env, attempts: int, known_fp: set[str], seed_offset: int) -> list[dict]:
    snap_dir = OUT / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    cand_path = OUT / "candidate_manifest.jsonl"
    if cand_path.is_file():
        cand_path.unlink()
    specs = _known_specs()
    reserved = _reserved_keys()
    candidates = []
    draw = seed_offset
    ran = 0
    t0 = time.perf_counter()
    while ran < attempts:
        spec = draw_hard_spec(draw)
        draw += 1
        key = _scenario_key(spec)
        if any(near(spec, item) for item in specs) or key in reserved:
            append_jsonl(OUT / "generation_log.jsonl", {
                "event": "near_duplicate_launch", "draw": draw - 1, **{k: spec[k] for k in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s")},
            })
            ran += 1
            continue
        eid = f"hard_{draw - 1:06d}"
        spec["split"] = "mine"
        spec["attempt_id"] = eid
        print("GEN", eid, f"speed={spec['speed_mps']:.2f}", f"phi={spec['phi_deg']:.1f}", flush=True)
        evidence, blob = run_episode(env, spec)
        evidence, blob = cylinder_first(evidence, blob)
        ran += 1
        row = {
            "id": eid,
            "draw": draw - 1,
            "category": evidence.get("category"),
            "reason": evidence.get("reason"),
            "states": evidence.get("states"),
            **{k: spec[k] for k in ("azimuth_deg", "speed_mps", "z_off_m", "phi_deg", "delay_s")},
            "scenario_key": list(key),
        }
        if blob is None or evidence.get("category") != "valid_impact_recovery_ic":
            append_jsonl(OUT / "generation_log.jsonl", {"event": "reject", **row})
            continue
        nominal = immediate_handoff(env, blob)
        row["nominal_class"] = nominal["nominal_class"]
        row["nominal_reason"] = nominal["reason"]
        if nominal["nominal_class"] != "RECOVERY_CANDIDATE":
            append_jsonl(OUT / "generation_log.jsonl", {"event": "nominal_not_candidate", **row})
            continue
        fp = _fingerprint(blob)
        row["fingerprint"] = fp
        if fp in known_fp:
            append_jsonl(OUT / "generation_log.jsonl", {"event": "fingerprint_duplicate", **row})
            continue
        known_fp.add(fp)
        blob["spec"]["id"] = eid
        blob["spec"]["protocol"] = "k48_hard_ic_mining_v1"
        rel = f"results/diagnostics/hard_ic_mining/snapshots/{eid}.pkl"
        path = ROOT / rel
        with path.open("wb") as handle:
            pickle.dump(blob, handle)
        row["snapshot"] = rel.replace("\\", "/")
        row["assigned_split"] = assigned_split(fp)
        row["pool"] = "recovery"
        append_jsonl(cand_path, row)
        candidates.append(row)
        print("CAND", eid, "split", row["assigned_split"], "states", row.get("states"), flush=True)
    print("GEN_DONE", "attempts", ran, "candidates", len(candidates), "sec", round(time.perf_counter() - t0, 1), flush=True)
    return candidates


def screen_one(env, model, book, device, row: dict) -> dict:
    blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
    # Deterministic SFT
    det = roll_policy(env, model, book, blob, device, cap_steps=200, limit_s=SELECTED_LIMIT_S)
    traced = free_rollout(env, model, book, blob, device, SELECTED_LIMIT_S, stochastic=False)
    mech = classify_mechanism(traced["samples"], traced["t0"], traced["handoff_t"])
    stoch = []
    for seed in STOCH_SEEDS:
        run = free_rollout(env, model, book, blob, device, SELECTED_LIMIT_S, stochastic=True, seed=seed)
        sk = classify_mechanism(run["samples"], run["t0"], run["handoff_t"])
        stoch.append({
            "seed": seed,
            "success": bool(run["outcome"].get("success")),
            "reason": run["outcome"].get("reason"),
            "mechanism": sk["mechanism"],
            "first_token": run["first_token"],
        })
    stoch_rate = float(np.mean([item["success"] for item in stoch])) if stoch else 0.0
    # Lightweight heuristic: sliding then a few vertical specs (contact mechanism from SFT det)
    return {
        **row,
        "sft_det_success": bool(det.get("success")),
        "sft_det_reason": det.get("reason"),
        "sft_det_recovery_s": det.get("recovery_elapsed"),
        "sft_det_mechanism": mech["mechanism"],
        "sft_det_first_token": traced.get("first_token"),
        "sft_stoch_successes": int(sum(1 for item in stoch if item["success"])),
        "sft_stoch_trials": len(stoch),
        "sft_stoch_rate": stoch_rate,
        "sft_stoch": json.dumps(stoch),
        "table_drop": det.get("reason") == "TABLE_DROP" or any(item["reason"] == "TABLE_DROP" for item in stoch),
        "hard_by_sft": (not bool(det.get("success"))) or stoch_rate < (2.0 / 3.0),
    }


def oracle_search(env, book, row: dict) -> dict:
    blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
    attempts = 0
    env_steps = 0
    best = None
    strategies = []

    # Sliding family
    attempts += 1
    slide = _try_sliding(env, book, blob)
    strategies.append({"name": "sliding_bundle", "ok": bool(slide and slide.get("ok")), "reason": None if not slide else slide.get("reason")})
    if slide and slide.get("ok"):
        best = {
            "ok": True,
            "strategy": slide.get("oracle_strategy") or "sliding",
            "schedule": slide.get("schedule"),
            "reason": "SUCCESS_12S",
            "mechanism_tag": "Contact-Preserving Adjustment",
            "recovery_elapsed": slide.get("recovery_elapsed"),
        }
        # estimate steps from recovery time
        if slide.get("recovery_elapsed") is not None:
            env_steps += int(round(float(slide["recovery_elapsed"]) / 0.02))

    # Vertical recatch — try the three legal chase specs
    if best is None:
        for spec in list(vertical_specs()):
            attempts += 1
            result = vertical_rollout(env, blob, book, spec)
            hold, _extra = _finish(env.sim, blob["ref"])
            summary = _summary(result)
            ok = bool(hold.get("success"))
            reason = hold.get("reason")
            strategies.append({
                "name": f"vertical:{spec.get('name')}",
                "ok": ok,
                "reason": reason,
                "t_off": summary.get("t_off"),
                "t_re": summary.get("t_re"),
            })
            env_steps += max(0, len(result.get("rows") or []))
            if ok:
                best = {
                    "ok": True,
                    "strategy": f"vertical_recatch:{spec.get('name')}",
                    "reason": "SUCCESS_12S",
                    "mechanism_tag": "Release-Recatch",
                    "recovery_elapsed": hold.get("recovery_elapsed"),
                }
                break

    # Stochastic SFT as recoverability evidence (already screened); count as search if det failed
    if best is None and row.get("sft_stoch_successes", 0) > 0:
        best = {
            "ok": True,
            "strategy": "sft_stochastic",
            "reason": "SUCCESS_12S",
            "mechanism_tag": "Unknown-from-stoch",
            "recovery_elapsed": None,
        }
        strategies.append({"name": "sft_stochastic_reuse", "ok": True, "reason": "SUCCESS_12S"})

    return {
        "id": row["id"],
        "attempts": attempts,
        "env_steps_est": env_steps,
        "strategies": json.dumps(strategies),
        "found_success": bool(best and best.get("ok")),
        "best_strategy": None if best is None else best.get("strategy"),
        "best_mechanism": None if best is None else best.get("mechanism_tag"),
        "best_reason": None if best is None else best.get("reason"),
        "best_recovery_s": None if best is None else best.get("recovery_elapsed"),
    }


def select_sets(screened: list[dict], search: dict[str, dict], retention_frac: float = 0.15):
    """Apply frozen splits; select hard verified + retention solved ICs."""
    by_split = {"train": [], "dev": [], "test": []}
    for row in screened:
        sid = row["assigned_split"]
        rec = search.get(row["id"], {})
        verified = bool(rec.get("found_success")) or bool(row.get("sft_det_success"))
        hard = bool(row.get("hard_by_sft"))
        # Prefer hard+verified; allow retention solved
        if hard and rec.get("found_success"):
            tier = "hard_verified"
        elif (not hard) and row.get("sft_det_success"):
            tier = "retention_solved"
        elif hard and not rec.get("found_success"):
            tier = "hard_unknown"
        else:
            tier = "other"
        item = {
            **{k: row[k] for k in (
                "id", "snapshot", "fingerprint", "assigned_split", "azimuth_deg", "speed_mps",
                "z_off_m", "phi_deg", "delay_s", "states", "nominal_reason",
                "sft_det_success", "sft_det_reason", "sft_det_mechanism", "sft_stoch_rate",
                "sft_stoch_successes", "hard_by_sft",
            )},
            "tier": tier,
            "oracle_found": bool(rec.get("found_success")),
            "oracle_strategy": rec.get("best_strategy"),
            "oracle_mechanism": rec.get("best_mechanism"),
            "pool": "recovery",
            "nominal_class": "RECOVERY_CANDIDATE",
            "split": sid if sid != "dev" else "train",  # GRPO readers reject valid; keep physical split in assigned_split
            "provenance": "hard_ic_mining_v1",
            "horizon": "recovery_limit_4s_then_nominal_12s",
            "recovery_limit_s": 4.0,
            "nominal_seconds": 12.0,
            "categories": _categories_for(row, rec),
        }
        by_split[sid].append(item)

    selected = {"train": [], "dev": [], "test": []}
    for sid, items in by_split.items():
        hard_v = [item for item in items if item["tier"] == "hard_verified"]
        retention = [item for item in items if item["tier"] == "retention_solved"]
        # Cap retention relative to hard set
        n_ret = min(len(retention), max(1, int(round(len(hard_v) * retention_frac)))) if hard_v else min(3, len(retention))
        keep = hard_v + retention[:n_ret]
        selected[sid] = keep
    return selected, by_split


def _categories_for(row, rec) -> list[str]:
    cats = []
    mech = rec.get("best_mechanism") or row.get("sft_det_mechanism") or ""
    if "Contact-Preserving" in str(mech) or (rec.get("best_strategy") or "").startswith(("controlled", "inward", "sliding")):
        cats.append("gravity_assisted_in_hand_sliding")
    if "Recatch" in str(mech) or "Hybrid" in str(mech) or "vertical" in str(rec.get("best_strategy") or ""):
        cats.append("vertical_recatch")
    if not cats:
        # Fall back from entry states
        states = row.get("states") or []
        if any(s in states for s in ("complete_grasp_loss", "airborne_object", "asymmetric_or_unilateral")):
            cats.append("vertical_recatch")
        else:
            cats.append("gravity_assisted_in_hand_sliding")
    return sorted(set(cats))


def write_manifests(selected: dict) -> None:
    for name in ("train", "dev", "test"):
        path = OUT / f"selected_{name}_manifest.jsonl"
        if path.is_file():
            path.unlink()
        for row in selected[name]:
            append_jsonl(path, row)
    # GRPO-readable copy for classic trainer (train only; split field must be train)
    grpo_path = OUT / "grpo_initial_conditions_hard.jsonl"
    if grpo_path.is_file():
        grpo_path.unlink()
    for row in selected["train"]:
        append_jsonl(grpo_path, {
            "id": row["id"],
            "snapshot": row["snapshot"],
            "pool": "recovery",
            "nominal_class": "RECOVERY_CANDIDATE",
            "split": "train",
            "horizon": row["horizon"],
            "recovery_limit_s": 4.0,
            "nominal_seconds": 12.0,
            "provenance": row["provenance"],
            "categories": row["categories"],
            "assigned_mining_split": row["assigned_split"],
            "tier": row["tier"],
        })


def baseline_table(screened: list[dict], selected: dict) -> list[dict]:
    rows = []
    pools = {
        "all_candidates": screened,
        "selected_train": selected["train"],
        "selected_dev": selected["dev"],
        "selected_test": selected["test"],
    }
    for name, items in pools.items():
        if not items:
            rows.append({"pool": name, "n": 0})
            continue
        # map back to screened metrics by id
        by_id = {row["id"]: row for row in screened}
        dets = [by_id[item["id"]] for item in items if item["id"] in by_id]
        n = len(dets)
        def mech_success(label_substr):
            return sum(
                1 for row in dets
                if row.get("sft_det_success") and label_substr in str(row.get("sft_det_mechanism") or "")
            )
        rows.append({
            "pool": name,
            "n": n,
            "nominal_success": 0,  # all RECOVERY_CANDIDATE by construction
            "sft_det_success": sum(1 for row in dets if row.get("sft_det_success")),
            "sft_det_success_rate": float(np.mean([row.get("sft_det_success") for row in dets])) if dets else 0.0,
            "mean_stoch_rate": float(np.mean([row.get("sft_stoch_rate") or 0.0 for row in dets])) if dets else 0.0,
            "cp_success": mech_success("Contact-Preserving"),
            "recatch_success": mech_success("Release-Recatch"),
            "hybrid_success": mech_success("Hybrid"),
            "mean_recovery_s_success": float(np.mean([
                row["sft_det_recovery_s"] for row in dets
                if row.get("sft_det_success") and row.get("sft_det_recovery_s") is not None
            ])) if any(row.get("sft_det_success") for row in dets) else None,
            "hard_count": sum(1 for row in dets if row.get("hard_by_sft")),
        })
    return rows


def render_examples(env, model, book, device, selected: dict, limit: int = 4) -> None:
    video_dir = OUT / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    examples = (selected["train"] + selected["test"] + selected["dev"])[:limit]
    for row in examples:
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        cam, renderer = open_camera(env.sim)

        def grab(sim):
            return shoot(sim, cam, renderer)

        # Policy frames via free_rollout sampling times — simplified: use compare_policy play if available
        from visualization.compare_policy_nominal import play_policy, play_nominal, restore_ic

        restore_ic(env, blob)
        left, left_label, _, _ = play_policy(env, model, book, blob, device, grab, SELECTED_LIMIT_S)
        restore_ic(env, blob)
        right, right_label = play_nominal(env, blob["ref"], grab)
        renderer.close()
        frames = stitch(left, right)
        dest = video_dir / f"{row['id']}__policy_vs_nominal.mp4"
        write_frames_mp4(frames, dest, fps=25)
        shutil.rmtree(dest.parent / f"_frames_{dest.stem}", ignore_errors=True)
        print("VIDEO", dest.name, left_label, right_label, flush=True)


def write_reports(meta: dict, screened: list[dict], search_rows: list[dict], selected: dict, baselines: list[dict]) -> None:
    search_by = {row["id"]: row for row in search_rows}
    hard = [row for row in screened if row.get("hard_by_sft")]
    hard_verified = [row for row in hard if search_by.get(row["id"], {}).get("found_success")]
    hard_unknown = [row for row in hard if not search_by.get(row["id"], {}).get("found_success")]
    mech_oracle = Counter(row.get("best_mechanism") for row in search_rows if row.get("found_success"))
    mech_sft = Counter(row.get("sft_det_mechanism") for row in screened)

    lines = [
        "# Hard IC mining report",
        "",
        "## Scope",
        "",
        f"- SFT checkpoint: `{SFT_CKPT.relative_to(ROOT).as_posix()}` (equal-weight Epoch 40)",
        "- Physics / nominal / K48 / step_legal / SUCCESS_12S unchanged",
        "- No GRPO training; no SFT modification",
        "- Candidate generation: legal continuous ball-impact via `run_episode` + `take_capture` (not teleports)",
        "",
        "## Frozen split rule",
        "",
        "See `split_rule.json`. Assigned before outcome-based selection.",
        "",
        "## Funnel",
        "",
        f"| Stage | Count |",
        f"|---|---:|",
        f"| Generation attempts | {meta.get('generate_attempts')} |",
        f"| Valid impact recovery ICs kept as candidates | {len(screened)} |",
        f"| Hard by SFT (det fail or stoch < 2/3) | {len(hard)} |",
        f"| Hard + oracle/stochastic SUCCESS evidence | {len(hard_verified)} |",
        f"| Hard + recoverability unknown under budget | {len(hard_unknown)} |",
        f"| Selected train / dev / test | {len(selected['train'])} / {len(selected['dev'])} / {len(selected['test'])} |",
        "",
        "## SFT stochastic rate distribution (candidates)",
        "",
    ]
    rates = [float(row.get("sft_stoch_rate") or 0.0) for row in screened]
    if rates:
        lines.append(f"- mean={np.mean(rates):.3f}, median={np.median(rates):.3f}, "
                     f"frac_zero={np.mean([r == 0 for r in rates]):.3f}, "
                     f"frac_ge_1={np.mean([r >= 1.0 for r in rates]):.3f}")
    lines.extend([
        "",
        "### SFT det mechanism counts (all candidates)",
        "",
        f"`{dict(mech_sft)}`",
        "",
        "### Oracle success mechanism tags",
        "",
        f"`{dict(mech_oracle)}`",
        "",
        "## Baseline comparison (SUCCESS rates)",
        "",
        "| pool | n | nominal SUCCESS | SFT det SUCCESS | mean stoch rate | CP succ | Recatch succ | Hybrid succ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in baselines:
        lines.append(
            f"| {row['pool']} | {row.get('n', 0)} | {row.get('nominal_success', 0)} | "
            f"{row.get('sft_det_success', 0)} | {row.get('mean_stoch_rate', 0):.3f} | "
            f"{row.get('cp_success', 0)} | {row.get('recatch_success', 0)} | {row.get('hybrid_success', 0)} |"
        )
    lines.extend([
        "",
        "All candidates are nominal-fail (`RECOVERY_CANDIDATE`) by construction.",
        "",
        "## Near-duplicate check",
        "",
        f"- Rejected fingerprints already in `data/snapshots/{{train,valid}}`: tracked during generation",
        f"- Known fingerprints loaded at start: {meta.get('known_fingerprints')}",
        f"- Historical 17 valid ICs excluded from GRPO mix: {sorted(meta.get('valid_ids') or [])}",
        "",
        "## Files",
        "",
        "- `candidate_manifest.jsonl`, `screening_results.csv`, `recoverability_search.csv`",
        "- `selected_{train,dev,test}_manifest.jsonl`, `grpo_initial_conditions_hard.jsonl`",
        "- `baseline_comparison.csv`, `grpo_readiness.md`, `snapshots/`, `videos/`",
        "",
        "## Repro",
        "",
        "```powershell",
        f"python -u experiments/sliding/mine_hard_ics.py --device cuda --generate-attempts {meta.get('generate_attempts')} --seed-offset {meta.get('seed_offset')}",
        "```",
        "",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # GRPO readiness
    n_train = len(selected["train"])
    n_hard_train = sum(1 for row in selected["train"] if row["tier"] == "hard_verified")
    n_ret = sum(1 for row in selected["train"] if row["tier"] == "retention_solved")
    cats = Counter()
    for row in selected["train"]:
        for cat in row.get("categories") or []:
            cats[cat] += 1
    readiness = [
        "# GRPO readiness",
        "",
        "## Evidence-based judgment",
        "",
        f"1. **Train IC count:** {n_train} (hard_verified={n_hard_train}, retention={n_ret}).",
        f"   Tree80 Stage A/B need reward variation; hard ICs with oracle success but SFT fail/stoch-low are suitable.",
        f"2. **Category coverage on selected train:** `{dict(cats)}`.",
        "3. **Dead ICs:** hard_unknown were **not** placed in selected manifests (avoid zero-success reward collapse).",
        "4. **Retention mix:** small solved set included to reduce catastrophic forgetting of Contact-Preserving / Recatch.",
        "5. **Init:** equal-weight SFT Epoch 40 is strong enough to initialize GRPO (17/17 on old valid).",
        "6. **Caveats:**",
        "   - First-round mining is vertical-grasp ballistic only; dedicated horizontal hard grid is thin/absent unless listed above.",
        "   - Oracle budget is limited (`_try_sliding` + ≤6 vertical specs); unknown ≠ impossible.",
        "   - Sliding skill mass in equal-weight SFT may bias exploration; monitor Contact-Preserving under GRPO.",
        "   - Adaptive tree80 currently loads skill manifests under `data/`; consume `grpo_initial_conditions_hard.jsonl` via a dedicated config or merge snapshots carefully without touching sealed valid.",
        "",
        "## Recommendation",
        "",
    ]
    if n_hard_train >= 8 and cats:
        readiness.append(
            "**Ready for a limited GRPO pilot** on `selected_train_manifest.jsonl` / "
            "`grpo_initial_conditions_hard.jsonl`, with Contact-Preserving count on `selected_dev` as the early-stop monitor. "
            "Do not use `selected_test` for checkpoint picking."
        )
    else:
        readiness.append(
            "**Not yet ready for full GRPO.** Increase generate-attempts and oracle budget until "
            "hard_verified train coverage is broader across Sliding and Recatch."
        )
    (OUT / "grpo_readiness.md").write_text("\n".join(readiness) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--generate-attempts", type=int, default=180)
    parser.add_argument("--seed-offset", type=int, default=HARD_DRAW_START)
    parser.add_argument("--skip-videos", action="store_true")
    parser.add_argument("--resume-screen", action="store_true", help="Reuse existing candidate_manifest.jsonl")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("snapshots", "traces", "contact_events", "videos"):
        (OUT / name).mkdir(exist_ok=True)

    rule = write_split_rule()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    book = load_codebook()
    print("LOAD", SFT_CKPT, flush=True)
    model, _ = load_trained_k48(SFT_CKPT, device)
    model.eval()
    env = make_training_env(device, model=model)

    known_fp = load_known_fingerprints()
    valid_ids = load_valid_ids()
    meta = {
        "generate_attempts": args.generate_attempts,
        "seed_offset": args.seed_offset,
        "known_fingerprints": len(known_fp),
        "valid_ids": sorted(valid_ids),
        "stoch_seeds": list(STOCH_SEEDS),
        "split_rule": rule,
        "sft_checkpoint": str(SFT_CKPT.relative_to(ROOT)).replace("\\", "/"),
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.resume_screen and (OUT / "candidate_manifest.jsonl").is_file():
        candidates = [
            json.loads(line) for line in (OUT / "candidate_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        print("RESUME_CANDIDATES", len(candidates), flush=True)
    else:
        candidates = generate_candidates(env, args.generate_attempts, known_fp, args.seed_offset)

    screened = []
    for row in candidates:
        print("SCREEN", row["id"], flush=True)
        screened.append(screen_one(env, model, book, device, row))

    with (OUT / "screening_results.csv").open("w", encoding="utf-8", newline="") as handle:
        if screened:
            writer = csv.DictWriter(handle, fieldnames=list(screened[0].keys()))
            writer.writeheader()
            writer.writerows(screened)

    search_rows = []
    for row in screened:
        # Search all hard ICs; also search a few solved for mechanism tags
        if row.get("hard_by_sft") or not row.get("sft_det_success"):
            print("ORACLE", row["id"], flush=True)
            search_rows.append(oracle_search(env, book, row))
        else:
            search_rows.append({
                "id": row["id"], "attempts": 0, "env_steps_est": 0, "strategies": "[]",
                "found_success": True, "best_strategy": "sft_deterministic",
                "best_mechanism": row.get("sft_det_mechanism"), "best_reason": "SUCCESS_12S",
                "best_recovery_s": row.get("sft_det_recovery_s"),
            })

    with (OUT / "recoverability_search.csv").open("w", encoding="utf-8", newline="") as handle:
        if search_rows:
            writer = csv.DictWriter(handle, fieldnames=list(search_rows[0].keys()))
            writer.writeheader()
            writer.writerows(search_rows)

    search_by = {row["id"]: row for row in search_rows}
    selected, _all_by_split = select_sets(screened, search_by)
    write_manifests(selected)

    baselines = baseline_table(screened, selected)
    with (OUT / "baseline_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baselines[0].keys()))
        writer.writeheader()
        writer.writerows(baselines)

    write_reports(meta, screened, search_rows, selected, baselines)

    if not args.skip_videos and (selected["train"] or selected["test"]):
        try:
            render_examples(env, model, book, device, selected, limit=4)
        except Exception as exc:
            print("VIDEO_FAIL", exc, flush=True)

    hard = sum(1 for row in screened if row.get("hard_by_sft"))
    hard_v = sum(1 for row in screened if row.get("hard_by_sft") and search_by.get(row["id"], {}).get("found_success"))
    print("DONE", json.dumps({
        "candidates": len(screened),
        "hard": hard,
        "hard_verified": hard_v,
        "train": len(selected["train"]),
        "dev": len(selected["dev"]),
        "test": len(selected["test"]),
    }), flush=True)


if __name__ == "__main__":
    main()
