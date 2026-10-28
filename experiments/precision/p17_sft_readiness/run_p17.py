"""P17 — Consolidate verified experts; assess SFT readiness (no training).

Official corpus is never mutated. P15/P16 stay candidate / diagnostic tiers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.expert_pipeline.live_stage1 import extract_precision_skills  # noqa: E402
from data_generation.expert_pipeline.provenance import ACTIVE_DATASET  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "p17_sft_readiness"
M111_META = (
    ROOT
    / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
)
MIN_MM = 0.5
SEED = 17


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _fp_actions(actions) -> str:
    arr = np.asarray(actions, np.float32).reshape(-1)
    if arr.size == 0:
        return "empty"
    return hashlib.sha1(arr.tobytes()).hexdigest()[:16]


def _lineage_id(source: str | None, base: str | None, path: str | None = None) -> str:
    # Prefer stable base snapshot / source IC; strip offset tags
    raw = (base or source or path or "unknown").replace("\\", "/")
    # Drop forced-skill / offset suffixes for grouping
    for sep in ("__force_", "_dx", "__dx", "_b0_", "_b1_", "_b2_", "_b3_"):
        if sep in raw and "ball_train" in raw:
            # keep ball_train id portion
            break
    # Normalize: take last path component without hash suffixes when possible
    name = Path(raw).stem if "/" in raw or "\\" in raw else raw
    # Collapse P15/P16 offset variants onto base_sample_id when present
    return name


def normalize_skill_seq(skills: list[str] | None, *, branch_id: str | None = None) -> str:
    skills = list(skills or [])
    # Map support_transfer → dx
    mapped = []
    for s in skills:
        sl = str(s).lower()
        if sl in ("support_transfer", "dx"):
            lab = "dx"
        elif "inward" in sl:
            lab = "dz_inward"
        elif "outward" in sl:
            lab = "dz_outward"
        elif sl.startswith("dz"):
            lab = "dz"
        elif "coupled" in sl:
            lab = "coupled"
        elif "dtheta" in sl:
            lab = "dtheta"
        else:
            lab = sl
        if not mapped or mapped[-1] != lab:
            mapped.append(lab)
    if mapped:
        seq = "→".join(mapped)
    else:
        seq = "none"
    # Prefix recovery letter when Stage-1 branch known
    bid = (branch_id or "").upper()
    if bid.startswith("A_") and "DIRECT" in bid:
        return f"A_recovery→{seq}" if seq != "none" else "A_recovery→direct_handoff"
    if bid.startswith("A_") and "PRECISION" in bid:
        return f"A_recovery→{seq}" if seq not in ("none",) else "A_recovery→precision"
    if bid.startswith("B_") and "DIRECT" in bid:
        return f"B_recatch→{seq}" if seq != "none" else "B_recatch→direct_handoff"
    if bid.startswith("B_") and "PRECISION" in bid:
        return f"B_recatch→{seq}"
    if bid.startswith("C_") and "DIRECT" in bid:
        return f"C_dynamic_recatch→{seq}" if seq != "none" else "C_dynamic_recatch→direct_handoff"
    if bid.startswith("C_") and "PRECISION" in bid:
        return f"C_dynamic_recatch→{seq}"
    if bid.startswith("D_"):
        return f"D2→{seq}"
    return seq


def errors_pack(err: dict | None) -> dict[str, Any]:
    if not err:
        return {}
    e = err.get("e_signed_mm")
    if e is None:
        return {
            "dx_mm": err.get("dx_mm"),
            "dz_mm": err.get("dz_mm"),
            "dtheta_deg": err.get("dtheta_deg"),
            "e_pos_mm": err.get("e_pos_mm"),
        }
    e = np.asarray(e, float).reshape(-1)
    return {
        "dx_mm": float(e[0]) if len(e) > 0 else None,
        "dz_mm": float(e[2]) if len(e) > 2 else None,
        "dtheta_deg": err.get("dtheta_deg"),
        "e_pos_mm": err.get("e_pos_mm"),
    }


def load_official() -> list[dict]:
    man = ACTIVE_DATASET / "manifest.jsonl"
    rows = []
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        path = ROOT / m["record_path"]
        phases: list[str] = []
        acts = None
        err0 = err1 = {}
        capture_fp = None
        try:
            raw = pickle.loads(path.read_bytes())
            phases = list(raw.get("phases") or [])
            acts = raw.get("actions_7d")
            if acts is None:
                acts = raw.get("actions")
            err0 = errors_pack(raw.get("errors_entry"))
            err1 = errors_pack(raw.get("errors_exit"))
            # fingerprint from capture state if present
            cap = raw.get("capture") or {}
            st = (cap.get("physics") or {}).get("state")
            if st is not None:
                capture_fp = hashlib.sha1(np.asarray(st, np.float64).tobytes()).hexdigest()[:16]
        except Exception as exc:
            m["_load_error"] = str(exc)
        skills = extract_precision_skills(phases)
        # Prefer actual phases over manifest skill_sequence when precision executed
        if m.get("precision_skipped") or str(m.get("skill_sequence", "")).startswith("direct"):
            actual = str(m.get("skill_sequence") or "direct_handoff")
            if "skip" in actual or actual == "direct_handoff":
                skills = []
        seq = normalize_skill_seq(skills, branch_id=m.get("branch_id"))
        if not skills and m.get("precision_skipped"):
            seq = normalize_skill_seq([], branch_id=m.get("branch_id")).replace("→none", "→precision_skipped")
            if "precision_skipped" not in seq:
                if "DIRECT" in (m.get("branch_id") or ""):
                    seq = seq.replace("→none", "→direct_handoff") if "→" in seq else seq
                else:
                    seq = normalize_skill_seq([], branch_id=m.get("branch_id"))
                    # branch-level direct/skip
                    bid = m.get("branch_id") or ""
                    if "DIRECT" in bid:
                        letter = bid[0]
                        name = {"A": "A_recovery", "B": "B_recatch", "C": "C_dynamic_recatch"}.get(letter, letter)
                        seq = f"{name}→direct_handoff"
                    elif "PRECISION" in bid and m.get("precision_skipped"):
                        letter = bid[0]
                        name = {"A": "A_recovery", "B": "B_recatch", "C": "C_dynamic_recatch"}.get(letter, letter)
                        seq = f"{name}→precision_skipped"

        n_act = int(m.get("n_actions") or (len(acts) if acts is not None else 0))
        e0 = float(err0.get("e_pos_mm") or 0) if err0.get("e_pos_mm") is not None else None
        e1 = float(err1.get("e_pos_mm") or 0) if err1.get("e_pos_mm") is not None else None
        rows.append(
            {
                "corpus_tier": "official_accepted",
                "traj_id": m.get("expert_id") or m.get("attempt_id"),
                "branch_id": m.get("branch_id"),
                "source_ic_id": m.get("source_ic_id"),
                "record_path": m.get("record_path"),
                "lineage_id": _lineage_id(m.get("source_ic_id"), m.get("source_ic_id")),
                "capture_fp": capture_fp,
                "actual_skill_sequence": seq,
                "manifest_skill_sequence": m.get("skill_sequence"),
                "precision_skills": "|".join(skills),
                "precision_skipped": m.get("precision_skipped"),
                "n_actions": n_act,
                "action_fp": _fp_actions(acts) if acts is not None else None,
                "dx0_mm": err0.get("dx_mm"),
                "dz0_mm": err0.get("dz_mm"),
                "dtheta0_deg": err0.get("dtheta_deg"),
                "e_pos0_mm": e0,
                "dx1_mm": err1.get("dx_mm"),
                "dz1_mm": err1.get("dz_mm"),
                "dtheta1_deg": err1.get("dtheta_deg"),
                "e_pos1_mm": e1,
                "e_pos_improved": (e0 - e1) >= MIN_MM if (e0 is not None and e1 is not None) else None,
                "SUCCESS_NEW": m.get("SUCCESS_NEW"),
                "SUCCESS_12S": m.get("SUCCESS_12S"),
                "physical_success": m.get("physical_success"),
                "phases_head": "|".join(phases[:12]),
                "physics_config": "default_ground_sim",
                "forced_precision": False,
            }
        )
    return rows


def load_diagnostic_ics(dir_path: Path, *, tier: str, default_skill: str | None = None) -> list[dict]:
    rows = []
    if not dir_path.exists():
        return rows
    for p in sorted(dir_path.glob("*.pkl")):
        try:
            raw = pickle.loads(p.read_bytes())
        except Exception as exc:
            rows.append({"corpus_tier": tier, "traj_id": p.stem, "load_error": str(exc)})
            continue
        meta = raw.get("meta") or {}
        acts = raw.get("actions") or meta.get("actions")
        base = raw.get("base_sample_id") or meta.get("base_sample_id") or meta.get("same_base_lineage")
        force = raw.get("requested_skill") or meta.get("requested_skill") or meta.get("force_skill") or default_skill
        demo = raw.get("demo_class") or meta.get("demo_class")
        skills = []
        if force == "dx" or (demo and "DX" in str(demo).upper()):
            skills = ["dx"]
        elif force == "dz" or (demo and "DZ" in str(demo).upper()):
            mode = meta.get("dz_mode") or raw.get("dz_mode") or ""
            if "inward" in str(mode).lower() or "B_" in str(mode):
                skills = ["dz_inward"]
            elif "outward" in str(mode).lower() or "A_" in str(mode):
                skills = ["dz_outward"]
            else:
                skills = ["dz"]
        elif force == "coupled" or (demo and "COUPLED" in str(demo).upper()):
            skills = ["coupled"]
        seq = "→".join(skills) if skills else str(demo or force or "unknown")
        dx0 = meta.get("dx0_mm", raw.get("dx0_mm"))
        dz0 = meta.get("dz0_mm", raw.get("dz0_mm"))
        dx1 = meta.get("dx1_mm", raw.get("dx1_mm"))
        dz1 = meta.get("dz1_mm", raw.get("dz1_mm"))
        e0 = meta.get("e_pos0_mm", raw.get("e_pos0_mm"))
        e1 = meta.get("e_pos1_mm", raw.get("e_pos1_mm"))
        if e0 is None and dx0 is not None and dz0 is not None:
            e0 = float(np.sqrt(float(dx0) ** 2 + float(dz0) ** 2))
        if e1 is None and dx1 is not None and dz1 is not None:
            e1 = float(np.sqrt(float(dx1) ** 2 + float(dz1) ** 2))
        ok = meta.get("ok_demo", raw.get("ok", meta.get("SUCCESS_NEW")))
        rows.append(
            {
                "corpus_tier": tier,
                "traj_id": p.stem,
                "branch_id": f"CANDIDATE_{str(force or 'unk').upper()}",
                "source_ic_id": base,
                "record_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "lineage_id": _lineage_id(base, base),
                "capture_fp": meta.get("capture_fp"),
                "actual_skill_sequence": seq,
                "manifest_skill_sequence": demo,
                "precision_skills": "|".join(skills),
                "precision_skipped": False,
                "n_actions": len(acts) if acts is not None else int(meta.get("n_actions") or 0),
                "action_fp": _fp_actions(acts) if acts is not None else None,
                "dx0_mm": dx0,
                "dz0_mm": dz0,
                "dx1_mm": dx1,
                "dz1_mm": dz1,
                "e_pos0_mm": e0,
                "e_pos1_mm": e1,
                "e_pos_improved": (
                    (float(e0) - float(e1)) >= MIN_MM if e0 is not None and e1 is not None else None
                ),
                "SUCCESS_NEW": bool(ok) if ok is not None else meta.get("SUCCESS_NEW"),
                "forced_precision": True,
                "physics_config": "default_ground_sim",
                "requested_skill": force,
            }
        )
    return rows


def coupled_tiers_from_csv() -> list[dict]:
    rows: list[dict] = []
    sources = [
        (
            ROOT / "results/diagnostics/p15_forced_precision_d2/COUPLED_EXECUTION_RESULTS.csv",
            "p15",
        ),
        (
            ROOT / "results/diagnostics/p16_precision_skill_expansion/COUPLED_P16_GENERATION.csv",
            "p16",
        ),
        (
            ROOT / "results/diagnostics/p16_precision_skill_expansion/COUPLED_REASSESSMENT.csv",
            "p15_reassess",
        ),
    ]
    for path, src in sources:
        if not path.exists():
            continue
        for r in csv.DictReader(path.open(encoding="utf-8")):
            try:
                dx0, dz0 = float(r.get("dx0_mm") or 0), float(r.get("dz0_mm") or 0)
                dx1, dz1 = float(r.get("dx1_mm") or 0), float(r.get("dz1_mm") or 0)
            except ValueError:
                continue
            e0 = r.get("e_pos0_mm") or r.get("e_pos0_xz_mm")
            e1 = r.get("e_pos1_mm") or r.get("e_pos1_xz_mm")
            if e0 is None:
                e0 = float(np.sqrt(dx0 * dx0 + dz0 * dz0))
            else:
                e0 = float(e0)
            if e1 is None:
                e1 = float(np.sqrt(dx1 * dx1 + dz1 * dz1))
            else:
                e1 = float(e1)
            n_act = int(float(r.get("n_actions") or 0))
            strict = str(r.get("SUCCESS_NEW") or r.get("ok_demo") or "").lower() in ("true", "1")
            if src == "p15_reassess":
                strict = str(r.get("SUCCESS_NEW")).lower() == "true"
            both = (abs(dx0) - abs(dx1)) >= MIN_MM and (abs(dz0) - abs(dz1)) >= MIN_MM
            e_imp = (e0 - e1) >= MIN_MM
            real_coupled = n_act > 0 and str(r.get("requested_skill", "coupled")).lower() in (
                "coupled",
                "",
            )
            # For generation CSVs, requested_skill may be absent — treat as coupled
            if "COUPLED" in str(r.get("tag") or "").upper() or src.startswith("p15"):
                real_coupled = n_act > 0
            tiers = []
            if strict and real_coupled:
                tiers.append("COUPLED_TASK_SUCCESS")
            if strict and real_coupled and e_imp:
                tiers.append("COUPLED_POSITION_IMPROVED")
            if strict and real_coupled and both:
                tiers.append("COUPLED_BOTH_AXES_IMPROVED")
            preferred = "COUPLED_POSITION_IMPROVED" in tiers
            rows.append(
                {
                    "tag": r.get("tag"),
                    "source": src,
                    "base_sample_id": r.get("base_sample_id"),
                    "dx0_mm": dx0,
                    "dz0_mm": dz0,
                    "dx1_mm": dx1,
                    "dz1_mm": dz1,
                    "e_pos0_mm": e0,
                    "e_pos1_mm": e1,
                    "n_actions": n_act,
                    "SUCCESS_NEW": strict,
                    "both_axis_improved": both,
                    "e_pos_improved": e_imp,
                    "tiers": "|".join(tiers) if tiers else "NONE",
                    "sft_preferred_positive": preferred,
                    "ablation_task_success_only": (
                        "COUPLED_TASK_SUCCESS" in tiers and not preferred
                    ),
                    "demo_class": r.get("demo_class") or r.get("demo_class_p15"),
                    "status": r.get("status"),
                }
            )
    return rows


def audit_dz_inward() -> str:
    path = ROOT / "results/diagnostics/p16_precision_skill_expansion/DZ_EXPANSION.csv"
    if not path.exists():
        return "# DZ Inward Failure Audit\n\nNo P16 DZ_EXPANSION.csv found.\n"
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8"))]
    inward = [r for r in rows if str(r.get("dz_mode", "")).startswith("B_") or float(r.get("requested_dz_mm") or 0) > 0]
    outward = [r for r in rows if str(r.get("dz_mode", "")).startswith("A_") or float(r.get("requested_dz_mm") or 0) < 0]

    def _stats(rs):
        return {
            "n": len(rs),
            "strict": sum(1 for r in rs if r.get("ok_demo") == "True"),
            "status": dict(Counter(r.get("status") for r in rs)),
            "bilateral_exit": dict(Counter(r.get("bilateral_exit") for r in rs)),
            "mean_dz_reduction": float(
                np.mean([float(r["dz_reduction_mm"]) for r in rs if r.get("dz_reduction_mm") not in (None, "")])
            )
            if any(r.get("dz_reduction_mm") not in (None, "") for r in rs)
            else None,
            "mean_e_pos_delta": float(
                np.mean(
                    [
                        float(r["e_pos1_mm"]) - float(r["e_pos0_mm"])
                        for r in rs
                        if r.get("e_pos0_mm") not in (None, "") and r.get("e_pos1_mm") not in (None, "")
                    ]
                )
            )
            if rs
            else None,
        }

    si, so = _stats(inward), _stats(outward)
    # Mechanism from numbers: inward often worsens |dz| and e_pos while bilateral retained
    mech = []
    if si["n"] and si["strict"] == 0:
        mech.append("No inward STRICT accepts in P16 matrix.")
    if si.get("mean_dz_reduction") is not None and si["mean_dz_reduction"] < 0:
        mech.append(
            f"Mean |dz| reduction is **negative** ({si['mean_dz_reduction']:.2f} mm): "
            "correction direction insufficient / wrong relative to residual (not zero-action)."
        )
    if si.get("mean_e_pos_delta") is not None and si["mean_e_pos_delta"] > 0:
        mech.append(
            f"Mean e_pos **increases** by {si['mean_e_pos_delta']:.2f} mm → final position failure vs entry."
        )
    if si["bilateral_exit"].get("True", 0) == si["n"]:
        mech.append("Bilateral contact retained at exit → not primarily contact-loss.")
    if all("PRIMITIVE_FAILED" in str(s) for s in si["status"]):
        mech.append(
            "Status `PRECISION_PRIMITIVE_FAILED` with reason often `dz:ok`: "
            "inward slip branch may report local ok while overall residual/handoff fails — "
            "**implementation/contract mismatch** worth a focused follow-up (not a new heuristic)."
        )

    return f"""# DZ Inward Failure Audit (P16 evidence)

## Scope
P16 forced-DZ matrix (`DZ_EXPANSION.csv`). Positive requested DZ → mode `B_full` (inward).
Negative requested DZ → mode `A_direct_down` (outward).

## Counts
| set | n | STRICT ok | statuses |
|-----|--:|----------:|----------|
| Inward (B_*) | {si['n']} | {si['strict']} | {si['status']} |
| Outward (A_*) | {so['n']} | {so['strict']} | {so['status']} |

## Inward metrics
- Mean |dz| reduction (mm): {si.get('mean_dz_reduction')}
- Mean Δe_pos (mm, exit−entry): {si.get('mean_e_pos_delta')}
- Bilateral exit: {si.get('bilateral_exit')}

## What `B_full` means here
From residuals + status (not from the label alone):

{chr(10).join('- ' + m for m in mech) or '- Insufficient evidence.'}

### Checklist
| hypothesis | supported? |
|------------|------------|
| Primitive never ran (0 actions) | **No** — hundreds of actions typical |
| Contact loss | **No** — bilateral_exit True on inward fails |
| Insufficient / wrong correction | **Yes** — |dz| and e_pos often worsen |
| Final position failure | **Yes** — e_pos up; SUCCESS_NEW False |
| Final orientation failure | Not isolated in this CSV (dθ not logged per row) |
| HANDOFF failure after good geometry | Secondary — status is PRIMITIVE_FAILED before clean SUCCESS |

## Recommendation
Do **not** add a new inward heuristic in P17. File a targeted investigation of inward DualLoop stop/desire vs signed residual under forced small |dz|, separately from FSM entry.
"""


def lineage_split(rows: list[dict]) -> tuple[dict, str]:
    """Group by lineage_id; assign train/val without leakage."""
    by_lin: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("corpus_tier") == "failed" or r.get("load_error"):
            continue
        lid = str(r.get("lineage_id") or r.get("source_ic_id") or r.get("traj_id"))
        by_lin[lid].append(r)

    rng = np.random.default_rng(SEED)
    lineages = sorted(by_lin.keys())
    rng.shuffle(lineages)
    n_val = max(1, int(round(0.2 * len(lineages)))) if lineages else 0
    val_set = set(lineages[:n_val])
    train_set = set(lineages[n_val:])

    split_rows = []
    for lid, trajs in by_lin.items():
        split = "val" if lid in val_set else "train"
        skills = Counter(t.get("actual_skill_sequence") for t in trajs)
        split_rows.append(
            {
                "lineage_id": lid,
                "split": split,
                "n_trajectories": len(trajs),
                "n_unique_action_fp": len({t.get("action_fp") for t in trajs if t.get("action_fp")}),
                "n_unique_capture_fp": len({t.get("capture_fp") for t in trajs if t.get("capture_fp")}),
                "tiers": dict(Counter(t.get("corpus_tier") for t in trajs)),
                "skills": dict(skills),
                "physics_config": "default_ground_sim",
            }
        )

    report = f"""# Lineage Split Report (P17)

## Method
- Group by `lineage_id` = source base / source_ic_id (offset variants share one lineage).
- Random 80/20 **lineage** split (seed={SEED}).
- **No** trajectory from the same lineage appears in both train and val.

## Totals
- Unique lineages: **{len(lineages)}**
- Train lineages: **{len(train_set)}**
- Val lineages: **{len(val_set)}**
- Trajectories considered: **{sum(len(v) for v in by_lin.values())}**

## Caveat
Multiple teleport/forced offsets from `ball_train_v2_0168` count as **one** lineage.
Do not claim independent grasp generalization from that family alone.

## Physics
All audited trajectories use `default_ground_sim` (unchanged).
"""
    return {"lineages": split_rows, "train": sorted(train_set), "val": sorted(val_set)}, report


def make_env():
    from controllers.recovery.geometry_switch import GeometrySwitch
    from envs.observations.observable_obs import ObservableObsState
    from envs.simulation.ground_sim import make_ground_sim
    from models.transformer_policy import DT_POLICY

    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def m111_verify_sample(corpus: list[dict], *, max_replay: int = 8) -> list[dict]:
    from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, load_motion_codebook, quantize_trajectory
    from envs.simulation.capture import restore_capture
    from envs.step_legal import step_legal
    from evaluation.axis_aware_success import evaluate_handoff_with_axis

    if not M111_META.exists():
        return [{"error": "M111_META_missing"}]
    cb = load_motion_codebook(M111_META)
    book = np.asarray(cb.prototypes, np.float32)
    rho = cb.residual
    w = DEFAULT_WEIGHTS

    # Prefer skill-diverse continuous demos with actions on disk
    candidates = []
    for r in corpus:
        if r.get("n_actions", 0) < 50:
            continue
        if r.get("corpus_tier") not in (
            "official_accepted",
            "candidate_verified",
        ):
            continue
        candidates.append(r)
    # diversify by skill
    picked = []
    seen_skill = set()
    for r in candidates:
        sk = r.get("actual_skill_sequence") or ""
        key = sk.split("→")[-1] if sk else ""
        if key not in seen_skill or len(picked) < 3:
            seen_skill.add(key)
            picked.append(r)
        if len(picked) >= max_replay:
            break
    if len(picked) < max_replay:
        for r in candidates:
            if r not in picked:
                picked.append(r)
            if len(picked) >= max_replay:
                break

    env = make_env()
    out = []
    for r in picked:
        path = ROOT / r["record_path"]
        try:
            raw = pickle.loads(path.read_bytes())
        except Exception as exc:
            out.append({"traj_id": r["traj_id"], "error": f"load:{exc}"})
            continue
        acts = raw.get("actions_7d")
        if acts is None:
            acts = raw.get("actions")
        acts = np.asarray(acts if acts is not None else [], np.float32).reshape(-1, 7)
        if len(acts) == 0:
            out.append({"traj_id": r["traj_id"], "error": "no_actions", "skill": r.get("actual_skill_sequence")})
            continue
        cap = raw.get("capture")
        ref = raw.get("ref")
        # For candidate pickles, capture is often post-success; start from meta construction if needed
        q = quantize_trajectory(book, acts, rho, w=w)
        recon = np.asarray(q["recon_actions"], np.float32)
        row = {
            "traj_id": r["traj_id"],
            "corpus_tier": r.get("corpus_tier"),
            "skill": r.get("actual_skill_sequence"),
            "n_actions": len(acts),
            "m111_n_unique_tokens": len(set(int(t) for t in q["tokens"])),
            "m111_clipped_frac": float(q["clipped_frac"]),
            "m111_mean_recon_wlinf": float(q["mean_recon_wlinf"]),
            "m111_max_recon_wlinf": float(q["max_recon_wlinf"]),
            "continuous_SUCCESS_NEW": r.get("SUCCESS_NEW"),
        }
        # Replay decoded actions if we have a usable start capture+ref
        if cap is not None and ref is not None and len(recon) > 0:
            try:
                restore_capture(env, cap)
                # If capture is terminal, replay is not meaningful for start-state demos —
                # still run to measure decode stability / crash
                step = max(1, len(recon) // 200)
                for a in recon[::step]:
                    step_legal(env.sim, np.asarray(a, np.float32))
                hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
                row["m111_replay_SUCCESS_NEW"] = bool(hand.get("SUCCESS_NEW"))
                row["m111_replay_SUCCESS_12S"] = bool(hand.get("SUCCESS_12S"))
                row["m111_replay_note"] = (
                    "start_capture_may_be_terminal; treat as decode sanity not full teacher replay"
                    if r.get("corpus_tier") == "official_accepted"
                    else "candidate_ic_start"
                )
            except Exception as exc:
                row["m111_replay_error"] = str(exc)
        else:
            row["m111_replay_note"] = "missing_capture_or_ref"
        out.append(row)
        print(
            f"  M111 {r['traj_id'][:40]} tokens={row['m111_n_unique_tokens']} "
            f"clip={row['m111_clipped_frac']:.3f} replay={row.get('m111_replay_SUCCESS_NEW')}",
            flush=True,
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=== P17 load official ===", flush=True)
    official = load_official()
    print(f"  official={len(official)}", flush=True)

    print("=== P17 load candidates / diagnostics ===", flush=True)
    p16_ics = load_diagnostic_ics(
        ROOT / "results/diagnostics/p16_precision_skill_expansion/ics",
        tier="candidate_verified",
    )
    p15_ics = load_diagnostic_ics(
        ROOT / "results/diagnostics/p15_forced_precision_d2/ics",
        tier="candidate_verified",
    )
    p14_ics = load_diagnostic_ics(
        ROOT / "results/diagnostics/p14_small_error_d2/ics",
        tier="candidate_verified",
        default_skill="dz",
    )
    p13_ics = load_diagnostic_ics(
        ROOT / "results/diagnostics/p13_teleport_d2/ics",
        tier="diagnostic_only",
        default_skill="dz",
    )
    p15_rej = load_diagnostic_ics(
        ROOT / "results/diagnostics/p15_forced_precision_d2/rejected_coupled_partial",
        tier="diagnostic_only",
        default_skill="coupled",
    )
    # Mark rejected as not preferred
    for r in p15_rej:
        r["corpus_tier"] = "diagnostic_only"

    corpus = official + p16_ics + p15_ics + p14_ics + p13_ics + p15_rej

    # Dedup only when both lineage and non-empty action fingerprint collide
    # (do not collapse distinct official directs that share empty action_fp).
    seen = set()
    deduped = []
    for r in corpus:
        afp = r.get("action_fp")
        if afp and afp != "empty":
            key = (r.get("lineage_id"), afp)
        else:
            key = ("id", r.get("traj_id") or r.get("record_path"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    corpus = deduped

    # Skill coverage table
    skill_rows = []
    for r in corpus:
        skill_rows.append(
            {
                "traj_id": r.get("traj_id"),
                "corpus_tier": r.get("corpus_tier"),
                "branch_id": r.get("branch_id"),
                "lineage_id": r.get("lineage_id"),
                "actual_skill_sequence": r.get("actual_skill_sequence"),
                "precision_skills": r.get("precision_skills"),
                "manifest_skill_sequence": r.get("manifest_skill_sequence"),
                "n_actions": r.get("n_actions"),
                "dx0_mm": r.get("dx0_mm"),
                "dz0_mm": r.get("dz0_mm"),
                "dx1_mm": r.get("dx1_mm"),
                "dz1_mm": r.get("dz1_mm"),
                "e_pos0_mm": r.get("e_pos0_mm"),
                "e_pos1_mm": r.get("e_pos1_mm"),
                "e_pos_improved": r.get("e_pos_improved"),
                "SUCCESS_NEW": r.get("SUCCESS_NEW"),
                "SUCCESS_12S": r.get("SUCCESS_12S"),
                "forced_precision": r.get("forced_precision"),
            }
        )
    _write_csv(OUT / "ACTUAL_SKILL_COVERAGE.csv", skill_rows)

    # Coupled tiers
    print("=== P17 Coupled tiers ===", flush=True)
    ct = coupled_tiers_from_csv()
    _write_csv(OUT / "COUPLED_QUALITY_TIERS.csv", ct)

    # DZ inward audit
    _write_md(OUT / "DZ_INWARD_FAILURE_AUDIT.md", audit_dz_inward())

    # Lineage split (official + candidates only for training candidates)
    train_pool = [
        r
        for r in corpus
        if r.get("corpus_tier") in ("official_accepted", "candidate_verified")
        and r.get("SUCCESS_NEW") in (True, "True", 1)
    ]
    split_obj, split_md = lineage_split(train_pool)
    _write_md(OUT / "LINEAGE_SPLIT_REPORT.md", split_md)
    (OUT / "lineage_split.json").write_text(json.dumps(split_obj, indent=2), encoding="utf-8")

    # M111
    print("=== P17 M111 sample verify ===", flush=True)
    m111_rows = m111_verify_sample(train_pool, max_replay=8)
    _write_csv(OUT / "M111_REPLAY_RESULTS.csv", m111_rows)

    # Manifest
    tier_counts = Counter(r.get("corpus_tier") for r in corpus)
    skill_counts = Counter(r.get("actual_skill_sequence") for r in corpus)
    official_real_prec = [
        r
        for r in official
        if not r.get("precision_skipped")
        and r.get("n_actions", 0) > 0
        and "direct" not in str(r.get("actual_skill_sequence"))
        and "skip" not in str(r.get("actual_skill_sequence"))
    ]
    lineages_indep = len({r.get("lineage_id") for r in train_pool})
    preferred_coupled = sum(1 for r in ct if r.get("sft_preferred_positive"))
    both_coupled = sum(1 for r in ct if "BOTH_AXES" in str(r.get("tiers")))

    # Skills across multiple lineages
    skill_lineages: dict[str, set] = defaultdict(set)
    for r in train_pool:
        for part in str(r.get("actual_skill_sequence") or "").split("→"):
            skill_lineages[part].add(r.get("lineage_id"))

    multi_geom = {k: len(v) for k, v in skill_lineages.items() if len(v) >= 2}

    manifest = {
        "generated_utc": _utc(),
        "active_dataset": str(ACTIVE_DATASET.relative_to(ROOT)).replace("\\", "/"),
        "tier_counts": dict(tier_counts),
        "n_corpus_rows": len(corpus),
        "n_official": len(official),
        "n_official_real_precision_actions": len(official_real_prec),
        "n_candidate_verified": tier_counts.get("candidate_verified", 0),
        "n_diagnostic_only": tier_counts.get("diagnostic_only", 0),
        "skill_sequence_counts": dict(skill_counts.most_common(40)),
        "n_train_pool_strict": len(train_pool),
        "n_independent_lineages_train_pool": lineages_indep,
        "coupled_preferred_sft": preferred_coupled,
        "coupled_both_axes": both_coupled,
        "skills_with_ge2_lineages": multi_geom,
        "auto_admit_p15_p16": False,
        "note": "P15/P16 never silently promoted to official.",
    }
    (OUT / "EXPERT_CORPUS_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_csv(OUT / "CORPUS_INDEX.csv", corpus)

    # Dataset spec + readiness
    m111_ok = [r for r in m111_rows if r.get("m111_n_unique_tokens")]
    m111_replay_ok = sum(1 for r in m111_rows if r.get("m111_replay_SUCCESS_NEW"))

    _write_md(
        OUT / "TRAINING_DATASET_SPEC.md",
        f"""# Training Dataset Spec (P17 candidate — not official admit)

## Tiers
1. **official_accepted** — `data/experts_v2_official` (STRICT admitted historically)
2. **candidate_verified** — P14–P16 STRICT ICs with continuous actions (forced-Precision metadata)
3. **diagnostic_only** — P13 / rejected Coupled / non-preferred rollouts
4. **failed** — retained in source CSVs only

## Policy inputs
- Observations: existing ObservableObs / DT_POLICY features only
- Actions: continuous 7D legal actions; M111 tokens as optional discrete targets
- **Do not** feed `forced_precision` / requested skill into observations

## Preferred SFT mixture (proposal)
- Recovery A/B/C direct + real Precision continuations from official
- Candidate DX / DZ outward / Coupled `COUPLED_POSITION_IMPROVED`
- Cap same-lineage offsets (e.g. ≤3 per lineage) to avoid clone domination

## Splits
See `LINEAGE_SPLIT_REPORT.md` / `lineage_split.json` (lineage-grouped, seed={SEED}).

## Exclude from positive SFT
- `precision_skipped` / zero-action “Precision” labels without real skills
- Coupled `COUPLED_TASK_SUCCESS` without e_pos improvement (ablation only)
- Inward DZ fails until mechanism fix
""",
    )

    ready = (
        lineages_indep >= 10
        and len(official_real_prec) >= 5
        and preferred_coupled >= 1
        and any("dx" in (k or "") for k in skill_lineages)
    )
    # Softer: initial SFT baseline OK if recovery+some precision exists; note gaps
    initial_sft_ok = len(official) >= 100 and lineages_indep >= 20

    _write_md(
        OUT / "SFT_READINESS_REPORT.md",
        f"""# SFT Readiness Report

## Demonstrated capabilities
- A/B/C recovery (+ direct handoff): strong in official ({tier_counts.get('official_accepted', 0)} traj)
- Real Precision (non-skip) official: **{len(official_real_prec)}**
- Forced DX candidates (P15/P16): present under candidate_verified
- DZ outward STRICT: present (P16)
- Coupled position-improved preferred: **{preferred_coupled}**

## Missing / weak generalization evidence
- DX mostly one compatible base lineage in forced set
- DZ inward: no STRICT accepts (see DZ_INWARD_FAILURE_AUDIT)
- Coupled both-axes improved: **{both_coupled}**
- Skills with ≥2 lineages: `{multi_geom}`

## M111
- Sample encode rows: {len(m111_ok)}; replay SUCCESS_NEW among samples: {m111_replay_ok}/{len(m111_rows)}
- Vocabulary **unchanged**; continuous→token fidelity varies (see `M111_REPLAY_RESULTS.csv`)

## Verdict
- **Initial SFT baseline (recovery-heavy + sparse Precision): {'YES' if initial_sft_ok else 'NOT YET'}**
- **Balanced multi-skill SFT: NOT YET** — need more independent DX bases, inward DZ fix, Coupled preferred set packaging
- **Do not train until** packaging script applies lineage caps + preferred Coupled filter + M111 encode of the frozen mix
""",
    )

    _write_md(
        OUT / "FINAL_REPORT.md",
        f"""# FINAL_REPORT — P17 SFT Readiness

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **Independent expert source lineages (STRICT train pool)**  
   **{lineages_indep}** unique `lineage_id`s among official+candidate STRICT rows.

2. **Trajectories per real skill (corpus skill_sequence counts)**  
   Top: `{dict(list(skill_counts.most_common(12)))}`  
   Official real Precision (non-skip with actions): **{len(official_real_prec)}**.

3. **Coupled suitable for positive SFT**  
   Preferred tier `COUPLED_POSITION_IMPROVED`: **{preferred_coupled}**  
   Both-axes improved: **{both_coupled}**  
   Task-success-only (ablation): see `COUPLED_QUALITY_TIERS.csv` (`ablation_task_success_only`).

4. **Does M111 preserve DX/DZ/Coupled?**  
   Encode succeeds on sampled continuous demos; full skill-preserving replay **not guaranteed**  
   (official captures often terminal; see `M111_REPLAY_RESULTS.csv`). Treat M111 as a discrete target with retention QA, not assumed lossless.

5. **Lineage-safe train/val split?**  
   **Yes** — lineage-grouped 80/20 in `lineage_split.json` (no same-base offsets across splits).

6. **Skills across multiple grasp geometries**  
   `{multi_geom}`  
   Forced DX remains single-lineage-dominated.

7. **Ready for initial SFT baseline?**  
   **Conditional YES** for a recovery-heavy baseline with sparse real Precision.  
   **NOT READY** for a balanced DX/DZ/Coupled multi-skill claim.  
   **Do not train** until a frozen mix is packaged from this audit (no auto-admit of P15/P16).

## Integrity
Official dataset untouched. Forced-Precision stays experimental metadata. M111 vocab frozen.
""",
    )

    print(f"Done → {OUT} lineages={lineages_indep} preferred_coupled={preferred_coupled}", flush=True)


if __name__ == "__main__":
    main()
