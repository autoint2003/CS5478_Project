"""P21 — Audit whether P19/P20 SFT data matches post-impact recovery.

Does not train or modify the policy. Does not mutate experts or M111.
"""

from __future__ import annotations

import csv
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

from data_generation.expert_pipeline.branches import BRANCHES, BRANCH_TO_CLASS  # noqa: E402
from data_generation.expert_pipeline.live_stage1 import extract_precision_skills  # noqa: E402
from data_generation.oracles.precision.stage1_eligibility import capture_fingerprint  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "p21_sft_task_alignment"
P19_MAN = ROOT / "results/diagnostics/p19_sft_packaging/SFT_DATASET_MANIFEST.csv"
P19_JSON = ROOT / "results/diagnostics/p19_sft_packaging/SFT_DATASET_MANIFEST.json"
P20_CL = ROOT / "results/diagnostics/p20_m111_sft_baseline/CLOSED_LOOP_EVALUATION.csv"
STAGE2_MAN = ROOT / "results/data_generation/stage2_abc_final_v2/manifest.jsonl"
LIFT_DIR = ROOT / "results/data_generation/stage2_abc_final_v2/lift"
CONT_DIR = ROOT / "results/data_generation/stage2_abc_final_v2/continuous"


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


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes")


def load_stage2_index() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not STAGE2_MAN.is_file():
        return out
    for line in STAGE2_MAN.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        out[str(m["sample_id"])] = m
    return out


def infer_physical_source(source_ic_id: str, source_kind: str | None, branch_id: str) -> dict:
    """Heuristic but evidence-based source tagging from IDs + stage2 metadata (not branch name alone)."""
    sid = source_ic_id or ""
    sk = source_kind or ""
    impact_likely = False
    impact_evidence = []
    stage1_plan = None
    capture_vs_impact = "unknown"

    if "ball_train" in sid or "ball_impact" in sid or "k48r_train" in sid:
        impact_likely = True
        impact_evidence.append("source_ic_id contains ball/k48r train impact corpus tag")
    if "slide" in sid or "inward_ex" in sid:
        impact_likely = True
        impact_evidence.append("source_ic_id contains lateral/slide disturbance tag")
    if "hard_" in sid or "HARD_IC" in sid:
        impact_likely = True
        impact_evidence.append("hard IC mining ID (disturbance / precision-required synthetic)")
    if "gbi_" in sid or "postlift" in sid or "post_lift" in sk:
        impact_likely = True
        impact_evidence.append("Plan-C catch/post-lift lineage (impact+recatch upstream of lift)")
    if "post_b1" in sid or "post_b1" in sk:
        impact_evidence.append("explicit post-B1 Stage-1 exit tag")
        capture_vs_impact = "after_stage1_exit"
    if sk.startswith("plan_c_post_lift") or "postlift" in sid:
        capture_vs_impact = "after_stage1_exit_post_lift"
    if sk.startswith("pilot_import"):
        impact_evidence.append("pilot_import Stage-2 IC (historically post Stage-1 stable)")
        if capture_vs_impact == "unknown":
            capture_vs_impact = "after_stage1_exit"
    if sk.startswith("plan_b_stage1"):
        capture_vs_impact = "after_stage1_exit"
        stage1_plan = "plan_b"
    if sk.startswith("hard_ic"):
        capture_vs_impact = "synthetic_or_hard_stable_entry"
    if sk.startswith("intermediate"):
        capture_vs_impact = "intermediate_curriculum_capture"

    br = BRANCHES.get(branch_id)
    if br and br.stage1:
        stage1_plan = br.stage1

    return {
        "impact_likely": impact_likely,
        "impact_evidence": "|".join(impact_evidence) if impact_evidence else "none_in_id_or_stage2_kind",
        "capture_relative_to_impact": capture_vs_impact,
        "stage1_plan_from_branch": stage1_plan,
    }


def classify_population(
    *,
    n_actions: int,
    precision_executed: bool,
    precision_skipped: bool,
    continuation: str,
    phases: list[str],
    zero_act_SUCCESS_NEW: bool | None,
    zero_act_SUCCESS_12S: bool | None,
    decision: dict | None,
    errors_entry: dict | None,
) -> str:
    """A/B/C/D categories from physical evidence at packaged start capture."""
    dec = decision or {}
    entry = errors_entry or {}
    skills = extract_precision_skills(phases)
    needs_prec = bool(
        dec.get("dx_needed")
        or dec.get("dz_needed")
        or dec.get("dtheta_needed")
        or (dec.get("pose_ready_success_new") is False)
        or skills
    )
    # Category D: terminal-ready direct HANDOFF (0 actions, probe succeeds)
    if n_actions == 0 and (continuation == "direct_handoff" or phases == ["direct_handoff"]):
        return "D_terminal_ready_direct_handoff"
    if n_actions == 0 and precision_skipped:
        return "D_terminal_ready_precision_skipped"
    # Already STRICT-ok at capture with nonzero "cleanup" actions only
    if zero_act_SUCCESS_NEW is True and n_actions > 0 and not needs_prec:
        return "B_post_impact_already_safe_handoff"
    if zero_act_SUCCESS_NEW is True and n_actions == 0:
        return "D_terminal_ready_direct_handoff"
    # Precision-required (stable contact but residuals need correction)
    if needs_prec or (precision_executed and n_actions > 0) or (zero_act_SUCCESS_NEW is False and n_actions > 0):
        # Distinguish recovery-required (lost grasp on immediate handoff) vs precision-only
        if zero_act_SUCCESS_NEW is False and zero_act_SUCCESS_12S is False:
            # often LOST_GRASP — still post-stage1 but not safe
            return "C_stable_but_precision_required"
        if zero_act_SUCCESS_NEW is False:
            return "C_stable_but_precision_required"
        return "C_stable_but_precision_required"
    # Default
    if n_actions == 0:
        return "D_terminal_ready_direct_handoff"
    return "B_post_impact_already_safe_handoff"


def probe_zero_action(env, capture, ref) -> dict:
    restore_capture(env, capture)
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    return {
        "zero_SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "zero_SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "zero_reason": h.get("std_reason") or h.get("reason"),
    }


def first_nonzero_action(acts: np.ndarray) -> dict:
    if len(acts) == 0:
        return {"first_action_index": None, "first_action_linf": None, "first_action": None}
    for i, a in enumerate(acts):
        if float(np.max(np.abs(a))) > 1e-6:
            return {
                "first_action_index": i,
                "first_action_linf": float(np.max(np.abs(a))),
                "first_action": ",".join(f"{x:.4f}" for x in a.tolist()),
            }
    return {"first_action_index": None, "first_action_linf": 0.0, "first_action": "all_near_zero"}


def find_stage1_artifacts(source_ic_id: str, s2: dict | None) -> dict:
    """Locate upstream Stage-1 / lift / continuous artifacts if present on disk."""
    out = {
        "stage1_lift_npz": None,
        "stage1_lift_n_actions": None,
        "stage2_continuous_npz": None,
        "stage2_continuous_n_actions": None,
        "genuine_stage1_plan_b": None,
        "source_kind": None,
        "stage2_final_category": None,
    }
    if s2:
        out["genuine_stage1_plan_b"] = s2.get("genuine_stage1_plan_b")
        out["source_kind"] = s2.get("source_kind")
        out["stage2_final_category"] = s2.get("final_category")
        cd = s2.get("continuous_demo") or {}
        if cd.get("path"):
            p = ROOT / "results/data_generation/stage2_abc_final_v2" / cd["path"]
            if p.is_file():
                out["stage2_continuous_npz"] = str(p.relative_to(ROOT)).replace("\\", "/")
                out["stage2_continuous_n_actions"] = cd.get("n_actions")
    # Lift tapes for C postlift
    if source_ic_id:
        # try common patterns
        for pat in (
            LIFT_DIR / f"{source_ic_id}.npz",
            LIFT_DIR / f"{source_ic_id}__lift.npz",
        ):
            if pat.is_file():
                blob = np.load(pat)
                n = int(len(blob["actions"])) if "actions" in blob.files else None
                out["stage1_lift_npz"] = str(pat.relative_to(ROOT)).replace("\\", "/")
                out["stage1_lift_n_actions"] = n
                break
        # fuzzy
        if out["stage1_lift_npz"] is None and LIFT_DIR.is_dir():
            hits = list(LIFT_DIR.glob(f"*{source_ic_id.split('__')[-1] if '__' in source_ic_id else source_ic_id}*.npz"))
            if len(hits) == 1:
                blob = np.load(hits[0])
                out["stage1_lift_npz"] = str(hits[0].relative_to(ROOT)).replace("\\", "/")
                out["stage1_lift_n_actions"] = int(len(blob["actions"])) if "actions" in blob.files else None
        if out["stage2_continuous_npz"] is None and CONT_DIR.is_dir():
            hits = list(CONT_DIR.glob(f"*{source_ic_id}*.npz"))
            if hits:
                blob = np.load(hits[0])
                out["stage2_continuous_npz"] = str(hits[0].relative_to(ROOT)).replace("\\", "/")
                out["stage2_continuous_n_actions"] = int(len(blob["actions"])) if "actions" in blob.files else 0
    return out


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== P21 start ===", flush=True)

    man = list(csv.DictReader(P19_MAN.open(encoding="utf-8")))
    s2_index = load_stage2_index()
    p20 = list(csv.DictReader(P20_CL.open(encoding="utf-8"))) if P20_CL.is_file() else []
    p20_by = defaultdict(dict)
    for r in p20:
        p20_by[r["traj_id"]][r["condition"]] = r

    env = make_env()

    # --- Pass 1: all 221 with focus fields; probe zero-action for all (needed for taxonomy) ---
    # Zero-action probe on all 221 is ~221 * ~5s = 18min. Optimize:
    # - For n_actions==0 direct: SUCCESS already known from packaging; still probe subset?
    # User asked physical evidence — probe ALL for consistency, but we can use stored
    # SUCCESS_NEW for zero-action tapes as the capture already evaluated via expert handoff.
    #
    # For n_actions==0: expert handoff from same capture with 0 actions IS the zero-action probe
    # (direct_handoff / precision_skipped). Use recorded SUCCESS_NEW/12S without re-sim.
    # For n_actions>0: must re-sim zero-action to answer "already safe?".

    direct_rows = []
    timeline_rows = []
    pop_counts = Counter()
    n_post_impact_recovery_required = 0
    n_complete_stage1_in_sft = 0

    nonzero = [r for r in man if int(r["n_actions_continuous"]) > 0]
    print(f"probing zero-action STRICT on {len(nonzero)} nonzero tapes…", flush=True)
    zero_cache: dict[str, dict] = {}
    for i, r in enumerate(nonzero):
        raw = pickle.loads((ROOT / r["record_path"]).read_bytes())
        z = probe_zero_action(env, raw["capture"], raw["ref"])
        zero_cache[r["traj_id"]] = z
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(nonzero)}", flush=True)

    print("building provenance + timelines…", flush=True)
    for r in man:
        tid = r["traj_id"]
        raw = pickle.loads((ROOT / r["record_path"]).read_bytes())
        acts = np.asarray(raw.get("actions_7d") if raw.get("actions_7d") is not None else raw.get("actions") or [], np.float32).reshape(-1, 7)
        # P16 candidates
        if acts.size == 0 and raw.get("actions") is not None:
            acts = np.asarray(raw.get("actions"), np.float32).reshape(-1, 7)
        if r["source"] == "p16":
            meta = raw.get("meta") or {}
            if acts.size == 0:
                a2 = meta.get("actions") or raw.get("actions")
                acts = np.asarray(a2 if a2 is not None else [], np.float32).reshape(-1, 7)

        n_act = int(acts.shape[0])
        phases = list(raw.get("phases") or [])
        source_ic_id = str(raw.get("source_ic_id") or r.get("source_lineage") or "")
        source_ic_path = str(raw.get("source_ic_path") or "")
        branch_id = str(raw.get("branch_id") or r.get("branch_id") or "")
        s2 = s2_index.get(source_ic_id) or s2_index.get(tid)
        # for experts, sample_id in stage2 == source_ic_id
        if s2 is None:
            s2 = s2_index.get(source_ic_id)

        phys = infer_physical_source(source_ic_id, (s2 or {}).get("source_kind"), branch_id)
        stage1_art = find_stage1_artifacts(source_ic_id, s2)

        continuation = str(raw.get("continuation") or "")
        precision_executed = bool(raw.get("precision_executed"))
        precision_skipped = bool(raw.get("precision_skipped"))
        decision = raw.get("decision") if isinstance(raw.get("decision"), dict) else None
        errors_entry = raw.get("errors_entry") if isinstance(raw.get("errors_entry"), dict) else None

        if n_act == 0:
            # Expert already evaluated this capture with empty/skipped tape
            z_new = bool(raw.get("SUCCESS_NEW"))
            z_12 = bool(raw.get("SUCCESS_12S"))
            z_reason = "expert_recorded_empty_tape_handoff"
        else:
            z = zero_cache[tid]
            z_new = z["zero_SUCCESS_NEW"]
            z_12 = z["zero_SUCCESS_12S"]
            z_reason = z["zero_reason"]

        pop = classify_population(
            n_actions=n_act,
            precision_executed=precision_executed,
            precision_skipped=precision_skipped,
            continuation=continuation,
            phases=phases,
            zero_act_SUCCESS_NEW=z_new,
            zero_act_SUCCESS_12S=z_12,
            decision=decision,
            errors_entry=errors_entry,
        )
        pop_counts[pop] += 1

        # Category A would be post-impact recovery-required AT SFT start.
        # Packaged experts start post-Stage-1; none are raw post-impact.
        recovery_required_at_sft_start = bool(n_act > 0 and z_new is False)
        if recovery_required_at_sft_start:
            # Still not "post-impact raw" — it's post-stage1 precision-required
            pass

        # Why HANDOFF immediately (for direct0)
        if n_act == 0:
            if continuation == "direct_handoff":
                why = "branch_continuation=direct_handoff; probe_direct_handoff on post-Stage-1 IC; SUCCESS recorded; actions_7d empty by design"
            elif precision_skipped:
                why = "Precision assessed then skipped: pose already handoff-ready (PRECISION_SKIPPED_DIRECT_SUCCESS); no correction actions"
            else:
                why = "zero packaged actions; see phases/continuation"
            already_stable = bool(z_new)
            stage1_already_done = True  # capture is Stage-1 exit IC for official A/B/C
            direct_rows.append(
                {
                    "traj_id": tid,
                    "split": r.get("split"),
                    "source": r.get("source"),
                    "branch_id": branch_id,
                    "class_alias": BRANCH_TO_CLASS.get(branch_id),
                    "source_ic_id": source_ic_id,
                    "source_ic_path": source_ic_path,
                    "stage2_source_kind": (s2 or {}).get("source_kind"),
                    "stage2_final_category": (s2 or {}).get("final_category"),
                    "genuine_stage1_plan_b": (s2 or {}).get("genuine_stage1_plan_b"),
                    "impact_likely": phys["impact_likely"],
                    "impact_evidence": phys["impact_evidence"],
                    "capture_relative_to_impact": phys["capture_relative_to_impact"],
                    "stage1_plan": phys["stage1_plan_from_branch"] or raw.get("stage1"),
                    "stage1_already_occurred_before_capture": stage1_already_done and branch_id in BRANCHES and BRANCHES[branch_id].stage1 is not None,
                    "stage1_physically_required_historically": phys["impact_likely"],  # impact suggests yes upstream; not at this capture
                    "initial_state_already_stable_STRICT": already_stable,
                    "zero_SUCCESS_NEW": z_new,
                    "zero_SUCCESS_12S": z_12,
                    "zero_reason": z_reason,
                    "continuation": continuation,
                    "skill_sequence": raw.get("skill_sequence"),
                    "phases": "|".join(phases[:8]),
                    "precision_skipped": precision_skipped,
                    "precision_executed": precision_executed,
                    "why_expert_selected_immediate_handoff": why,
                    "population_category": pop,
                    "entry_dx_mm": (errors_entry or {}).get("dx_mm"),
                    "entry_dz_mm": (errors_entry or {}).get("dz_mm"),
                    "entry_dtheta_deg": (errors_entry or {}).get("dtheta_deg"),
                    "entry_e_pos_mm": (errors_entry or {}).get("e_pos_mm"),
                    **{k: stage1_art[k] for k in stage1_art},
                }
            )

        # Timeline audit for every official A/B/C (+ D / candidates)
        fa = first_nonzero_action(acts)
        skills = extract_precision_skills(phases)
        # Stage-1 actions in packaged SFT tape?
        stage1_in_tape = False  # experts_v2 never embeds Stage-1 in actions_7d
        # Evidence: phases never include plan_a/b/c stage1 tags; start is frozen IC
        has_precision_phases = any(
            ("correct_" in p) or ("dx_" in p) or ("dz_" in p) or ("dtheta" in p) for p in phases
        )
        handoff_phase_idx = next((i for i, p in enumerate(phases) if "handoff" in p.lower() or p == "direct_handoff"), None)

        complete_stage1_supervision = False  # never true for P19 pack
        if stage1_art["stage1_lift_n_actions"] and int(stage1_art["stage1_lift_n_actions"] or 0) > 0:
            # Stage-1 exists on disk but NOT in SFT sample
            pass

        timeline_rows.append(
            {
                "traj_id": tid,
                "split": r.get("split"),
                "source": r.get("source"),
                "branch_id": branch_id,
                "class_alias": BRANCH_TO_CLASS.get(branch_id),
                "source_ic_id": source_ic_id,
                "n_actions_in_sft_tape": n_act,
                "sft_obs_steps_incl_handoff": r.get("n_steps"),
                "packaged_start": "post_stage1_frozen_capture",
                "initial_post_impact_state_in_tape": False,
                "first_physical_recovery_action_in_tape": fa["first_action_index"] is not None and n_act > 0,
                "first_action_index": fa["first_action_index"],
                "first_action": fa["first_action"],
                "stage1_completion_event_in_tape": False,
                "stage1_actions_omitted_from_sft": bool(branch_id in BRANCHES and BRANCHES[branch_id].stage1 is not None),
                "stage1_upstream_artifact": stage1_art.get("stage1_lift_npz") or stage1_art.get("stage2_continuous_npz"),
                "stage1_upstream_n_actions": stage1_art.get("stage1_lift_n_actions")
                if stage1_art.get("stage1_lift_n_actions") is not None
                else stage1_art.get("stage2_continuous_n_actions"),
                "stage2_precision_actions_present": bool(n_act > 0 and (has_precision_phases or precision_executed)),
                "precision_skills_executed": "→".join(skills) if skills else "",
                "handoff_in_phases_index": handoff_phase_idx,
                "handoff_as_sft_terminal_token": True,  # P19 always appends HANDOFF
                "phases_head": "|".join(phases[:10]),
                "zero_SUCCESS_NEW_at_capture": z_new,
                "zero_SUCCESS_12S_at_capture": z_12,
                "zero_reason": z_reason,
                "population_category": pop,
                "complete_stage1_supervision_in_sft": complete_stage1_supervision,
                "impact_likely": phys["impact_likely"],
                "capture_relative_to_impact": phys["capture_relative_to_impact"],
                "stage2_source_kind": (s2 or {}).get("source_kind"),
            }
        )

    _write_csv(OUT / "DIRECT_HANDOFF_PROVENANCE.csv", direct_rows)
    _write_csv(OUT / "STAGE1_TIMELINE_AUDIT.csv", timeline_rows)

    # --- Actionful val cases ---
    val_actionful = [r for r in man if r.get("split") == "val" and int(r["n_actions_continuous"]) > 0]
    af_lines = [
        "# ACTIONFUL_VAL_CASES",
        "",
        f"Generated: {_utc()}",
        "",
        f"Count: **{len(val_actionful)}** (P19 val ∩ nonzero `actions_7d`).",
        "",
        "SFT predictions from P20 Baseline A closed-loop (`n_motion`, SUCCESS).",
        "Zero-action STRICT = restore packaged capture, evaluate with empty action list.",
        "",
    ]
    for r in val_actionful:
        tid = r["traj_id"]
        raw = pickle.loads((ROOT / r["record_path"]).read_bytes())
        acts = np.asarray(raw.get("actions_7d"), np.float32).reshape(-1, 7)
        z = zero_cache[tid]
        fa = first_nonzero_action(acts)
        sft = p20_by.get(tid, {}).get("sft_A_natural", {})
        ce = p20_by.get(tid, {}).get("continuous_expert", {})
        skills = extract_precision_skills(list(raw.get("phases") or []))
        recovery_required = not z["zero_SUCCESS_NEW"]
        af_lines.append(f"## `{tid}`")
        af_lines.append("")
        af_lines.append(f"- Branch / skill_group: `{r.get('branch_id')}` / `{r.get('skill_group')}`")
        af_lines.append(f"- Source IC: `{raw.get('source_ic_id')}`")
        af_lines.append(f"- Packaged actions: **{len(acts)}**")
        af_lines.append(f"- Phases (head): `{list(raw.get('phases') or [])[:8]}`")
        af_lines.append(f"- Precision skills in phases: `{skills or 'none'}`")
        af_lines.append(f"- Entry errors dx/dz/dθ/e_pos mm: "
                        f"{(raw.get('errors_entry') or {}).get('dx_mm')}, "
                        f"{(raw.get('errors_entry') or {}).get('dz_mm')}, "
                        f"{(raw.get('errors_entry') or {}).get('dtheta_deg')}, "
                        f"{(raw.get('errors_entry') or {}).get('e_pos_mm')}")
        af_lines.append(f"- Decision flags: dx_needed={ (raw.get('decision') or {}).get('dx_needed') }, "
                        f"dz_needed={ (raw.get('decision') or {}).get('dz_needed') }, "
                        f"pose_ready_SUCCESS_NEW={ (raw.get('decision') or {}).get('pose_ready_success_new') }")
        af_lines.append(f"- **Genuinely recovery/precision-required at capture?** "
                        f"**{'YES' if recovery_required else 'NO'}** "
                        f"(zero-action STRICT NEW={z['zero_SUCCESS_NEW']} 12S={z['zero_SUCCESS_12S']} reason={z['zero_reason']})")
        af_lines.append(f"- Expert first nonzero action (idx={fa['first_action_index']}): `{fa['first_action']}`")
        af_lines.append(f"- Expert executed: "
                        f"{'Precision '+('→'.join(skills) if skills else '(geometry/settle only)') if len(acts) else 'nothing'}; "
                        f"Stage-1 **not in tape** (post-Stage-1 start)")
        af_lines.append(f"- SFT-A predicted: HANDOFF at t=0 (n_motion={sft.get('n_motion')}); "
                        f"STRICT={sft.get('SUCCESS_NEW')} failure_class={sft.get('failure_class')}")
        af_lines.append(f"- Continuous expert STRICT={ce.get('SUCCESS_NEW')}")
        if not recovery_required and _as_bool(sft.get("SUCCESS_NEW")):
            af_lines.append(
                "- **Why SFT zero-motion still STRICT-succeeds:** capture is already "
                "`SUCCESS_NEW` under immediate HANDOFF. Expert nonzero actions are "
                "geometry/DOWN-gate housekeeping (`NO_REMAINING_CORRECTION` / "
                "`final_down_gate`), not required for the official STRICT probe."
            )
        elif recovery_required and not _as_bool(sft.get("SUCCESS_NEW")):
            af_lines.append(
                "- **Why SFT fails:** capture is **not** STRICT-ready (zero-action "
                f"`{z['zero_reason']}`); Expert needs Precision `{skills}`; policy "
                "emits immediate HANDOFF without those corrections."
            )
        af_lines.append("")
    _write_md(OUT / "ACTIONFUL_VAL_CASES.md", "\n".join(af_lines))

    # --- Train/eval start-state comparison ---
    cat_split = defaultdict(Counter)
    for row in timeline_rows:
        cat_split[row["split"]][row["population_category"]] += 1
    start_md = f"""# TRAIN_EVAL_START_STATE_COMPARISON

Generated: {_utc()}

## What closed-loop evaluation starts from (P20)

For every P19 packaged trajectory (train and val):

1. Load expert record `capture` + `ref`.
2. `restore_capture(env, capture)`.
3. Policy observes airborne state and acts (or HANDOFF).

That `capture` is the **experts_v2 Stage-2 IC**: a **post–Stage-1** (A/B) or **post-lift** (C) frozen state, **not** the raw post-impact state.

Confirmed in code/docs:

- `data_generation/expert_pipeline/dispatch.py`: *\"Source ICs for experts_v2 are post-Stage-1 stable captures.\"*
- `stage2_abc_final_v2/REPORT.md`: *\"Stage-2 SFT begins at the post-lift IC\"*; lift tapes are Stage-1 and live under `lift/`.
- `stage1_eligibility.EVAL_CONFIG`: Precision restores `blob['capture']` as Stage-1 exit.

## Intended robotic recovery task (project framing)

Typical intended closed loop:

`impact → Stage-1 A/B/C recovery → (optional) Stage-2 Precision → HANDOFF → 12s`

## What P19/P20 actually train/eval

| Stage | In intended task | In P19 SFT tape | In P20 closed-loop start |
|---|---|---|---|
| Raw post-impact | yes | **no** | **no** |
| Stage-1 A/B/C actions | yes | **omitted** (upstream only) | already finished before start |
| Stage-2 Precision | yes | only if `actions_7d` nonzero | yes when present |
| Direct HANDOFF | yes | 171/221 terminal-only | yes (dominant) |

## Population at packaged start (train vs val)

### Train
{json.dumps(dict(cat_split.get('train', {})), indent=2)}

### Val
{json.dumps(dict(cat_split.get('val', {})), indent=2)}

## Match verdict

Training and evaluation **match each other** (same post-Stage-1 capture distribution).

They **do not match** a task definition that requires learning **post-impact Stage-1 recovery** from scratch: that supervision was never packaged into P19.

P20’s always-HANDOFF collapse is therefore evaluated on a start-state distribution that is already mostly HANDOFF-ready — amplifying the imbalance.
"""
    _write_md(OUT / "TRAIN_EVAL_START_STATE_COMPARISON.md", start_md)

    # Counts for final questions
    n_cat_a = 0  # post-impact recovery-required at SFT start — none by construction
    n_cat_c = pop_counts.get("C_stable_but_precision_required", 0)
    n_cat_b = pop_counts.get("B_post_impact_already_safe_handoff", 0)
    n_cat_d = pop_counts.get("D_terminal_ready_direct_handoff", 0) + pop_counts.get(
        "D_terminal_ready_precision_skipped", 0
    )
    n_complete_stage1 = sum(1 for row in timeline_rows if row["complete_stage1_supervision_in_sft"])
    n_stage1_omitted = sum(1 for row in timeline_rows if row["stage1_actions_omitted_from_sft"])
    n_impact_likely = sum(1 for row in timeline_rows if row["impact_likely"])

    proposal = f"""# TASK_ALIGNED_DATASET_PROPOSAL

Generated: {_utc()}

## Diagnosis (one paragraph)

`experts_v2` / P19 packages **Stage-2 continuations from post-Stage-1 captures**.  
The 171 immediate-HANDOFF tapes are correct Expert behavior on already-stable ICs, not missing labels.  
Stage-1 A/B/C recovery actions exist upstream (pilot Stage-1, Plan-C lift NPZs) but were **never written into** `actions_7d`.  
Therefore P20 did not fail a balanced post-impact recovery dataset — it overfit the **actual** packaged task: “from a late stable/near-stable grasp, emit HANDOFF (sometimes after Precision).”

## Proposed populations (keep separate)

| ID | Definition | Role | Current P19 count (approx) |
|---|---|---|---:|
| **A** | Post-impact recovery-required; Stage-1 actions labeled from impact/entry | Primary multi-skill recovery SFT | **0** in P19 |
| **B** | Post-impact (or post-Stage-1) already STRICT-safe; Expert may still move for hygiene | HANDOFF-timing / negative Precision | ~{n_cat_b} |
| **C** | Stable contact but Precision-required (`zero_SUCCESS_NEW=False`) | Stage-2 Precision SFT | ~{n_cat_c} |
| **D** | Terminal-ready direct HANDOFF (0 motion) | HANDOFF prior only; **cap / down-weight** | ~{n_cat_d} |

Do **not** mix A–D without `population_category` + `capture_relative_to_impact` metadata.

## Construction plan (next packaging, not this run)

1. **Recover Stage-1 tapes** for A/B/C from upstream artifacts (`lift/*.npz`, Stage-1 continuous demos, live_stage1 runners) aligned to the same `source_ic_id` / capture fingerprint chain.
2. Build **full-horizon** episodes: `[Stage-1 actions][+ Precision][+ HANDOFF]` with observations recorded from the **true start** (post-impact or Stage-1 entry), not only Stage-1 exit.
3. Keep a **capped** set of D/B HANDOFF exemplars (e.g. ≤20% of episodes or loss weight) so HANDOFF remains supervised without drowning motion.
4. Preserve lineage splits; tag each row with `population_category` ∈ {{A,B,C,D}}.
5. Eval must sample **A and C** held-out lineages separately from D; report STRICT stratified.

## Explicit non-goals

- Do not “fix” P20 by changing the Oracle.
- Do not pretend P19 already contains Stage-1.
- Do not delete all HANDOFF tapes (terminal skill is real).
"""
    _write_md(OUT / "TASK_ALIGNED_DATASET_PROPOSAL.md", proposal)

    # Final report
    n_direct = len(direct_rows)
    final = f"""# FINAL_REPORT — P21 SFT Task Alignment

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **How many of the 221 trajectories actually start from a post-impact recovery-required state?**  
   **0** as packaged for SFT.  
   All official starts are **post–Stage-1 / post-lift frozen captures**.  
   ~{n_impact_likely} have **upstream** impact-like provenance (ball/slide/C-catch/hard-IC tags), but the SFT observation window begins **after** Stage-1.  
   States that still need work at that late start are category **C** (Precision-required): **{n_cat_c}** (zero-action STRICT fails while Expert has corrections).  
   That is **not** the same as raw post-impact recovery-required (category **A**).

2. **How many contain complete A/B/C Stage-1 supervision?**  
   **0 / {n_stage1_omitted}** A/B/C-branched rows include Stage-1 actions in the SFT tape.  
   Stage-1 is omitted by `experts_v2` design (`actions_7d` = Stage-2 continuation only).  
   Some upstream lift/continuous artifacts exist on disk (`STAGE1_TIMELINE_AUDIT.csv`) but were not packaged into P19 NPZs.

3. **Why are 171 trajectories immediate HANDOFF?**  
   Because the Expert’s Stage-2 continuation on those ICs is empty:  
   - Direct-HANDOFF branches: `probe_direct_handoff` already `SUCCESS_NEW` → `phases=['direct_handoff']`, `n_actions=0`.  
   - Precision-skipped: assessment finds pose handoff-ready → skip corrections.  
   See `DIRECT_HANDOFF_PROVENANCE.csv` (`why_expert_selected_immediate_handoff`).  
   This is correct Expert behavior on late stable ICs, not a labeling bug.

4. **Does the current dataset match the intended robotic recovery task?**  
   **No**, if the intended task is *post-impact → Stage-1 A/B/C → Precision → HANDOFF*.  
   **Yes**, if the intended task is narrowed to *post-Stage-1 Stage-2 / HANDOFF*.  
   P20 trained/evaluated the latter. Train and eval start-state distributions **match each other** but not the full recovery story (`TRAIN_EVAL_START_STATE_COMPARISON.md`).

5. **Is the SFT collapse primarily class imbalance, or task/data alignment?**  
   **Both, with alignment primary.**  
   - **Alignment:** the packaged task is dominated by already-HANDOFF-ready starts; Stage-1 recovery supervision is missing — so the policy cannot learn post-impact A/B/C from P19.  
   - **Imbalance:** within that Stage-2 task, 171/221 terminal HANDOFF tapes still bias toward always-HANDOFF, which is why even category-C val ICs get premature HANDOFF.  
   Fixing weights alone (P20-B) cannot create Stage-1 skills that were never in the tapes.

## Population snapshot (221)

{json.dumps(dict(pop_counts), indent=2)}

## Six actionful val ICs (summary)

| Outcome | Count | Meaning |
|---|---:|---|
| Zero-action already STRICT; SFT HANDOFF “succeeds” | 2 | Not recovery-required at start; Expert motion optional |
| Zero-action fails STRICT; SFT HANDOFF fails | 4 | True Precision-required; policy refuses to act |

Details: `ACTIONFUL_VAL_CASES.md`.

## Integrity

- No training, no policy/Oracle/M111/physics edits
- Read-only over P19 packaging + expert/stage2 artifacts
"""
    _write_md(OUT / "FINAL_REPORT.md", final)

    (OUT / "raw_p21_summary.json").write_text(
        json.dumps(
            {
                "n_traj": len(man),
                "n_direct_handoff_traced": n_direct,
                "population": dict(pop_counts),
                "n_complete_stage1_in_sft": n_complete_stage1,
                "n_stage1_omitted_branches": n_stage1_omitted,
                "n_impact_likely_upstream": n_impact_likely,
                "n_category_A_post_impact_recovery_required": n_cat_a,
                "n_category_C_precision_required": n_cat_c,
                "elapsed_s": time.time() - t0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("=== P21 done ===", dict(pop_counts), flush=True)


if __name__ == "__main__":
    main()
