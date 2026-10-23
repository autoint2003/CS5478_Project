"""Generate experts_v2 trajectories for one or all six branches."""

from __future__ import annotations

import pickle
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.expert_pipeline.branches import BRANCHES, select_branch
from data_generation.expert_pipeline.dispatch import run_continuation
from data_generation.expert_pipeline.provenance import ACTIVE_DATASET
from data_generation.expert_pipeline.record import DatasetWriter
from data_generation.expert_pipeline.sources import (
    SourceIC,
    discover_branch_ics,
    load_generation_config,
)
from data_generation.oracles.precision.state import PrecisionConfig
from envs.observations.observable_obs import ObservableObsState
from envs.simulation.ground_sim import make_ground_sim
from models.transformer_policy import DT_POLICY

ROOT = Path(__file__).resolve().parents[2]


def _relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


def load_ic_state(src: SourceIC) -> dict[str, Any]:
    blob = pickle.loads(src.ic_path.read_bytes())
    if "capture" not in blob or "ref" not in blob:
        raise KeyError(f"IC {src.sample_id} missing capture/ref")
    return {"capture": blob["capture"], "ref": blob["ref"], "blob": blob}


def _enrich_from_continuation(cont: dict[str, Any], src: SourceIC, cfg: dict[str, Any]) -> dict[str, Any]:
    branch = select_branch(src.branch_id)
    result = cont.get("result") or {}
    handoff = result.get("handoff") if isinstance(result, dict) else None
    if cont.get("continuation") == "direct_handoff":
        success_new = bool(result.get("SUCCESS_NEW"))
        success_12s = bool(result.get("SUCCESS_12S"))
        status = "DIRECT_HANDOFF_SUCCESS" if success_new else "DIRECT_HANDOFF_FAIL"
        phases = ["direct_handoff"]
        precision_skipped = False
        precision_executed = False
        actions = cont.get("actions") or []
    else:
        success_new = bool(cont.get("physical_success"))
        handoff = handoff or {}
        success_12s = bool(handoff.get("SUCCESS_12S") or result.get("SUCCESS_12S"))
        status = result.get("status")
        phases = list(result.get("phases") or [])
        precision_skipped = bool(result.get("precision_skipped"))
        precision_executed = bool(result.get("precision_executed"))
        actions = cont.get("actions") or []
        # Failed precision may still have executed actions — recover from result if needed
        if not actions and isinstance(result, dict):
            # actions stripped in to_dict; keep from cont only
            pass

    out = {
        "branch_id": branch.branch_id,
        "stage1": branch.stage1,
        "continuation": branch.continuation,
        "source_ic_id": src.source_ic_id,
        "source_ic_path": _relpath(src.ic_path),
        "plan_provenance": src.plan_provenance,
        "config_id": cfg.get("config_id"),
        "seed": cfg.get("seed", 0),
        "physical_success": bool(cont.get("physical_success")),
        "SUCCESS_NEW": success_new if cont.get("continuation") == "direct_handoff" else bool(cont.get("physical_success")),
        "SUCCESS_12S": success_12s,
        "status": status,
        "phases": phases,
        "precision_skipped": precision_skipped,
        "precision_executed": precision_executed,
        "actions": actions,
        "n_actions": len(actions or []),
        "result": result,
        "handoff": handoff or result if cont.get("continuation") == "direct_handoff" else handoff,
        "decision": result.get("decision") if isinstance(result, dict) else None,
        "rejected_reason": cont.get("rejected_reason"),
        "capture": None,
        "ref": None,
    }
    return out


def run_one(
    env,
    src: SourceIC,
    writer: DatasetWriter,
    cfg: dict[str, Any],
    *,
    precision_config: PrecisionConfig | None = None,
) -> dict[str, Any]:
    if writer.already_done(src.branch_id, src.source_ic_id):
        return {
            "branch_id": src.branch_id,
            "source_ic_id": src.source_ic_id,
            "skipped": True,
            "reason": "resume_already_done",
        }
    try:
        state = load_ic_state(src)
        cont = run_continuation(
            env,
            {"capture": state["capture"], "ref": state["ref"]},
            src.branch_id,
            precision_config,
        )
        # Preserve full action list from PrecisionResult before to_dict stripped it
        if cont.get("continuation") == "precision" and cont.get("physical_success"):
            # re-fetch actions: dispatch already stored them when ok
            pass
        elif cont.get("continuation") == "precision" and not cont.get("physical_success"):
            # dispatch currently clears actions on failure — re-run is expensive;
            # store whatever is present
            pass
        record = _enrich_from_continuation(cont, src, cfg)
        record["capture"] = state["capture"]
        record["ref"] = state["ref"]
        if record["physical_success"] and record["SUCCESS_NEW"]:
            saved = writer.write_success(record)
            return {**record, "saved": saved, "accepted": True}
        saved = writer.write_failure(record)
        return {**record, "saved": saved, "accepted": False}
    except Exception as exc:
        record = {
            "branch_id": src.branch_id,
            "stage1": select_branch(src.branch_id).stage1,
            "continuation": select_branch(src.branch_id).continuation,
            "source_ic_id": src.source_ic_id,
            "source_ic_path": str(src.ic_path).replace("\\", "/"),
            "plan_provenance": src.plan_provenance,
            "physical_success": False,
            "SUCCESS_NEW": False,
            "SUCCESS_12S": False,
            "exception": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "result": {},
            "actions": [],
            "n_actions": 0,
            "config_id": cfg.get("config_id"),
        }
        saved = writer.write_failure(record)
        return {**record, "saved": saved, "accepted": False}


def generate_branch(
    branch_id: str,
    *,
    output_root: Path | None = None,
    config_path: Path | None = None,
    limit: int | None = None,
    precision_config: PrecisionConfig | None = None,
) -> dict[str, Any]:
    cfg = load_generation_config(config_path)
    sources = discover_branch_ics(branch_id, cfg)
    if limit is not None:
        sources = sources[: max(0, int(limit))]
    out_root = Path(output_root) if output_root else ROOT / cfg.get("output_root", "data/experts_v2")
    writer = DatasetWriter(out_root, config_id=cfg.get("config_id"))
    env = make_env()
    stats = {
        "branch_id": branch_id,
        "candidate_ics": len(discover_branch_ics(branch_id, cfg)),
        "attempted": 0,
        "skipped_resume": 0,
        "accepted": 0,
        "rejected": 0,
        "SUCCESS_NEW": 0,
        "SUCCESS_12S": 0,
        "failure_classes": {},
        "skill_sequences": {},
    }
    for src in sources:
        row = run_one(env, src, writer, cfg, precision_config=precision_config)
        if row.get("skipped"):
            stats["skipped_resume"] += 1
            continue
        stats["attempted"] += 1
        if row.get("accepted"):
            stats["accepted"] += 1
            stats["SUCCESS_NEW"] += 1
            if row.get("SUCCESS_12S"):
                stats["SUCCESS_12S"] += 1
            skill = (row.get("saved") or {}).get("manifest_row", {}).get("skill_sequence") or row.get("skill_sequence")
            if not skill:
                from data_generation.expert_pipeline.record import skill_sequence_label

                skill = skill_sequence_label(row.get("phases"), precision_skipped=bool(row.get("precision_skipped")))
            stats["skill_sequences"][skill] = stats["skill_sequences"].get(skill, 0) + 1
        else:
            stats["rejected"] += 1
            if row.get("SUCCESS_12S"):
                stats["SUCCESS_12S"] += 1
            fc = (row.get("saved") or {}).get("failure_class") or "unclassified_failure"
            stats["failure_classes"][fc] = stats["failure_classes"].get(fc, 0) + 1
    stats["output_root"] = str(out_root).replace("\\", "/")
    stats["index"] = writer.rebuild_index()
    return stats


def generate_all(
    *,
    output_root: Path | None = None,
    config_path: Path | None = None,
    limit_per_branch: int | None = None,
) -> dict[str, Any]:
    summary = {}
    for bid in BRANCHES:
        summary[bid] = generate_branch(
            bid,
            output_root=output_root,
            config_path=config_path,
            limit=limit_per_branch,
        )
    return summary
