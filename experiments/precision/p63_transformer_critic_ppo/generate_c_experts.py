"""Generate additional Plan-C ICs + SUCCESS_NEW expert trajectories (impact-grid method).

Uses the same pipeline as ``scale_plan_c_sidewall_experts``:
``diverse_impact_configs`` → ``build_physical_root`` → ``search_catch`` →
``lift_and_validate`` → ``exact_replay_3of3`` → ``pack_expert``.

Writes under ``results/experiments/p63_transformer_critic_ppo/c_experts/``.
Skips fingerprints already present in scale50 / golden / P21 C train pool.
Also writes MixedStart-ready episode pickles for GRPO/PPO training.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path

import numpy as np

from data_generation.oracles.plan_c.expert_search import (
    build_physical_root,
    diverse_impact_configs,
    exact_replay_3of3,
    lift_and_validate,
    pack_expert,
    physical_root_fingerprint,
    search_catch,
)
from envs.simulation.ground_sim import make_ground_sim
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs
from experiments.precision.p63_transformer_critic_ppo.paths import OUT, ROOT, SEED

SCALE50 = ROOT / "results/diagnostics/ground_plan_c_scale50/raw/accepted"
GOLDEN = ROOT / "results/diagnostics/ground_plan_c_final_success/raw"
SIDEWALL = ROOT / "results/diagnostics/ground_plan_c_sidewall_research/raw"
C_OUT = OUT / "c_experts"
ACCEPTED = C_OUT / "accepted"
EPISODES = C_OUT / "episodes"


def _jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): _jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsafe(x) for x in o]
    return str(o)


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsafe(obj), indent=2), encoding="utf-8")


def _append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    keys = list(row.keys())
    if exists:
        with path.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            keys = list(r.fieldnames or keys)
            for k in row:
                if k not in keys:
                    keys.append(k)
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: _jsafe(row.get(k)) for k in keys})


def _load_known_fingerprints() -> set[str]:
    known: set[str] = set()
    for folder in (SCALE50, GOLDEN, SIDEWALL, ACCEPTED):
        if not folder.is_dir():
            continue
        for p in folder.glob("*SUCCESS_NEW*.pkl"):
            try:
                blob = pickle.loads(p.read_bytes())
            except Exception:
                continue
            fp = blob.get("fingerprint")
            if fp:
                known.add(str(fp))
            snap = blob.get("catch_snap") or blob.get("snap")
            cfg = blob.get("cfg") or {}
            if snap is not None and cfg:
                try:
                    known.add(physical_root_fingerprint(cfg, snap))
                except Exception:
                    pass
    # Also block P21 C traj stems loosely via root_id in jobs
    try:
        for j in collect_jobs():
            if j.get("skill") != "C":
                continue
            tid = str(j.get("traj_id") or "")
            # gbi_XXX often appears in traj_id
            for part in tid.replace("-", "_").split("_"):
                if part.startswith("gbi") and len(part) >= 6:
                    known.add(f"traj::{tid}")
                    break
            known.add(f"traj::{tid}")
    except Exception as e:
        print(f"warn: could not load P21 jobs ({e})", flush=True)
    return known


def _episode_from_expert(expert: dict, traj_id: str) -> dict:
    acts = expert.get("actions")
    if acts is None:
        acts = expert["prefix_actions"]
    acts = np.asarray(acts, np.float32)
    return {
        "traj_id": traj_id,
        "family": "C",
        "lineage_id": expert.get("fingerprint") or traj_id,
        "skill": "C",
        "source_corpus": "p63_plan_c_generate",
        "start_kind": "catch_snap",
        "catch_snap": expert["catch_snap"],
        "capture": None,
        "ref": expert["ref"],
        "actions": acts,
        "n_stage1": int(len(acts)),
        "n_precision": 0,
        "includes_precision": False,
        "mechanism": expert.get("mechanism"),
        "horizon_kind": "recovery_stage1_only",
        "cfg": expert.get("cfg"),
        "fingerprint": expert.get("fingerprint"),
        "grasp_kind": expert.get("grasp_kind"),
        "SUCCESS_NEW": True,
    }


def _mixed_start_blob(ep: dict) -> dict:
    """Serializable MixedStart payload (ep kind)."""
    return {
        "skill": "C",
        "traj_id": ep["traj_id"],
        "family": "C",
        "kind": "ep",
        "ep": ep,
    }


def run(
    *,
    target: int = 32,
    max_configs: int = 600,
    rollouts_per_root: int = 36,
    max_sidewall_try: int = 4,
    wall_s: float = 20000.0,
    seed: int = SEED + 63063,
) -> dict:
    C_OUT.mkdir(parents=True, exist_ok=True)
    ACCEPTED.mkdir(parents=True, exist_ok=True)
    EPISODES.mkdir(parents=True, exist_ok=True)

    known = _load_known_fingerprints()
    print(f"=== known fingerprints/traj keys: {len(known)} ===", flush=True)

    # Resume prior P63 accepts
    success_rows: list[dict] = []
    accepted_fps: set[str] = set()
    for p in sorted(ACCEPTED.glob("*__SUCCESS_NEW_expert.pkl")):
        blob = pickle.loads(p.read_bytes())
        fp = str(blob.get("fingerprint") or "")
        if not fp or fp in accepted_fps:
            continue
        accepted_fps.add(fp)
        known.add(fp)
        success_rows.append(
            {
                "root_id": blob.get("root_id"),
                "fingerprint": fp,
                "traj_id": blob.get("traj_id"),
                "SUCCESS_NEW": 1,
                "resumed": 1,
                "approach": (blob.get("cfg") or {}).get("approach"),
                "speed": (blob.get("cfg") or {}).get("speed"),
            }
        )
    print(f"=== resume {len(success_rows)} P63 accepted ===", flush=True)

    rng = np.random.default_rng(seed)
    configs = diverse_impact_configs(max_configs, seed=seed)
    # Prefer configs whose config_id / knob set is less likely already mined under scale50 seed
    rng.shuffle(configs)
    print(f"=== search {len(configs)} configs; target={target}; seed={seed} ===", flush=True)

    t0 = time.time()
    funnel = {
        "configs_tried": 0,
        "roots_built": 0,
        "genuine_loss": 0,
        "dup_skipped": 0,
        "sidewall_hit": 0,
        "success_new": len(success_rows),
        "replay_ok": len(success_rows),
    }

    for ci, cfg in enumerate(configs):
        if len(success_rows) >= target:
            break
        if time.time() - t0 > wall_s:
            print("wall-clock budget exhausted", flush=True)
            break

        funnel["configs_tried"] += 1
        sim, _ = make_ground_sim()
        root = build_physical_root(sim, cfg, max_s=1.2)
        funnel["roots_built"] += 1
        if root is None:
            _append_csv(C_OUT / "FUNNEL.csv", {"config_id": cfg["config_id"], "stage": "build", "ok": 0})
            continue
        funnel["genuine_loss"] += 1
        fp = str(root["fingerprint"])
        if fp in known or fp in accepted_fps:
            funnel["dup_skipped"] += 1
            _append_csv(
                C_OUT / "FUNNEL.csv",
                {"config_id": cfg["config_id"], "stage": "dup_fp", "ok": 0, "fingerprint": fp},
            )
            continue

        rid = cfg["config_id"]
        print(
            f"[{len(success_rows)}/{target}] {rid} slip={root['slip_s']:.3f} "
            f"ap={cfg['approach']} sp={cfg['speed']} z={cfg['z']} "
            f"tilt={cfg['tilt_deg']} ax={cfg['axial_offset']}",
            flush=True,
        )

        catch = search_catch(
            sim, root["snap"], root["ref"], rng=rng, max_rollouts=rollouts_per_root
        )
        sidewall = catch["sidewall"]
        if sidewall:
            funnel["sidewall_hit"] += 1

        accepted_here = False
        row = {
            "config_id": rid,
            "fingerprint": fp,
            "slip_s": root["slip_s"],
            "n_sidewall": len(sidewall),
            "SUCCESS_NEW": 0,
        }
        for hi, hit in enumerate(sidewall[:max_sidewall_try]):
            val = lift_and_validate(sim, root["snap"], root["ref"], hit)
            if not val.get("SUCCESS_NEW"):
                row[f"try{hi}_reason"] = val.get("reason")
                continue
            funnel["success_new"] += 1
            rep = exact_replay_3of3(
                root["snap"], root["ref"], val["prefix_actions"], val["elev_snap"]
            )
            if not rep["ok"]:
                row["replay_fail"] = 1
                print(f"  replay fail elev={rep['elev_wins']}/3", flush=True)
                continue
            funnel["replay_ok"] += 1
            expert = pack_expert(
                root_id=rid,
                fingerprint=fp,
                cfg=cfg,
                snap=root["snap"],
                ref=root["ref"],
                validated=val,
                events=root["events"],
            )
            expert["exact_replay"] = rep
            traj_id = f"plan_c__{rid}__{fp[:12]}"
            expert["traj_id"] = traj_id
            out_name = f"{rid}__{fp[:10]}__SUCCESS_NEW_expert.pkl"
            (ACCEPTED / out_name).write_bytes(pickle.dumps(expert))

            ep = _episode_from_expert(expert, traj_id)
            (EPISODES / f"{traj_id}.pkl").write_bytes(pickle.dumps(ep))
            (EPISODES / f"{traj_id}__mixed_start.pkl").write_bytes(
                pickle.dumps(_mixed_start_blob(ep))
            )

            accepted_fps.add(fp)
            known.add(fp)
            success_rows.append(
                {
                    "root_id": rid,
                    "fingerprint": fp,
                    "traj_id": traj_id,
                    "SUCCESS_NEW": 1,
                    "grasp_kind": expert.get("grasp_kind"),
                    "approach": cfg.get("approach"),
                    "speed": cfg.get("speed"),
                    "z": cfg.get("z"),
                    "tilt_deg": cfg.get("tilt_deg"),
                    "axial_offset": cfg.get("axial_offset"),
                    "hit_i": hi,
                    "expert_pkl": str((ACCEPTED / out_name).relative_to(ROOT)).replace("\\", "/"),
                    "episode_pkl": str((EPISODES / f"{traj_id}.pkl").relative_to(ROOT)).replace(
                        "\\", "/"
                    ),
                }
            )
            row["SUCCESS_NEW"] = 1
            accepted_here = True
            print(
                f"  ACCEPTED {traj_id} kind={expert.get('grasp_kind')} n={len(success_rows)}",
                flush=True,
            )
            break

        _append_csv(C_OUT / "FUNNEL.csv", row)
        _write_json(
            C_OUT / "PROGRESS.json",
            {
                "n_success": len(success_rows),
                "target": target,
                "configs_tried": ci + 1,
                "wall_s": time.time() - t0,
                "funnel": funnel,
            },
        )
        if not accepted_here and (ci + 1) % 10 == 0:
            print(f"  … tried {ci+1} configs, accepted {len(success_rows)}", flush=True)

    # Manifests
    man_path = C_OUT / "C_NEW_IC_MANIFEST.jsonl"
    with man_path.open("w", encoding="utf-8") as f:
        for r in success_rows:
            f.write(json.dumps(_jsafe(r)) + "\n")

    # Also refresh expand-style jsonl for training loaders
    train_man = OUT / "C_DATA_MANIFEST.jsonl"
    with train_man.open("w", encoding="utf-8") as f:
        for i, r in enumerate(success_rows):
            f.write(
                json.dumps(
                    {
                        "index": i,
                        "skill": "C",
                        "traj_id": r["traj_id"],
                        "family": r.get("root_id"),
                        "kind": "ep",
                        "source": "p63_generate_c_experts",
                        "fingerprint": r.get("fingerprint"),
                        "expert_pkl": r.get("expert_pkl"),
                        "episode_pkl": r.get("episode_pkl"),
                        "provenance": {
                            "method": "diverse_impact_configs+search_catch+lift_and_validate",
                            "excluded_scale50_golden_p21": True,
                            "replay_3of3": True,
                            "SUCCESS_NEW": True,
                        },
                    }
                )
                + "\n"
            )

    summary = {
        "n_success": len(success_rows),
        "target": target,
        "seed": seed,
        "wall_s": time.time() - t0,
        "funnel": funnel,
        "accepted_dir": str(ACCEPTED),
        "episodes_dir": str(EPISODES),
        "manifest": str(man_path),
        "train_manifest": str(train_man),
    }
    _write_json(C_OUT / "SUMMARY.json", summary)
    (C_OUT / "REPORT.md").write_text(
        "\n".join(
            [
                "# P63 new Plan-C IC + expert generation",
                "",
                f"- Accepted SUCCESS_NEW (replay-verified): **{len(success_rows)}** / {target}",
                f"- Seed: `{seed}` (distinct from scale50 default)",
                f"- Wall: {summary['wall_s']:.1f}s",
                f"- Funnel: `{json.dumps(funnel)}`",
                "",
                "Experts: `c_experts/accepted/*__SUCCESS_NEW_expert.pkl`",
                "Episodes (MixedStart-ready): `c_experts/episodes/*.pkl`",
                "Manifest: `C_DATA_MANIFEST.jsonl`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def load_new_c_mixed_starts() -> list:
    """Load MixedStart objects from generated episode pickles."""
    from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart

    out = []
    if not EPISODES.is_dir():
        return out
    for p in sorted(EPISODES.glob("*__mixed_start.pkl")):
        blob = pickle.loads(p.read_bytes())
        out.append(
            MixedStart(
                skill=blob["skill"],
                traj_id=blob["traj_id"],
                family=blob["family"],
                kind=blob["kind"],
                ep=blob["ep"],
            )
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=32)
    ap.add_argument("--max-configs", type=int, default=600)
    ap.add_argument("--rollouts-per-root", type=int, default=36)
    ap.add_argument("--max-sidewall-try", type=int, default=4)
    ap.add_argument("--wall-s", type=float, default=20000.0)
    ap.add_argument("--seed", type=int, default=SEED + 63063)
    args = ap.parse_args()
    run(
        target=args.target,
        max_configs=args.max_configs,
        rollouts_per_root=args.rollouts_per_root,
        max_sidewall_try=args.max_sidewall_try,
        wall_s=args.wall_s,
        seed=args.seed,
    )
