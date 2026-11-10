"""Phase 4 — Build expert BC replay from successful takeover continuations."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from envs.observations.airborne_obs import observe_airborne
from envs.observations.observable_obs import ObservableObsState
from envs.step_legal import step_legal
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p52_independent_vs_tree_sampling.branch_state import restore_branch_state
from experiments.precision.p55_expert_takeover.paths import OUT, REPLAY_DIR
from models.m111_checkpoint import load_m111_book
from models.transformer_policy import DT_POLICY
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory


def _candidate_captures() -> dict[str, str]:
    cand_path = OUT / "TAKEOVER_CANDIDATES.json"
    if not cand_path.is_file():
        return {}
    blob = json.loads(cand_path.read_text(encoding="utf-8"))
    out = {}
    for c in blob.get("candidates") or []:
        cp = c.get("capture_path")
        if cp and c.get("candidate_id"):
            out[c["candidate_id"]] = cp
    return out


def _replay_obs_from_capture(env, capture_path: str, acts: np.ndarray) -> np.ndarray | None:
    """Restore branch capture and replay expert actions to record 39-D obs before each step."""
    try:
        branch = pickle.loads(Path(capture_path).read_bytes())
        restore_branch_state(env, branch)
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        obs_list = []
        for a in acts:
            obs = observe_airborne(
                env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist
            )
            obs_list.append(np.asarray(obs, np.float32))
            aa = np.asarray(a, np.float32).reshape(7)
            step_legal(env.sim, aa)
            env._prev = aa.copy()
        if not obs_list:
            return None
        return np.stack(obs_list, axis=0)
    except Exception:
        return None


def build_replay(*, attach_obs: bool = True) -> dict:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    cont = json.loads((OUT / "EXPERT_CONTINUATION_RESULTS.json").read_text(encoding="utf-8"))
    book, rho, _cb = load_m111_book()
    caps = _candidate_captures()
    env = make_env() if attach_obs else None
    rows = []
    rejected = []
    n_obs = 0
    for r in cont["results"]:
        if not r.get("SUCCESS_RELAXED_12S"):
            continue
        ap = r.get("expert_actions_path")
        if not ap or not Path(ap).is_file():
            rejected.append({**{k: r[k] for k in ("candidate_id", "ic_id", "skill")}, "reason": "missing_actions"})
            continue
        acts = np.load(ap).astype(np.float32)
        if acts.ndim != 2 or acts.shape[1] != 7 or len(acts) == 0:
            rejected.append({"candidate_id": r["candidate_id"], "reason": "bad_action_shape"})
            continue
        try:
            q = quantize_trajectory(book, acts, rho, w=DEFAULT_WEIGHTS)
            tokens = np.asarray(q["tokens"], np.int64)
            residuals = np.asarray(q["residuals"], np.float32)
            recon_a = np.asarray(q["recon_actions"], np.float32)
            err = float(np.mean(np.linalg.norm(recon_a - acts, axis=-1)))
            valid = err < 0.35  # loose M111 fidelity gate
        except Exception as e:
            rejected.append({"candidate_id": r["candidate_id"], "reason": f"quantize:{e}"})
            continue

        capture_path = caps.get(r["candidate_id"])
        obs = None
        if attach_obs and env is not None and capture_path and Path(capture_path).is_file():
            obs = _replay_obs_from_capture(env, capture_path, acts)
            if obs is not None and len(obs) == len(acts):
                n_obs += 1
            else:
                obs = None

        sample_path = REPLAY_DIR / f"sample_{r['candidate_id'].replace(':','_').replace('/','_')}.npz"
        save_kw = dict(
            actions=acts,
            tokens=tokens,
            residuals=residuals,
            recon_err=np.array([err], np.float32),
        )
        if obs is not None:
            save_kw["obs"] = obs
        np.savez_compressed(sample_path, **save_kw)
        rows.append(
            {
                "candidate_id": r["candidate_id"],
                "ic_id": r["ic_id"],
                "skill": r["skill"],
                "class": r.get("class"),
                "n_expert_actions": int(len(acts)),
                "m111_recon_err": err,
                "m111_valid": bool(valid),
                "bc_obs_valid": bool(obs is not None),
                "sample_path": str(sample_path),
                "capture_path": capture_path,
                "SUCCESS_RELAXED_12S": True,
                "SUCCESS_12S": bool(r.get("SUCCESS_12S")),
                "SUCCESS_NEW": bool(r.get("SUCCESS_NEW")),
                "bc_target": "expert_suffix_only",
            }
        )
        if not valid:
            rejected.append({"candidate_id": r["candidate_id"], "reason": f"high_recon_err:{err:.3f}"})

    manifest = {
        "n_success_continuations": sum(1 for r in cont["results"] if r.get("SUCCESS_RELAXED_12S")),
        "n_replay_samples": len(rows),
        "n_valid_m111": sum(1 for r in rows if r["m111_valid"]),
        "n_bc_obs_valid": n_obs,
        "n_rejected": len(rejected),
        "by_skill": {
            sk: sum(1 for r in rows if r["skill"] == sk) for sk in ("A1", "B1", "C", "D")
        },
        "samples": rows,
        "rejected": rejected,
        "note": (
            "Only expert-controlled suffix actions are BC targets; not on-policy GRPO samples. "
            "BC requires paired obs (bc_obs_valid); zero-obs stub is disabled."
        ),
    }
    (OUT / "EXPERT_REPLAY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# EXPERT_REPLAY_AUDIT",
        "",
        f"- Successful expert continuations: {manifest['n_success_continuations']}",
        f"- Replay samples written: {manifest['n_replay_samples']}",
        f"- M111-valid (recon err < 0.35): {manifest['n_valid_m111']}",
        f"- BC obs attached (state-conditioned): {manifest['n_bc_obs_valid']}",
        f"- Rejected/flagged: {manifest['n_rejected']}",
        f"- By skill: {manifest['by_skill']}",
        "",
        "Policy prefix is NOT used as BC target. Expert suffix only.",
        "Sequences are **not** treated as on-policy GRPO samples.",
        "train_comparison BC skips samples without `obs` in the npz.",
        "",
    ]
    (OUT / "EXPERT_REPLAY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build_replay()
    print(json.dumps({k: m[k] for k in m if k not in ("samples", "rejected")}, indent=2))
