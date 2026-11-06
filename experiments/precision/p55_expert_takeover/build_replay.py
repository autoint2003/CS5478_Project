"""Phase 4 — Build expert BC replay from successful takeover continuations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from models.m111_checkpoint import HANDOFF, load_m111_book
from experiments.precision.p55_expert_takeover.paths import OUT, REPLAY_DIR
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS, quantize_trajectory


def build_replay() -> dict:
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    cont = json.loads((OUT / "EXPERT_CONTINUATION_RESULTS.json").read_text(encoding="utf-8"))
    book, rho, _cb = load_m111_book()
    rows = []
    rejected = []
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

        sample_path = REPLAY_DIR / f"sample_{r['candidate_id'].replace(':','_').replace('/','_')}.npz"
        np.savez_compressed(
            sample_path,
            actions=acts,
            tokens=tokens,
            residuals=residuals,
            recon_err=np.array([err], np.float32),
        )
        rows.append(
            {
                "candidate_id": r["candidate_id"],
                "ic_id": r["ic_id"],
                "skill": r["skill"],
                "class": r.get("class"),
                "n_expert_actions": int(len(acts)),
                "m111_recon_err": err,
                "m111_valid": bool(valid),
                "sample_path": str(sample_path),
                "capture_path": None,  # prefix capture is in TAKEOVER_CANDIDATES
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
        "n_rejected": len(rejected),
        "by_skill": {
            sk: sum(1 for r in rows if r["skill"] == sk) for sk in ("A1", "B1", "C", "D")
        },
        "samples": rows,
        "rejected": rejected,
        "note": "Only expert-controlled suffix actions are BC targets; not on-policy GRPO samples.",
    }
    (OUT / "EXPERT_REPLAY_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# EXPERT_REPLAY_AUDIT",
        "",
        f"- Successful expert continuations: {manifest['n_success_continuations']}",
        f"- Replay samples written: {manifest['n_replay_samples']}",
        f"- M111-valid (recon err < 0.35): {manifest['n_valid_m111']}",
        f"- Rejected/flagged: {manifest['n_rejected']}",
        f"- By skill: {manifest['by_skill']}",
        "",
        "Policy prefix is NOT used as BC target. Expert suffix only.",
        "Sequences are **not** treated as on-policy GRPO samples.",
        "",
    ]
    (OUT / "EXPERT_REPLAY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    m = build_replay()
    print(json.dumps({k: m[k] for k in m if k not in ("samples", "rejected")}, indent=2))
