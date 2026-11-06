"""Two independent processes: same ckpt / IC / seeds → compare traces."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments.precision.p61_reproducible_joint_eval.paths import (
    GLOBAL_SEED,
    OUT,
    P59_PASS1,
    PROTOCOL_VERSION,
)


WORKER = r'''
import json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"{root}")
sys.path.insert(0, str(ROOT))

from experiments.precision.p53_independent_grpo64.probabilistic_eval import load_heldout_starts
from experiments.precision.p61_reproducible_joint_eval.eval_core import (
    load_actor, make_eval_env, rollout_seeded,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed
from experiments.precision.p61_reproducible_joint_eval.paths import P59_PASS1, PER_CLASS_HELDOUT

device_s = sys.argv[1]
out_path = Path(sys.argv[2])
n_roll = int(sys.argv[3])

device = torch.device(device_s if torch.cuda.is_available() and device_s.startswith("cuda") else "cpu")
starts = load_heldout_starts(per_class=PER_CLASS_HELDOUT)
# pick first C IC
st = starts["C"][0]
actor, book, rho, _ = load_actor(P59_PASS1, device)
env = make_eval_env(book, rho)
rows = []
for i in range(n_roll):
    seed = rollout_seed(split="heldout", traj_id=st.traj_id, rollout_i=i)
    m = rollout_seeded(
        env, actor, st, device, seed=seed, record_trace=True, deterministic_torch=True
    )
    rows.append({{
        "traj_id": st.traj_id,
        "skill": st.skill,
        "rollout_i": i,
        "seed": seed,
        "device": str(device),
        "obs0_sha16": m["obs0_sha16"],
        "qpos0_sha16": m["qpos0_sha16"],
        "tokens": m["tokens"],
        "actions_resid_sha16": [
            __import__("hashlib").sha256(
                np.asarray(a["resid"], np.float32).tobytes()
            ).hexdigest()[:16]
            for a in m["actions"]
        ],
        "qpos_trace_sha16": m["qpos_trace_sha16"],
        "SUCCESS_RELAXED_12S": m["SUCCESS_RELAXED_12S"],
        "SUCCESS_12S": m["SUCCESS_12S"],
        "SUCCESS_NEW": m["SUCCESS_NEW"],
        "drop": m["drop"],
        "handoff": m["handoff"],
        "n_steps": m["n_steps"],
    }})
out_path.write_text(json.dumps({{"rows": rows}}, indent=2), encoding="utf-8")
print("WROTE", out_path, flush=True)
'''


def _run_worker(device: str, out_path: Path, n_roll: int) -> dict:
    root = Path(__file__).resolve().parents[3]
    code = WORKER.format(root=str(root).replace("\\", "\\\\"))
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    r = subprocess.run(
        [sys.executable, "-c", code, device, str(out_path), str(n_roll)],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"worker failed:\n{r.stdout}\n{r.stderr}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def _compare(a: dict, b: dict) -> dict:
    diffs = []
    exact = True
    for ra, rb in zip(a["rows"], b["rows"]):
        keys = [
            "obs0_sha16",
            "qpos0_sha16",
            "tokens",
            "actions_resid_sha16",
            "qpos_trace_sha16",
            "SUCCESS_RELAXED_12S",
            "SUCCESS_12S",
            "SUCCESS_NEW",
            "drop",
            "handoff",
            "n_steps",
        ]
        row_diff = {"rollout_i": ra["rollout_i"], "seed": ra["seed"]}
        mismatch = False
        for k in keys:
            if ra.get(k) != rb.get(k):
                mismatch = True
                exact = False
                row_diff[k] = {"a": ra.get(k), "b": rb.get(k)}
        if mismatch:
            diffs.append(row_diff)
    n = len(a["rows"])
    succ_a = sum(r["SUCCESS_RELAXED_12S"] for r in a["rows"])
    succ_b = sum(r["SUCCESS_RELAXED_12S"] for r in b["rows"])
    return {
        "exact_match": exact,
        "n_rollouts": n,
        "n_mismatched_rollouts": len(diffs),
        "success_count_a": succ_a,
        "success_count_b": succ_b,
        "diffs": diffs[:8],  # cap
    }


def run_reproducibility_test(*, device: str = "cpu", n_rollouts: int = 4) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    if not P59_PASS1.is_file():
        raise FileNotFoundError(P59_PASS1)
    p_a = OUT / "repro_worker_a.json"
    p_b = OUT / "repro_worker_b.json"
    print(f"[repro] device={device} n={n_rollouts} process A...", flush=True)
    a = _run_worker(device, p_a, n_rollouts)
    print(f"[repro] process B...", flush=True)
    b = _run_worker(device, p_b, n_rollouts)
    cmp = _compare(a, b)

    # Diagnose first mismatch source if any
    diagnosis = []
    if cmp["exact_match"]:
        diagnosis.append("Exact match across independent processes.")
    else:
        d0 = (cmp["diffs"] or [{}])[0]
        if d0.get("obs0_sha16") or d0.get("qpos0_sha16"):
            diagnosis.append(
                "Initial observation/qpos mismatch → IC restore or pre-policy RNG/env init issue."
            )
        elif d0.get("tokens") or d0.get("actions_resid_sha16"):
            diagnosis.append(
                "Action mismatch with matching init → policy sampling / torch RNG / CUDA nondeterminism."
            )
        elif d0.get("qpos_trace_sha16"):
            diagnosis.append(
                "Tokens match but physics diverge → simulator nondeterminism or float reduction order."
            )
        elif any(k.startswith("SUCCESS") or k in ("drop", "handoff") for k in d0):
            diagnosis.append(
                "Terminal label mismatch with earlier fields matching → success/handoff eval path drift."
            )
        else:
            diagnosis.append("Mismatch present; see diffs.")
        if device.startswith("cuda"):
            diagnosis.append(
                "CUDA device used: some kernels may remain nondeterministic despite seeding; "
                "re-run with --device cpu to isolate."
            )

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "global_seed": GLOBAL_SEED,
        "ckpt": str(P59_PASS1),
        "device": device,
        "n_rollouts": n_rollouts,
        "PYTHONHASHSEED": "0",
        "comparison": cmp,
        "diagnosis": diagnosis,
    }
    (OUT / "REPRODUCIBILITY_TEST.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# REPRODUCIBILITY_TEST",
        "",
        f"Checkpoint: `{P59_PASS1.name}` | device: **{device}** | rollouts: {n_rollouts}",
        f"Protocol: `{PROTOCOL_VERSION}` | PYTHONHASHSEED=0 | SHA-256 seeds",
        "",
        f"**Exact match: {cmp['exact_match']}** "
        f"({cmp['n_mismatched_rollouts']}/{cmp['n_rollouts']} mismatched).",
        f"Success counts: A={cmp['success_count_a']} B={cmp['success_count_b']}",
        "",
        "## Diagnosis",
        "",
    ]
    for d in diagnosis:
        lines.append(f"- {d}")
    lines += ["", "## Compared fields", ""]
    lines.append(
        "obs0, qpos0, tokens, residual hashes, qpos trajectory hashes, "
        "SUCCESS_RELAXED/12S/NEW, drop, handoff, n_steps."
    )
    lines.append("")
    (OUT / "REPRODUCIBILITY_TEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("device", "comparison", "diagnosis")}, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-rollouts", type=int, default=4)
    args = ap.parse_args()
    run_reproducibility_test(device=args.device, n_rollouts=args.n_rollouts)
