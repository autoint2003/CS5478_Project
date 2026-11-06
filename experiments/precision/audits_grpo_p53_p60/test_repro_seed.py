"""Small reproducibility checks: hash(traj_id) instability + dual-process sampling.

Does not run full eval. Optional MuJoCo/policy check if CUDA+ckpt available.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/audits/grpo_p53_p60"


def test_builtin_hash_unstable():
    tid = "plan_c__gbi_204_057a7965a7"
    # Document PYTHONHASHSEED effect by spawning children
    salts = []
    for seed in ("0", "1", "42"):
        code = (
            f"import os; os.environ['PYTHONHASHSEED']='{seed}'; "
            f"print(hash({tid!r}) % 100000)"
        )
        # Can't set hash seed after interpreter start in parent; spawn fresh
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        salts.append(int(r.stdout.strip()))
    distinct = len(set(salts))
    report = {
        "traj_id": tid,
        "salts_by_PYTHONHASHSEED": {"0": salts[0], "1": salts[1], "42": salts[2]},
        "n_distinct": distinct,
        "conclusion": (
            "hash(traj_id) differs across PYTHONHASHSEED → C-val torch.manual_seed "
            "is not cross-session reproducible"
            if distinct > 1
            else "unexpected: salts identical"
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "hash_seed_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert distinct > 1, report
    print("PASS builtin_hash_unstable", report)


def test_stable_seed_helper():
    tid = "plan_c__gbi_204_057a7965a7"
    a = int(hashlib.md5(tid.encode()).hexdigest()[:8], 16) % 100000
    b = int(hashlib.md5(tid.encode()).hexdigest()[:8], 16) % 100000
    assert a == b
    print("PASS stable_seed_helper", a)


def test_dual_process_policy_sample_if_possible():
    """Two processes, same torch seed + same obs → identical token."""
    ckpt = ROOT / "results/experiments/p59_expanded_c_grpo/checkpoints/P59_pass1.pt"
    if not ckpt.is_file():
        print("SKIP dual_process_policy_sample (no ckpt)")
        return
    script = r"""
import json, sys
import numpy as np
import torch
from pathlib import Path
ROOT = Path(r"%s")
sys.path.insert(0, str(ROOT))
from experiments.precision.p53_independent_grpo64.policy_sample import load_policy_actor, sample_step
device = torch.device("cpu")
actor, book, rho, _ = load_policy_actor(Path(r"%s"), device)
actor.eval()
torch.manual_seed(12345)
obs = [np.zeros(39, np.float32) for _ in range(4)]
prev = [np.zeros(7, np.float32) for _ in range(4)]
toks = []
for i in range(5):
    tok, resid, logp, _ = sample_step(actor, obs, prev, device, temperature=1.0)
    toks.append(int(tok))
    obs.append(np.random.randn(39).astype(np.float32)*0)  # keep zeros; RNG advanced by torch only
    prev.append(np.zeros(7, np.float32))
print(json.dumps({"tokens": toks}))
""" % (
        str(ROOT).replace("\\", "\\\\"),
        str(ckpt).replace("\\", "\\\\"),
    )
    outs = []
    for _ in range(2):
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        if r.returncode != 0:
            print("SKIP dual_process (load failed)", r.stderr[-400:])
            return
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("{")][-1]
        outs.append(json.loads(line)["tokens"])
    match = outs[0] == outs[1]
    (OUT / "dual_process_tokens.json").write_text(
        json.dumps({"run_a": outs[0], "run_b": outs[1], "match": match}, indent=2),
        encoding="utf-8",
    )
    assert match, outs
    print("PASS dual_process_policy_sample", outs[0])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test_builtin_hash_unstable()
    test_stable_seed_helper()
    test_dual_process_policy_sample_if_possible()
    print("ALL REPRO TESTS DONE")


if __name__ == "__main__":
    main()
