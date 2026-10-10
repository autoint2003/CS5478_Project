"""Train, evaluate, and check the canonical 39-D balanced policy.

One torch seed. The demonstrations, batch weights, tokenizer, optimizer, and
24-case benchmark match the balanced run. The observation is observe_airborne,
which is the former 45-vector without the duplicate hand-twist channels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import training.natural_loss_recatch_transformer_v3 as v3
from envs.airborne_obs import OBS_DIM_AIR, OBS_NAMES_AIR, OBS_VERSION, observe_airborne
from envs.observable_obs import ObservableObsState
from training.balanced_unified_recovery import CKPT_39, CKPT_45, RAW, train_equal
from training.observation_robustness_check import conditions
from training.observation_study_common import (
    build_cases,
    collect_train,
    eval_policy,
    family_counts,
    run_suite,
)

OUT = ROOT / "results" / "diagnostics" / "raw" / "balanced_39d"
REPORT = ROOT / "results" / "diagnostics" / "FINAL_39D_BALANCED_POLICY.md"
SKILLS = ("impact", "sliding", "wrist", "recatch")


def prototypes(device):
    """K=32 prototypes from the archived 45-D balanced checkpoint. Weights are not loaded."""
    saved = torch.load(CKPT_45, map_location=device, weights_only=False)
    weight = saved["model"]["in_proj.weight"]
    if tuple(weight.shape)[-1] == OBS_DIM_AIR + 7:
        raise RuntimeError("archived checkpoint already matches the 39-D input; refusing to treat it as the 45-D source")
    return np.asarray(saved["proto"], np.float32)


def published_retained():
    blob = json.loads((RAW / "summary.json").read_text(encoding="utf-8"))
    kept = []
    for row in blob["rows"]:
        if row["family"] in SKILLS and row["balanced"]["t12"] == "RETAINED_TO_12":
            kept.append(row["id"])
    return kept


def evaluate(ctx, model):
    rows = []
    for case in ctx["cases"]:
        got = eval_policy(ctx, model, case)
        rows.append({"id": case["id"], "family": case["family"], **got})
        print("CASE", case["id"], got["t12"], got["mechanism"], flush=True)
    ok = sum(r["t12"] == "RETAINED_TO_12" for r in rows)
    return {"ok": int(ok), "n": len(rows), "families": family_counts(rows), "rows": rows}


def train_and_eval():
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the final 39-D run requires CUDA")
    print("build cases", flush=True)
    ctx = build_cases()
    trajs = collect_train(ctx)
    widths = {int(np.asarray(tr["obs"]).shape[1]) for tr in trajs}
    if widths != {OBS_DIM_AIR}:
        raise RuntimeError(f"training observations are {widths}, expected {OBS_DIM_AIR}")
    proto = prototypes(device)
    print("trajs", len(trajs), "obs", OBS_DIM_AIR, "proto", tuple(proto.shape), flush=True)
    model, loss, info = train_equal(trajs, proto, device)
    payload = {
        "seed": 1,
        "proto": proto,
        "model": model.state_dict(),
        "obs_dim": OBS_DIM_AIR,
        "obs_names": list(OBS_NAMES_AIR),
        "obs_version": OBS_VERSION,
        "prev_action_dim": 7,
        "train": info,
        "train_loss": None if loss is None else float(loss),
    }
    torch.save(payload, CKPT_39)
    print("saved", CKPT_39, "train", payload["train_loss"], flush=True)

    clean = evaluate(ctx, model)
    kept = published_retained()
    lost = [name for name in kept if next(r["t12"] for r in clean["rows"] if r["id"] == name) != "RETAINED_TO_12"]
    print("CLEAN", clean["ok"], clean["families"], "lost", lost, flush=True)

    spec = next(c for c in conditions() if c["name"] == "combined_moderate")
    noisy = []
    for seed in spec["seeds"]:
        ev = run_suite(ctx, model, spec["spec"], seed)
        noisy.append({
            "seed": seed,
            "ok": ev["ok"],
            "families": family_counts(ev["rows"]),
            "rows": [
                {"id": r["id"], "family": r["family"], "t12": r["t12"], "mechanism": r["mechanism"]}
                for r in ev["rows"]
            ],
        })
        print("NOISY", seed, ev["ok"], family_counts(ev["rows"]), flush=True)

    blob = {
        "obs_dim": OBS_DIM_AIR,
        "obs_names": list(OBS_NAMES_AIR),
        "obs_version": OBS_VERSION,
        "checkpoint": str(CKPT_39),
        "archived_45d_checkpoint": str(CKPT_45),
        "train": info,
        "train_loss": payload["train_loss"],
        "clean": {
            "ok": clean["ok"],
            "families": clean["families"],
            "rows": [
                {"id": r["id"], "family": r["family"], "t12": r["t12"], "mechanism": r["mechanism"]}
                for r in clean["rows"]
            ],
        },
        "published_skill_cases": kept,
        "lost_published_skills": lost,
        "robust": noisy,
        "robust_spec": spec["level"],
    }
    (OUT / "summary.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print("SUMMARY", OUT / "summary.json", flush=True)


def verify():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, proto, saved = v3.load_airborne_policy(CKPT_39, device)
    if int(saved.get("obs_dim", -1)) != 39:
        raise RuntimeError("checkpoint obs_dim is not 39")
    if tuple(model.in_proj.weight.shape) != (64, 46):
        raise RuntimeError(f"input layer is {tuple(model.in_proj.weight.shape)}")
    try:
        v3.load_airborne_policy(CKPT_45, device)
    except RuntimeError as exc:
        print("refused 45-D checkpoint:", exc, flush=True)
    else:
        raise RuntimeError("the 45-D checkpoint was loaded into the 39-D model")
    ctx = build_cases()
    case = next(c for c in ctx["cases"] if c["id"] == "wrist_suffix_3")
    sim = ctx["sim_i"]
    from training.recovery_runtime import restore_dyn
    from training.replay_core import park_impact_ball
    restore_dyn(sim, case["snap"], ctx["nominal_i"], ctx["pads_i"], ctx["pad_mu_i"], None, None)
    park_impact_ball(sim)
    hist = ObservableObsState(v3.DT_POLICY)
    obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
    if obs.shape != (39,):
        raise RuntimeError(f"live observation shape {obs.shape}")
    if obs.shape[0] == 45:
        raise RuntimeError("observation fell back to 45-D")
    prev = np.zeros((1, 7), np.float32)
    with torch.no_grad():
        out = model(
            torch.tensor(obs, device=device).unsqueeze(0),
            torch.tensor(prev, device=device),
        )
    if tuple(out[2].shape) != (1, 7):
        raise RuntimeError(f"action shape {tuple(out[2].shape)}")
    got = eval_policy(ctx, model, case)
    print(
        "VERIFY",
        "obs", tuple(obs.shape),
        "prev", 7,
        "in_proj", tuple(model.in_proj.weight.shape),
        "proto", tuple(proto.shape),
        "rollout", case["id"], got["t12"],
        flush=True,
    )
    if got["t12"] != "RETAINED_TO_12":
        raise RuntimeError(f"wrist_suffix_3 rollout returned {got['t12']}")
    print("VERIFY OK", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        verify()
    else:
        train_and_eval()
