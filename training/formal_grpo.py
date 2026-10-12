"""Formal critic-free GRPO.

Running this file starts closed-loop training. The preparation check does not
call it. Outputs go to the directory in the config and do not replace the
frozen reference, the frozen initialization, or the 40-update sanity run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.config_util import load_yaml
from training.action_representation_expansion import build_policy
from training.autonomous_episode import make_training_env
from training.final_rl_readiness import CATALOG, GROUP, SIGMA
from training.future_state_ablation import FuturePolicy
from training.pre_grpo_sanity import (
    CLIP,
    KL_WEAK,
    LR,
    TRAIN_ORDER,
    one_update,
    sha256,
)
from training.supervised_coverage_warmstart import roll_ic

IC_DIR = ROOT / "results" / "diagnostics" / "raw" / "final_rl_readiness" / "ics"
SANITY_DIR = ROOT / "results" / "diagnostics" / "raw" / "pre_grpo_sanity"
# Frozen readiness result: this validation impact never entered.
INVALID_NO_ENTRY = frozenset({"imp_f"})


def split_ids(name):
    return tuple(spec["id"] for spec in CATALOG if spec["split"] == name)


def assert_config(cfg):
    expected = {
        "kl_coef": KL_WEAK,
        "lr": LR,
        "clip": CLIP,
        "group": GROUP,
        "sigma": SIGMA,
        "grad_clip": 1.0,
    }
    for key, value in expected.items():
        if abs(float(cfg[key]) - float(value)) > 1e-12:
            raise RuntimeError(f"{key} is {cfg[key]} and the frozen value is {value}")
    updates = int(cfg["updates"])
    every = int(cfg["eval_every"])
    if updates < 1 or every < 1:
        raise RuntimeError("updates and eval_every must be positive")
    if set(TRAIN_ORDER) != set(split_ids("train")):
        raise RuntimeError("TRAIN_ORDER does not match the catalog train split")
    final_ids = set(split_ids("final_test"))
    if final_ids & set(TRAIN_ORDER) or final_ids & set(split_ids("validation")):
        raise RuntimeError("FINAL_TEST overlaps the train or validation split")
    init = ROOT / str(cfg["initialization"])
    ref = ROOT / str(cfg["reference"])
    out = ROOT / str(cfg["output_dir"])
    if out.resolve() == SANITY_DIR.resolve():
        raise RuntimeError("output_dir would overwrite the pre-GRPO sanity files")
    if not init.is_file() or not ref.is_file():
        raise RuntimeError("initialization or reference checkpoint is missing")
    return updates, every, init, ref, out


def refuse_final(episode_id):
    if episode_id in split_ids("final_test"):
        raise RuntimeError(f"FINAL_TEST is not used: {episode_id}")


def load_models(init_path, ref_path, device):
    saved_ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_model = build_policy(np.asarray(saved_ref["proto"], np.float32), device, float(saved_ref["res_cap"]))
    ref_model.load_state_dict(saved_ref["model"])
    ref_model.eval()
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    saved_init = torch.load(init_path, map_location="cpu", weights_only=False)
    proto = np.asarray(saved_init["proto"], np.float32)
    if proto.shape != (36, 7):
        raise RuntimeError(f"expected K=36 initialization, got {proto.shape}")
    if abs(float(saved_ref["res_cap"]) - 0.25) > 1e-9 or abs(float(saved_init["res_cap"]) - 0.25) > 1e-9:
        raise RuntimeError("residual cap is not 0.25")
    model = FuturePolicy(proto).to(device)
    model.load_state_dict(saved_init["model"])
    for parameter in model.fut.parameters():
        parameter.requires_grad_(False)
    return model, ref_model


def validation_eval(env, model):
    model.eval()
    env.model = model
    rows = []
    for episode_id in split_ids("validation"):
        refuse_final(episode_id)
        path = IC_DIR / f"{episode_id}.pkl"
        if episode_id in INVALID_NO_ENTRY:
            if path.exists():
                raise RuntimeError(f"{episode_id} is the frozen no-entry case and now has a capture")
            rows.append({"id": episode_id, "rolled": False, "label": "INVALID_IC", "success": False})
            continue
        if not path.is_file():
            raise RuntimeError(f"missing validation capture {episode_id}")
        out = roll_ic(env, episode_id)
        rows.append({
            "id": episode_id,
            "rolled": True,
            "label": out.get("label"),
            "success": bool(out.get("success")),
            "held_10": out.get("held_10"),
            "held_12": out.get("held_12"),
        })
    score = int(sum(row["success"] for row in rows if row["rolled"]))
    return score, rows


def save_model(path, model, update_index, score):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "update": update_index,
            "validation_success": score,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Formal critic-free GRPO")
    parser.add_argument("--config", default="config/formal_grpo.yaml")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    updates, every, init_path, ref_path, out = assert_config(cfg)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ref_model = load_models(init_path, ref_path, device)
    env = make_training_env(device, model=model)
    log = {
        "initialization": {"path": str(init_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(init_path)},
        "reference": {"path": str(ref_path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha256(ref_path)},
        "train_ids": list(TRAIN_ORDER),
        "validation_ids": list(split_ids("validation")),
        "validation_not_rolled": sorted(INVALID_NO_ENTRY),
        "final_test_not_rolled": list(split_ids("final_test")),
        "updates": updates,
        "eval_every": every,
        "kl_coef": KL_WEAK,
        "objective": "mean recovery-step clipped GRPO plus state-scoped KL 0.05",
        "selected": None,
        "evals": [],
        "rows": [],
    }
    path = out / "formal_grpo.json"
    best = {"score": -1, "update": None}

    def consider(update_index):
        score, rows = validation_eval(env, model)
        model.train()
        log["evals"].append({"update": update_index, "validation_success": score, "rows": rows})
        save_model(out / f"checkpoint_u{update_index:04d}.pt", model, update_index, score)
        if score > best["score"]:
            best["score"] = score
            best["update"] = update_index
            save_model(out / "selected_validation.pt", model, update_index, score)
            log["selected"] = {"update": update_index, "validation_success": score}
        path.write_text(json.dumps(log), encoding="utf-8")
        print("VAL", update_index, score, "best", best["update"], best["score"], flush=True)

    print("FORMAL_GRPO", log["initialization"]["sha256"][:12], log["reference"]["sha256"][:12], flush=True)
    consider(0)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    for index in range(1, updates + 1):
        episode_id = TRAIN_ORDER[(index - 1) % len(TRAIN_ORDER)]
        refuse_final(episode_id)
        rng = np.random.default_rng(int(cfg["seed_base"]) + 19 * index + sum(ord(char) for char in episode_id))
        row, _audits = one_update(env, model, ref_model, opt, episode_id, KL_WEAK, rng, index)
        log["rows"].append(row)
        path.write_text(json.dumps(log), encoding="utf-8")
        if index % every == 0:
            consider(index)
    print("SELECTED", log["selected"], flush=True)
    print("WROTE", path, flush=True)


if __name__ == "__main__":
    main()
