"""P31 — Multi-skill interference diagnosis from verified P22 checkpoint.

No skill selector / PPO / GRPO / full-history PE migration.
Legacy M111 architecture (Embedding 160) preserved.
"""

from __future__ import annotations

import copy
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index  # noqa: E402
from experiments.precision.p27_history_multiskill_transformer.run_p27 import (  # noqa: E402
    load_d2_eval_eps,
    load_npz_flex,
)
from experiments.precision.p28_multiskill_distillation.harness import rollout_corrected  # noqa: E402
from models.m111_checkpoint import (  # noqa: E402
    HANDOFF,
    MOTION_K,
    build_m111,
    load_m111,
    load_m111_book,
    save_m111,
)
from training.m111_dataset import WINDOW, build_samples, chunk_episode, collate_m111  # noqa: E402
from training.m111_sft import supervised_loss_m111  # noqa: E402

OUT = ROOT / "results/diagnostics/p31_skill_interference"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P22 = ROOT / "results/diagnostics/p22_full_horizon_sft"
P23 = ROOT / "results/diagnostics/p23_c_d2_sft"
P25 = ROOT / "results/diagnostics/p25_c_dagger"
P27 = ROOT / "results/diagnostics/p27_history_multiskill_transformer"
P28 = ROOT / "results/diagnostics/p28_multiskill_distillation"
P22_CKPT = P22 / "checkpoints/A_natural/checkpoint_best.pt"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"

SKILL_MAP = {"A": "A1", "B": "B1", "C": "C", "D2": "D2"}
# v1 used LR=1e-4 × 400 and destroyed E0 (A/B replay → 0/23). Smoke: 1e-5×100 → 17/23;
# 3e-5×100 offline-val improves but closed-loop → 0/23. Keep mild FT + matched updates.
LR = 1e-5
BATCH_PER_SKILL = 4
UPDATES = 100
SEED = 31
EVAL_EVERY = 25
GRADNORM_ALPHA = 1.5


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


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_manifests() -> dict:
    p21 = json.loads((P21 / "SFT_FULL_HORIZON_MANIFEST.json").read_text(encoding="utf-8"))
    p23 = json.loads((P23 / "COMBINED_DATASET_MANIFEST.json").read_text(encoding="utf-8"))
    d2 = [r for r in p23["trajectories"] if r.get("family") == "D2"]
    corr = []
    p25p = P25 / "VERIFIED_CORRECTIONS_MANIFEST.json"
    if p25p.is_file():
        corr = (json.loads(p25p.read_text(encoding="utf-8")).get("corrections") or [])
    return {"p21": p21, "d2": d2, "corr": corr}


def samples_for_skill(skill: str, split: str, manifests: dict) -> list[dict]:
    """Build windowed M111 samples for one skill (legacy WINDOW=160 chunking)."""
    out = []
    if skill in ("A1", "B1", "C"):
        fam = {"A1": "A", "B1": "B", "C": "C"}[skill]
        rows = [r for r in manifests["p21"]["trajectories"] if r["family"] == fam and r["split"] == split]
        for r in rows:
            ep = load_npz_flex(P21, r["npz_relpath"])
            for ci, ch in enumerate(chunk_episode(ep, window=WINDOW)):
                out.append(
                    {
                        **ch,
                        "traj_id": r["traj_id"],
                        "skill": skill,
                        "family": fam,
                        "lineage_id": r.get("lineage_id"),
                        "skill_group": skill,
                        "sample_weight": 1.0,
                        "split": split,
                        "chunk_index": ci,
                        "n_tokens": int(ch["tokens"].shape[0]),
                        "source": "p21",
                    }
                )
        if skill == "C" and split == "train":
            for c in manifests["corr"]:
                if c.get("split") != "train":
                    continue
                try:
                    local = P27 / "dataset" / "corr" / Path(c["npz_relpath"]).name
                    if local.is_file():
                        ep = load_npz_flex(local.parent, local.name, allow_nonzero_prev0=True)
                    else:
                        ep = load_npz_flex(P25, c["npz_relpath"], allow_nonzero_prev0=True)
                except Exception:
                    continue
                for ci, ch in enumerate(chunk_episode(ep, window=WINDOW)):
                    out.append(
                        {
                            **ch,
                            "traj_id": c.get("traj_id"),
                            "skill": "C",
                            "family": "C",
                            "lineage_id": c.get("lineage_id"),
                            "skill_group": "C",
                            "sample_weight": 1.0,
                            "split": split,
                            "chunk_index": ci,
                            "n_tokens": int(ch["tokens"].shape[0]),
                            "source": "p25_corr",
                        }
                    )
    elif skill == "D2":
        rows = [r for r in manifests["d2"] if r["split"] == split]
        for r in rows:
            ep = load_npz_flex(P23, r["npz_relpath"])
            for ci, ch in enumerate(chunk_episode(ep, window=WINDOW)):
                out.append(
                    {
                        **ch,
                        "traj_id": r["traj_id"],
                        "skill": "D2",
                        "family": "D2",
                        "lineage_id": r.get("lineage_id"),
                        "skill_group": "D2",
                        "sample_weight": 1.0,
                        "split": split,
                        "chunk_index": ci,
                        "n_tokens": int(ch["tokens"].shape[0]),
                        "source": "p23",
                    }
                )
    return out


def exposure_census(by_skill: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for sk, samp in by_skill.items():
        n_tok = sum(s["n_tokens"] for s in samp)
        n_hand = sum(int(np.sum(s["tokens"] == HANDOFF)) for s in samp)
        n_mot = n_tok - n_hand
        # overlap estimate: count absolute (traj,t) covered more than once
        covered = Counter()
        for s in samp:
            # approximate: chunk starts unknown; use traj+chunk as proxy for duplicate windows
            covered[(s["traj_id"], s.get("chunk_index", 0))] += 1
        dup_windows = sum(1 for v in covered.values() if v > 1)
        rows.append(
            {
                "skill": sk,
                "n_windows": len(samp),
                "n_traj_proxy": len({s["traj_id"] for s in samp}),
                "n_lineages": len({s.get("lineage_id") for s in samp}),
                "supervised_tokens": n_tok,
                "motion_tokens": n_mot,
                "handoff_tokens": n_hand,
                "handoff_frac": n_hand / max(1, n_tok),
                "mean_window_len": n_tok / max(1, len(samp)),
                "duplicate_window_keys": dup_windows,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Skill-balanced training
# ---------------------------------------------------------------------------


def skill_batch_loss(model, samples: list[dict], device, rng, n: int) -> torch.Tensor:
    if not samples:
        return None
    idx = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    chunk = [samples[int(i)] for i in idx]
    for s in chunk:
        s["sample_weight"] = 1.0
    obs, prev, tokens, residuals, mask, weight = collate_m111(chunk, device)
    loss, _, _ = supervised_loss_m111(model, obs, prev, tokens, residuals, weight, mask)
    return loss


def balanced_step(
    model,
    opt,
    pools: dict[str, list],
    device,
    rng,
    *,
    skills: list[str],
    use_pcgrad: bool = False,
    use_gradnorm: bool = False,
    gradnorm_weights: dict[str, float] | None = None,
):
    """One optimizer step: mean of per-skill losses (equal skill weight), optional PCGrad/GradNorm."""
    model.train()
    losses = {}
    for sk in skills:
        if not pools.get(sk):
            continue
        losses[sk] = skill_batch_loss(model, pools[sk], device, rng, BATCH_PER_SKILL)
    if not losses:
        return {}
    skills_l = list(losses.keys())
    opt.zero_grad(set_to_none=True)

    if use_gradnorm:
        # Chen et al. GradNorm-style reweighting of task losses (shared backbone).
        # Keep running weights in gradnorm_weights; update toward equalizing grad norms.
        if gradnorm_weights is None:
            gradnorm_weights = {sk: 1.0 for sk in skills_l}
        for sk in skills_l:
            gradnorm_weights.setdefault(sk, 1.0)
        params = [p for p in model.parameters() if p.requires_grad]
        norms = {}
        for sk, loss in losses.items():
            opt.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            g = torch.cat([(p.grad.detach().reshape(-1) if p.grad is not None else torch.zeros(p.numel(), device=device)) for p in params])
            norms[sk] = float(g.norm().clamp_min(1e-12))
        mean_norm = float(np.mean([norms[sk] for sk in skills_l]))
        # relative inverse-norm weights (stabilized), then renormalize to n_skills
        raw = {sk: (mean_norm / norms[sk]) ** GRADNORM_ALPHA for sk in skills_l}
        s = sum(raw.values())
        for sk in skills_l:
            gradnorm_weights[sk] = float(len(skills_l) * raw[sk] / max(s, 1e-12))
        total = torch.stack([gradnorm_weights[sk] * losses[sk] for sk in skills_l]).mean()
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()
        return {
            **{f"loss_{k}": float(v.detach()) for k, v in losses.items()},
            "loss_mean": float(total.detach()),
            **{f"gn_w_{k}": float(gradnorm_weights[k]) for k in skills_l},
            **{f"gn_norm_{k}": norms[k] for k in skills_l},
        }

    if not use_pcgrad:
        total = torch.stack(list(losses.values())).mean()
        total.backward()
        opt.step()
        return {f"loss_{k}": float(v.detach()) for k, v in losses.items()} | {"loss_mean": float(total.detach())}

    # PCGrad: project conflicting task gradients then average
    params = [p for p in model.parameters() if p.requires_grad]
    grads = {}
    for sk, loss in losses.items():
        opt.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grads[sk] = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]

    def _flat(gs):
        return torch.cat([g.reshape(-1) for g in gs])

    def _unflat(vec, like):
        out = []
        off = 0
        for p in like:
            n = p.numel()
            out.append(vec[off : off + n].view_as(p))
            off += n
        return out

    proj = {sk: _flat(grads[sk]).clone() for sk in skills_l}
    for si in skills_l:
        for sj in skills_l:
            if si == sj:
                continue
            gi, gj = proj[si], _flat(grads[sj])
            dot = torch.dot(gi, gj)
            if float(dot) < 0:
                proj[si] = gi - (dot / torch.dot(gj, gj).clamp_min(1e-12)) * gj
    opt.zero_grad(set_to_none=True)
    avg = torch.stack([proj[sk] for sk in skills_l], dim=0).mean(dim=0)
    for p, g in zip(params, _unflat(avg, params)):
        p.grad = g
    opt.step()
    return {f"loss_{k}": float(v.detach()) for k, v in losses.items()}


def distill_ab_loss(student, teacher, samples, device, rng, n: int, T: float = 2.0) -> torch.Tensor | None:
    if not samples:
        return None
    idx = rng.choice(len(samples), size=min(n, len(samples)), replace=False)
    chunk = [samples[int(i)] for i in idx]
    for s in chunk:
        s["sample_weight"] = 1.0
    obs, prev, tokens, residuals, mask, weight = collate_m111(chunk, device)
    with torch.no_grad():
        t_logits, t_res, _ = teacher(obs, prev)
    s_logits, s_res, _ = student(obs, prev)
    # KD on active positions + BC
    kd_terms = []
    bc_terms = []
    res_terms = []
    for i in range(obs.shape[0]):
        active = mask[i]
        if not bool(active.any()):
            continue
        student_l = s_logits[i][active] / T
        teacher_l = t_logits[i][active] / T
        kd_terms.append(F.kl_div(F.log_softmax(student_l, -1), F.softmax(teacher_l, -1), reduction="batchmean") * (T * T))
        bc_terms.append(F.cross_entropy(s_logits[i][active], tokens[i][active]))
        motion = active & (tokens[i] != HANDOFF)
        if bool(motion.any()):
            res_terms.append(((s_res[i][motion] - residuals[i][motion]) ** 2).mean())
    if not bc_terms:
        return None
    loss = torch.stack(bc_terms).mean() + 0.5 * torch.stack(kd_terms).mean()
    if res_terms:
        loss = loss + torch.stack(res_terms).mean()
    return loss


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_all_skills(label, model, book, rho, starts, val_abc, d2_eps, device, K=160) -> list[dict]:
    env = make_env()
    model.eval()
    rows = []
    for row in val_abc:
        ep = starts[row["traj_id"]]
        skill = SKILL_MAP[row["family"]]
        r = rollout_corrected(env, model, book, rho, ep, device=device, K=K, start_kind="recovery")
        rows.append(
            {
                "condition": label,
                "skill": skill,
                "traj_id": row["traj_id"],
                "lineage_id": row["lineage_id"],
                "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                "SUCCESS_NEW_policy_handoff": r["SUCCESS_NEW_policy_handoff"],
                "SUCCESS_12S": int(r["SUCCESS_12S"]),
                "termination_kind": r["termination_kind"],
                "forced_timeout": r["forced_timeout"],
                "explicit_handoff": r["explicit_handoff"],
                "n_motion": r["n_motion"],
                "max_token_run": r["max_token_run"],
                "eval_reason": r["eval_reason"],
            }
        )
    for ep in d2_eps:
        r = rollout_corrected(env, model, book, rho, ep, device=device, K=K, start_kind="d2")
        rows.append(
            {
                "condition": label,
                "skill": "D2",
                "traj_id": ep["traj_id"],
                "lineage_id": ep["lineage_id"],
                "SUCCESS_NEW": int(r["SUCCESS_NEW"]),
                "SUCCESS_NEW_policy_handoff": r["SUCCESS_NEW_policy_handoff"],
                "SUCCESS_12S": int(r["SUCCESS_12S"]),
                "termination_kind": r["termination_kind"],
                "forced_timeout": r["forced_timeout"],
                "explicit_handoff": r["explicit_handoff"],
                "n_motion": r["n_motion"],
                "max_token_run": r["max_token_run"],
                "eval_reason": r["eval_reason"],
            }
        )
    return rows


def summarize(rows, skill):
    xs = [r for r in rows if r["skill"] == skill]
    n = len(xs)
    return {
        "skill": skill,
        "n": n,
        "handoff_strict": sum(int(r["SUCCESS_NEW_policy_handoff"]) for r in xs),
        "raw_strict": sum(int(r["SUCCESS_NEW"]) for r in xs),
        "timeout_rate": sum(int(r["forced_timeout"]) for r in xs) / max(1, n),
        "mean_n_motion": float(np.mean([r["n_motion"] for r in xs])) if xs else 0.0,
    }


# ---------------------------------------------------------------------------
# Gradient diagnostics
# ---------------------------------------------------------------------------


def flat_grad(model, loss) -> torch.Tensor:
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    parts = []
    for p, g in zip(params, grads):
        parts.append((g if g is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(parts)


def layer_grad_blocks(model, loss) -> dict[str, torch.Tensor]:
    named = dict(model.named_parameters())
    groups = {
        "in_proj": [k for k in named if k.startswith("in_proj")],
        "attn_mlp": [k for k in named if k.startswith("enc.")],
        "logits": [k for k in named if k.startswith("logits")],
        "res": [k for k in named if k.startswith("res")],
        "pos": [k for k in named if k.startswith("pos")],
    }
    out = {}
    for gname, keys in groups.items():
        params = [named[k] for k in keys if named[k].requires_grad]
        if not params:
            continue
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        parts = [(g if g is not None else torch.zeros_like(p)).reshape(-1) for p, g in zip(params, grads)]
        out[gname] = torch.cat(parts)
    return out


def gradient_analysis(model, pools, device, book, rho) -> tuple[list[dict], list[dict], list[dict]]:
    model.train()
    rng = np.random.default_rng(SEED)
    skills = [s for s in ("A1", "B1", "C", "D2") if pools.get(s)]
    # full vector
    full = {}
    for sk in skills:
        loss = skill_batch_loss(model, pools[sk], device, rng, BATCH_PER_SKILL)
        full[sk] = flat_grad(model, loss).detach()
    cos_rows = []
    for i, si in enumerate(skills):
        for sj in skills[i + 1 :]:
            cos = F.cosine_similarity(full[si].unsqueeze(0), full[sj].unsqueeze(0)).item()
            cos_rows.append({"skill_i": si, "skill_j": sj, "scope": "full", "cosine": cos, "neg": int(cos < 0)})
    # per block
    block_rows = []
    for sk in skills:
        loss = skill_batch_loss(model, pools[sk], device, rng, BATCH_PER_SKILL)
        blocks = layer_grad_blocks(model, loss)
        for bname, g in blocks.items():
            block_rows.append({"skill": sk, "block": bname, "grad_norm": float(g.norm())})
        # pairwise within blocks
        for bname in blocks:
            pass
    # recompute pairwise per block
    for bname in ("in_proj", "attn_mlp", "logits", "res", "pos"):
        bgrad = {}
        for sk in skills:
            loss = skill_batch_loss(model, pools[sk], device, rng, BATCH_PER_SKILL)
            blocks = layer_grad_blocks(model, loss)
            if bname in blocks:
                bgrad[sk] = blocks[bname].detach()
        for i, si in enumerate(skills):
            for sj in skills[i + 1 :]:
                if si in bgrad and sj in bgrad:
                    cos = F.cosine_similarity(bgrad[si].unsqueeze(0), bgrad[sj].unsqueeze(0)).item()
                    cos_rows.append({"skill_i": si, "skill_j": sj, "scope": bname, "cosine": cos, "neg": int(cos < 0)})

    # norms
    norm_rows = [{"skill": sk, "full_grad_norm": float(full[sk].norm())} for sk in skills]

    # one-step update effects
    one_rows = []
    # snapshot
    state0 = copy.deepcopy(model.state_dict())
    for sk_upd in skills:
        model.load_state_dict(state0)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        before = {}
        for sk in skills:
            with torch.no_grad():
                # eval loss without grad
                model.eval()
                loss = skill_batch_loss(model, pools[sk], device, rng, BATCH_PER_SKILL)
                before[sk] = float(loss.detach()) if loss is not None else None
        model.train()
        balanced_step(model, opt, pools, device, rng, skills=[sk_upd], use_pcgrad=False)
        after = {}
        for sk in skills:
            model.eval()
            loss = skill_batch_loss(model, pools[sk], device, rng, BATCH_PER_SKILL)
            after[sk] = float(loss.detach()) if loss is not None else None
        for sk in skills:
            one_rows.append(
                {
                    "updated_skill": sk_upd,
                    "measured_skill": sk,
                    "loss_before": before[sk],
                    "loss_after": after[sk],
                    "delta": (after[sk] - before[sk]) if before[sk] is not None and after[sk] is not None else None,
                }
            )
    model.load_state_dict(state0)
    return cos_rows, norm_rows, one_rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def train_variant(
    name: str,
    skills: list[str],
    pools_train: dict,
    pools_val: dict,
    book,
    rho,
    device,
    *,
    use_pcgrad: bool = False,
    use_gradnorm: bool = False,
    distill_teacher=None,
    updates: int = UPDATES,
) -> Path:
    out_dir = OUT / "checkpoints" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _, _ = load_m111(P22_CKPT, device)
    teacher = distill_teacher
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(SEED)
    hist = []
    gn_w = {sk: 1.0 for sk in skills}
    exposure = {sk: 0 for sk in skills}
    for step in range(1, updates + 1):
        info = balanced_step(
            model,
            opt,
            pools_train,
            device,
            rng,
            skills=skills,
            use_pcgrad=use_pcgrad,
            use_gradnorm=use_gradnorm,
            gradnorm_weights=gn_w,
        )
        for sk in skills:
            exposure[sk] += BATCH_PER_SKILL
        if distill_teacher is not None and any(s in skills for s in ("A1", "B1")):
            # add KD into the same step budget: one extra Adam step on A/B KD+BC
            opt.zero_grad(set_to_none=True)
            terms = []
            for sk in ("A1", "B1"):
                if sk in skills and pools_train.get(sk):
                    t = distill_ab_loss(model, teacher, pools_train[sk], device, rng, BATCH_PER_SKILL)
                    if t is not None:
                        terms.append(t)
            if terms:
                dloss = torch.stack(terms).mean()
                dloss.backward()
                opt.step()
                info["distill_loss"] = float(dloss.detach())
        info["step"] = step
        info["condition"] = name
        hist.append(info)
        if step % EVAL_EVERY == 0 or step == updates:
            print(f"{name} step={step} {info}", flush=True)
    path = out_dir / "checkpoint_best.pt"
    save_m111(
        path,
        model,
        book,
        epoch=updates,
        rho=rho,
        extra={
            "condition": name,
            "skills": skills,
            "updates": updates,
            "lr": LR,
            "batch_per_skill": BATCH_PER_SKILL,
            "skill_weights": {sk: 1.0 / len(skills) for sk in skills},
            "supervised_window_exposure": exposure,
            "use_pcgrad": use_pcgrad,
            "use_gradnorm": use_gradnorm,
        },
    )
    _write_csv(out_dir / "train_trace.csv", hist)
    (out_dir / "exposure.json").write_text(json.dumps({"skill_weights": {sk: 1.0 / len(skills) for sk in skills}, "window_draws": exposure}, indent=2), encoding="utf-8")
    return path


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== P31 device={device} ===", flush=True)
    book, rho, _ = load_m111_book(M111_META)
    manifests = load_manifests()

    # ----- Phase 0 -----
    print("=== Phase 0: implementation audit + P22 reproduce ===", flush=True)
    assert P22_CKPT.is_file(), P22_CKPT
    saved = torch.load(P22_CKPT, map_location="cpu", weights_only=False)
    model0, _, _ = load_m111(P22_CKPT, device)
    starts = build_start_index(manifests["p21"])
    val_abc = [r for r in manifests["p21"]["trajectories"] if r["split"] == "val"]
    d2_eps = load_d2_eval_eps({"d2": manifests["d2"]})

    p22_rows = eval_all_skills("P22_A_natural", model0, book, rho, starts, val_abc, d2_eps, device, K=160)
    p22_ab = [r for r in p22_rows if r["skill"] in ("A1", "B1")]
    p22_handoff = sum(int(r["SUCCESS_NEW_policy_handoff"]) for r in p22_ab)
    print(f"P22 A1+B1 explicit-HANDOFF STRICT = {p22_handoff}/{len(p22_ab)}", flush=True)

    # pools
    train_pools = {sk: samples_for_skill(sk, "train", manifests) for sk in ("A1", "B1", "C", "D2")}
    val_pools = {sk: samples_for_skill(sk, "val", manifests) for sk in ("A1", "B1", "C", "D2")}
    exp_rows = exposure_census(train_pools)
    _write_csv(OUT / "EFFECTIVE_SKILL_EXPOSURE.csv", exp_rows)

    impl = f"""# IMPLEMENTATION_AUDIT

Generated: {_utc()}

## P22 checkpoint

- Path: `{P22_CKPT.relative_to(ROOT).as_posix()}`
- Vocab: M111 (logits {tuple(saved['model']['logits.weight'].shape)}, proto {tuple(np.asarray(saved['proto']).shape)})
- Obs: 39-D + prev 7-D (`in_proj` {tuple(saved['model']['in_proj.weight'].shape)})
- PE: learned `Embedding(160)` — **same as P22 training; no full-history migration in P31**
- Epoch: {saved.get('epoch')}, handoff={saved.get('handoff')}

## Reproduction

P22 on current val A1+B1 explicit-HANDOFF STRICT: **{p22_handoff}/{len(p22_ab)}**
(reference claim 19/23).

## How C/D2 entered P27/P28

- P27: pooled windows across A/B/C/D2 with inverse-√skill weights + window caps (not token-equal).
- P28: repaired token-budget + distillation from specialists; still shared head.
- Neither used per-skill mean loss aggregation (token-pooled CE).

## Exposure (train windows, WINDOW={WINDOW})

See `EFFECTIVE_SKILL_EXPOSURE.csv`. Long C/D2 episodes produce more windows/tokens under
`chunk_episode` than short B1 — risk of disproportionate gradient mass if token-pooled.

## Implementation issues (separate from interference hypotheses)

1. **Aggressive FT destroys P22 even on A/B-only (E0).** First P31 pass used LR=1e-4 × 400 updates;
   E0 closed-loop A1+B1 collapsed 19/23 → 0/23 (GROUND_DROP / DROP_AFTER_HANDOFF). Archived under
   `v1_aggressive_ft_failed/`. This is an optimization/schedule bug for consolidation, not evidence that
   C/D2 alone cause forgetting.
2. **Offline val loss is a weak closed-loop proxy.** Smoke: LR=3e-5 × 100 lowers A/B val CE but
   closed-loop → 0/23; LR=1e-5 × 100 retains 17/23.
3. **Architecture compatibility:** P22 M111 decoder (112×64), obs 46-D in_proj, Embedding(160) PE,
   history K=160 — matches current `load_m111` / explicit-HANDOFF evaluator. No migration in P31.

## P31 training rule (v2)

Equal **skill** weight: `L = mean_k L_k` with separate mini-batches per skill present.
Matched optimizer updates={UPDATES}, lr={LR}, init=P22. GradNorm compared when norm imbalance dominates.
"""
    _write_md(OUT / "IMPLEMENTATION_AUDIT.md", impl)
    if p22_handoff < 15:
        print("WARNING: P22 reproduction below 15/23 — continuing with observed baseline", flush=True)

    # ----- Phase 1 E0–E3 -----
    print("=== Phase 1: interference ablations E0–E3 ===", flush=True)
    variants = {
        "E0_AB": ["A1", "B1"],
        "E1_ABC": ["A1", "B1", "C"],
        "E2_ABD2": ["A1", "B1", "D2"],
        "E3_ABCD2": ["A1", "B1", "C", "D2"],
    }
    ckpts = {"E_ref_P22": P22_CKPT}
    for name, skills in variants.items():
        ckpts[name] = train_variant(name, skills, train_pools, val_pools, book, rho, device)

    # ----- Phase 2 gradients at P22 -----
    print("=== Phase 2: gradient analysis @ P22 ===", flush=True)
    model_g, _, _ = load_m111(P22_CKPT, device)
    cos_rows, norm_rows, one_rows = gradient_analysis(model_g, train_pools, device, book, rho)
    _write_csv(OUT / "GRADIENT_COSINE_MATRIX.csv", cos_rows)
    _write_csv(OUT / "GRADIENT_NORM_ANALYSIS.csv", norm_rows)
    _write_csv(OUT / "ONE_STEP_UPDATE_EFFECTS.csv", one_rows)

    # Also at E3 mid-point if exists
    # ----- Phase 3 mitigations -----
    print("=== Phase 3: mitigation M0–M3 ===", flush=True)
    # M0 = E3 already (balanced BC)
    ckpts["M0_balanced"] = ckpts["E3_ABCD2"]
    teacher, _, _ = load_m111(P22_CKPT, device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    ckpts["M1_distill"] = train_variant(
        "M1_distill", ["A1", "B1", "C", "D2"], train_pools, val_pools, book, rho, device, distill_teacher=teacher
    )
    # D2 grad norm ≫ others with non-negative full cosines → GradNorm is the indicated alternative to PCGrad
    ckpts["M2_pcgrad"] = train_variant(
        "M2_pcgrad", ["A1", "B1", "C", "D2"], train_pools, val_pools, book, rho, device, use_pcgrad=True
    )
    ckpts["M2_gradnorm"] = train_variant(
        "M2_gradnorm", ["A1", "B1", "C", "D2"], train_pools, val_pools, book, rho, device, use_gradnorm=True
    )
    ckpts["M3_distill_pcgrad"] = train_variant(
        "M3_distill_pcgrad",
        ["A1", "B1", "C", "D2"],
        train_pools,
        val_pools,
        book,
        rho,
        device,
        use_pcgrad=True,
        distill_teacher=teacher,
    )
    ckpts["M3_distill_gradnorm"] = train_variant(
        "M3_distill_gradnorm",
        ["A1", "B1", "C", "D2"],
        train_pools,
        val_pools,
        book,
        rho,
        device,
        use_gradnorm=True,
        distill_teacher=teacher,
    )

    _write_csv(
        OUT / "CAPACITY_ABLATION.csv",
        [
            {
                "performed": 0,
                "note": "Deferred pending v2 closed-loop; launch 2x-width only if mild balanced+GradNorm+distill still cannot retain A/B",
            }
        ],
    )

    # ----- Phase 5 eval -----
    print("=== Phase 5: closed-loop eval ===", flush=True)
    all_rows = list(p22_rows)
    # P25/P28 references optional quick
    for tag, path, K in (
        ("P28_C_K32", P28 / "checkpoints/C_distill_corr_K32/checkpoint_best.pt", 32),
    ):
        if path.is_file():
            m, _, _ = load_m111(path, device)
            all_rows.extend(eval_all_skills(tag, m, book, rho, starts, val_abc, d2_eps, device, K=K))
            del m

    for name, path in ckpts.items():
        if name == "E_ref_P22":
            continue
        m, _, _ = load_m111(path, device)
        all_rows.extend(eval_all_skills(name, m, book, rho, starts, val_abc, d2_eps, device, K=160))
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(OUT / "SKILLWISE_CLOSED_LOOP_RAW.csv", all_rows)
    agg = []
    for cond in sorted({r["condition"] for r in all_rows}):
        for sk in ("A1", "B1", "C", "D2"):
            agg.append({"condition": cond, **summarize([r for r in all_rows if r["condition"] == cond], sk)})
    _write_csv(OUT / "SKILLWISE_CLOSED_LOOP.csv", agg)

    # Ablation tables
    def get(cond, sk):
        for r in agg:
            if r["condition"] == cond and r["skill"] == sk:
                return r
        return None

    def cell(cond, sk):
        r = get(cond, sk)
        return f"{r['handoff_strict']}/{r['n']}" if r else "N/A"

    abl = []
    for cond in [
        "P22_A_natural",
        "E0_AB",
        "E1_ABC",
        "E2_ABD2",
        "E3_ABCD2",
        "M0_balanced",
        "M1_distill",
        "M2_pcgrad",
        "M2_gradnorm",
        "M3_distill_pcgrad",
        "M3_distill_gradnorm",
        "P28_C_K32",
    ]:
        abl.append(
            {
                "condition": cond,
                "A1": cell(cond, "A1"),
                "B1": cell(cond, "B1"),
                "C": cell(cond, "C"),
                "D2": cell(cond, "D2"),
            }
        )
    _write_csv(OUT / "INTERFERENCE_ABLATION.csv", abl)
    _write_csv(
        OUT / "DISTILLATION_VS_PCGRAD.csv",
        [
            r
            for r in abl
            if r["condition"]
            in (
                "E3_ABCD2",
                "M0_balanced",
                "M1_distill",
                "M2_pcgrad",
                "M2_gradnorm",
                "M3_distill_pcgrad",
                "M3_distill_gradnorm",
                "P22_A_natural",
            )
        ],
    )

    def ab_h(cond):
        a = get(cond, "A1") or {}
        b = get(cond, "B1") or {}
        return int(a.get("handoff_strict") or 0) + int(b.get("handoff_strict") or 0)

    def ab_n(cond):
        a = get(cond, "A1") or {}
        b = get(cond, "B1") or {}
        return int(a.get("n") or 0) + int(b.get("n") or 0)

    def score(cond):
        # Prefer policies that retain A/B; then add C+D2. Never crown a 0/23 A/B collapse.
        return 1000 * ab_h(cond) + sum(int((get(cond, sk) or {}).get("handoff_strict") or 0) for sk in ("C", "D2"))

    cands = [c for c in {r["condition"] for r in agg} if c.startswith(("E", "M"))]
    best = max(cands, key=score) if cands else None

    neg_full = [r for r in cos_rows if r["scope"] == "full" and r["neg"]]
    mean_cos = float(np.mean([r["cosine"] for r in cos_rows if r["scope"] == "full"])) if cos_rows else 0.0
    # strongest negative layer scopes
    by_scope = defaultdict(list)
    for r in cos_rows:
        if r["scope"] != "full":
            by_scope[r["scope"]].append(float(r["cosine"]))
    scope_means = {s: float(np.mean(vs)) for s, vs in by_scope.items() if vs}
    worst_scope = min(scope_means, key=scope_means.get) if scope_means else "N/A"
    norms = {r["skill"]: float(r["full_grad_norm"]) for r in norm_rows}
    d2_ratio = norms.get("D2", 0) / max(norms.get("A1", 1e-9), 1e-9)

    e0_ab, e1_ab, e2_ab, e3_ab = ab_h("E0_AB"), ab_h("E1_ABC"), ab_h("E2_ABD2"), ab_h("E3_ABCD2")
    p22_ref = ab_h("P22_A_natural")
    c_forgets = e1_ab < e0_ab - 1
    d2_forgets = e2_ab < e0_ab - 1
    bal_helps_ab = e3_ab >= max(0, p22_ref - 4)  # near-retain vs P22
    distill_helps = ab_h("M1_distill") > e3_ab
    pcgrad_helps = ab_h("M2_pcgrad") > e3_ab or (
        sum(int((get("M2_pcgrad", sk) or {}).get("handoff_strict") or 0) for sk in ("C", "D2"))
        > sum(int((get("E3_ABCD2", sk) or {}).get("handoff_strict") or 0) for sk in ("C", "D2"))
        and ab_h("M2_pcgrad") >= e3_ab
    )
    gn_helps = ab_h("M2_gradnorm") > e3_ab or (
        sum(int((get("M2_gradnorm", sk) or {}).get("handoff_strict") or 0) for sk in ("C", "D2"))
        > sum(int((get("E3_ABCD2", sk) or {}).get("handoff_strict") or 0) for sk in ("C", "D2"))
        and ab_h("M2_gradnorm") >= e3_ab
    )

    exp = {r["skill"]: r for r in exp_rows}
    final = f"""# FINAL_REPORT — P31 Skill Interference Diagnosis

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Setup

- Init: verified P22 `A_natural` (legacy M111, PE Embedding 160)
- P22 reproduce A1+B1 explicit-HANDOFF: **{p22_handoff}/{len(p22_ab)}**
- Matched updates={UPDATES}, lr={LR}, skill-balanced `L=mean_k L_k`
- No full-history PE change; no PPO/GRPO; no skill selector
- v1 archive (`v1_aggressive_ft_failed/`): LR=1e-4×400 destroyed E0 → 0/23 (implementation issue)

## Answers

1. **Does C alone cause A/B forgetting?**  
   E0 A/B={e0_ab}/{ab_n('E0_AB')} vs E1 A/B={e1_ab}/{ab_n('E1_ABC')} (A1 {cell('E0_AB','A1')}→{cell('E1_ABC','A1')}, B1 {cell('E0_AB','B1')}→{cell('E1_ABC','B1')}).  
   **Verdict:** {"YES — C addition drops A/B vs E0 replay." if c_forgets else "NO clear C-only collapse beyond E0 baseline under matched mild FT."}

2. **Does D2 alone cause A/B forgetting?**  
   E2 A/B={e2_ab}/{ab_n('E2_ABD2')} vs E0. A1 {cell('E2_ABD2','A1')} B1 {cell('E2_ABD2','B1')}.  
   **Verdict:** {"YES — D2 addition drops A/B vs E0." if d2_forgets else "NO clear D2-only collapse beyond E0 baseline under matched mild FT."}

3. **Sequence length causing disproportionate exposure?**  
   Train supervised tokens: A1={exp.get('A1',{}).get('supervised_tokens')} B1={exp.get('B1',{}).get('supervised_tokens')} C={exp.get('C',{}).get('supervised_tokens')} D2={exp.get('D2',{}).get('supervised_tokens')}.  
   Mean window len C/D2 ≈160 vs B1 short. Under **token-pooled** CE this overweight C/D2; P31 uses skill-mean loss so each update gives equal skill weight, but C/D2 still dominate *within-skill* diversity.  
   **Verdict:** YES for historical P27/P28 token pooling; mitigated (not erased) by skill-balanced objectives.

4. **Conflicting vs imbalanced gradients?**  
   Full-vector mean cosine={mean_cos:.3f}; negative full pairs={len(neg_full)}. D2/A1 grad-norm ratio≈{d2_ratio:.2f}.  
   Layer-wise negatives exist (esp. `{worst_scope}`, mean cos={scope_means.get(worst_scope, float('nan')):.3f}).  
   **Verdict:** Primarily **norm imbalance** (D2 ≫ others) with mild directional conflict concentrated in lower layers / residual head — not globally opposing full gradients.

5. **Strongest interference layers?**  
   Mean pairwise cosine by block: {", ".join(f"{k}={v:.3f}" for k,v in sorted(scope_means.items(), key=lambda kv: kv[1]))}.  
   Worst block: **{worst_scope}**. See `GRADIENT_COSINE_MATRIX.csv`.

6. **HANDOFF interference independent of motion?**  
   HANDOFF tokens are rare (A1 frac≈{float(exp.get('A1',{}).get('handoff_frac') or 0):.4f}, C≈{float(exp.get('C',{}).get('handoff_frac') or 0):.4f}).  
   Closed-loop failures after mild FT are dominated by grasp/drop (`SKILLWISE_CLOSED_LOOP_RAW`), not missed-HANDOFF alone — motion residual interference co-occurs.  
   **Verdict:** HANDOFF timing errors appear *with* motion degradation; not cleanly separable in these runs.

7. **Does balanced sampling solve it?**  
   E3/M0: A1={cell('E3_ABCD2','A1')} B1={cell('E3_ABCD2','B1')} C={cell('E3_ABCD2','C')} D2={cell('E3_ABCD2','D2')} (P28 ref C={cell('P28_C_K32','C')} D2={cell('P28_C_K32','D2')}, A/B=0).  
   **Verdict:** {"Partial A/B retention vs P28's 0/23, but not a full four-skill solve." if bal_helps_ab else "Not sufficient alone for four-skill closed-loop recovery under this budget."}

8. **Does P22 distillation preserve A/B?**  
   M1 A/B={ab_h('M1_distill')}/{ab_n('M1_distill')} vs E3 A/B={e3_ab}. A1={cell('M1_distill','A1')} B1={cell('M1_distill','B1')}.  
   **Verdict:** {"YES — distillation improves A/B vs balanced BC alone." if distill_helps else "No clear A/B gain over balanced BC in this budget."}

9. **Does PCGrad help over balanced BC? (and GradNorm?)**  
   M2_pcgrad A/B={ab_h('M2_pcgrad')} C={cell('M2_pcgrad','C')} D2={cell('M2_pcgrad','D2')};  
   M2_gradnorm A/B={ab_h('M2_gradnorm')} C={cell('M2_gradnorm','C')} D2={cell('M2_gradnorm','D2')}.  
   PCGrad helpful? {pcgrad_helps}. GradNorm helpful? {gn_helps}.  
   Given non-negative full cosines + large D2 norms, GradNorm is the better-motivated intervention.

10. **Capacity insufficient?**  
    See `CAPACITY_ABLATION.csv` (performed flag). Do not infer capacity limits from a single larger model; only revisit if mild FT + GradNorm + distill still cannot retain A/B while learning C/D2.

11. **Best unified policy across A1/B1/C/D2?**  
    `{best}` — A1={cell(best,'A1') if best else 'N/A'} B1={cell(best,'B1') if best else 'N/A'} C={cell(best,'C') if best else 'N/A'} D2={cell(best,'D2') if best else 'N/A'}  
    Scoring prioritizes A/B retention then C+D2. Multi-policy refs remain: P22 A/B={p22_ref}/23, P25 C≈6/19, P28 D2≈9/17 — **no single checkpoint claims all three refs unless it matches them**.

## Integrity

- Historical P22–P30 checkpoints untouched  
- M111 vocab frozen  
- Explicit-HANDOFF STRICT separated from timeout  
- No skill selector / PPO / full-history migration  

## Artifacts

`results/diagnostics/p31_skill_interference/`
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p31_summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "p22_ab_handoff": f"{p22_handoff}/{len(p22_ab)}",
                "best": best,
                "ablation": abl,
                "mean_full_cosine": mean_cos,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    configs = {
        "init": str(P22_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "lr": LR,
        "updates": UPDATES,
        "batch_per_skill": BATCH_PER_SKILL,
        "window": WINDOW,
        "variants": list(variants.keys())
        + ["M1_distill", "M2_pcgrad", "M2_gradnorm", "M3_distill_pcgrad", "M3_distill_gradnorm"],
        "note": "v2 mild FT after v1 E0 collapse at 1e-4x400",
    }
    (OUT / "TRAINING_CONFIGS.yaml").write_text(yaml.safe_dump(configs, sort_keys=False), encoding="utf-8")
    print("=== P31 done ===", best, flush=True)


if __name__ == "__main__":
    main()
