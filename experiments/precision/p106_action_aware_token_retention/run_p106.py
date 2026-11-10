"""P106 — Action-aware token retention vs strength-matched KL control.

Arms:
  A — frozen P64
  B — reuse P104 v2 Token-only checkpoint (no retrain)
  C — token-only SFT + stronger KL retention (matched control)
  D — token-only SFT + KL + action-aware retention (primary)

One seed. Backbone + residual frozen. No PPO/GRPO. No sealed ind panel.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.final_independent_success_rate.paths import P64_ACTOR  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
from models.p104_policy import build_p104_from_p64  # noqa: E402

# Reuse P104 data / CL / correction utilities (same recipe).
import experiments.precision.p104_joint_token_residual_pilot.run_p104 as p104  # noqa: E402

OUT = ROOT / "results/experiments/p106_action_aware_token_retention"
CKPT = OUT / "checkpoints"
LOGS = OUT / "logs"
P104_OUT = ROOT / "results/experiments/p104_joint_token_residual_pilot"
P104B_CKPT = P104_OUT / "checkpoints" / "armB_seed0_final.pt"

EXPECTED_SHA = "ec08f59816aa2fdd"
SEED = 0
UPDATES = 2000
LR = 1e-3
GRAD_CLIP = 5.0
CKPT_SELECT_RULE = "final_update_2000_no_cl_selection"

# Retention calibration (locked a priori; total strength ≈ P104's 3.0)
CALIB_TEMP_T = 2.0
STRENGTH_KL = 1.5
STRENGTH_AA = 1.5
LAMBDA_MAX = 10000.0  # AA_ref from peaked softmax diffs is small; allow λ so λ*AA_ref = STRENGTH*L_corr
KL_REF_MIN = 1e-4
AA_REF_MIN = 1e-8
LARGE_ACT_FLIP_THRESH = 0.5  # L2 on executed action under frozen residual


def utc() -> str:
    return p104.utc()


def write_json(path: Path, obj) -> None:
    p104.write_json(path, obj)


def write_csv(path: Path, rows: list[dict]) -> None:
    p104.write_csv(path, rows)


def sha16(path: Path) -> str:
    return p104.sha16(path)


def phys_root(tid: str) -> str:
    return p104.phys_root(tid)


@torch.no_grad()
def compute_action_distance_table(p64, demo, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-step (T,K) squared executed-action distances from P64 argmax token.

    Uses clipped decode(proto[z] + r_P64(H)), HANDOFF → zero action.
    Returns (d2, z0, p_ref) with HANDOFF-masked softmax already applied to p_ref.
    """
    obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
    prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
    logits, res, _ = p64(obs, prev)
    masked = mask_handoff_logits(logits.float())
    p_ref = F.softmax(masked, dim=-1)
    z0 = torch.argmax(masked, dim=-1)  # [T]
    proto = p64.proto  # [K_or_n_motion+?, 7] — FullHistoryPolicy: proto is motion table
    # Align with policy decode: motion tokens 0..n_motion-1; handoff → zeros
    n_motion = int(p64.n_motion)
    handoff = int(p64.handoff_index) if p64.handoff_index is not None else None
    K = int(masked.shape[-1])
    T = int(masked.shape[0])
    # Build executed actions for every token index
    a_all = torch.zeros(T, K, 7, device=device, dtype=torch.float32)
    motion_idx = torch.arange(K, device=device).clamp(0, n_motion - 1)
    proto_exp = proto[motion_idx]  # [K,7]
    a_motion = torch.clamp(proto_exp.unsqueeze(0) + res.unsqueeze(1), -1.0, 1.0)  # [T,K,7]
    a_all = a_motion
    if handoff is not None and 0 <= handoff < K:
        a_all[:, handoff, :] = 0.0
    a0 = a_all[torch.arange(T, device=device), z0]  # [T,7]
    d2 = ((a_all - a0.unsqueeze(1)) ** 2).sum(-1)  # [T,K]
    if handoff is not None and 0 <= handoff < K:
        d2[:, handoff] = 0.0
    return d2, z0, p_ref


def estimate_d_norm(p64, pres_demos, device) -> dict:
    """Frozen normalization: mean positive action-distance over preservation."""
    vals = []
    n_steps = 0
    for d in pres_demos:
        d2, _, _ = compute_action_distance_table(p64, d, device)
        # mean over tokens with d2>0 and steps
        pos = d2[d2 > 1e-12]
        if pos.numel():
            vals.append(float(pos.mean().item()))
        n_steps += int(d2.shape[0])
    mean_pos = float(np.mean(vals)) if vals else 1.0
    if not np.isfinite(mean_pos) or mean_pos < 1e-8:
        mean_pos = 1.0
    return {
        "D_norm": mean_pos,
        "method": "mean_positive_d2_over_preservation_tokens",
        "n_pres_demos": len(pres_demos),
        "n_steps": n_steps,
        "note": "d_z := ||a_z - a_z0||^2 / D_norm; frozen before training",
    }


def action_aware_retention(
    actor, p64, demo, device, d_norm: float, *, pi_new: torch.Tensor | None = None
) -> tuple[torch.Tensor, dict]:
    """L_AA = mean_t sum_z d_z_norm * (pi_new - pi_P64)^2. Zero at P64 init."""
    with torch.no_grad():
        d2, z0, p_ref = compute_action_distance_table(p64, demo, device)
        d_normed = d2 / float(d_norm)
    if pi_new is None:
        obs = torch.as_tensor(demo["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(demo["prev"], device=device, dtype=torch.float32)
        l_new, _, _ = actor(obs, prev)
        pi_new = F.softmax(mask_handoff_logits(l_new.float()), dim=-1)
    diff = pi_new - p_ref
    per_t = (d_normed * diff.pow(2)).sum(-1)
    loss = per_t.mean()
    with torch.no_grad():
        z_new = torch.argmax(pi_new, dim=-1)
        agree = float((z_new == z0).float().mean().item())
        # action-weighted mass on non-z0
        aw = float((d_normed * pi_new).sum(-1).mean().item())
    return loss, {
        "L_AA": float(loss.detach()),
        "argmax_agree": agree,
        "action_weighted_mass": aw,
        "n": int(p_ref.shape[0]),
    }


def token_kl_retention(actor, p64, demo, device) -> tuple[torch.Tensor, dict]:
    return p104.token_kl_retention(actor, p64, demo, device)


def correction_losses(actor, demo, book, rho, device):
    return p104.correction_losses(actor, demo, book, rho, device, joint=False)


def calibrate_retention(p64, corr_demos, pres_demos, book, rho, device, d_norm: float) -> dict:
    """Training-only calibration + C↔D retention strength matching. No eval peeking.

    KL reference: temperature T≠1 (never uniform logit+1).
    AA reference: fixed prior P104-B v2 Token-only on preservation (nontrivial harmful
    deviation documented before P106 CL; not selected from P106 eval outcomes).
    Strength match uses intended scales: λ_KL_control*KL_ref = (STRENGTH_KL+STRENGTH_AA)*L_corr.
    """
    actor = build_p104_from_p64(p64, "token_only", device)
    b1 = [d for d in corr_demos if d["skill"] == "B1"]
    c = [d for d in corr_demos if d["skill"] == "C"]
    if not b1 or not c:
        raise RuntimeError("calibration requires nonempty B1 and C correction pools")

    tok_pair_vals = []
    for db, dc in zip(b1, c + c[: max(0, len(b1) - len(c))]):
        lt_b, _, sb = correction_losses(actor, db, book, rho, device)
        lt_c, _, sc = correction_losses(actor, dc, book, rho, device)
        if sb["n_sup"] + sc["n_sup"] > 0:
            tok_pair_vals.append(float(lt_b.detach()) + float(lt_c.detach()))
    l_tok = float(np.mean(tok_pair_vals)) if tok_pair_vals else 1e-3

    by_skill: dict[str, list] = {"A1": [], "B1": [], "D": []}
    for d in pres_demos:
        by_skill[d["skill"]].append(d)
    for sk in ("A1", "B1", "D"):
        if not by_skill[sk]:
            raise RuntimeError(f"calibration requires nonempty preservation class {sk}")

    T = float(CALIB_TEMP_T)
    if abs(T - 1.0) < 1e-9:
        raise RuntimeError("CALIB_TEMP_T must differ from 1.0")

    if not P104B_CKPT.is_file():
        raise FileNotFoundError(f"P104-B required for AA reference: {P104B_CKPT}")
    harm = load_token_ckpt(P104B_CKPT, p64, device)

    kl_refs = []
    aa_refs = []
    aa_temp_refs = []
    for sk, pool in by_skill.items():
        kl_vals, aa_vals, aa_t_vals = [], [], []
        for d in pool:
            obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
            prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
            with torch.no_grad():
                l_ref, _, _ = p64(obs, prev)
                masked = mask_handoff_logits(l_ref.float())
                p_ref = F.softmax(masked, dim=-1)
                log_q = F.log_softmax(masked / T, dim=-1)
                kl = (p_ref * (p_ref.clamp_min(1e-8).log() - log_q)).sum(-1).mean()
                pi_T = F.softmax(masked / T, dim=-1)
                l_h, _, _ = harm(obs, prev)
                pi_h = F.softmax(mask_handoff_logits(l_h.float()), dim=-1)
            laa_h, _ = action_aware_retention(actor, p64, d, device, d_norm, pi_new=pi_h)
            laa_t, _ = action_aware_retention(actor, p64, d, device, d_norm, pi_new=pi_T)
            kl_vals.append(float(kl))
            aa_vals.append(float(laa_h.detach()))
            aa_t_vals.append(float(laa_t.detach()))
        kl_refs.append(float(np.mean(kl_vals)))
        aa_refs.append(float(np.mean(aa_vals)))
        aa_temp_refs.append(float(np.mean(aa_t_vals)))
    kl_ref = float(np.mean(kl_refs))
    aa_ref = float(np.mean(aa_refs))
    aa_temp_ref = float(np.mean(aa_temp_refs))
    if not np.isfinite(kl_ref) or kl_ref < KL_REF_MIN:
        raise RuntimeError(f"token KL reference too small/invalid: {kl_ref}")
    if not np.isfinite(aa_ref) or aa_ref < AA_REF_MIN:
        raise RuntimeError(f"AA reference (P104-B) too small/invalid: {aa_ref}; temp_ref={aa_temp_ref}")

    lam_kl = float(STRENGTH_KL * l_tok / kl_ref)
    lam_aa = float(STRENGTH_AA * l_tok / aa_ref)
    lam_kl_uncapped, lam_aa_uncapped = lam_kl, lam_aa
    lam_kl = min(lam_kl, LAMBDA_MAX)
    lam_aa = min(lam_aa, LAMBDA_MAX)

    # Intended total retention scale match (robust if individual λ hits a cap)
    lam_kl_control = float((STRENGTH_KL + STRENGTH_AA) * l_tok / kl_ref)
    lam_kl_control = min(lam_kl_control, LAMBDA_MAX)

    seed_all(SEED, deterministic_torch=True)
    actor_g = build_p104_from_p64(p64, "token_only", device)
    actor_g.train()
    actor_g.base.eval()
    batch = [by_skill[sk][0] for sk in ("A1", "B1", "D")]

    def ret_losses(act):
        L_kl = torch.zeros((), device=device)
        L_aa = torch.zeros((), device=device)
        for d in batch:
            kli, _ = token_kl_retention(act, p64, d, device)
            aai, _ = action_aware_retention(act, p64, d, device, d_norm)
            L_kl = L_kl + kli
            L_aa = L_aa + aai
        return L_kl / len(batch), L_aa / len(batch)

    with torch.no_grad():
        actor_g.base.logits.weight.add_(0.01 * torch.randn_like(actor_g.base.logits.weight))

    L_kl, L_aa = ret_losses(actor_g)
    loss_D = float(lam_kl) * L_kl + float(lam_aa) * L_aa
    actor_g.zero_grad(set_to_none=True)
    loss_D.backward(retain_graph=True)
    g_D = torch.cat([p.grad.reshape(-1) for p in actor_g.parameters() if p.requires_grad and p.grad is not None])
    norm_D = float(g_D.norm().item()) if g_D.numel() else 0.0

    actor_g.zero_grad(set_to_none=True)
    L_kl2, _ = ret_losses(actor_g)
    (float(lam_kl_control) * L_kl2).backward()
    g_C = torch.cat([p.grad.reshape(-1) for p in actor_g.parameters() if p.requires_grad and p.grad is not None])
    norm_C = float(g_C.norm().item()) if g_C.numel() else 0.0
    grad_ratio = float(norm_C / max(norm_D, 1e-12))

    actor0 = build_p104_from_p64(p64, "token_only", device)
    aa0_vals = [float(action_aware_retention(actor0, p64, d, device, d_norm)[0].detach()) for d in batch]
    aa_at_init = float(np.mean(aa0_vals))

    return {
        "created": utc(),
        "method": "train_only_temp_KL_plus_P104B_AA_ref_intended_strength_match_v2",
        "mean_L_tok_corr_pair_init": l_tok,
        "CALIB_TEMP_T": T,
        "kl_ref_temperature": kl_ref,
        "aa_ref_P104B": aa_ref,
        "aa_ref_temperature_audit": aa_temp_ref,
        "aa_ref_source": str(P104B_CKPT),
        "STRENGTH_KL": STRENGTH_KL,
        "STRENGTH_AA": STRENGTH_AA,
        "lambda_KL": lam_kl,
        "lambda_AA": lam_aa,
        "lambda_KL_uncapped": lam_kl_uncapped,
        "lambda_AA_uncapped": lam_aa_uncapped,
        "lambda_KL_control": lam_kl_control,
        "lambda_KL_capped": lam_kl >= LAMBDA_MAX - 1e-12,
        "lambda_AA_capped": lam_aa >= LAMBDA_MAX - 1e-12,
        "lambda_KL_control_capped": lam_kl_control >= LAMBDA_MAX - 1e-12,
        "retention_loss_scale_D_intended": float((STRENGTH_KL + STRENGTH_AA) * l_tok),
        "retention_loss_scale_C_at_temp_ref": float(lam_kl_control * kl_ref),
        "retention_loss_scale_D_at_P104B_AA_ref": float(lam_kl * kl_ref + lam_aa * aa_ref),
        "grad_norm_D_perturbed": norm_D,
        "grad_norm_C_matched_perturbed": norm_C,
        "grad_norm_ratio_C_over_D": grad_ratio,
        "L_AA_at_P64_init_mean": aa_at_init,
        "LAMBDA_MAX": LAMBDA_MAX,
        "d_norm": d_norm,
        "note": (
            "λ_KL_control*(KL_ref) matched to (STRENGTH_KL+STRENGTH_AA)*L_corr. "
            "λ_AA from P104-B AA_ref on preservation. No post-hoc eval tuning."
        ),
    }


def train_arm(
    arm: str,
    seed: int,
    p64,
    book,
    rho,
    corr_demos,
    pres_demos,
    lam_kl: float,
    lam_aa: float,
    d_norm: float,
    device,
):
    """arm in {'C','D'}."""
    assert arm in ("C", "D")
    seed_all(seed, deterministic_torch=True)
    actor = build_p104_from_p64(p64, "token_only", device)
    n_train = actor.trainable_param_count()
    opt = torch.optim.Adam([p for p in actor.parameters() if p.requires_grad], lr=LR)
    b1 = [d for d in corr_demos if d["skill"] == "B1"]
    c = [d for d in corr_demos if d["skill"] == "C"]
    by_skill: dict[str, list] = {"A1": [], "B1": [], "D": []}
    for d in pres_demos:
        by_skill[d["skill"]].append(d)
    rng = np.random.default_rng(seed)
    log_rows = []
    actor.train()
    actor.base.eval()
    t0 = time.time()
    use_aa = arm == "D"
    for u in range(1, UPDATES + 1):
        db = b1[int(rng.integers(0, len(b1)))]
        dc = c[int(rng.integers(0, len(c)))]
        lt_b, _, sb = correction_losses(actor, db, book, rho, device)
        lt_c, _, sc = correction_losses(actor, dc, book, rho, device)
        L_corr = lt_b + lt_c

        pres_batch = p104.sample_balanced_pres(by_skill, rng)
        L_kl = torch.zeros((), device=device)
        L_aa = torch.zeros((), device=device)
        for dp in pres_batch:
            kli, _ = token_kl_retention(actor, p64, dp, device)
            L_kl = L_kl + kli
            if use_aa:
                aai, _ = action_aware_retention(actor, p64, dp, device, d_norm)
                L_aa = L_aa + aai
        if pres_batch:
            L_kl = L_kl / len(pres_batch)
            if use_aa:
                L_aa = L_aa / len(pres_batch)

        if use_aa:
            loss = L_corr + float(lam_kl) * L_kl + float(lam_aa) * L_aa
        else:
            loss = L_corr + float(lam_kl) * L_kl

        opt.zero_grad(set_to_none=True)
        if float(loss.detach()) > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
        if u % 50 == 0 or u == 1 or u == UPDATES:
            row = {
                "arm": arm,
                "seed": seed,
                "update": u,
                "loss": float(loss.detach()),
                "L_corr": float(L_corr.detach()),
                "L_tok_B": sb["L_tok"],
                "L_tok_C": sc["L_tok"],
                "L_KL": float(L_kl.detach()),
                "L_AA": float(L_aa.detach()) if use_aa else 0.0,
                "mismatch_C": sc["teacher_pred_mismatch_frac"],
                "elapsed_s": time.time() - t0,
            }
            log_rows.append(row)
            print(
                f"  [P106-{arm} s={seed}] u={u}/{UPDATES} loss={row['loss']:.5f} "
                f"corr={row['L_corr']:.5f} kl={row['L_KL']:.5f} aa={row['L_AA']:.5f}",
                flush=True,
            )
    path = CKPT / f"arm{arm}_seed{seed}_final.pt"
    torch.save(
        {
            "arm": arm,
            "mode": "token_only",
            "seed": seed,
            "update": UPDATES,
            "p64_sha": EXPECTED_SHA,
            "lambda_KL": float(lam_kl) if use_aa else None,
            "lambda_AA": float(lam_aa) if use_aa else None,
            "lambda_KL_control": float(lam_kl) if not use_aa else None,
            "d_norm": float(d_norm),
            "actor": actor.state_dict(),
            "trainable_params": n_train,
            "ckpt_select_rule": CKPT_SELECT_RULE,
        },
        path,
    )
    actor.eval()
    return actor, log_rows, path, n_train


def load_token_ckpt(path: Path, p64, device):
    actor = build_p104_from_p64(p64, "token_only", device)
    blob = torch.load(path, map_location=device, weights_only=False)
    actor.load_state_dict(blob["actor"], strict=True)
    actor.eval()
    return actor


@torch.no_grad()
def mechanistic_flip_analysis(p64, actors: dict, demos, device, d_norm: float, book, rho) -> list[dict]:
    """Per-step token/action metrics on identical recorded histories."""
    rows = []
    p64.eval()
    for d in demos:
        obs = torch.as_tensor(d["obs"], device=device, dtype=torch.float32)
        prev = torch.as_tensor(d["prev"], device=device, dtype=torch.float32)
        l64, r64, a64 = p64(obs, prev)
        masked64 = mask_handoff_logits(l64.float())
        p64_pi = F.softmax(masked64, dim=-1)
        z64 = torch.argmax(masked64, dim=-1)
        # margin
        top2 = torch.topk(masked64, k=2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        d2, _, _ = compute_action_distance_table(p64, d, device)
        d_normed = (d2 / float(d_norm)).cpu().numpy()

        arm_out = {}
        for arm, actor in actors.items():
            actor.eval()
            ln, rn, an = actor(obs, prev)
            masked = mask_handoff_logits(ln.float())
            pi = F.softmax(masked, dim=-1)
            zn = torch.argmax(masked, dim=-1)
            kl = (p64_pi * (p64_pi.clamp_min(1e-8).log() - F.log_softmax(masked, dim=-1))).sum(-1)
            aw = (torch.as_tensor(d_normed, device=device) * (pi - p64_pi).pow(2)).sum(-1)
            # executed action under frozen P64 residual + each policy's argmax token
            # a from actor already uses frozen residual for token_only
            act_l2 = ((an - a64) ** 2).sum(-1).sqrt()
            arm_out[arm] = {
                "z": zn.cpu().numpy(),
                "kl": kl.cpu().numpy(),
                "aw": aw.cpu().numpy(),
                "act_l2": act_l2.cpu().numpy(),
                "pi": pi,
            }

        for t in range(len(d["obs"])):
            z0 = int(z64[t].item())
            phase = "early" if t < p104.EARLY_K else ("mid" if t < 2 * len(d["obs"]) // 3 else "late")
            base = {
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "skill": d["skill"],
                "role": d.get("role", ""),
                "t": t,
                "phase": phase,
                "z_P64": z0,
                "margin_P64": float(margin[t]),
            }
            for arm in actors:
                za = int(arm_out[arm]["z"][t])
                flip = int(za != z0)
                # action distance of flipped token under frozen residual
                act_dist = float(np.sqrt(d2[t, za].item())) if flip else 0.0
                large_flip = int(flip and act_dist >= LARGE_ACT_FLIP_THRESH)
                rows.append(
                    {
                        **base,
                        "arm": arm,
                        "z_new": za,
                        "argmax_changed": flip,
                        "KL": float(arm_out[arm]["kl"][t]),
                        "action_weighted_dev": float(arm_out[arm]["aw"][t]),
                        "executed_act_l2_vs_P64": float(arm_out[arm]["act_l2"][t]),
                        "flip_act_l2_frozenRes": act_dist,
                        "large_action_flip": large_flip,
                    }
                )
    return rows


@torch.no_grad()
def action_aware_metric_rows(p64, actors: dict, demos, device, d_norm: float) -> list[dict]:
    rows = []
    for d in demos:
        for arm, actor in actors.items():
            kli, sk = token_kl_retention(actor, p64, d, device)
            aai, sa = action_aware_retention(actor, p64, d, device, d_norm)
            rows.append(
                {
                    "arm": arm,
                    "skill": d["skill"],
                    "traj_id": d["traj_id"],
                    "phys_root": d["phys_root"],
                    "role": d.get("role", "preservation"),
                    "n": sk["n"],
                    "token_kl": sk["kl"],
                    "L_AA": sa["L_AA"],
                    "argmax_agree": sa["argmax_agree"],
                    "action_weighted_mass": sa["action_weighted_mass"],
                    "act_l2": sk["act_l2"],
                }
            )
    return rows


def outcome_label(sums, mech, paired, b1_corr_roots, c_corr_roots) -> dict:
    """Map to authorization outcomes A–E (exploratory)."""
    def protected_ok(arm):
        s = sums[arm]
        return s["a1_lost"] == 0 and s["b1_success_lost"] == 0 and s["d_lost"] == 0

    def any_gain(arm):
        gains = [
            r
            for r in paired
            if r["arm"] == arm
            and int(r.get("gain", 0)) == 1
            and r["skill"] in ("B1", "C_Feasible")
            and int(r["P64_SUCCESS"]) == 0
        ]
        for r in gains:
            pr = r["phys_root"]
            if r["skill"] == "B1" and pr not in b1_corr_roots:
                return True
            if r["skill"] == "C_Feasible" and pr not in c_corr_roots:
                return True
        return False

    # Mechanistic: fewer large-action flips on A1 preservation for D vs C
    def large_flips(arm, skill="A1"):
        return sum(
            1
            for r in mech
            if r["arm"] == arm and r["skill"] == skill and int(r.get("large_action_flip", 0)) == 1
        )

    c_ok = protected_ok("C")
    d_ok = protected_ok("D")
    c_gain = any_gain("C")
    d_gain = any_gain("D")
    p104b_c_gain = sums.get("B", {}).get("c_feasible_gain", 0)

    # Known A1 root retention
    known = "plan_a__ball_train_0017"
    known_rows = {r["arm"]: r for r in paired if r.get("traj_id") == known and r["skill"] == "A1"}

    if d_ok and d_gain and (not c_ok or not c_gain) and large_flips("D") <= large_flips("C"):
        label = "A_AA_SUPPORTED"
    elif d_ok and c_ok and (d_gain == c_gain):
        label = "B_SIMILAR_STRONGER_RETENTION"
    elif d_ok and c_ok and not d_gain and not c_gain:
        label = "C_RETAIN_NO_ACQUIRE"
    elif (not d_ok) and (not c_ok):
        label = "D_BOTH_LOSE_OLD"
    elif d_ok and not d_gain and p104b_c_gain > 0 and sums["D"]["c_feasible_gain"] == 0:
        label = "E_AA_OVERCONSTRAINS_ACQUISITION"
    elif d_ok and d_gain:
        label = "A_AA_SUPPORTED" if (not c_ok or large_flips("D") < large_flips("C")) else "B_SIMILAR_STRONGER_RETENTION"
    else:
        label = "MIXED_INCONCLUSIVE"

    return {
        "outcome_label": label,
        "protected_ok": {"C": c_ok, "D": d_ok, "B": protected_ok("B") if "B" in sums else None},
        "root_disjoint_gain": {"C": c_gain, "D": d_gain},
        "large_action_flips_A1_pres": {"C": large_flips("C"), "D": large_flips("D"), "B": large_flips("B")},
        "known_A1_0017": {
            arm: {
                "P64_SUCCESS": known_rows[arm]["P64_SUCCESS"],
                "CAND_SUCCESS": known_rows[arm]["CAND_SUCCESS"],
                "loss": known_rows[arm]["loss"],
            }
            for arm in known_rows
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print("[P106] Phase 0 — provenance, parity, calibration", flush=True)
    p64_sha = sha16(P64_ACTOR)
    if p64_sha != EXPECTED_SHA:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": f"sha {p64_sha}"})
        raise SystemExit(f"P64 sha mismatch: {p64_sha}")
    if not P104B_CKPT.is_file():
        raise SystemExit(f"missing P104-B v2 checkpoint: {P104B_CKPT}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64, book, rho, saved = load_actor(P64_ACTOR, device)
    p64.eval()
    for p in p64.parameters():
        p.requires_grad_(False)
    book = np.asarray(book, np.float64)
    rho = p104.rho_arr(rho)

    corr_raw = p104.load_correction_demos()
    pres_demos = p104.load_preservation_demos()
    ban = p104.eval_banned_phys()
    for d in corr_raw:
        d["phys_root"] = phys_root(d["phys_root"])
    for d in pres_demos:
        d["phys_root"] = phys_root(d["phys_root"])
    corr_roots = {d["phys_root"] for d in corr_raw}
    pres_roots = {d["phys_root"] for d in pres_demos}
    leak = sorted((corr_roots | pres_roots) & ban)
    if leak:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": "sealed_leak", "leak": leak})
        raise SystemExit(f"sealed leak: {leak}")
    if any(d["skill"] == "C" for d in pres_demos):
        raise SystemExit("C preservation must not exist")
    role_overlap = sorted(
        {(d["skill"], d["phys_root"]) for d in corr_raw} & {(d["skill"], d["phys_root"]) for d in pres_demos}
    )
    if role_overlap:
        raise SystemExit(f"correction∩preservation overlap: {role_overlap}")

    exposed_a1 = "ball_train_0017"
    exposed_in_pres = exposed_a1 in {phys_root(d["phys_root"]) for d in pres_demos if d["skill"] == "A1"}
    # Do NOT add exposed eval root to training merely to fix the observed failure.

    corr_demos = p104.annotate_correction_targets(corr_raw, p64, book, rho, device)
    b1_corr_roots = {d["phys_root"] for d in corr_demos if d["skill"] == "B1"}
    c_corr_roots = {d["phys_root"] for d in corr_demos if d["skill"] == "C"}

    arm0 = build_p104_from_p64(p64, "token_only", device)
    parity = {"token_only_init": p104.init_parity(p64, arm0, corr_demos, device)}
    write_json(OUT / "INIT_PARITY.json", parity)
    if not parity["token_only_init"]["pass"]:
        raise SystemExit("init parity failed")

    dnorm_info = estimate_d_norm(p64, pres_demos, device)
    d_norm = float(dnorm_info["D_norm"])
    print(f"[P106] D_norm={d_norm:.6g}", flush=True)

    calib = calibrate_retention(p64, corr_demos, pres_demos, book, rho, device, d_norm)
    write_json(OUT / "RETENTION_CALIBRATION.json", {**calib, "d_norm_info": dnorm_info})
    lam_kl = float(calib["lambda_KL"])
    lam_aa = float(calib["lambda_AA"])
    lam_kl_c = float(calib["lambda_KL_control"])
    print(
        f"[P106] λ_KL={lam_kl:.6g} λ_AA={lam_aa:.6g} λ_KL_control={lam_kl_c:.6g} "
        f"grad_ratio_C/D={calib['grad_norm_ratio_C_over_D']:.4f} L_AA@init={calib['L_AA_at_P64_init_mean']:.2e}",
        flush=True,
    )
    if calib["L_AA_at_P64_init_mean"] > 1e-6:
        raise SystemExit(f"L_AA not ~0 at init: {calib['L_AA_at_P64_init_mean']}")

    provenance = {
        "created": utc(),
        "p64_sha": p64_sha,
        "p104b_ckpt": str(P104B_CKPT),
        "p104b_sha16": sha16(P104B_CKPT),
        "correction_roots": sorted(corr_roots),
        "preservation_roots": sorted(pres_roots),
        "banned_overlap": leak,
        "role_skill_root_overlap": role_overlap,
        "exposed_A1_failure_root": exposed_a1,
        "exposed_A1_in_legitimate_preservation": exposed_in_pres,
        "exposed_A1_added_to_train": False,
        "c_preservation_count": 0,
        "sealed_panel_opened": False,
        "backbone_frozen": True,
        "residual_frozen": True,
        "note": "Same P98 pools as P104. Do not add exposed A1 eval root merely to fix observed failure.",
    }
    write_json(OUT / "PROVENANCE_AUDIT.json", provenance)

    splits = json.loads((p104.P98 / "SPLITS.json").read_text(encoding="utf-8"))
    manifest = {
        "created": utc(),
        "p64_sha": p64_sha,
        "seed": SEED,
        "updates": UPDATES,
        "lr": LR,
        "ckpt_select_rule": CKPT_SELECT_RULE,
        "arms": {
            "A": "immutable_P64",
            "B": "reuse_P104v2_token_only_checkpoint",
            "C": "token_only + strength_matched_KL_control",
            "D": "token_only + KL + action_aware_retention",
        },
        "lambda_KL": lam_kl,
        "lambda_AA": lam_aa,
        "lambda_KL_control": lam_kl_c,
        "d_norm": d_norm,
        "calibration": calib,
        "correction_demos": [
            {"skill": d["skill"], "traj_id": d["traj_id"], "phys_root": d["phys_root"], "n": int(len(d["obs"]))}
            for d in corr_demos
        ],
        "preservation_demos": [
            {"skill": d["skill"], "traj_id": d["traj_id"], "phys_root": d["phys_root"], "n": int(len(d["obs"]))}
            for d in pres_demos
        ],
        "no_ppo_grpo": True,
        "p64_immutable": True,
    }
    write_json(OUT / "TRAIN_MANIFEST.json", manifest)

    starts = p104.lookup_starts()
    panel_dev = {
        "B1": list(splits["dev"]["B1"]),
        "C_Feasible": list(splits["dev"]["C"]),
        "A1": list(splits["dev"]["A1"]),
        "D": list(splits["dev"]["D"]),
        "B1_success": [d["traj_id"] for d in pres_demos if d["skill"] == "B1"],
        "C_train": [d["traj_id"] for d in corr_demos if d["skill"] == "C"],
    }
    for tid in panel_dev["C_Feasible"] + panel_dev["C_train"]:
        p104.resolve_c_start(tid, phys_root(tid), starts)

    print("[P106] Phase 1 — train Arms C and D", flush=True)
    all_logs = []
    ckpts = {"B": str(P104B_CKPT)}
    nparams = {}
    # Arm C: KL control only
    print("[P106] train Arm C (KL control)", flush=True)
    actor_c, logs_c, path_c, n_c = train_arm(
        "C", SEED, p64, book, rho, corr_demos, pres_demos, lam_kl_c, 0.0, d_norm, device
    )
    all_logs.extend(logs_c)
    ckpts["C"] = str(path_c)
    nparams["C"] = n_c
    del actor_c
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("[P106] train Arm D (action-aware)", flush=True)
    actor_d, logs_d, path_d, n_d = train_arm(
        "D", SEED, p64, book, rho, corr_demos, pres_demos, lam_kl, lam_aa, d_norm, device
    )
    all_logs.extend(logs_d)
    ckpts["D"] = str(path_d)
    nparams["D"] = n_d
    del actor_d
    if device.type == "cuda":
        torch.cuda.empty_cache()
    write_csv(LOGS / "TRAIN_CURVES.csv", all_logs)

    print("[P106] Phase 2 — mechanistic + closed-loop", flush=True)
    actors = {
        "B": load_token_ckpt(P104B_CKPT, p64, device),
        "C": load_token_ckpt(Path(ckpts["C"]), p64, device),
        "D": load_token_ckpt(Path(ckpts["D"]), p64, device),
    }
    nparams["B"] = actors["B"].trainable_param_count()

    # Mechanistic on preservation + correction (identical histories)
    flip_rows = mechanistic_flip_analysis(p64, actors, pres_demos + corr_demos, device, d_norm, book, rho)
    write_csv(OUT / "TOKEN_FLIP_ANALYSIS.csv", flip_rows)
    aa_rows = action_aware_metric_rows(p64, actors, pres_demos, device, d_norm)
    # also correction early C action error
    feas_rows = []
    for arm, actor in actors.items():
        feas_rows.extend(p104.feasibility_transitions(p64, actor, corr_demos, book, rho, device, arm))
    write_csv(OUT / "ACTION_AWARE_RETENTION.csv", aa_rows)
    write_csv(OUT / "TOKEN_FEASIBILITY_TRANSITIONS.csv", feas_rows)

    paired = []
    print("[P106] CL Arm A", flush=True)
    rows_a = p104.eval_panels(p64, {}, starts, panel_dev, book, rho, device, train_seed=-1)
    paired.extend(rows_a)
    sums = {"A": p104.summarize(rows_a, "A")}

    for arm in ("B", "C", "D"):
        print(f"[P106] CL Arm {arm}", flush=True)
        rows = p104.eval_panels(p64, {arm: actors[arm]}, starts, panel_dev, book, rho, device, train_seed=SEED)
        rows = [r for r in rows if r["arm"] == arm]
        # first-divergence already partially in eval_panels; for newly lost do CF
        lost = [r for r in rows if r["skill"] in ("A1", "B1_success", "D") and int(r["loss"]) == 1]
        for r in lost:
            st = starts.get(r["traj_id"])
            if st is None:
                continue
            # force P64 token 1 step then resume cand (nondeg diagnostic)
            # reuse closed_loop_swap as coarse token/residual separation
            swap = p104.closed_loop_swap(actors[arm], p64, st, book, rho, device)
            r["swap_candTok_P64res_SUCCESS"] = swap["SUCCESS_RELAXED_12S"]
            swap2 = p104.closed_loop_swap(p64, actors[arm], st, book, rho, device)
            r["swap_P64tok_candRes_SUCCESS"] = swap2["SUCCESS_RELAXED_12S"]
        paired.extend(rows)
        sums[arm] = p104.summarize(rows, arm)
        print(
            f"[P106] Arm {arm}: a1_lost={sums[arm]['a1_lost']} b1s_lost={sums[arm]['b1_success_lost']} "
            f"d_lost={sums[arm]['d_lost']} b1_new={sums[arm]['b1_new_recover']} "
            f"c_feas={sums[arm]['c_feasible_succ']}",
            flush=True,
        )

    write_csv(OUT / "PAIRED_CLOSED_LOOP.csv", paired)

    outcome = outcome_label(sums, flip_rows, paired, b1_corr_roots, c_corr_roots)

    def early_c_stats(arm):
        sub = [r for r in feas_rows if r["arm"] == arm and r["skill"] == "C" and int(r["early"]) == 1]
        n = len(sub) or 1
        return {
            "n": len(sub),
            "feas_p64_rate": sum(int(r["feas_p64"]) for r in sub) / n,
            "feas_new_rate": sum(int(r["feas_new"]) for r in sub) / n,
            "C_to_feasible": sum(1 for r in sub if r["transition"] == "C_to_feasible"),
            "mean_a_err": float(np.mean([r["a_err_actor_l2"] for r in sub])) if sub else float("nan"),
        }

    def mean_aa(arm):
        sub = [r for r in aa_rows if r["arm"] == arm]
        if not sub:
            return {}
        return {
            "token_kl": float(np.mean([r["token_kl"] for r in sub])),
            "L_AA": float(np.mean([r["L_AA"] for r in sub])),
            "argmax_agree": float(np.mean([r["argmax_agree"] for r in sub])),
            "act_l2": float(np.mean([r["act_l2"] for r in sub])),
        }

    def flip_summary(arm, skill=None):
        sub = [r for r in flip_rows if r["arm"] == arm and (skill is None or r["skill"] == skill)]
        return {
            "n": len(sub),
            "n_argmax_flip": sum(int(r["argmax_changed"]) for r in sub),
            "n_large_action_flip": sum(int(r["large_action_flip"]) for r in sub),
            "mean_KL_on_flip": float(
                np.mean([r["KL"] for r in sub if int(r["argmax_changed"])]) or [float("nan")]
            ),
        }

    ec = {a: early_c_stats(a) for a in ("B", "C", "D")}
    aa = {a: mean_aa(a) for a in ("B", "C", "D")}
    fl_a1 = {a: flip_summary(a, "A1") for a in ("B", "C", "D")}

    # Answers
    known = outcome["known_A1_0017"]
    q1 = bool(known.get("D", {}).get("CAND_SUCCESS") == 1) if "D" in known else None
    q2 = (
        outcome["protected_ok"]["D"]
        and (
            not outcome["protected_ok"]["C"]
            or fl_a1["D"]["n_large_action_flip"] < fl_a1["C"]["n_large_action_flip"]
            or (sums["D"]["a1_lost"] < sums["C"]["a1_lost"])
        )
    )
    q3 = sums["D"]["c_feasible_gain"] > 0 or (
        sums["D"]["c_feasible_succ"] >= sums["B"]["c_feasible_succ"] and sums["D"]["a1_lost"] == 0
    )
    # clearer: did D keep a C-Feasible success if B had one, or any C gain
    q3_preserve_c = sums["D"]["c_feasible_succ"] >= 1 or sums["D"]["c_feasible_gain"] >= 1
    q4 = outcome["root_disjoint_gain"]["D"] or sums["D"]["b1_new_recover"] > 0 or sums["D"]["c_feasible_gain"] > 0
    q5_conflict = (sums["D"]["a1_lost"] == 0 and sums["D"]["c_feasible_gain"] == 0 and sums["B"]["c_feasible_gain"] > 0) or (
        fl_a1["D"]["n_large_action_flip"] == 0 and ec["D"]["C_to_feasible"] == 0 and ec["B"]["C_to_feasible"] > 0
    )
    q6 = outcome["outcome_label"] in ("A_AA_SUPPORTED",) and outcome["protected_ok"]["D"] and outcome["root_disjoint_gain"]["D"]
    q7 = (
        "One seed + tiny C-Feasible panel (n=2); exposed A1 root absent from preservation "
        f"(in_pres={exposed_in_pres}); mechanism exploration only."
        if not q6
        else "Still one-seed exploratory; multi-seed would test stability of AA vs KL match."
    )

    summary = {
        "case": "P106_COMPLETE",
        "created": utc(),
        "p64_sha": p64_sha,
        "p64_replaced": False,
        "seed": SEED,
        "lambda_KL": lam_kl,
        "lambda_AA": lam_aa,
        "lambda_KL_control": lam_kl_c,
        "d_norm": d_norm,
        "trainable_params": nparams,
        "ckpt_paths": ckpts,
        "init_parity": parity,
        "arms": sums,
        "early_c_feasibility": ec,
        "retention_offline": aa,
        "flip_summary_A1_pres": fl_a1,
        "outcome": outcome,
        "exposed_A1_in_preservation": exposed_in_pres,
        "questions": {
            "1_AA_prevented_known_A1_failure": q1,
            "2_outperformed_matched_KL_control": q2,
            "3_preserved_C_recovery_capability": q3_preserve_c,
            "4_new_B1_or_C_physical_success": bool(q4),
            "5_acquisition_retention_conflict": bool(q5_conflict),
            "6_justify_multiseed": bool(q6),
            "7_principal_limitation": q7,
        },
        "mechanism_validation_only": True,
    }
    write_json(OUT / "SUMMARY.json", summary)

    report = f"""# FINAL_REPORT — P106 Action-Aware Token Retention Pilot

Generated: {utc()}

## Setup

- P64 sha `{p64_sha}` (**immutable**).
- Arm A: frozen P64. Arm B: **reused** P104 v2 Token-only ckpt (not retrained).
- Arm C: token-only + strength-matched KL control (`λ_KL_control={lam_kl_c:.6g}`).
- Arm D: token-only + KL (`λ_KL={lam_kl:.6g}`) + action-aware (`λ_AA={lam_aa:.6g}`).
- `D_norm={d_norm:.6g}`. L_AA@init={calib['L_AA_at_P64_init_mean']:.2e}.
- Retention loss scale (intended): C={calib['retention_loss_scale_C_at_temp_ref']:.6g}, D={calib['retention_loss_scale_D_intended']:.6g}; D@P104B-AA_ref={calib['retention_loss_scale_D_at_P104B_AA_ref']:.6g}.
- Grad-norm ratio C/D (perturbed audit)={calib['grad_norm_ratio_C_over_D']:.4f}.
- Seed={SEED}, updates={UPDATES}. Backbone+residual frozen. No PPO/GRPO. Sealed panel closed.
- Exposed A1 root `ball_train_0017` in legitimate preservation? **{exposed_in_pres}** (not added).

## Closed-loop SUCCESS_RELAXED_12S

| Arm | A1 ret/base (lost) | B1-succ ret/base (lost) | D ret/base (lost) | B1 new | C-Feasible | C-train |
|---|---:|---:|---:|---:|---:|---:|
| A | {sums['A']['a1_retained']}/{sums['A']['a1_base']} ({sums['A']['a1_lost']}) | {sums['A']['b1_success_retained']}/{sums['A']['b1_success_base']} ({sums['A']['b1_success_lost']}) | {sums['A']['d_retained']}/{sums['A']['d_base']} ({sums['A']['d_lost']}) | {sums['A']['b1_new_recover']} | {sums['A']['c_feasible_succ']}/{sums['A']['c_feasible_n']} | {sums['A']['c_train_succ']}/{sums['A']['c_train_n']} |
| B | {sums['B']['a1_retained']}/{sums['B']['a1_base']} ({sums['B']['a1_lost']}) | {sums['B']['b1_success_retained']}/{sums['B']['b1_success_base']} ({sums['B']['b1_success_lost']}) | {sums['B']['d_retained']}/{sums['B']['d_base']} ({sums['B']['d_lost']}) | {sums['B']['b1_new_recover']} | {sums['B']['c_feasible_succ']}/{sums['B']['c_feasible_n']} | {sums['B']['c_train_succ']}/{sums['B']['c_train_n']} |
| C | {sums['C']['a1_retained']}/{sums['C']['a1_base']} ({sums['C']['a1_lost']}) | {sums['C']['b1_success_retained']}/{sums['C']['b1_success_base']} ({sums['C']['b1_success_lost']}) | {sums['C']['d_retained']}/{sums['C']['d_base']} ({sums['C']['d_lost']}) | {sums['C']['b1_new_recover']} | {sums['C']['c_feasible_succ']}/{sums['C']['c_feasible_n']} | {sums['C']['c_train_succ']}/{sums['C']['c_train_n']} |
| D | {sums['D']['a1_retained']}/{sums['D']['a1_base']} ({sums['D']['a1_lost']}) | {sums['D']['b1_success_retained']}/{sums['D']['b1_success_base']} ({sums['D']['b1_success_lost']}) | {sums['D']['d_retained']}/{sums['D']['d_base']} ({sums['D']['d_lost']}) | {sums['D']['b1_new_recover']} | {sums['D']['c_feasible_succ']}/{sums['D']['c_feasible_n']} | {sums['D']['c_train_succ']}/{sums['D']['c_train_n']} |

Known A1 `plan_a__ball_train_0017`: {json.dumps(known)}

## Mechanistic (A1 preservation flips)

| Arm | argmax flips | large-action flips | mean KL | L_AA | argmax agree |
|---|---:|---:|---:|---:|---:|
| B | {fl_a1['B']['n_argmax_flip']} | {fl_a1['B']['n_large_action_flip']} | {aa['B'].get('token_kl', float('nan')):.4f} | {aa['B'].get('L_AA', float('nan')):.4f} | {aa['B'].get('argmax_agree', float('nan')):.3f} |
| C | {fl_a1['C']['n_argmax_flip']} | {fl_a1['C']['n_large_action_flip']} | {aa['C'].get('token_kl', float('nan')):.4f} | {aa['C'].get('L_AA', float('nan')):.4f} | {aa['C'].get('argmax_agree', float('nan')):.3f} |
| D | {fl_a1['D']['n_argmax_flip']} | {fl_a1['D']['n_large_action_flip']} | {aa['D'].get('token_kl', float('nan')):.4f} | {aa['D'].get('L_AA', float('nan')):.4f} | {aa['D'].get('argmax_agree', float('nan')):.3f} |

Early C feas (skill=C, t<4): B c2f={ec['B']['C_to_feasible']} feas={ec['B']['feas_new_rate']:.2f}; C c2f={ec['C']['C_to_feasible']} feas={ec['C']['feas_new_rate']:.2f}; D c2f={ec['D']['C_to_feasible']} feas={ec['D']['feas_new_rate']:.2f}.

## Outcome label

**{outcome['outcome_label']}** (exploratory, one seed; C-Feasible n=2 not decisive).

## Answers

1. **Did AA prevent known A1 token failure?** {q1}
2. **Did it outperform matched KL control?** {q2}
3. **Preserve new C recovery capability?** {q3_preserve_c}
4. **New B1/C physical successes?** {bool(q4)}
5. **Acquisition–retention conflict?** {bool(q5_conflict)}
6. **Justify multi-seed validation?** {bool(q6)}
7. **Principal remaining limitation?** {q7}

## Stop

Diagnostics+pilot complete. **Do not** launch P107 / PPO / backbone unfreeze / ind panel / P64 replacement without approval.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[P106] DONE outcome={outcome['outcome_label']}", flush=True)


if __name__ == "__main__":
    main()
