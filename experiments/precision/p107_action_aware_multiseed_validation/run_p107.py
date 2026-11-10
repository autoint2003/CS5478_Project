"""P107 — Action-aware token retention multi-seed validation.

Arms:
  A — frozen P64
  B — KL-only, P106 loss-scale-matched λ (frozen)
  C — KL-only, retention-gradient-matched λ (calibrated training-only)
  D — P106 Action-Aware recipe (λ_KL, λ_AA, D_norm frozen)

Seeds: 1,2,3 (exclude P106 discovery seed 0).
No PPO/GRPO, no backbone/residual unfreeze, no sealed ind panel, no P64 replacement.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.final_independent_success_rate.paths import P64_ACTOR  # noqa: E402
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
from models.p104_policy import build_p104_from_p64  # noqa: E402

import experiments.precision.p104_joint_token_residual_pilot.run_p104 as p104  # noqa: E402
import experiments.precision.p106_action_aware_token_retention.run_p106 as p106  # noqa: E402

OUT = ROOT / "results/experiments/p107_action_aware_multiseed_validation"
CKPT = OUT / "checkpoints"
LOGS = OUT / "logs"
P106_OUT = ROOT / "results/experiments/p106_action_aware_token_retention"
P106_CALIB = P106_OUT / "RETENTION_CALIBRATION.json"
P106_MANIFEST = P106_OUT / "TRAIN_MANIFEST.json"
P97_REC = ROOT / "results/experiments/p97_c_expert_recoverability/C_EXPERT_RECOVERABILITY.csv"
P97_NEW = ROOT / "results/experiments/p97_c_expert_recoverability/new_ics/accepted"

EXPECTED_SHA = "ec08f59816aa2fdd"
SEEDS = [1, 2, 3]  # exclude P106 discovery seed 0
UPDATES = 2000
LR = 1e-3
GRAD_CLIP = 5.0
CKPT_SELECT_RULE = "final_update_2000_no_cl_selection"
PERTURB_STD = 0.01
GRAD_MATCH_BATCHES = 5
LAMBDA_MAX = 10000.0
LARGE_ACT_FLIP_THRESH = 0.5


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


def train_banned_phys() -> set[str]:
    """Same train ban as P104/P106 (exact manifest pools)."""
    return set(p104.eval_banned_phys())


def sealed_eval_phys_roots() -> set[str]:
    """Ban sealed independent panel roots from eval expansion (panel not opened)."""
    ban = set(p104.eval_banned_phys())
    splits = json.loads((p104.P98 / "SPLITS.json").read_text(encoding="utf-8"))
    for block in (splits.get("ind") or {}).values():
        if isinstance(block, list):
            ban |= {phys_root(t) for t in block}
    return ban


def build_eval_panels(pres_demos, corr_demos, starts) -> tuple[dict, dict]:
    """Root-disjoint development panels by eligibility rules (not model performance)."""
    ban = sealed_eval_phys_roots()
    splits = json.loads((p104.P98 / "SPLITS.json").read_text(encoding="utf-8"))
    train_c = {phys_root(x) for x in splits["c_train_phys"]}
    exposed_c = {phys_root(x) for x in splits["c_dev_phys"]}
    exposed_a1 = {phys_root(t) for t in splits["dev"]["A1"]}
    exposed_b1 = {phys_root(t) for t in splits["dev"]["B1"]}
    exposed_d = {phys_root(t) for t in splits["dev"]["D"]}
    p106_exposed = {
        "A1": sorted(exposed_a1),
        "B1": sorted(exposed_b1),
        "D": sorted(exposed_d),
        "C": sorted(exposed_c),
    }

    # C-Feasible: expert-recoverable, not sealed
    feas_rows = list(csv_dict(P97_REC))
    c_feas: list[dict] = []
    for r in feas_rows:
        if r.get("expert_SUCCESS_RELAXED_12S") != "1":
            continue
        pr = phys_root(r["phys_root"])
        if pr in ban:
            continue
        tid = r["traj_id"]
        if pr in train_c:
            exposure = "train"
        elif pr in exposed_c:
            exposure = "previously_exposed"
        else:
            exposure = "newly_selected"
        c_feas.append({"traj_id": tid, "phys_root": pr, "exposure": exposure})
    # unique by phys_root, prefer previously_exposed then newly_selected then train
    rank = {"previously_exposed": 0, "newly_selected": 1, "train": 2}
    c_feas.sort(key=lambda x: (rank[x["exposure"]], x["phys_root"]))
    seen = set()
    c_feas_u = []
    for e in c_feas:
        if e["phys_root"] in seen:
            continue
        seen.add(e["phys_root"])
        c_feas_u.append(e)

    # C-Unfiltered: train correction C + exposed + expert-feasible + new_ics (non-sealed)
    c_unf_map: dict[str, dict] = {}
    for d in corr_demos:
        if d["skill"] != "C":
            continue
        pr = phys_root(d["phys_root"])
        if pr in ban:
            continue
        c_unf_map[pr] = {"traj_id": d["traj_id"], "phys_root": pr, "exposure": "train"}
    if P97_NEW.is_dir():
        for p in sorted(P97_NEW.glob("*.pkl")):
            pr = phys_root(p.stem)
            if pr in ban:
                continue
            c_unf_map.setdefault(
                pr, {"traj_id": p.stem, "phys_root": pr, "exposure": "train"}
            )
    for e in c_feas_u:
        c_unf_map.setdefault(e["phys_root"], dict(e))
    for pr in exposed_c:
        if pr in ban:
            continue
        tid = None
        for t in splits["dev"]["C"]:
            if phys_root(t) == pr:
                tid = t
                break
        c_unf_map.setdefault(
            pr,
            {"traj_id": tid or f"plan_c__{pr}", "phys_root": pr, "exposure": "previously_exposed"},
        )

    c_unf = sorted(c_unf_map.values(), key=lambda x: (rank.get(x["exposure"], 9), x["phys_root"]))

    # Broader preservation panels: union of P98 preservation + previously exposed P106/P104 dev
    a1_pres = []
    for d in pres_demos:
        if d["skill"] == "A1":
            a1_pres.append(
                {
                    "traj_id": d["traj_id"],
                    "phys_root": d["phys_root"],
                    "exposure": "previously_exposed" if d["phys_root"] in exposed_a1 else "preservation_train",
                }
            )
    for tid in splits["dev"]["A1"]:
        pr = phys_root(tid)
        if pr in ban:
            continue
        if any(x["phys_root"] == pr for x in a1_pres):
            continue
        a1_pres.append({"traj_id": tid, "phys_root": pr, "exposure": "previously_exposed"})

    b1_succ = []
    for d in pres_demos:
        if d["skill"] == "B1":
            b1_succ.append(
                {
                    "traj_id": d["traj_id"],
                    "phys_root": d["phys_root"],
                    "exposure": "previously_exposed" if d["phys_root"] in exposed_b1 else "preservation_train",
                }
            )
    d_panel = []
    for d in [x for x in pres_demos if x["skill"] == "D"]:
        d_panel.append(
            {
                "traj_id": d["traj_id"],
                "phys_root": d["phys_root"],
                "exposure": "previously_exposed" if d["phys_root"] in exposed_d else "preservation_train",
            }
        )
    for tid in splits["dev"]["D"]:
        pr = phys_root(tid)
        if pr in ban or any(x["phys_root"] == pr for x in d_panel):
            continue
        d_panel.append({"traj_id": tid, "phys_root": pr, "exposure": "previously_exposed"})

    b1_fail = [{"traj_id": t, "phys_root": phys_root(t), "exposure": "previously_exposed"} for t in splits["dev"]["B1"]]

    # Resolve starts; drop unresolved C entries (eligibility still documented)
    c_feas_res, c_unf_res = [], []
    unresolved = []
    for e in c_feas_u:
        st = resolve_c_start_robust(e["traj_id"], e["phys_root"], starts)
        if st is None:
            unresolved.append(("C_Feasible", e))
        else:
            c_feas_res.append(e)
    for e in c_unf:
        st = resolve_c_start_robust(e["traj_id"], e["phys_root"], starts)
        if st is None:
            unresolved.append(("C_Unfiltered", e))
        else:
            c_unf_res.append(e)
    c_feas_u, c_unf = c_feas_res, c_unf_res

    meta = {
        "selection_rule": (
            "C-Feasible: p97 expert_SUCCESS=1, exclude sealed/P98-ind/frozen ban; "
            "C-Unfiltered: all non-sealed train+exposed+feasible C with resolvable IC; "
            "A1/B1/D: P98 preservation union P104/P106 exposed dev. Not selected by candidate scores."
        ),
        "p106_previously_exposed": p106_exposed,
        "sealed_excluded_n": len(ban),
        "C_Feasible": c_feas_u,
        "C_Unfiltered": c_unf,
        "A1": a1_pres,
        "B1_success": b1_succ,
        "D": d_panel,
        "B1": b1_fail,
        "targets": {"C_Feasible_min": 6, "C_Unfiltered_min": 8},
        "achieved": {"C_Feasible": len(c_feas_u), "C_Unfiltered": len(c_unf)},
        "unresolved_dropped": unresolved,
        "panel_expansion_ok": len(c_feas_u) >= 6 and len(c_unf) >= 8,
        "study_class": (
            "expanded_dev_validation"
            if (len(c_feas_u) >= 6 and len(c_unf) >= 8)
            else "exploratory_insufficient_panel_expansion"
        ),
    }

    panels = {
        "A1": [e["traj_id"] for e in a1_pres],
        "B1_success": [e["traj_id"] for e in b1_succ],
        "D": [e["traj_id"] for e in d_panel],
        "B1": [e["traj_id"] for e in b1_fail],
        "C_Feasible": [e["traj_id"] for e in c_feas_u],
        "C_Unfiltered": [e["traj_id"] for e in c_unf],
        "C_train": [d["traj_id"] for d in corr_demos if d["skill"] == "C"],
    }
    # exposure lookup
    exp_lookup = {}
    for skill, lst in (
        ("A1", a1_pres),
        ("B1_success", b1_succ),
        ("D", d_panel),
        ("B1", b1_fail),
        ("C_Feasible", c_feas_u),
        ("C_Unfiltered", c_unf),
    ):
        for e in lst:
            exp_lookup[(skill, e["traj_id"])] = e["exposure"]
    meta["exposure_lookup"] = {f"{k[0]}::{k[1]}": v for k, v in exp_lookup.items()}
    return panels, meta


def csv_dict(path: Path):
    import csv

    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


class AmbiguousICResolutionError(RuntimeError):
    """Raised when multiple exact-ID candidates exist or prefix matching would be required."""


def _load_c_start_from_pkl(tid: str, pkl: Path, starts: dict):
    import pickle

    from experiments.precision.p51_four_class_mixed_ppo.env_mixed import MixedStart

    ep = pickle.loads(pkl.read_bytes())
    if isinstance(ep, dict) and "ep" in ep:
        ep = ep["ep"]
    st = MixedStart(skill="C", traj_id=tid, family="C", kind="ep", ep=ep)
    starts[tid] = st
    return st


def resolve_c_start_robust(tid: str, pr: str, starts: dict):
    """Fail-closed C IC resolution by exact canonical traj_id only.

    Accepts, in order:
      1) exact tid already present in starts
      2) exact file P97_NEW / '{tid}.pkl'
      3) exact file stage2_abc_final/ics / '{tid}.pkl'

    Never uses phys_root / gbi prefix matching, never picks the first glob hit.
    Returns None when no exact match exists (caller drops the root).
    Raises AmbiguousICResolutionError if more than one exact path exists.
    """
    if tid in starts:
        return starts[tid]

    exact_paths: list[Path] = []
    p97 = P97_NEW / f"{tid}.pkl"
    if p97.is_file():
        exact_paths.append(p97)
    stage2 = ROOT / "results/data_generation/stage2_abc_final/ics" / f"{tid}.pkl"
    if stage2.is_file():
        exact_paths.append(stage2)

    # De-duplicate identical files by resolved path; distinct paths for same tid → ambiguous.
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in exact_paths:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    if len(uniq) > 1:
        raise AmbiguousICResolutionError(
            f"multiple exact IC snapshots for tid={tid}: {[str(p) for p in uniq]}"
        )
    if len(uniq) == 1:
        return _load_c_start_from_pkl(tid, uniq[0], starts)

    # Exact schedule/corr alias only when p104 returns a start whose traj_id equals tid.
    st = p104.resolve_c_start(tid, pr, starts)
    if st is not None and getattr(st, "traj_id", None) == tid:
        starts[tid] = st
        return st
    # Fail closed: do not accept phys_root-base / prefix aliases.
    return None


def param_grad_vec(module) -> torch.Tensor:
    grads = [p.grad.reshape(-1) for p in module.parameters() if p.requires_grad and p.grad is not None]
    if not grads:
        return torch.zeros(1, device=next(module.parameters()).device)
    return torch.cat(grads)


def retention_losses(actor, p64, batch, device, d_norm, *, use_aa: bool):
    L_kl = torch.zeros((), device=device)
    L_aa = torch.zeros((), device=device)
    for d in batch:
        kli, _ = p106.token_kl_retention(actor, p64, d, device)
        L_kl = L_kl + kli
        if use_aa:
            aai, _ = p106.action_aware_retention(actor, p64, d, device, d_norm)
            L_aa = L_aa + aai
    n = max(len(batch), 1)
    return L_kl / n, L_aa / n


def correction_pair_loss(actor, corr_demos, book, rho, device, rng):
    b1 = [d for d in corr_demos if d["skill"] == "B1"]
    c = [d for d in corr_demos if d["skill"] == "C"]
    db = b1[int(rng.integers(0, len(b1)))]
    dc = c[int(rng.integers(0, len(c)))]
    lt_b, _, _ = p106.correction_losses(actor, db, book, rho, device)
    lt_c, _, _ = p106.correction_losses(actor, dc, book, rho, device)
    return lt_b + lt_c


def calibrate_gradient_match(
    p64,
    corr_demos,
    pres_demos,
    book,
    rho,
    device,
    *,
    lam_kl_d: float,
    lam_aa: float,
    d_norm: float,
    lam_kl_loss_scale: float,
) -> dict:
    """Training-only gradient-norm matching for Arm C vs frozen Arm-D retention."""
    by_skill: dict[str, list] = {"A1": [], "B1": [], "D": []}
    for d in pres_demos:
        by_skill[d["skill"]].append(d)
    for sk in ("A1", "B1", "D"):
        if not by_skill[sk]:
            raise RuntimeError(f"missing preservation class {sk}")

    # Fixed batches (seeded, training-only)
    rng = np.random.default_rng(107)
    batches = []
    for i in range(GRAD_MATCH_BATCHES):
        batches.append([by_skill[sk][int(rng.integers(0, len(by_skill[sk])))] for sk in ("A1", "B1", "D")])

    per_batch = []
    ratios = []
    for bi, batch in enumerate(batches):
        seed_all(1000 + bi, deterministic_torch=True)
        actor = build_p104_from_p64(p64, "token_only", device)
        actor.train()
        actor.base.eval()
        with torch.no_grad():
            actor.base.logits.weight.add_(PERTURB_STD * torch.randn_like(actor.base.logits.weight))
            if actor.base.logits.bias is not None:
                actor.base.logits.bias.add_(PERTURB_STD * torch.randn_like(actor.base.logits.bias))

        # Retention D
        L_kl, L_aa = retention_losses(actor, p64, batch, device, d_norm, use_aa=True)
        loss_D = float(lam_kl_d) * L_kl + float(lam_aa) * L_aa
        actor.zero_grad(set_to_none=True)
        loss_D.backward(retain_graph=True)
        g_D = param_grad_vec(actor)
        n_D = float(g_D.norm().item())

        # Unit KL retention grad
        actor.zero_grad(set_to_none=True)
        L_kl2, _ = retention_losses(actor, p64, batch, device, d_norm, use_aa=False)
        L_kl2.backward(retain_graph=True)
        g_KL = param_grad_vec(actor)
        n_KL = float(g_KL.norm().item())

        # Correction grad (same perturbed actor)
        actor.zero_grad(set_to_none=True)
        L_corr = correction_pair_loss(actor, corr_demos, book, rho, device, rng)
        L_corr.backward(retain_graph=True)
        g_corr = param_grad_vec(actor)
        n_corr = float(g_corr.norm().item())

        # Per-class retention grads for D
        class_norms = {}
        for sk in ("A1", "B1", "D"):
            actor.zero_grad(set_to_none=True)
            L_kl_c, L_aa_c = retention_losses(actor, p64, [by_skill[sk][0]], device, d_norm, use_aa=True)
            (float(lam_kl_d) * L_kl_c + float(lam_aa) * L_aa_c).backward(retain_graph=True)
            class_norms[sk] = float(param_grad_vec(actor).norm().item())

        cos_cr = float(
            torch.nn.functional.cosine_similarity(g_corr.unsqueeze(0), g_D.unsqueeze(0)).item()
        ) if g_corr.numel() == g_D.numel() else float("nan")

        ratio = n_D / max(n_KL, 1e-12)
        ratios.append(ratio)
        per_batch.append(
            {
                "batch": bi,
                "L_KL": float(L_kl.detach()),
                "L_AA": float(L_aa.detach()),
                "L_ret_D": float(loss_D.detach()),
                "L_corr": float(L_corr.detach()),
                "grad_norm_ret_D": n_D,
                "grad_norm_unit_KL": n_KL,
                "grad_norm_corr": n_corr,
                "corr_ret_cosine": cos_cr,
                "per_class_ret_grad_norm": class_norms,
                "lambda_ratio_batch": ratio,
            }
        )

    lam_grad = float(np.median(ratios))
    lam_grad = float(min(max(lam_grad, 0.0), LAMBDA_MAX))

    # Verify matched norms at same protocol
    verify = []
    for bi, batch in enumerate(batches):
        seed_all(2000 + bi, deterministic_torch=True)
        actor = build_p104_from_p64(p64, "token_only", device)
        actor.train()
        actor.base.eval()
        with torch.no_grad():
            actor.base.logits.weight.add_(PERTURB_STD * torch.randn_like(actor.base.logits.weight))
        L_kl, L_aa = retention_losses(actor, p64, batch, device, d_norm, use_aa=True)
        actor.zero_grad(set_to_none=True)
        (float(lam_kl_d) * L_kl + float(lam_aa) * L_aa).backward(retain_graph=True)
        n_D = float(param_grad_vec(actor).norm().item())
        actor.zero_grad(set_to_none=True)
        L_kl2, _ = retention_losses(actor, p64, batch, device, d_norm, use_aa=False)
        (float(lam_grad) * L_kl2).backward()
        n_C = float(param_grad_vec(actor).norm().item())
        verify.append({"batch": bi, "grad_norm_D": n_D, "grad_norm_C_matched": n_C, "ratio_C_over_D": n_C / max(n_D, 1e-12)})

    return {
        "created": utc(),
        "method": "retention_grad_norm_match_median_over_fixed_pres_batches_v1",
        "perturb_std": PERTURB_STD,
        "n_batches": GRAD_MATCH_BATCHES,
        "frozen_D_lambda_KL": lam_kl_d,
        "frozen_D_lambda_AA": lam_aa,
        "frozen_D_d_norm": d_norm,
        "P106_lambda_KL_loss_scale_control": lam_kl_loss_scale,
        "lambda_KL_grad_matched": lam_grad,
        "batch_details": per_batch,
        "verify_after_lock": verify,
        "median_ratio": float(np.median(ratios)),
        "mean_verify_ratio_C_over_D": float(np.mean([v["ratio_C_over_D"] for v in verify])),
        "note": "Calibrated on training-only preservation batches; no A1/C eval outcomes used.",
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
    *,
    use_aa: bool,
):
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
    for u in range(1, UPDATES + 1):
        db = b1[int(rng.integers(0, len(b1)))]
        dc = c[int(rng.integers(0, len(c)))]
        lt_b, _, sb = p106.correction_losses(actor, db, book, rho, device)
        lt_c, _, sc = p106.correction_losses(actor, dc, book, rho, device)
        L_corr = lt_b + lt_c
        pres_batch = p104.sample_balanced_pres(by_skill, rng)
        L_kl = torch.zeros((), device=device)
        L_aa = torch.zeros((), device=device)
        demos_batch = pres_batch
        for dp in demos_batch:
            kli, _ = p106.token_kl_retention(actor, p64, dp, device)
            L_kl = L_kl + kli
            if use_aa:
                aai, _ = p106.action_aware_retention(actor, p64, dp, device, d_norm)
                L_aa = L_aa + aai
        if demos_batch:
            L_kl = L_kl / len(demos_batch)
            if use_aa:
                L_aa = L_aa / len(demos_batch)
        loss = L_corr + float(lam_kl) * L_kl + (float(lam_aa) * L_aa if use_aa else 0.0)
        opt.zero_grad(set_to_none=True)
        if float(loss.detach()) > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], GRAD_CLIP)
            opt.step()
        if u % 100 == 0 or u == 1 or u == UPDATES:
            row = {
                "arm": arm,
                "seed": seed,
                "update": u,
                "loss": float(loss.detach()),
                "L_corr": float(L_corr.detach()),
                "L_KL": float(L_kl.detach()),
                "L_AA": float(L_aa.detach()) if use_aa else 0.0,
                "elapsed_s": time.time() - t0,
            }
            log_rows.append(row)
            print(
                f"  [P107-{arm} s={seed}] u={u}/{UPDATES} loss={row['loss']:.4f} "
                f"corr={row['L_corr']:.4f} kl={row['L_KL']:.5f} aa={row['L_AA']:.5f}",
                flush=True,
            )
    path = CKPT / f"arm{arm}_seed{seed}_final.pt"
    torch.save(
        {
            "arm": arm,
            "seed": seed,
            "update": UPDATES,
            "p64_sha": EXPECTED_SHA,
            "lambda_KL": float(lam_kl),
            "lambda_AA": float(lam_aa) if use_aa else None,
            "d_norm": float(d_norm),
            "use_aa": use_aa,
            "actor": actor.state_dict(),
            "trainable_params": n_train,
            "ckpt_select_rule": CKPT_SELECT_RULE,
        },
        path,
    )
    actor.eval()
    return actor, log_rows, path, n_train


def load_token_ckpt(path: Path, p64, device):
    return p106.load_token_ckpt(path, p64, device)


def summarize_arm_rows(rows: list[dict], arm: str, seed: int, exposure_lookup: dict) -> dict:
    sub = [r for r in rows if r["arm"] == arm and int(r.get("train_seed", seed)) == seed]

    def sk(name):
        return [r for r in sub if r["skill"] == name]

    def retained(name):
        base = [r for r in sk(name) if int(r["P64_SUCCESS"]) == 1]
        return {
            "base": len(base),
            "lost": sum(1 for r in base if int(r["CAND_SUCCESS"]) == 0),
            "retained": sum(1 for r in base if int(r["CAND_SUCCESS"]) == 1),
        }

    a1 = retained("A1")
    b1s = retained("B1_success")
    d = retained("D")
    b1 = sk("B1")
    cf = sk("C_Feasible")
    cu = sk("C_Unfiltered")
    # root-disjoint gains: exclude train exposure for C_Feasible
    def gains(skill, allow_exposures):
        out = []
        for r in sk(skill):
            if int(r["P64_SUCCESS"]) != 0 or int(r["CAND_SUCCESS"]) != 1:
                continue
            exp = exposure_lookup.get(f"{skill}::{r['traj_id']}", "")
            if allow_exposures is None or exp in allow_exposures:
                out.append(r)
        return out

    cf_gain_new = gains("C_Feasible", {"newly_selected", "previously_exposed"})
    cf_gain_new_only = gains("C_Feasible", {"newly_selected"})
    cu_gain = gains("C_Unfiltered", None)
    b1_new = [r for r in b1 if int(r["P64_SUCCESS"]) == 0 and int(r["CAND_SUCCESS"]) == 1]
    return {
        "arm": arm,
        "seed": seed,
        "a1_base": a1["base"],
        "a1_lost": a1["lost"],
        "a1_retained": a1["retained"],
        "b1_success_base": b1s["base"],
        "b1_success_lost": b1s["lost"],
        "b1_success_retained": b1s["retained"],
        "d_base": d["base"],
        "d_lost": d["lost"],
        "d_retained": d["retained"],
        "b1_n": len(b1),
        "b1_succ": sum(int(r["CAND_SUCCESS"]) for r in b1),
        "b1_new_recover": len(b1_new),
        "c_feasible_n": len(cf),
        "c_feasible_succ": sum(int(r["CAND_SUCCESS"]) for r in cf),
        "c_feasible_gain": sum(int(r["gain"]) for r in cf),
        "c_feasible_gain_nontrain": len(cf_gain_new),
        "c_feasible_gain_newly_selected": len(cf_gain_new_only),
        "c_unfiltered_n": len(cu),
        "c_unfiltered_succ": sum(int(r["CAND_SUCCESS"]) for r in cu),
        "c_unfiltered_gain": len(cu_gain),
        "protected_ok": a1["lost"] == 0 and b1s["lost"] == 0 and d["lost"] == 0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    print("[P107] Phase 0 — lock P106 coeffs, panels, grad-match calib", flush=True)
    p64_sha = sha16(P64_ACTOR)
    if p64_sha != EXPECTED_SHA:
        write_json(OUT / "SUMMARY.json", {"case": "ABORT", "reason": f"sha {p64_sha}"})
        raise SystemExit(f"P64 sha mismatch: {p64_sha}")
    if not P106_CALIB.is_file() or not P106_MANIFEST.is_file():
        raise SystemExit("missing P106 calibration/manifest")

    p106_calib = json.loads(P106_CALIB.read_text(encoding="utf-8"))
    p106_man = json.loads(P106_MANIFEST.read_text(encoding="utf-8"))
    lam_kl_d = float(p106_calib["lambda_KL"])
    lam_aa = float(p106_calib["lambda_AA"])
    d_norm = float(p106_calib["d_norm"])
    lam_kl_loss = float(p106_calib["lambda_KL_control"])  # Arm B

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p64, book, rho, saved = load_actor(P64_ACTOR, device)
    p64.eval()
    for p in p64.parameters():
        p.requires_grad_(False)
    book = np.asarray(book, np.float64)
    rho = p104.rho_arr(rho)

    # Exact P106 data pools
    corr_raw = p104.load_correction_demos()
    pres_demos = p104.load_preservation_demos()
    ban = train_banned_phys()
    for d in corr_raw:
        d["phys_root"] = phys_root(d["phys_root"])
    demos = pres_demos
    for d in demos:
        d["phys_root"] = phys_root(d["phys_root"])
    leak = sorted(({d["phys_root"] for d in corr_raw} | {d["phys_root"] for d in demos}) & ban)
    if leak:
        raise SystemExit(f"sealed leak in train data: {leak}")
    corr_demos = p104.annotate_correction_targets(corr_raw, p64, book, rho, device)

    parity = {"token_only_init": p104.init_parity(p64, build_p104_from_p64(p64, "token_only", device), corr_demos, device)}
    if not parity["token_only_init"]["pass"]:
        raise SystemExit("init parity failed")
    write_json(OUT / "INIT_PARITY.json", parity)

    starts = p104.lookup_starts()
    panels, panel_meta = build_eval_panels(pres_demos, corr_demos, starts)
    print(
        f"[P107] panels C_Feasible={panel_meta['achieved']['C_Feasible']} "
        f"C_Unfiltered={panel_meta['achieved']['C_Unfiltered']} class={panel_meta['study_class']}",
        flush=True,
    )

    grad_calib = calibrate_gradient_match(
        p64,
        corr_demos,
        pres_demos,
        book,
        rho,
        device,
        lam_kl_d=lam_kl_d,
        lam_aa=lam_aa,
        d_norm=d_norm,
        lam_kl_loss_scale=lam_kl_loss,
    )
    lam_kl_grad = float(grad_calib["lambda_KL_grad_matched"])
    write_json(
        OUT / "RETENTION_CALIBRATION.json",
        {
            "P106_frozen": {
                "lambda_KL": lam_kl_d,
                "lambda_AA": lam_aa,
                "d_norm": d_norm,
                "lambda_KL_loss_scale_control": lam_kl_loss,
                "source": str(P106_CALIB),
            },
            "arm_B": {"recipe": "P106_loss_scale_matched_KL", "lambda_KL": lam_kl_loss},
            "arm_C": {"recipe": "retention_gradient_matched_KL", **grad_calib},
            "arm_D": {"recipe": "P106_action_aware_frozen", "lambda_KL": lam_kl_d, "lambda_AA": lam_aa, "d_norm": d_norm},
        },
    )
    print(
        f"[P107] λ_B(loss-scale)={lam_kl_loss:.6g} λ_C(grad-match)={lam_kl_grad:.6g} "
        f"verify_ratio={grad_calib['mean_verify_ratio_C_over_D']:.4f}",
        flush=True,
    )

    provenance = {
        "created": utc(),
        "p64_sha": p64_sha,
        "seeds": SEEDS,
        "excluded_p106_discovery_seed": 0,
        "p106_coeffs_frozen": True,
        "sealed_panel_opened": False,
        "train_leak": leak,
        "panel_meta": {k: v for k, v in panel_meta.items() if k != "exposure_lookup"},
        "correction_n": len(corr_demos),
        "preservation_n": len(pres_demos),
        "exposed_A1_failure_root_in_train": "ball_train_0017"
        in {d["phys_root"] for d in demos if d["skill"] == "A1"},
        "note": "Do not add exposed eval roots to training. Sealed ind excluded via P98 ind ∪ frozen ban.",
    }
    write_json(OUT / "PROVENANCE_AUDIT.json", provenance)

    manifest = {
        "created": utc(),
        "p64_sha": p64_sha,
        "seeds": SEEDS,
        "updates": UPDATES,
        "lr": LR,
        "ckpt_select_rule": CKPT_SELECT_RULE,
        "arms": {
            "A": "immutable_P64",
            "B": "KL_only_P106_loss_scale_matched",
            "C": "KL_only_retention_gradient_matched",
            "D": "P106_action_aware_frozen_coeffs",
        },
        "lambda_B": lam_kl_loss,
        "lambda_C": lam_kl_grad,
        "lambda_D_KL": lam_kl_d,
        "lambda_D_AA": lam_aa,
        "d_norm": d_norm,
        "panels": panels,
        "panel_expansion_ok": panel_meta["panel_expansion_ok"],
        "study_class": panel_meta["study_class"],
        "p106_manifest_sha16": sha16(P106_MANIFEST),
        "correction_demos": p106_man.get("correction_demos"),
        "preservation_demos": p106_man.get("preservation_demos"),
        "no_ppo_grpo": True,
        "p64_immutable": True,
    }
    write_json(OUT / "TRAIN_MANIFEST.json", manifest)

    # ---- Train ----
    print("[P107] Phase 1 — train B/C/D × seeds", flush=True)
    all_logs = []
    ckpts: dict[str, dict[int, str]] = {"B": {}, "C": {}, "D": {}}
    arm_specs = {
        "B": {"lam_kl": lam_kl_loss, "lam_aa": 0.0, "use_aa": False},
        "C": {"lam_kl": lam_kl_grad, "lam_aa": 0.0, "use_aa": False},
        "D": {"lam_kl": lam_kl_d, "lam_aa": lam_aa, "use_aa": True},
    }
    for arm, spec in arm_specs.items():
        for seed in SEEDS:
            print(f"[P107] train Arm {arm} seed={seed}", flush=True)
            _, logs, path, _ = train_arm(
                arm,
                seed,
                p64,
                book,
                rho,
                corr_demos,
                demos,
                spec["lam_kl"],
                spec["lam_aa"],
                d_norm,
                device,
                use_aa=spec["use_aa"],
            )
            all_logs.extend(logs)
            ckpts[arm][seed] = str(path)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    write_csv(LOGS / "TRAIN_CURVES.csv", all_logs)

    # ---- Eval ----
    print("[P107] Phase 2 — mechanistic + closed-loop", flush=True)
    exposure_lookup = panel_meta["exposure_lookup"]
    paired_all = []
    per_seed = []
    flip_rows = []
    compat_rows = []

    # Arm A once
    print("[P107] CL Arm A", flush=True)
    rows_a = p104.eval_panels(p64, {}, starts, panels, book, rho, device, train_seed=-1)
    for r in rows_a:
        r["exposure"] = exposure_lookup.get(f"{r['skill']}::{r['traj_id']}", "")
        r["arm"] = "A"
    paired_all.extend(rows_a)
    sum_a = summarize_arm_rows(rows_a, "A", -1, exposure_lookup)
    sum_a["seed"] = -1
    per_seed.append(sum_a)

    actors_by_seed: dict[int, dict] = {}
    for seed in SEEDS:
        actors = {
            arm: load_token_ckpt(Path(ckpts[arm][seed]), p64, device) for arm in ("B", "C", "D")
        }
        actors_by_seed[seed] = actors
        # mechanistic on preservation (seed0 actors only once aggregated later — do all seeds)
        flips = p106.mechanistic_flip_analysis(p64, actors, demos + corr_demos, device, d_norm, book, rho)
        for r in flips:
            r["seed"] = seed
        flip_rows.extend(flips)
        for arm, actor in actors.items():
            feas = p104.feasibility_transitions(p64, actor, corr_demos, book, rho, device, arm)
            for r in feas:
                r["seed"] = seed
            compat_rows.extend(feas)

        for arm in ("B", "C", "D"):
            print(f"[P107] CL Arm {arm} seed={seed}", flush=True)
            rows = p104.eval_panels(p64, {arm: actors[arm]}, starts, panels, book, rho, device, train_seed=seed)
            rows = [r for r in rows if r["arm"] == arm]
            for r in rows:
                r["exposure"] = exposure_lookup.get(f"{r['skill']}::{r['traj_id']}", "")
            # first-divergence CF on protected losses
            lost = [r for r in rows if r["skill"] in ("A1", "B1_success", "D") and int(r["loss"]) == 1]
            for r in lost:
                st = starts.get(r["traj_id"])
                if st is None:
                    continue
                swap = p104.closed_loop_swap(actors[arm], p64, st, book, rho, device)
                r["swap_candTok_P64res_SUCCESS"] = swap["SUCCESS_RELAXED_12S"]
                swap2 = p104.closed_loop_swap(p64, actors[arm], st, book, rho, device)
                r["swap_P64tok_candRes_SUCCESS"] = swap2["SUCCESS_RELAXED_12S"]
            paired_all.extend(rows)
            s = summarize_arm_rows(rows, arm, seed, exposure_lookup)
            per_seed.append(s)
            print(
                f"  -> a1_lost={s['a1_lost']} b1s_lost={s['b1_success_lost']} d_lost={s['d_lost']} "
                f"c_feas={s['c_feasible_succ']}/{s['c_feasible_n']} c_unf={s['c_unfiltered_succ']}/{s['c_unfiltered_n']}",
                flush=True,
            )

    write_csv(OUT / "PER_SEED_RESULTS.csv", per_seed)
    write_csv(OUT / "PAIRED_ROOT_OUTCOMES.csv", paired_all)
    write_csv(OUT / "TOKEN_FLIP_ANALYSIS.csv", flip_rows)
    write_csv(OUT / "ACTION_COMPATIBILITY.csv", compat_rows)

    # Aggregate across seeds
    def agg(arm: str) -> dict:
        rows = [r for r in per_seed if r["arm"] == arm and int(r["seed"]) >= 0]
        if not rows:
            return {}
        keys = [
            "a1_lost",
            "a1_retained",
            "a1_base",
            "b1_success_lost",
            "d_lost",
            "b1_new_recover",
            "c_feasible_succ",
            "c_feasible_gain",
            "c_feasible_gain_newly_selected",
            "c_unfiltered_succ",
            "c_unfiltered_gain",
            "protected_ok",
        ]
        out = {"arm": arm, "n_seeds": len(rows)}
        for k in keys:
            vals = [float(r[k]) for r in rows]
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
            out[f"{k}_per_seed"] = vals
        out["protected_ok_all_seeds"] = all(bool(r["protected_ok"]) for r in rows)
        out["a1_lost_any_seed"] = any(int(r["a1_lost"]) > 0 for r in rows)
        return out

    aggs = {a: agg(a) for a in ("B", "C", "D")}

    def flip_sum(arm: str) -> dict:
        sub = [r for r in flip_rows if r["arm"] == arm and r["skill"] == "A1"]
        # mean over seeds of large flips count per seed
        by_seed = {}
        for r in sub:
            by_seed.setdefault(int(r["seed"]), []).append(r)
        large = [sum(int(x["large_action_flip"]) for x in v) for v in by_seed.values()]
        flips = [sum(int(x["argmax_changed"]) for x in v) for v in by_seed.values()]
        return {
            "large_action_flips_mean": float(np.mean(large)) if large else float("nan"),
            "argmax_flips_mean": float(np.mean(flips)) if flips else float("nan"),
        }

    flips_agg = {a: flip_sum(a) for a in ("B", "C", "D")}

    def early_c(arm: str) -> dict:
        sub = [r for r in compat_rows if r["arm"] == arm and r["skill"] == "C" and int(r["early"]) == 1]
        # average feas rate across seeds
        by_seed = {}
        for r in sub:
            by_seed.setdefault(int(r["seed"]), []).append(r)
        rates = []
        c2f = []
        aerr = []
        for v in by_seed.values():
            n = len(v) or 1
            rates.append(sum(int(x["feas_new"]) for x in v) / n)
            c2f.append(sum(1 for x in v if x["transition"] == "C_to_feasible"))
            aerr.append(float(np.mean([x["a_err_actor_l2"] for x in v])))
        return {
            "feas_new_rate_mean": float(np.mean(rates)) if rates else float("nan"),
            "C_to_feasible_mean": float(np.mean(c2f)) if c2f else float("nan"),
            "mean_a_err_mean": float(np.mean(aerr)) if aerr else float("nan"),
        }

    ec = {a: early_c(a) for a in ("B", "C", "D")}

    # Decision
    d_better_b = (
        aggs["D"]["a1_lost_mean"] < aggs["B"]["a1_lost_mean"] - 1e-9
        or (aggs["D"]["protected_ok_all_seeds"] and not aggs["B"]["protected_ok_all_seeds"])
    ) and (
        aggs["D"]["c_feasible_succ_mean"] + aggs["D"]["c_unfiltered_succ_mean"]
        >= aggs["B"]["c_feasible_succ_mean"] + aggs["B"]["c_unfiltered_succ_mean"] - 1e-9
    )
    d_better_c = (
        aggs["D"]["a1_lost_mean"] < aggs["C"]["a1_lost_mean"] - 1e-9
        or (aggs["D"]["protected_ok_all_seeds"] and not aggs["C"]["protected_ok_all_seeds"])
    ) and (
        aggs["D"]["c_feasible_succ_mean"] + aggs["D"]["c_unfiltered_succ_mean"]
        >= aggs["C"]["c_feasible_succ_mean"] + aggs["C"]["c_unfiltered_succ_mean"] - 1e-9
    )
    # comparable grad-matched on the A1/C axes (do not require full protected_ok:
    # D-skill losses can fail protected_ok for both C and D without speaking to AA uniqueness)
    c_comparable = (
        abs(aggs["D"]["a1_lost_mean"] - aggs["C"]["a1_lost_mean"]) < 0.34  # <1 loss on avg of 3
        and abs(aggs["D"]["c_feasible_succ_mean"] - aggs["C"]["c_feasible_succ_mean"]) < 0.5
    )
    # Q3 is specifically A1 retention across seeds — not full A1/B1/D protected panel.
    a1_generalizes = (not aggs["D"]["a1_lost_any_seed"]) and aggs["D"]["a1_lost_mean"] == 0.0
    c_new_roots = aggs["D"]["c_feasible_gain_newly_selected_mean"] > 0
    inconsistent = (
        (aggs["D"]["a1_lost_any_seed"] and aggs["D"]["a1_lost_mean"] < 1.0)
        or (aggs["D"]["c_feasible_succ_std"] > 0.8)
    )

    if d_better_b and d_better_c and a1_generalizes and not inconsistent:
        decision = "SUPPORTS_ACTION_AWARE"
        next_step = "consider_sealed_independent_validation_after_human_approval"
    elif c_comparable and a1_generalizes and not d_better_c:
        # Grad-matched KL matches AA on A1 (/~C). Do NOT claim regularization strength
        # is causally proven to explain all AA effects — only that unique AA benefit
        # is not demonstrated under this matching protocol.
        decision = "NO_DEMONSTRATED_UNIQUE_AA_BENEFIT"
        next_step = "stop_no_p108"
    elif inconsistent or not panel_meta["panel_expansion_ok"]:
        decision = "INCONCLUSIVE"
        next_step = "data_expansion_or_stop"
    elif d_better_b and not d_better_c:
        decision = "BEATS_LOSS_SCALE_NOT_GRAD_MATCH"
        next_step = "stop_or_reassess_AA_vs_strength"
    else:
        decision = "MIXED_INCONCLUSIVE"
        next_step = "data_expansion_or_stop"

    q1 = bool(d_better_b)
    q2 = bool(d_better_c)
    q3 = bool(a1_generalizes)
    q4 = bool(c_new_roots)  # require newly_selected root gains; do not credit train-root C successes
    q5 = bool(d_better_b and d_better_c and a1_generalizes)
    q6 = next_step

    summary = {
        "case": "P107_COMPLETE",
        "created": utc(),
        "p64_sha": p64_sha,
        "p64_replaced": False,
        "seeds": SEEDS,
        "study_class": panel_meta["study_class"],
        "panel_expansion_ok": panel_meta["panel_expansion_ok"],
        "lambdas": {"B_loss_scale": lam_kl_loss, "C_grad_match": lam_kl_grad, "D_KL": lam_kl_d, "D_AA": lam_aa},
        "grad_match_verify_ratio_mean": grad_calib["mean_verify_ratio_C_over_D"],
        "aggregates": aggs,
        "flips_A1": flips_agg,
        "early_c": ec,
        "decision": decision,
        "next_step": next_step,
        "questions": {
            "1_outperform_loss_scale_KL": q1,
            "2_outperform_grad_matched_KL": q2,
            "3_A1_retention_generalizes_across_seeds": q3,
            "4_C_acquisition_generalizes_new_roots": q4,
            "5_genuine_tradeoff_improvement": q5,
            "6_next_step": q6,
        },
        "ckpts": ckpts,
        "mechanism_validation_only": True,
        "replacement_authorized": False,
    }
    write_json(OUT / "SUMMARY.json", summary)

    report = f"""# FINAL_REPORT — P107 Action-Aware Multi-Seed Validation

Generated: {utc()}

## Setup

- P64 `{p64_sha}` immutable. Seeds **{SEEDS}** (P106 discovery seed 0 excluded).
- Arm A: frozen P64.
- Arm B: KL-only, P106 **loss-scale** λ=`{lam_kl_loss:.6g}` (frozen).
- Arm C: KL-only, **gradient-matched** λ=`{lam_kl_grad:.6g}` (train-only calib; verify C/D grad ratio mean=`{grad_calib['mean_verify_ratio_C_over_D']:.4f}`).
- Arm D: P106 Action-Aware frozen (`λ_KL={lam_kl_d:.6g}`, `λ_AA={lam_aa:.6g}`, `D_norm={d_norm:.6g}`).
- Updates={UPDATES}. Backbone+residual frozen. Sealed ind **not** opened.
- Study class: **{panel_meta['study_class']}** (C-Feasible n={panel_meta['achieved']['C_Feasible']}, C-Unfiltered n={panel_meta['achieved']['C_Unfiltered']}).

## Aggregates (mean±std over seeds)

| Arm | A1 lost | B1-succ lost | D lost | C-Feasible succ | C-Unf succ | protected_ok all |
|---|---:|---:|---:|---:|---:|---|
| B | {aggs['B']['a1_lost_mean']:.2f}±{aggs['B']['a1_lost_std']:.2f} | {aggs['B']['b1_success_lost_mean']:.2f} | {aggs['B']['d_lost_mean']:.2f} | {aggs['B']['c_feasible_succ_mean']:.2f}±{aggs['B']['c_feasible_succ_std']:.2f} | {aggs['B']['c_unfiltered_succ_mean']:.2f} | {aggs['B']['protected_ok_all_seeds']} |
| C | {aggs['C']['a1_lost_mean']:.2f}±{aggs['C']['a1_lost_std']:.2f} | {aggs['C']['b1_success_lost_mean']:.2f} | {aggs['C']['d_lost_mean']:.2f} | {aggs['C']['c_feasible_succ_mean']:.2f}±{aggs['C']['c_feasible_succ_std']:.2f} | {aggs['C']['c_unfiltered_succ_mean']:.2f} | {aggs['C']['protected_ok_all_seeds']} |
| D | {aggs['D']['a1_lost_mean']:.2f}±{aggs['D']['a1_lost_std']:.2f} | {aggs['D']['b1_success_lost_mean']:.2f} | {aggs['D']['d_lost_mean']:.2f} | {aggs['D']['c_feasible_succ_mean']:.2f}±{aggs['D']['c_feasible_succ_std']:.2f} | {aggs['D']['c_unfiltered_succ_mean']:.2f} | {aggs['D']['protected_ok_all_seeds']} |

A1 large-action flips (mean over seeds): B={flips_agg['B']['large_action_flips_mean']:.1f}, C={flips_agg['C']['large_action_flips_mean']:.1f}, D={flips_agg['D']['large_action_flips_mean']:.1f}.

Early C feas rate: B={ec['B']['feas_new_rate_mean']:.2f}, C={ec['C']['feas_new_rate_mean']:.2f}, D={ec['D']['feas_new_rate_mean']:.2f}.

## Decision

**{decision}** → next: `{next_step}`

P107 does **not** authorize P64 replacement or sealed-panel opening.

## Answers

1. **Outperform loss-scale-matched KL?** {q1}
2. **Outperform gradient-matched KL?** {q2}
3. **A1 retention generalizes across seeds?** {q3}
4. **C acquisition generalizes to new physical roots?** {q4}
5. **Genuine acquisition–retention tradeoff improvement?** {q5}
6. **Next step?** {q6}

## Stop

Stop after P107. No P108 / PPO / backbone unfreeze / ind panel / P64 replacement without approval.
"""
    (OUT / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(report, flush=True)
    print(f"[P107] DONE decision={decision}", flush=True)


if __name__ == "__main__":
    main()
