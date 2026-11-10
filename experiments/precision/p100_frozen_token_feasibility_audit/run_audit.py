"""P100 Phase 0 — Frozen P64 token feasibility audit (read-only).

Does NOT train, modify checkpoints, change M111/R3F, or run Arms B/C.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/experiments/p100_frozen_token_feasibility_audit"
P64 = ROOT / "results/experiments/p64_ppo_multiround/checkpoints/P64_step4_actor.pt"
P98 = ROOT / "results/experiments/p98_unified_b1_c_consolidation"
P87_TEACHER = ROOT / "results/experiments/p87_k48_actor_retention/teacher"
P99_CACHE = ROOT / "results/experiments/p99_unified_onpolicy_skill_learning/cache"
P99_VER = ROOT / "results/experiments/p99_unified_onpolicy_skill_learning/VERIFIED_CORRECTION_DATASET.csv"

WINDOW = 160
ACT_TOL = 1e-5  # exact representability tolerance
RES_L2_OK = 1e-3  # residual/action match threshold within class A


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def phys_root(tid: str) -> str:
    s = str(tid)
    s = re.sub(r"^(plan_[abcd]__|[ABCD]__)", "", s)
    s = re.sub(r"__[0-9a-fA-F]{8,}$", "", s)
    return s


def load_actor():
    from experiments.precision.p53_independent_grpo64.policy_sample import (
        load_policy_actor,
        mask_handoff_logits,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor, book, rho, saved = load_policy_actor(P64, device)
    actor.eval()
    rho_arr = np.asarray(rho if np.ndim(rho) else np.full(7, float(rho)), np.float64).reshape(7)
    if rho_arr.size == 1:
        rho_arr = np.full(7, float(rho_arr), np.float64)
    return actor, np.asarray(book, np.float64), rho_arr, saved, device, mask_handoff_logits


def token_can_represent(book: np.ndarray, k: int, a: np.ndarray, rho: np.ndarray, tol: float = ACT_TOL) -> bool:
    proto = book[int(k)]
    a = np.asarray(a, np.float64).reshape(7)
    lo = np.maximum(-1.0, proto - rho)
    hi = np.minimum(1.0, proto + rho)
    return bool(np.all(a >= lo - tol) and np.all(a <= hi + tol))


def feasible_tokens(book: np.ndarray, a: np.ndarray, rho: np.ndarray, tol: float = ACT_TOL) -> list[int]:
    return [k for k in range(len(book)) if token_can_represent(book, k, a, rho, tol)]


def best_under_token(book: np.ndarray, k: int, a: np.ndarray, rho: np.ndarray) -> dict:
    proto = book[int(k)]
    a = np.asarray(a, np.float64).reshape(7)
    raw = a - proto
    r = np.clip(raw, -rho, rho)
    recon = np.clip(proto + r, -1.0, 1.0)
    err = recon - a
    return {
        "residual": r.astype(np.float32),
        "recon": recon.astype(np.float32),
        "l2": float(np.linalg.norm(err)),
        "linf": float(np.max(np.abs(err))),
        "feasible": bool(np.max(np.abs(err)) <= ACT_TOL),
    }


@torch.no_grad()
def predict_tokens(actor, obs: np.ndarray, prev: np.ndarray, device, mask_handoff_logits) -> tuple[np.ndarray, np.ndarray]:
    """Causal P64 predictions for each prefix length 1..T."""
    T = int(obs.shape[0])
    z_hat = np.zeros(T, dtype=np.int64)
    r_hat = np.zeros((T, 7), dtype=np.float32)
    for t in range(T):
        o = torch.as_tensor(obs[max(0, t + 1 - WINDOW) : t + 1], device=device, dtype=torch.float32)
        p = torch.as_tensor(prev[max(0, t + 1 - WINDOW) : t + 1], device=device, dtype=torch.float32)
        logits, res, _ = actor(o, p)
        z_hat[t] = int(torch.argmax(mask_handoff_logits(logits[-1].float())).item())
        r_hat[t] = res[-1].detach().cpu().numpy().astype(np.float32)
    return z_hat, r_hat


def classify_step(
    *,
    z_hat: int,
    z_star: int,
    a_star: np.ndarray,
    r_hat: np.ndarray,
    book: np.ndarray,
    rho: np.ndarray,
) -> dict:
    Z = feasible_tokens(book, a_star, rho)
    best_hat = best_under_token(book, z_hat, a_star, rho)
    best_star = best_under_token(book, z_star, a_star, rho) if 0 <= z_star < len(book) else None
    # P64 executed action under predicted token + P64 residual
    proto_h = book[int(z_hat)]
    a_p64 = np.clip(proto_h + np.clip(r_hat, -rho, rho), -1.0, 1.0)
    a_err_p64 = float(np.linalg.norm(a_p64 - a_star))

    if len(Z) == 0:
        cat = "D"
    elif int(z_hat) not in Z:
        cat = "C"
    elif int(z_hat) != int(z_star):
        cat = "B"
    else:
        cat = "A"

    residual_ok = bool(a_err_p64 <= RES_L2_OK) if cat == "A" else None
    return {
        "category": cat,
        "z_hat": int(z_hat),
        "z_star": int(z_star),
        "n_feasible": len(Z),
        "z_hat_in_Z": int(z_hat) in Z,
        "z_star_in_Z": (int(z_star) in Z) if 0 <= z_star < len(book) else False,
        "best_l2_under_z_hat": best_hat["l2"],
        "best_linf_under_z_hat": best_hat["linf"],
        "best_l2_under_z_star": None if best_star is None else best_star["l2"],
        "a_err_p64_l2": a_err_p64,
        "residual_ok_within_A": residual_ok,
        "exact_feasible_under_z_hat": int(best_hat["feasible"]),
    }


def load_npz_demo(path: Path, *, skill: str, traj_id: str, source: str) -> dict | None:
    if not path.is_file():
        return None
    z = np.load(path, allow_pickle=True)
    need = ("tokens", "actions", "obs", "prev")
    if any(k not in z.files for k in need):
        return None
    return {
        "traj_id": traj_id,
        "phys_root": phys_root(traj_id),
        "skill": skill,
        "source": source,
        "tokens": np.asarray(z["tokens"], np.int64),
        "residuals": np.asarray(z["residuals"], np.float32) if "residuals" in z.files else None,
        "actions": np.asarray(z["actions"], np.float32),
        "obs": np.asarray(z["obs"], np.float32),
        "prev": np.asarray(z["prev"], np.float32),
    }


def collect_demos() -> list[dict]:
    demos: list[dict] = []
    # Training-only B1/C from P98 correction targets
    with (P98 / "B1_C_CORRECTION_TARGETS.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            skill = r["skill"]
            tid = r["traj_id"]
            pr = r.get("phys_root") or phys_root(tid)
            if skill == "B1":
                d = load_npz_demo(P87_TEACHER / f"{tid}.npz", skill="B1", traj_id=tid, source="b1_train_expert")
            else:
                d = load_npz_demo(P98 / "cache" / f"c_corr_{pr}.npz", skill="C", traj_id=tid, source="c_train_expert")
            if d is None:
                print(f"[warn] missing demo {skill} {tid}", flush=True)
                continue
            d["phys_root"] = pr
            demos.append(d)

    # Physically verified student-state expert continuations (matched-state only)
    if P99_VER.is_file():
        with P99_VER.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("source") != "student_state":
                    continue
                if int(r.get("verified") or 0) != 1:
                    continue
                skill = r["skill"]
                pr = r["phys_root"]
                st = r.get("student_t") or "0"
                path = P99_CACHE / f"stu_{skill}_{pr}_t{st}.npz"
                d = load_npz_demo(
                    path,
                    skill=skill,
                    traj_id=r["traj_id"],
                    source=f"student_state_t{st}",
                )
                if d is None:
                    print(f"[warn] missing student demo {path.name}", flush=True)
                    continue
                d["phys_root"] = pr
                demos.append(d)
    return demos


def audit_demo(demo: dict, actor, book, rho, device, mask_fn) -> list[dict]:
    obs, prev = demo["obs"], demo["prev"]
    toks, acts = demo["tokens"], demo["actions"]
    T = min(len(toks), len(acts), len(obs), len(prev))
    z_hat, r_hat = predict_tokens(actor, obs[:T], prev[:T], device, mask_fn)
    rows = []
    for t in range(T):
        info = classify_step(
            z_hat=int(z_hat[t]),
            z_star=int(toks[t]),
            a_star=acts[t],
            r_hat=r_hat[t],
            book=book,
            rho=rho,
        )
        rows.append(
            {
                "traj_id": demo["traj_id"],
                "phys_root": demo["phys_root"],
                "skill": demo["skill"],
                "source": demo["source"],
                "t": t,
                "early": int(t <= 4),
                **info,
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    def _rates(subset: list[dict]) -> dict:
        n = len(subset) or 1
        c = Counter(r["category"] for r in subset)
        feasible_hat = sum(int(r["exact_feasible_under_z_hat"]) for r in subset)
        return {
            "n_steps": len(subset),
            "A": c["A"],
            "B": c["B"],
            "C": c["C"],
            "D": c["D"],
            "pct_A": 100.0 * c["A"] / n,
            "pct_B": 100.0 * c["B"] / n,
            "pct_C": 100.0 * c["C"] / n,
            "pct_D": 100.0 * c["D"] / n,
            "pct_exact_feasible_under_z_hat": 100.0 * feasible_hat / n,  # A+B
            "pct_A_plus_B": 100.0 * (c["A"] + c["B"]) / n,
            "mean_best_l2_under_z_hat": float(np.mean([r["best_l2_under_z_hat"] for r in subset])) if subset else 0.0,
            "mean_a_err_p64_l2": float(np.mean([r["a_err_p64_l2"] for r in subset])) if subset else 0.0,
            "A_residual_ok": sum(1 for r in subset if r["category"] == "A" and r["residual_ok_within_A"]),
            "A_residual_bad": sum(1 for r in subset if r["category"] == "A" and r["residual_ok_within_A"] is False),
        }

    by_skill = {sk: _rates([r for r in rows if r["skill"] == sk]) for sk in sorted({r["skill"] for r in rows})}
    early = _rates([r for r in rows if r["early"] == 1])
    late = _rates([r for r in rows if r["early"] == 0])
    expert_only = _rates([r for r in rows if str(r["source"]).endswith("expert") or r["source"] in ("b1_train_expert", "c_train_expert")])
    student = _rates([r for r in rows if str(r["source"]).startswith("student_state")])
    per_root = {}
    for pr in sorted({r["phys_root"] for r in rows}):
        sub = [r for r in rows if r["phys_root"] == pr]
        per_root[pr] = {
            "skill": sub[0]["skill"],
            "source_set": sorted({r["source"] for r in sub}),
            **_rates(sub),
        }
    return {
        "overall": _rates(rows),
        "expert_train": expert_only,
        "student_state": student,
        "early_t0_4": early,
        "t_gt_4": late,
        "by_skill": by_skill,
        "per_root": per_root,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p64_sha = sha16(P64)
    assert p64_sha == "ec08f59816aa2fdd", f"unexpected P64 sha {p64_sha}"

    actor, book, rho, saved, device, mask_fn = load_actor()
    demos = collect_demos()
    print(f"[P100] demos={len(demos)} device={device} rho={rho.tolist()}", flush=True)

    all_rows: list[dict] = []
    for d in demos:
        rows = audit_demo(d, actor, book, rho, device, mask_fn)
        all_rows.extend(rows)
        print(f"  audited {d['skill']} {d['phys_root']} T={len(rows)} src={d['source']}", flush=True)

    summary = summarize(all_rows)
    write_csv(OUT / "STEP_LEVEL_AUDIT.csv", all_rows)
    (OUT / "SUMMARY.json").write_text(json.dumps({"p64_sha": p64_sha, "generated": utc(), **summary}, indent=2), encoding="utf-8")

    # per-root CSV
    root_rows = []
    for pr, st in summary["per_root"].items():
        root_rows.append({"phys_root": pr, **{k: v for k, v in st.items() if k != "source_set"}, "sources": "|".join(st["source_set"])})
    write_csv(OUT / "PER_ROOT_SUMMARY.csv", root_rows)

    # brief markdown for artifacts; full narrative lives in docs/
    ov = summary["overall"]
    ex = summary["expert_train"]
    early = summary["early_t0_4"]
    md = f"""# P100 Frozen-Token Feasibility Audit — Artifact Summary

Generated: {utc()}
P64 sha: `{p64_sha}`
Tolerance: exact action match `{ACT_TOL}`, class-A residual-ok L2 `{RES_L2_OK}`

## Overall (all audited steps)

| Cat | n | pct |
|---|---:|---:|
| A (z_hat=z* and feasible) | {ov['A']} | {ov['pct_A']:.2f}% |
| B (z_hat≠z* but feasible) | {ov['B']} | {ov['pct_B']:.2f}% |
| C (z_hat infeasible; other token feasible) | {ov['C']} | {ov['pct_C']:.2f}% |
| D (no token exactly represents a*) | {ov['D']} | {ov['pct_D']:.2f}% |
| **Exact feasible under z_hat (A+B)** | | **{ov['pct_A_plus_B']:.2f}%** |

Within A: residual_ok={ov['A_residual_ok']}, residual_bad={ov['A_residual_bad']}

## Expert train-only

Exact feasible under z_hat: **{ex['pct_A_plus_B']:.2f}%** (A={ex['pct_A']:.2f}%, B={ex['pct_B']:.2f}%, C={ex['pct_C']:.2f}%, D={ex['pct_D']:.2f}%)

## Early t=0–4

Exact feasible under z_hat: **{early['pct_A_plus_B']:.2f}%** (A={early['pct_A']:.2f}%, B={early['pct_B']:.2f}%, C={early['pct_C']:.2f}%, D={early['pct_D']:.2f}%)

## By skill

"""
    for sk, st in summary["by_skill"].items():
        md += f"- **{sk}**: feasible@z_hat={st['pct_A_plus_B']:.2f}% (A={st['pct_A']:.2f} B={st['pct_B']:.2f} C={st['pct_C']:.2f} D={st['pct_D']:.2f})\n"
    md += "\nSee `docs/saup_frozen_token_feasibility_audit.md` for interpretation and GO/NO-GO.\n"
    (OUT / "ARTIFACT_SUMMARY.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    print(f"[P100] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
