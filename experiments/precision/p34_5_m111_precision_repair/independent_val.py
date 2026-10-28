"""Phase 5 — independent validation (lineage-disjoint from vocab design pool)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env, replay_actions
from experiments.precision.p34_5_m111_precision_repair.io_util import load_csv, utc, write_csv, write_md
from experiments.precision.p34_5_m111_precision_repair.paths import (
    M111_REPLAY,
    OUT,
    P34_MANIFEST,
    ROOT,
)
from experiments.precision.p34_5_m111_precision_repair.repairs import load_book, recon_with_book
from experiments.precision.p34_precision_data_expansion.builder import (
    restore_default_dynamics,
    snapshot_default_dynamics,
)


def _split_fail_lineages(fail_rows: list[dict], *, n_holdout: int = 10, seed: int = 5478):
    """Hold out lineages for validation; rest for vocab design.

    Prefer ~70/30 design/holdout by unique physical roots. Always leave ≥1 design
    root when ≥2 roots exist.
    """
    rng = np.random.default_rng(seed)
    by_root: dict[str, list[dict]] = {}
    for r in fail_rows:
        lin = r.get("lineage_id") or r.get("tag") or ""
        parts = lin.split("__")
        root = parts[1] if len(parts) >= 2 else (r.get("tag") or lin)
        # strip p34_ prefix variants
        if root.startswith("p34_"):
            root = root[len("p34_") :]
        by_root.setdefault(root, []).append(r)
    roots = sorted(by_root.keys())
    rng.shuffle(roots)
    n_roots = len(roots)
    if n_roots <= 1:
        # single root: hold out ~30% of rows by tag, rest design
        rs = list(fail_rows)
        rng.shuffle(rs)
        n_h = max(1, int(round(0.3 * len(rs)))) if len(rs) > 1 else 0
        return rs[n_h:], rs[:n_h], ["row_holdout"]
    n_h = min(n_holdout, max(1, n_roots // 3))
    n_h = min(n_h, n_roots - 1)  # keep ≥1 design root
    hold_roots = set(roots[:n_h])
    design, hold = [], []
    for root, rs in by_root.items():
        (hold if root in hold_roots else design).extend(rs)
    return design, hold, sorted(hold_roots)


def run_independent_validation(
    book_v0: np.ndarray,
    book_chosen: np.ndarray,
    rho,
    *,
    chosen_label: str,
) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    fails = [r for r in load_csv(M111_REPLAY) if str(r.get("ok")).lower() not in ("1", "true")]
    design, hold, hold_roots = _split_fail_lineages(fails, n_holdout=10)
    # Also evaluate P33 test split if present in manifest
    man = load_csv(P34_MANIFEST) if P34_MANIFEST.is_file() else []
    p33_test = [r for r in man if r.get("source") == "P33" and r.get("split") == "test"]

    env = make_env()
    defaults = snapshot_default_dynamics(env.sim)
    detail = []

    def eval_set(name: str, rows: list[dict], book: np.ndarray, label: str):
        ok = 0
        for r in rows:
            p = Path(r.get("ic_path") or "")
            if not p.is_file():
                # P33 paths may point elsewhere
                if not p.is_absolute():
                    p = ROOT / p
            if not p.is_file():
                detail.append(
                    {
                        "set": name,
                        "vocab": label,
                        "tag": r.get("tag"),
                        "ok": 0,
                        "reason": "missing_ic",
                    }
                )
                continue
            blob = pickle.loads(p.read_bytes())
            acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
            h = replay_actions(
                env, blob["capture"], blob["ref"], recon_with_book(book, rho, acts)
            )
            s = int(bool(h.get("SUCCESS_NEW")))
            ok += s
            detail.append(
                {
                    "set": name,
                    "vocab": label,
                    "tag": r.get("tag") or blob.get("tag"),
                    "lineage_id": r.get("lineage_id") or blob.get("lineage_id"),
                    "SUCCESS_NEW": s,
                    "SUCCESS_12S": int(bool(h.get("SUCCESS_12S"))),
                    "reason": h.get("reason"),
                }
            )
        return ok, len(rows)

    print(
        f"=== Independent val: design_fails={len(design)} holdout_fails={len(hold)} "
        f"p33_test={len(p33_test)} chosen={chosen_label} ===",
        flush=True,
    )

    # Dev (design) — for reporting only
    d0, n_d = eval_set("dev_design_fails", design, book_v0, "V0")
    d1, _ = eval_set("dev_design_fails", design, book_chosen, chosen_label)
    # Holdout fails
    h0, n_h = eval_set("holdout_fails", hold, book_v0, "V0")
    h1, _ = eval_set("holdout_fails", hold, book_chosen, chosen_label)
    # P33 test
    p0, n_p = eval_set("p33_test", p33_test, book_v0, "V0")
    p1, _ = eval_set("p33_test", p33_test, book_chosen, chosen_label)

    restore_default_dynamics(env.sim, defaults)
    write_csv(OUT / "INDEPENDENT_VALIDATION_DETAIL.csv", detail)

    hold_lins = {r.get("lineage_id") for r in hold}
    write_csv(
        OUT / "FAIL_DESIGN_HOLDOUT_SPLIT.csv",
        [
            {
                "tag": r.get("tag"),
                "lineage_id": r.get("lineage_id"),
                "partition": "holdout" if r.get("lineage_id") in hold_lins else "design",
            }
            for r in fails
        ],
    )

    text = f"""# INDEPENDENT_VALIDATION — P34.5

Generated: {utc()}

## Protocol

1. Split the 38 continuous-only fails by **physical root lineage**.
2. **Design pool** ({len(design)} roots): may be used to mine added prototypes.
3. **Holdout pool** ({len(hold)} roots, roots={hold_roots}): frozen before selection; not used for prototype mining when rebuilding.
4. Also evaluate **P33 test** split ({len(p33_test)} ICs) under V0 vs chosen vocab.
5. Freeze chosen vocab before reading holdout numbers (this report records both).

## Results (SUCCESS_NEW)

| Set | V0 | {chosen_label} | n |
|---|---:|---:|---:|
| Dev design fails | {d0} | {d1} | {n_d} |
| Holdout fails | {h0} | {h1} | {n_h} |
| P33 test | {p0} | {p1} | {n_p} |

## Interpretation

- Holdout improvement indicates the extension captures a **shared** representation gap, not only memorized design tapes.
- P33 test checks that the extension does not harm previously M111-compatible Precision roots.
- Do **not** claim Precision policy learnability from these numbers.
"""
    write_md(OUT / "INDEPENDENT_VALIDATION.md", text)
    print(text, flush=True)
    return {
        "design_n": n_d,
        "hold_n": n_h,
        "p33_n": n_p,
        "dev_v0": d0,
        "dev_chosen": d1,
        "hold_v0": h0,
        "hold_chosen": h1,
        "p33_v0": p0,
        "p33_chosen": p1,
        "hold_roots": hold_roots,
        "design": design,
        "hold": hold,
    }
