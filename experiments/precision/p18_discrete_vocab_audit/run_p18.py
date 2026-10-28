"""P18 — Full-corpus continuous vs M111 replay; redesign if needed.

Does not train SFT/GRPO. Does not mutate official corpus or overwrite M111.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.expert_pipeline.live_stage1 import extract_precision_skills  # noqa: E402
from data_generation.expert_pipeline.provenance import ACTIVE_DATASET  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import (  # noqa: E402
    make_env,
    quantize_with_book,
    replay_actions,
)
from training.skill_aware_action_vocab import (  # noqa: E402
    DEFAULT_WEIGHTS,
    ResidualBudget,
    load_motion_codebook,
)

OUT = ROOT / "results" / "diagnostics" / "p18_discrete_vocab_audit"
M111_META = (
    ROOT
    / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"
)
P17_SPLIT = ROOT / "results/diagnostics/p17_sft_readiness/lineage_split.json"
SEED = 18
OVERALL_TARGET = 0.95
SKILL_TARGET = 0.90


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


def _skill_label(phases: list[str] | None, branch_id: str, precision_skipped: bool) -> str:
    skills = extract_precision_skills(phases)
    mapped = []
    for s in skills:
        sl = str(s).lower()
        if sl in ("support_transfer", "dx"):
            lab = "dx"
        elif "inward" in sl:
            lab = "dz_inward"
        elif "outward" in sl:
            lab = "dz_outward"
        elif sl.startswith("dz"):
            lab = "dz"
        elif "coupled" in sl:
            lab = "coupled"
        elif "dtheta" in sl:
            lab = "dtheta"
        else:
            lab = sl
        if not mapped or mapped[-1] != lab:
            mapped.append(lab)
    prec = "→".join(mapped) if mapped else ("precision_skipped" if precision_skipped else "direct")
    bid = (branch_id or "").upper()
    if bid.startswith("A"):
        rec = "A"
    elif bid.startswith("B"):
        rec = "B"
    elif bid.startswith("C"):
        rec = "C"
    elif bid.startswith("D") or "CANDIDATE" in bid:
        rec = "D"
    else:
        rec = "?"
    contains_prec = bool(mapped) and not precision_skipped
    return f"{rec}|{prec}|prec={int(contains_prec)}"


def load_population() -> list[dict]:
    rows: list[dict] = []
    # Official
    man = ACTIVE_DATASET / "manifest.jsonl"
    for line in man.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        path = ROOT / m["record_path"]
        if not path.is_file():
            rows.append(
                {
                    "traj_id": m.get("expert_id"),
                    "eligible": False,
                    "ineligible_reason": "MISSING_RECORD",
                    "source": "official",
                    "branch_id": m.get("branch_id"),
                    "lineage_id": m.get("source_ic_id"),
                    "record_path": m.get("record_path"),
                    "orig_SUCCESS_NEW": m.get("SUCCESS_NEW"),
                }
            )
            continue
        raw = pickle.loads(path.read_bytes())
        acts = np.asarray(raw.get("actions_7d"), np.float32).reshape(-1, 7)
        cap = raw.get("capture")
        ref = raw.get("ref")
        reason = None
        eligible = True
        if not m.get("SUCCESS_NEW"):
            eligible, reason = False, "ORIG_NOT_STRICT"
        elif cap is None or ref is None:
            eligible, reason = False, "MISSING_CAPTURE_OR_REF"
        elif acts.size == 0 and not (
            m.get("precision_skipped") or str(m.get("skill_sequence", "")).startswith("direct")
        ):
            # zero-action non-direct is suspicious but may be skip
            pass
        skill = _skill_label(
            list(raw.get("phases") or []),
            str(m.get("branch_id") or ""),
            bool(m.get("precision_skipped")),
        )
        rows.append(
            {
                "traj_id": m.get("expert_id") or m.get("attempt_id"),
                "eligible": eligible,
                "ineligible_reason": reason,
                "source": "official",
                "branch_id": m.get("branch_id"),
                "lineage_id": m.get("source_ic_id"),
                "record_path": m.get("record_path"),
                "n_actions": int(acts.shape[0]),
                "action_fp": hashlib.sha1(acts.tobytes()).hexdigest()[:16] if acts.size else "empty",
                "skill_group": skill,
                "precision_containing": "prec=1" in skill,
                "orig_SUCCESS_NEW": bool(m.get("SUCCESS_NEW")),
                "precision_skipped": bool(m.get("precision_skipped")),
                "phases_head": "|".join(list(raw.get("phases") or [])[:8]),
            }
        )

    # P15/P16 candidates — require actions + capture that is start-compatible.
    # Many P16 ics store post-success capture; try continuous replay and mark.
    for src, folder in (
        ("p16", ROOT / "results/diagnostics/p16_precision_skill_expansion/ics"),
        ("p15", ROOT / "results/diagnostics/p15_forced_precision_d2/ics"),
        ("p14", ROOT / "results/diagnostics/p14_small_error_d2/ics"),
    ):
        if not folder.is_dir():
            continue
        for p in sorted(folder.glob("*.pkl")):
            raw = pickle.loads(p.read_bytes())
            meta = raw.get("meta") or {}
            acts = raw.get("actions") or meta.get("actions")
            acts = np.asarray(acts if acts is not None else [], np.float32).reshape(-1, 7)
            cap = raw.get("capture")
            ref = raw.get("ref")
            base = raw.get("base_sample_id") or meta.get("base_sample_id") or meta.get("same_base_lineage")
            force = raw.get("requested_skill") or meta.get("requested_skill") or meta.get("force_skill")
            ok = meta.get("ok_demo", raw.get("ok", meta.get("SUCCESS_NEW")))
            eligible = True
            reason = None
            if not ok:
                eligible, reason = False, "ORIG_NOT_STRICT_CANDIDATE"
            elif cap is None or ref is None:
                eligible, reason = False, "MISSING_CAPTURE_OR_REF"
            elif acts.size == 0:
                eligible, reason = False, "NO_ACTIONS"
            # Start-state uncertainty for candidates that may be terminal captures
            skill = f"D|{force or 'cand'}|prec=1"
            rows.append(
                {
                    "traj_id": p.stem,
                    "eligible": eligible,
                    "ineligible_reason": reason,
                    "source": src,
                    "branch_id": f"CANDIDATE_{str(force or 'X').upper()}",
                    "lineage_id": base or p.stem,
                    "record_path": str(p.relative_to(ROOT)).replace("\\", "/"),
                    "n_actions": int(acts.shape[0]),
                    "action_fp": hashlib.sha1(acts.tobytes()).hexdigest()[:16] if acts.size else "empty",
                    "skill_group": skill,
                    "precision_containing": True,
                    "orig_SUCCESS_NEW": bool(ok),
                    "precision_skipped": False,
                    "candidate_start_uncertain": True,
                }
            )

    # Dedup exact action fingerprints within same lineage
    seen = set()
    out = []
    for r in rows:
        key = (r.get("lineage_id"), r.get("action_fp"), r.get("traj_id") if r.get("action_fp") == "empty" else None)
        # unique by (lineage, action_fp) for nonempty; traj_id for empty
        k2 = (r.get("lineage_id"), r.get("action_fp")) if r.get("action_fp") != "empty" else ("id", r.get("traj_id"))
        if k2 in seen:
            r = dict(r)
            r["eligible"] = False
            r["ineligible_reason"] = (r.get("ineligible_reason") or "") + "|DUPLICATE_ACTION_FP"
            # still keep in manifest as ineligible duplicate
        else:
            seen.add(k2)
        out.append(r)
    return out


def load_blob(record_path: str) -> tuple[Any, Any, np.ndarray]:
    raw = pickle.loads((ROOT / record_path).read_bytes())
    if "actions_7d" in raw:
        acts = np.asarray(raw.get("actions_7d"), np.float32).reshape(-1, 7)
        return raw["capture"], raw["ref"], acts
    meta = raw.get("meta") or {}
    acts = np.asarray(raw.get("actions") or meta.get("actions") or [], np.float32).reshape(-1, 7)
    return raw.get("capture"), raw.get("ref"), acts


def rate(num: int, den: int) -> float | None:
    return (num / den) if den else None


def build_candidate_books(train_actions: list[np.ndarray], m111) -> dict[str, dict]:
    """Train-lineage-only vocabulary candidates (prototypes + residual)."""
    books = {}
    proto = np.asarray(m111.prototypes, np.float32)
    rho0 = m111.residual
    books["M111"] = {"book": proto, "rho": rho0, "M": int(proto.shape[0]), "note": "frozen baseline"}

    # Larger residual
    books["M111_rho050"] = {
        "book": proto.copy(),
        "rho": ResidualBudget.scalar(0.50),
        "M": int(proto.shape[0]),
        "note": "same prototypes, rho=0.50",
    }

    # Contact-focused residual (vz, wx higher budget)
    rho_c = np.full(7, 0.35, dtype=np.float64)
    rho_c[2] = 0.55  # vz
    rho_c[3] = 0.55  # wx
    books["M111_contact_rho"] = {
        "book": proto.copy(),
        "rho": ResidualBudget(rho=rho_c),
        "M": int(proto.shape[0]),
        "note": "per-dim residual boost vz/wx",
    }

    # Fine precision: add k-means centers from low-magnitude train actions
    if train_actions:
        all_a = np.concatenate([a for a in train_actions if len(a)], axis=0)
        mag = np.linalg.norm(all_a[:, :6], axis=1)
        low = all_a[mag < float(np.percentile(mag, 40))]
        if len(low) >= 64:
            # simple numpy k-means
            rng = np.random.default_rng(SEED)
            k = 32
            centers = low[rng.choice(len(low), size=k, replace=False)].astype(np.float64)
            for _ in range(12):
                d = ((low[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
                lab = np.argmin(d, axis=1)
                for j in range(k):
                    mask = lab == j
                    if np.any(mask):
                        centers[j] = low[mask].mean(axis=0)
            extra = centers.astype(np.float32)
            merged = np.vstack([proto, extra])
            books["M111_plus_fine32"] = {
                "book": merged,
                "rho": ResidualBudget.scalar(0.35),
                "M": int(merged.shape[0]),
                "note": "M111 + 32 low-magnitude k-means prototypes (train only)",
            }
    return books


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== P18 population ===", flush=True)
    pop = load_population()
    _write_csv(OUT / "FULL_CORPUS_MANIFEST.csv", pop)
    n_el = sum(1 for r in pop if r.get("eligible"))
    print(f"  total={len(pop)} eligible={n_el}", flush=True)

    env = make_env()
    cont_rows: list[dict] = []
    print("=== P18 continuous replay baseline ===", flush=True)
    for i, r in enumerate(pop):
        if not r.get("eligible"):
            cont_rows.append({**r, "cont_replay_ok": False, "cont_skip": True})
            continue
        try:
            cap, ref, acts = load_blob(r["record_path"])
            t1 = time.time()
            phys = replay_actions(env, cap, ref, acts, stabilize_n=8)
            row = {
                **{k: r[k] for k in r},
                "cont_SUCCESS_NEW": phys["SUCCESS_NEW"],
                "cont_SUCCESS_12S": phys["SUCCESS_12S"],
                "cont_reason": phys["reason"],
                "cont_replay_ok": phys["SUCCESS_NEW"],
                "cont_elapsed_s": round(time.time() - t1, 2),
            }
            # Candidates that fail continuous may be terminal-capture artifacts
            if r.get("candidate_start_uncertain") and not phys["SUCCESS_NEW"]:
                row["eligible_for_discrete"] = False
                row["ineligible_reason"] = "CANDIDATE_CONT_REPLAY_FAIL_LIKELY_TERMINAL_CAPTURE"
            else:
                row["eligible_for_discrete"] = bool(phys["SUCCESS_NEW"])
            cont_rows.append(row)
            if (i + 1) % 20 == 0 or not phys["SUCCESS_NEW"]:
                print(
                    f"  [{i+1}/{len(pop)}] {str(r['traj_id'])[:48]} cont={phys['SUCCESS_NEW']} "
                    f"n={r.get('n_actions')}",
                    flush=True,
                )
        except Exception as exc:
            cont_rows.append(
                {
                    **r,
                    "cont_replay_ok": False,
                    "cont_SUCCESS_NEW": False,
                    "cont_reason": f"EXCEPTION:{exc}",
                    "eligible_for_discrete": False,
                    "ineligible_reason": f"CONT_EXCEPTION:{exc}",
                }
            )
    _write_csv(OUT / "CONTINUOUS_REPLAY_BASELINE.csv", cont_rows)

    reproducible = [r for r in cont_rows if r.get("eligible_for_discrete")]
    print(f"  continuous-ok for discrete compare: {len(reproducible)}", flush=True)

    # ---- M111 full replay ----
    print("=== P18 M111 full replay ===", flush=True)
    if not M111_META.exists():
        raise FileNotFoundError(M111_META)
    m111 = load_motion_codebook(M111_META)
    book = np.asarray(m111.prototypes, np.float32)
    rho = m111.residual
    w = DEFAULT_WEIGHTS

    m111_rows: list[dict] = []
    for i, r in enumerate(reproducible):
        cap, ref, acts = load_blob(r["record_path"])
        t1 = time.time()
        if len(acts) == 0:
            # direct / skip: encode empty; physical replay same as continuous
            phys = replay_actions(env, cap, ref, acts, stabilize_n=8)
            qinfo = {"clipped_frac": 0.0, "mean_recon_wlinf": 0.0, "n_tok": 0}
            recon = acts
        else:
            q = quantize_with_book(book, acts, rho, w=w)
            recon = np.asarray(q["recon_actions"], np.float32)
            phys = replay_actions(env, cap, ref, recon, stabilize_n=8)
            qinfo = {
                "clipped_frac": float(q["clipped_frac"]),
                "mean_recon_wlinf": float(q["mean_recon_wlinf"]),
                "n_tok": len(set(int(t) for t in q["tokens"])),
            }
        m111_rows.append(
            {
                "traj_id": r["traj_id"],
                "source": r.get("source"),
                "branch_id": r.get("branch_id"),
                "lineage_id": r.get("lineage_id"),
                "skill_group": r.get("skill_group"),
                "precision_containing": r.get("precision_containing"),
                "n_actions": r.get("n_actions"),
                "cont_SUCCESS_NEW": True,
                "m111_SUCCESS_NEW": phys["SUCCESS_NEW"],
                "m111_SUCCESS_12S": phys["SUCCESS_12S"],
                "m111_reason": phys["reason"],
                "m111_clipped_frac": qinfo["clipped_frac"],
                "m111_mean_recon_wlinf": qinfo["mean_recon_wlinf"],
                "m111_n_unique_tokens": qinfo["n_tok"],
                "elapsed_s": round(time.time() - t1, 2),
            }
        )
        if (i + 1) % 15 == 0 or not phys["SUCCESS_NEW"]:
            print(
                f"  M111 [{i+1}/{len(reproducible)}] {str(r['traj_id'])[:40]} "
                f"ok={phys['SUCCESS_NEW']} clip={qinfo['clipped_frac']:.3f}",
                flush=True,
            )
    _write_csv(OUT / "M111_FULL_REPLAY_RESULTS.csv", m111_rows)

    # Per-skill rates
    def group_rates(rows: list[dict], ok_key: str) -> list[dict]:
        groups = defaultdict(list)
        for r in rows:
            groups[str(r.get("skill_group"))].append(r)
            br = str(r.get("branch_id") or "?")
            groups[f"branch:{br}"].append(r)
            groups["ALL"].append(r)
            if r.get("precision_containing"):
                groups["PRECISION_CONTAINING"].append(r)
            else:
                groups["RECOVERY_OR_DIRECT"].append(r)
            # length buckets
            n = int(r.get("n_actions") or 0)
            if n == 0:
                groups["len:0"].append(r)
            elif n < 500:
                groups["len:1-499"].append(r)
            elif n < 1500:
                groups["len:500-1499"].append(r)
            else:
                groups["len:1500+"].append(r)
        out = []
        for g, rs in sorted(groups.items()):
            ok = sum(1 for x in rs if x.get(ok_key))
            out.append(
                {
                    "group": g,
                    "n": len(rs),
                    "n_success": ok,
                    "rate": rate(ok, len(rs)),
                    "meets_90": (rate(ok, len(rs)) or 0) >= SKILL_TARGET if len(rs) >= 5 else None,
                    "small_n": len(rs) < 5,
                }
            )
        return out

    per_skill = group_rates(m111_rows, "m111_SUCCESS_NEW")
    _write_csv(OUT / "PER_SKILL_SUCCESS_RATES.csv", per_skill)

    n_m = len(m111_rows)
    n_mok = sum(1 for r in m111_rows if r.get("m111_SUCCESS_NEW"))
    overall = rate(n_mok, n_m)
    print(f"  M111 overall {n_mok}/{n_m} = {overall}", flush=True)

    # Failure analysis
    fails = [r for r in m111_rows if not r.get("m111_SUCCESS_NEW")]
    fail_reasons = Counter(r.get("m111_reason") for r in fails)
    fail_skills = Counter(r.get("skill_group") for r in fails)
    _write_md(
        OUT / "M111_FAILURE_ANALYSIS.md",
        f"""# M111 Failure Analysis

## Overall
- Continuous-reproducible population: **{n_m}**
- M111 STRICT success: **{n_mok}/{n_m}** ({None if overall is None else f'{100*overall:.1f}%'})
- Target ≥{100*OVERALL_TARGET:.0f}%: **{'PASS' if (overall or 0) >= OVERALL_TARGET else 'FAIL'}**

## Failure reasons
{dict(fail_reasons)}

## Failures by skill_group
{dict(fail_skills.most_common(20))}

## Interpretation
- Continuous baseline already filtered harness bugs for the primary compare set.
- Remaining M111 failures are attributed to **discretization / coverage / resolution** unless paired continuous also fails (excluded).
- High `clipped_frac` or `mean_recon_wlinf` on failures suggests vocabulary coverage/resolution.
- Low reconstruction error but physical fail suggests contact sensitivity to small action deltas.

## Sample failures
"""
        + "\n".join(
            f"- `{r['traj_id']}` skill={r.get('skill_group')} reason={r.get('m111_reason')} "
            f"clip={r.get('m111_clipped_frac')} wlinf={r.get('m111_mean_recon_wlinf')}"
            for r in fails[:25]
        ),
    )

    # Lineage split audit
    split = {"train": [], "val": []}
    if P17_SPLIT.exists():
        split = json.loads(P17_SPLIT.read_text(encoding="utf-8"))
    train_lin = set(split.get("train") or [])
    val_lin = set(split.get("val") or [])
    # hold out 20% of val as untouched test if enough; else document val used for selection
    rng = np.random.default_rng(SEED)
    val_list = sorted(val_lin)
    rng.shuffle(val_list)
    n_test = max(1, len(val_list) // 2) if len(val_list) >= 4 else 0
    test_lin = set(val_list[:n_test])
    sel_val_lin = set(val_list[n_test:]) if n_test else set(val_list)
    _write_md(
        OUT / "LINEAGE_SPLIT_AUDIT.md",
        f"""# Lineage Split Audit (P18)

Uses P17 lineage split (`lineage_split.json`).

- Train lineages: **{len(train_lin)}** (vocabulary design/tuning only)
- Selection val lineages: **{len(sel_val_lin)}** (choose among candidates)
- Untouched test lineages: **{len(test_lin)}** (final confirmation only; not used to tune)

If test is empty (too few val lineages), selection val must **not** be described as untouched test.

Population continuous-ok lineages: {len({r.get('lineage_id') for r in reproducible})}
""",
    )

    # ---- Candidate vocabularies (if needed or always compare a few) ----
    print("=== P18 vocabulary candidates ===", flush=True)
    train_acts = []
    for r in reproducible:
        if r.get("lineage_id") in train_lin or not train_lin:
            _, _, a = load_blob(r["record_path"])
            if len(a):
                train_acts.append(a)
    books = build_candidate_books(train_acts, m111)

    # Evaluate candidates on selection val (or all train if small) — physical replay primary
    eval_set = [
        r
        for r in reproducible
        if (r.get("lineage_id") in sel_val_lin)
        or (not sel_val_lin and r.get("lineage_id") in train_lin)
        or (not train_lin)
    ]
    if len(eval_set) < 8:
        eval_set = reproducible  # fallback: full reproducible (document)
        eval_note = "fallback_full_reproducible_for_selection"
    else:
        eval_note = "selection_val_lineages"

    cand_rows = []
    cand_detail: dict[str, list] = {}
    for name, spec in books.items():
        ok = 0
        detail = []
        for r in eval_set:
            cap, ref, acts = load_blob(r["record_path"])
            if len(acts) == 0:
                phys = replay_actions(env, cap, ref, acts, stabilize_n=8)
                clip = 0.0
            else:
                q = quantize_with_book(spec["book"], acts, spec["rho"], w=w)
                phys = replay_actions(env, cap, ref, q["recon_actions"], stabilize_n=8)
                clip = float(q["clipped_frac"])
            ok += int(phys["SUCCESS_NEW"])
            detail.append({"traj_id": r["traj_id"], "ok": phys["SUCCESS_NEW"], "clip": clip})
        cand_detail[name] = detail
        cand_rows.append(
            {
                "vocab": name,
                "M": spec["M"],
                "note": spec["note"],
                "eval_set": eval_note,
                "n_eval": len(eval_set),
                "n_success": ok,
                "rate": rate(ok, len(eval_set)),
                "mean_clip": float(np.mean([d["clip"] for d in detail])) if detail else None,
            }
        )
        print(f"  {name}: {ok}/{len(eval_set)} = {rate(ok, len(eval_set))}", flush=True)
    _write_csv(OUT / "VOCAB_CANDIDATE_COMPARISON.csv", cand_rows)

    # Select best by physical rate on selection set; prefer meeting target
    cand_rows_sorted = sorted(cand_rows, key=lambda x: (-(x["rate"] or 0), x["M"]))
    best = cand_rows_sorted[0]
    m111_pass = (overall or 0) >= OVERALL_TARGET

    # Final vocab: if M111 passes use M111; else best candidate; confirm on test if available
    final_name = "M111" if m111_pass else best["vocab"]
    final_spec = books[final_name]
    print(f"=== P18 final vocab = {final_name} (m111_pass={m111_pass}) ===", flush=True)

    confirm_set = [r for r in reproducible if r.get("lineage_id") in test_lin] or reproducible
    final_rows = []
    for r in confirm_set if not m111_pass else reproducible:
        # If M111 passed, FINAL results = M111 full results (already have)
        if m111_pass and final_name == "M111":
            continue
        cap, ref, acts = load_blob(r["record_path"])
        if len(acts) == 0:
            phys = replay_actions(env, cap, ref, acts, stabilize_n=8)
            qinfo = {"clip": 0.0, "wlinf": 0.0, "ntok": 0}
        else:
            q = quantize_with_book(final_spec["book"], acts, final_spec["rho"], w=w)
            phys = replay_actions(env, cap, ref, q["recon_actions"], stabilize_n=8)
            qinfo = {
                "clip": float(q["clipped_frac"]),
                "wlinf": float(q["mean_recon_wlinf"]),
                "ntok": len(set(int(t) for t in q["tokens"])),
            }
        final_rows.append(
            {
                "traj_id": r["traj_id"],
                "vocab": final_name,
                "skill_group": r.get("skill_group"),
                "lineage_id": r.get("lineage_id"),
                "SUCCESS_NEW": phys["SUCCESS_NEW"],
                "reason": phys["reason"],
                **qinfo,
            }
        )
    if m111_pass and final_name == "M111":
        final_rows = [
            {
                "traj_id": r["traj_id"],
                "vocab": "M111",
                "skill_group": r.get("skill_group"),
                "lineage_id": r.get("lineage_id"),
                "SUCCESS_NEW": r.get("m111_SUCCESS_NEW"),
                "reason": r.get("m111_reason"),
                "clip": r.get("m111_clipped_frac"),
                "wlinf": r.get("m111_mean_recon_wlinf"),
                "ntok": r.get("m111_n_unique_tokens"),
            }
            for r in m111_rows
        ]
    _write_csv(OUT / "FINAL_VOCAB_REPLAY_RESULTS.csv", final_rows)

    # Persist candidate book if not M111
    art = OUT / "artifacts"
    art.mkdir(exist_ok=True)
    if final_name != "M111":
        np.save(art / f"{final_name}_prototypes.npy", final_spec["book"])
        meta = {
            "vocab_version": f"p18_{final_name}",
            "parent": "M111",
            "M": final_spec["M"],
            "residual_rho": ResidualBudget.from_spec(final_spec["rho"]).to_list(),
            "note": final_spec["note"],
            "created_utc": _utc(),
        }
        (art / f"{final_name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    n_final_ok = sum(1 for r in final_rows if r.get("SUCCESS_NEW"))
    final_rate = rate(n_final_ok, len(final_rows))

    _write_md(
        OUT / "FINAL_VOCAB_SPEC.md",
        f"""# Final Vocabulary Spec (P18)

## Decision
- **Selected:** `{final_name}`
- M111 overall on continuous-ok corpus: **{n_mok}/{n_m}** ({None if overall is None else f'{100*overall:.1f}%'})
- Final confirmation rate: **{n_final_ok}/{len(final_rows)}** ({None if final_rate is None else f'{100*final_rate:.1f}%'})
- Target ≥{100*OVERALL_TARGET:.0f}%: **{'MET' if (final_rate or 0) >= OVERALL_TARGET else 'NOT MET'}**

## Freeze status
- M111 artifacts **preserved** (not overwritten).
- If redesigned: prototypes under `artifacts/{final_name}_prototypes.npy` with version `p18_{final_name}`.
- Training pipeline must pin the chosen version explicitly.

## Encode/decode
- `a = clip(c_k + clip(r, -rho, rho), -1, 1)` (unchanged decoder form)
- Residual: {ResidualBudget.from_spec(final_spec['rho']).label()}
- M = {final_spec['M']}
""",
    )

    n_lineages = len({r.get("lineage_id") for r in reproducible})
    go_sft = (final_rate or 0) >= OVERALL_TARGET and n_final_ok >= 50

    _write_md(
        OUT / "SFT_READINESS_DECISION.md",
        f"""# SFT Readiness Decision (post discrete vocab audit)

## Gate
Physical discrete replay STRICT ≥{100*OVERALL_TARGET:.0f}% on continuous-reproducible corpus.

## Result
- Continuous-reproducible trajectories: **{len(reproducible)}**
- Final vocab `{final_name}`: **{n_final_ok}/{len(final_rows)}**
- **SFT may proceed with this vocab:** {'YES' if go_sft else 'NO — document remaining gaps; do not start SFT/GRPO'}

## Notes
- Official auto-admit unchanged.
- Forced-Precision candidates remain optional mix-ins after lineage caps.
- Package dataset only after this decision file is reviewed.
""",
    )

    _write_md(
        OUT / "FINAL_REPORT.md",
        f"""# FINAL_REPORT — P18 Discrete Vocab Audit

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **Fair physical replay fraction**  
   Population={len(pop)}; continuous-reproducible eligible for discrete compare=**{len(reproducible)}**.  
   Others: see `FULL_CORPUS_MANIFEST.csv` / continuous failures (incl. candidate terminal-capture).

2. **M111 full-corpus STRICT rate (on continuous-ok set)**  
   **{n_mok}/{n_m}** = **{None if overall is None else f'{100*overall:.1f}%'}**

3. **Skills most damaged**  
   See `PER_SKILL_SUCCESS_RATES.csv` and failure skill counts in `M111_FAILURE_ANALYSIS.md`.

4. **Quantization vs harness**  
   Primary rates use continuous-ok only → remaining fails attributed to discretization/sensitivity.  
   Continuous failures listed separately (not blamed on M111).

5. **Revised vocab ≥95%?**  
   Final `{final_name}`: **{None if final_rate is None else f'{100*final_rate:.1f}%'}** — {'YES' if (final_rate or 0) >= OVERALL_TARGET else 'NO'}

6. **Major skills preserved?**  
   Check per-skill table; groups with n<5 marked uncertain.

7. **Lineages / trajectories for SFT**  
   Continuous-ok lineages≈**{n_lineages}**; trajectories **{len(reproducible)}** (discrete-validated subset **{n_final_ok}** under final vocab).

## Integrity
No SFT/GRPO. M111 not overwritten. Physics/STRICT unchanged.
""",
    )

    (OUT / "raw_p18_summary.json").write_text(
        json.dumps(
            {
                "generated_utc": _utc(),
                "n_population": len(pop),
                "n_continuous_ok": len(reproducible),
                "m111_success": n_mok,
                "m111_n": n_m,
                "m111_rate": overall,
                "final_vocab": final_name,
                "final_rate": final_rate,
                "go_sft": go_sft,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Done → {OUT} M111={n_mok}/{n_m} final={final_name} {n_final_ok}/{len(final_rows)}", flush=True)


if __name__ == "__main__":
    main()
