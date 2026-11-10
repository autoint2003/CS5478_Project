"""P107 integrity audit — generate mapping/provenance CSVs and classify IC resolution.

Does not train. Does not open sealed panel. Read-only except writing audit deliverables under OUT.
"""
from __future__ import annotations

import csv
import hashlib
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.p104_joint_token_residual_pilot.run_p104 import (  # noqa: E402
    P97_NEW,
    lookup_starts,
    phys_root,
    resolve_c_start,
)
import experiments.precision.p107_action_aware_multiseed_validation.run_p107 as p107  # noqa: E402

OUT = ROOT / "results/experiments/p107_action_aware_multiseed_validation"
STAGE2 = ROOT / "results/data_generation/stage2_abc_final/ics"
P97_REC = ROOT / "results/experiments/p97_c_expert_recoverability/C_EXPERT_RECOVERABILITY.csv"
P98_SPLITS = ROOT / "results/experiments/p98_unified_b1_c_consolidation/SPLITS.json"


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def strip_plan(tid: str) -> str:
    return re.sub(r"^(plan_[abc]__|A__|B__|C__|D__|PILOT_C__|PILOT_B__)", "", str(tid), flags=re.I)


def canonical_physical_id(tid: str) -> str:
    """Full identity after plan/prefix strip; keep hash suffix (no phys_root truncation)."""
    return strip_plan(tid)


def find_exact_pkl(tid: str) -> tuple[Path | None, str]:
    p = P97_NEW / f"{tid}.pkl"
    if p.is_file():
        return p, "p97_new_exact"
    p2 = STAGE2 / f"{tid}.pkl"
    if p2.is_file():
        return p2, "stage2_exact"
    return None, ""


def prefix_candidates(tid: str, pr: str) -> list[Path]:
    pools: list[Path] = []
    if P97_NEW.is_dir():
        pools.extend(sorted(P97_NEW.glob("*.pkl")))
    if STAGE2.is_dir():
        pools.extend(sorted(STAGE2.glob("*.pkl")))
    pr_l = pr.lower()
    tid_l = tid.lower()
    parts = pr_l.split("_")
    pr_prefix = parts[0] + "_" + parts[1] if len(parts) >= 2 else pr_l
    out = []
    for pkl in pools:
        stem = pkl.stem.lower()
        if (
            pr_l in stem
            or tid_l in stem
            or stem in tid_l
            or (pr_prefix and pr_prefix in stem and "gbi_" in pr_prefix)
        ):
            out.append(pkl)
    return out


def ep_fingerprint(ep) -> dict:
    if isinstance(ep, dict) and "ep" in ep:
        ep = ep["ep"]
    if not isinstance(ep, dict):
        return {"type": type(ep).__name__}
    out = {
        "traj_id": ep.get("traj_id"),
        "lineage_id": ep.get("lineage_id"),
        "fingerprint": ep.get("fingerprint"),
    }
    return out


def main() -> None:
    man = json.loads((OUT / "TRAIN_MANIFEST.json").read_text(encoding="utf-8"))
    paired = list(csv.DictReader((OUT / "PAIRED_ROOT_OUTCOMES.csv").open(encoding="utf-8")))
    panels = man["panels"]
    starts = lookup_starts()
    splits = json.loads(P98_SPLITS.read_text(encoding="utf-8"))
    train_c_raw = list(splits["c_train_phys"])
    exposed_c_raw = list(splits["c_dev_phys"])
    train_c = {phys_root(x) for x in train_c_raw}
    exposed_c = {phys_root(x) for x in exposed_c_raw}
    train_canon = {canonical_physical_id(x) for x in train_c_raw}
    exposed_canon = {canonical_physical_id(x) for x in exposed_c_raw}

    p97_by_tid = {
        r["traj_id"]: r
        for r in csv.DictReader(P97_REC.open(encoding="utf-8"))
    }

    # --- CANONICAL_ROOT_MAPPING + collisions ---
    all_ids: set[str] = set()
    for v in panels.values():
        all_ids.update(v)
    all_ids.update(train_c_raw)
    all_ids.update(exposed_c_raw)
    for block in (splits.get("dev") or {}).values():
        if isinstance(block, list):
            all_ids.update(block)
    for r in p97_by_tid.values():
        all_ids.add(r["traj_id"])
        all_ids.add(r["phys_root"])
    if P97_NEW.is_dir():
        all_ids.update(p.stem for p in P97_NEW.glob("*.pkl"))
    for t, s in starts.items():
        if getattr(s, "skill", None) == "C":
            all_ids.add(t)

    by_trunc: dict[str, set[str]] = defaultdict(set)
    mapping_rows = []
    for tid in sorted(all_ids):
        trunc = phys_root(tid)
        canon = canonical_physical_id(tid)
        by_trunc[trunc].add(tid)
        mapping_rows.append(
            {
                "source_id": tid,
                "canonical_physical_id": canon,
                "phys_root_truncated": trunc,
                "truncation_strips_hash": int(canon != trunc and "__" in strip_plan(tid)),
                "in_C_Feasible": int(tid in panels["C_Feasible"]),
                "in_C_Unfiltered": int(tid in panels["C_Unfiltered"]),
                "in_C_train_panel": int(tid in panels.get("C_train", [])),
                "in_c_train_phys": int(tid in train_c_raw or trunc in train_c),
                "in_c_dev_phys": int(tid in exposed_c_raw or trunc in exposed_c),
            }
        )

    collision_rows = []
    for trunc, tids in sorted(by_trunc.items()):
        if len(tids) <= 1:
            continue
        canons = sorted({canonical_physical_id(t) for t in tids})
        # distinct if canonical IDs differ beyond truncated-hex alias of same stem
        hex_suffixes = set()
        bases = set()
        for c in canons:
            m = re.match(r"^(.*?)__([0-9a-f]{8,})$", c, re.I)
            if m:
                bases.add(m.group(1))
                hex_suffixes.add(m.group(2))
            else:
                bases.add(c)
                hex_suffixes.add("")
        # same base with different non-nested hex → true distinct collision
        by_base: dict[str, set[str]] = defaultdict(set)
        for c in canons:
            m = re.match(r"^(.*?)__([0-9a-f]{8,})$", c, re.I)
            if m:
                by_base[m.group(1)].add(m.group(2))
            else:
                by_base[c].add("")
        true_distinct = False
        for b, hs in by_base.items():
            nonempty = [h for h in hs if h]
            # aliases if one hex is prefix of another
            if len(nonempty) <= 1:
                continue
            nested = all(
                any(a == b_ or a.startswith(b_) or b_.startswith(a) for b_ in nonempty if b_ != a)
                for a in nonempty
            )
            if not nested:
                true_distinct = True
        collision_rows.append(
            {
                "phys_root_truncated": trunc,
                "n_source_ids": len(tids),
                "source_ids": "|".join(sorted(tids)),
                "canonical_ids": "|".join(canons),
                "alias_only_truncated_hex": int(not true_distinct),
                "true_distinct_physical_collision": int(true_distinct),
            }
        )

    # --- IC_RESOLUTION_AUDIT ---
    ic_rows = []
    for skill in ("C_Feasible", "C_Unfiltered"):
        for tid in panels[skill]:
            pr = phys_root(tid)
            canon = canonical_physical_id(tid)
            starts_local = dict(starts)
            st = resolve_c_start(tid, pr, starts_local)
            method = "unresolved"
            path = ""
            snap_id = ""
            sha = ""
            amb = 0
            n_pref = 0
            exact_ok = 0
            fail_closed = "FAIL"
            fp = {}
            if st is not None:
                p, src = find_exact_pkl(tid)
                if p is not None:
                    method = src
                    path = str(p)
                    snap_id = p.stem
                    sha = sha16(p)
                    exact_ok = 1
                    fail_closed = "PASS_EXACT_PKL"
                    try:
                        fp = ep_fingerprint(pickle.loads(p.read_bytes()))
                    except Exception as e:
                        fp = {"error": str(e)}
                elif tid in starts:
                    method = "schedule_exact_tid"
                    path = f"schedule:{tid}"
                    snap_id = tid
                    exact_ok = 1
                    fail_closed = "PASS_SCHEDULE_EXACT"
                    fp = ep_fingerprint(starts[tid].ep)
                else:
                    # p104 matched via phys_root base / corr alias
                    method = "p104_nonexact_alias"
                    snap_id = getattr(st, "traj_id", "")
                    path = f"alias:{snap_id}"
                    amb = 1
                    fail_closed = "FAIL_NONEXACT_P104"
                    fp = ep_fingerprint(st.ep)
            else:
                cands = prefix_candidates(tid, pr)
                n_pref = len(cands)
                if cands:
                    method = "prefix_first_match"
                    path = str(cands[0])
                    snap_id = cands[0].stem
                    sha = sha16(cands[0])
                    amb = 1
                    fail_closed = "FAIL_PREFIX_AMBIGUOUS" if n_pref != 1 else "FAIL_PREFIX_SINGLE"
                    try:
                        fp = ep_fingerprint(pickle.loads(cands[0].read_bytes()))
                    except Exception as e:
                        fp = {"error": str(e)}
                else:
                    method = "unresolved"
                    fail_closed = "FAIL_MISSING"

            # membership via truncated (historical) and canonical
            if pr in train_c:
                memb_trunc = "train"
            elif pr in exposed_c:
                memb_trunc = "previously_exposed"
            else:
                memb_trunc = "newly_selected"

            # canonical membership: train if any train id shares trunc OR exact canon prefix-match of train hashes
            if any(phys_root(x) == pr for x in train_c_raw) or canon in train_canon:
                memb_canon = "train"
            elif any(phys_root(x) == pr for x in exposed_c_raw) or canon in exposed_canon:
                memb_canon = "previously_exposed"
            else:
                memb_canon = "newly_selected"

            # recorded exposure from paired Arm A
            exp_rec = ""
            for r in paired:
                if r["skill"] == skill and r["traj_id"] == tid and r["arm"] == "A":
                    exp_rec = r.get("exposure", "")
                    break

            src_manifest = ""
            if tid in p97_by_tid:
                src_manifest = "p97_C_EXPERT_RECOVERABILITY"
            elif tid in panels.get("C_train", []):
                src_manifest = "correction_train_C"
            elif (P97_NEW / f"{tid}.pkl").is_file():
                src_manifest = "p97_new_ics"
            else:
                src_manifest = "panel_other"

            succ_cells = {}
            for arm in ("A", "B", "C", "D"):
                seeds = [-1] if arm == "A" else [1, 2, 3]
                for seed in seeds:
                    hits = [
                        r
                        for r in paired
                        if r["traj_id"] == tid
                        and r["skill"] == skill
                        and r["arm"] == arm
                        and int(r["train_seed"]) == seed
                    ]
                    key = f"succ_{arm}_s{seed}"
                    succ_cells[key] = int(hits[0]["CAND_SUCCESS"]) if hits else ""

            invalid = int(amb == 1 or exact_ok == 0)
            ic_rows.append(
                {
                    "panel": skill,
                    "traj_id": tid,
                    "canonical_physical_id": canon,
                    "phys_root_truncated": pr,
                    "source_manifest": src_manifest,
                    "p97_phys_root_field": (p97_by_tid.get(tid) or {}).get("phys_root", ""),
                    "exposure_recorded": exp_rec,
                    "membership_truncated_phys_root": memb_trunc,
                    "membership_canonical": memb_canon,
                    "original_resolve_method": method,
                    "fail_closed_status": fail_closed,
                    "exact_match": exact_ok,
                    "ambiguous_or_prefix": amb,
                    "n_prefix_candidates": n_pref,
                    "resolved_snapshot_path": path,
                    "resolved_snapshot_id": snap_id,
                    "resolved_snapshot_sha16": sha,
                    "ep_traj_id": fp.get("traj_id", ""),
                    "ep_lineage_id": fp.get("fingerprint") or fp.get("lineage_id") or "",
                    "result_validity": "INVALID_AMBIGUOUS_IC" if invalid else "VALID_EXACT_IC",
                    **succ_cells,
                }
            )

    # --- ROOT_PROVENANCE_SPLITS ---
    overlap = set(panels["C_Feasible"]) & set(panels["C_Unfiltered"])
    prov_rows = []
    for skill in ("C_Feasible", "C_Unfiltered"):
        for tid in panels[skill]:
            ic = next(r for r in ic_rows if r["panel"] == skill and r["traj_id"] == tid)
            cat = ic["membership_canonical"]
            if ic["result_validity"] != "VALID_EXACT_IC":
                cat = f"{cat}|INVALID_IC"
            # per arm/seed succ
            row = {
                "panel": skill,
                "traj_id": tid,
                "canonical_physical_id": ic["canonical_physical_id"],
                "phys_root_truncated": ic["phys_root_truncated"],
                "provenance_category": cat,
                "in_both_C_panels": int(tid in overlap),
                "exposure_recorded": ic["exposure_recorded"],
                "result_validity": ic["result_validity"],
            }
            for arm in ("A", "B", "C", "D"):
                seeds = [-1] if arm == "A" else [1, 2, 3]
                for seed in seeds:
                    row[f"succ_{arm}_s{seed}"] = ic.get(f"succ_{arm}_s{seed}", "")
            # aggregate success counts by category helpers
            for arm in ("B", "C", "D"):
                vals = [int(ic[f"succ_{arm}_s{s}"]) for s in (1, 2, 3) if ic.get(f"succ_{arm}_s{s}") != ""]
                row[f"succ_sum_{arm}"] = sum(vals)
            prov_rows.append(row)

    # category success summary (printed)
    print("=== Truncation collisions (true distinct) ===")
    n_true = sum(int(r["true_distinct_physical_collision"]) for r in collision_rows)
    print("groups_with_multi_tid", len(collision_rows), "true_distinct", n_true)
    for r in collision_rows:
        if int(r["true_distinct_physical_collision"]):
            print(" TRUE", r)

    print("=== IC validity ===")
    for r in ic_rows:
        if r["panel"] == "C_Feasible" or r["traj_id"] not in panels["C_Feasible"]:
            print(
                r["panel"],
                r["traj_id"][:48],
                r["original_resolve_method"],
                r["fail_closed_status"],
                r["result_validity"],
                r["exposure_recorded"],
            )

    print("=== Successes ===")
    for arm in ("B", "C", "D"):
        for seed in (1, 2, 3):
            for skill in ("C_Feasible", "C_Unfiltered"):
                for r in paired:
                    if (
                        r["arm"] == arm
                        and int(r["train_seed"]) == seed
                        and r["skill"] == skill
                        and int(r["CAND_SUCCESS"]) == 1
                    ):
                        print(arm, seed, skill, r["traj_id"], r.get("exposure"), "P64", r["P64_SUCCESS"])

    # write CSVs
    def write_csv(path: Path, rows: list[dict]) -> None:
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
            w.writerows(rows)

    write_csv(OUT / "CANONICAL_ROOT_MAPPING.csv", mapping_rows)
    write_csv(OUT / "IC_RESOLUTION_AUDIT.csv", ic_rows)
    write_csv(OUT / "ROOT_PROVENANCE_SPLITS.csv", prov_rows)
    write_csv(OUT / "_TRUNCATION_COLLISIONS.csv", collision_rows)

    summary = {
        "n_mapping_rows": len(mapping_rows),
        "n_trunc_multi_groups": len(collision_rows),
        "n_true_distinct_trunc_collisions": n_true,
        "n_ic_rows": len(ic_rows),
        "n_invalid_ic": sum(1 for r in ic_rows if r["result_validity"] != "VALID_EXACT_IC"),
        "invalid_traj_ids": sorted(
            {r["traj_id"] for r in ic_rows if r["result_validity"] != "VALID_EXACT_IC"}
        ),
        "overlap_C_Feasible_Unfiltered": sorted(overlap),
        "n_overlap": len(overlap),
    }
    (OUT / "_AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("AUDIT_SUMMARY", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
