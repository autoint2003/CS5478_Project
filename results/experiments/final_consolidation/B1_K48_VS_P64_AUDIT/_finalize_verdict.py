"""Finalize CASE A verdict using tree80 + v2 paired raw rolls."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main() -> None:
    k = json.loads((OUT / "RAW_K48_ROLLS.json").read_text(encoding="utf-8"))
    p = json.loads((OUT / "RAW_P64_ROLLS.json").read_text(encoding="utf-8"))
    paired_old = list(csv.DictReader((OUT / "PAIRED_B1_RESULTS.csv").open(encoding="utf-8")))

    v2 = {r["traj_id"]: r for r in k if r.get("policy") == "K48_v2_u0312"}
    t80 = {r["traj_id"]: r for r in k if r.get("policy") == "K48_tree80_e3"}
    p64 = {r["traj_id"]: r for r in p}

    rows = []
    for old in paired_old:
        tid = old["traj_id"]
        a = v2[tid]
        b = t80[tid]
        c = p64[tid]
        rows.append(
            {
                "traj_id": tid,
                "root_id": old["root_id"],
                "frozen_P64_SUCCESS_RELAXED_12S": old["frozen_P64_SUCCESS_RELAXED_12S"],
                "frozen_P64_fail": old["frozen_P64_fail"],
                "posthoc_P64_SUCCESS_RELAXED_12S": c.get("SUCCESS_RELAXED_12S"),
                "posthoc_P64_SUCCESS_12S": c.get("SUCCESS_12S"),
                "posthoc_P64_SUCCESS_NEW": c.get("SUCCESS_NEW"),
                "posthoc_P64_fail": c.get("earliest_decisive_failure"),
                "posthoc_P64_frozen_match": c.get("frozen_match"),
                "K48_v2_SUCCESS_12S": a.get("SUCCESS_12S"),
                "K48_v2_reason": a.get("hist_reason"),
                "K48_v2_handoff_mode": a.get("handoff_mode"),
                "K48_v2_recovery_steps": a.get("recovery_steps"),
                "K48_v2_first_token": a.get("first_token"),
                "K48_tree80_SUCCESS_12S": b.get("SUCCESS_12S"),
                "K48_tree80_reason": b.get("hist_reason"),
                "K48_tree80_handoff_mode": b.get("handoff_mode"),
                "K48_tree80_recovery_steps": b.get("recovery_steps"),
                "K48_tree80_first_token": b.get("first_token"),
                "K48_tree80_e9": b.get("e_pos_le9"),
                "K48_v2_fc": a.get("first_contact"),
                "K48_tree80_fc": b.get("first_contact"),
                "P64_fc": c.get("first_contact"),
                "K48_v2_opp": a.get("opposing_ever"),
                "K48_tree80_opp": b.get("opposing_ever"),
                "P64_opp": c.get("opposing_ever"),
                "K48_v2_e9": a.get("e_pos_le9"),
                "P64_e9": c.get("e_pos_le9"),
                "P64_ep_len": c.get("ep_len"),
                "evidence_class": "post_hoc_diagnostic_not_independent_final_test",
            }
        )

    keys = list(rows[0].keys())
    with (OUT / "PAIRED_B1_RESULTS.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    utc = datetime.now(timezone.utc).isoformat()
    t80_s = sum(int(r["K48_tree80_SUCCESS_12S"] or 0) for r in rows)
    v2_s = sum(int(r["K48_v2_SUCCESS_12S"] or 0) for r in rows)
    p_s12 = sum(int(r["posthoc_P64_SUCCESS_12S"] or 0) for r in rows)
    p_rel = sum(int(r["posthoc_P64_SUCCESS_RELAXED_12S"] or 0) for r in rows)

    lines = [
        "# TRAJECTORY_DIVERGENCE",
        "",
        f"Generated: {utc}",
        "",
        "Scope: post-hoc paired closed-loop on the **same eight frozen B1 ICs**.",
        "Independent final-test estimate remains **P64 0/8** unchanged.",
        "",
        "## Aggregate",
        "",
        f"- K48 `tree80_e3` native SUCCESS_12S: **{t80_s}/8** (B-relevant historical GRPO lineage)",
        f"- K48 `v2_u0312` native SUCCESS_12S: **{v2_s}/8** (archive ball_train_v2 GRPO; premature HANDOFF)",
        f"- P64 post-hoc SUCCESS_12S: **{p_s12}/8**",
        f"- P64 post-hoc SUCCESS_RELAXED_12S: **{p_rel}/8** (frozen independent: **0/8**)",
        "- K48 SUCCESS_RELAXED_12S: **N/A** (R3F-path metric)",
        "",
        "## Checkpoint behavior contrast",
        "",
        "- **v2_u0312:** on all 8 ICs emits token `12` ×4 then HANDOFF=`47` (~0.08 s) → `LOST_GRASP`.",
        "- **tree80_e3:** ~36–37 motion steps (~0.72–0.74 s) with non-trivial token sequences ending in HANDOFF=`47` → **SUCCESS_12S on 8/8**.",
        "- **P64:** first-contact + opposing on all 8; stalls at `no_e_pos_le9` (4) or `no_r3_ready` (4).",
        "",
        "## Per-IC earliest meaningful divergence",
        "",
    ]
    for r in rows:
        tid = r["traj_id"]
        a, b, c = v2[tid], t80[tid], p64[tid]
        earliest = (
            "terminal_success_gap (tree80 SUCCESS_12S; P64 fails R3F/relaxed) — after shared FC/opp"
            if int(b.get("SUCCESS_12S") or 0) and not int(c.get("SUCCESS_RELAXED_12S") or 0)
            else "see contact table"
        )
        lines += [
            f"### `{tid}`",
            f"- Frozen P64 fail: `{r['frozen_P64_fail']}`",
            f"- v2_u0312: s12={a.get('SUCCESS_12S')} reason={a.get('hist_reason')} "
            f"steps={a.get('recovery_steps')} tokens={a.get('tokens_head')}",
            f"- tree80: s12={b.get('SUCCESS_12S')} reason={b.get('hist_reason')} "
            f"steps={b.get('recovery_steps')} e9={b.get('e_pos_le9')} tokens_head={b.get('tokens_head')}",
            f"- P64: relaxed={c.get('SUCCESS_RELAXED_12S')} 12S={c.get('SUCCESS_12S')} "
            f"fail={c.get('earliest_decisive_failure')} ep_len={c.get('ep_len')} "
            f"family_tail={c.get('family_tail')}",
            f"- Contact ever fc/bil/opp/pad4: tree80="
            f"{b.get('first_contact')}/{b.get('bilateral_ever')}/{b.get('opposing_ever')}/{b.get('min_pad_ge4')}; "
            f"P64={c.get('first_contact')}/{c.get('bilateral_ever')}/{c.get('opposing_ever')}/{c.get('min_pad_ge4')}",
            f"- **Earliest meaningful divergence:** {earliest}",
            "",
        ]
    lines += [
        "## Mechanism classes (diagnostic)",
        "",
        "1. **Policy skill regression (P64 vs tree80 K48)** — **supported** on these 8 ICs.",
        "2. **Task / IC distribution shift vs archive v2 30/56** — **also supported** (v2 does not transfer).",
        "3. **Action representation** — K48≠M111; cannot weight-copy tree80 into P64.",
        "4. **Inference / eval mismatch** — present; does not erase tree80 SUCCESS_12S.",
        "5. **Statistical uncertainty** — n=8 post-hoc; not a new independent final-test estimate.",
        "",
    ]
    (OUT / "TRAJECTORY_DIVERGENCE.md").write_text("\n".join(lines), encoding="utf-8")

    verdict = f"""# B1_REGRESSION_VERDICT

Generated: {utc}

## Verdict: **CASE A** — Historical K48 GRPO succeeds on B1 ICs where P64 fails

**Primary B-relevant checkpoint:** `results/grpo/tree80/checkpoint_epoch_0003.pt` (K48 adaptive GRPO tree80 epoch 3), the lineage used historically for Plan-B / Tree80 B expert generation.

On the exact eight frozen independent B1 MixedStarts, under the original K48 `roll_policy` wrapper (argmax + decode_action + HANDOFF=47 + SUCCESS_12S):

| Policy | Metric | Score |
|---|---|---|
| K48 tree80_e3 | SUCCESS_12S | **{t80_s}/8** |
| K48 v2_u0312 (archive 30/56 ckpt) | SUCCESS_12S | **{v2_s}/8** (premature HANDOFF → LOST_GRASP) |
| P64 (post-hoc replay) | SUCCESS_12S | **{p_s12}/8** |
| P64 (post-hoc / frozen) | SUCCESS_RELAXED_12S | **{p_rel}/8** / **0/8** |

This is evidence of **B1 policy regression of the frozen M111 P64 Actor relative to earlier K48 tree80 GRPO** on the current B1 IC set.

### Important secondary finding

Archive `v2_u0312` (**30/56** on `ball_train_v2` ids÷5) does **not** succeed on these Plan-B B1 ICs. That historical score is **not directly comparable** to current B1 (distribution shift + premature-HANDOFF transfer failure). Alone it would look like CASE B/C; tree80 paired results drive the overall verdict to **CASE A** for the B-class K48 GRPO lineage that mattered for Plan-B.

HANDOFF contracts still differ (K48 early token vs P64 R3F). SUCCESS_RELAXED_12S is N/A for K48. Paired scores are **post-hoc diagnostic**, not a replacement independent final-test estimate.

## Preserved independent final-test result

| Item | Value |
|---|---|
| Frozen P64 B1 | **0/8** SUCCESS_RELAXED_12S |
| Artifact | `results/experiments/final_independent_success_rate/EPISODE_RESULTS.csv` |
| Status | **Unchanged** |

## Phase 2 — is 0/8 an eval bug?

- Infra errors: none.
- Failures: `no_e_pos_le9` (4) / `no_r3_ready` (4); all have first_contact=1 and opposing_ever=1.
- Snapshot restore: **8/8** OK (`RESTORE_AUDIT.csv`).
- Post-hoc P64 replay matches frozen success bits: **8/8**.

**Conclusion: 0/8 is not an evaluation/restoration bug (not CASE D).**

## Phase 4 mechanism read

See `TRAJECTORY_DIVERGENCE.md`.

1. Policy skill regression (P64 vs tree80) — **primary**.
2. IC/distribution shift vs archive v2 30/56 — **also true** (explains v2_u0312 failure here).
3. Action representation K48≠M111 — structural continuity break.
4. Inference/eval mismatch — present; dual metrics reported.
5. Statistical uncertainty — n=8 diagnostic only.

## What this audit does **not** authorize

- Restoring K48 / tree80 as the final policy
- Retraining or checkpoint-selecting P64 from these eight ICs
- Replacing the independent 0/8 with post-hoc scores

## Artifacts

- `HISTORICAL_K48_RESULTS.md`
- `PROTOCOL_COMPATIBILITY.md`
- `PAIRED_B1_RESULTS.csv`
- `TRAJECTORY_DIVERGENCE.md`
- `B1_REGRESSION_VERDICT.md`
- `RESTORE_AUDIT.csv`
- `RAW_K48_ROLLS.json` / `RAW_P64_ROLLS.json`
"""
    (OUT / "B1_REGRESSION_VERDICT.md").write_text(verdict, encoding="utf-8")

    # Patch PROTOCOL with tree80 paired note
    proto = (OUT / "PROTOCOL_COMPATIBILITY.md").read_text(encoding="utf-8")
    if "tree80_e3 paired" not in proto:
        proto += f"""

## Phase 3 paired outcome note (post-run)

| Checkpoint | SUCCESS_12S on 8 B1 ICs |
|---|---|
| K48 tree80_e3 | **{t80_s}/8** |
| K48 v2_u0312 | **{v2_s}/8** |
| P64 SUCCESS_12S | **{p_s12}/8** |
| P64 SUCCESS_RELAXED_12S | **{p_rel}/8** |

Common physical restore worked. Action vocabularies remain incompatible (separate decoders used). See `B1_REGRESSION_VERDICT.md`.
"""
        (OUT / "PROTOCOL_COMPATIBILITY.md").write_text(proto, encoding="utf-8")

    hist = (OUT / "HISTORICAL_K48_RESULTS.md").read_text(encoding="utf-8")
    if "Paired B1 transfer" not in hist:
        hist += f"""

## Paired B1 transfer (post-hoc diagnostic)

On the eight frozen independent B1 ICs (not the historical 56):

| Checkpoint | SUCCESS_12S |
|---|---|
| tree80_e3 | **{t80_s}/8** |
| v2_u0312 | **{v2_s}/8** |

tree80 is the B-relevant historical GRPO for Plan-B comparison; v2_u0312's archive 30/56 does not transfer to this IC set.
"""
        (OUT / "HISTORICAL_K48_RESULTS.md").write_text(hist, encoding="utf-8")

    print(json.dumps({"case": "A", "tree80": t80_s, "v2": v2_s, "p64_rel": p_rel}, indent=2))


if __name__ == "__main__":
    main()
