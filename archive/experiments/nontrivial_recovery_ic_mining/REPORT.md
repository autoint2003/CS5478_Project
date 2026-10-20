# Nontrivial Recovery IC Mining — PILOT REPORT

**Stop after pilot.** No large-scale expansion. No SFT/GRPO. No physics/reward/env edits.

Artifacts: `SEARCH_COVERAGE.md`, `SHORT_HANDOFF_BASELINE.csv`, `ACTIVE_RECOVERY_SEARCH.csv`, `PREFIX_HANDOFF_AUDIT.csv`, `MECHANISM_COMPETITION.csv`, `IC_MANIFEST.csv`, `rediscovery_3647.json`, `raw/`, script `scripts/pilot_nontrivial_recovery_ic_mining.py`.

---

## Executive verdict

On a 22-IC pilot drawn from known `[36,36,47]` cases, prior Sliding/Recatch-favorable ICs, BOTH_VERIFIED, and hard-sliding oracles:

| Metric | Result |
|---|---|
| Short-handoff SUCCESS_12S | **22/22** (Level1) |
| Immediate HANDOFF (`[]`) | **0/22** |
| Level2 (short-fail + long oracle success) | **0** |
| Pure codebook `[36,36]` rediscovery | **1/7** known GRPO ICs |
| Dominant shortest prefixes | `[44]`, `[30]`, `[30,30]`, `[41]` (all grip-openish) |

**Pilot conclusion:** the current “recoverable / mechanism-competition” IC pool is dominated by **20–40 ms open-grip → HANDOFF** shortcuts. It does **not** yet supply credible Level2/Level3 nontrivial active-recovery evidence. Next mining must **filter out** short-handoff recoverable ICs before claiming Sliding vs Recatch competition.

---

## Search coverage (Q1)

See `SEARCH_COVERAGE.md`.

**VERIFIED (code):** `vertical_recatch_search` is a continuous open→chase→close→brake→lift→settle FSM; HANDOFF only after settle. `_try_sliding` uses schedule libraries. Neither enumerates K48 short open→HANDOFF prefixes, so `[36,36,47]` was outside their proposal class.

**Additional VERIFIED (this pilot):** even when we *do* enumerate short prefixes with **zero residual**, pure codebook `[36,36]` is **not** the reliable shortcut on the known GRPO-only ICs. Prior GRPO traces decode `action = codebook[token] + residual` (`decode_action`); residual-free `[36,36]` yields `DROP_AFTER_HANDOFF` on 6/7 known ICs, while **`[44]` (20 ms openish)** reaches `SUCCESS_12S` on all 7. So “GRPO found `[36,36,47]`” is **token-level** shorthand; continuous residual motion matters.

---

## Pilot protocol

- ICs: 22 (7 known_3647 + 4 slide_fav + 5–6 recatch_fav + both + hard_sliding_oracle)
- Short search per IC (~183 unique prefixes): immediate HANDOFF; all openish 1-step (+ full 1-step on known); `[36,36]`; openish self-pairs; 40 random 2-steps; open→close samples; budgeted 3/5-step open chains
- Full 47¹ / 47² deferred (cost)
- Success = restore snapshot → `step_legal` only → original HANDOFF → full nominal → `SUCCESS_12S`
- Failures logged separately (`DROP_AFTER_HANDOFF`, `LOST_GRASP`, `TABLE_DROP`, …)
- Active oracles (`_try_sliding` + `vertical_specs`) on short-fail ICs (none) and a Level1 competition subset
- Wall time ≈ **45 min** (~122 s wall / IC short search)

---

## Short-Handoff Baseline (Q2)

| | |
|---|---|
| ICs with ≥1 short SUCCESS_12S | **22/22** |
| Level1 | **22** |
| Depth of shortest success | 1-step: **19**; 2-step: **3**; ≥3: **0** |
| Immediate HANDOFF success | **0/22** (all `LOST_GRASP` or `DROP_AFTER_HANDOFF`) |
| Bucket successes (rollout counts) | 20 ms: 49; 40 ms: 263; 60–100 ms: 2; 100–200 ms: 0 |
| All depth-1 successes | grip-openish (`a6 < -0.5`) |

Shortest-success token modes: `[44]`×9, `[30]`×6, `[30,30]`×3, `[41]`×2, `[3]`×1, `[32]`×1.

**Label: `SUPPORTED_BUT_INCOMPLETE`** — pilot pool only; not exhaustive 47²; not the full hard benchmark.

---

## Active recovery / Level2 (Q3)

- Level2: **0**
- Unresolved (no short, no oracle): **0**
- Every pilot IC already had a short SUCCESS_12S, so longer oracles were never the *only* evidence.

**Label: `SUPPORTED_BUT_INCOMPLETE` for “no Level2 in this pool”; `UNRESOLVED` for “nontrivial active recovery does not exist physically.”**

---

## Prefix HANDOFF audit (Q4)

From `PREFIX_HANDOFF_AUDIT.csv` on each IC’s shortest short success:

- `prefix_len=0` always fails on these ICs
- For 19/22, `prefix_len=1` already `SUCCESS_12S` → the single openish step is **necessary** vs immediate HANDOFF; longer oracle motions are **not required** for success
- For 3/22 (`hrv1_920039/067/070`), 1×`[30]` fails but 2×`[30]` succeeds → **40 ms** open needed; still not long active recovery

**Label: `VERIFIED` for these 22 shortest paths; `SUPPORTED_BUT_INCOMPLETE` for necessity of longer Sliding/Recatch oracles (they are redundant whenever short succeeds).**

---

## Mechanism competition under short filter (Q5–Q7)

Active competition pass (Level1 subset only) tags: SLIDING_FAVORABLE=4, RECATCH_FAVORABLE=1, BOTH_VERIFIED=7.

**These tags are not Level3 evidence** when Level1 already holds: both mechanisms (or neither) can be “favorable” among long oracles while a 20 ms open→HANDOFF already solves the IC.

| Question | Pilot answer | Confidence |
|---|---|---|
| Sliding-favorable nontrivial? | **No credible Level3 IC in pilot** (all short-recoverable) | `UNRESOLVED` (need short-fail pool) |
| Recatch-favorable nontrivial? | same | `UNRESOLVED` |
| Budget asymmetry? | **Yes** — short K48 prefixes vs 3 vertical FSM specs vs sliding schedules | `VERIFIED` |

---

## 39D distinguishability (Q8)

Not re-probed this pilot. Prior `mechanism_selection_feasibility/OBSERVATION_AUDIT.md` remains the reference.

**Caveat from this pilot:** if most “competition” ICs are short-handoff shortcuts, mechanism-selection from 39D history is a **different** question than distinguishing long Sliding vs Recatch on Level2 ICs.

**Label: `SUPPORTED_BUT_INCOMPLETE` (deferred).**

---

## Next scale recommendation (Q9)

**Do not expand the same recoverability-screened pool.** Priority:

1. **Short-Handoff Baseline as a hard filter** on a larger candidate set (hard mining leftovers, failed SFT/GRPO ICs, new contact-slip generators). Keep only ICs with **no** SUCCESS_12S under a documented short budget (recommend: exhaustive openish 1-step + full 47 1-step + exhaustive openish×openish 2-step + `[36,36]` + residual-sensitive probes).
2. On short-fail survivors, run **shared-budget** active search (beam/random K48 + sliding schedules + vertical FSM as proposals only).
3. Prefix-HANDOFF audit before claiming Level2.
4. Holdout: family-disjoint; never promote holdout successes into train.

**Budget estimate (from pilot):** ~180 short rollouts / IC ≈ **~120 s wall / IC** on this machine for the budgeted (non-exhaustive) short search. Exhaustive 47¹ ≈ +24 rollouts; full 47² ≈ +2209 rollouts (~10–20× cost). Target **50–100 short-fail candidates** screened from ≥200–400 raw ICs; expected short-pass rate unknown — if similar to this pool (~100%), need much harder generation.

**Scale-up gate:** find ≥5 Level2 ICs with prefix audit showing success only after ≥100 ms of recovery control, then decide on Level3 competition.

---

## Answers checklist (required)

1. **Why old Recatch missed `[36,36,47]`?** `VERIFIED` — FSM proposal class; see `SEARCH_COVERAGE.md`. Residual-free codebook `[36,36]` also often fails even when GRPO tokens match.
2. **How many short-handoff recoverable?** **22/22** pilot — `SUPPORTED_BUT_INCOMPLETE`.
3. **True longer active recovery?** **0 Level2 in pilot** — `SUPPORTED_BUT_INCOMPLETE` / existence `UNRESOLVED`.
4. **Necessary vs redundant?** Shortest path usually **1 openish step**; immediate HANDOFF fails; long oracles redundant on these ICs — `VERIFIED` (pilot set).
5. **Sliding-favorable nontrivial?** `UNRESOLVED` (no Level2).
6. **Recatch-favorable nontrivial?** `UNRESOLVED` (no Level2).
7. **Budget asymmetry?** `VERIFIED`.
8. **39D enough for mechanism choice?** `SUPPORTED_BUT_INCOMPLETE` (prior report; pool contaminated by short shortcuts).
9. **Next expand?** Short-fail filter first; then shared active search; ~120 s/IC short screen + higher for active — see above.

---

## Stop

Pilot complete. Awaiting decision before large-scale mining or any training.
