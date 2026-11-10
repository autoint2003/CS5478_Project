# Workflow Audit Addendum (read-only)

Date: 2026-11-06. No code behavior changes; no re-runs; no P85.

Sources: primary consolidation audit + [frozen-path audit](3f39613c-53f3-4b2c-a38f-d31de7709092) + [P70–P84 pipeline audit](99412420-5e66-4455-8035-c96eee06e195) + [P51–P64 audit](7b0a85d3-c7e4-415a-ac15-0904852613c4) + [P65–P84 audit](26d23fa2-8861-4e5a-b8c0-7f768daf00b9). See also `ABCD_EXPERIMENT_CODE_AUDIT.md`.

## Verdict

**No CRITICAL code-path bugs that auto-train or evaluate final-test.**  
**One HIGH scientific/protocol finding:** on the P82/P83 diagnostic panel, **P70 V2 almost never acts**, so reported MSP ≈ 0.273 is behaviorally **≈ P64 Actor-only**, not a measured P64+V2 closed-loop rate on that panel.

## HIGH — must correct prior wording

### V2 inactive on P83/P82 panel

Verified on `results/experiments/p83_recoverable_failure_mass/cache/panel_32.pkl`:

| Metric | Value |
|---|---:|
| Episodes | 128 |
| Episodes with `forced_steps > 0` | **2** (both B1; no first contact; both fail) |
| A1/C/D with any forced step | **0 / 96** |
| Total forced steps (sum) | 18 |

**Mechanism:** V2 requires `require_no_contact` eligibility (`_elig` in `p71_contact_safe_alignment/controllers.py`). Panel starts typically already report pad contact in observation, so V2 never activates; control is Actor tokens.

**Implications:**

1. Do **not** attribute P83 MSP (or P82–P84 failure budgets) to V2 closed-loop control.
2. P70’s V2 lift (SUCCESS/R3 ≈ 0.240 vs V0 ≈ 0.083) was measured on a **different IC distribution** (C-focused P69 split with activation rate 1.0) and **does not transfer as evidence** onto the A1/B1/C/D P83 panel under this start protocol.
3. Frozen *code path* still wires V2 (`rule=None` → V2 then Actor); frozen *measured panel behavior* is Actor-dominated.
4. Report frozen system as: **P64 Actor (+ P70 V2 pre-contact gate; essentially never triggered on P83-like contact-at-start ICs).**

Unresolved (reported, not repaired): why P70 validation ICs activate V2 while P83 C ICs do not — likely start-contact / IC-pool difference; no new eval launched.

## Must disclose (other)

1. **P83 panel ≠ train-independent.** ~120/128 panel traj/roots in `IC_EXPOSURE.jsonl`. MSP close to P64 dev (≈0.2715).
2. Label P82–P84 “baseline” carefully: intended stack is P64+V2, but **panel execution ≈ Actor-only** (above).
3. **P83 Case C mass comparison was asymmetric** (C contact-formation vs non-C any-failure SUM). **Revised 2026-11-07 to Case D** (`stop_consolidate_baseline`) using fair mean; see `POST_AUDIT_DECISION_REVISION.md`.
4. P61/P62 final-test ICs not in P54 schedule; not used for P64 step-4 selection. (P62 historically evaluated an ancestor ckpt on final — not the frozen step-4.)

## Trust table (updated)

| Claim family | Status |
|---|---|
| No auto-train; overlays off (`rule=None`); hashes match | Trustworthy |
| V2 uses obs-only ready_score (no `e_pos`) | Trustworthy |
| P83 MSP as “P64+V2 system performance” | **Incorrect attribution** — treat as ≈ Actor-only on this panel |
| P70 V2 lift on P70 held panel | Separate experiment; do not paste onto P83 |
| Stop at P84 / reject overlays & fixed-token | Still supported (negative full-episode interventions) |
| P79/P81 matched-CF static-bias story | Do not overclaim |
| P83 “non-C dominates” | Opportunity framing only |

## Latent hygiene — code fixes applied

- Post-contact V2 re-eligibility closed: after `saw_contact`, baseline/`rule=None` returns Actor (no `select_v2_token`).
- `blocked_eval` uses `is_blocked_eval` (traj + `root_id`) in P75/P82/P83/P69.
- P58/P59/P60/P82 seeds → `stable_u32`; P53 PID lock Windows-safe; P76 `P73_CF` → `p73_first_contact_bilateral`.
- P82/P83 write `v2_activation` meta and warn if activation_rate < 0.05.
- Remaining (unchanged): handoff/oracle can mutate sim before some ever-flags; failure taxonomy still lacks explicit drop bucket.
- Historical panel numbers unchanged until deliberate re-collect.

## Explicitly verified OK

- Canonical collect uses `select_token(..., rule=None)`.
- V2 **spec** matches `BASELINE_FREEZE.json` (wiring present even when inactive).
- Decision modules do not auto-launch training.
- Checkpoint MD5s; success key `SUCCESS_RELAXED_12S` → `SUCCESS_RELAXED`.
- `generate_figures` `parents[3]` → repo root; figs 04–06 match panel counts.
