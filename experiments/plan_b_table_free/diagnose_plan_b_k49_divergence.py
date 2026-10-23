#!/usr/bin/env python
"""Phase-1: continuous vs K49 divergence + hybrid phase localization for Plan-B nontrivial.

Writes results/data_generation/stage2_abc_mining/v2/b_k49_diagnostics/
Does not modify Oracle, K49, physics, or success criteria.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.precision import state_machine as upo  # noqa: E402
from data_generation.oracles.precision.corrections.contact_primitives import _wrist, audit_grasp  # noqa: E402
from data_generation.oracles.precision.state_machine import (  # noqa: E402
    PrecisionConfig,
    measure_errors,
    run_precision,
)
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from training.action_vocab_k49 import CAP, HANDOFF, K49_TABLE_ROWS, build_k49_table  # noqa: E402

V1 = ROOT / "results/data_generation/stage2_abc_mining/v1"
OUT = ROOT / "results/data_generation/stage2_abc_mining/v2/b_k49_diagnostics"
K49_CONFIG = "K49_tok48_wy0p25"

# Deduped genuine Plan-B nontrivial continuous-OK / K49-fail cases
CASES = [
    "B__hard_910166__post_b1__7b45f78c0ce7796e",
    "B__hard_910220__0412621ee978627d",
]


def jsafe(o):
    if o is None or isinstance(o, (bool, str, int)):
        return o
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsafe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsafe(x) for x in o]
    return str(o)


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    return env


def nearest(table, a):
    a = np.asarray(a, np.float32).reshape(7)
    bi, bd = 0, 1e9
    for i in range(K49_TABLE_ROWS):
        if i == HANDOFF:
            continue
        d = float(np.linalg.norm(table[i] - a))
        if d < bd:
            bi, bd = i, d
    return bi, table[bi].copy(), bd


def decode_k49(tokens, residuals, table):
    out = []
    for i, t in enumerate(tokens):
        a = table[int(t)].copy()
        if residuals is not None and i < len(residuals):
            a = np.clip(a + np.clip(residuals[i], -CAP, CAP), -1, 1).astype(np.float32)
        out.append(a)
    return np.asarray(out, np.float32)


def snap_phys(sim):
    err = measure_errors(sim)
    geom = audit_grasp(sim)
    e = np.asarray(err["e_signed_mm"], float)
    return {
        "e_signed_mm": e.tolist(),
        "e_pos_mm": float(err["e_pos_mm"]),
        "dtheta_deg": float(err["dtheta_deg"]),
        "bilateral": bool(err["bilateral"]),
        "v_rel": float(err["v_rel_norm"]),
        "w_rel": float(err["w_rel_norm"]),
        "fn_L": float(err.get("fn_L") or 0.0),
        "fn_R": float(err.get("fn_R") or 0.0),
        "obj_z": float(err.get("obj_z") or 0.0),
        "grasp_kind": geom.get("grasp_kind"),
    }


def replay_tape(env, cap, actions, ref, *, record_every: int = 1, label: str = ""):
    restore_capture(env, cap)
    series = []
    acts = np.asarray(actions, np.float32)
    for i, a in enumerate(acts):
        step_legal(env.sim, np.asarray(a, np.float32))
        if i % record_every == 0 or i + 1 == len(acts):
            row = snap_phys(env.sim)
            row["i"] = i
            row["a"] = np.asarray(a, float).tolist()
            series.append(row)
    for _ in range(8):
        step_legal(env.sim, _wrist(0.0, 0.0, 1.0))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "label": label,
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
        "n": len(acts),
        "series": series,
        "final": snap_phys(env.sim),
    }


def first_divergence(cont_series, k49_series, *, e_pos_tol=0.5, bil_flip=True, fn_tol=1.5):
    n = min(len(cont_series), len(k49_series))
    for i in range(n):
        c, k = cont_series[i], k49_series[i]
        reasons = []
        if abs(c["e_pos_mm"] - k["e_pos_mm"]) > e_pos_tol:
            reasons.append(f"e_pos_delta={k['e_pos_mm']-c['e_pos_mm']:.2f}")
        if abs(c["dtheta_deg"] - k["dtheta_deg"]) > 1.0:
            reasons.append(f"dtheta_delta={k['dtheta_deg']-c['dtheta_deg']:.2f}")
        if bil_flip and bool(c["bilateral"]) != bool(k["bilateral"]):
            reasons.append(f"bilateral c={c['bilateral']} k={k['bilateral']}")
        if abs(c["fn_L"] - k["fn_L"]) > fn_tol or abs(c["fn_R"] - k["fn_R"]) > fn_tol:
            reasons.append(
                f"fn_delta L={k['fn_L']-c['fn_L']:.2f} R={k['fn_R']-c['fn_R']:.2f}"
            )
        ec = np.asarray(c["e_signed_mm"], float)
        ek = np.asarray(k["e_signed_mm"], float)
        if float(np.linalg.norm(ec - ek)) > 1.0:
            reasons.append(f"e_signed_l2={float(np.linalg.norm(ec-ek)):.2f}")
        if reasons:
            return {
                "step": int(c["i"]),
                "reasons": reasons,
                "cont": c,
                "k49": k,
                "action_l2": float(np.linalg.norm(np.asarray(c["a"]) - np.asarray(k["a"]))),
                "action_linf": float(np.max(np.abs(np.asarray(c["a"]) - np.asarray(k["a"])))),
            }
    return None


def instrumented_phase_spans(env, cap, ref, plan="B"):
    """Re-run Oracle with monkeypatched primitives to record accepted action counts per phase."""
    spans = []  # (phase_name, n_actions, ok)

    def wrap(fn, name):
        def inner(*args, **kwargs):
            r = fn(*args, **kwargs)
            n = len(r.get("actions") or [])
            # Only count if actions would be extended (approx: always log)
            spans.append(
                {
                    "phase": name,
                    "n_actions": n,
                    "ok": bool(r.get("ok")),
                    "skipped": bool(r.get("skipped")),
                    "rejected_branch": bool(r.get("rejected_branch")),
                    "stop_reason": r.get("stop_reason") or (r.get("slip") or {}).get("stop_reason"),
                }
            )
            return r

        return inner

    # Patch symbols used inside run_precision
    orig = {
        "correct_dx_horizontal_attempt": upo.correct_dx_horizontal_attempt,
        "correct_dx_verified": upo.correct_dx_verified,
        "correct_dz_inward": upo.correct_dz_inward,
        "correct_dz_outward": upo.correct_dz_outward,
        "correct_dtheta": upo.correct_dtheta,
    }
    upo.correct_dx_horizontal_attempt = wrap(orig["correct_dx_horizontal_attempt"], "dx_horizontal_attempt")
    upo.correct_dx_verified = wrap(orig["correct_dx_verified"], "dx_verified_lateral")
    upo.correct_dz_inward = wrap(orig["correct_dz_inward"], "correct_dz_inward")
    upo.correct_dz_outward = wrap(orig["correct_dz_outward"], "correct_dz_outward")
    upo.correct_dtheta = wrap(orig["correct_dtheta"], "correct_dtheta")
    try:
        restore_capture(env, cap)
        res = run_precision(env, {"capture": cap, "ref": ref}, plan, PrecisionConfig())
    finally:
        for k, v in orig.items():
            setattr(upo, k, v)
    # Build cumulative spans for accepted tape (horizontal may be rejected — match by mechanism)
    mech = (res.primitive_logs or {}).get("dx_mechanism")
    accepted = []
    cursor = 0
    for s in spans:
        # Horizontal actions only on tape if mechanism says so
        if s["phase"] == "dx_horizontal_attempt" and mech != "horizontal_dx":
            accepted.append({**s, "on_tape": False, "tape_start": None, "tape_end": None})
            continue
        if s["n_actions"] <= 0:
            accepted.append({**s, "on_tape": False, "tape_start": None, "tape_end": None})
            continue
        accepted.append(
            {
                **s,
                "on_tape": True,
                "tape_start": cursor,
                "tape_end": cursor + s["n_actions"],
            }
        )
        cursor += s["n_actions"]
    return {
        "oracle_n_actions": len(res.actions or []),
        "oracle_SUCCESS_NEW": bool((res.handoff or {}).get("SUCCESS_NEW")),
        "phases": list(res.phases or []),
        "dx_mechanism": mech,
        "primitive_spans": accepted,
        "span_total": cursor,
    }


def hybrid_replay(env, cap, cont_acts, k49_acts, ref, spans, *, continuous_phases: set[str]):
    """Offline diagnostic: continuous for named phases, K49 elsewhere."""
    n = len(cont_acts)
    blend = np.asarray(k49_acts[:n], np.float32).copy()
    used = []
    for s in spans:
        if not s.get("on_tape"):
            continue
        a, b = s["tape_start"], s["tape_end"]
        if a is None or b is None:
            continue
        a, b = int(a), int(min(b, n))
        if s["phase"] in continuous_phases:
            blend[a:b] = cont_acts[a:b]
            used.append({"phase": s["phase"], "mode": "continuous", "a": a, "b": b})
        else:
            used.append({"phase": s["phase"], "mode": "k49", "a": a, "b": b})
    # If span_total mismatches saved tape, fall back to all-k49 with note
    note = None
    if abs(spans and max((s.get("tape_end") or 0) for s in spans) - n) > 5:
        note = f"span_end_mismatch_vs_saved_tape n={n}"
    rep = replay_tape(env, cap, blend, ref, record_every=max(1, n // 40), label="hybrid")
    return {
        "continuous_phases": sorted(continuous_phases),
        "segments": used,
        "SUCCESS_NEW": rep["SUCCESS_NEW"],
        "SUCCESS_12S": rep["SUCCESS_12S"],
        "reason": rep["reason"],
        "note": note,
        "final": rep["final"],
    }


def classify_failure(div, hybrids, enc_stats, cont_ok, k49_ok) -> str:
    if not cont_ok:
        return "continuous_teacher_not_reproducible"
    if k49_ok:
        return "k49_now_succeeds_recheck"
    # Encoding path
    if enc_stats["max_recon_l2"] < 1e-6 and enc_stats["residual_clipped_frac"] < 1e-9:
        # Perfect reconstruction but fail → controller/state issue
        return "numerical_or_controller_state_mismatch"
    # Hybrid localization
    if hybrids:
        # Find which single continuous phase restores success
        rescue = [h for h in hybrids if h.get("SUCCESS_NEW") and len(h.get("continuous_phases") or []) == 1]
        if rescue:
            return f"quantization_induced_contact_failure_in_phase:{rescue[0]['continuous_phases'][0]}"
        # All-continuous hybrid of all phases should succeed
        all_c = [h for h in hybrids if set(h.get("continuous_phases") or []) == {"ALL"}]
        if all_c and all_c[0].get("SUCCESS_NEW") and not k49_ok:
            if enc_stats["residual_clipped_frac"] > 0.2 or enc_stats["max_recon_l2"] > 0.15:
                return "genuine_limitation_or_heavy_quantization_of_frozen_k49"
            return "temporal_or_controller_state_mismatch"
    if enc_stats["residual_clipped_frac"] > 0.2 or enc_stats["max_recon_l2"] > 0.35:
        return "genuine_limitation_of_frozen_k49_action_set"
    if div and div.get("action_linf", 0) > CAP + 1e-6:
        return "incorrect_encoding_or_decoding"  # residual should cap
    if div and div.get("step", 0) < 5:
        return "incorrect_encoding_or_decoding_or_restore"
    return "unresolved"


def diagnose_one(env, sample_id: str, table) -> dict:
    outcome = json.loads((V1 / "outcomes" / f"{sample_id}.json").read_text(encoding="utf-8"))
    ic = pickle.loads((V1 / outcome["ic_pkl"]).read_bytes())
    cont = np.load(V1 / outcome["continuous_demo"]["path"])["actions"].astype(np.float32)
    tok = np.load(V1 / outcome["k49_demo"]["path"])
    tokens = tok["tokens"]
    residuals = tok["residuals"] if "residuals" in tok.files else None
    k49_acts = decode_k49(tokens, residuals, table)

    # Encoding stats
    recon_l2 = np.linalg.norm(k49_acts - cont[: len(k49_acts)], axis=1) if len(k49_acts) else np.zeros(0)
    raw_l2 = []
    clipped = 0
    for i, a in enumerate(cont):
        ti, proto, d = nearest(table, a)
        raw = a - proto
        res = np.clip(raw, -CAP, CAP)
        if float(np.max(np.abs(raw - res))) > 1e-8:
            clipped += 1
        raw_l2.append(d)
    enc_stats = {
        "n": int(len(cont)),
        "max_recon_l2": float(recon_l2.max()) if len(recon_l2) else 0.0,
        "mean_recon_l2": float(recon_l2.mean()) if len(recon_l2) else 0.0,
        "max_proto_l2": float(max(raw_l2) if raw_l2 else 0.0),
        "residual_clipped_frac": float(clipped / max(1, len(cont))),
        "cap": CAP,
        "dim_max_abs_err": float(np.max(np.abs(k49_acts - cont[: len(k49_acts)]))) if len(k49_acts) else 0.0,
        "dim_argmax": int(np.argmax(np.max(np.abs(k49_acts - cont[: len(k49_acts)]), axis=0))) if len(k49_acts) else -1,
    }

    rec_every = max(1, len(cont) // 80)
    print(f"=== {sample_id}: continuous replay ===", flush=True)
    cont_rep = replay_tape(env, ic["capture"], cont, ic["ref"], record_every=rec_every, label="continuous")
    print(f"  NEW={cont_rep['SUCCESS_NEW']} 12S={cont_rep['SUCCESS_12S']}", flush=True)

    print(f"=== {sample_id}: K49 replay ===", flush=True)
    k49_rep = replay_tape(env, ic["capture"], k49_acts, ic["ref"], record_every=rec_every, label="k49")
    print(f"  NEW={k49_rep['SUCCESS_NEW']} 12S={k49_rep['SUCCESS_12S']} reason={k49_rep['reason']}", flush=True)

    div = first_divergence(cont_rep["series"], k49_rep["series"])
    if div:
        print(f"  first_div step={div['step']} reasons={div['reasons']} a_linf={div['action_linf']:.3f}", flush=True)

    print(f"=== {sample_id}: instrumented Oracle for phase spans ===", flush=True)
    span_info = instrumented_phase_spans(env, ic["capture"], ic["ref"], plan="B")
    print(
        f"  oracle_n={span_info['oracle_n_actions']} saved_n={len(cont)} mech={span_info['dx_mechanism']} "
        f"SUCCESS_NEW={span_info['oracle_SUCCESS_NEW']}",
        flush=True,
    )

    # Align spans to saved tape length if oracle n matches
    hybrids = []
    phases_on_tape = [s["phase"] for s in span_info["primitive_spans"] if s.get("on_tape")]
    # Scale spans if lengths differ slightly
    scale = len(cont) / max(1, span_info["span_total"]) if span_info["span_total"] else 1.0
    scaled_spans = []
    for s in span_info["primitive_spans"]:
        s2 = dict(s)
        if s.get("on_tape") and s.get("tape_start") is not None:
            s2["tape_start"] = int(round(s["tape_start"] * scale))
            s2["tape_end"] = int(round(s["tape_end"] * scale))
        scaled_spans.append(s2)

    # Hybrid: each phase continuous alone
    for ph in phases_on_tape:
        print(f"  hybrid continuous-only phase={ph}", flush=True)
        h = hybrid_replay(env, ic["capture"], cont, k49_acts, ic["ref"], scaled_spans, continuous_phases={ph})
        hybrids.append(h)
        print(f"    NEW={h['SUCCESS_NEW']} reason={h['reason']}", flush=True)

    # Hybrid: all correction phases continuous
    if phases_on_tape:
        h_all = hybrid_replay(
            env, ic["capture"], cont, k49_acts, ic["ref"], scaled_spans, continuous_phases=set(phases_on_tape)
        )
        h_all["continuous_phases"] = ["ALL"]
        hybrids.append(h_all)
        print(f"  hybrid ALL continuous phases NEW={h_all['SUCCESS_NEW']}", flush=True)

    # Prefix continuous / suffix K49 at first divergence
    if div:
        cut = int(div["step"]) + 1
        blend = np.concatenate([cont[:cut], k49_acts[cut: len(cont)]], axis=0)
        pref = replay_tape(env, ic["capture"], blend, ic["ref"], record_every=max(1, len(cont) // 40), label="prefix_cont")
        hybrids.append(
            {
                "continuous_phases": [f"prefix_0_{cut}"],
                "SUCCESS_NEW": pref["SUCCESS_NEW"],
                "SUCCESS_12S": pref["SUCCESS_12S"],
                "reason": pref["reason"],
                "note": "prefix continuous through first divergence step",
            }
        )
        print(f"  hybrid prefix_cont[:{cut}] NEW={pref['SUCCESS_NEW']}", flush=True)

    verdict = classify_failure(div, hybrids, enc_stats, cont_rep["SUCCESS_NEW"], k49_rep["SUCCESS_NEW"])

    # Per-dim residual saturation histogram
    sat = np.zeros(7, np.int64)
    for a in cont:
        _, proto, _ = nearest(table, a)
        raw = np.abs(a - proto)
        sat += (raw > CAP + 1e-8).astype(np.int64)

    report = {
        "sample_id": sample_id,
        "phases_outcome": (outcome.get("precision") or {}).get("phases"),
        "encoding": enc_stats,
        "residual_saturation_counts_by_dim": sat.tolist(),
        "dim_names": ["vx", "vy", "vz", "wx", "wy", "wz", "grip"],
        "continuous_replay": {k: cont_rep[k] for k in ("SUCCESS_NEW", "SUCCESS_12S", "reason", "n", "final")},
        "k49_replay": {k: k49_rep[k] for k in ("SUCCESS_NEW", "SUCCESS_12S", "reason", "n", "final")},
        "first_divergence": div,
        "phase_spans": span_info,
        "hybrids": [
            {**{k: v for k, v in h.items() if k != "final"}, "final": h.get("final")} for h in hybrids
        ],
        "verdict": verdict,
        "notes": [
            "Hybrid trajectories are offline diagnostics only — not verified K49 SFT demos.",
            "Oracle not modified; K49 codebook not modified.",
        ],
    }
    # Trim series from hybrids already trimmed
    out_path = OUT / f"{sample_id}__divergence.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(jsafe(report), indent=2), encoding="utf-8")

    # Compact markdown
    md = [
        f"# K49 Divergence — `{sample_id}`",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        "## Encoding",
        f"- n_actions: {enc_stats['n']}",
        f"- max_recon_l2: {enc_stats['max_recon_l2']:.4f}",
        f"- residual_clipped_frac: {enc_stats['residual_clipped_frac']:.4f}",
        f"- dim_argmax abs err: {enc_stats['dim_argmax']} ({report['dim_names'][enc_stats['dim_argmax']] if enc_stats['dim_argmax']>=0 else 'n/a'})",
        f"- saturation counts by dim: {sat.tolist()}",
        "",
        "## Replays",
        f"- Continuous SUCCESS_NEW: {cont_rep['SUCCESS_NEW']} ({cont_rep['reason']})",
        f"- K49 SUCCESS_NEW: {k49_rep['SUCCESS_NEW']} ({k49_rep['reason']})",
        "",
        "## First divergence",
        f"```json\n{json.dumps(jsafe(div), indent=2)}\n```" if div else "_none within sampled series_",
        "",
        "## Phase spans (fresh instrumented Oracle)",
        f"- dx_mechanism: `{span_info['dx_mechanism']}`",
        f"- oracle_n / saved_n: {span_info['oracle_n_actions']} / {len(cont)}",
        "",
    ]
    for s in scaled_spans:
        md.append(
            f"- `{s['phase']}` on_tape={s.get('on_tape')} n={s['n_actions']} "
            f"range=[{s.get('tape_start')},{s.get('tape_end')}] ok={s.get('ok')}"
        )
    md += ["", "## Hybrids (diagnostic only)", ""]
    for h in hybrids:
        md.append(
            f"- continuous_phases={h.get('continuous_phases')}: NEW={h.get('SUCCESS_NEW')} "
            f"12S={h.get('SUCCESS_12S')} reason={h.get('reason')}"
        )
    (OUT / f"{sample_id}__REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} verdict={verdict}", flush=True)
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    env = make_env()
    table = build_k49_table(config_id=K49_CONFIG)
    reports = []
    for sid in CASES:
        reports.append(diagnose_one(env, sid, table))
    summary = {
        "cases": [
            {
                "sample_id": r["sample_id"],
                "verdict": r["verdict"],
                "cont_NEW": r["continuous_replay"]["SUCCESS_NEW"],
                "k49_NEW": r["k49_replay"]["SUCCESS_NEW"],
                "first_div_step": (r.get("first_divergence") or {}).get("step"),
                "max_recon_l2": r["encoding"]["max_recon_l2"],
                "clipped_frac": r["encoding"]["residual_clipped_frac"],
                "dx_mechanism": (r.get("phase_spans") or {}).get("dx_mechanism"),
                "hybrid_rescues": [
                    h.get("continuous_phases")
                    for h in r.get("hybrids") or []
                    if h.get("SUCCESS_NEW") and h.get("continuous_phases") not in (["ALL"],)
                ],
            }
            for r in reports
        ]
    }
    (OUT / "SUMMARY.json").write_text(json.dumps(jsafe(summary), indent=2), encoding="utf-8")
    lines = ["# Plan B K49 Divergence Summary", ""]
    for c in summary["cases"]:
        lines.append(f"## `{c['sample_id']}`")
        lines.append(f"- Verdict: `{c['verdict']}`")
        lines.append(f"- Cont NEW / K49 NEW: {c['cont_NEW']} / {c['k49_NEW']}")
        lines.append(f"- First div step: {c['first_div_step']}")
        lines.append(f"- max_recon_l2={c['max_recon_l2']:.3f} clipped_frac={c['clipped_frac']:.3f}")
        lines.append(f"- dx_mechanism: {c['dx_mechanism']}")
        lines.append(f"- Hybrid single-phase rescues: {c['hybrid_rescues']}")
        lines.append("")
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(jsafe(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
