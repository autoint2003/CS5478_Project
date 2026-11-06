"""Phase 2 — Gate failure attribution for sustained-bilateral / stable but not R3-ready."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.precision.p67_r3f_gate_diagnosis.paths import OUT, TRACE_DIR


def _load_timelines() -> list[dict]:
    path = OUT / "GATE_PREDICATE_TIMELINES.json"
    return json.loads(path.read_text(encoding="utf-8"))["episodes"]


def _load_full_trace(traj_id: str, seed_hint: int | None = None) -> dict | None:
    for p in TRACE_DIR.glob("*.json"):
        if traj_id.replace("/", "_")[:40] in p.name:
            return json.loads(p.read_text(encoding="utf-8"))
    # any matching prefix
    for p in TRACE_DIR.glob("*.json"):
        blob = json.loads(p.read_text(encoding="utf-8"))
        if blob.get("summary", {}).get("traj_id") == traj_id:
            return blob
    return None


def classify_episode(ep: dict, trace: dict | None) -> dict[str, Any]:
    """Multi-label categories A–E (not exclusive)."""
    st = ep["stages"]
    cats = []
    detail: dict[str, Any] = {}

    if not st.get("sustained_bilateral_5") and not st.get("stable_hold"):
        return {"categories": ["NOT_SUSTAINED"], "detail": {}}

    if st.get("r3_ready") or st.get("handoff"):
        return {"categories": ["REACHED_R3"], "detail": {}}

    # Analyze from near_breaks and max_ok_s
    breaks = ep.get("near_breaks") or []
    if trace:
        breaks = trace.get("near_breaks") or breaks
        tl = trace.get("timeline") or []
    else:
        tl = []

    # Predicate failure counts on frames where bilateral but not frame_ok
    fail_counts = Counter()
    co_fail = Counter()
    n_bil_not_frame = 0
    sidewall_when_bil = 0
    bil_frames = 0
    theta_trend = []
    epos_trend = []
    asym_at_break = []

    for p in tl:
        if not p.get("bilateral"):
            continue
        bil_frames += 1
        if p.get("correct_sidewall"):
            sidewall_when_bil += 1
        if p.get("frame_ok"):
            continue
        n_bil_not_frame += 1
        falses = []
        for k in (
            "working_ok",
            "min_pad_ok",
            "margin_ok",
            "r3_theta_ok",
            "r3_pos_ok",
            "working_theta_ok",
            "working_pos_ok",
        ):
            if not p.get(k):
                fail_counts[k] += 1
                falses.append(k)
        if falses:
            co_fail[tuple(sorted(falses))] += 1
        if p.get("theta") is not None:
            theta_trend.append(float(p["theta"]))
            epos_trend.append(float(p["e_pos"]))

    for b in breaks:
        for k in b.get("false_preds") or []:
            fail_counts[f"break:{k}"] += 1
        if b.get("fn_asym") is not None:
            asym_at_break.append(float(b["fn_asym"]))

    # Categories
    frac_sw = sidewall_when_bil / max(1, bil_frames)
    if frac_sw < 0.3:
        cats.append("A_incorrect_grasp_geometry")
    if st.get("near_ready") and ep.get("max_ok_s", 0) < 0.5:
        cats.append("B_insufficient_contact_stability")
    # pose outside gate while bilateral+min_pad
    pose_block = fail_counts["margin_ok"] + fail_counts["r3_theta_ok"] + fail_counts["r3_pos_ok"]
    pad_block = fail_counts["min_pad_ok"]
    if pose_block > pad_block and pose_block > 0:
        cats.append("C_pose_outside_gate")
    if pad_block > 0 and pad_block >= pose_block:
        cats.append("B_insufficient_contact_stability")
    if breaks and ep.get("max_ok_s", 0) >= 0.15:
        cats.append("D_near_ready_disrupted")
    # FSM: not in gate — note no FSM predicate; timeout without handoff
    if not st.get("handoff") and st.get("stable_hold"):
        # eligibility via timeout fallback only at t_max=3100 which never hits in 80 steps
        cats.append("E_no_timeout_fallback_in_horizon")

    if not cats:
        cats.append("OTHER_GATE_BLOCK")

    # Trends
    def _slope(xs):
        if len(xs) < 4:
            return None
        x = np.arange(len(xs), dtype=float)
        y = np.asarray(xs, float)
        return float(np.polyfit(x, y, 1)[0])

    detail = {
        "fail_counts": dict(fail_counts),
        "top_co_fail": [
            {"preds": list(k), "n": v} for k, v in co_fail.most_common(8)
        ],
        "n_bil_not_frame": n_bil_not_frame,
        "bil_frames": bil_frames,
        "sidewall_frac_when_bilateral": frac_sw,
        "max_ok_s": ep.get("max_ok_s"),
        "n_near_breaks": ep.get("n_near_breaks"),
        "theta_slope": _slope(theta_trend),
        "epos_slope": _slope(epos_trend),
        "mean_fn_asym_at_break": float(np.mean(asym_at_break)) if asym_at_break else None,
    }
    return {"categories": cats, "detail": detail}


def run_attribution() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = _load_timelines()
    rows = []
    cat_counter = Counter()
    pred_fail = Counter()
    representatives: dict[str, list] = defaultdict(list)

    for ep in episodes:
        if not (ep["stages"].get("sustained_bilateral_5") or ep["stages"].get("stable_hold")):
            continue
        if ep["stages"].get("r3_ready") or ep["stages"].get("handoff"):
            continue
        trace = _load_full_trace(ep["traj_id"])
        clf = classify_episode(ep, trace)
        for c in clf["categories"]:
            cat_counter[c] += 1
            if len(representatives[c]) < 3:
                representatives[c].append(
                    {
                        "traj_id": ep["traj_id"],
                        "seed": ep.get("seed"),
                        "max_ok_s": ep.get("max_ok_s"),
                        "stages": ep["stages"],
                        "near_breaks_head": (ep.get("near_breaks") or [])[:3],
                    }
                )
        for k, v in (clf["detail"].get("fail_counts") or {}).items():
            pred_fail[k] += v
        rows.append(
            {
                "traj_id": ep["traj_id"],
                "seed": ep.get("seed"),
                "categories": clf["categories"],
                "detail": clf["detail"],
                "stages": ep["stages"],
                "max_ok_s": ep.get("max_ok_s"),
            }
        )

    report = {
        "n_sustained_fail_r3": len(rows),
        "category_counts": dict(cat_counter),
        "predicate_failure_totals": dict(pred_fail.most_common()),
        "representatives": {k: v for k, v in representatives.items()},
        "rows": rows,
    }
    (OUT / "C_FAILURE_ATTRIBUTION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = ["# C Failure Attribution (P67 Phase 2)\n"]
    md.append(f"Sustained bilateral/stable but not R3-ready: **{len(rows)}** episodes\n")
    md.append("## Category counts (multi-label)\n")
    for k, v in cat_counter.most_common():
        md.append(f"- **{k}**: {v}")
    md.append("\n## Predicate failure totals (bilateral ∧ ¬frame_ok frames + break labels)\n")
    for k, v in pred_fail.most_common(15):
        md.append(f"- `{k}`: {v}")
    md.append("\n## Notes\n")
    md.append(
        "- Categories are **not exclusive** when multiple gate conditions fail.\n"
        "- RecatchTracker `stable_hold` uses min_pad≥2; R3 needs min_pad≥4.\n"
        "- No force-balance gate; `fn_asym` is diagnostic only.\n"
        "- Timeout fallback (t_max=3100) never fires within max_steps=80 → "
        "category E flags horizon/eligibility gap for non-early HANDOFF.\n"
    )
    (OUT / "C_FAILURE_ATTRIBUTION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return report


if __name__ == "__main__":
    run_attribution()
