"""Phase 1 — contact geometry ground-truth label audit."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from experiments.precision.p63_transformer_ppo_c_grasp.contact_events import is_correct_sidewall
from experiments.precision.p77_contact_observability.paths import FC_WINDOWS, OUT, P74_AUDIT, P75_DATA


def _audit_p74() -> dict:
    raw = json.loads(P74_AUDIT.read_text(encoding="utf-8"))
    event_counts = Counter()
    timing = defaultdict(list)
    label_consistency = {
        "opposing_implies_sw_and_bilateral": 0,
        "opposing_total": 0,
        "sw_without_bilateral": 0,
        "sw_total": 0,
        "future_leak_check_notes": [],
    }
    # Per-episode first times
    first_times = []
    for ep in raw["rows"]:
        steps = ep["steps"]
        t_fc = t_sw = t_opp = t_loss = t_r3 = None
        saw_contact = False
        for s in steps:
            w = s.get("window")
            if w:
                event_counts[f"window:{w}"] += 1
            hc = int(s.get("has_pad_contact") or 0)
            sw = int(s.get("correct_sidewall") or 0)
            opp = int(s.get("opposing") or 0)
            if hc and t_fc is None:
                t_fc = int(s["t"])
                event_counts["first_contact"] += 1
            if sw and t_sw is None:
                t_sw = int(s["t"])
                event_counts["first_correct_sidewall"] += 1
            if opp and t_opp is None:
                t_opp = int(s["t"])
                event_counts["first_opposing"] += 1
            if saw_contact and not hc and t_loss is None:
                t_loss = int(s["t"])
                event_counts["first_contact_loss"] += 1
            if hc:
                saw_contact = True
            if int(s.get("frame_ok") or 0) and t_r3 is None and int(ep.get("r3_ready") or 0):
                # approximate: first frame_ok on r3 episodes
                pass
            if opp:
                label_consistency["opposing_total"] += 1
                if sw and int(s.get("bilateral") or 0):
                    label_consistency["opposing_implies_sw_and_bilateral"] += 1
            if sw:
                label_consistency["sw_total"] += 1
                if not int(s.get("bilateral") or 0):
                    label_consistency["sw_without_bilateral"] += 1
        if int(ep.get("r3_ready") or 0):
            event_counts["episode_r3_ready"] += 1
        first_times.append(
            {
                "traj_id": ep["traj_id"],
                "t_fc": t_fc,
                "t_sw": t_sw,
                "t_opp": t_opp,
                "t_loss": t_loss,
                "reached_opposing": int(ep.get("reached_opposing") or 0),
                "lost_contact_after_fc": int(ep.get("lost_contact_after_fc") or 0),
                "r3_ready": int(ep.get("r3_ready") or 0),
                "fc_t_stored": ep.get("fc_t"),
            }
        )
        if t_fc is not None and ep.get("fc_t") is not None:
            timing["fc_t_match"].append(int(t_fc == int(ep["fc_t"]) or abs(t_fc - int(ep["fc_t"])) <= 1))

    # Definition string
    definition = {
        "correct_sidewall": (
            "is_correct_sidewall(audit): grasp_kind=='opposing_sidewall' AND opposing_sidewall "
            "AND n_end_face==0 AND bilateral (p63 contact_events)."
        ),
        "opposing_bilateral": "bilateral AND correct_sidewall (P74/P75 stage logic).",
        "first_contact": "first has_pad_contact after episode start.",
        "contact_loss": "first transition contact→no-contact after having had contact post-FC.",
        "r3_ready": "episode-level (dwell∧frame)∨handoff; not a per-step geometry label.",
        "source": str(is_correct_sidewall.__code__.co_filename),
    }
    label_consistency["frac_opposing_consistent"] = (
        label_consistency["opposing_implies_sw_and_bilateral"] / max(label_consistency["opposing_total"], 1)
    )
    label_consistency["future_leak_check_notes"] = [
        "Labels are post-step geometry at the same decision index as stored obs (instrument order: act→step→geometry).",
        "Do not use episode-final reached_opposing as a current-state feature (outcome only).",
        "Window tags are causal stage machines updated after each step.",
    ]
    return {
        "n_episodes": len(raw["rows"]),
        "definition": definition,
        "event_counts": dict(event_counts),
        "fc_t_match_rate": float(np.mean(timing["fc_t_match"])) if timing["fc_t_match"] else None,
        "label_consistency": label_consistency,
        "first_times_sample": first_times[:5],
        "n_with_sw": sum(1 for f in first_times if f["t_sw"] is not None),
        "n_with_opp": sum(1 for f in first_times if f["t_opp"] is not None),
    }


def _audit_p75() -> dict:
    import pickle

    train = pickle.loads((P75_DATA / "train_episodes.pkl").read_bytes())
    # C only stage transitions
    stage_trans = Counter()
    n_c = 0
    causal_ok = 0
    for ep in train:
        if ep["skill"] != "C":
            continue
        n_c += 1
        stages = ep["stages"]
        for t in range(len(stages) - 1):
            if stages[t] != stages[t + 1]:
                stage_trans[f"{stages[t]}->{stages[t + 1]}"] += 1
        # causal: stage at t must not encode future opposing if never reached
        if not ep["reached_opposing"]:
            if "opposing_to_frame" not in stages and "frame_to_r3" not in stages:
                causal_ok += 1
        else:
            causal_ok += 1
    return {
        "n_train_C": n_c,
        "stage_transitions": dict(stage_trans.most_common(20)),
        "frac_causal_stage_consistent": float(causal_ok / max(n_c, 1)),
        "fc_windows": list(FC_WINDOWS),
        "note": "P75 stores stages without privileged sidewall; use P74/sense-diag for sidewall labels.",
    }


def run_label_audit() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"p74": _audit_p74(), "p75": _audit_p75()}
    (OUT / "CONTACT_LABEL_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "CONTACT_LABEL_AUDIT.md").write_text(
        "# Contact Label Audit (P77 Phase 1)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps({"p74_n": report["p74"]["n_episodes"], "def": report["p74"]["definition"]["correct_sidewall"]}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_label_audit()
