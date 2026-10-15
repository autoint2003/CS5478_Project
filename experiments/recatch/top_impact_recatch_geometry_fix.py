"""Fix over-chase depth and close geometry on one 6 m/s top-impact IC.

Uses official step_legal. Does not change the robot, nominal controller,
TABLE_DROP, or the K48 codebook.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_generation.strategies.recatch_oracle import MAIN_HALF, MAIN_CENTER, _geom_name, _geom_kind, _class_of, contacts_now, _pad_band, _class_force, _wrist_sat, _metrics, rollout, staged_choose, _depth_vz, _blend_depth, _base_spec

from controllers.recovery_actions import TAU_OPEN, TAU_SECURE
from experiments.ball_impact.ball_impact_recoverability import _action_from, _nominal_fail, _setup
from envs.airborne_obs import observe_airborne
from envs.observable_obs import ObservableObsState
from data_generation.strategies.capture_geometry import capture_geometry
from envs.horizon_outcome import NOMINAL_S
from experiments.ball_impact.privileged_ball_impact_search import representability, step_legal
from envs.table_drop import latched, restore_state
from experiments.recatch.top_impact_open_overchase_recatch import (
    _cells,
    _finish,
    _impact,
    _pose,
    _video,
)
from training.unified_k47 import load_codebook
from experiments.recatch.legal_recatch_transfer import _bottom
from training.replay_core import TABLE_TOP
from training.recovery_runtime import restore_ballistic

OUT = ROOT / "results" / "diagnostics" / "top_impact_recatch_geometry_fix"
IC_ID = "s6.0_p10_a+0.000_e0.000"




















def _xy(pose):
    return np.array([
        float(np.clip(3.0 * pose["p_rel"][0], -0.2, 0.2)),
        float(np.clip(3.0 * pose["p_rel"][1], -0.2, 0.2)),
    ])


def original_choose(pose, _raw, state):
    """The over-deep chase that recontacted around 0.14 s."""
    rh_z = -float(2.2)
    centered = abs(float(pose["p_rel"][0])) < 0.025 and abs(float(pose["p_rel"][1])) < 0.02
    below = float(pose["pinch"][2]) <= float(pose["p_obj"][2]) - 0.004
    both = pose["nL"] > 0 and pose["nR"] > 0
    mode = state["mode"]
    if mode in ("open", "chase"):
        tau = TAU_OPEN if pose["gap"] < 0.044 else 0.0
        mode = "chase" if pose["gap"] >= 0.044 else "open"
        if below and centered and pose["gap"] >= 0.042 and pose["bottom"] > 0.50:
            mode = "close"
        vz = rh_z
    elif mode == "close":
        tau = TAU_SECURE if both else -8.0
        vz = float(min(pose["v_obj"][2] - 0.30, -0.20))
        state["both"] = state.get("both", 0) + 1 if both else 0
        if state["both"] >= 2:
            mode = "brake"
    elif mode == "brake":
        tau = TAU_SECURE
        speed = float(pose["v_obj"][2])
        vz = 0.0 if speed >= -0.02 else min(0.0, speed + 6.0 * 0.02)
        if both and pose["v_obj"][2] > -0.05 and pose["v_hand"][2] > -0.08:
            mode = "lift"
        if not both:
            mode = "close"
    else:
        tau = TAU_SECURE
        vz = 0.08
    state["mode"] = mode
    hand_xy = _xy(pose)
    # _xy is in hand frame; world XY correction stays small and is applied in world
    # by the same mapping the previous oracle used after rotating. Here the
    # stored p_rel is hand frame, and world X/Y are nearly hand Y/X. Keep the
    # correction in world by using the object velocity for X/Y during close.
    if mode in ("open", "chase"):
        v = np.array([hand_xy[1], hand_xy[0], vz])
    else:
        v = np.array([
            float(np.clip(pose["v_obj"][0], -0.3, 0.3)),
            float(np.clip(pose["v_obj"][1], -0.3, 0.3)),
            vz,
        ])
    v = np.clip(v, [-1.0, -1.0, -3.8], [1.0, 1.0, 1.0])
    return {"v": v, "tau": tau}










def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _public(stage, spec, result):
    row = {"stage": stage, **{key: spec[key] for key in spec}}
    row.update(result["metrics"])
    return row


def _save_contacts(path, contacts):
    if not contacts:
        return
    np.savez_compressed(
        path,
        t=np.array([item["t"] for item in contacts]),
        side=np.array([item["side"] for item in contacts]),
        klass=np.array([item["class"] for item in contacts]),
        local=np.array([item["local"] for item in contacts]),
        fn=np.array([item["fn"] for item in contacts]),
        ft=np.array([item["ft"] for item in contacts]),
        fz_n=np.array([item["fz_n"] for item in contacts]),
        dist=np.array([item["dist"] for item in contacts]),
        cz=np.array([item["cz"] for item in contacts]),
        gap=np.array([item["gap"] for item in contacts]),
    )


def _plot(path, contacts, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "LOWER_VERTICAL_PAD": "#1b7f4e",
        "UPPER_SLOPED_PART": "#c0392b",
        "EDGE_CONTACT": "#d68910",
        "OTHER": "#5d6d7e",
    }
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), sharey=True)
    for axis, side in zip(axes, ("L", "R")):
        chosen = [item for item in contacts if item["side"] == side]
        for name, color in colors.items():
            pts = [item for item in chosen if item["class"] == name]
            if not pts:
                continue
            axis.scatter(
                [item["local"][1] * 1000.0 for item in pts],
                [item["local"][2] * 1000.0 for item in pts],
                s=[8.0 + 2.0 * item["fn"] for item in pts],
                c=color, label=name, alpha=0.85,
            )
        axis.add_patch(plt.Rectangle(
            ((MAIN_CENTER[1] - MAIN_HALF[1]) * 1000.0, (MAIN_CENTER[2] - MAIN_HALF[2]) * 1000.0),
            2 * MAIN_HALF[1] * 1000.0, 2 * MAIN_HALF[2] * 1000.0,
            fill=False, linestyle="--", color="#1b7f4e",
        ))
        axis.set_title(f"{side} finger local y–z")
        axis.set_xlabel("local y (mm)")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("local z (mm), tip is up")
    axes[1].legend(fontsize=7, loc="best")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _better(candidate, incumbent):
    if incumbent is None:
        return True
    left = _rank(candidate["metrics"])
    right = _rank(incumbent["metrics"])
    return left > right


def _rank(metrics):
    return (
        int(metrics["t_pad_hold"] is not None),
        int(not metrics["table"]),
        int(metrics["lifted"]),
        int(metrics["t_pad"] is not None),
        metrics["min_bottom"],
        -abs((metrics["re_cz"] if metrics["re_cz"] is not None else 0.0) - 0.102),
        metrics["min_fz_n"],
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "VIDEOS").mkdir(exist_ok=True)
    sim, _base, ref, _target, _book = _setup()
    book = load_codebook()
    cell = next(item for item in _cells() if item["id"] == IC_ID)
    info, err = _impact(sim, _base, cell)
    if err:
        raise RuntimeError(err)
    band = _pad_band(sim)
    print("PAD", [round(v, 4) for v in band], "cz0", round(info["pose"]["cz"], 4), flush=True)
    nominal = _nominal_fail(sim, info["snap"], ref)
    print("NOMINAL", nominal["why"], nominal["t"], flush=True)

    original = rollout(sim, info["snap"], original_choose, 1.6)
    _save_contacts(OUT / "original_contacts.npz", original["contacts"])
    _plot(OUT / "original_contact_map.png", original["contacts"], "original over-chase contacts")
    window = [item for item in original["contacts"] if 0.02 - 1e-6 <= item["t"] <= 0.30 + 1e-6]
    at_14 = [item for item in original["contacts"] if abs(item["t"] - 0.14) < 0.011]
    audit_contacts = []
    for item in at_14:
        audit_contacts.append({
            "t": round(item["t"], 3),
            "side": item["side"],
            "geom": item["geom"],
            "class": item["class"],
            "kind": item["kind"],
            "local_mm": [round(1000.0 * float(v), 2) for v in item["local"]],
            "offset_mm": [round(1000.0 * float(v), 2) for v in item["offset"]],
            "normal": [round(float(v), 3) for v in item["normal"]],
            "fn": round(item["fn"], 2),
            "ft": round(item["ft"], 2),
            "fz_n": round(item["fz_n"], 2),
            "dist_mm": round(1000.0 * item["dist"], 3),
            "dim": item["dim"],
            "efc": item["efc"],
            "cz": round(item["cz"], 4),
            "gap_mm": round(1000.0 * item["gap"], 1),
        })
    by_t = {}
    for item in original["contacts"]:
        by_t.setdefault(round(item["t"], 3), []).append(item)
    spike_t = min(by_t, key=lambda key: sum(item["fz_n"] for item in by_t[key])) if by_t else None
    spike_rows = [] if spike_t is None else sorted(by_t[spike_t], key=lambda item: item["fz_n"])[:8]
    def _jsonable(value):
        if isinstance(value, dict):
            return {key: _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    (OUT / "contact_audit_0p14.json").write_text(json.dumps(_jsonable({
        "pad_band_hand_z": [round(band[0], 4), round(band[1], 4)],
        "n_window": len(window),
        "n_at_0p14": len(at_14),
        "class_counts": {
            name: sum(1 for item in at_14 if item["class"] == name)
            for name in ("LOWER_VERTICAL_PAD", "UPPER_SLOPED_PART", "EDGE_CONTACT", "OTHER")
        },
        "contacts": audit_contacts,
        "worst_normal": [
            {
                "t": spike_t,
                "class": item["class"],
                "kind": item["kind"],
                "fn": round(item["fn"], 2),
                "ft": round(item["ft"], 2),
                "fz_n": round(item["fz_n"], 2),
                "local_mm": [round(1000.0 * float(v), 2) for v in item["local"]],
            }
            for item in spike_rows
        ],
        "metrics": original["metrics"],
    }), indent=2), encoding="utf-8")
    print("AUDIT 0.14", json.loads((OUT / "contact_audit_0p14.json").read_text(encoding="utf-8"))["class_counts"], "spike", spike_t, flush=True)

    depth_rows = []
    best = None
    best_spec = None
    for stop_cz in (0.110, 0.106, 0.102, 0.098, 0.094):
        for close_dive in (True, False):
            spec = _base_spec()
            spec["stop_cz"] = stop_cz
            spec["close_dive"] = close_dive
            result = rollout(sim, info["snap"], staged_choose(spec), 1.4)
            depth_rows.append(_public("A", spec, result))
            print("A", stop_cz, "dive", close_dive, result["metrics"]["re_class"], result["metrics"]["re_cz"], "pad", result["metrics"]["t_pad"], "bot", result["metrics"]["min_bottom"], result["metrics"]["end_mode"], flush=True)
            if _better(result, best):
                best, best_spec = result, dict(spec)
    _write(OUT / "OVERCHASE_DEPTH_SWEEP.csv", depth_rows)

    close_rows = []
    spec_b = dict(best_spec)
    spec_b["close_dive"] = False
    for open_gap, close_gap, arrest in (
        (0.044, 0.042, 0.0),
        (0.050, 0.046, -1.0),
        (0.058, 0.048, -1.0),
        (0.070, 0.050, -1.0),
    ):
        for early in (0.0, 0.004):
            spec = dict(spec_b)
            spec["open_gap"] = open_gap
            spec["close_gap"] = close_gap
            spec["arrest"] = arrest
            spec["stop_cz"] = spec_b["stop_cz"] + early
            result = rollout(sim, info["snap"], staged_choose(spec), 1.4)
            close_rows.append(_public("B", spec, result))
            print("B", open_gap, early, result["metrics"]["re_class"], result["metrics"]["t_re"], result["metrics"]["gap_max_mm"], result["metrics"]["min_bottom"], flush=True)
            if _better(result, best):
                best, best_spec = result, dict(spec)

    spec_c = dict(best_spec)
    spec_c["close_dive"] = False
    for close_tau in (-4.0, -6.0, -8.0, -12.0):
        for hold_tau in (-6.0, -10.0, -18.0):
            spec = dict(spec_c)
            spec["close_tau"] = close_tau
            spec["hold_tau"] = hold_tau
            result = rollout(sim, info["snap"], staged_choose(spec), 1.4)
            close_rows.append(_public("C", spec, result))
            print("C", close_tau, hold_tau, result["metrics"]["re_class"], result["metrics"]["min_fz_n"], result["metrics"]["gap_min_mm"], result["metrics"]["t_pad_hold"], flush=True)
            if _better(result, best):
                best, best_spec = result, dict(spec)
    _write(OUT / "CLOSE_TIMING_FORCE_SWEEP.csv", close_rows)

    brake_rows = []
    spec_d = dict(best_spec)
    for brake in (3.0, 6.0, 10.0):
        for pad_steps in (1, 3):
            spec = dict(spec_d)
            spec["brake"] = brake
            spec["pad_steps"] = pad_steps
            result = rollout(sim, info["snap"], staged_choose(spec), 1.6)
            brake_rows.append(_public("D", spec, result))
            print("D", brake, pad_steps, result["metrics"]["min_bottom"], result["metrics"]["t_vz0"], result["metrics"]["end_bottom"], result["metrics"]["table"], flush=True)
            if _better(result, best):
                best, best_spec = result, dict(spec)
    for lift in (0.08, 0.15):
        spec = dict(best_spec)
        spec["lift"] = lift
        result = rollout(sim, info["snap"], staged_choose(spec), 2.2)
        brake_rows.append(_public("E", spec, result))
        print("E", lift, result["metrics"]["end_mode"], result["metrics"]["end_bottom"], result["metrics"]["lifted"], result["metrics"]["end_cz"], flush=True)
        if _better(result, best):
            best, best_spec = result, dict(spec)
    _write(OUT / "BRAKE_LIFT_RESULTS.csv", brake_rows)

    label = "NO_VERTICAL_PAD_HOLD"
    formal = None
    extension = None
    k48 = None
    held = best["metrics"]["t_pad_hold"] is not None and not best["metrics"]["table"]
    if held and best["metrics"]["end_mode"] == "handoff":
        actions = np.stack(best["actions"])
        packed = representability(actions, book)
        k48 = None if packed["max_linf"] is None else float(packed["max_linf"])
        if k48 is not None and k48 <= 0.25 + 1e-8:
            restore_ballistic(sim, info["snap"])
            restore_state(sim, None)
            for action in packed["quantized"]:
                step_legal(sim, action)
            out, extension = _finish(sim, ref, extra=5.0)
            formal = out.get("reason")
            label = formal if out.get("success") and extension == "HELD_EXTENSION" else (
                "FORMAL_SUCCESS_ONLY" if out.get("success") else formal
            )
            np.savez_compressed(
                OUT / "best_actions.npz",
                actions=actions,
                tokens=packed["tokens"],
                residual=(actions - book[packed["tokens"]]).astype(np.float32),
            )
        else:
            label = "K48_OUT_OF_RANGE"
    elif held:
        label = "PARTIAL_RECATCH"
    elif best["metrics"]["table"]:
        label = "TABLE"
    print("BEST", label, best_spec["stop_cz"], best["metrics"]["end_mode"], formal, extension, k48, flush=True)

    _save_contacts(OUT / "best_contacts.npz", best["contacts"])
    _plot(OUT / "best_contact_map.png", best["contacts"], f"best {label}")
    np.savez_compressed(
        OUT / "best_trace.npz",
        t=np.array([row["t"] for row in best["rows"]]),
        cz=np.array([row["cz"] for row in best["rows"]]),
        gap=np.array([row["gap"] for row in best["rows"]]),
        bottom=np.array([row["bottom"] for row in best["rows"]]),
        v_obj=np.array([row["v_obj"] for row in best["rows"]]),
        v_hand=np.array([row["v_hand"] for row in best["rows"]]),
        p_hand=np.array([row["p_hand"] for row in best["rows"]]),
        mode=np.array([row["mode"] for row in best["rows"]]),
        qpos=np.array([row["qpos"] for row in best["rows"]]),
        nL=np.array([row["nL"] for row in best["rows"]]),
        nR=np.array([row["nR"] for row in best["rows"]]),
    )
    t_off = info["rows"][-1]["t"] + 0.02
    video_rows = list(info["rows"])
    for row in best["rows"]:
        copied = dict(row)
        copied["t"] = row["t"] + t_off
        video_rows.append(copied)
    _video(video_rows, OUT / "VIDEOS" / "best.mp4", IC_ID)
    timeline = {
        "label": label,
        "formal": formal,
        "extension": extension,
        "k48_linf": k48,
        "nominal": nominal["why"],
        "nominal_t": nominal["t"],
        "spec": best_spec,
        "metrics": best["metrics"],
        "pad_band": [round(band[0], 4), round(band[1], 4)],
    }
    (OUT / "best.json").write_text(json.dumps(_jsonable(timeline), indent=2), encoding="utf-8")
    with (OUT / "EVENT_TIMELINE.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "t_open", "t_off", "t_re", "t_pad", "t_pad_hold", "t_vz0", "end_mode", "formal", "extension"])
        writer.writeheader()
        writer.writerow({"label": label, "formal": formal, "extension": extension, **{key: best["metrics"][key] for key in ("t_open", "t_off", "t_re", "t_pad", "t_pad_hold", "t_vz0", "end_mode")}})
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
