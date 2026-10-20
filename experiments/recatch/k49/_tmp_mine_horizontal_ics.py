"""Mine horizontal-center-grasp ICs for Type3 end-catch (no teleport)."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.simulation.ballistic_scene import CYL_H
from envs.simulation.capture import restore_capture
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from evaluation.grasp_geometry import cylinder_axis_world, measure, object_R
from models.k48_checkpoint import load_trained_k48
from scripts.generate_k49_dynamic_recatch_experts import contact_axial_offset
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID,
    SFT,
    build_k49_table,
    decompose,
    load_codebook,
    make_training_env,
)
from training.action_vocab_k49 import assert_k48_prefix_identical

CYL_HALF = 0.5 * float(CYL_H)
OUT = ROOT / "results" / "expert_generation" / "k49_horizontal_endcatch_recatch"
NEAR = [
    "hrv1_920109",
    "hrv1_920031",
]


def horiz_metrics(sim):
    m = measure(sim, U_TGT, P_TGT)
    d = decompose(sim)
    cs = contact_summary(sim)
    u = np.asarray(m["u_c_h"], float)
    u = u / (np.linalg.norm(u) + 1e-12)
    ut = np.asarray(U_TGT, float)
    ut = ut / (np.linalg.norm(ut) + 1e-12)
    # horizontal: axis nearly perpendicular to target
    align = abs(float(np.dot(u, ut)))  # 0 = horizontal, 1 = aligned
    horiz_deg = float(np.degrees(np.arccos(np.clip(align, 0, 1))))  # 90 = horizontal
    ax, _ = contact_axial_offset(sim)
    center_grasp = True
    axial_mean = None
    if ax is not None:
        axial_mean = ax["mean_abs_m"]
        center_grasp = bool(ax["mean_abs_m"] <= 0.45 * CYL_HALF)
    # also use COM axial error as proxy
    e_ax = abs(float(d["e_ax_mm"]))
    if e_ax > 12.0:
        center_grasp = False
    return {
        "e_theta_deg": float(m["e_theta_deg"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_ax_mm": float(d["e_ax_mm"]),
        "e_lat_mm": float(d["e_lat_mm"]),
        "u_c_h": u.tolist(),
        "align_abs": align,
        "horiz_angle_from_target_deg": horiz_deg,  # ~90 desired
        "bilateral": bool(cs["bilateral"]),
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "v_rel_h": np.asarray(m["v_rel_h"], float).tolist(),
        "w_rel_h": np.asarray(m["w_rel_h"], float).tolist(),
        "contact_axial_mean_m": axial_mean,
        "center_grasp": center_grasp,
        "obj_z": float(m["obj_z"]),
    }


def score_ic(h):
    # prefer near-horizontal, center grasp, some contact, large theta
    if not (h["nL"] + h["nR"] > 0):
        return -1e9
    # horiz_angle ~90 is best; also e_theta large
    horiz_score = 90.0 - abs(90.0 - h["horiz_angle_from_target_deg"])
    th = h["e_theta_deg"]
    center = 20.0 if h["center_grasp"] else -30.0
    bil = 5.0 if h["bilateral"] else 0.0
    # need room to correct: theta above SUCCESS threshold
    room = min(th, 80.0)
    return horiz_score + 0.8 * room + center + bil - 0.05 * h["e_pos_mm"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "RAW").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    assert_k48_prefix_identical(table, book48)
    model, _ = load_trained_k48(SFT, device)
    env = make_training_env(device, model=model)

    pools = []
    for d in [
        ROOT / "results/benchmarks/hard_recovery_v1/snapshots",
        ROOT / "results/diagnostics/hard_ic_mining/snapshots",
        ROOT / "results/expert_generation/k49_in_recovery_recatch/snapshots",
        ROOT / "results/expert_generation/k49_dynamic_recatch_experts/snapshots",
        ROOT / "results/diagnostics/large_tilt_eccentric_grasp_mining/snapshots",
    ]:
        if d.exists():
            pools.extend(sorted(d.glob("*.pkl")))

    rows = []
    seen = set()
    for sp in pools:
        ic = sp.stem
        if ic in seen:
            continue
        seen.add(ic)
        try:
            blob = pickle.loads(sp.read_bytes())
            restore_capture(env, blob["capture"])
            h = horiz_metrics(env.sim)
        except Exception as e:
            continue
        sc = score_ic(h)
        rows.append({
            "ic_id": ic,
            "snapshot": str(sp),
            "score": sc,
            "near_miss_seed": ic in NEAR,
            **h,
        })

    rows.sort(key=lambda r: -r["score"])
    # filters for "suitable"
    suitable = [
        r for r in rows
        if r["score"] > -1e8
        and r["e_theta_deg"] >= 25.0
        and r["horiz_angle_from_target_deg"] >= 55.0
        and r["center_grasp"]
        and (r["nL"] + r["nR"]) > 0
    ]
    # also keep near-miss seeds even if milder
    for r in rows:
        if r["ic_id"] in NEAR and r["ic_id"] not in {x["ic_id"] for x in suitable}:
            suitable.insert(0, r)

    # diversify by episode prefix
    picked = []
    eps = set()
    for r in suitable:
        ep = r["ic_id"].rsplit("_", 1)[0] if "_" in r["ic_id"] else r["ic_id"]
        # allow a few per episode family (hrv1 vs hard)
        key = r["ic_id"][:8]
        n_same = sum(1 for p in picked if p["ic_id"][:8] == key)
        if n_same >= 3 and r["ic_id"] not in NEAR:
            continue
        picked.append(r)
        if len(picked) >= 16:
            break

    out = {
        "n_scanned": len(rows),
        "n_suitable": len(suitable),
        "picked": picked,
        "top20": rows[:20],
        "u_tgt": U_TGT.tolist(),
        "cyl_half_m": CYL_HALF,
    }
    (OUT / "RAW" / "horizontal_ic_mine.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print("scanned", len(rows), "suitable", len(suitable), "picked", len(picked))
    for r in picked[:16]:
        print(
            f"{r['ic_id']:20s} th={r['e_theta_deg']:5.1f} horiz={r['horiz_angle_from_target_deg']:5.1f} "
            f"ax={r['e_ax_mm']:6.1f} bil={r['bilateral']} center={r['center_grasp']} "
            f"sc={r['score']:.1f} near={r['near_miss_seed']}"
        )


if __name__ == "__main__":
    main()
