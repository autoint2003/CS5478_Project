"""Geometry control: cylinder vs flat box, same nominal freeze-here hold.

Diagnostic in-memory geom swap only. Official assets/scene.xml is not written.
No recovery, disturbance, or parameter tuning.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from envs.dynamics import set_finger_object_sliding_mu
from envs.physical_recovery import physical_pack
from training.demo_teleport_recovery_state import PAIR_MU, advance_to_parent, make_parent_sim
from training.nominal_grasp_creep_root_cause import (
    HOLD_S,
    LOG_DT,
    TAU,
    apply_harder,
    dump_json,
    robust_slope,
)
from training.replay_core import freeze, tick_vw
from training.vertical_slip_contact_mechanics_audit import extract_contacts

OUT = ROOT / "results" / "diagnostics" / "box_vs_cylinder_creep"
RAW = OUT / "raw"
FIG = OUT / "figures"


def convert_object_to_box(sim) -> dict:
    """Replace the object geom with a box of the cylinder bounding size.

    Grasp width: cylinder diameter 2*0.018 = 0.036 m. Box half-extent X = 0.018 m
    so flat faces are 36 mm apart, parallel to the finger pads (pinch along world X
    at identity / nominal grasp). Height Z matches cylinder half-height 0.03 m.
    Contact solref/solimp/condim/friction stay those of the original object geom.
    Body inertial stays the XML+runtime mass scaling (m=0.20 after reset).
    """
    m, ids = sim.model, sim.ids
    gid = int(ids.object_geom)
    before = {
        "type": int(m.geom_type[gid]),
        "size": np.array(m.geom_size[gid], float).tolist(),
        "solref": np.array(m.geom_solref[gid], float).tolist(),
        "solimp": np.array(m.geom_solimp[gid], float).tolist(),
        "condim": int(m.geom_condim[gid]),
        "friction": np.array(m.geom_friction[gid], float).tolist(),
        "rbound": float(m.geom_rbound[gid]),
    }
    r = float(before["size"][0])
    hh = float(before["size"][1])
    m.geom_type[gid] = int(mujoco.mjtGeom.mjGEOM_BOX)
    sz = np.array(m.geom_size[gid], float)
    sz[0], sz[1], sz[2] = r, r, hh
    m.geom_size[gid] = sz
    m.geom_rbound[gid] = float(np.linalg.norm(sz[:3]))
    mujoco.mj_forward(m, sim.data)
    after = {
        "type": int(m.geom_type[gid]),
        "type_name": "BOX",
        "size": np.array(m.geom_size[gid], float).tolist(),
        "solref": np.array(m.geom_solref[gid], float).tolist(),
        "solimp": np.array(m.geom_solimp[gid], float).tolist(),
        "condim": int(m.geom_condim[gid]),
        "friction": np.array(m.geom_friction[gid], float).tolist(),
        "rbound": float(m.geom_rbound[gid]),
        "grasp_width_m": 2.0 * r,
        "note": "in-memory only; assets/scene.xml unchanged",
    }
    return {"before": before, "after": after}


def geom_label(sim) -> str:
    t = int(sim.model.geom_type[sim.ids.object_geom])
    return "box" if t == int(mujoco.mjtGeom.mjGEOM_BOX) else "cylinder"


def _centroid_span(cons, side):
    pts = [c for c in cons if c["side"] == side]
    if not pts:
        return {
            "n": 0,
            "cx": float("nan"),
            "cz": float("nan"),
            "span_x": float("nan"),
            "span_z": float("nan"),
        }
    xs = np.array([c["pos_h"][0] for c in pts], float)
    zs = np.array([c["pos_h"][2] for c in pts], float)
    return {
        "n": len(pts),
        "cx": float(np.mean(xs)),
        "cz": float(np.mean(zs)),
        "span_x": float(np.max(xs) - np.min(xs)),
        "span_z": float(np.max(zs) - np.min(zs)),
    }


def sample(sim, t_hold: float, ez0: float) -> dict:
    cons, mech = extract_contacts(sim)
    o = physical_pack(sim)
    L = _centroid_span(cons, "L")
    R = _centroid_span(cons, "R")
    rhos = [c["rho"] for c in cons]
    fts = [c["Ft"] for c in cons]
    keys_mm = []
    keys_pair = []
    for c in cons:
        p = np.round(1e3 * np.array(c["pos_w"], float), 1)  # 0.1 mm
        keys_mm.append((int(c["geom1"]), int(c["geom2"]), float(p[0]), float(p[1]), float(p[2])))
        keys_pair.append((int(c["geom1"]), int(c["geom2"])))
    return {
        "t_hold": t_hold,
        "e_z": float(mech["rh"][2]),
        "dez": float(mech["rh"][2] - ez0),
        "v_rel_z": float(mech["v_rel_h"][2]),
        "aperture": float(o["aperture"]),
        "nL": mech["nL"],
        "nR": mech["nR"],
        "Fn_L": float(sum(abs(c["Fn"]) for c in cons if c["side"] == "L")),
        "Fn_R": float(sum(abs(c["Fn"]) for c in cons if c["side"] == "R")),
        "Ft_sum": float(sum(fts)) if fts else 0.0,
        "rho_mean": float(np.nanmean(rhos)) if rhos else float("nan"),
        "rho_max": float(np.nanmax(rhos)) if rhos else float("nan"),
        "residual_norm": mech["residual_norm"],
        "cx_L": L["cx"],
        "cz_L": L["cz"],
        "spanx_L": L["span_x"],
        "spanz_L": L["span_z"],
        "cx_R": R["cx"],
        "cz_R": R["cz"],
        "spanx_R": R["span_x"],
        "spanz_R": R["span_z"],
        "keys_mm": keys_mm,
        "keys_pair": keys_pair,
        "solref_live": sorted({tuple(c["solref"]) for c in cons if c.get("solref")}),
        "dim_live": sorted({c["dim"] for c in cons}),
        "mu_live": sorted({c["mu"] for c in cons}),
        "geom": geom_label(sim),
        "obj_z": float(o["obj_z"]),
        "hand_z": float(mech["ph"][2]),
    }


def churn_stats(rows):
    def run(keyname):
        life = defaultdict(int)
        births = deaths = 0
        prev = set()
        for r in rows:
            cur = set(tuple(x) for x in r[keyname])
            if prev:
                births += len(cur - prev)
                deaths += len(prev - cur)
            for k in cur:
                life[k] += 1
            prev = cur
        t = float(rows[-1]["t_hold"] - rows[0]["t_hold"])
        dt = float(np.median(np.diff([r["t_hold"] for r in rows]))) if len(rows) > 1 else LOG_DT
        return {
            "births_per_s": births / max(t, 1e-6),
            "deaths_per_s": deaths / max(t, 1e-6),
            "unique": len(life),
            "median_life_s": float(np.median(list(life.values())) * dt) if life else 0.0,
        }

    return {"pos_0.1mm": run("keys_mm"), "geom_pair": run("keys_pair")}


def summarize(rows, extra=None):
    t = np.array([r["t_hold"] for r in rows], float)
    ez = np.array([r["e_z"] for r in rows], float)
    ap = np.array([r["aperture"] for r in rows], float)
    m = (t >= 1.0) & (t <= 10.0)
    vz = np.array([r["v_rel_z"] for r in rows], float)
    i10 = int(np.argmin(np.abs(t - 10.0)))
    s = {
        "slope_ez_1_10_mm_s": 1e3 * robust_slope(t, ez),
        "d_ez_10s_mm": 1e3 * float(ez[i10] - ez[0]),
        "median_vz_1_10": float(np.median(vz[m])) if m.any() else float("nan"),
        "d_aperture_mm": 1e3 * float(ap[-1] - ap[0]),
        "Fn_L_med": float(np.median([r["Fn_L"] for r in rows])),
        "Fn_R_med": float(np.median([r["Fn_R"] for r in rows])),
        "Ft_sum_med": float(np.median([r["Ft_sum"] for r in rows])),
        "rho_mean_med": float(np.nanmedian([r["rho_mean"] for r in rows])),
        "rho_max_max": float(np.nanmax([r["rho_max"] for r in rows])),
        "nL_range": [min(r["nL"] for r in rows), max(r["nL"] for r in rows)],
        "nR_range": [min(r["nR"] for r in rows), max(r["nR"] for r in rows)],
        "cz_L_drift_mm": 1e3 * float(rows[-1]["cz_L"] - rows[0]["cz_L"]),
        "cz_R_drift_mm": 1e3 * float(rows[-1]["cz_R"] - rows[0]["cz_R"]),
        "spanz_L_med_mm": 1e3 * float(np.nanmedian([r["spanz_L"] for r in rows])),
        "spanz_R_med_mm": 1e3 * float(np.nanmedian([r["spanz_R"] for r in rows])),
        "residual_median": float(np.nanmedian([r["residual_norm"] for r in rows])),
        "solref_live": [list(x) for x in rows[len(rows) // 2]["solref_live"]],
        "dim_live": rows[0]["dim_live"],
        "mu_live": rows[0]["mu_live"],
        "geom": rows[0]["geom"],
        "churn": churn_stats(rows),
        "parent_ok": True,
        "n_samples": len(rows),
        "dropped_both": any(r["nL"] == 0 and r["nR"] == 0 for r in rows),
    }
    if extra:
        s.update(extra)
    return s


def make_shape_sim(shape: str):
    sim = make_parent_sim()
    meta = {"shape": shape, "box_convert": None}
    if shape == "box":
        meta["box_convert"] = convert_object_to_box(sim)
    return sim, meta


def capture_parent(shape: str):
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim, meta = make_shape_sim(shape)
    parent = advance_to_parent(sim, cfg, mode="ZERO")
    if not parent.get("ok"):
        return None, meta, parent
    freeze(sim)
    snap = sim.snapshot()
    meta["parent_rel"] = parent.get("rel")
    meta["t_parent"] = float(sim.data.time)
    meta["geom"] = geom_label(sim)
    return snap, meta, parent


def run_hold(snap, shape: str, variant: str):
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim, meta = make_shape_sim(shape)
    sim.load_snapshot(snap)
    set_finger_object_sliding_mu(sim.model, sim.ids, PAIR_MU, data=None)
    mujoco.mj_forward(sim.model, sim.data)
    if variant == "harder":
        apply_harder(sim)
    elif variant == "g0":
        sim.model.opt.gravity[:] = 0.0
        mujoco.mj_forward(sim.model, sim.data)
    elif variant == "g0.5":
        sim.model.opt.gravity[:] = np.array(sim.model.opt.gravity, float) * 0.5
        mujoco.mj_forward(sim.model, sim.data)
    freeze(sim)
    gains = gains_from_cfg(cfg)
    dt = float(sim.model.opt.timestep)
    n = int(round(HOLD_S / dt))
    stride = max(int(round(LOG_DT / dt)), 1)
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    rh0 = float(
        (Rh.T @ (np.array(sim.data.xpos[sim.ids.object_body]) - np.array(sim.data.xpos[sim.ids.hand_body])))[2]
    )
    rows = [sample(sim, 0.0, rh0)]
    for k in range(n):
        tick_vw(sim, np.zeros(3), np.zeros(3), TAU, gains)
        t_hold = (k + 1) * dt
        if k % stride == 0 or k + 1 == n:
            rows.append(sample(sim, t_hold, rh0))
    extra = {"variant": variant, "shape": shape, "box_meta": meta.get("box_convert")}
    return rows, summarize(rows, extra)


def overlay(viewer, title, phase, t_hold, aperture, dez_mm):
    if viewer is None or not hasattr(viewer, "add_overlay"):
        return
    try:
        pos = mujoco.mjtGridPos.mjGRID_TOPLEFT
        viewer.add_overlay(pos, "TITLE", title)
        viewer.add_overlay(pos, "PHASE", str(phase))
        viewer.add_overlay(pos, "HOLD t", f"{t_hold:.2f} s")
        viewer.add_overlay(pos, "APERTURE", f"{aperture:.5f} m")
        viewer.add_overlay(pos, "Delta r_h.z", f"{dez_mm:.2f} mm")
    except Exception:
        pass


def run_viewer(shape: str, variant: str, playback: float, title: str):
    import mujoco.viewer as mjviewer
    from training.demo_teleport_recovery_state import PARENT_HOLD, Z_AIR, _cam
    from training.impact_demo_core import RENDER_HZ, RealtimePacer
    from training.impact_visualization_utils import apply_viewer_camera, enable_viewer_flags

    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim, _ = make_shape_sim(shape)
    if variant == "g0":
        # gravity off only for the hold; construction uses 1g so the grasp exists
        pass
    dt = float(sim.model.opt.timestep)
    cm = mjviewer.launch_passive(sim.model, sim.data)
    viewer = cm.__enter__()
    apply_viewer_camera(viewer, _cam(sim))
    enable_viewer_flags(viewer, show_contact_points=False, show_contact_forces=False)
    pacer = RealtimePacer(playback, RENDER_HZ, dt)
    pacer.start(0.0)
    pacer.note_sync(0.0)
    air_ok = 0.0
    ez0 = None
    try:
        n_max = int(round(12.0 / dt))
        for _ in range(n_max):
            sim.physics_step(None, in_recovery=False)
            sim.maybe_capture_reference()
            o = physical_pack(sim)
            t = float(sim.data.time)
            if pacer.should_render(t):
                overlay(viewer, title, str(sim.fsm.phase), 0.0, float(o["aperture"]), 0.0)
                viewer.sync()
                pacer.note_sync(t)
                pacer.wait_if_ahead(t)
            if sim.fsm.phase == "lift" and sim.captured and float(o["obj_z"]) >= Z_AIR:
                if int(o["nL"]) > 0 and int(o["nR"]) > 0 and float(o["v_rel"]) < 0.05:
                    air_ok += dt
                else:
                    air_ok = 0.0
                if air_ok >= PARENT_HOLD and abs(float(o["e_x"])) < 0.003:
                    freeze(sim)
                    ez0 = float(np.asarray(o["rh"])[2])
                    break
        if ez0 is None:
            freeze(sim)
            ez0 = float(physical_pack(sim)["rh"][2])
        if variant == "g0":
            sim.model.opt.gravity[:] = 0.0
            mujoco.mj_forward(sim.model, sim.data)
            freeze(sim)
        gains = gains_from_cfg(cfg)
        n = int(round(HOLD_S / dt))
        for k in range(n):
            tick_vw(sim, np.zeros(3), np.zeros(3), TAU, gains)
            t_hold = (k + 1) * dt
            o = physical_pack(sim)
            dez = 1e3 * (float(np.asarray(o["rh"])[2]) - ez0)
            if pacer.should_render(float(sim.data.time)):
                overlay(viewer, title, "hold", t_hold, float(o["aperture"]), dez)
                viewer.sync()
                pacer.note_sync(float(sim.data.time))
                pacer.wait_if_ahead(float(sim.data.time))
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    break
    finally:
        cm.__exit__(None, None, None)


def save_traj(tag, rows, summ):
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RAW / f"{tag}.npz",
        t_hold=np.array([r["t_hold"] for r in rows], float),
        e_z=np.array([r["e_z"] for r in rows], float),
        v_rel_z=np.array([r["v_rel_z"] for r in rows], float),
        aperture=np.array([r["aperture"] for r in rows], float),
        Fn_L=np.array([r["Fn_L"] for r in rows], float),
        Fn_R=np.array([r["Fn_R"] for r in rows], float),
        Ft_sum=np.array([r["Ft_sum"] for r in rows], float),
        rho_mean=np.array([r["rho_mean"] for r in rows], float),
        nL=np.array([r["nL"] for r in rows], float),
        nR=np.array([r["nR"] for r in rows], float),
        cz_L=np.array([r["cz_L"] for r in rows], float),
        cz_R=np.array([r["cz_R"] for r in rows], float),
        residual=np.array([r["residual_norm"] for r in rows], float),
    )
    dump_json(RAW / f"{tag}.json", summ)


def figures():
    FIG.mkdir(parents=True, exist_ok=True)

    def L(tag):
        return np.load(RAW / f"{tag}.npz")

    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, lab in (
        ("cylinder_g1", "cylinder 1g"),
        ("box_g1", "box 1g"),
        ("cylinder_g05", "cylinder 0.5g"),
        ("box_g05", "box 0.5g"),
        ("cylinder_g0", "cylinder 0g"),
        ("box_g0", "box 0g"),
    ):
        z = L(tag)
        ax.plot(z["t_hold"], 1e3 * (z["e_z"] - z["e_z"][0]), label=lab)
    ax.set_xlabel("hold t (s)")
    ax.set_ylabel("Delta r_h.z (mm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_rh_z.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for tag, lab in (("cylinder_g1", "cylinder 1g"), ("box_g1", "box 1g"), ("cylinder_harder", "cyl harder"), ("box_harder", "box harder")):
        z = L(tag)
        ax.plot(z["t_hold"], 1e3 * (z["e_z"] - z["e_z"][0]), label=lab)
    ax.set_xlabel("hold t (s)")
    ax.set_ylabel("Delta r_h.z (mm)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_harder.png", dpi=130)
    plt.close(fig)

    names = ["g1", "g05", "g0", "harder"]
    xc = [np.load(RAW / f"cylinder_{k}.npz") for k in ("g1", "g05", "g0", "harder")]
    xb = [np.load(RAW / f"box_{k}.npz") for k in ("g1", "g05", "g0", "harder")]
    # use metrics slopes later; plot trajectories already above
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(L("cylinder_g1")["t_hold"], L("cylinder_g1")["nL"], label="cyl nL")
    ax.plot(L("box_g1")["t_hold"], L("box_g1")["nL"], label="box nL")
    ax.plot(L("cylinder_g1")["t_hold"], L("cylinder_g1")["nR"], ls=":", label="cyl nR")
    ax.plot(L("box_g1")["t_hold"], L("box_g1")["nR"], ls=":", label="box nR")
    ax.set_ylabel("contact count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ncon.png", dpi=130)
    plt.close(fig)
    _ = (names, xc, xb)


def classify(metrics):
    c1 = metrics["cylinder_g1"]["slope_ez_1_10_mm_s"]
    b1 = metrics["box_g1"]["slope_ez_1_10_mm_s"]
    c0 = metrics["cylinder_g0"]["slope_ez_1_10_mm_s"]
    b0 = metrics["box_g0"]["slope_ez_1_10_mm_s"]
    c5 = metrics["cylinder_g05"]["slope_ez_1_10_mm_s"]
    b5 = metrics["box_g05"]["slope_ez_1_10_mm_s"]
    ratio = abs(b1) / max(abs(c1), 1e-6)
    g_scale_c = abs(c5) / max(abs(c1), 1e-6)
    g_scale_b = abs(b5) / max(abs(b1), 1e-9)
    both_g = abs(c0) < 0.15 and abs(b0) < 0.15
    if both_g and 0.7 < ratio < 1.3 and abs(c1) > 0.4 and abs(b1) > 0.4:
        return "A. BOX ~ CYLINDER", "Flat faces creep at a comparable gravity-loaded rate. Curvature is not required."
    if both_g and ratio < 0.25 and abs(c1) > 0.4:
        if abs(b1) < 0.15:
            return "C. BOX ~0, CYLINDER large", "Cylinder-specific contact generation/curvature implicated."
        return "B. BOX << CYLINDER", "Geometry/manifold amplifies creep; box still non-zero."
    if both_g and abs(b1) < 0.15 and abs(c1) > 0.4:
        return "C. BOX ~0, CYLINDER large", "Cylinder-specific contact generation/curvature implicated."
    if both_g and abs(b1) > 0.3 and abs(c1) > 0.3 and (ratio < 0.7 or ratio > 1.3):
        return (
            "D. Both scale with gravity but at different coefficients",
            f"1g box/cyl={ratio:.2f}; 0.5g/1g cyl={g_scale_c:.2f} box={g_scale_b:.2f}.",
        )
    return (
        "UNRESOLVED / inspect table",
        f"cyl 1g={c1:.3f} box 1g={b1:.3f} cyl0={c0:.3f} box0={b0:.3f}",
    )


def write_md(metrics, classif, box_meta):
    p = OUT / "BOX_VS_CYLINDER_CREEP.md"
    L = []
    a = L.append
    a("# Cylinder vs box: nominal gravity-loaded pinch creep")
    a("")
    a("Diagnostic only. Official XML/recovery/controller unchanged. Box is an in-memory geom-type swap of the same object body.")
    a("")
    a("## Evidence classes")
    a("- VERIFIED CODE FACT")
    a("- RAW TRAJECTORY EVIDENCE")
    a("- USER VISUAL OBSERVATION (viewer commands; not claimed here)")
    a("- INTERPRETATION")
    a("")
    a("## Box construction (CODE)")
    a("")
    a("Cylinder XML size is radius 0.018 m, half-height 0.03 m. Box half-extents `(0.018, 0.018, 0.03)` m: grasp faces 36 mm apart, parallel to pads. Same geom solref/solimp/condim/friction, m=0.20 kg, pair mu=1.0, same Panda/tendon/freeze-here hold.")
    a("")
    a("```")
    a(json.dumps(box_meta, indent=2, default=str))
    a("```")
    a("")
    a("## Main table (RAW) — slope `r_h.z` 1–10 s (mm/s) and Δ at 10 s (mm)")
    a("")
    a("| condition | cylinder slope | cylinder Δ10s | box slope | box Δ10s |")
    a("|---|---|---|---|---|")
    for cond, ck, bk in (
        ("1g baseline", "cylinder_g1", "box_g1"),
        ("0.5g baseline", "cylinder_g05", "box_g05"),
        ("0g", "cylinder_g0", "box_g0"),
        ("harder 1g", "cylinder_harder", "box_harder"),
    ):
        c, b = metrics[ck], metrics[bk]
        a(
            f"| {cond} | {c['slope_ez_1_10_mm_s']:.3f} | {c['d_ez_10s_mm']:.2f} | "
            f"{b['slope_ez_1_10_mm_s']:.3f} | {b['d_ez_10s_mm']:.2f} |"
        )
    a("")
    a("## Per-cell details (RAW)")
    a("")
    a("| cell | ap. drift mm | Fn L/R | Ft sum | rho mean/max | nL nR | cz drift L/R mm | pair churn birth/s | |res| |")
    a("|---|---|---|---|---|---|---|---|---|")
    for k, s in metrics.items():
        ch = s["churn"]["geom_pair"]
        a(
            f"| {k} | {s['d_aperture_mm']:.4f} | {s['Fn_L_med']:.2f}/{s['Fn_R_med']:.2f} | "
            f"{s['Ft_sum_med']:.3f} | {s['rho_mean_med']:.3f}/{s['rho_max_max']:.3f} | "
            f"{s['nL_range']}/{s['nR_range']} | {s['cz_L_drift_mm']:.2f}/{s['cz_R_drift_mm']:.2f} | "
            f"{ch['births_per_s']:.1f} | {s['residual_median']:.4f} |"
        )
    a("")
    a("Position-hashed (0.1 mm) churn is also in the json; geom-pair churn is the regeneration diagnostic.")
    a("")
    a("## INTERPRETATION")
    a("")
    a(f"**{classif[0]}**")
    a("")
    a(classif[1])
    a("")
    a("Not a fix. Not a hardware claim. Official model unchanged.")
    a("")
    a("## Viewer")
    a("")
    a("```")
    a("python training/box_vs_cylinder_creep.py --view cylinder --g 1 --playback-speed 0.35")
    a("python training/box_vs_cylinder_creep.py --view box --g 1 --playback-speed 0.35")
    a("python training/box_vs_cylinder_creep.py --view box --g 0 --playback-speed 0.35")
    a("```")
    a("")
    a("Full normal grasp + 10 s freeze-here hold.")
    p.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--view", choices=("cylinder", "box"), default=None)
    p.add_argument("--g", type=float, default=1.0)
    p.add_argument("--playback-speed", type=float, default=0.35)
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    if args.view:
        var = "g0" if abs(args.g) < 1e-9 else "g1"
        title = f"{args.view.upper()} {('0g' if var=='g0' else '1g')} NOMINAL PINCH"
        run_viewer(args.view, var, args.playback_speed, title)
        return 0

    metrics = {}
    box_meta = None
    for shape in ("cylinder", "box"):
        print("parent", shape, "...", flush=True)
        snap, meta, parent = capture_parent(shape)
        if snap is None:
            print("FAILED parent", shape, parent)
            return 2
        ser = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in snap.items()}
        dump_json(RAW / f"state_{shape}.json", ser)
        dump_json(RAW / f"parent_meta_{shape}.json", meta)
        if shape == "box":
            box_meta = meta.get("box_convert")
        for var, tag in (("g1", f"{shape}_g1"), ("g0.5", f"{shape}_g05"), ("g0", f"{shape}_g0"), ("harder", f"{shape}_harder")):
            print("hold", tag, "...", flush=True)
            rows, summ = run_hold(snap, shape, var)
            save_traj(tag, rows, summ)
            metrics[tag] = summ
            print(" ", tag, "slope", round(summ["slope_ez_1_10_mm_s"], 3), "d10", round(summ["d_ez_10s_mm"], 2), flush=True)

    figures()
    classif = classify(metrics)
    dump_json(RAW / "metrics.json", metrics)
    write_md(metrics, classif, box_meta)
    print("class", classif[0])
    print("wrote", OUT / "BOX_VS_CYLINDER_CREEP.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
