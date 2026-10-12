"""CENTER-6 controlled-slip sufficiency (duration only). Not heuristic optimization.

Frozen: impact, EARLY, +120°, τ=-5, τ=-18 brake, corrected RETURN, noslip=1.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from controllers.jacobian_controller import gains_from_cfg
from envs.config_util import load_yaml, merge_sim_config
from envs.physical_recovery import TABLE_DROP, physical_pack
from sensors.spatial_tactile import PAD1_HALF, measure_spatial_tactile
from training.ballistic_large_angle import (
    BRAKE_S,
    LA_RAW,
    RAW,
    dump,
    load_early,
    ori_pack,
    return_to_nominal,
    rotate_to,
    tick_log,
)
from training.demo_ballistic_impact import make_sim, z_tgt_of
from training.demo_ballistic_recovery import HOLD_S, OUT, continue_zero, measure, restore_ballistic, slim_m
from training.demo_teleport_recovery_state import d_in_pack, pad_span_hand
from training.replay_core import freeze
from training.vertical_slip_contact_mechanics_audit import extract_contacts
from training.write_ballistic_impact_scene import CYL_R

DURS = (0.70, 1.00, 1.30, 1.60)
PAD_U_HALF = float(PAD1_HALF[0])


def _tac(sim) -> dict:
    r = measure_spatial_tactile(sim.model, sim.data, sim.ids)
    def mm(ok, x):
        return None if not ok else 1e3 * float(x)
    return {
        "u_L_mm": mm(r.left.valid, r.left.u),
        "v_L_mm": mm(r.left.valid, r.left.v),
        "u_R_mm": mm(r.right.valid, r.right.u),
        "v_R_mm": mm(r.right.valid, r.right.v),
        "fn_L": float(r.left.fn),
        "fn_R": float(r.right.fn),
        "n_L": int(r.left.n_contacts),
        "n_R": int(r.right.n_contacts),
        "valid_L": bool(r.left.valid),
        "valid_R": bool(r.right.valid),
        "u_L_edge": None if not r.left.valid else abs(float(r.left.u)) / PAD_U_HALF,
        "u_R_edge": None if not r.right.valid else abs(float(r.right.u)) / PAD_U_HALF,
    }


def geometry_snapshot(sim, R_nom, tag: str) -> dict:
    fc, _ = extract_contacts(sim)
    o = physical_pack(sim)
    rh = np.asarray(o["rh"], float)
    pad = pad_span_hand(sim)
    din = d_in_pack(sim, rh)
    Ro = np.array(sim.data.xmat[sim.ids.object_body].reshape(3, 3), float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    rows = []
    for c in fc:
        pos_w = np.asarray(c["pos_w"], float)
        pos_cyl = Ro.T @ (pos_w - po)
        pos_h = np.asarray(c["pos_h"], float)
        rows.append(
            {
                "side": c["side"],
                "Fn": float(c["Fn"]),
                "rho": c["rho"],
                "pos_h_mm": (1e3 * pos_h).tolist(),
                "pos_cyl_mm": (1e3 * pos_cyl).tolist(),
                "cyl_radial_mm": 1e3 * float(np.hypot(pos_cyl[0], pos_cyl[1])),
                "cyl_axial_mm": 1e3 * float(pos_cyl[2]),
                "pad_x_frac": float(
                    (pos_h[0] - pad["x_lo"]) / max(pad["x_hi"] - pad["x_lo"], 1e-9)
                ),
            }
        )
    tac = _tac(sim)
    edge_u = [x for x in (tac["u_L_edge"], tac["u_R_edge"]) if x is not None]
    return {
        "tag": tag,
        "t": float(sim.data.time),
        "rh_mm": (1e3 * rh).tolist(),
        "tilt_deg": float(np.degrees(np.arccos(np.clip(np.asarray(o["R_rel"], float).reshape(3, 3)[2, 2], -1, 1)))),
        "v_rel_h": np.asarray(o["v_rel_h"], float).tolist(),
        "w_rel_h": np.asarray(o["w_rel_h"], float).tolist(),
        "nL": int(o["nL"]),
        "nR": int(o["nR"]),
        "Fn_L": float(sum(abs(c["Fn"]) for c in fc if c["side"] == "L")),
        "Fn_R": float(sum(abs(c["Fn"]) for c in fc if c["side"] == "R")),
        "rho_max": float(np.nanmax([c["rho"] for c in fc])) if fc else None,
        "aperture": float(o["aperture"]),
        "obj_z": float(o["obj_z"]),
        "pad_x_lo_mm": 1e3 * pad["x_lo"],
        "pad_x_hi_mm": 1e3 * pad["x_hi"],
        "from_plus_x_rim_mm": din["from_plus_x_rim_mm"],
        "from_distal_tip_mm": din["from_distal_tip_mm"],
        "d_in_mm": din["d_in_mm"],
        "cyl_R_mm": 1e3 * CYL_R,
        "tactile": tac,
        "max_pad_u_edge": max(edge_u) if edge_u else None,
        "contacts": rows,
        "ori": {
            "err_des_deg": ori_pack(sim, R_nom)["err_des_deg"],
            "err_hand_deg": ori_pack(sim, R_nom)["err_hand_deg"],
        },
    }


def run_duration(sim, gains, snap, plan, slip_s: float) -> dict:
    restore_ballistic(sim, snap)
    R_nom = np.asarray(sim.fsm.r_des, float).reshape(3, 3).copy()
    Rh0 = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float).copy()
    freeze(sim)
    z_tgt = float(snap.get("z_tgt", z_tgt_of(sim)))
    log = [measure(sim, -18.0, Rh0)]
    lost_at = None
    rot = rotate_to(sim, gains, Rh0, float(plan["theta_deg"]), -18.0, log)
    geo_rot = geometry_snapshot(sim, R_nom, "post_rotate")
    rh_slip0 = float(log[-1]["rh"][0])
    dt = float(sim.model.opt.timestep)
    n_slip = int(round(float(slip_s) / dt))
    vrels = []
    t_table, _ = tick_log(
        sim, gains, 0.0, 0.0, 0.0, float(plan["tau_slip"]), Rh0, log, n_slip, label="WEAK_SLIP"
    )
    for m in log:
        if m["t"] >= geo_rot["t"] - 1e-6:
            vrels.append(float(np.linalg.norm(m["v_rel_h"])))
        if m["obj_z"] < TABLE_DROP or (m["nL"] == 0 and m["nR"] == 0):
            lost_at = "slip"
            break
    geo_slip = geometry_snapshot(sim, R_nom, "post_slip")
    rh_slip1 = float(log[-1]["rh"][0])
    out = {
        "slip_s": slip_s,
        "rotate": {k: rot[k] for k in ("target_deg", "actual_deg", "reached", "t_table")},
        "rhx_slip_start_mm": 1e3 * rh_slip0,
        "rhx_slip_end_mm": 1e3 * rh_slip1,
        "drhx_mm": 1e3 * (rh_slip1 - rh_slip0),
        "vrel_slip": {
            "n": len(vrels),
            "mean": float(np.mean(vrels)) if vrels else None,
            "max": float(np.max(vrels)) if vrels else None,
            "p90": float(np.percentile(vrels, 90)) if vrels else None,
        },
        "lost_at": lost_at,
        "table": t_table,
        "geo_rot": geo_rot,
        "geo_slip": geo_slip,
    }
    if t_table is not None or log[-1]["obj_z"] < TABLE_DROP:
        out["lost_at"] = out["lost_at"] or "slip_table"
        return out
    n_br = int(round(BRAKE_S / dt))
    t_table, _ = tick_log(sim, gains, 0.0, 0.0, 0.0, -18.0, Rh0, log, n_br, label="BRAKE")
    out["geo_brake"] = geometry_snapshot(sim, R_nom, "post_brake")
    if t_table is not None:
        out["lost_at"] = "brake"
        out["table"] = t_table
        return out
    ret = return_to_nominal(sim, gains, R_nom, Rh0, log)
    out["geo_return"] = geometry_snapshot(sim, R_nom, "post_return")
    out["return"] = {
        "err_des_deg": ret["end"]["err_des_deg"],
        "err_hand_deg": ret["end"]["err_hand_deg"],
        "dt": ret["dt"],
    }
    if ret["t_table"] is not None:
        out["lost_at"] = "return"
        out["table"] = ret["t_table"]
        return out
    t_hold0 = float(sim.data.time)
    t_table = continue_zero(sim, gains, HOLD_S + 4.0, log, Rh0, z_tgt)
    hold = []
    for m in log:
        if m["t"] + 1e-9 < t_hold0:
            continue
        if (not hold) or m["t"] - hold[-1]["t"] >= 0.20:
            hold.append(
                {
                    "t": m["t"],
                    "rhx_mm": 1e3 * float(m["rh"][0]),
                    "tilt_deg": m["tilt_deg"],
                    "vrel": float(np.linalg.norm(m["v_rel_h"])),
                    "nL": m["nL"],
                    "nR": m["nR"],
                    "Fn_L": m["Fn_L"],
                    "Fn_R": m["Fn_R"],
                }
            )
    out["geo_hold_end"] = geometry_snapshot(sim, R_nom, "hold_end")
    out["hold"] = hold
    out["table"] = t_table
    out["t_end"] = float(sim.data.time)
    out["held"] = bool(
        t_table is None
        and log[-1]["nL"] > 0
        and log[-1]["nR"] > 0
        and log[-1]["obj_z"] > 0.55
        and log[-1]["t"] >= t_hold0 + HOLD_S - 0.5
    )
    if hold:
        out["hold_drhx_mm"] = hold[-1]["rhx_mm"] - hold[0]["rhx_mm"]
        out["hold_vrel_mean"] = float(np.mean([h["vrel"] for h in hold]))
    return out


def _edge_call(g: dict) -> str:
    rim = g.get("from_plus_x_rim_mm")
    u = g.get("max_pad_u_edge")
    bits = []
    if rim is not None:
        if rim < 2.0:
            bits.append("COM within 2 mm of +x pad AABB rim")
        elif rim < 5.0:
            bits.append("COM near +x pad AABB (2–5 mm)")
        else:
            bits.append(f"COM {rim:.1f} mm inside +x pad AABB rim")
    if u is not None:
        if u > 0.85:
            bits.append(f"CoP u at {100*u:.0f}% of pad half-width (near pad-width edge)")
        elif u > 0.60:
            bits.append(f"CoP u at {100*u:.0f}% of pad half-width")
        else:
            bits.append(f"CoP u at {100*u:.0f}% of pad half-width (interior)")
    return "; ".join(bits) if bits else "insufficient contacts"


def write_report(rows: list[dict], verdict: str) -> None:
    p = OUT / "CONTROLLED_SLIP_SUFFICIENCY.md"
    a = []
    ap = a.append
    ap("# Controlled-slip sufficiency (CENTER-6)")
    ap("")
    ap("**THIS IS NOT HEURISTIC OPTIMIZATION.**")
    ap("")
    ap("One control variable: weak-slip **duration** at frozen `τ=-5`.")
    ap("The physical quantity is `Δr_h.x`. Duration is not a policy.")
    ap("")
    ap("## 1. Frozen mechanism")
    ap("")
    ap("- CENTER-6 ballistic translational progressive failure, EARLY intervention")
    ap("- secure +120° (`τ=-18`), weak `τ=-5`, brake `τ=-18` 0.40 s")
    ap("- corrected RETURN to saved `R_nominal = r_des` (not modified in this task)")
    ap("- noslip=1; no `v_x`/`v_z`; no impact/angle/τ change")
    ap("")
    ap("## 2. Small duration experiment")
    ap("")
    ap("| slip_s | Δr_h.x mm | start mm | end mm | lost | held | return nL/nR |")
    ap("|---:|---:|---:|---:|---|---|---|")
    for r in rows:
        g = r.get("geo_return") or r.get("geo_slip") or {}
        ap(
            f"| {r['slip_s']:.2f} | {r['drhx_mm']:.2f} | {r['rhx_slip_start_mm']:.2f} | "
            f"{r['rhx_slip_end_mm']:.2f} | {r.get('lost_at')} | {r.get('held')} | "
            f"{g.get('nL')}/{g.get('nR')} |"
        )
    ap("")
    ap("## 3. Actual Δr_h.x and slip kinematics")
    ap("")
    for r in rows:
        v = r["vrel_slip"]
        def mm_s(x):
            return "—" if x is None else f"{1e3 * float(x):.1f}"
        ap(
            f"- **{r['slip_s']:.2f} s**: Δr_h.x={r['drhx_mm']:.2f} mm; "
            f"|v_rel| mean={mm_s(v['mean'])} mm/s, max={mm_s(v['max'])} mm/s, p90={mm_s(v['p90'])} mm/s."
        )
    ap("")
    ap("## 4. Contact geometry after return")
    ap("")
    ap("| slip_s | r_h.x mm | rim+x mm | d_in mm | CoP u_L/u_R mm | u_edge | nL/nR | Fn L/R |")
    ap("|---:|---:|---:|---:|---|---:|---|---|")
    for r in rows:
        g = r.get("geo_return")
        if not g:
            ap(f"| {r['slip_s']:.2f} | — | — | — | lost | — | — | — |")
            continue
        t = g["tactile"]
        ap(
            f"| {r['slip_s']:.2f} | {g['rh_mm'][0]:.2f} | {g['from_plus_x_rim_mm']:.2f} | "
            f"{g['d_in_mm']:.2f} | {t['u_L_mm']}/{t['u_R_mm']} | {g['max_pad_u_edge']} | "
            f"{g['nL']}/{g['nR']} | {g['Fn_L']:.1f}/{g['Fn_R']:.1f} |"
        )
    ap("")
    ap("Pad AABB and per-contact positions: `raw/large_angle/slip_sufficiency.json`.")
    ap("Cylinder radius 18 mm, pad-1 half-width 8.5 mm (XML).")
    ap("")
    ap("## 5. Edge-catch audit")
    ap("")
    ap("Privileged `r_h` and contact.pos are used only as diagnostics. CoP `(u,v)` is the existing tactile observable.")
    ap("")
    for r in rows:
        g = r.get("geo_return") or r.get("geo_hold_end")
        ap(f"- **{r['slip_s']:.2f} s** post-return: {_edge_call(g) if g else r.get('lost_at')}")
        if g and g.get("contacts"):
            for c in g["contacts"][:8]:
                ap(
                    f"  - {c['side']} Fn={c['Fn']:.2f} N pos_h={np.round(c['pos_h_mm'],1).tolist()} mm "
                    f"cyl(r,z)={c['cyl_radial_mm']:.1f},{c['cyl_axial_mm']:.1f} mm pad_x_frac={c['pad_x_frac']:.2f}"
                )
    ap("")
    ap("## 6. Long-hold comparison")
    ap("")
    ap("Same horizon after RETURN (`HOLD_S+4`).")
    ap("")
    ap("| slip_s | held | hold Δr_h.x mm | hold |v_rel| mean | end r_h.x mm | end nL/nR |")
    ap("|---:|---|---:|---:|---:|---|")
    for r in rows:
        ge = r.get("geo_hold_end") or {}
        ap(
            f"| {r['slip_s']:.2f} | {r.get('held')} | {r.get('hold_drhx_mm')} | "
            f"{r.get('hold_vrel_mean')} | {None if not ge else ge['rh_mm'][0]:.2f} | "
            f"{None if not ge else str(ge['nL'])+'/'+str(ge['nR'])} |"
        )
    ap("")
    ap("## 7. Observable differences (no obs-vector change)")
    ap("")
    ap("Signals that can distinguish marginal vs more-supported states without privileged `r_h.x` as a policy target:")
    ap("")
    ap("- tactile CoP `u` (pad-width); large `|u|/8.5 mm` is an edge signature")
    ap("- bilateral `n` / Fn left-right balance")
    ap("- aperture")
    ap("- relative-motion estimates (if available to the policy)")
    ap("")
    ap("Do not treat privileged `r_h.x` as the learned-policy criterion.")
    ap("")
    ap("## 8. Mechanism-level conclusion")
    ap("")
    ap(verdict)
    ap("")
    ap("Even if one duration looks better, that is **not** “the policy should slip for that many seconds.”")
    ap("It only shows whether the validated mechanism has enough slip authority to change the support basin.")
    ap("")
    ap("## 9. Representative viewer commands")
    ap("")
    ap("```text")
    ap("python training/demo_ballistic_recovery.py --mode gravity_large_angle")
    if (RAW / "more_slip_plan.json").is_file():
        ap("python training/demo_ballistic_recovery.py --mode gravity_more_slip")
    ap("```")
    ap("")
    ap("No MP4. No SAC. No recatch. RETURN not modified.")
    p.write_text("\n".join(a), encoding="utf-8")
    print("wrote", p, flush=True)


def conclude(rows: list[dict]) -> str:
    base = next((r for r in rows if abs(r["slip_s"] - 0.70) < 1e-9), rows[0])
    others = [r for r in rows if r is not base]
    gb = base.get("geo_return")
    if gb is None:
        return "D. Baseline itself failed; not expected. Inspect raw."
    rim0 = float(gb["from_plus_x_rim_mm"])
    u0 = gb.get("max_pad_u_edge") or 0.0
    creep0 = base.get("hold_drhx_mm")
    lost_long = [r for r in others if r.get("lost_at")]
    better = []
    for r in others:
        g = r.get("geo_return")
        if not g or r.get("lost_at"):
            continue
        rim = float(g["from_plus_x_rim_mm"])
        u = g.get("max_pad_u_edge") or 0.0
        creep = r.get("hold_drhx_mm")
        moved = abs(r["drhx_mm"]) > abs(base["drhx_mm"]) + 0.4
        interior = (rim > rim0 + 1.5) or (u + 0.08 < u0)
        calmer = creep is not None and creep0 is not None and creep < creep0 - 0.3
        if moved and (interior or calmer) and r.get("held"):
            better.append(r)
    if lost_long and not better:
        return (
            f"**D.** Longer exposure approaches a loss boundary "
            f"(lost: {[r['slip_s'] for r in lost_long]}). "
            f"0.70 s remains the surviving short-slip arrest "
            f"(rim {rim0:.1f} mm, CoP u_edge={u0:.2f})."
        )
    if better:
        ds = ", ".join(f"{r['slip_s']:.2f}s Δ={r['drhx_mm']:.2f} mm" for r in better)
        return (
            f"**B.** Modestly more slip moves the object from a more-marginal support "
            f"into a more interior/robust geometry. Examples: {ds}. "
            f"Baseline 0.70 s: rim {rim0:.1f} mm, u_edge={u0:.2f}, hold Δr_h.x={creep0} mm. "
            "This is authority/sufficiency, not a recommended duration."
        )
    if u0 < 0.70 and rim0 > 5.0 and base.get("held"):
        return (
            f"**A.** 0.70 s is already physically well supported on pad metrics "
            f"(rim {rim0:.1f} mm from +x AABB, CoP u_edge={u0:.2f}). "
            "Visual ‘edge catch’ is not matched by these contact locations. "
            "Longer slip did not produce a clearly different basin."
        )
    return (
        f"**C.** More slip changes Δr_h.x but does not produce a meaningfully more robust "
        f"support geometry (baseline rim {rim0:.1f} mm, u_edge={u0:.2f}; "
        f"longer cases similar or mixed). No duration is selected as a heuristic."
    )


def headless_slip_sufficiency():
    LA_RAW.mkdir(parents=True, exist_ok=True)
    cfg = merge_sim_config(load_yaml(Path(__file__).resolve().parents[1] / "config" / "nominal.yaml"))
    gains = gains_from_cfg(cfg)
    snap = load_early()
    plan = json.loads((RAW / "large_angle_plan.json").read_text(encoding="utf-8"))
    rows = []
    for d in DURS:
        sim, _ = make_sim()
        r = run_duration(sim, gains, snap, plan, d)
        rows.append(r)
        print(
            f"slip {d:.2f}s drhx={r['drhx_mm']:.2f} mm lost={r.get('lost_at')} "
            f"held={r.get('held')} rim={None if not r.get('geo_return') else r['geo_return']['from_plus_x_rim_mm']}",
            flush=True,
        )
    dump(LA_RAW / "slip_sufficiency.json", rows)
    verdict = conclude(rows)
    # representative longer-slip viewer only if B and a survivor with more interior geometry
    base = next(r for r in rows if abs(r["slip_s"] - 0.70) < 1e-9)
    pick = None
    if verdict.startswith("**B"):
        cands = [
            r
            for r in rows
            if r.get("held")
            and abs(r["slip_s"] - 0.70) > 1e-9
            and r.get("geo_return")
            and float(r["geo_return"]["from_plus_x_rim_mm"])
            > float(base["geo_return"]["from_plus_x_rim_mm"]) + 1.0
        ]
        if cands:
            pick = min(cands, key=lambda r: r["slip_s"])
    more_p = RAW / "more_slip_plan.json"
    if pick is not None:
        mp = deepcopy(plan)
        mp["slip_s"] = pick["slip_s"]
        mp["note"] = "representative longer-slip case for viewer; not a heuristic"
        dump(more_p, mp)
        print("wrote more_slip_plan", pick["slip_s"], flush=True)
    elif more_p.is_file():
        more_p.unlink()
    write_report(rows, verdict)
    print(verdict, flush=True)
    return rows, verdict
