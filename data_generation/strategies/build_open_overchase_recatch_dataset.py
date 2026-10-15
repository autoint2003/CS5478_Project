"""Build the Open-overchase-close-brake recatch set.

The previous data/recatch tree is moved to data/regrasp. New demonstrations
are only trajectories that pass the physical recatch checks, official
SUCCESS_12S, a 5 s hold, and K48 residual within 0.25.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.strategies.recatch_oracle import _nominal_fail, _setup
from data_generation.strategies.capture_geometry import capture_geometry
from data_generation.oracle_search import representability
from envs.step_legal import step_legal
from training.recovery_runtime import apply_dyn, restore_ballistic
from envs.recovery_success import nominal_step
from envs.table_drop import latched, restore_state
from data_generation.strategies.recatch_oracle import _bottom
from data_generation.strategies.recatch_oracle import _finish, _impact
from data_generation.strategies.recatch_oracle import _base_spec, rollout, staged_choose
from training.unified_k47 import load_codebook

DATA = ROOT / "data" / "recatch"
DIAG = ROOT / "results" / "diagnostics" / "open_overchase_recatch_dataset"
REFERENCE = {"speed": 6.0, "pitch": 10.0, "axial": 0.0, "ecc": 0.0, "mass": None, "friction": None}


def _migrate():
    src = ROOT / "data" / "recatch"
    dst = ROOT / "data" / "regrasp"
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
    for path in (ROOT / "data").rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
            continue
        text = path.read_text(encoding="utf-8")
        if "data/recatch/" not in text:
            continue
        path.write_text(text.replace("data/recatch/", "data/regrasp/"), encoding="utf-8")


def _nominal_dynamics(sim):
    body = int(sim.ids.object_body)
    geom = int(sim.ids.object_geom)
    return (
        float(sim.model.body_mass[body]),
        np.array(sim.model.body_inertia[body], float).copy(),
        np.array(sim.model.geom_friction[geom], float).copy(),
    )


def _cells():
    rows = [dict(REFERENCE, id="ref_s6_p10_a0_e0", split="train", kind="reference")]
    for speed in (5.0, 6.0, 7.0):
        for pitch in (8.0, 15.0, 25.0):
            for axial in (0.0, 0.012, -0.012, 0.022):
                for ecc in (0.0, 0.008):
                    split = "valid" if speed == 5.0 else "train"
                    rows.append({
                        "speed": speed, "pitch": pitch, "axial": axial, "ecc": ecc,
                        "mass": None, "friction": None, "split": split, "kind": "impact",
                        "id": f"s{speed:.0f}_p{pitch:.0f}_a{axial:+.3f}_e{ecc:.3f}",
                    })
    rows.append({
        "speed": 5.0, "pitch": 15.0, "axial": 0.0, "ecc": 0.0,
        "mass": None, "friction": 0.7, "split": "valid", "kind": "friction",
        "id": "s5_p15_mu0.7",
    })
    rows.append({
        "speed": 7.0, "pitch": 15.0, "axial": 0.0, "ecc": 0.0,
        "mass": 0.16, "friction": None, "split": "train", "kind": "mass",
        "id": "s7_p15_m0.16",
    })
    rows.append({
        "speed": 6.0, "pitch": 10.0, "axial": 0.0, "ecc": 0.008,
        "mass": None, "friction": 1.3, "split": "train", "kind": "friction",
        "id": "ref_ecc_mu1.3",
    })
    return rows


def _specs():
    base = _base_spec()
    base.update({"stop_cz": 0.11, "close_dive": True, "close_tau": -18.0, "hold_tau": -18.0, "lift": 0.15, "settle_bottom": 0.56})
    shallow = _base_spec()
    shallow.update({"stop_cz": 0.106, "close_dive": False, "close_tau": -18.0, "hold_tau": -18.0, "brake": 8.0, "pad_steps": 1, "lift": 0.12, "open_gap": 0.048, "arrest": -1.0, "settle_bottom": 0.54})
    early = _base_spec()
    early.update({"stop_cz": 0.102, "close_dive": False, "close_tau": -18.0, "hold_tau": -12.0, "brake": 10.0, "pad_steps": 1, "lift": 0.10, "down": 2.6, "settle_bottom": 0.52})
    return [("reference", base), ("shallow_stop", shallow), ("early_brake", early)]


def _reapply(sim, nominal, cell):
    def hook():
        apply_dyn(sim, nominal, cell["mass"], cell["friction"])
    return hook


def _baseline(sim, snap, ref, nominal, cell):
    restore_ballistic(sim, snap)
    _reapply(sim, nominal, cell)()
    restore_state(sim, None)

    class Env:
        pass

    env = Env()
    env.sim = sim
    from envs.horizon_outcome import NOMINAL_S, nominal_verify

    out = nominal_verify(env, ref, seconds=NOMINAL_S)
    return {"ok": bool(out.get("success")), "why": out.get("reason"), "t": round(float(out.get("nominal_elapsed") or 0.0), 3)}


def _hold_ok(sim):
    t0 = float(sim.data.time)
    while float(sim.data.time) < t0 + 0.4:
        nominal_step(sim)
        geo = capture_geometry(sim)
        if int(geo["nL"]) == 0 or int(geo["nR"]) == 0 or _bottom(sim) < 0.50:
            return False
    return True


def _prepare(sim, base, nominal, cell):
    restore_ballistic(sim, base)
    restore_state(sim, None)
    apply_dyn(sim, nominal, cell["mass"], cell["friction"])
    if not _hold_ok(sim):
        apply_dyn(sim, nominal, None, None)
        return False
    return True


def _physical(metrics):
    if metrics.get("t_off") is None:
        return "no_release"
    if metrics.get("re_class") != "LOWER_VERTICAL_PAD":
        return "wrong_surface"
    if float(metrics.get("re_fn_slope") or 0.0) > 0.5:
        return "slope_force"
    if metrics.get("t_pad_hold") is None:
        return "contact_not_held"
    if metrics.get("table"):
        return "table"
    if metrics.get("t_vz0") is None:
        return "no_brake"
    if metrics.get("end_mode") != "handoff":
        return "no_handoff"
    if float(metrics.get("end_gap_mm") or 0.0) < 33.0:
        return "stays_squeezed"
    if float(metrics.get("end_bottom") or 0.0) < float(metrics.get("min_bottom") or 0.0) + 0.03:
        return "no_lift"
    if not (0.090 <= float(metrics.get("re_cz") or 0.0) <= 0.112):
        return "cz_outside"
    return "pass"


def _verify(sim, snap, result, ref, book, nominal):
    reason = _physical(result["metrics"])
    if reason != "pass":
        return reason, None
    actions = np.stack(result["actions"])
    if len(result["observations"]) != len(actions):
        return "obs_mismatch", None
    packed = representability(actions, book)
    if packed["max_linf"] is None or float(packed["max_linf"]) > 0.25 + 1e-8:
        return "k48_range", packed
    apply_dyn(sim, nominal, None, None)
    restore_ballistic(sim, snap)
    restore_state(sim, None)
    apply_dyn(sim, nominal, snap.get("mass"), snap.get("friction"))
    for action in packed["quantized"]:
        step_legal(sim, action)
    out, extra = _finish(sim, ref, extra=5.0)
    if not out.get("success"):
        return out.get("reason") or "nominal_fail", packed
    if extra != "HELD_EXTENSION":
        return extra or "extension_fail", packed
    return "ok", packed


def _different(actions, previous):
    if previous is None:
        return True
    n = min(len(actions), len(previous))
    if n < 5:
        return True
    return float(np.mean(np.abs(actions[:n] - previous[:n]))) > 0.04


def _family(cell):
    return f"v{cell['speed']:.0f}_p{cell['pitch']:.0f}_a{np.sign(cell['axial']):+.0f}_e{1 if cell['ecc'] else 0}_m{cell['mass']}_f{cell['friction']}"


def main():
    _migrate()
    DATA.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        (DATA / split).mkdir(exist_ok=True)
    DIAG.mkdir(parents=True, exist_ok=True)
    sim, base, ref, _target, _book = _setup()
    nominal = _nominal_dynamics(sim)
    book = load_codebook()
    specs = _specs()
    manifest = []
    boundary = []
    kept_actions = []
    for cell in _cells():
        if not _prepare(sim, base, nominal, cell):
            boundary.append({"id": cell["id"], "split": cell["split"], "class": "INVALID_GRASP"})
            print("INVALID", cell["id"], flush=True)
            continue
        seated = sim.snapshot() if hasattr(sim, "snapshot") else None
        info, err = _impact(sim, base, cell, after_restore=_reapply(sim, nominal, cell))
        apply_dyn(sim, nominal, cell["mass"], cell["friction"])
        if err:
            boundary.append({"id": cell["id"], "split": cell["split"], "class": err})
            print("MISS", cell["id"], err, flush=True)
            continue
        pose = info["pose"]
        slipped = float(pose["p_rel"][2] - info["before"]["p_rel"][2])
        fast = float(pose["v_obj"][2]) < -1.2 or pose["bottom"] < 0.48
        if fast:
            boundary.append({"id": cell["id"], "split": cell["split"], "class": "TOO_FAST", "dz": round(slipped, 4), "vz": round(float(pose["v_obj"][2]), 3)})
            print("FAST", cell["id"], round(float(pose["v_obj"][2]), 2), flush=True)
            continue
        if not info["progressive"] and slipped < 0.004 and pose["nL"] > 0 and pose["nR"] > 0:
            boundary.append({"id": cell["id"], "split": cell["split"], "class": "ABSORBED", "dz": round(slipped, 4)})
            print("ABSORB", cell["id"], round(slipped, 4), flush=True)
            continue
        info["snap"]["mass"] = cell["mass"]
        info["snap"]["friction"] = cell["friction"]
        apply_dyn(sim, nominal, cell["mass"], cell["friction"])
        nominal_out = _baseline(sim, info["snap"], ref, nominal, cell)
        apply_dyn(sim, nominal, cell["mass"], cell["friction"])
        print("BASE", cell["id"], nominal_out["why"], flush=True)
        if nominal_out["ok"]:
            boundary.append({"id": cell["id"], "split": cell["split"], "class": "TRIVIAL", "nominal": nominal_out["why"]})
            continue
        saved = None
        for name, spec in specs:
            apply_dyn(sim, nominal, cell["mass"], cell["friction"])
            result = rollout(sim, info["snap"], staged_choose(spec), 2.4, after_restore=_reapply(sim, nominal, cell))
            apply_dyn(sim, nominal, cell["mass"], cell["friction"])
            if saved is not None and not _different(np.stack(result["actions"]) if result["actions"] else np.zeros((1, 7)), saved):
                print(" ", name, "duplicate", flush=True)
                continue
            reason, packed = _verify(sim, info["snap"], result, ref, book, nominal)
            print(" ", name, reason, result["metrics"].get("end_mode"), result["metrics"].get("min_bottom"), flush=True)
            boundary.append({
                "id": cell["id"], "split": cell["split"], "strategy": name, "class": reason,
                "re_cz": result["metrics"].get("re_cz"), "bottom": result["metrics"].get("min_bottom"),
                "end_mode": result["metrics"].get("end_mode"),
            })
            if reason != "ok":
                continue
            actions = np.stack(result["actions"])
            stem = f"{cell['id']}__{name}"
            folder = DATA / cell["split"]
            snap_dir = ROOT / "data" / "snapshots" / cell["split"]
            snap_dir.mkdir(parents=True, exist_ok=True)
            tokens = np.asarray(packed["tokens"], np.int32)
            residual = (actions - book[tokens]).astype(np.float32)
            np.savez_compressed(
                folder / f"{stem}.npz",
                actions=actions.astype(np.float32),
                observations=np.asarray(result["observations"], np.float32),
                handoff_obs=np.asarray(result["handoff_obs"], np.float32),
            )
            np.savez_compressed(folder / f"{stem}__tokens.npz", tokens=tokens, residuals=residual)
            with (snap_dir / f"{cell['id']}.pkl").open("wb") as handle:
                pickle.dump(info["snap"], handle)
            manifest.append({
                "id": cell["id"],
                "split": cell["split"],
                "strategy": name,
                "family": "recatch",
                "impact_family": _family(cell),
                "mechanism": "open_overchase_close_brake",
                "continuous_npz": f"data/recatch/{cell['split']}/{stem}.npz",
                "token_residual_npz": f"data/recatch/{cell['split']}/{stem}__tokens.npz",
                "snapshot": f"data/snapshots/{cell['split']}/{cell['id']}.pkl",
                "source_snapshot": f"data/snapshots/{cell['split']}/{cell['id']}.pkl",
                "handoff_label": "SUCCESS_12S",
                "nominal_class": "RECOVERY_CANDIDATE",
                "provenance": "top_impact_open_overchase",
                "source_kind": "generated",
                "sample_weight": 1.0,
                "verified_success": True,
                "speed": cell["speed"],
                "pitch": cell["pitch"],
                "axial": cell["axial"],
                "ecc": cell["ecc"],
                "mass": cell["mass"],
                "friction": cell["friction"],
                "k48_linf": round(float(packed["max_linf"]), 4),
                "extension": "HELD_EXTENSION",
                "formal": "SUCCESS_12S",
                "dz": round(slipped, 4),
                "re_cz": result["metrics"].get("re_cz"),
                "min_bottom": result["metrics"].get("min_bottom"),
                "t_open": result["metrics"].get("t_open"),
                "t_re": result["metrics"].get("t_re"),
                "t_brake": result["metrics"].get("t_vz0"),
            })
            saved = actions
            kept_actions.append(actions)
            print("KEEP", stem, flush=True)
            if sum(1 for row in manifest if row["id"] == cell["id"]) >= 2:
                break
        del seated
    _write_outputs(manifest, boundary, kept_actions)
    print("DONE", len(manifest), flush=True)


def _write_outputs(manifest, boundary, kept_actions):
    with (DATA / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row) + "\n")
    with (DATA / "boundary.jsonl").open("w", encoding="utf-8") as handle:
        for row in boundary:
            handle.write(json.dumps(row) + "\n")
    splits = {name: sum(1 for row in manifest if row["split"] == name) for name in ("train", "valid")}
    families = {name: len({row.get("impact_family", row["family"]) for row in manifest if row["split"] == name}) for name in splits}
    near = 0
    for i, left in enumerate(kept_actions):
        for right in kept_actions[i + 1:]:
            n = min(len(left), len(right))
            if n > 5 and float(np.mean(np.abs(left[:n] - right[:n]))) < 0.02:
                near += 1
    summary = {
        "successes": len(manifest),
        "ics": len({row["id"] for row in manifest}),
        "splits": splits,
        "families": families,
        "near_duplicate_pairs": near,
        "boundary": {},
    }
    for row in boundary:
        summary["boundary"][row["class"]] = summary["boundary"].get(row["class"], 0) + 1
    (DIAG / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    entry_path = ROOT / "data" / "entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8")) if entry_path.exists() else {}
    entry.setdefault("categories", {})
    entry["categories"]["regrasp"] = "data/regrasp/manifest.jsonl"
    entry["categories"]["recatch"] = "data/recatch/manifest.jsonl"
    entry_path.write_text(json.dumps(entry, indent=2), encoding="utf-8")
    lines = [
        "# Open → Over-chase → Close → Brake recatch data",
        "",
        f"成功示范 {summary['successes']} 条，来自 {summary['ics']} 个冲击条件。",
        f"划分：train {splits['train']}，valid {splits['valid']}。",
        f"近重复动作对 {near}。",
        "",
        "边界计数：",
    ]
    for key, value in sorted(summary["boundary"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend([
        "",
        "成功判据包括真实上方撞击、名义失败、张开后在指尖竖直内侧面重新夹持、制动、上举、无桌面掉落、K48 残差不超过 0.25、`SUCCESS_12S` 和额外 5 s 保持。",
        "搜索用了接触分类。保存的 observation 是现有 airborne 观测，其中有相对位姿，没有接触面类别。",
        "旧的 `data/recatch` 已改名为 `data/regrasp`。",
    ])
    (DIAG / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
