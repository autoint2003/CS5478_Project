"""Phase 1 — diagnose continuous-ok / M111-fail P34 P1 roots."""

from __future__ import annotations

import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.precision.p18_discrete_vocab_audit.replay_harness import (
    make_env,
    quantize_with_book,
    replay_actions,
)
from experiments.precision.p34_5_m111_precision_repair.io_util import utc, write_csv, write_md
from experiments.precision.p34_5_m111_precision_repair.paths import (
    HANDOFF_M111,
    M111_META,
    M111_REPLAY,
    OUT,
)
from experiments.precision.p34_precision_data_expansion.builder import (
    restore_default_dynamics,
    restore_ic,
    snapshot_default_dynamics,
)
from training.skill_aware_action_vocab import ACT_DIMS, DEFAULT_WEIGHTS, load_motion_codebook


def _load_fail_ok() -> tuple[list[dict], list[dict]]:
    rows = []
    with M111_REPLAY.open(encoding="utf-8") as f:
        import csv

        rows = list(csv.DictReader(f))
    fails = [r for r in rows if str(r.get("ok")).lower() not in ("1", "true")]
    oks = [r for r in rows if str(r.get("ok")).lower() in ("1", "true")]
    return fails, oks


def _phys_probe(env) -> dict[str, float]:
    sim = env.sim
    from evaluation.grasp_geometry import measure
    from evaluation.success_criteria import P_TGT, U_TGT, contact_summary

    cs = contact_summary(sim)
    try:
        m = measure(sim, U_TGT, P_TGT)
        e_pos = float(m.get("e_pos_mm") or 0.0)
        e_th = float(m.get("e_theta_deg") or 0.0)
        obj_z = float(m.get("obj_z") or sim.data.xpos[sim.ids.object_body][2])
    except Exception:
        e_pos = e_th = float("nan")
        obj_z = float(sim.data.xpos[sim.ids.object_body][2])
    return {
        "obj_z": obj_z,
        "bilateral": float(bool(cs.get("bilateral"))),
        "nL": float(cs.get("nL") or 0),
        "nR": float(cs.get("nR") or 0),
        "e_pos_mm": e_pos,
        "e_theta_deg": e_th,
        "time": float(sim.data.time),
    }


def _step_compare(
    env,
    capture,
    cont: np.ndarray,
    recon: np.ndarray,
    *,
    sample_every: int = 20,
) -> dict[str, Any]:
    """Find first action / physical divergence between cont and recon."""
    from envs.simulation.capture import restore_capture
    from envs.step_legal import step_legal
    from envs.observations.observable_obs import ObservableObsState
    from models.transformer_policy import DT_POLICY

    cont = np.asarray(cont, np.float32).reshape(-1, 7)
    recon = np.asarray(recon, np.float32).reshape(-1, 7)
    n = min(len(cont), len(recon))
    first_act = None
    for i in range(n):
        if float(np.max(np.abs(cont[i] - recon[i]))) > 1e-4:
            first_act = i
            break
    # dual rollouts for physical divergence
    restore_capture(env, capture)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    phys_c = []
    for i in range(n):
        step_legal(env.sim, cont[i])
        if i % sample_every == 0 or i == n - 1:
            p = _phys_probe(env)
            p["i"] = i
            phys_c.append(p)
    end_c = phys_c[-1]

    restore_capture(env, capture)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    phys_r = []
    first_phys = None
    for i in range(n):
        step_legal(env.sim, recon[i])
        if i % sample_every == 0 or i == n - 1:
            p = _phys_probe(env)
            p["i"] = i
            phys_r.append(p)
            if first_phys is None and len(phys_c) >= len(phys_r):
                pc = phys_c[len(phys_r) - 1]
                if abs(pc["obj_z"] - p["obj_z"]) > 5e-4 or abs(pc["bilateral"] - p["bilateral"]) > 0.5:
                    first_phys = i
    end_r = phys_r[-1]
    return {
        "first_action_div_i": first_act,
        "first_action_div_linf": float(np.max(np.abs(cont[first_act] - recon[first_act])))
        if first_act is not None
        else 0.0,
        "first_action_div_dims": [
            ACT_DIMS[j]
            for j in range(7)
            if first_act is not None and abs(float(cont[first_act, j] - recon[first_act, j])) > 1e-4
        ],
        "first_phys_div_i": first_phys,
        "end_cont_obj_z": end_c["obj_z"],
        "end_recon_obj_z": end_r["obj_z"],
        "end_dz": float(end_r["obj_z"] - end_c["obj_z"]),
        "end_cont_bilateral": end_c["bilateral"],
        "end_recon_bilateral": end_r["bilateral"],
    }


def diagnose_one(env, defaults, book, rho, row: dict, ref) -> dict[str, Any]:
    path = Path(row["ic_path"])
    if not path.is_absolute():
        from experiments.precision.p34_5_m111_precision_repair.paths import ROOT

        path = ROOT / path
    blob = pickle.loads(path.read_bytes())
    acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
    q = quantize_with_book(book, acts, rho, w=DEFAULT_WEIGHTS)
    recon = np.asarray(q["recon_actions"], np.float32).reshape(-1, 7)
    tokens = np.asarray(q["tokens"], np.int64).reshape(-1)
    err = recon - acts
    from training.skill_aware_action_vocab import ResidualBudget

    rho_a = ResidualBudget.from_spec(rho).as_array()
    beyond = np.maximum(np.abs(acts - book[tokens]) - rho_a, 0.0)

    # continuous + m111 STRICT
    hc = replay_actions(env, blob["capture"], blob["ref"], acts)
    hm = replay_actions(env, blob["capture"], blob["ref"], recon)

    # after m111 traj, is state handoff-ok?
    restore_ic(env, blob["capture"], defaults=defaults)
    from envs.simulation.capture import restore_capture
    from envs.step_legal import step_legal
    from envs.observations.observable_obs import ObservableObsState
    from models.transformer_policy import DT_POLICY
    from evaluation.axis_aware_success import evaluate_handoff_with_axis

    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    for a in recon:
        step_legal(env.sim, a)
    h_imm = evaluate_handoff_with_axis(env.sim, blob["ref"], stabilize_n=8)

    # first divergence — skip full dual sim on long tapes (STRICT pair already run)
    n = len(acts)
    if n > 400:
        first_act = None
        for i in range(n):
            if float(np.max(np.abs(acts[i] - recon[i]))) > 1e-4:
                first_act = i
                break
        div = {
            "first_action_div_i": first_act,
            "first_action_div_linf": float(np.max(np.abs(acts[first_act] - recon[first_act])))
            if first_act is not None
            else 0.0,
            "first_action_div_dims": [
                ACT_DIMS[j]
                for j in range(7)
                if first_act is not None
                and abs(float(acts[first_act, j] - recon[first_act, j])) > 1e-4
            ],
            "first_phys_div_i": None,
            "end_cont_obj_z": float("nan"),
            "end_recon_obj_z": float("nan"),
            "end_dz": float("nan"),
            "end_cont_bilateral": float("nan"),
            "end_recon_bilateral": float("nan"),
        }
    else:
        div = _step_compare(env, blob["capture"], acts, recon, sample_every=20)
    restore_default_dynamics(env.sim, defaults)

    sat_dims = [ACT_DIMS[i] for i in range(7) if float(beyond[:, i].max()) > 1e-8]
    per_dim_max = {ACT_DIMS[i]: float(np.max(np.abs(err[:, i]))) for i in range(7)}
    per_dim_mean = {ACT_DIMS[i]: float(np.mean(np.abs(err[:, i]))) for i in range(7)}
    grip_cont_min = float(np.min(acts[:, 6]))
    grip_recon_min = float(np.min(recon[:, 6]))

    # taxonomy label
    tax = "UNKNOWN"
    if not hc.get("SUCCESS_NEW"):
        tax = "CONT_ALSO_FAILS"  # reproducibility / impl issue
    elif hm.get("SUCCESS_NEW"):
        tax = "FALSE_NEGATIVE_AUDIT"
    elif hm.get("SUCCESS_12S") and not hm.get("SUCCESS_NEW"):
        tax = "BORDERLINE_HOLD"  # 12s ok, dwell/working hold fails
    elif float(per_dim_max.get("grip", 0)) > 0.05 and (
        not sat_dims or sat_dims == ["grip"] or "grip" in sat_dims
    ):
        tax = "GRIP_QUANTIZATION"
    elif any(d in sat_dims for d in ("vx", "vy", "vz", "wx", "wy", "wz")):
        tax = "MOTION_RESIDUAL_SATURATION"
    else:
        tax = "ACCUMULATED_STATE_DRIFT"

    if div["first_action_div_dims"] == ["grip"] or (
        div["first_action_div_dims"] and set(div["first_action_div_dims"]) <= {"grip"}
    ):
        if tax == "BORDERLINE_HOLD":
            tax = "GRIP_INDUCED_BORDERLINE_HOLD"
        elif tax == "ACCUMULATED_STATE_DRIFT":
            tax = "GRIP_QUANTIZATION"

    return {
        "tag": row.get("tag") or blob.get("tag"),
        "lineage_id": row.get("lineage_id") or blob.get("lineage_id"),
        "ic_path": str(path).replace("\\", "/"),
        "family": blob.get("family"),
        "regime_id": blob.get("regime_id"),
        "n_actions": int(n),
        "cont_SUCCESS_NEW": int(bool(hc.get("SUCCESS_NEW"))),
        "cont_SUCCESS_12S": int(bool(hc.get("SUCCESS_12S"))),
        "cont_reason": hc.get("reason"),
        "cont_pre_pos_mm": hc.get("pre_pos_mm"),
        "cont_pre_theta": hc.get("pre_theta"),
        "m111_SUCCESS_NEW": int(bool(hm.get("SUCCESS_NEW"))),
        "m111_SUCCESS_12S": int(bool(hm.get("SUCCESS_12S"))),
        "m111_reason": hm.get("reason"),
        "m111_pre_pos_mm": hm.get("pre_pos_mm"),
        "m111_pre_theta": hm.get("pre_theta"),
        "post_m111_imm_SUCCESS_NEW": int(bool(h_imm.get("SUCCESS_NEW"))),
        "post_m111_imm_SUCCESS_12S": int(bool(h_imm.get("SUCCESS_12S"))),
        "post_m111_imm_reason": h_imm.get("reason"),
        "clipped_frac": float(q.get("clipped_frac") or 0),
        "max_recon_wlinf": float(q.get("max_recon_wlinf") or 0),
        "mean_recon_wlinf": float(q.get("mean_recon_wlinf") or 0),
        "sat_dims": sat_dims,
        "per_dim_max_err": per_dim_max,
        "per_dim_mean_err": per_dim_mean,
        "grip_cont_min": grip_cont_min,
        "grip_recon_min": grip_recon_min,
        "grip_gap": float(grip_recon_min - grip_cont_min),
        "unique_tokens": int(len(set(tokens.tolist()))),
        "handoff_token_id": HANDOFF_M111,
        "taxonomy": tax,
        **{f"max_err_{d}": per_dim_max[d] for d in ACT_DIMS},
        **{f"mean_err_{d}": per_dim_mean[d] for d in ACT_DIMS},
        **div,
    }


def run_diagnosis(*, limit_fail: int | None = None, verify_ok_n: int = 10) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    fails, oks = _load_fail_ok()
    if limit_fail is not None:
        fails = fails[:limit_fail]
    cb = load_motion_codebook(M111_META)
    book = np.asarray(cb.prototypes, np.float32)
    rho = cb.residual
    env = make_env()
    defaults = snapshot_default_dynamics(env.sim)

    print(f"=== P34.5 diagnose fails={len(fails)} verify_ok={verify_ok_n} ===", flush=True)
    diag_rows = []
    for i, r in enumerate(fails):
        print(f"  fail {i+1}/{len(fails)} {r.get('tag')}", flush=True)
        diag_rows.append(diagnose_one(env, defaults, book, rho, r, None))

    # Quantization error table
    qerr = []
    for d in diag_rows:
        qerr.append(
            {
                "tag": d["tag"],
                "lineage_id": d["lineage_id"],
                "taxonomy": d["taxonomy"],
                **{f"max_err_{k}": d[f"max_err_{k}"] for k in ACT_DIMS},
                **{f"mean_err_{k}": d[f"mean_err_{k}"] for k in ACT_DIMS},
                "clipped_frac": d["clipped_frac"],
                "max_recon_wlinf": d["max_recon_wlinf"],
                "sat_dims": d["sat_dims"],
                "grip_cont_min": d["grip_cont_min"],
                "grip_recon_min": d["grip_recon_min"],
            }
        )
    write_csv(OUT / "QUANTIZATION_ERROR.csv", qerr)
    write_csv(OUT / "REPLAY_DIAGNOSTICS.csv", diag_rows)

    # Spot-check R0: original oks still succeed
    ok_verify = []
    for r in oks[:verify_ok_n]:
        path = Path(r["ic_path"])
        from experiments.precision.p34_5_m111_precision_repair.paths import ROOT

        if not path.is_absolute():
            path = ROOT / path
        blob = pickle.loads(path.read_bytes())
        acts = np.asarray(blob["actions"], np.float32).reshape(-1, 7)
        q = quantize_with_book(book, acts, rho, w=DEFAULT_WEIGHTS)
        hm = replay_actions(env, blob["capture"], blob["ref"], q["recon_actions"])
        ok_verify.append(
            {
                "tag": r.get("tag"),
                "m111_SUCCESS_NEW": int(bool(hm.get("SUCCESS_NEW"))),
                "m111_reason": hm.get("reason"),
            }
        )
        print(f"  ok-verify {r.get('tag')} NEW={hm.get('SUCCESS_NEW')}", flush=True)
    write_csv(OUT / "R0_OK_SPOTCHECK.csv", ok_verify)
    restore_default_dynamics(env.sim, defaults)

    tax = Counter(d["taxonomy"] for d in diag_rows)
    n_cont_fail = sum(1 for d in diag_rows if d["taxonomy"] == "CONT_ALSO_FAILS")
    n_impl = n_cont_fail + sum(1 for d in diag_rows if d["taxonomy"] == "FALSE_NEGATIVE_AUDIT")
    n_12_only = sum(1 for d in diag_rows if "BORDERLINE" in d["taxonomy"])
    n_grip = sum(1 for d in diag_rows if "GRIP" in d["taxonomy"])
    n_motion = sum(1 for d in diag_rows if d["taxonomy"] == "MOTION_RESIDUAL_SATURATION")
    n_drift = sum(1 for d in diag_rows if d["taxonomy"] == "ACCUMULATED_STATE_DRIFT")
    post_imm_ok = sum(1 for d in diag_rows if d["post_m111_imm_SUCCESS_NEW"])

    text = f"""# FAILURE_TAXONOMY — P34.5

Generated: {utc()}

## Cohort

- Continuous-ok / M111-fail P34 P1 roots diagnosed: **{len(diag_rows)}**
- Original M111-ok spot-check (R0 sample): {sum(1 for r in ok_verify if r['m111_SUCCESS_NEW'])}/{len(ok_verify)}
- Expected baseline from P34 audit: 85/123 M111 STRICT success

## Aggregate taxonomy

| Class | Count | Meaning |
|---|---:|---|
| CONT_ALSO_FAILS | {tax.get('CONT_ALSO_FAILS',0)} | Continuous replay also fails STRICT now (impl/repro) |
| FALSE_NEGATIVE_AUDIT | {tax.get('FALSE_NEGATIVE_AUDIT',0)} | M111 now succeeds (audit flake) |
| BORDERLINE_HOLD | {tax.get('BORDERLINE_HOLD',0)} | SUCCESS_12S but working-hold / SUCCESS_NEW fails |
| GRIP_INDUCED_BORDERLINE_HOLD | {tax.get('GRIP_INDUCED_BORDERLINE_HOLD',0)} | First action error is grip; ends as borderline hold |
| GRIP_QUANTIZATION | {tax.get('GRIP_QUANTIZATION',0)} | Grip residual saturation drives failure |
| MOTION_RESIDUAL_SATURATION | {tax.get('MOTION_RESIDUAL_SATURATION',0)} | Non-grip dims saturate residual ρ |
| ACCUMULATED_STATE_DRIFT | {tax.get('ACCUMULATED_STATE_DRIFT',0)} | Small per-step errors accumulate |
| **Total** | **{len(diag_rows)}** | |

Counts: implementation-related≈{n_impl}; representation-related≈{len(diag_rows)-n_impl}
(grip-related≈{n_grip}; borderline≈{n_12_only}; motion-sat≈{n_motion}; drift≈{n_drift})

## Key observations

1. P34 audit `m111_reason` often shows `SUCCESS_12S` even when `SUCCESS_NEW=0` — failures are frequently **dwell/working-hold** misses after a completed 12 s nominal, not hard drops.
2. Grip weight in nearest-prototype matching is 0.5 (lowest), so motion prototypes are selected for vx/vy/vz/wx first; grip is corrected only within residual ρ=0.35. Extreme expert grip commands (min≈−0.3…−0.6) often cannot be reconstructed.
3. After quantized replay, immediate HANDOFF SUCCESS_NEW on the terminal state: **{post_imm_ok}/{len(diag_rows)}** — many tapes never reach a STRICT-safe corrected state under M111.

## Artifacts

- `QUANTIZATION_ERROR.csv`
- `REPLAY_DIAGNOSTICS.csv`
- `R0_OK_SPOTCHECK.csv`
"""
    write_md(OUT / "FAILURE_TAXONOMY.md", text)
    print(text, flush=True)
    return {
        "n_fail": len(diag_rows),
        "taxonomy": dict(tax),
        "n_impl": n_impl,
        "n_repr": len(diag_rows) - n_impl,
        "ok_spot": sum(1 for r in ok_verify if r["m111_SUCCESS_NEW"]),
        "ok_spot_n": len(ok_verify),
        "diag_rows": diag_rows,
    }
