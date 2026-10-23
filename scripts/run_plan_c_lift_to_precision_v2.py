#!/usr/bin/env python
"""Plan C: catch-secure IC → grip settle → controlled lift → Precision eval.

Writes results/data_generation/stage2_abc_final_v2/ (does not overwrite final v1).
Lift is Stage-1 provenance; Stage-2 Precision starts at the post-lift IC.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch
from data_generation.oracles.precision.primitives import _wrist
from data_generation.oracles.precision.stable_grasp import hold_and_confirm_stable
from data_generation.oracles.precision.unified_precision_oracle import (
    PrecisionConfig,
    entry_gate,
    measure_errors,
    probe_direct_handoff,
    run_precision,
)
from envs.observations.airborne_obs import observe_airborne
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.ground_drop import latched
from envs.simulation.capture import restore_capture, take_capture
from envs.simulation.ground_sim import make_ground_sim
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from models.transformer_policy import DT_POLICY
from training.action_vocab_k49 import CAP, HANDOFF, build_k49_table

V2_MINING = ROOT / "results/data_generation/stage2_abc_mining/v2"
V1_MINING = ROOT / "results/data_generation/stage2_abc_mining/v1"
OUT = ROOT / "results/data_generation/stage2_abc_final_v2"
K49_CONFIG = "K49_tok48_wy0p25"
# Verified A/B airborne operating region (object world z).
TARGET_OBJ_Z = 0.52
LIFT_VZ = 0.12
LIFT_MAX_S = 8.0
DEDUP = 1e-3


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


def atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsafe(obj), indent=2), encoding="utf-8")
    tmp.replace(path)


def make_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    env._obs_seq = []
    env._prev_seq = []
    return env


def capture_fp(cap):
    q = np.asarray(cap["physics"]["qpos"], float)
    return hashlib.sha256(np.round(q / DEDUP).astype(np.int64).tobytes()).hexdigest()[:16]


def world_state(env):
    err = measure_errors(env.sim)
    return {
        "obj_z": float(env.sim.data.xpos[env.sim.ids.object_body][2]),
        "hand_z": float(env.sim.data.xpos[env.sim.ids.hand_body][2]),
        "e_pos_mm": float(err["e_pos_mm"]),
        "dtheta_deg": float(err["dtheta_deg"]),
        "e_signed_mm": [float(x) for x in err["e_signed_mm"]],
        "bilateral": bool(err["bilateral"]),
        "fn_L": float(err["fn_L"]),
        "fn_R": float(err["fn_R"]),
        "v_rel_norm": float(err["v_rel_norm"]),
        "w_rel_norm": float(err["w_rel_norm"]),
        "ground_drop": bool(latched(env.sim)),
        "alpha_up_deg": float(err.get("alpha_up_deg") or 0.0),
    }


def grip_settle(env, hold_s: float = 0.20) -> list[np.ndarray]:
    n = max(1, int(round(hold_s / 0.02)))
    acts = []
    for _ in range(n):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        step_legal(env.sim, a)
        acts.append(a.copy())
    return acts


def controlled_airborne_lift(
    env,
    *,
    target_obj_z: float = TARGET_OBJ_Z,
    vz_up: float = LIFT_VZ,
    max_s: float = LIFT_MAX_S,
    grip: float = 1.0,
) -> dict:
    """Legal 7D world-+z lift from current capture state (no teleport)."""
    from data_generation.oracles.plan_c.catch_stabilization import _world_up_to_action

    t0 = float(env.sim.data.time)
    pre = world_state(env)
    actions: list[np.ndarray] = []
    series = []
    reached = False
    fail = None
    while float(env.sim.data.time) - t0 < max_s - 1e-9:
        st = world_state(env)
        if st["ground_drop"]:
            fail = "GROUND_DROP"
            break
        if not st["bilateral"] or st["fn_L"] < 1.0 or st["fn_R"] < 1.0:
            fail = "CONTACT_LOSS"
            break
        if st["obj_z"] >= target_obj_z - 0.02:
            reached = True
            break
        a = np.asarray(_world_up_to_action(env.sim, float(vz_up), grip=float(grip)), np.float32).reshape(7)
        actions.append(a.copy())
        step_legal(env.sim, a)
        if len(actions) % 5 == 0:
            series.append(world_state(env))
    # Stabilization hold
    for _ in range(15):
        a = np.zeros(7, np.float32)
        a[6] = 1.0
        actions.append(a.copy())
        step_legal(env.sim, a)
        st = world_state(env)
        if st["ground_drop"] or not st["bilateral"]:
            fail = fail or ("GROUND_DROP" if st["ground_drop"] else "CONTACT_LOSS_HOLD")
            break
    post = world_state(env)
    if fail is None and not (post["bilateral"] and post["fn_L"] >= 1.0 and post["fn_R"] >= 1.0):
        fail = "UNSTABLE_AFTER_LIFT"
    if fail is None and post["obj_z"] < target_obj_z - 0.05:
        fail = "BELOW_WORKING_HEIGHT"
        if not reached:
            fail = "LIFT_TIMEOUT_BELOW_TARGET"
    return {
        "ok": fail is None and post["obj_z"] >= target_obj_z - 0.05,
        "fail_reason": fail,
        "reached_target_z": bool(reached or post["obj_z"] >= target_obj_z - 0.02),
        "n_actions": len(actions),
        "actions": actions,
        "pre": pre,
        "post": post,
        "series_tail": series[-8:],
        "target_obj_z": target_obj_z,
        "vz_up": vz_up,
    }


def observe(env):
    return observe_airborne(
        env.sim.model, env.sim.data, env.sim.ids, getattr(env.sim, "fsm", None), env._hist
    )


def replay_cont(env, cap, ref, actions):
    restore_capture(env, cap)
    env._hist = ObservableObsState(DT_POLICY)
    obs = []
    for a in actions:
        obs.append(observe(env))
        step_legal(env.sim, np.asarray(a, np.float32))
    for _ in range(8):
        step_legal(env.sim, _wrist(0, 0, 1))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "SUCCESS_NEW": bool(h.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
        "reason": h.get("std_reason") or h.get("reason"),
        "observations": np.asarray(obs, np.float32) if obs else np.zeros((0, 39), np.float32),
    }


def quantize_and_replay_k49(env, cap, ref, actions, table):
    acts = [np.asarray(a, np.float32).reshape(7) for a in actions]
    if not acts:
        return {
            "verified": False,
            "nontrivial_representability": False,
            "tokens": np.zeros(0, np.int64),
            "residuals": np.zeros((0, 7), np.float32),
            "max_recon_l2": 0.0,
            "residual_clipped_frac": 0.0,
            "reason": "empty",
        }
    toks, res_a = [], []
    clipped = 0
    max_l2 = 0.0
    rows = np.asarray(table, np.float32)
    for a in acts:
        d = np.linalg.norm(rows[:, :7] - a.reshape(1, 7), axis=1)
        t = int(np.argmin(d))
        proto = rows[t, :7]
        raw = a - proto
        r = np.clip(raw, -CAP, CAP).astype(np.float32)
        if np.any(np.abs(raw) > CAP + 1e-8):
            clipped += 1
        max_l2 = max(max_l2, float(np.linalg.norm(a - (proto + r))))
        toks.append(t)
        res_a.append(r)
    restore_capture(env, cap)
    for i, t in enumerate(toks):
        a = np.clip(table[int(t)] + np.clip(res_a[i], -CAP, CAP), -1, 1).astype(np.float32)
        step_legal(env.sim, a)
    for _ in range(8):
        step_legal(env.sim, _wrist(0, 0, 1))
    h = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=0)
    return {
        "verified": bool(h.get("SUCCESS_NEW")),
        "nontrivial_representability": bool(h.get("SUCCESS_NEW")) and len(acts) > 0,
        "tokens": np.asarray(toks, np.int64),
        "residuals": np.asarray(res_a, np.float32),
        "max_recon_l2": max_l2,
        "residual_clipped_frac": float(clipped / len(acts)),
        "reason": h.get("std_reason") or h.get("reason"),
        "SUCCESS_12S": bool(h.get("SUCCESS_12S")),
    }


def evaluate_post_lift(env, table, cfg, *, sample_id, catch_id, cap_lift, ref, lift_meta, lift_actions):
    restore_capture(env, cap_lift)
    gate = entry_gate(env)
    restore_capture(env, cap_lift)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, cap_lift)
    err = measure_errors(env.sim)
    row = {
        "sample_id": sample_id,
        "plan_provenance": "C",
        "source_kind": "plan_c_post_lift_v2",
        "catch_sample_id": catch_id,
        "lift": {k: lift_meta[k] for k in lift_meta if k != "actions"},
        "lift_n_actions": len(lift_actions),
        "errors_entry": err,
        "gate_ok": bool(gate.get("ok")),
        "stable": bool(stab.get("stable")),
    }
    if not (gate.get("ok") and stab.get("stable")):
        row["sft_category"] = "C_POST_LIFT_UNSUITABLE"
        row["status"] = "POST_LIFT_GATE_FAIL"
        return row

    direct = probe_direct_handoff(env, cap_lift, ref, stabilize_n=8)
    restore_capture(env, cap_lift)
    t0 = time.time()
    res = run_precision(env, {"capture": cap_lift, "ref": ref}, "C", cfg)
    wall = time.time() - t0
    restore_capture(env, cap_lift)
    prec_new = bool((res.handoff or {}).get("SUCCESS_NEW")) if res.handoff else False
    actions = list(res.actions or [])
    if direct.get("SUCCESS_NEW") and len(actions) == 0:
        cat = "DIRECT_HANDOFF"
    elif prec_new and not direct.get("SUCCESS_NEW"):
        cat = "NONTRIVIAL_PRECISION"
    elif prec_new:
        cat = "DIRECT_HANDOFF"
    else:
        cat = "FAILED_ORACLE"
    row.update(
        {
            "status": "OK",
            "sft_category": cat,
            "direct": {
                "SUCCESS_NEW": bool(direct.get("SUCCESS_NEW")),
                "SUCCESS_12S": bool(direct.get("SUCCESS_12S")),
                "reason": direct.get("reason"),
            },
            "precision": {
                "SUCCESS_NEW": prec_new,
                "SUCCESS_12S": bool((res.handoff or {}).get("SUCCESS_12S")) if res.handoff else False,
                "n_actions": len(actions),
                "phases": list(res.phases or []),
                "failures": list(res.failures or []),
                "status": res.status,
                "wall_s": round(wall, 2),
            },
            "capture_fp": capture_fp(cap_lift),
        }
    )

    # Save IC + lift tape (Stage-1)
    ic_pkl = OUT / "ics" / f"{sample_id}.pkl"
    ic_pkl.parent.mkdir(parents=True, exist_ok=True)
    ic_pkl.write_bytes(pickle.dumps({"capture": cap_lift, "ref": ref, "catch_sample_id": catch_id}))
    lift_path = OUT / "lift" / f"{sample_id}__lift.npz"
    lift_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(lift_path, actions=np.asarray(lift_actions, np.float32))
    row["ic_pkl"] = str(ic_pkl.relative_to(OUT))
    row["lift_npz"] = str(lift_path.relative_to(OUT))
    row["stage1_lift_provenance"] = True
    row["stage2_starts_at_post_lift_ic"] = True

    if cat in ("DIRECT_HANDOFF", "NONTRIVIAL_PRECISION"):
        crep = replay_cont(env, cap_lift, ref, actions)
        cpath = OUT / "continuous" / f"{sample_id}__continuous.npz"
        cpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cpath, actions=np.asarray(actions, np.float32), observations=crep["observations"]
        )
        row["continuous_demo"] = {
            "path": str(cpath.relative_to(OUT)),
            "n_actions": len(actions),
            "verified": bool(crep["SUCCESS_NEW"]),
            "SUCCESS_12S": bool(crep["SUCCESS_12S"]),
            "reason": crep["reason"],
        }
        k = quantize_and_replay_k49(env, cap_lift, ref, actions, table)
        tpath = OUT / "tokens" / f"{sample_id}__k49.npz"
        tpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            tpath,
            tokens=k["tokens"],
            residuals=k["residuals"],
            handoff_token=np.asarray([HANDOFF], np.int64),
            continuous_actions=np.asarray(actions, np.float32),
            cap=np.asarray([CAP], np.float32),
        )
        row["k49_demo"] = {
            "path": str(tpath.relative_to(OUT)),
            "n_tokens": int(len(k["tokens"])),
            "verified": k["verified"],
            "nontrivial_representability": k["nontrivial_representability"] and len(actions) > 0,
            "max_recon_l2": k["max_recon_l2"],
            "residual_clipped_frac": k["residual_clipped_frac"],
            "reason": k["reason"],
            "config_id": K49_CONFIG,
        }
    else:
        fpath = OUT / "failures" / f"{sample_id}__failure.json"
        atomic_json(
            fpath,
            {
                "sample_id": sample_id,
                "failures": (row.get("precision") or {}).get("failures"),
                "phases": (row.get("precision") or {}).get("phases"),
                "status": (row.get("precision") or {}).get("status"),
                "lift_post": lift_meta.get("post"),
            },
        )
        row["failure_diag"] = str(fpath.relative_to(OUT))
    return row


def main():
    for sub in ("ics", "lift", "continuous", "tokens", "outcomes", "failures", "videos", "diagnostics"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    env = make_env()
    table = build_k49_table(config_id=K49_CONFIG)
    cfg = PrecisionConfig()

    # Collect C catch ICs from mining v2
    catch_rows = []
    for p in sorted((V2_MINING / "outcomes").glob("C__*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        ic = V2_MINING / "ics" / f"{o['sample_id']}.pkl"
        if ic.exists():
            catch_rows.append(o)

    funnel = Counter()
    eval_rows = []
    lift_report_rows = []

    print(f"Plan-C lift-to-Precision: {len(catch_rows)} catch ICs", flush=True)
    for o in catch_rows:
        catch_id = o["sample_id"]
        funnel["catch_total"] += 1
        pack = pickle.loads((V2_MINING / "ics" / f"{catch_id}.pkl").read_bytes())
        cap0, ref = pack["capture"], pack["ref"]
        restore_capture(env, cap0)
        pre_catch = world_state(env)

        # Grip settle (Stage-1)
        settle_acts = grip_settle(env, 0.20)
        after_settle = world_state(env)
        if after_settle["ground_drop"] or not after_settle["bilateral"]:
            funnel["settle_fail"] += 1
            lift_report_rows.append(
                {
                    "catch_sample_id": catch_id,
                    "status": "C_SETTLE_FAILURE",
                    "pre_catch": pre_catch,
                    "after_settle": after_settle,
                }
            )
            continue

        lift = controlled_airborne_lift(env, target_obj_z=TARGET_OBJ_Z, vz_up=LIFT_VZ)
        lift_acts = settle_acts + list(lift["actions"])
        lift_meta = {k: v for k, v in lift.items() if k != "actions"}
        lift_meta["settle_n"] = len(settle_acts)
        lift_meta["pre_catch"] = pre_catch
        lift_meta["after_settle"] = after_settle

        if not lift["ok"]:
            funnel["C_LIFT_FAILURE"] += 1
            funnel[f"lift_fail:{lift['fail_reason']}"] += 1
            lift_report_rows.append(
                {
                    "catch_sample_id": catch_id,
                    "status": "C_LIFT_FAILURE",
                    "fail_reason": lift["fail_reason"],
                    **{k: lift_meta[k] for k in ("pre", "post", "pre_catch", "reached_target_z", "n_actions")},
                }
            )
            sid = f"C_LIFTFAIL__{catch_id}"
            atomic_json(OUT / "failures" / f"{sid}.json", lift_report_rows[-1])
            print(f"  LIFT_FAIL {catch_id} {lift['fail_reason']} z {pre_catch['obj_z']:.3f}->{lift['post']['obj_z']:.3f}", flush=True)
            continue

        funnel["lift_ok"] += 1
        cap_lift = take_capture(env)
        fp = capture_fp(cap_lift)
        sample_id = f"C__postlift__{catch_id.split('__')[1] if '__' in catch_id else catch_id}__{fp}"

        # Fair paired eval from post-lift IC
        row = evaluate_post_lift(
            env,
            table,
            cfg,
            sample_id=sample_id,
            catch_id=catch_id,
            cap_lift=cap_lift,
            ref=ref,
            lift_meta=lift_meta,
            lift_actions=lift_acts,
        )
        funnel[row.get("sft_category", "?")] += 1
        if (row.get("k49_demo") or {}).get("nontrivial_representability"):
            funnel["NONTRIVIAL_K49"] += 1
        if (row.get("continuous_demo") or {}).get("verified") and row.get("sft_category") == "NONTRIVIAL_PRECISION":
            funnel["NONTRIVIAL_CONT"] += 1
        atomic_json(OUT / "outcomes" / f"{sample_id}.json", row)
        eval_rows.append(row)
        lift_report_rows.append(
            {
                "catch_sample_id": catch_id,
                "post_lift_sample_id": sample_id,
                "status": "LIFT_OK",
                "pre_catch_obj_z": pre_catch["obj_z"],
                "post_lift_obj_z": lift["post"]["obj_z"],
                "post_lift_hand_z": lift["post"]["hand_z"],
                "sft_category": row.get("sft_category"),
                "direct_NEW": (row.get("direct") or {}).get("SUCCESS_NEW"),
                "prec_NEW": (row.get("precision") or {}).get("SUCCESS_NEW"),
                "k49_NEW": (row.get("k49_demo") or {}).get("verified"),
            }
        )
        print(
            f"  [{row.get('sft_category')}] {sample_id} z={lift['post']['obj_z']:.3f} "
            f"direct={row['direct'].get('SUCCESS_NEW')} prec={row['precision'].get('SUCCESS_NEW')} "
            f"n={row['precision'].get('n_actions')}",
            flush=True,
        )

    atomic_json(OUT / "C_POST_LIFT_EVALUATION.json", {"funnel": dict(funnel), "rows": eval_rows})
    atomic_json(OUT / "diagnostics/c_lift_rows.json", {"rows": lift_report_rows, "funnel": dict(funnel)})

    # Markdown lift report
    ab_mean = 0.515
    md = f"""# Plan C Lift-to-Precision Report

Created: {datetime.now(timezone.utc).isoformat()}

## Working height target

Verified A/B Precision entry object world-z ≈ **{ab_mean:.3f} m** (hand ≈ 0.61 m).  
Plan-C catch-secure ICs typically sit at object z ≈ **0.20–0.40 m**.

Lift target: object world-z ≥ **{TARGET_OBJ_Z:.2f} m** via legal 7D world-+z (`vz={LIFT_VZ}`), grip=1.0, then 0.30 s hold.

Lift is **Stage-1** provenance. Stage-2 Precision / SFT starts at the **post-lift IC**.

## Funnel

```json
{json.dumps(dict(funnel), indent=2)}
```

## Counts

- Catch ICs screened: {funnel.get('catch_total', 0)}
- Lift OK: {funnel.get('lift_ok', 0)}
- Lift failures: {funnel.get('C_LIFT_FAILURE', 0)}
- Post-lift DIRECT_HANDOFF: {funnel.get('DIRECT_HANDOFF', 0)}
- Post-lift NONTRIVIAL continuous: {funnel.get('NONTRIVIAL_CONT', 0)} / category NONTRIVIAL: {funnel.get('NONTRIVIAL_PRECISION', 0)}
- Post-lift K49 nontrivial: {funnel.get('NONTRIVIAL_K49', 0)}
- Post-lift FAILED_ORACLE: {funnel.get('FAILED_ORACLE', 0)}
- Post-lift unsuitable gate: {funnel.get('C_POST_LIFT_UNSUITABLE', 0)}

## Notes

- Original catch ICs preserved under mining v2; this tree does not overwrite `stage2_abc_final/`.
- Fair Direct vs Precision comparison uses the **same post-lift capture**.
"""
    (OUT / "C_LIFT_REPORT.md").write_text(md, encoding="utf-8")
    print("Done", dict(funnel), "->", OUT, flush=True)
    return dict(funnel)


if __name__ == "__main__":
    # Ensure _action_vw exists or fallback
    try:
        from data_generation.oracles.precision.primitives import _action_vw  # noqa: F401
    except ImportError:
        pass
    main()
