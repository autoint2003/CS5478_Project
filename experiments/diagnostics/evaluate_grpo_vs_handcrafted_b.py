#!/usr/bin/env python
"""Evaluate historical GRPO recatch vs handcrafted Plan-B Stage-1.

Diagnostic only. Does not retrain GRPO, modify C experts, M111 vocab, physics,
or success criteria. Preserves all historical artifacts.

Results → results/diagnostics/grpo_vs_handcrafted_b/
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from controllers.recovery.geometry_switch import GeometrySwitch  # noqa: E402
from data_generation.oracles.plan_b.vertical_recatch import (  # noqa: E402
    make_env as make_vb_env,
)
from data_generation.oracles.plan_b.vertical_recatch import rollout as vb_rollout
from data_generation.oracles.plan_b.vertical_recatch import specs as vb_specs
from data_generation.oracles.precision.force_slip_control import sample_state  # noqa: E402
from data_generation.oracles.precision.precision_goals import (  # noqa: E402
    assess_need,
    entry_gate,
    measure_errors,
)
from data_generation.oracles.precision.precision_handoff import probe_direct_handoff  # noqa: E402
from data_generation.oracles.precision.precision_types import PrecisionConfig  # noqa: E402
from data_generation.oracles.precision.primitives import _wrist  # noqa: E402
from data_generation.oracles.precision.stable_grasp import hold_and_confirm_stable  # noqa: E402
from data_generation.oracles.precision.unified_precision_oracle import run_precision  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.horizon_outcome import recovery_failed  # noqa: E402
from envs.outcomes.support_drop import latched, terminal_reason  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.simulation.ground_sim import make_ground_sim  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from models.k48_checkpoint import HANDOFF, load_trained_k48  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k47_codebook import decode_action, load_codebook  # noqa: E402
from training.k48_config import SELECTED_LIMIT_S  # noqa: E402
from training.recovery_episode import make_training_env  # noqa: E402

DS = ROOT / "results/data_generation/stage2_abc_recovery_fullchain_v1"
HARD = ROOT / "results/diagnostics/hard_ic_mining"
OUT = ROOT / "results/diagnostics/grpo_vs_handcrafted_b"
TREE80 = ROOT / "results/grpo/tree80/checkpoint_epoch_0003.pt"
ADAPTIVE = ROOT / "results/grpo/adaptive_g/full_run_v2_natural_sampling/checkpoint_epoch_0005.pt"
DT = float(DT_POLICY)
SEED = 5478
LIMIT_S = float(SELECTED_LIMIT_S)
CAP_STEPS = 200


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


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def snap(sim) -> dict:
    err = measure_errors(sim)
    st = sample_state(sim)
    return {
        "bilateral": bool(err["bilateral"]),
        "e_pos_mm": float(err["e_pos_mm"]),
        "dtheta_deg": float(err["dtheta_deg"]),
        "dx_mm": float(err["dx_mm"]),
        "dy_mm": float(err["dy_mm"]),
        "dz_mm": float(err["dz_mm"]),
        "v_rel": float(st.get("v_rel_norm") or 0.0),
        "w_rel": float(st.get("w_rel_norm") or 0.0),
        "obj_z": float(err["obj_z"]),
        "fn_L": float(err["fn_L"]),
        "fn_R": float(err["fn_R"]),
        "latched": bool(latched(sim)),
    }


def make_plain_env():
    class E:
        pass

    env = E()
    env.sim, _ = make_ground_sim()
    env.switch = GeometrySwitch()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    return env


# ---------------------------------------------------------------------------
# IC pools (fixed; not selected by GRPO success)
# ---------------------------------------------------------------------------


def load_easy_b() -> list[dict]:
    rows = [
        json.loads(l)
        for l in (DS / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    out = []
    for r in rows:
        if r.get("source_label") != "B":
            continue
        if not (r.get("continuous_demo") or {}).get("verified"):
            continue
        ic = pickle.loads((DS / r["ic_pkl"]).read_bytes())
        cont = np.load(DS / r["continuous_demo"]["path"])
        acts = np.asarray(cont["actions"], np.float32)
        n_s1 = int(r.get("n_actions_stage1") or len(acts))
        out.append({
            "ic_id": r["ic_id"],
            "sample_id": r["sample_id"],
            "pool": "easy_fullchain_B",
            "capture": ic["capture"],
            "ref": ic["ref"],
            "handcrafted_actions": acts[:n_s1],
            "handcrafted_source": "fullchain_stage1_prefix",
        })
    return out


def load_hard_b(n: int = 20, seed: int = SEED) -> list[dict]:
    """Hard vertical_recatch Recovery ICs from mining — fixed shuffle, not GRPO-filtered."""
    paths = [
        HARD / "selected_train_manifest.jsonl",
        HARD / "selected_dev_manifest.jsonl",
        HARD / "selected_test_manifest.jsonl",
    ]
    cands = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cats = r.get("categories") or []
            if isinstance(cats, str):
                cats = [cats]
            mech = str(r.get("oracle_mechanism") or "")
            if "vertical_recatch" not in cats and "Release-Recatch" not in mech:
                continue
            snap_path = ROOT / r["snapshot"]
            if not snap_path.exists():
                continue
            cands.append(r)
    # Dedup by id
    by_id = {r["id"]: r for r in cands}
    ids = sorted(by_id)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    ids = ids[:n]
    out = []
    for i in ids:
        r = by_id[i]
        blob = pickle.loads((ROOT / r["snapshot"]).read_bytes())
        if "capture" not in blob or "ref" not in blob:
            continue
        out.append({
            "ic_id": r["id"],
            "sample_id": f"hardB__{r['id']}",
            "pool": "hard_vertical_recatch",
            "capture": blob["capture"],
            "ref": blob["ref"],
            "handcrafted_actions": None,  # run oracle at eval time
            "handcrafted_source": "vertical_recatch_search",
            "oracle_strategy": r.get("oracle_strategy"),
            "tier": r.get("tier"),
        })
    return out


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


def run_action_prefix(env, cap, ref, actions: np.ndarray) -> dict:
    restore_capture(env, cap)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    lost = 0
    failed = False
    reason = None
    both_off = False
    bil_after_off = False
    for a in actions:
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
        st = snap(env.sim)
        if not st["bilateral"]:
            both_off = True
        elif both_off:
            bil_after_off = True
        failed, lost = recovery_failed(env.sim, ref, lost)
        if failed:
            reason = terminal_reason(env.sim) or "DROP"
            break
    exit_cap = take_capture(env)
    exit_st = snap(env.sim)
    restore_capture(env, exit_cap)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, exit_cap)
    gate = entry_gate(env)
    restore_capture(env, exit_cap)
    return {
        "ok": not failed,
        "fail_reason": reason,
        "n_steps": int(len(actions)),
        "actions": np.asarray(actions, np.float32),
        "exit_capture": exit_cap,
        "exit_state": exit_st,
        "hold_stable": bool(stab.get("stable")),
        "gate_ok": bool(gate.get("ok")),
        "saw_both_off": both_off,
        "recatch_bilateral": bil_after_off or (exit_st["bilateral"] and both_off),
        "retained_bilateral": exit_st["bilateral"] and not both_off,
    }


def run_handcrafted_hard(vb_env, book, ic: dict) -> dict:
    blob = {"capture": ic["capture"], "ref": ic["ref"]}
    best = None
    for spec in vb_specs():
        res = vb_rollout(vb_env, blob, book, spec)
        acts = np.asarray(res["actions"], np.float32)
        # Evaluate exit without full finish first
        restore_capture(vb_env, ic["capture"])
        for a in acts:
            step_legal(vb_env.sim, a)
        exit_cap = take_capture(vb_env)
        exit_st = snap(vb_env.sim)
        restore_capture(vb_env, exit_cap)
        stab = hold_and_confirm_stable(vb_env, hold_s=0.20)
        restore_capture(vb_env, exit_cap)
        gate = entry_gate(vb_env)
        restore_capture(vb_env, exit_cap)
        summary = {
            "ok": res["mode"] == "handoff" and bool(exit_st["bilateral"]),
            "fail_reason": None if res["mode"] == "handoff" else res["mode"],
            "n_steps": int(len(acts)),
            "actions": acts,
            "exit_capture": exit_cap,
            "exit_state": exit_st,
            "hold_stable": bool(stab.get("stable")),
            "gate_ok": bool(gate.get("ok")),
            "spec": spec["name"],
            "mode": res["mode"],
            "saw_both_off": True,  # vertical recatch is release-recatch by design
            "recatch_bilateral": bool(exit_st["bilateral"]),
            "retained_bilateral": False,
        }
        # Prefer SUCCESS-capable exits via Direct probe later; pick best bilateral+handoff
        score = (
            int(summary["ok"]),
            int(summary["hold_stable"]),
            int(summary["gate_ok"]),
            -float(exit_st["e_pos_mm"]),
        )
        if best is None or score > best[0]:
            best = (score, summary)
    if best is not None:
        return best[1]
    restore_capture(vb_env, ic["capture"])
    return {
        "ok": False,
        "fail_reason": "no_spec",
        "n_steps": 0,
        "actions": np.zeros((0, 7), np.float32),
        "exit_capture": ic["capture"],
        "exit_state": snap(vb_env.sim),
        "hold_stable": False,
        "gate_ok": False,
        "saw_both_off": False,
        "recatch_bilateral": False,
        "retained_bilateral": False,
    }


def run_grpo_stage1(env, model, book, device, ic: dict) -> dict:
    """Argmax K48 until HANDOFF / limit / fail. Capture exit before nominal."""
    blob = {"capture": ic["capture"], "ref": ic["ref"]}
    restore_capture(env, blob["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._obs_seq = []
    env._prev_seq = []
    env._prev = np.zeros(7, np.float32)
    t0 = float(env.sim.data.time)
    lost = 0
    tokens = []
    actions = []
    both_off = False
    bil_after = False
    how = None
    reason = None
    for _ in range(CAP_STEPS):
        if float(env.sim.data.time) - t0 >= LIMIT_S - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _ = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        tokens.append(token)
        if token == HANDOFF:
            how = "handoff"
            break
        if token < 0 or token >= HANDOFF:
            how = "illegal_token"
            reason = f"token_{token}"
            break
        _res, action = decode_action(book, token, mean[-1].detach().float().cpu().numpy())
        executed = np.asarray(step_legal(env.sim, action), np.float32)
        env._prev = executed
        actions.append(executed)
        st = snap(env.sim)
        if not st["bilateral"]:
            both_off = True
        elif both_off:
            bil_after = True
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        if failed:
            how = "fail"
            reason = terminal_reason(env.sim) or "DROP"
            break
    exit_cap = take_capture(env)
    exit_st = snap(env.sim)
    restore_capture(env, exit_cap)
    stab = hold_and_confirm_stable(env, hold_s=0.20)
    restore_capture(env, exit_cap)
    gate = entry_gate(env)
    restore_capture(env, exit_cap)
    ok = how == "handoff" and bool(exit_st["bilateral"]) and not latched(env.sim)
    return {
        "ok": ok,
        "fail_reason": reason if how == "fail" else (None if ok else how),
        "how": how,
        "n_steps": len(actions),
        "n_tokens_incl_handoff": len(tokens),
        "tokens": tokens,
        "actions": np.asarray(actions, np.float32) if actions else np.zeros((0, 7), np.float32),
        "exit_capture": exit_cap,
        "exit_state": exit_st,
        "hold_stable": bool(stab.get("stable")),
        "gate_ok": bool(gate.get("ok")),
        "saw_both_off": both_off,
        "recatch_bilateral": bil_after,
        "retained_bilateral": bool(exit_st["bilateral"]) and not both_off,
        "handoff_token_emitted": how == "handoff",
    }


def probe_exit(env, exit_cap, ref) -> dict:
    restore_capture(env, exit_cap)
    err = measure_errors(env.sim)
    need = assess_need(err, PrecisionConfig())
    direct = probe_direct_handoff(env, exit_cap, ref, stabilize_n=8)
    return {
        "need_dtheta": bool(need.get("dtheta_needed")),
        "need_dx": bool(need.get("dx_needed")),
        "need_dz": bool(need.get("dz_needed")),
        "need_ready": bool(need.get("ready_for_direct_handoff_assessment")),
        "direct_NEW": bool(direct.get("SUCCESS_NEW")),
        "direct_12S": bool(direct.get("SUCCESS_12S")),
        "direct_reason": direct.get("reason"),
    }


def run_precision_from_exit(env, exit_cap, ref, family: str = "B") -> dict:
    cfg = PrecisionConfig()
    res = run_precision(env, {"capture": exit_cap, "ref": ref}, family, cfg)
    h = res.handoff or {}
    return {
        "precision_executed": bool(res.precision_executed),
        "precision_skipped": bool(res.precision_skipped),
        "precision_status": res.status,
        "n_precision_actions": len(res.actions or []),
        "phases": list(res.phases or []),
        "prec_NEW": bool(h.get("SUCCESS_NEW")),
        "prec_12S": bool(h.get("SUCCESS_12S")),
        "prec_reason": h.get("std_reason") or h.get("reason"),
        "failures": list(res.failures or [])[:6],
    }


def flatten_stage1(controller: str, ic: dict, s1: dict, probe: dict) -> dict:
    es = s1.get("exit_state") or {}
    return {
        "controller": controller,
        "ic_id": ic["ic_id"],
        "sample_id": ic["sample_id"],
        "pool": ic["pool"],
        "stage1_ok": s1.get("ok"),
        "fail_reason": s1.get("fail_reason"),
        "how": s1.get("how"),
        "n_steps": s1.get("n_steps"),
        "hold_stable": s1.get("hold_stable"),
        "gate_ok": s1.get("gate_ok"),
        "saw_both_off": s1.get("saw_both_off"),
        "recatch_bilateral": s1.get("recatch_bilateral"),
        "retained_bilateral": s1.get("retained_bilateral"),
        "bilateral": es.get("bilateral"),
        "e_pos_mm": es.get("e_pos_mm"),
        "dtheta_deg": es.get("dtheta_deg"),
        "v_rel": es.get("v_rel"),
        "w_rel": es.get("w_rel"),
        "obj_z": es.get("obj_z"),
        "fn_L": es.get("fn_L"),
        "fn_R": es.get("fn_R"),
        **probe,
        "spec": s1.get("spec"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard-n", type=int, default=20)
    ap.add_argument("--skip-adaptive", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # Phase 1 inventory
    inventory = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {
            "tree80_e3": {
                "path": str(TREE80.relative_to(ROOT)).replace("\\", "/"),
                "exists": TREE80.exists(),
                "role": "tree-augmented GRPO; primary historical recatch candidate",
            },
            "adaptive_e5": {
                "path": str(ADAPTIVE.relative_to(ROOT)).replace("\\", "/"),
                "exists": ADAPTIVE.exists(),
                "role": "Independent-G natural sampling; later hard-IC GRPO",
            },
        },
        "interface": {
            "obs": "39D obs_gt_airborne_39 (no extra privileged channel beyond MuJoCo GT object-rel)",
            "action": "K48 token 0-46 + residual±0.25 → legal 7D; HANDOFF=47",
            "horizon": "recovery_limit_s=4.0 then nominal 12s",
            "compatible_with_current_recovery": True,
            "note": "Uses K48 not M111/K49; legal via same step_legal / AIRBORNE_V3 map",
        },
        "usable_without_retrain": TREE80.exists(),
    }
    atomic_json(OUT / "PHASE1_GRPO_INVENTORY.json", inventory)

    if not TREE80.exists():
        raise SystemExit(f"missing {TREE80}")

    print("Loading policies…", flush=True)
    model_t, book = load_trained_k48(TREE80, device)
    model_a = None
    if not args.skip_adaptive and ADAPTIVE.exists():
        model_a, _ = load_trained_k48(ADAPTIVE, device)

    env_pol = make_training_env(device, model=model_t)
    env_plain = make_plain_env()
    vb_env = make_vb_env()
    book48 = load_codebook()

    easy = load_easy_b()
    hard = load_hard_b(n=args.hard_n, seed=SEED)
    ics = easy + hard
    atomic_json(
        OUT / "EVAL_ICS.json",
        {
            "seed": SEED,
            "n_easy": len(easy),
            "n_hard": len(hard),
            "easy_ids": [x["ic_id"] for x in easy],
            "hard_ids": [x["ic_id"] for x in hard],
            "selection": "fixed seed shuffle of vertical_recatch hard manifests; not GRPO-success filtered",
        },
    )
    print(f"ICs: easy={len(easy)} hard={len(hard)}", flush=True)

    stage1_rows = []
    integration_rows = []
    candidates = []

    controllers = [("handcrafted_B", None), ("grpo_tree80_e3", model_t)]
    if model_a is not None:
        controllers.append(("grpo_adaptive_e5", model_a))

    for ci, ic in enumerate(ics):
        print(f"\n=== [{ci+1}/{len(ics)}] {ic['sample_id']} ({ic['pool']}) ===", flush=True)
        s1_by = {}

        # Handcrafted
        t0 = time.time()
        if ic["handcrafted_actions"] is not None:
            s1_h = run_action_prefix(env_plain, ic["capture"], ic["ref"], ic["handcrafted_actions"])
            s1_h["spec"] = "stored_prefix"
        else:
            s1_h = run_handcrafted_hard(vb_env, book48, ic)
        ph = probe_exit(env_plain, s1_h["exit_capture"], ic["ref"]) if s1_h.get("exit_capture") else {}
        row_h = flatten_stage1("handcrafted_B", ic, s1_h, ph)
        row_h["wall_s"] = time.time() - t0
        stage1_rows.append(row_h)
        s1_by["handcrafted_B"] = (s1_h, ph)
        print(f"  handcrafted ok={s1_h.get('ok')} steps={s1_h.get('n_steps')} direct_NEW={ph.get('direct_NEW')}", flush=True)

        # GRPO policies
        for cname, model in controllers[1:]:
            t0 = time.time()
            env_pol.model = model  # observation path independent of model weights
            # ensure policy env uses same model for forward — roll uses passed model
            s1_g = run_grpo_stage1(env_pol, model, book, device, ic)
            pg = probe_exit(env_plain, s1_g["exit_capture"], ic["ref"])
            row_g = flatten_stage1(cname, ic, s1_g, pg)
            row_g["wall_s"] = time.time() - t0
            stage1_rows.append(row_g)
            s1_by[cname] = (s1_g, pg)
            print(
                f"  {cname} ok={s1_g.get('ok')} how={s1_g.get('how')} steps={s1_g.get('n_steps')} "
                f"both_off={s1_g.get('saw_both_off')} recatch={s1_g.get('recatch_bilateral')} "
                f"direct_NEW={pg.get('direct_NEW')}",
                flush=True,
            )

            # Phase 3: integration from GRPO exit when Stage-1 produced a usable capture
            if s1_g.get("exit_capture") is None:
                continue
            if s1_g.get("how") == "fail" and not s1_g.get("exit_state", {}).get("bilateral"):
                integration_rows.append({
                    "controller": cname,
                    "ic_id": ic["ic_id"],
                    "sample_id": ic["sample_id"],
                    "pool": ic["pool"],
                    "branch": "skipped_failed_exit",
                    "stage1_ok": False,
                })
                continue

            # A) Direct (already in pg)
            # B) Precision from same exit
            t1 = time.time()
            prec = run_precision_from_exit(env_plain, s1_g["exit_capture"], ic["ref"], family="B")
            integ = {
                "controller": cname,
                "ic_id": ic["ic_id"],
                "sample_id": ic["sample_id"],
                "pool": ic["pool"],
                "stage1_ok": s1_g.get("ok"),
                "stage1_how": s1_g.get("how"),
                "n_steps": s1_g.get("n_steps"),
                "saw_both_off": s1_g.get("saw_both_off"),
                "recatch_bilateral": s1_g.get("recatch_bilateral"),
                "exit_bilateral": (s1_g.get("exit_state") or {}).get("bilateral"),
                "exit_e_pos": (s1_g.get("exit_state") or {}).get("e_pos_mm"),
                "exit_theta": (s1_g.get("exit_state") or {}).get("dtheta_deg"),
                "exit_v_rel": (s1_g.get("exit_state") or {}).get("v_rel"),
                "hold_stable": s1_g.get("hold_stable"),
                "gate_ok": s1_g.get("gate_ok"),
                "direct_NEW": pg.get("direct_NEW"),
                "direct_12S": pg.get("direct_12S"),
                "direct_reason": pg.get("direct_reason"),
                "need_any": bool(pg.get("need_dtheta") or pg.get("need_dx") or pg.get("need_dz")),
                **prec,
                "wall_s_precision": time.time() - t1,
            }
            # Multi-stage candidate: GRPO restored contact, Direct fails, Precision succeeds
            integ["multi_stage_candidate"] = bool(
                (s1_g.get("recatch_bilateral") or s1_g.get("ok"))
                and (s1_g.get("exit_state") or {}).get("bilateral")
                and not pg.get("direct_NEW")
                and prec.get("prec_NEW")
            )
            integ["direct_success_no_precision_needed"] = bool(pg.get("direct_NEW"))
            integration_rows.append(integ)
            if integ["multi_stage_candidate"]:
                candidates.append(integ)
                print(f"  ** MULTI-STAGE CANDIDATE **", flush=True)
            elif pg.get("direct_NEW"):
                print(f"  Direct SUCCESS (Precision not forced)", flush=True)
            else:
                print(
                    f"  Direct fail / Prec NEW={prec.get('prec_NEW')} status={prec.get('precision_status')}",
                    flush=True,
                )

        write_csv(OUT / "STAGE1_COMPARISON.csv", stage1_rows)
        write_csv(OUT / "GRPO_PRECISION_INTEGRATION.csv", integration_rows)

    # Summaries
    def summarize(controller: str, pool: str | None = None):
        xs = [r for r in stage1_rows if r["controller"] == controller]
        if pool:
            xs = [r for r in xs if r["pool"] == pool]
        n = len(xs)
        if n == 0:
            return {}
        return {
            "n": n,
            "stage1_ok": sum(1 for r in xs if r["stage1_ok"] in (True, "True")),
            "direct_NEW": sum(1 for r in xs if r.get("direct_NEW") in (True, "True")),
            "direct_12S": sum(1 for r in xs if r.get("direct_12S") in (True, "True")),
            "recatch_bilateral": sum(1 for r in xs if r.get("recatch_bilateral") in (True, "True")),
            "saw_both_off": sum(1 for r in xs if r.get("saw_both_off") in (True, "True")),
            "mean_n_steps": float(np.mean([float(r["n_steps"] or 0) for r in xs])),
            "mean_e_pos": float(np.mean([float(r["e_pos_mm"]) for r in xs if r.get("e_pos_mm") not in (None, "")])),
        }

    phase2 = {
        "overall": {c: summarize(c) for c, _ in controllers},
        "easy": {c: summarize(c, "easy_fullchain_B") for c, _ in controllers},
        "hard": {c: summarize(c, "hard_vertical_recatch") for c, _ in controllers},
    }
    atomic_json(OUT / "PHASE2_SUMMARY.json", phase2)

    # Failure diagnosis on GRPO→Precision fails
    fail_rows = [
        r for r in integration_rows
        if r.get("direct_NEW") in (False, "False", False)
        and r.get("prec_NEW") in (False, "False", False)
        and r.get("branch") != "skipped_failed_exit"
    ]
    diagnosis = []
    for r in fail_rows:
        causes = []
        if r.get("stage1_ok") in (False, "False"):
            causes.append("grpo_stage1_not_ok")
        if r.get("exit_bilateral") in (False, "False"):
            causes.append("no_bilateral_at_exit")
        if r.get("hold_stable") in (False, "False") or r.get("gate_ok") in (False, "False"):
            causes.append("unstable_exit_gate")
        if r.get("need_any") in (False, "False") and "NO_REMAINING_CORRECTION" in str(r.get("phases")):
            causes.append("precision_no_remaining_correction_in_band_fragile")
        if r.get("need_any") in (True, "True") and r.get("prec_NEW") in (False, "False"):
            causes.append("precision_controller_failed_needed_correction")
        if r.get("exit_v_rel") not in (None, "") and float(r["exit_v_rel"]) > 0.08:
            causes.append("residual_velocity_high")
        if not causes:
            causes.append("unclassified")
        diagnosis.append({**{k: r[k] for k in (
            "controller", "ic_id", "pool", "direct_reason", "prec_reason", "precision_status", "phases", "n_steps"
        ) if k in r}, "causes": causes})
    atomic_json(OUT / "PHASE4_FAILURE_DIAGNOSIS.json", {"n": len(diagnosis), "rows": diagnosis})
    write_csv(OUT / "MULTI_STAGE_CANDIDATES.csv", candidates)
    atomic_json(OUT / "MULTI_STAGE_CANDIDATES.json", candidates)

    # Recommendation
    h_easy = phase2["easy"].get("handcrafted_B", {})
    g_easy = phase2["easy"].get("grpo_tree80_e3", {})
    h_hard = phase2["hard"].get("handcrafted_B", {})
    g_hard = phase2["hard"].get("grpo_tree80_e3", {})
    n_multi = len(candidates)

    if g_hard.get("direct_NEW", 0) > h_hard.get("direct_NEW", 0) + 2 and n_multi >= 3:
        rec = "supplement"
        rec_detail = (
            "GRPO recovers more hard Direct successes and yields multi-stage candidates; "
            "keep handcrafted for easy B Direct demos; use GRPO (or GRPO→Precision) to expand hard B."
        )
    elif g_hard.get("direct_NEW", 0) >= h_hard.get("direct_NEW", 0) and g_easy.get("direct_NEW", 0) >= h_easy.get("direct_NEW", 0) - 1:
        if n_multi >= 1:
            rec = "supplement"
            rec_detail = "GRPO competitive; multi-stage candidates exist — supplement handcrafted B, do not replace yet."
        else:
            rec = "retain_handcrafted"
            rec_detail = (
                "GRPO does not clearly beat handcrafted on matched ICs and produced no "
                "GRPO→Precision multi-stage successes; retain handcrafted B; optional GRPO as research probe only."
            )
    elif g_hard.get("direct_NEW", 0) + 2 < h_hard.get("direct_NEW", 0):
        rec = "retain_handcrafted"
        rec_detail = "Handcrafted vertical_recatch outperforms historical GRPO on hard ICs; do not replace."
    else:
        rec = "supplement"
        rec_detail = "Mixed results — keep handcrafted; consider GRPO only where it uniquely recovers contact."

    recommendation = {
        "recommendation": rec,
        "detail": rec_detail,
        "phase2": phase2,
        "n_multi_stage_candidates": n_multi,
        "do_not": [
            "do not train new GRPO yet",
            "do not modify C experts / M111 / physics / SUCCESS_*",
            "do not start SFT/RL packing",
        ],
    }
    atomic_json(OUT / "RECOMMENDATION.json", recommendation)

    md = []
    md.append("# GRPO Recatch vs Handcrafted Plan-B — Evaluation Report")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("")
    md.append("## Phase 1 — Historical GRPO")
    md.append("")
    md.append("- **Policy:** K48 token+residual (±0.25) → legal 7D; HANDOFF=47; 39D airborne obs.")
    md.append("- **Primary ckpt:** `results/grpo/tree80/checkpoint_epoch_0003.pt` (usable, no retrain).")
    md.append("- **Secondary:** Adaptive-G E5 if present.")
    md.append("- **Privileged:** no extra oracle channel beyond MuJoCo GT object-relative (formal 39D).")
    md.append("- Historical SUCCESS metrics ≠ full-chain Stage-2 expert validity.")
    md.append("")
    md.append("## Phase 2 — Matched Stage-1 comparison")
    md.append("")
    md.append("### Easy fullchain B ICs")
    md.append("")
    md.append("| Controller | n | Stage-1 ok | Direct NEW | Direct 12S | Recatch bil | mean steps |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for c, _ in controllers:
        s = phase2["easy"].get(c, {})
        md.append(
            f"| {c} | {s.get('n')} | {s.get('stage1_ok')} | {s.get('direct_NEW')} | "
            f"{s.get('direct_12S')} | {s.get('recatch_bilateral')} | {s.get('mean_n_steps')} |"
        )
    md.append("")
    md.append("### Hard vertical_recatch ICs (fixed seed, not GRPO-filtered)")
    md.append("")
    md.append("| Controller | n | Stage-1 ok | Direct NEW | Direct 12S | Recatch bil | mean steps |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for c, _ in controllers:
        s = phase2["hard"].get(c, {})
        md.append(
            f"| {c} | {s.get('n')} | {s.get('stage1_ok')} | {s.get('direct_NEW')} | "
            f"{s.get('direct_12S')} | {s.get('recatch_bilateral')} | {s.get('mean_n_steps')} |"
        )
    md.append("")
    md.append("## Phase 3 — GRPO → Precision")
    md.append("")
    md.append(f"- Multi-stage candidates (Direct fail ∧ Precision SUCCESS_NEW): **{n_multi}**")
    md.append(f"- See `MULTI_STAGE_CANDIDATES.csv` / `GRPO_PRECISION_INTEGRATION.csv`")
    md.append("")
    md.append("## Phase 4 — Failures")
    md.append("")
    md.append(f"- GRPO exit with Direct fail & Precision fail: {len(diagnosis)}")
    cause_c = Counter()
    for d in diagnosis:
        for c in d.get("causes") or []:
            cause_c[c] += 1
    for k, v in cause_c.most_common():
        md.append(f"- `{k}`: {v}")
    md.append("")
    md.append("## Phase 5 — Recommendation")
    md.append("")
    md.append(f"**{rec}** — {rec_detail}")
    md.append("")
    md.append("No new GRPO training. No C/M111/physics changes. No SFT packing.")
    (OUT / "REPORT.md").write_text("\n".join(md), encoding="utf-8")
    print("\nDONE", rec, "multi=", n_multi, flush=True)


if __name__ == "__main__":
    main()
