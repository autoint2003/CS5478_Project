"""Closed-loop eval with shared WORKING HANDOFF (learned HANDOFF masked)."""

from __future__ import annotations

import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from envs.observations.airborne_obs import observe_airborne
from envs.observations.observable_obs import ObservableObsState
from envs.outcomes.support_drop import latched
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.axis_aware_success import evaluate_handoff_with_axis
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start
from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index
from experiments.precision.p27_history_multiskill_transformer.run_p27 import load_d2_eval_eps
from experiments.precision.p28_multiskill_distillation.harness import policy_step_K
from experiments.precision.p37_working_handoff_d.handoff_rules import (
    init_labels,
    working_geometry,
)
from experiments.precision.p38_unified_motion_only_sft.paths import (
    OUT,
    P21_MAN,
    P23_MAN,
    P35,
    P35_MAN,
    ROOT,
)
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111
from models.transformer_policy import DT_POLICY
from training.m111_dataset import WINDOW as K_WIN

MAX_STEPS = 2500
EVAL_OUT = OUT / "eval"


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)

    def cell(v):
        if isinstance(v, (list, dict, tuple)):
            return json.dumps(v)
        return v

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: cell(r.get(k)) for k in keys})


def abs_path(p) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


def rollout_working(
    env,
    model,
    book,
    rho,
    *,
    restore_fn: Callable,
    ref,
    device,
    max_steps: int = MAX_STEPS,
    meta: dict | None = None,
) -> dict[str, Any]:
    restore_fn()
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    labels = init_labels(env.sim)  # AXIS+WORKING init flags (WORKING is what matters)
    init_wk = working_geometry(env.sim)
    obs_seq: list = []
    prev_seq: list = []
    n_motion = 0
    termination = "timeout"
    drop = False
    handoff_step = None
    ever_working = bool(init_wk["working_ok"])
    reach_step = 0 if ever_working else None

    for t in range(max_steps):
        if latched(env.sim):
            drop = True
            termination = "drop"
            break
        wk = working_geometry(env.sim)
        if wk["working_ok"]:
            ever_working = True
            if reach_step is None:
                reach_step = t
            handoff_step = t
            termination = "rule_handoff"
            break

        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action, _ = policy_step_K(model, book, rho, obs_seq, prev_seq, device, K_WIN)
        if action is None or tok == HANDOFF:
            with torch.no_grad():
                logits, resid, _ = model(
                    torch.as_tensor(np.stack(obs_seq[-K_WIN:]), device=device),
                    torch.as_tensor(np.stack(prev_seq[-K_WIN:]), device=device),
                )
                logits = logits[-1].clone()
                logits[HANDOFF] = -1e9
                tok = int(torch.argmax(logits).item())
                rr = resid[-1].float().cpu().numpy().astype(np.float32)
                action = decode_m111_action(book, tok, rr, rho=rho)

        step_legal(env.sim, action)
        env._prev = action
        n_motion += 1

    intended = termination == "rule_handoff"
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8) if (intended or True) else {}
    # Always run STRICT probe for diagnostics; count SUCCESS only if intended handoff
    if not intended and not drop:
        # still evaluate current state for raw flags
        pass
    s12 = bool(hand.get("SUCCESS_12S"))
    s_new = bool(hand.get("SUCCESS_NEW"))
    strict = bool(intended and s_new)
    win = hand.get("window_SUCCESS_NEW") or {}

    if strict:
        fail = "SUCCESS"
    elif drop:
        fail = "DROP"
    elif termination == "timeout":
        fail = "TIMEOUT_NO_WORKING"
    elif intended and s12 and not s_new:
        fail = "WORKING_HOLD_FAIL"
    elif intended and not s12:
        fail = f"POST_HANDOFF:{hand.get('reason')}"
    else:
        fail = f"OTHER:{termination}"

    return {
        **(meta or {}),
        "init_working_satisfied": int(init_wk["working_ok"]),
        "init_theta": init_wk["dtheta_deg"],
        "init_e_pos": init_wk["e_pos_mm"],
        "init_axis_satisfied": labels.get("init_axis_satisfied"),
        "n_motion": n_motion,
        "termination": termination,
        "handoff": int(intended),
        "ever_working": int(ever_working),
        "reach_step": reach_step,
        "SUCCESS_12S": int(s12) if intended else 0,
        "SUCCESS_NEW": int(strict),
        "SUCCESS_NEW_raw": int(s_new),
        "timeout": int(termination == "timeout"),
        "drop": int(drop),
        "fail_category": fail,
        "eval_reason": hand.get("reason"),
        "win_mean_pos": win.get("mean_pos"),
        "win_mean_theta": win.get("mean_theta"),
        "nontrivial": int(not init_wk["working_ok"]),
    }


def load_abc_val_eps() -> list[dict]:
    man = json.loads(P21_MAN.read_text(encoding="utf-8"))
    index = build_start_index(man)
    out = []
    for r in man["trajectories"]:
        if r["split"] != "val":
            continue
        fam = r["family"]
        if fam not in ("A", "B", "C"):
            continue
        skill = {"A": "A1", "B": "B1", "C": "C"}[fam]
        ep = index[r["traj_id"]]
        out.append({"skill": skill, "traj_id": r["traj_id"], "split": "val", "ep": ep, "suite": skill})
    return out


def load_d2_val() -> list[dict]:
    man = json.loads(P23_MAN.read_text(encoding="utf-8"))
    d2 = [r for r in man["trajectories"] if r.get("family") == "D2"]
    eps = load_d2_eval_eps({"d2": d2})
    out = []
    for ep in eps:
        out.append(
            {
                "skill": "D2",
                "suite": "D2",
                "traj_id": ep.get("traj_id") or ep.get("lineage_id") or "d2",
                "split": "val",
                "ep": ep,
            }
        )
    return out


def load_d_nontrivial_test() -> list[dict]:
    """P36/P37 D nontrivial test (11): P1 test not initially AXIS-satisfied ≈ nontrivial."""
    from experiments.precision.p37_working_handoff_d.handoff_rules import axis_geometry

    env = make_env()
    rows = [
        r
        for r in csv.DictReader(P35_MAN.open(encoding="utf-8"))
        if r.get("class") == "P1" and r.get("split") == "test"
    ]
    out = []
    for r in rows:
        blob = pickle.loads(Path(r["ic_path"]).read_bytes())
        restore_capture(env, blob["capture"])
        ax = axis_geometry(env.sim)
        if ax["axis_ok"]:
            continue  # nontrivial = not AXIS-sat (P36/P37 definition)
        out.append(
            {
                "skill": "D",
                "suite": "D_nt_test",
                "traj_id": r["traj_id"],
                "split": "test",
                "capture": blob["capture"],
                "ref": blob["ref"],
            }
        )
    return out


def eval_checkpoint(ckpt: Path, device, book, rho, *, tag: str) -> list[dict]:
    model, _, _ = load_m111(ckpt, device)
    env = make_env()
    jobs = load_abc_val_eps() + load_d2_val() + load_d_nontrivial_test()
    rows = []
    for i, job in enumerate(jobs):
        if "ep" in job:
            ep = job["ep"]

            def restore(ep=ep):
                # D2 eps may use capture key; A/B/C use restore_start (capture may be null)
                cap = ep.get("capture")
                if cap is not None:
                    restore_capture(env, cap)
                    env._hist = ObservableObsState(DT_POLICY)
                    env._prev = np.zeros(7, np.float32)
                else:
                    restore_start(env, ep)

            ref = ep["ref"]
        else:

            def restore(cap=job["capture"]):
                restore_capture(env, cap)

            ref = job["ref"]

        row = rollout_working(
            env,
            model,
            book,
            rho,
            restore_fn=restore,
            ref=ref,
            device=device,
            meta={
                "tag": tag,
                "checkpoint": str(ckpt.as_posix()),
                "skill": job["skill"],
                "suite": job["suite"],
                "traj_id": job["traj_id"],
                "split": job["split"],
            },
        )
        rows.append(row)
        print(
            f"  [{tag}] {i+1}/{len(jobs)} {job['suite']} {str(job['traj_id'])[:36]} "
            f"NEW={row['SUCCESS_NEW']} wk0={row['init_working_satisfied']} cat={row['fail_category']}",
            flush=True,
        )
    del model
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for suite in sorted({r["suite"] for r in rows}):
        for nontrivial_only in (False, True):
            sub = [r for r in rows if r["suite"] == suite]
            if nontrivial_only:
                sub = [r for r in sub if int(r["nontrivial"]) == 1]
            n = len(sub)
            out.append(
                {
                    "suite": suite,
                    "nontrivial_only": int(nontrivial_only),
                    "n": n,
                    "WORKING_reach": sum(int(r["ever_working"]) for r in sub),
                    "HANDOFF": sum(int(r["handoff"]) for r in sub),
                    "SUCCESS_12S": sum(int(r["SUCCESS_12S"]) for r in sub),
                    "SUCCESS_NEW": sum(int(r["SUCCESS_NEW"]) for r in sub),
                    "timeout": sum(int(r["timeout"]) for r in sub),
                    "drop": sum(int(r["drop"]) for r in sub),
                    "init_working": sum(int(r["init_working_satisfied"]) for r in sub),
                    "fail_cats": dict(Counter(r["fail_category"] for r in sub)),
                }
            )
    return out


def eval_all_kinds(device, book, rho, kinds=("E0", "E1", "E2")) -> dict:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    for kind in kinds:
        ckpt = OUT / "checkpoints" / kind / "checkpoint_best.pt"
        if not ckpt.is_file():
            print(f"skip missing {ckpt}", flush=True)
            continue
        print(f"=== Eval {kind} {ckpt} ===", flush=True)
        rows = eval_checkpoint(ckpt, device, book, rho, tag=kind)
        write_csv(EVAL_OUT / f"ROLLOUTS_{kind}.csv", rows)
        summ = summarize(rows)
        for s in summ:
            s["kind"] = kind
        summaries.extend(summ)
        all_rows.extend(rows)
    write_csv(EVAL_OUT / "SUMMARY.csv", summaries)
    write_csv(EVAL_OUT / "ALL_ROLLOUTS.csv", all_rows)
    return {"rows": all_rows, "summary": summaries}
