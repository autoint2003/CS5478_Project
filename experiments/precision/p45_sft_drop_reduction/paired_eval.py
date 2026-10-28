"""P45 Phase 4 — Paired closed-loop eval on identical P44 holdout ICs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p28_multiskill_distillation.harness import policy_step_K  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META  # noqa: E402
from experiments.precision.p39_working_hold_handoff.expert_validate import collect_jobs  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from experiments.precision.p45_sft_drop_reduction.paths import OUT, P44_CKPT, P44_EVAL, SEED, T_MAX  # noqa: E402
from models.m111_checkpoint import HANDOFF, decode_m111_action, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import WINDOW as K_WIN  # noqa: E402


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def rollout(env, model, book, rho, restore_fn, ref, device):
    restore_fn()
    mon = R3FOnline.create(t_max=T_MAX)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    handoff = False
    reason = "NONE"
    step_h = None
    drop = False
    n_steps = 0
    for t in range(T_MAX + 1):
        if latched(env.sim):
            drop = True
            break
        st = mon.observe(env.sim, t, tape_ended=False)
        if st["ready"]:
            handoff = True
            reason = st["reason"]
            step_h = t
            break
        if t >= T_MAX:
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
                lg, resid, _ = model(
                    torch.as_tensor(np.stack(obs_seq[-K_WIN:]), device=device),
                    torch.as_tensor(np.stack(prev_seq[-K_WIN:]), device=device),
                )
                lg = lg[-1].clone()
                lg[HANDOFF] = -1e9
                tok = int(torch.argmax(lg).item())
                rr = resid[-1].float().cpu().numpy().astype(np.float32)
                action = decode_m111_action(book, tok, rr, rho=rho)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32).reshape(7)
        n_steps += 1

    out = {
        "handoff": int(handoff or (not drop and True)),  # timeout counts
        "handoff_reason": reason if handoff else ("DROP_BEFORE" if drop else "TIMEOUT_FALLBACK"),
        "handoff_step": step_h if handoff else (None if drop else T_MAX),
        "drop_before": int(drop),
        "n_policy_steps": n_steps,
        "SUCCESS_12S": 0,
        "SUCCESS_NEW": 0,
        "fail_category": None,
    }
    if drop:
        out["handoff"] = 0
        out["fail_category"] = "DROP_BEFORE"
        return out
    if not handoff:
        out["handoff"] = 1
        out["handoff_reason"] = "TIMEOUT_FALLBACK"
        out["handoff_step"] = T_MAX
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=8)
    s12 = int(bool(hand.get("SUCCESS_12S")))
    snew = int(bool(hand.get("SUCCESS_NEW")))
    out.update({"SUCCESS_12S": s12, "SUCCESS_NEW": snew, "eval_reason": hand.get("reason")})
    if snew:
        out["fail_category"] = "SUCCESS"
    elif s12:
        out["fail_category"] = "HOLD_FAIL_12S_OK"
    else:
        out["fail_category"] = f"POST_HANDOFF:{hand.get('reason')}"
    return out


def eval_ckpt(tag: str, ckpt: Path, jobs_by_key: dict, hold_rows: list) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, _ = load_m111(ckpt, device)
    model.eval()
    _b2, rho, _ = load_m111_book(M111_META)
    book = book if book is not None else _b2
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []
    rows = []
    print(f"=== eval {tag} n={len(hold_rows)} ===", flush=True)
    for i, hr in enumerate(hold_rows):
        key = (hr["skill"], hr["traj_id"])
        job = jobs_by_key.get(key)
        if job is None:
            cands = [
                j
                for (sk, tid), j in jobs_by_key.items()
                if sk == hr["skill"] and (tid == hr["traj_id"] or hr["traj_id"] in tid or tid in hr["traj_id"])
            ]
            job = cands[0] if cands else None
        if job is None:
            continue
        skill = job["skill"]
        if skill in ("A1", "B1", "C"):
            ep = job["ep"]
            ref = ep["ref"]

            def restore(ep=ep):
                restore_start(env, ep)

        else:
            cap, ref = job["capture"], job["ref"]

            def restore(cap=cap):
                restore_capture(env, cap)

        ev = rollout(env, model, book, rho, restore, ref, device)
        row = {
            "tag": tag,
            "skill": skill,
            "traj_id": job["traj_id"],
            "p44_12S": int(hr.get("SUCCESS_12S") or 0),
            "p44_fail": hr.get("fail_category"),
            **ev,
        }
        rows.append(row)
        write_csv(OUT / "PAIRED_EVAL_ROLLOUTS.csv", rows)
        print(
            f"  [{i+1}/{len(hold_rows)}] {tag} {skill} {job['traj_id'][:36]} "
            f"{ev['fail_category']} 12S={ev['SUCCESS_12S']} drop={ev['drop_before']}",
            flush=True,
        )
    return rows


def summarize(rows: list[dict], tag: str) -> dict:
    out = {}
    for skill in ("A1", "B1", "C", "D"):
        sub = [r for r in rows if r["tag"] == tag and r["skill"] == skill]
        out[skill] = {
            "n": len(sub),
            "SUCCESS_12S": sum(int(r["SUCCESS_12S"]) for r in sub),
            "SUCCESS_NEW": sum(int(r["SUCCESS_NEW"]) for r in sub),
            "DROP_BEFORE": sum(int(r["drop_before"]) for r in sub),
            "handoff": sum(int(r["handoff"]) for r in sub),
            "reasons": dict(Counter(r["handoff_reason"] for r in sub)),
            "fails": dict(Counter(r["fail_category"] for r in sub)),
        }
    return out


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hold = [r for r in csv.DictReader(P44_EVAL.open(encoding="utf-8")) if r.get("tag") == "holdout"]
    # Deduplicate by skill,traj_id keeping last (v2 eval)
    uniq = {}
    for r in hold:
        uniq[(r["skill"], r["traj_id"])] = r
    hold = list(uniq.values())
    jobs_by_key = {(j["skill"], j["traj_id"]): j for j in collect_jobs()}

    all_rows = []
    # P44 baseline numbers from CSV (no re-run) + optional re-eval tags
    for r in hold:
        all_rows.append(
            {
                "tag": "P44",
                "skill": r["skill"],
                "traj_id": r["traj_id"],
                "SUCCESS_12S": int(r.get("SUCCESS_12S") or 0),
                "SUCCESS_NEW": int(r.get("SUCCESS_NEW") or 0),
                "drop_before": int(r.get("fail_category") == "DROP_BEFORE"),
                "handoff": int(r.get("handoff") or 0) if "handoff" in r else int(r.get("fail_category") != "DROP_BEFORE"),
                "handoff_reason": r.get("handoff_reason"),
                "fail_category": r.get("fail_category"),
                "p44_12S": int(r.get("SUCCESS_12S") or 0),
                "p44_fail": r.get("fail_category"),
            }
        )

    for tag, ckpt in (
        ("P45A", OUT / "checkpoints/P45A_best.pt"),
        ("P45B", OUT / "checkpoints/P45B_best.pt"),
    ):
        if not ckpt.exists():
            print(f"SKIP missing {ckpt}", flush=True)
            continue
        all_rows.extend(eval_ckpt(tag, ckpt, jobs_by_key, hold))

    write_csv(OUT / "PAIRED_EVAL_ROLLOUTS.csv", all_rows)
    summaries = {tag: summarize(all_rows, tag) for tag in ("P44", "P45A", "P45B")}

    # Paired flips P44 -> P45B
    lines = [
        "# PAIRED_EVAL_REPORT",
        "",
        f"Generated: {utc()}",
        "Same holdout ICs as P44. R3F T_max=3100 unchanged.",
        "",
        "## Aggregate SUCCESS_12S",
        "",
        "| Skill | P44 | P45-A | P45-B | P44 DROP | P45-A DROP | P45-B DROP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sk in ("A1", "B1", "C", "D"):
        a = summaries.get("P44", {}).get(sk, {})
        b = summaries.get("P45A", {}).get(sk, {})
        c = summaries.get("P45B", {}).get(sk, {})
        lines.append(
            f"| {sk} | {a.get('SUCCESS_12S', 0)}/{a.get('n', 0)} | "
            f"{b.get('SUCCESS_12S', 0)}/{b.get('n', 0)} | "
            f"{c.get('SUCCESS_12S', 0)}/{c.get('n', 0)} | "
            f"{a.get('DROP_BEFORE', 0)} | {b.get('DROP_BEFORE', 0)} | {c.get('DROP_BEFORE', 0)} |"
        )

    # flips
    def map_tag(tag):
        return {(r["skill"], r["traj_id"]): r for r in all_rows if r["tag"] == tag}

    p44m, p45am, p45bm = map_tag("P44"), map_tag("P45A"), map_tag("P45B")
    lines += ["", "## Paired flips vs P44 (SUCCESS_12S)", ""]
    for name, mp in (("P45A", p45am), ("P45B", p45bm)):
        gain = loss = both = neither = 0
        per = Counter()
        for k, a in p44m.items():
            if k not in mp:
                continue
            b = mp[k]
            aok, bok = int(a["SUCCESS_12S"]), int(b["SUCCESS_12S"])
            if aok and bok:
                both += 1
            elif (not aok) and bok:
                gain += 1
                per[k[0]] += 1
            elif aok and (not bok):
                loss += 1
            else:
                neither += 1
        lines.append(
            f"- **{name}**: gain={gain} loss={loss} both_ok={both} both_fail={neither} "
            f"gain_by_skill=`{dict(per)}`"
        )

    # acceptance
    lines += ["", "## Acceptance checks", ""]
    ok_all = True
    for sk in ("A1", "B1", "C", "D"):
        for tag in ("P45A", "P45B"):
            s = summaries.get(tag, {}).get(sk)
            if not s or s["n"] == 0:
                continue
            if s["SUCCESS_12S"] <= 0:
                ok_all = False
                lines.append(f"- FAIL {tag} {sk} SUCCESS_12S=0")
    p44 = summaries["P44"]
    for tag in ("P45A", "P45B"):
        if tag not in summaries or not summaries[tag]:
            continue
        for sk in ("B1", "C"):
            if summaries[tag].get(sk, {}).get("n", 0) == 0:
                continue
            if summaries[tag][sk]["SUCCESS_12S"] < p44[sk]["SUCCESS_12S"]:
                lines.append(
                    f"- WARN {tag} {sk} below P44 "
                    f"({summaries[tag][sk]['SUCCESS_12S']} < {p44[sk]['SUCCESS_12S']})"
                )
            else:
                lines.append(f"- OK {tag} {sk} ≥ P44")
    lines.append(f"- All classes >0 under best run: see FINAL_SUMMARY")
    (OUT / "PAIRED_EVAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "PAIRED_EVAL_SUMMARY.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    run()


if __name__ == "__main__":
    main()
