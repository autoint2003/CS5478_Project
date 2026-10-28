"""P49 Phase 1–3: B policy rollout collection + oracle recoverability + DAgger pack.

Starts from P47T. Non-holdout ball/slide/k48r/inward only. No table_free/hard_*.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data_generation.oracles.plan_c.future_trajectory import pad_bilateral  # noqa: E402
from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.outcomes.support_drop import latched  # noqa: E402
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.diagnostics.run_b_grpo_multistage_dataset import (  # noqa: E402
    DEFAULT_WEIGHTS,
    make_plain_env,
    probe_direct,
    quantize_and_replay,
    run_prec,
    run_tree80_stage1,
)
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p28_multiskill_distillation.harness import policy_step_K  # noqa: E402
from experiments.precision.p38_unified_motion_only_sft.paths import M111_META as _  # noqa: E402
from experiments.precision.p44_r3f_unified_sft.r3f_online import R3FOnline  # noqa: E402
from experiments.precision.p49_b_targeted_dagger.paths import (  # noqa: E402
    BALL_V2,
    DATA_SNAP,
    M111_META,
    OUT,
    P44_EVAL,
    P45B_DATA,
    P47T_CKPT,
    SEED,
    T_MAX,
)
from models.k48_checkpoint import load_trained_k48  # noqa: E402
from models.m111_checkpoint import HANDOFF, MOTION_K, decode_m111_action, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.k48_config import SELECTED_LIMIT_S  # noqa: E402
from training.m111_dataset import WINDOW, chunk_episode  # noqa: E402
from training.recovery_episode import make_training_env  # noqa: E402
from training.skill_aware_action_vocab import load_motion_codebook  # noqa: E402

TREE80 = ROOT / "results/grpo/tree80/checkpoint_epoch_0003.pt"
CUT_TIMES = (1, 3, 5, 8, 12, 20)
FAMILIES = ("ball", "slide", "k48r", "inward")


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


def root_id(tid: str) -> str:
    t = re.sub(r"^(plan_b__)?", "", tid)
    t = re.sub(r"__([0-9a-f]{12,})$", "", t)
    return t


def holdout_b() -> set[str]:
    return {
        root_id(r["traj_id"])
        for r in csv.DictReader(P44_EVAL.open(encoding="utf-8"))
        if r.get("skill") == "B1" and r.get("tag") == "holdout"
    }


def family_of(ic_id: str) -> str | None:
    if ic_id.startswith("hard_"):
        return None
    if "ball" in ic_id:
        return "ball"
    if "slide" in ic_id:
        return "slide"
    if "k48r" in ic_id:
        return "k48r"
    if "inward" in ic_id:
        return "inward"
    return None


def load_family_ics(held: set[str], *, per_family: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    by: dict[str, list] = {f: [] for f in FAMILIES}
    # ball from ball_v2
    for p in sorted(BALL_V2.glob("*.pkl")):
        ic = p.stem
        if ic in held:
            continue
        fam = family_of(ic)
        if fam != "ball":
            continue
        blob = pickle.loads(p.read_bytes())
        if "capture" not in blob or "ref" not in blob:
            continue
        by["ball"].append(
            {"ic_id": ic, "family": "ball", "capture": blob["capture"], "ref": blob["ref"], "path": str(p)}
        )
    # slide/k48r/inward from data/snapshots/train
    for p in sorted(DATA_SNAP.glob("*.pkl")):
        ic = p.stem
        if ic in held:
            continue
        fam = family_of(ic)
        if fam not in ("slide", "k48r", "inward"):
            continue
        blob = pickle.loads(p.read_bytes())
        if "capture" not in blob or "ref" not in blob:
            continue
        by[fam].append(
            {"ic_id": ic, "family": fam, "capture": blob["capture"], "ref": blob["ref"], "path": str(p)}
        )
    out = []
    for fam in FAMILIES:
        pool = by[fam]
        rng.shuffle(pool)
        out.extend(pool[:per_family])
        print(f"  family {fam}: selected {min(per_family, len(pool))}/{len(pool)}", flush=True)
    return out


def observe(env) -> np.ndarray:
    return np.asarray(
        observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, env._hist),
        np.float32,
    )


def contact_info(sim) -> dict:
    pb = pad_bilateral(sim)
    return {
        "bilateral": int(pb["bilateral"]),
        "nL": int(pb["nL"]),
        "nR": int(pb["nR"]),
        "asym": int(abs(int(pb["nL"]) - int(pb["nR"]))),
    }


def policy_roll_collect(env, model, book, rho, ic, device, *, horizon: int = 80) -> dict:
    restore_capture(env, ic["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    mon = R3FOnline.create(t_max=T_MAX)
    obs_seq, prev_seq, toks, caps = [], [], [], {}
    first_bil_loss = None
    first_asym = None
    drop = False
    hand = False
    hand_reason = None
    for t in range(horizon + 1):
        if latched(env.sim):
            drop = True
            break
        st = mon.observe(env.sim, t, tape_ended=False)
        if st["ready"]:
            hand = True
            hand_reason = st["reason"]
            break
        if t >= horizon:
            break
        ci = contact_info(env.sim)
        if first_bil_loss is None and not ci["bilateral"] and t > 0:
            first_bil_loss = t
        if first_asym is None and ci["asym"] >= 2 and t > 0:
            first_asym = t
        # snapshot cut candidates BEFORE action
        if t in CUT_TIMES or (first_bil_loss is not None and t == max(1, first_bil_loss - 1)):
            caps[t] = {
                "capture": take_capture(env),
                "prev": env._prev.copy(),
                "obs": observe(env),
                "contact": ci,
                "t": t,
            }
        obs = observe(env)
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action, _ = policy_step_K(model, book, rho, obs_seq, prev_seq, device, WINDOW)
        if action is None or tok == HANDOFF:
            with torch.no_grad():
                lg, resid, _ = model(
                    torch.as_tensor(np.stack(obs_seq[-WINDOW:]), device=device),
                    torch.as_tensor(np.stack(prev_seq[-WINDOW:]), device=device),
                )
                lg = lg[-1].clone()
                lg[HANDOFF] = -1e9
                tok = int(torch.argmax(lg).item())
                rr = resid[-1].float().cpu().numpy().astype(np.float32)
                action = decode_m111_action(book, tok, rr, rho=rho)
        toks.append(tok)
        step_legal(env.sim, action)
        env._prev = np.asarray(action, np.float32).reshape(7)
    return {
        "drop": int(drop),
        "handoff": int(hand),
        "hand_reason": hand_reason,
        "n_steps": len(toks),
        "first_bil_loss": first_bil_loss,
        "first_asym": first_asym,
        "cuts": caps,
        "tokens": toks,
    }


def classify_probe(ok_init: bool, cont_ok: bool, s12: bool, reason: str | None) -> str:
    if not ok_init:
        return "INVALID_RESTORE"
    if reason == "unsupported":
        return "ORACLE_UNSUPPORTED"
    if s12:
        return "RECOVERABLE"
    return "UNRECOVERABLE"


def oracle_probe_from_cut(
    env_plain,
    env_pol,
    tree_model,
    tree_book,
    device,
    cut: dict,
    ref,
    cb,
) -> dict:
    """Try direct → precision → Tree80+precision from full capture."""
    cap = cut["capture"]
    try:
        restore_capture(env_plain, cap)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "INVALID_RESTORE",
            "ok_init": 0,
            "SUCCESS_12S": 0,
            "method": None,
            "fail": f"restore:{e}",
        }
    ok_init = 1
    # 1) direct
    try:
        d = probe_direct(env_plain, cap, ref)
    except Exception as e:  # noqa: BLE001
        d = {"SUCCESS_12S": False, "SUCCESS_NEW": False, "reason": f"direct_err:{e}"}
    if d.get("SUCCESS_12S"):
        # no motion labels — mark recoverable but skip for dagger labels
        return {
            "status": "RECOVERABLE",
            "ok_init": ok_init,
            "SUCCESS_12S": 1,
            "SUCCESS_NEW": int(bool(d.get("SUCCESS_NEW"))),
            "method": "direct_handoff",
            "actions": np.zeros((0, 7), np.float32),
            "n_actions": 0,
            "usable_dagger": 0,
            "reason": d.get("reason"),
        }
    # 2) precision from cut
    try:
        prec = run_prec(env_plain, cap, ref)
    except Exception as e:  # noqa: BLE001
        prec = {
            "SUCCESS_12S": False,
            "actions": np.zeros((0, 7), np.float32),
            "reason": f"prec_err:{e}",
            "executed": False,
        }
    if prec.get("SUCCESS_12S") and len(prec.get("actions") or []):
        acts = np.asarray(prec["actions"], np.float32)
        q = quantize_and_replay(env_plain, cap, ref, acts, cb.prototypes, cb.residual, DEFAULT_WEIGHTS, "M111")
        if q["SUCCESS_12S"]:
            return {
                "status": "RECOVERABLE",
                "ok_init": ok_init,
                "SUCCESS_12S": 1,
                "SUCCESS_NEW": int(bool(q["SUCCESS_NEW"])),
                "method": "precision",
                "actions": acts,
                "tokens": q["tokens"],
                "residuals": q["residuals"],
                "recon": q["recon_actions"],
                "n_actions": int(len(acts)),
                "usable_dagger": 1,
                "clip": q.get("clipped_frac"),
            }
        return {
            "status": "UNRECOVERABLE",
            "ok_init": ok_init,
            "SUCCESS_12S": 0,
            "method": "precision_m111_fail",
            "fail": q.get("reason"),
            "usable_dagger": 0,
        }
    # 3) Tree80 stage1 from cut then precision
    try:
        s1 = run_tree80_stage1(env_pol, tree_model, tree_book, device, cap, ref)
    except Exception as e:  # noqa: BLE001
        return {
            "status": "ORACLE_UNSUPPORTED",
            "ok_init": ok_init,
            "SUCCESS_12S": 0,
            "method": "tree80_err",
            "fail": str(e),
            "usable_dagger": 0,
        }
    if not s1.get("ok") or s1.get("exit_capture") is None:
        return {
            "status": "UNRECOVERABLE",
            "ok_init": ok_init,
            "SUCCESS_12S": 0,
            "method": "tree80_stage1_fail",
            "fail": s1.get("fail_reason") or s1.get("how"),
            "usable_dagger": 0,
        }
    d2 = probe_direct(env_plain, s1["exit_capture"], ref)
    acts = np.asarray(s1["actions"], np.float32)
    if d2.get("SUCCESS_12S"):
        method = "tree80_direct"
    else:
        prec2 = run_prec(env_plain, s1["exit_capture"], ref)
        if not prec2.get("SUCCESS_12S"):
            return {
                "status": "UNRECOVERABLE",
                "ok_init": ok_init,
                "SUCCESS_12S": 0,
                "method": "tree80_prec_fail",
                "fail": prec2.get("reason"),
                "usable_dagger": 0,
            }
        if len(prec2.get("actions") or []):
            acts = np.concatenate([acts, np.asarray(prec2["actions"], np.float32)], axis=0)
        method = "tree80_precision"
    if len(acts) == 0:
        return {
            "status": "RECOVERABLE",
            "ok_init": ok_init,
            "SUCCESS_12S": 1,
            "method": method,
            "usable_dagger": 0,
            "n_actions": 0,
            "actions": acts,
        }
    q = quantize_and_replay(env_plain, cap, ref, acts, cb.prototypes, cb.residual, DEFAULT_WEIGHTS, "M111")
    if not q["SUCCESS_12S"]:
        return {
            "status": "UNRECOVERABLE",
            "ok_init": ok_init,
            "SUCCESS_12S": 0,
            "method": f"{method}_m111_fail",
            "fail": q.get("reason"),
            "usable_dagger": 0,
        }
    return {
        "status": "RECOVERABLE",
        "ok_init": ok_init,
        "SUCCESS_12S": 1,
        "SUCCESS_NEW": int(bool(q["SUCCESS_NEW"])),
        "method": method,
        "actions": acts,
        "tokens": q["tokens"],
        "residuals": q["residuals"],
        "recon": q["recon_actions"],
        "n_actions": int(len(acts)),
        "usable_dagger": 1,
        "clip": q.get("clipped_frac"),
    }


def continuation_to_chunks(env, cut, probe, ic, *, sample_weight: float = 2.0) -> list[dict]:
    """Build supervised chunks from cut state along verified M111 continuation."""
    if not probe.get("usable_dagger"):
        return []
    toks = np.asarray(probe["tokens"], np.int64)
    res = np.asarray(probe["residuals"], np.float32)
    recon = np.asarray(probe["recon"], np.float32)
    n = len(toks)
    if n and int(toks[-1]) == HANDOFF:
        n -= 1
        toks, res, recon = toks[:n], res[:n], recon[:n]
    if n == 0:
        return []
    restore_capture(env, cut["capture"])
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.asarray(cut["prev"], np.float32).reshape(7)
    obs_list, prev_list = [], []
    for t in range(n):
        obs_list.append(observe(env))
        prev_list.append(env._prev.copy())
        step_legal(env.sim, recon[t])
        env._prev = recon[t].reshape(7)
    mask = (toks != HANDOFF) & (toks >= 0) & (toks < MOTION_K)
    if not bool(mask.any()):
        return []
    ep = {
        "observations": np.asarray(obs_list, np.float32),
        "prev_actions": np.asarray(prev_list, np.float32),
        "tokens": toks,
        "residuals": res,
        "loss_mask": mask,
    }
    out = []
    for ci, ch in enumerate(chunk_episode(ep, window=WINDOW)):
        m = np.asarray(ch["loss_mask"], dtype=np.bool_)
        m &= np.asarray(ch["tokens"]) != HANDOFF
        m &= (np.asarray(ch["tokens"]) >= 0) & (np.asarray(ch["tokens"]) < MOTION_K)
        if not bool(m.any()):
            continue
        # flat weight — do not upweight long continuations
        w = float(sample_weight)
        out.append(
            {
                **ch,
                "loss_mask": m,
                "traj_id": f"plan_b__p49dagger__{ic['ic_id']}__t{cut['t']}",
                "skill": "B1",
                "split": "train",
                "chunk_index": ci,
                "n_tokens": int(m.sum()),
                "sample_weight": w,
                "source": "p49_dagger",
                "ic_id": ic["ic_id"],
                "family": ic["family"],
                "cut_step": cut["t"],
                "oracle_method": probe.get("method"),
                "SUCCESS_12S": 1,
                "policy_ckpt": "P47T",
            }
        )
    return out


def run(*, per_family: int = 6, pilot: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset").mkdir(parents=True, exist_ok=True)
    (OUT / "raw").mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    held = holdout_b()
    print(f"Holdout B excluded: {len(held)}", flush=True)
    ics = load_family_ics(held, per_family=per_family, seed=SEED)
    print(f"Collect ICs: {len(ics)} (pilot={pilot})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, book, _ = load_m111(P47T_CKPT, device)
    model.eval()
    _b2, rho, _ = load_m111_book(M111_META)
    book = book if book is not None else _b2
    cb = load_motion_codebook(M111_META)
    tree_model, tree_book = load_trained_k48(TREE80, device)
    env = make_env()
    env._obs_seq = []
    env._prev_seq = []
    env_plain = make_plain_env()
    env_pol = make_training_env(device, model=tree_model)

    collect_rows = []
    probe_rows = []
    dagger_chunks: list[dict] = []
    status_by_t = defaultdict(Counter)

    for i, ic in enumerate(ics):
        print(f"=== [{i+1}/{len(ics)}] {ic['family']} {ic['ic_id']} ===", flush=True)
        roll = policy_roll_collect(env, model, book, rho, ic, device)
        collect_rows.append(
            {
                "ic_id": ic["ic_id"],
                "family": ic["family"],
                "drop": roll["drop"],
                "handoff": roll["handoff"],
                "hand_reason": roll["hand_reason"],
                "n_steps": roll["n_steps"],
                "first_bil_loss": roll["first_bil_loss"],
                "first_asym": roll["first_asym"],
                "n_cuts": len(roll["cuts"]),
            }
        )
        # probe each cut (pilot: all; full: prefer early + pre-bil-loss)
        cuts = sorted(roll["cuts"].items())
        if pilot and len(cuts) > 4:
            # keep earliest 3 + last before bil loss if any
            prefer = [c for c in cuts if c[0] <= 8]
            cuts = prefer[:4] if prefer else cuts[:4]
        for t, cut in cuts:
            print(f"  probe t={t} bil={cut['contact']['bilateral']} asym={cut['contact']['asym']}", flush=True)
            probe = oracle_probe_from_cut(
                env_plain, env_pol, tree_model, tree_book, device, cut, ic["ref"], cb
            )
            status = probe["status"]
            status_by_t[t][status] += 1
            row = {
                "ic_id": ic["ic_id"],
                "family": ic["family"],
                "cut_t": t,
                "bilateral": cut["contact"]["bilateral"],
                "asym": cut["contact"]["asym"],
                "status": status,
                "method": probe.get("method"),
                "SUCCESS_12S": int(probe.get("SUCCESS_12S") or 0),
                "usable_dagger": int(probe.get("usable_dagger") or 0),
                "n_actions": probe.get("n_actions"),
                "fail": probe.get("fail") or probe.get("reason"),
                "dt_from_bil_loss": (
                    None
                    if roll["first_bil_loss"] is None
                    else int(t - roll["first_bil_loss"])
                ),
            }
            probe_rows.append(row)
            write_csv(OUT / "ORACLE_PROBE_ROWS.csv", probe_rows)
            if probe.get("usable_dagger"):
                chs = continuation_to_chunks(env_plain, cut, probe, ic, sample_weight=2.0)
                dagger_chunks.extend(chs)
                print(
                    f"    ADMIT {status} method={probe.get('method')} "
                    f"acts={probe.get('n_actions')} chunks={len(chs)}",
                    flush=True,
                )
            else:
                print(f"    {status} method={probe.get('method')} fail={row.get('fail')}", flush=True)

    write_csv(OUT / "POLICY_COLLECTION_ROWS.csv", collect_rows)
    (OUT / "dataset" / "chunks_B1_dagger_only.pkl").write_bytes(pickle.dumps(dagger_chunks))

    # recoverability summary
    st_counts = Counter(r["status"] for r in probe_rows)
    usable = sum(int(r["usable_dagger"]) for r in probe_rows)
    by_t = {str(t): dict(c) for t, c in sorted(status_by_t.items())}
    # intervention window: earliest t with recoverable rate >= 30% among probed
    window = None
    for t in sorted(status_by_t):
        c = status_by_t[t]
        n = sum(c.values()) or 1
        rate = c.get("RECOVERABLE", 0) / n
        if rate >= 0.3:
            window = {"t": t, "recoverable_rate": rate, "n": n}
            break

    rec = {
        "generated": utc(),
        "n_ics": len(ics),
        "n_probes": len(probe_rows),
        "status_counts": dict(st_counts),
        "usable_dagger_probes": usable,
        "n_dagger_chunks": len(dagger_chunks),
        "recoverability_by_cut_t": by_t,
        "intervention_window_hint": window,
        "collection_drop_rate": float(np.mean([r["drop"] for r in collect_rows])) if collect_rows else None,
    }
    (OUT / "ORACLE_RECOVERABILITY.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    # merge with P45B replay pools for training later
    pools = {}
    for sk in ("A1", "B1", "C", "D"):
        base = pickle.loads((P45B_DATA / f"chunks_{sk}_P45B.pkl").read_bytes())
        # Prefer P47T-era A/C/D: use P45B as clean replay (no table_free). For B add dagger.
        pools[sk] = list(base)
    # Also try load P47T B mix without table_free if exists
    p47t_b = OUT.parent / "p47_b_ordinary_expansion/dataset/chunks_B1_P47T.pkl"
    # Keep P45B B as replay base + dagger only (avoid P47T ordinary flooding)
    pools["B1"] = list(pools["B1"]) + list(dagger_chunks)
    for sk, chs in pools.items():
        (OUT / "dataset" / f"chunks_{sk}_P49.pkl").write_bytes(pickle.dumps(chs))
    (OUT / "dataset" / "dagger_meta.json").write_text(
        json.dumps(
            {
                "n_dagger_chunks": len(dagger_chunks),
                "replay_B1": len(pickle.loads((P45B_DATA / "chunks_B1_P45B.pkl").read_bytes())),
                "total_B1": len(pools["B1"]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # markdowns
    (OUT / "POLICY_STATE_COLLECTION.md").write_text(
        "\n".join(
            [
                "# POLICY_STATE_COLLECTION",
                "",
                f"Generated: {utc()}",
                f"Policy: P47T; ICs={len(ics)}; families={FAMILIES}",
                f"Holdout excluded: {len(held)}",
                f"Drop rate on collect: {rec['collection_drop_rate']}",
                f"Cuts probed: {len(probe_rows)}; cut schedule={CUT_TIMES}",
                "",
                f"Per-family IC counts: {dict(Counter(i['family'] for i in ics))}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "ORACLE_RECOVERABILITY.md").write_text(
        "\n".join(
            [
                "# ORACLE_RECOVERABILITY",
                "",
                f"Generated: {utc()}",
                "",
                f"Status counts: `{dict(st_counts)}`",
                f"Usable DAgger probes (SUCCESS_12S + motion labels): **{usable}** / {len(probe_rows)}",
                f"DAgger chunks: **{len(dagger_chunks)}**",
                f"Recoverability by cut_t: `{by_t}`",
                f"Intervention window hint (first t with ≥30% recoverable): `{window}`",
                "",
                "Methods: direct_handoff (no labels) → precision → Tree80(+precision), then M111 requantize gate.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "DAGGER_DATASET_SUMMARY.md").write_text(
        "\n".join(
            [
                "# DAGGER_DATASET_SUMMARY",
                "",
                f"Generated: {utc()}",
                f"DAgger chunks: {len(dagger_chunks)}",
                f"By family: {dict(Counter(c.get('family') for c in dagger_chunks))}",
                f"By method: {dict(Counter(c.get('oracle_method') for c in dagger_chunks))}",
                f"By cut_t: {dict(Counter(c.get('cut_step') for c in dagger_chunks))}",
                "No table_free / hard_*. Flat sample_weight=2.0 (not length-weighted).",
                f"Merged B1 pool size: {len(pools['B1'])} (P45B replay + dagger)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(rec, indent=2), flush=True)
    if usable == 0:
        print("STOP: no usable recoverable DAgger — adjust intervention timing before expand.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=6)
    ap.add_argument("--full", action="store_true", help="more cuts per IC")
    args = ap.parse_args()
    run(per_family=args.per_family, pilot=not args.full)


if __name__ == "__main__":
    main()
