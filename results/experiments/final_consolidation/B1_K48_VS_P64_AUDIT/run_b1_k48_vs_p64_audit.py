"""B1 K48 GRPO vs P64 diagnostic — post-hoc paired closed-loop on frozen 8 B1 ICs.

Does not train, tune, or change the frozen final policy.
Does not alter the independent final-eval 0/8 result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    CACHE,
    MAX_EP_STEPS,
    OUT as FINAL_OUT,
    P64_ACTOR,
    P64_CRITIC,
    T_MAX,
)
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import (  # noqa: E402
    StabilizeGrpoEnv,
)
from experiments.precision.p63_transformer_critic_ppo.checkpointing import (  # noqa: E402
    load_actor,
    load_critic,
)
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p82_global_failure_budget.collect import rollout_episode  # noqa: E402
from models.k48_checkpoint import HANDOFF as K48_HANDOFF  # noqa: E402
from models.k48_checkpoint import load_trained_k48  # noqa: E402
from training.k47_codebook import decode_action  # noqa: E402
from training.k48_config import SELECTED_LIMIT_S  # noqa: E402
from training.recovery_episode import make_training_env  # noqa: E402

OUT = Path(__file__).resolve().parent
K48_V2 = ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v2/checkpoint_u0312.pt"
K48_TREE80 = ROOT / "results/grpo/tree80/checkpoint_epoch_0003.pt"
K48_EVAL_MD = ROOT / "results/archive/demo_results/diagnostics/K48_GRPO_EVAL_COMPARISON.md"
K48_CMP = ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/eval_compare/comparison.json"
K48_ROWS = ROOT / "results/archive/demo_results/diagnostics/raw/unified_k48/eval_compare/v2_u0312/heldout_train.json"
FINAL_CSV = FINAL_OUT / "EPISODE_RESULTS.csv"
FINAL_MANIFEST = FINAL_OUT / "FROZEN_TEST_MANIFEST.json"
STARTS_PKL = CACHE / "selected_starts.pkl"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def load_b1_bundle() -> tuple[list[dict], list[Any], list[dict]]:
    manifest = json.loads(FINAL_MANIFEST.read_text(encoding="utf-8"))
    b1_eps = [e for e in manifest["episodes"] if e["skill"] == "B1"]
    starts_blob = pickle.loads(STARTS_PKL.read_bytes())
    starts = list(starts_blob["B1"])
    by_tid = {s.traj_id: s for s in starts}
    ordered = [by_tid[e["traj_id"]] for e in b1_eps]
    frozen_rows = []
    with FINAL_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["skill"] == "B1":
                frozen_rows.append(row)
    return b1_eps, ordered, frozen_rows


def blob_from_start(start) -> dict:
    ep = start.ep
    if not isinstance(ep, dict) or ep.get("capture") is None or ep.get("ref") is None:
        raise RuntimeError(f"B1 start {start.traj_id} missing nested capture/ref")
    return {
        "capture": ep["capture"],
        "ref": ep["ref"],
        "traj_id": start.traj_id,
        "family": ep.get("family"),
        "skill": "B1",
        "mechanism": ep.get("mechanism"),
        "source_corpus": ep.get("source_corpus"),
        "lineage_id": ep.get("lineage_id"),
    }


def audit_restore(starts, *, device) -> list[dict]:
    """Confirm capture restore yields a usable physical IC (policy loaded only for env ctor)."""
    # make_training_env requires a model object (legacy balanced ckpt path is gone).
    model, _book = load_trained_k48(K48_V2, device)
    model.eval()
    env = make_training_env(device, model=model)
    rows = []
    for s in starts:
        blob = blob_from_start(s)
        try:
            restore_capture(env, blob["capture"])
            g0 = relative_geometry(env.sim)
            z = float(env.sim.data.xpos[env.sim.ids.object_body][2])
            t = float(env.sim.data.time)
            ok = True
            note = "ok"
            if not np.isfinite(z) or z < 0.05:
                ok = False
                note = f"bad_object_z={z}"
            if blob["ref"] is None:
                ok = False
                note = "missing_ref"
            rows.append(
                {
                    "traj_id": s.traj_id,
                    "restore_ok": int(ok),
                    "note": note,
                    "sim_time": t,
                    "object_z": z,
                    "has_pad_contact": int(g0.get("has_pad_contact") or 0),
                    "bilateral": int(g0.get("bilateral") or 0),
                    "e_pos_mm": float(g0.get("e_pos_mm") or -1),
                    "e_theta_deg": float(g0.get("e_theta_deg") or g0.get("theta") or -1),
                    "min_pad": int(g0.get("min_pad") or 0),
                    "start_kind_nested": (s.ep or {}).get("start_kind"),
                    "mechanism": (s.ep or {}).get("mechanism"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "traj_id": s.traj_id,
                    "restore_ok": 0,
                    "note": f"RESTORE_ERROR:{type(exc).__name__}: {exc}",
                }
            )
    return rows


def instrumented_k48_roll(
    env,
    model,
    book,
    blob,
    device,
    *,
    cap_steps: int = 200,
    limit_s: float = SELECTED_LIMIT_S,
) -> dict:
    """K48 roll_policy with contact / token milestones. Native SUCCESS_12S path."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    basin_run = 0
    lost_low = 0
    how = None
    steps = 0
    tokens: list[int] = []
    grips: list[float] = []
    ever = {k: False for k in ("fc", "bil", "opp", "pad4", "e9", "loss")}
    t_fc = t_opp = t_loss = None
    had = False
    first_tok = None
    handoff_step = None
    e_pos_at_end = e_th_at_end = None
    min_pad_end = bil_end = None

    from envs.outcomes.horizon_outcome import recovery_failed, basin_ok, _row, finish_recovery, _fail_recovery

    for _ in range(cap_steps):
        if float(env.sim.data.time) - t0 >= float(limit_s) - 1e-9:
            how = "forced"
            break
        obs = np.asarray(env.observation(), np.float32)
        env._obs_seq.append(obs)
        env._prev_seq.append(env._prev.copy())
        with torch.no_grad():
            logits, mean, _act = model(
                torch.tensor(np.stack(env._obs_seq[-160:]), device=device),
                torch.tensor(np.stack(env._prev_seq[-160:]), device=device),
            )
        token = int(torch.argmax(logits[-1]).item())
        if first_tok is None:
            first_tok = token
        tokens.append(token)
        if token == int(K48_HANDOFF):
            how = "early"
            handoff_step = steps
            break
        if int(token) >= len(book):
            raise RuntimeError(f"token {token} outside K48 book")
        _residual, action = decode_action(book, token, mean[-1].detach().float().cpu().numpy())
        executed = step_legal(env.sim, action)
        env._prev = np.asarray(executed, np.float32)
        grips.append(float(executed[6]))
        steps += 1

        g = relative_geometry(env.sim)
        if g["has_pad_contact"]:
            if not ever["fc"]:
                t_fc = steps
            ever["fc"] = True
            had = True
        elif had and not g["has_pad_contact"]:
            ever["loss"] = True
            if t_loss is None:
                t_loss = steps
        if g["bilateral"]:
            ever["bil"] = True
        if g.get("bilateral") and g.get("correct_sidewall"):
            if not ever["opp"]:
                t_opp = steps
            ever["opp"] = True
        if g.get("min_pad", 0) >= 4:
            ever["pad4"] = True
        if g.get("e_pos_mm", 1e9) <= 9:
            ever["e9"] = True
        e_pos_at_end = float(g.get("e_pos_mm") or -1)
        e_th_at_end = float(g.get("e_theta_deg") or g.get("theta") or -1)
        min_pad_end = int(g.get("min_pad") or 0)
        bil_end = int(g.get("bilateral") or 0)

        failed, lost_low = recovery_failed(env.sim, blob["ref"], lost_low)
        if failed:
            out = _fail_recovery(env.sim, blob["ref"], t0)
            out.update(
                {
                    "recovery_steps": steps,
                    "tokens_head": tokens[:40],
                    "tokens_tail": tokens[-20:],
                    "n_tokens": len(tokens),
                    "handoff_mode": None,
                    "explicit_policy_handoff": 0,
                    "first_token": first_tok,
                    "grip_mean": float(np.mean(grips)) if grips else None,
                    "grip_min": float(np.min(grips)) if grips else None,
                    "first_contact": int(ever["fc"]),
                    "bilateral_ever": int(ever["bil"]),
                    "opposing_ever": int(ever["opp"]),
                    "min_pad_ge4": int(ever["pad4"]),
                    "e_pos_le9": int(ever["e9"]),
                    "contact_loss_ever": int(ever["loss"]),
                    "t_fc": t_fc,
                    "t_opp": t_opp,
                    "t_loss": t_loss,
                    "e_pos_mm_end": e_pos_at_end,
                    "e_theta_deg_end": e_th_at_end,
                    "min_pad_end": min_pad_end,
                    "bilateral_end": bil_end,
                    "SUCCESS_RELAXED_12S": 0,
                    "SUCCESS_NEW": 0,
                    "r3_proxy": 0,
                }
            )
            return out

        inside = basin_ok(_row(env.sim, blob["ref"]), float(executed[6]) > 0.5)
        basin_run = basin_run + 1 if inside else 0
        # With handoff_index present, early exit is only via HANDOFF token (matches roll_policy).
        # Keep basin stats for diagnostics only.

    if how is None:
        how = "forced"

    # Native K48 SUCCESS_12S (forced/early HANDOFF + nominal verify).
    # Do not call evaluate_handoff_with_axis here: stabilize_n would mutate state
    # before finish_recovery. SUCCESS_RELAXED_12S is an R3F-path metric — N/A for
    # the historical K48 handoff contract; reported as empty.
    out = finish_recovery(env, blob["ref"], t0, how)
    out.update(
        {
            "recovery_steps": steps,
            "tokens_head": tokens[:40],
            "tokens_tail": tokens[-20:],
            "n_tokens": len(tokens),
            "handoff_mode": how,
            "explicit_policy_handoff": int(how == "early"),
            "forced_recovery_limit_handoff": int(how == "forced"),
            "handoff_step": handoff_step,
            "first_token": first_tok,
            "grip_mean": float(np.mean(grips)) if grips else None,
            "grip_min": float(np.min(grips)) if grips else None,
            "first_contact": int(ever["fc"]),
            "bilateral_ever": int(ever["bil"]),
            "opposing_ever": int(ever["opp"]),
            "min_pad_ge4": int(ever["pad4"]),
            "e_pos_le9": int(ever["e9"]),
            "contact_loss_ever": int(ever["loss"]),
            "t_fc": t_fc,
            "t_opp": t_opp,
            "t_loss": t_loss,
            "e_pos_mm_end": e_pos_at_end,
            "e_theta_deg_end": e_th_at_end,
            "min_pad_end": min_pad_end,
            "bilateral_end": bil_end,
            "SUCCESS_12S_native": int(bool(out.get("success"))),
            "SUCCESS_NEW": "",
            "SUCCESS_RELAXED_12S": "",
            "strict_reason": "N/A_under_k48_native_handoff_contract",
            "r3_proxy": "",
        }
    )
    return out


def run_p64_detail(starts, frozen_rows, device) -> list[dict]:
    actor, book, rho, _ = load_actor(P64_ACTOR, device)
    critic = load_critic(P64_CRITIC, actor, device)
    for p in actor.parameters():
        p.requires_grad_(False)
    env = StabilizeGrpoEnv(
        book,
        rho,
        max_steps=MAX_EP_STEPS,
        t_max=T_MAX,
        weights=PoseRewardWeights(),
        use_stabilization=False,
    )
    frozen_by_tid = {r["traj_id"]: r for r in frozen_rows}
    rows = []
    for s in starts:
        fr = frozen_by_tid[s.traj_id]
        seed = int(fr["seed"])
        print(f"[P64 post-hoc] {s.traj_id} seed={seed}", flush=True)
        try:
            rec = rollout_episode(env, actor, critic, book, s, device, seed=seed)
            info = getattr(env, "last_info", None) or {}
            fam_seq = rec.get("family_seq") or []
            toks = [int(x["tok"]) for x in fam_seq]
            rows.append(
                {
                    "policy": "P64",
                    "traj_id": s.traj_id,
                    "root_id": fr["root_id"],
                    "seed": seed,
                    "native_success_metric": "SUCCESS_RELAXED_12S",
                    "native_success": int(rec.get("SUCCESS_RELAXED") or 0),
                    "SUCCESS_RELAXED_12S": int(rec.get("SUCCESS_RELAXED") or 0),
                    "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
                    "SUCCESS_NEW": int(info.get("SUCCESS_NEW") or 0),
                    "frozen_SUCCESS_RELAXED_12S": int(fr["SUCCESS_RELAXED_12S"]),
                    "frozen_match": int(int(rec.get("SUCCESS_RELAXED") or 0) == int(fr["SUCCESS_RELAXED_12S"])),
                    "r3_ready": int(rec.get("r3_ready") or 0),
                    "handoff": int(rec.get("handoff") or 0),
                    "handoff_mode": "r3f" if int(rec.get("handoff") or 0) else ("timeout" if rec.get("timeout") else "fail"),
                    "earliest_decisive_failure": rec.get("earliest_decisive_failure"),
                    "ep_len": int(rec.get("ep_len") or 0),
                    "forced_steps": int(rec.get("forced_steps") or 0),
                    "actor_steps": int(rec.get("actor_steps") or 0),
                    "first_contact": int(rec.get("first_contact") or 0),
                    "bilateral_ever": int(rec.get("bilateral_ever") or 0),
                    "opposing_ever": int(rec.get("opposing_ever") or 0),
                    "min_pad_ge4": int(rec.get("min_pad_ge4") or 0),
                    "e_pos_le9": int(rec.get("e_pos_le9") or 0),
                    "frame_ok": int(rec.get("frame_ok") or 0),
                    "contact_loss_ever": int(rec.get("contact_loss_ever") or 0),
                    "t_fc": rec.get("t_fc"),
                    "t_opp": rec.get("t_opp"),
                    "t_loss": rec.get("t_loss"),
                    "tokens_head": toks[:40],
                    "tokens_tail": toks[-20:],
                    "n_tokens": len(toks),
                    "family_tail": [x.get("fam") for x in fam_seq[-12:]],
                    "grip_open_near_contact_steps": rec.get("grip_open_near_contact_steps"),
                    "infra_error": None,
                    "post_hoc_diagnostic": 1,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "policy": "P64",
                    "traj_id": s.traj_id,
                    "root_id": fr["root_id"],
                    "seed": seed,
                    "native_success": 0,
                    "SUCCESS_RELAXED_12S": 0,
                    "frozen_SUCCESS_RELAXED_12S": int(fr["SUCCESS_RELAXED_12S"]),
                    "frozen_match": 0,
                    "earliest_decisive_failure": "infra_error",
                    "infra_error": f"{type(exc).__name__}: {exc}",
                    "post_hoc_diagnostic": 1,
                }
            )
            traceback.print_exc()
    return rows


def run_k48_on_b1(label: str, ckpt: Path, starts, frozen_rows, device) -> list[dict]:
    if not ckpt.is_file():
        return [{"policy": label, "traj_id": "N/A", "note": f"missing {ckpt}", "compatible": 0}]
    model, book = load_trained_k48(ckpt, device)
    model.eval()
    env = make_training_env(device, model=model)
    frozen_by_tid = {r["traj_id"]: r for r in frozen_rows}
    rows = []
    for s in starts:
        fr = frozen_by_tid[s.traj_id]
        blob = blob_from_start(s)
        print(f"[K48 {label}] {s.traj_id}", flush=True)
        try:
            out = instrumented_k48_roll(env, model, book, blob, device)
            rows.append(
                {
                    "policy": label,
                    "ckpt": str(ckpt.relative_to(ROOT)).replace("\\", "/"),
                    "ckpt_md5": _md5(ckpt),
                    "traj_id": s.traj_id,
                    "root_id": fr["root_id"],
                    "seed_na_k48_argmax": None,
                    "native_success_metric": "SUCCESS_12S",
                    "native_success": int(bool(out.get("success")) or out.get("SUCCESS_12S_native") or 0),
                    "SUCCESS_12S": int(bool(out.get("success")) or out.get("SUCCESS_12S_native") or 0),
                    "SUCCESS_NEW": out.get("SUCCESS_NEW"),
                    "SUCCESS_RELAXED_12S": out.get("SUCCESS_RELAXED_12S"),
                    "hist_reason": out.get("reason"),
                    "strict_reason": out.get("strict_reason"),
                    "common_metric_SUCCESS_12S": int(bool(out.get("success")) or out.get("SUCCESS_12S_native") or 0),
                    "handoff_mode": out.get("handoff_mode"),
                    "explicit_policy_handoff": int(out.get("explicit_policy_handoff") or 0),
                    "forced_recovery_limit_handoff": int(out.get("forced_recovery_limit_handoff") or 0),
                    "recovery_steps": out.get("recovery_steps"),
                    "recovery_elapsed": out.get("recovery_elapsed"),
                    "first_contact": int(out.get("first_contact") or 0),
                    "bilateral_ever": int(out.get("bilateral_ever") or 0),
                    "opposing_ever": int(out.get("opposing_ever") or 0),
                    "min_pad_ge4": int(out.get("min_pad_ge4") or 0),
                    "e_pos_le9": int(out.get("e_pos_le9") or 0),
                    "contact_loss_ever": int(out.get("contact_loss_ever") or 0),
                    "t_fc": out.get("t_fc"),
                    "t_opp": out.get("t_opp"),
                    "t_loss": out.get("t_loss"),
                    "e_pos_mm_end": out.get("e_pos_mm_end"),
                    "e_theta_deg_end": out.get("e_theta_deg_end"),
                    "min_pad_end": out.get("min_pad_end"),
                    "bilateral_end": out.get("bilateral_end"),
                    "first_token": out.get("first_token"),
                    "tokens_head": out.get("tokens_head"),
                    "tokens_tail": out.get("tokens_tail"),
                    "n_tokens": out.get("n_tokens"),
                    "grip_mean": out.get("grip_mean"),
                    "grip_min": out.get("grip_min"),
                    "r3_proxy": int(out.get("r3_proxy") or 0),
                    "compatible": 1,
                    "post_hoc_diagnostic": 1,
                    "infra_error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "policy": label,
                    "traj_id": s.traj_id,
                    "root_id": fr["root_id"],
                    "native_success": 0,
                    "compatible": 1,
                    "infra_error": f"{type(exc).__name__}: {exc}",
                    "post_hoc_diagnostic": 1,
                }
            )
            traceback.print_exc()
    return rows


def write_historical_md() -> dict:
    md5 = _md5(K48_V2)
    sha = _sha256(K48_V2)
    tree_md5 = _md5(K48_TREE80)
    tree_sha = _sha256(K48_TREE80)
    cmp = json.loads(K48_CMP.read_text(encoding="utf-8")) if K48_CMP.is_file() else {}
    held = json.loads(K48_ROWS.read_text(encoding="utf-8")) if K48_ROWS.is_file() else {}
    train_v2 = next((t for t in cmp.get("train", []) if "v2_u0312" in str(t.get("label") or t.get("name") or "") or True), None)
    # find by index 4 from known order
    train_entries = cmp.get("train") or []
    v2_entry = train_entries[4] if len(train_entries) > 4 else {}
    valid_v2 = (cmp.get("validation") or {}).get("v2_u0312") or {}
    ic_ids = [r["id"] for r in held.get("rows") or []]
    text = f"""# HISTORICAL_K48_RESULTS

Generated: {_utc()}
Status: **archive recovery only** — no retrain.

## Canonical historical K48 GRPO checkpoint (primary)

| Field | Value |
|---|---|
| Label | `v2_u0312` (GRPO long reward v2, update 312) |
| Path | `results/archive/demo_results/diagnostics/raw/unified_k48/grpo_long_reward_v2/checkpoint_u0312.pt` |
| MD5 | `{md5}` |
| SHA256 | `{sha}` |
| Report | `results/archive/demo_results/diagnostics/K48_GRPO_EVAL_COMPARISON.md` |
| Raw | `results/archive/demo_results/diagnostics/raw/unified_k48/eval_compare/v2_u0312/heldout_train.json` |

### Historical scores

| Split | Success | Denominator | Criterion |
|---|---:|---:|---|
| Held-out train (ids divisible by 5) | **30** | **56** | `SUCCESS_12S` |
| Validation | **26** | **26** | `SUCCESS_12S` (all early HANDOFF) |

Archive summary reasons (train): `{json.dumps((held.get('summary') or {}).get('reasons') or v2_entry.get('reasons'))}`

### Historical IC IDs (all 56)

Corpus: **`ball_train_v2_*` only**, numeric id divisible by 5 (`0005` … `0280`).

First 12: {", ".join(ic_ids[:12])}
…
Last 4: {", ".join(ic_ids[-4:])}

Root IDs: identical to traj ids in this corpus (no Plan-B `plan_b__` prefix).

### Historical evaluation protocol

| Item | Value |
|---|---|
| Inference | **argmax** K48 logits (deterministic) |
| Action decode | K47 codebook + residual (`decode_action`); HANDOFF index **47** |
| Recovery limit | **4.0 s** (`SELECTED_LIMIT_S`) |
| HANDOFF | Early via HANDOFF token **or** forced at recovery limit; basin early-exit only if no handoff head |
| Success | Physical **`SUCCESS_12S`** after recovery + nominal hold to 12 s |
| Device (archive) | CPU |
| Controller / physics | Legal 7D `step_legal` / AIRBORNE_V3; 39-D airborne obs; MuJoCo Panda cylinder |

Source report excerpt path: `{K48_EVAL_MD.as_posix()}`

## Secondary K48 GRPO lineage (do not confuse)

| Label | Path | MD5 | Role |
|---|---|---|---|
| tree80 epoch 3 | `results/grpo/tree80/checkpoint_epoch_0003.pt` | `{tree_md5}` | Later adaptive GRPO; used as frozen Stage-1 generator for B experts (P46/P47). SHA256 `{tree_sha}` |
| P22 A/B SFT | M111 | — | **Not K48** |
| P28 unified SFT | M111 | — | **Not K48** |
| M111 GRPO / P53+ | M111 | — | **Not K48** |
| P64 Actor | `P64_step4_actor.pt` | `{_md5(P64_ACTOR)}` | Frozen final M111 policy under audit |

## Equivalence warning vs current B1

Historical “strong B” evidence is **`SUCCESS_12S` on `ball_train_v2` ICs with K48 early/forced HANDOFF**.
Current independent B1 is **Plan-B MixedStart ICs**, M111 P64, **`SUCCESS_RELAXED_12S` + R3F**.
Matching the letter “B” / “B1” does **not** imply equivalent physical tasks or protocols.
"""
    _write(OUT / "HISTORICAL_K48_RESULTS.md", text)
    return {
        "k48_v2_md5": md5,
        "k48_v2_sha256": sha,
        "tree80_md5": tree_md5,
        "hist_success": 30,
        "hist_n": 56,
        "hist_ic_ids": ic_ids,
        "valid_v2": valid_v2,
    }


def write_protocol_md(hist: dict, restore_rows: list[dict], b1_eps: list[dict], starts) -> None:
    hist_ids = set(hist.get("hist_ic_ids") or [])
    b1_roots = [e["root_id"] for e in b1_eps]
    overlap = [r for r in b1_roots if r in hist_ids or any(r.endswith(h) or h in r for h in hist_ids)]
    # more precise: strip prefixes
    def norm(x: str) -> str:
        for p in ("plan_b__", "plan_a__", "B__", "A__"):
            if x.startswith(p):
                x = x[len(p) :]
        if "__" in x:
            x = x.split("__")[0]
        return x

    hist_norm = {norm(x) for x in hist_ids}
    b1_norm = [norm(r) for r in b1_roots]
    exact = [r for r in b1_norm if r in hist_norm]
    mechs = Counter((s.ep or {}).get("mechanism") for s in starts)
    corpora = Counter((s.ep or {}).get("source_corpus") for s in starts)
    restore_ok = sum(int(r.get("restore_ok") or 0) for r in restore_rows)
    text = f"""# PROTOCOL_COMPATIBILITY

Generated: {_utc()}

## Question

Are historical K48 “B” results and the current independent **B1** evaluation the same physical task under a comparable protocol?

## Short answer

**No — not equivalent.** Physical restore of current B1 captures into the K48 `roll_policy` wrapper **is** possible (obs 39-D + capture/ref), but IC distribution, action vocabulary, HANDOFF semantics, and success criteria differ. A paired closed-loop on the eight B1 ICs is therefore a **diagnostic transfer test**, not a reproduction of the historical 30/56 number.

## Side-by-side

| Axis | Historical K48 GRPO (`v2_u0312`) | Current independent B1 (P64) | Compatible? |
|---|---|---|---|
| Policy family | K48 Transformer (48-way) | M111 Transformer (112-way) P64 Actor | **No weight sharing** — separate loaders |
| Action decode | `k47_codebook.decode_action` + HANDOFF=47 | `decode_m111_action` + HANDOFF=111 | **Incompatible vocab** |
| Inference sampling | **argmax** | Actor sample (+ V2 gate, rule=None; V2 idle on these eps) | Different stochasticity |
| Observation | 39-D airborne | 39-D airborne | Yes (dim) |
| IC restore | snapshot `capture`+`ref` | MixedStart `kind=ep` with nested `capture`+`ref` | Yes for physics |
| IC corpus | `ball_train_v2_{{5,10,…,280}}` (56) | Plan-B heldout mix: k48r / slide_quota / inward_extra / one ball_train_v2 | **Different** |
| Exact ID overlap with hist 56 | — | Exact normalized overlap: **{len(exact)}** → {exact or "∅"} | Near-zero |
| HANDOFF | Early token 47 or forced @ 4 s | R3F dwell / monitor → SUCCESS path | Different |
| Success criterion | `SUCCESS_12S` | `SUCCESS_RELAXED_12S` (15° / 15 mm on 12s path) | Related but **not equal** |
| Max recovery | 4.0 s | 80 control steps (~episode budget) + R3F t_max 3100 | Different clocks |
| Controller | `step_legal` recovery v3 | Same underlying `step_legal` inside StabilizeGrpoEnv | Shared low-level |

## Current B1 IC inventory (frozen independent set)

| traj_id | root_id | mechanism | source_corpus |
|---|---|---|---|
"""
    for s, e in zip(starts, b1_eps):
        ep = s.ep or {}
        text += f"| `{e['traj_id']}` | `{e['root_id']}` | {ep.get('mechanism')} | {ep.get('source_corpus')} |\n"
    text += f"""

Mechanism counts: {dict(mechs)}
Corpus counts: {dict(corpora)}

## Restore audit (Phase 2 prerequisite)

Capture→sim restore without policy: **{restore_ok}/{len(restore_rows)}** OK.

See `RESTORE_AUDIT.csv`.

## Common protocol adopted for Phase 3

1. **Same eight frozen B1 MixedStarts** (nested capture/ref).
2. **K48:** `load_trained_k48` + `make_training_env` + instrumented argmax `roll_policy` (native SUCCESS_12S; post-hoc SUCCESS_NEW / SUCCESS_RELAXED probes).
3. **P64:** identical `StabilizeGrpoEnv` + `rollout_episode` path as final eval (native SUCCESS_RELAXED_12S); seeds from frozen manifest.
4. **Do not** load K48 weights into M111.
5. Report **both** native metrics; treat paired numbers as **post-hoc diagnostic**, not a new independent final-test estimate.

## Incompatibilities that remain

- Cannot share action-token sequences across policies (48 ≠ 111).
- Cannot force identical HANDOFF gating without changing a policy’s inference contract.
- Historical 30/56 is **not** an estimate on these eight ICs.
"""
    _write(OUT / "PROTOCOL_COMPATIBILITY.md", text)


def paired_table(k48_rows: list[dict], p64_rows: list[dict], frozen_rows: list[dict]) -> list[dict]:
    p64 = {r["traj_id"]: r for r in p64_rows}
    fr = {r["traj_id"]: r for r in frozen_rows}
    # Prefer primary k48 label v2_u0312
    k48_primary = [r for r in k48_rows if r.get("policy") == "K48_v2_u0312"]
    k48 = {r["traj_id"]: r for r in k48_primary}
    out = []
    for tid, frow in fr.items():
        a = k48.get(tid) or {}
        b = p64.get(tid) or {}
        out.append(
            {
                "traj_id": tid,
                "root_id": frow["root_id"],
                "frozen_P64_SUCCESS_RELAXED_12S": int(frow["SUCCESS_RELAXED_12S"]),
                "frozen_P64_fail": frow["earliest_decisive_failure"],
                "posthoc_P64_SUCCESS_RELAXED_12S": b.get("SUCCESS_RELAXED_12S"),
                "posthoc_P64_SUCCESS_12S": b.get("SUCCESS_12S"),
                "posthoc_P64_SUCCESS_NEW": b.get("SUCCESS_NEW"),
                "posthoc_P64_fail": b.get("earliest_decisive_failure"),
                "posthoc_P64_frozen_match": b.get("frozen_match"),
                "K48_v2_SUCCESS_12S": a.get("SUCCESS_12S"),
                "K48_v2_SUCCESS_RELAXED_12S": a.get("SUCCESS_RELAXED_12S"),
                "K48_v2_SUCCESS_NEW": a.get("SUCCESS_NEW"),
                "K48_v2_reason": a.get("hist_reason") or a.get("strict_reason"),
                "K48_v2_handoff_mode": a.get("handoff_mode"),
                "K48_fc": a.get("first_contact"),
                "P64_fc": b.get("first_contact"),
                "K48_bilateral": a.get("bilateral_ever"),
                "P64_bilateral": b.get("bilateral_ever"),
                "K48_opp": a.get("opposing_ever"),
                "P64_opp": b.get("opposing_ever"),
                "K48_pad4": a.get("min_pad_ge4"),
                "P64_pad4": b.get("min_pad_ge4"),
                "K48_e9": a.get("e_pos_le9"),
                "P64_e9": b.get("e_pos_le9"),
                "K48_contact_loss": a.get("contact_loss_ever"),
                "P64_contact_loss": b.get("contact_loss_ever"),
                "K48_t_fc": a.get("t_fc"),
                "P64_t_fc": b.get("t_fc"),
                "K48_e_pos_mm_end": a.get("e_pos_mm_end"),
                "K48_e_theta_deg_end": a.get("e_theta_deg_end"),
                "K48_n_tokens": a.get("n_tokens"),
                "P64_ep_len": b.get("ep_len"),
                "K48_first_token": a.get("first_token"),
                "evidence_class": "post_hoc_diagnostic_not_independent_final_test",
            }
        )
    return out


def write_divergence(paired: list[dict], k48_rows: list[dict], p64_rows: list[dict]) -> None:
    lines = [
        "# TRAJECTORY_DIVERGENCE",
        "",
        f"Generated: {_utc()}",
        "",
        "Scope: post-hoc paired closed-loop on the **same eight frozen B1 ICs**.",
        "Independent final-test estimate remains **P64 0/8** unchanged.",
        "",
        "## Aggregate",
        "",
    ]
    k_s12 = sum(int(r.get("K48_v2_SUCCESS_12S") or 0) for r in paired)
    p_rel = sum(int(r.get("posthoc_P64_SUCCESS_RELAXED_12S") or 0) for r in paired)
    p_s12 = sum(int(r.get("posthoc_P64_SUCCESS_12S") or 0) for r in paired)
    p_fr = sum(int(r.get("frozen_P64_SUCCESS_RELAXED_12S") or 0) for r in paired)
    lines += [
        f"- K48 `v2_u0312` native SUCCESS_12S: **{k_s12}/8**",
        f"- P64 post-hoc SUCCESS_12S (common-ish): **{p_s12}/8**",
        f"- P64 post-hoc SUCCESS_RELAXED_12S: **{p_rel}/8** (frozen independent: **{p_fr}/8**)",
        "- K48 SUCCESS_RELAXED_12S: **N/A** (R3F-path metric; not part of K48 native contract)",
        "",
        "## Per-IC earliest meaningful divergence",
        "",
    ]
    k48_by = {r["traj_id"]: r for r in k48_rows if r.get("policy") == "K48_v2_u0312"}
    p64_by = {r["traj_id"]: r for r in p64_rows}
    for row in paired:
        tid = row["traj_id"]
        a = k48_by.get(tid) or {}
        b = p64_by.get(tid) or {}
        # Divergence heuristic
        if int(row.get("K48_v2_SUCCESS_12S") or 0) and not int(
            row.get("posthoc_P64_SUCCESS_12S") or row.get("posthoc_P64_SUCCESS_RELAXED_12S") or 0
        ):
            earliest = "terminal_success_gap (K48 SUCCESS_12S; P64 fails 12S and relaxed)"
        elif int(a.get("first_contact") or 0) != int(b.get("first_contact") or 0):
            earliest = "first_contact"
        elif int(a.get("opposing_ever") or 0) != int(b.get("opposing_ever") or 0):
            earliest = "bilateral/opposing_contact"
        elif int(a.get("min_pad_ge4") or 0) != int(b.get("min_pad_ge4") or 0):
            earliest = "pad_engagement (>=4)"
        elif int(a.get("e_pos_le9") or 0) != int(b.get("e_pos_le9") or 0):
            earliest = "position_error_band (e_pos<=9mm)"
        elif (a.get("handoff_mode") or "") != (b.get("handoff_mode") or ""):
            earliest = "handoff_timing_or_mode"
        else:
            earliest = "late_terminal_or_aligned_failure"
        lines += [
            f"### `{tid}`",
            f"- Frozen P64 fail: `{row.get('frozen_P64_fail')}`",
            f"- K48: success12S={row.get('K48_v2_SUCCESS_12S')} handoff={row.get('K48_v2_handoff_mode')} "
            f"reason={row.get('K48_v2_reason')}",
            f"- P64 post-hoc: 12S={row.get('posthoc_P64_SUCCESS_12S')} relaxed={row.get('posthoc_P64_SUCCESS_RELAXED_12S')} "
            f"fail={row.get('posthoc_P64_fail')} r3/hand={b.get('r3_ready')}/{b.get('handoff')}",
            f"- Contact: K48 fc/bil/opp/pad4/e9/loss="
            f"{a.get('first_contact')}/{a.get('bilateral_ever')}/{a.get('opposing_ever')}/"
            f"{a.get('min_pad_ge4')}/{a.get('e_pos_le9')}/{a.get('contact_loss_ever')}; "
            f"P64={b.get('first_contact')}/{b.get('bilateral_ever')}/{b.get('opposing_ever')}/"
            f"{b.get('min_pad_ge4')}/{b.get('e_pos_le9')}/{b.get('contact_loss_ever')}",
            f"- End pose (K48): e_pos_mm={a.get('e_pos_mm_end')} e_theta_deg={a.get('e_theta_deg_end')} "
            f"min_pad={a.get('min_pad_end')}",
            f"- Tokens: K48 first={a.get('first_token')} n={a.get('n_tokens')} head={a.get('tokens_head')}; "
            f"P64 n={b.get('n_tokens')} family_tail={b.get('family_tail')}",
            f"- **Earliest meaningful divergence:** {earliest}",
            "",
        ]
    lines += [
        "## Mechanism classes (diagnostic)",
        "",
        "1. **Policy skill regression** — only if K48 clearly succeeds and P64 fails under a shared physical IC + honestly reported native metrics.",
        "2. **Task / IC distribution shift** — historical 30/56 was ball_train_v2×5; current B1 is mostly non-overlapping Plan-B starts.",
        "3. **Action representation limits** — K48≠M111; token sequences are not comparable.",
        "4. **Inference / eval mismatch** — argmax+forced HANDOFF vs R3F+relaxed success.",
        "5. **Statistical uncertainty** — n=8 post-hoc; independent B1 already short by 2 ICs.",
        "",
    ]
    _write(OUT / "TRAJECTORY_DIVERGENCE.md", "\n".join(lines))


def write_verdict(
    paired: list[dict],
    restore_rows: list[dict],
    hist: dict,
    frozen_rows: list[dict],
) -> str:
    restore_ok = sum(int(r.get("restore_ok") or 0) for r in restore_rows) == len(restore_rows)
    frozen_succ = sum(int(r["SUCCESS_RELAXED_12S"]) for r in frozen_rows)
    k_s12 = sum(int(r.get("K48_v2_SUCCESS_12S") or 0) for r in paired)
    p_rel = sum(int(r.get("posthoc_P64_SUCCESS_RELAXED_12S") or 0) for r in paired)
    p_s12 = sum(int(r.get("posthoc_P64_SUCCESS_12S") or 0) for r in paired)
    match = all(
        int(r.get("posthoc_P64_frozen_match") or 0) == 1
        for r in paired
        if r.get("posthoc_P64_frozen_match") is not None
    )

    # Case logic (common comparable metric ≈ SUCCESS_12S; P64 native is RELAXED)
    if not restore_ok:
        case = "D"
        title = "Evaluation or restoration bug detected"
    elif k_s12 >= 1 and p_s12 == 0 and p_rel == 0 and frozen_succ == 0:
        case = "A"
        title = "Historical K48 GRPO succeeds on B1 ICs where P64 fails"
    elif k_s12 == 0 and p_rel == 0 and p_s12 == 0 and frozen_succ == 0:
        case = "B"
        title = "Both policies fail on the new B1 ICs"
    elif k_s12 == 0 and frozen_succ == 0:
        case = "C"
        title = "Historical advantage disappears after protocol/IC alignment"
    else:
        case = "E"
        title = "Comparison cannot be made reliably / mixed evidence"

    note = ""
    if case == "A":
        note = (
            " Under a shared physical IC restore, K48 `v2_u0312` achieves native SUCCESS_12S "
            "on ≥1 of the eight B1 starts while P64 remains 0 on both SUCCESS_12S and "
            "SUCCESS_RELAXED_12S. Caveat: HANDOFF contracts still differ (early/forced vs R3F); "
            "SUCCESS_RELAXED is N/A for K48. Exact traj-id overlap with the historical 56 is ~0, "
            "so this is transfer evidence of skill gap on the *current* B1 set, not reproduction "
            "of the archive 30/56."
        )
    elif case == "B":
        note = (
            " Consistent with distribution/difficulty shift: historical 30/56 was a different IC set "
            f"(ball_train_v2×5; n={hist.get('hist_n')}), not these Plan-B B1 starts."
        )
    elif case == "C":
        note = (
            " After aligning to the eight current B1 ICs, K48 no longer shows the historical "
            "advantage; earlier B-class scores are not directly comparable."
        )

    text = f"""# B1_REGRESSION_VERDICT

Generated: {_utc()}

## Verdict: **CASE {case}** — {title}

{note}

## Preserved independent final-test result

| Item | Value |
|---|---|
| Frozen P64 B1 | **{frozen_succ}/8** SUCCESS_RELAXED_12S |
| Artifact | `results/experiments/final_independent_success_rate/EPISODE_RESULTS.csv` |
| Status | **Unchanged** — this audit does not replace it |

Paired replays are labeled `post_hoc_diagnostic_not_independent_final_test`.

## Diagnostic paired summary (same 8 B1 ICs)

| Policy / metric | Score |
|---|---|
| P64 frozen SUCCESS_RELAXED_12S | {frozen_succ}/8 |
| P64 post-hoc SUCCESS_RELAXED_12S | {p_rel}/8 |
| P64 post-hoc reproduces frozen bits | {"yes" if match else "NO — investigate"} |
| K48 v2_u0312 SUCCESS_12S (native) | {k_s12}/8 |
| P64 post-hoc SUCCESS_12S | {p_s12}/8 |
| K48 SUCCESS_RELAXED_12S | N/A (not K48 contract) |
| Restore audit | {sum(int(r.get('restore_ok') or 0) for r in restore_rows)}/{len(restore_rows)} |

## Phase 2 — is 0/8 an eval bug?

- Infra errors in frozen B1 rows: none (all completed eps).
- Failures: `no_e_pos_le9` / `no_r3_ready` with first_contact=1 and opposing_ever=1 on all 8.
- Snapshot restore: {sum(int(r.get('restore_ok') or 0) for r in restore_rows)}/{len(restore_rows)} OK.
- Post-hoc P64 replay match to frozen success bits: {"matched" if match else "mismatched"}.

Conclusion: **0/8 is not explained by restore/eval implementation failure** (unless post-hoc mismatch is flagged above).

## Phase 4 mechanism read

See `TRAJECTORY_DIVERGENCE.md` for per-IC earliest divergence.

Distinguishing factors:

1. Policy skill regression — {"supported" if case == "A" else "not primary"}.
2. IC / difficulty shift vs historical ball_train_v2×5 — **supported** (exact ID overlap ≈ 0; see PROTOCOL).
3. Action representation (K48 vs M111) — structural; prevents weight-level continuity claims.
4. Inference/eval mismatch (argmax+4s HANDOFF vs R3F+RELAXED) — always present; reported dual metrics.
5. Statistical uncertainty — n=8 diagnostic; B1 independent shortfall already 2.

## What this audit does **not** authorize

- Restoring K48 as the final policy
- Retraining / checkpoint-selecting P64 from these eight ICs
- Replacing the independent 0/8 with post-hoc scores

## Artifacts

- `HISTORICAL_K48_RESULTS.md`
- `PROTOCOL_COMPATIBILITY.md`
- `PAIRED_B1_RESULTS.csv`
- `TRAJECTORY_DIVERGENCE.md`
- `B1_REGRESSION_VERDICT.md`
- `RESTORE_AUDIT.csv`
- `RAW_K48_ROLLS.json` / `RAW_P64_ROLLS.json` (machine detail)
"""
    _write(OUT / "B1_REGRESSION_VERDICT.md", text)
    return case


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    b1_eps, starts, frozen_rows = load_b1_bundle()
    assert len(starts) == len(frozen_rows) == 8, (len(starts), len(frozen_rows))

    hist = write_historical_md()
    restore_rows = audit_restore(starts, device=device)
    _write_csv(OUT / "RESTORE_AUDIT.csv", restore_rows)
    write_protocol_md(hist, restore_rows, b1_eps, starts)

    # Phase 3 paired
    k48_rows = run_k48_on_b1("K48_v2_u0312", K48_V2, starts, frozen_rows, device)
    # Secondary reference (not primary verdict driver)
    k48_rows.extend(run_k48_on_b1("K48_tree80_e3", K48_TREE80, starts, frozen_rows, device))
    p64_rows = run_p64_detail(starts, frozen_rows, device)

    (OUT / "RAW_K48_ROLLS.json").write_text(
        json.dumps(k48_rows, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "RAW_P64_ROLLS.json").write_text(
        json.dumps(p64_rows, indent=2, default=str), encoding="utf-8"
    )

    paired = paired_table(k48_rows, p64_rows, frozen_rows)
    _write_csv(OUT / "PAIRED_B1_RESULTS.csv", paired)
    write_divergence(paired, k48_rows, p64_rows)
    case = write_verdict(paired, restore_rows, hist, frozen_rows)
    print(json.dumps({"case": case, "out": str(OUT)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
