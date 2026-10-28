"""P20 — M111 SFT trainer migration, smoke tests, Baseline A/B train + closed-loop eval.

Does not modify Expert Oracle, M111 vocabulary, physics, or STRICT criteria.
Does not launch GRPO.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.observations.airborne_obs import observe_airborne  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from envs.simulation.capture import restore_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import (  # noqa: E402
    make_env,
    quantize_with_book,
)
from models.m111_checkpoint import (  # noqa: E402
    HANDOFF,
    LOGIT_K,
    MOTION_K,
    decode_m111_action,
    load_m111,
    load_m111_book,
)
from models.transformer_policy import DT_POLICY  # noqa: E402
from training.m111_dataset import build_samples, load_p19_manifest  # noqa: E402
from training.m111_sft import eval_token_metrics, smoke_train_tiny, train_m111  # noqa: E402
from training.skill_aware_action_vocab import DEFAULT_WEIGHTS  # noqa: E402

OUT = ROOT / "results" / "diagnostics" / "p20_m111_sft_baseline"
CFG_PATH = ROOT / "config" / "training" / "m111_sft_baseline.yaml"
P19 = ROOT / "results" / "diagnostics" / "p19_sft_packaging"
WINDOW = 160


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_cfg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


def load_record(record_path: str) -> dict:
    raw = pickle.loads((ROOT / record_path).read_bytes())
    acts = raw.get("actions_7d")
    if acts is None:
        acts = raw.get("actions")
        meta = raw.get("meta") or {}
        if acts is None:
            acts = meta.get("actions")
    return {
        "actions": np.asarray(acts if acts is not None else [], np.float32).reshape(-1, 7),
        "capture": raw.get("capture"),
        "ref": raw.get("ref"),
        "phases": list(raw.get("phases") or []),
    }


def run_interface_tests() -> str:
    import pytest

    report = []
    report.append("# M111_INTERFACE_TESTS")
    report.append("")
    report.append(f"Generated: {_utc()}")
    report.append("")
    ret = pytest.main(["-q", str(ROOT / "tests" / "test_m111_interface.py")])
    report.append(f"pytest exit code: **{ret}** (`0` = all passed)")
    report.append("")
    report.append("Covered:")
    report.append("- 112-way head / HANDOFF=111 / tokens in [0,111]")
    report.append("- Decode parity with hybrid_decode; HANDOFF not a control vector")
    report.append("- K48 checkpoint rejection")
    report.append("- P19 loader: obs 39-D, prev 7-D, no target leakage into inputs")
    if ret != 0:
        raise RuntimeError("M111 interface unit tests failed")
    return "\n".join(report)


def policy_step(model, book, rho, obs_seq, prev_seq, device):
    """One inference decision: returns token, residual, action_or_None (None=HANDOFF)."""
    obs_in = obs_seq[-WINDOW:]
    prev_in = prev_seq[-WINDOW:]
    with torch.no_grad():
        logits, resid, act = model(
            torch.as_tensor(np.stack(obs_in), device=device),
            torch.as_tensor(np.stack(prev_in), device=device),
        )
    tok = int(torch.argmax(logits[-1]).item())
    if tok == HANDOFF:
        return tok, np.zeros(7, np.float32), None
    if tok < 0 or tok >= MOTION_K:
        raise RuntimeError(f"illegal token {tok}")
    r = resid[-1].float().cpu().numpy().astype(np.float32)
    a = decode_m111_action(book, tok, r, rho=rho)
    return tok, r, a


def rollout_policy(
    env,
    model,
    book,
    rho,
    capture,
    ref,
    *,
    device,
    max_steps: int,
    stabilize_n: int,
) -> dict:
    from envs.outcomes.physical_recovery import TABLE_TOP, physical_pack

    restore_capture(env, capture)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)
    obs_seq: list[np.ndarray] = []
    prev_seq: list[np.ndarray] = []
    tokens: list[int] = []
    n_motion = 0
    premature = False
    timeout = False
    dropped = False
    reason = None

    for t in range(int(max_steps)):
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        tok, _r, action = policy_step(model, book, rho, obs_seq, prev_seq, device)
        tokens.append(tok)
        if action is None:
            if t == 0 and max_steps > 1:
                premature = True  # immediate HANDOFF; may still be correct for direct demos
            break
        step_legal(env.sim, action)
        env._prev = action
        n_motion += 1
        o = physical_pack(env.sim)
        if float(o.get("obj_z", 99)) < TABLE_TOP + 0.01:
            dropped = True
            reason = "DROP_DURING_POLICY"
            break
    else:
        timeout = True
        reason = "NEVER_HANDOFF_TIMEOUT"

    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=int(stabilize_n))
    return {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")) and not dropped,
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")) and not dropped,
        "eval_reason": reason or hand.get("std_reason") or hand.get("reason"),
        "n_motion": n_motion,
        "n_decisions": len(tokens),
        "premature_handoff": bool(premature and n_motion == 0),
        "never_handoff_timeout": bool(timeout),
        "dropped_during_policy": bool(dropped),
        "predicted_tokens_head": tokens[:16],
        "predicted_tokens_tail": tokens[-8:],
        "unique_tokens": len(set(tokens)),
        "failure_reason": reason
        or (None if hand.get("SUCCESS_NEW") else hand.get("std_reason") or hand.get("reason")),
    }


def replay_open_loop(env, capture, ref, actions, *, stabilize_n: int) -> dict:
    restore_capture(env, capture)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    acts = np.asarray(actions, np.float32).reshape(-1, 7)
    for a in acts:
        step_legal(env.sim, np.asarray(a, np.float32))
        env._prev = np.asarray(a, np.float32)
    hand = evaluate_handoff_with_axis(env.sim, ref, stabilize_n=int(stabilize_n))
    return {
        "SUCCESS_NEW": bool(hand.get("SUCCESS_NEW")),
        "SUCCESS_12S": bool(hand.get("SUCCESS_12S")),
        "eval_reason": hand.get("std_reason") or hand.get("reason"),
        "n_motion": int(len(acts)),
        "n_decisions": int(len(acts)),
        "premature_handoff": False,
        "never_handoff_timeout": False,
        "failure_reason": None if hand.get("SUCCESS_NEW") else hand.get("std_reason") or hand.get("reason"),
    }


def classify_failure(row: dict) -> str:
    if row.get("SUCCESS_NEW"):
        return "success"
    if row.get("never_handoff_timeout"):
        return "never_handoff_timeout"
    if row.get("premature_handoff") and int(row.get("n_motion") or 0) == 0:
        # Distinguish: expert was also direct handoff vs model aborted early on actionful demo
        if int(row.get("expert_n_actions") or 0) == 0:
            return "immediate_handoff_match_expert_zero"
        return "premature_handoff"
    reason = str(row.get("failure_reason") or row.get("eval_reason") or "")
    rl = reason.lower()
    if "drop" in rl or "table" in rl:
        return "contact_instability_or_drop"
    if "timeout" in rl:
        return "timeout"
    if int(row.get("n_motion") or 0) > int(row.get("expert_n_actions") or 0) + 50:
        return "excessive_recovery_actions"
    if int(row.get("expert_n_actions") or 0) > 0 and int(row.get("n_motion") or 0) < 5:
        return "premature_handoff"
    return f"physical_fail:{reason or 'unknown'}"


def maybe_render(env, out_path: Path, n_frames: int = 60) -> None:
    """Best-effort short video; skip silently if renderer unavailable."""
    try:
        import imageio.v2 as imageio
        from envs.simulation import render_utils  # type: ignore
    except Exception:
        try:
            # Fallback: skip
            return
        except Exception:
            return
    # Skip complex render if no helper — keep P20 unblocked
    _ = (imageio, render_utils, out_path, n_frames, env)
    return


def closed_loop_eval(
    *,
    val_rows: list[dict],
    book: np.ndarray,
    rho,
    models: dict[str, Any],
    cfg: dict,
    device,
) -> list[dict]:
    env = make_env()
    stabilize_n = int(cfg.get("stabilize_n", 8))
    max_steps = int(cfg.get("max_policy_steps", 2500))
    results = []
    for i, row in enumerate(val_rows):
        rec = load_record(row["record_path"])
        if rec["capture"] is None or rec["ref"] is None:
            continue
        tid = row["traj_id"]
        expert_n = int(row.get("n_actions_continuous") or len(rec["actions"]))
        # 1) Continuous expert
        c = replay_open_loop(env, rec["capture"], rec["ref"], rec["actions"], stabilize_n=stabilize_n)
        c.update(
            {
                "traj_id": tid,
                "condition": "continuous_expert",
                "source_lineage": row.get("source_lineage"),
                "skill_group": row.get("skill_group"),
                "expert_n_actions": expert_n,
                "split": "val",
            }
        )
        c["failure_class"] = classify_failure(c)
        results.append(c)
        # 2) M111 decoded expert
        q = quantize_with_book(book, rec["actions"], rho, w=DEFAULT_WEIGHTS)
        m = replay_open_loop(
            env, rec["capture"], rec["ref"], q["recon_actions"], stabilize_n=stabilize_n
        )
        m.update(
            {
                "traj_id": tid,
                "condition": "m111_expert",
                "source_lineage": row.get("source_lineage"),
                "skill_group": row.get("skill_group"),
                "expert_n_actions": expert_n,
                "split": "val",
            }
        )
        m["failure_class"] = classify_failure(m)
        results.append(m)
        # 3/4) SFT policies
        for name, model in models.items():
            model.eval()
            r = rollout_policy(
                env,
                model,
                book,
                rho,
                rec["capture"],
                rec["ref"],
                device=device,
                max_steps=max_steps,
                stabilize_n=stabilize_n,
            )
            r.update(
                {
                    "traj_id": tid,
                    "condition": name,
                    "source_lineage": row.get("source_lineage"),
                    "skill_group": row.get("skill_group"),
                    "expert_n_actions": expert_n,
                    "split": "val",
                    "meta_executes_dx": row.get("meta_executes_dx"),
                    "meta_executes_dz": row.get("meta_executes_dz"),
                    "meta_executes_coupled": row.get("meta_executes_coupled"),
                    "meta_is_direct_handoff_only": row.get("meta_is_direct_handoff_only"),
                    "meta_recovery_letter": row.get("meta_recovery_letter"),
                }
            )
            r["failure_class"] = classify_failure(r)
            results.append(r)
        if (i + 1) % 5 == 0:
            print(f"  eval {i+1}/{len(val_rows)} trajs", flush=True)
    return results


def write_training_configs_yaml(cfg: dict) -> None:
    text = yaml.safe_dump(
        {
            "shared": {k: v for k, v in cfg.items() if k != "baselines"},
            "baselines": cfg.get("baselines"),
            "notes": [
                "Architecture identical across A/B; only weight_scheme differs.",
                "Checkpoint selection uses validation token accuracy, not train loss alone.",
                "M111 vocabulary frozen; no K48 remapping.",
            ],
        },
        sort_keys=False,
    )
    (OUT / "TRAINING_CONFIGS.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    device = torch.device(cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        cfg["device"] = "cpu"
    print(f"=== P20 device={device} ===", flush=True)

    write_training_configs_yaml(cfg)

    # Phase 2/5 unit tests
    print("=== interface tests ===", flush=True)
    tests_md = run_interface_tests()
    _write_md(OUT / "M111_INTERFACE_TESTS.md", tests_md)

    book, rho, cb = load_m111_book(ROOT / cfg["m111_meta"])
    assert book.shape == (MOTION_K, 7)

    manifest = load_p19_manifest(ROOT / cfg["dataset_manifest"])
    print("=== build samples ===", flush=True)
    train_A = build_samples(manifest, split="train", scheme="natural")
    val_A = build_samples(manifest, split="val", scheme="natural")
    train_B = build_samples(manifest, split="train", scheme="lineage_length")
    val_B = build_samples(manifest, split="val", scheme="lineage_length")
    print(f"samples A train/val {len(train_A)}/{len(val_A)}; B {len(train_B)}/{len(val_B)}", flush=True)

    # Smoke
    print("=== smoke ===", flush=True)
    smoke = smoke_train_tiny(train_A, book, rho, device, steps=40)
    (OUT / "smoke_result.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    print("smoke OK", smoke, flush=True)

    # Train baselines
    metrics_rows = []
    models = {}
    best_paths = {}
    for key, bcfg in cfg["baselines"].items():
        name = bcfg["name"]
        scheme = bcfg["weight_scheme"]
        train_s = train_A if scheme == "natural" else train_B
        val_s = val_A if scheme == "natural" else val_B
        out_dir = OUT / "checkpoints" / name
        print(f"=== train {name} scheme={scheme} ===", flush=True)
        tcfg = {
            **{k: v for k, v in cfg.items() if k != "baselines"},
            "name": name,
            "device": str(device),
        }
        result = train_m111(
            train_samples=train_s,
            val_samples=val_s,
            cfg=tcfg,
            out_dir=out_dir,
            book=book,
            rho=rho,
        )
        for h in result["history"]:
            metrics_rows.append({"baseline": name, "weight_scheme": scheme, **h})
        best_paths[name] = result["best"]["path"]
        model, _b, _s = load_m111(out_dir / "checkpoint_best.pt", device)
        models[f"sft_{name}"] = model
        # Final confusion-ish report snippet
        final_val = eval_token_metrics(model, val_s, device)
        (out_dir / "final_val_metrics.json").write_text(
            json.dumps({k: v for k, v in final_val.items() if k != "pred_token_hist"}, indent=2),
            encoding="utf-8",
        )
        (out_dir / "pred_token_hist.json").write_text(
            json.dumps(final_val.get("pred_token_hist") or {}, indent=2), encoding="utf-8"
        )

    _write_csv(OUT / "TRAINING_METRICS.csv", metrics_rows)

    # Token confusion report
    conf_lines = [
        "# TOKEN_CONFUSION_REPORT",
        "",
        f"Generated: {_utc()}",
        "",
        "Step-level teacher-forced metrics on lineage-held-out validation windows.",
        "Not a substitute for closed-loop STRICT success.",
        "",
    ]
    for name, model in models.items():
        val_s = val_A if "natural" in name else val_B
        m = eval_token_metrics(model, val_s, device)
        conf_lines.append(f"## {name}")
        conf_lines.append("")
        conf_lines.append(f"- token_acc: **{m['token_acc']:.4f}**")
        conf_lines.append(f"- motion_token_acc: **{m['motion_token_acc']:.4f}**")
        conf_lines.append(f"- HANDOFF precision/recall: **{m['handoff_precision']:.3f}** / **{m['handoff_recall']:.3f}**")
        conf_lines.append(f"- n_steps: {m['n_steps']}")
        top = sorted((m.get("pred_token_hist") or {}).items(), key=lambda x: -x[1])[:15]
        conf_lines.append(f"- top predicted tokens: {top}")
        conf_lines.append("- per-skill acc (manifest skill_group):")
        for sk, acc in sorted((m.get("per_skill_acc") or {}).items(), key=lambda x: -x[1])[:12]:
            conf_lines.append(f"  - `{sk}`: {acc:.3f}")
        conf_lines.append("")
    _write_md(OUT / "TOKEN_CONFUSION_REPORT.md", "\n".join(conf_lines))

    # Closed-loop eval
    print("=== closed-loop eval ===", flush=True)
    val_trajs = [r for r in manifest["trajectories"] if r.get("split") == "val"]
    cl_rows = closed_loop_eval(
        val_rows=val_trajs,
        book=book,
        rho=rho,
        models=models,
        cfg=cfg,
        device=device,
    )
    # JSON-serialize lists
    for r in cl_rows:
        for k in ("predicted_tokens_head", "predicted_tokens_tail"):
            if isinstance(r.get(k), list):
                r[k] = "|".join(str(x) for x in r[k])
    _write_csv(OUT / "CLOSED_LOOP_EVALUATION.csv", cl_rows)

    # Aggregate comparison
    def agg(condition: str) -> dict:
        rows = [r for r in cl_rows if r.get("condition") == condition]
        n = len(rows)
        return {
            "condition": condition,
            "n": n,
            "strict_rate": sum(1 for r in rows if r.get("SUCCESS_NEW")) / max(1, n),
            "survival_12s_rate": sum(1 for r in rows if r.get("SUCCESS_12S")) / max(1, n),
            "premature_handoff_rate": sum(1 for r in rows if r.get("premature_handoff")) / max(1, n),
            "timeout_rate": sum(1 for r in rows if r.get("never_handoff_timeout")) / max(1, n),
            "mean_n_motion": float(np.mean([r.get("n_motion") or 0 for r in rows])) if rows else 0.0,
        }

    conditions = ["continuous_expert", "m111_expert"] + list(models.keys())
    aggs = [agg(c) for c in conditions]
    _write_csv(OUT / "CLOSED_LOOP_SUMMARY.csv", aggs)

    # Failure analysis
    fail_lines = [
        "# FAILURE_ANALYSIS",
        "",
        f"Generated: {_utc()}",
        "",
    ]
    for cond in models:
        rows = [r for r in cl_rows if r["condition"] == cond]
        fails = [r for r in rows if not r.get("SUCCESS_NEW")]
        fail_lines.append(f"## {cond}")
        fail_lines.append(f"- failures: {len(fails)} / {len(rows)}")
        fc = Counter(r.get("failure_class") for r in fails)
        for k, v in fc.most_common():
            fail_lines.append(f"  - `{k}`: {v}")
        # skill slices
        for tag, key in (
            ("direct0", "meta_is_direct_handoff_only"),
            ("dx", "meta_executes_dx"),
            ("dz", "meta_executes_dz"),
            ("coupled", "meta_executes_coupled"),
        ):
            sub = [r for r in rows if str(r.get(key)).lower() in ("true", "1")]
            if not sub:
                continue
            ok = sum(1 for r in sub if r.get("SUCCESS_NEW"))
            fail_lines.append(f"  - slice {tag}: STRICT {ok}/{len(sub)}")
        fail_lines.append("")

    fail_lines.extend(
        [
            "## Dominant limitation (judgment)",
            "",
            "See FINAL_REPORT for the ranked diagnosis after comparing A vs B and teacher baselines.",
            "",
        ]
    )
    _write_md(OUT / "FAILURE_ANALYSIS.md", "\n".join(fail_lines))

    # Baseline comparison
    a = agg("sft_A_natural")
    b = agg("sft_B_lineage_length")
    ce = agg("continuous_expert")
    me = agg("m111_expert")
    comp = f"""# BASELINE_COMPARISON

Generated: {_utc()}

## Closed-loop STRICT (held-out val lineages)

| Condition | N | STRICT | 12S | premature H | timeout | mean motion steps |
|---|---:|---:|---:|---:|---:|---:|
| Continuous Expert | {ce['n']} | {ce['strict_rate']:.3f} | {ce['survival_12s_rate']:.3f} | {ce['premature_handoff_rate']:.3f} | {ce['timeout_rate']:.3f} | {ce['mean_n_motion']:.1f} |
| M111 Expert | {me['n']} | {me['strict_rate']:.3f} | {me['survival_12s_rate']:.3f} | {me['premature_handoff_rate']:.3f} | {me['timeout_rate']:.3f} | {me['mean_n_motion']:.1f} |
| SFT A (natural) | {a['n']} | {a['strict_rate']:.3f} | {a['survival_12s_rate']:.3f} | {a['premature_handoff_rate']:.3f} | {a['timeout_rate']:.3f} | {a['mean_n_motion']:.1f} |
| SFT B (lineage/length) | {b['n']} | {b['strict_rate']:.3f} | {b['survival_12s_rate']:.3f} | {b['premature_handoff_rate']:.3f} | {b['timeout_rate']:.3f} | {b['mean_n_motion']:.1f} |

## Weighting effect

- Δ STRICT (B − A): **{b['strict_rate'] - a['strict_rate']:+.3f}**
- Δ mean motion steps (B − A): **{b['mean_n_motion'] - a['mean_n_motion']:+.1f}**

Higher motion steps with better STRICT suggests improved nontrivial recovery; higher motion with worse STRICT suggests meandering / failed precision.
"""
    _write_md(OUT / "BASELINE_COMPARISON.md", comp)

    # Final report answers
    # Teacher-forced learning signal
    ta = next(r for r in metrics_rows if r["baseline"] == "A_natural" and r["epoch"] == max(x["epoch"] for x in metrics_rows if x["baseline"] == "A_natural"))
    tb = next(r for r in metrics_rows if r["baseline"] == "B_lineage_length" and r["epoch"] == max(x["epoch"] for x in metrics_rows if x["baseline"] == "B_lineage_length"))

    # Diagnose dominant limitation
    gap_teacher = me["strict_rate"] - max(a["strict_rate"], b["strict_rate"])
    if max(a["strict_rate"], b["strict_rate"]) < 0.15 and me["strict_rate"] > 0.8:
        dominant = "4. Closed-loop distribution shift (teacher open-loop OK; policy fails closed-loop)"
    elif abs(b["strict_rate"] - a["strict_rate"]) > 0.05 and b["strict_rate"] > a["strict_rate"]:
        dominant = "1. Dataset imbalance (weighting helps — natural mix under-emphasizes actionful tapes)"
    elif max(a.get("val_token_acc") or 0, tb.get("val_token_acc") or 0) < 0.3:
        dominant = "3. Training configuration / underfitting"
    elif me["strict_rate"] < 0.9:
        dominant = "5. Missing expert skill coverage / discrete teacher gaps on val"
    else:
        dominant = "4. Closed-loop distribution shift (primary), with residual imbalance effects"

    final = f"""# FINAL_REPORT — P20 M111 SFT Baseline

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **Does the 112-way trainer learn the M111 distribution?**  
   Val token acc A={ta.get('val_token_acc'):.4f}, B={tb.get('val_token_acc'):.4f};  
   motion-token acc A={ta.get('val_motion_token_acc'):.4f}, B={tb.get('val_motion_token_acc'):.4f}.  
   See `TOKEN_CONFUSION_REPORT.md`.

2. **Can the policy learn when to HANDOFF?**  
   HANDOFF precision/recall A={ta.get('val_handoff_precision'):.3f}/{ta.get('val_handoff_recall'):.3f},  
   B={tb.get('val_handoff_precision'):.3f}/{tb.get('val_handoff_recall'):.3f}.  
   Closed-loop premature-H rates: A={a['premature_handoff_rate']:.3f}, B={b['premature_handoff_rate']:.3f};  
   timeout: A={a['timeout_rate']:.3f}, B={b['timeout_rate']:.3f}.

3. **Does lineage/length weighting help nontrivial recovery?**  
   STRICT A={a['strict_rate']:.3f} vs B={b['strict_rate']:.3f} (Δ={b['strict_rate']-a['strict_rate']:+.3f});  
   mean motion A={a['mean_n_motion']:.1f} vs B={b['mean_n_motion']:.1f}.

4. **Recognizable A/B/C and Precision skills?**  
   Slice STRICT rates in `FAILURE_ANALYSIS.md`. Teacher-forced per-skill acc in `TOKEN_CONFUSION_REPORT.md`.  
   Closed-loop skill execution is inferred from motion length + failure class (no privileged skill input).

5. **Physical STRICT on held-out lineages?**  
   Continuous={ce['strict_rate']:.3f}, M111 expert={me['strict_rate']:.3f},  
   SFT-A={a['strict_rate']:.3f}, SFT-B={b['strict_rate']:.3f} (N={ce['n']} val trajs).

6. **Useful GRPO initialization?**  
   {"Yes, as a discrete-compatible warm start if HANDOFF timing is competent and STRICT is non-trivial on direct-handoff slices." if max(a["strict_rate"], b["strict_rate"]) >= 0.25 or max(float(ta.get("val_handoff_recall") or 0), float(tb.get("val_handoff_recall") or 0)) >= 0.5 else "Marginal — prefer fixing closed-loop HANDOFF / motion competence before GRPO."}  
   **GRPO not started in P20.**

## Dominant limitation

{dominant}

Teacher−policy STRICT gap (M111 expert − best SFT): **{gap_teacher:+.3f}**

## Artifacts

- Checkpoints: `checkpoints/A_natural/`, `checkpoints/B_lineage_length/` (`checkpoint_best.pt`)
- Metrics: `TRAINING_METRICS.csv`, `CLOSED_LOOP_EVALUATION.csv`
- Audit: `TRAINER_MIGRATION_AUDIT.md`, `M111_INTERFACE_TESTS.md`

## Integrity

- Official corpus / M111 prototypes untouched
- K48 checkpoints refused by loader
- No Oracle or STRICT changes
- No GRPO
"""
    _write_md(OUT / "FINAL_REPORT.md", final)

    (OUT / "raw_p20_summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "device": str(device),
                "aggregates": aggs,
                "best_paths": {k: str(v) for k, v in best_paths.items()},
                "dominant": dominant,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("=== P20 done ===", f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
