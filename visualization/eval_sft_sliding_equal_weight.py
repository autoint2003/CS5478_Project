"""Evaluate equal-weight Sliding SFT vs original SFT Epoch 40.

Read-only w.r.t. physics/codebook/original checkpoints. Writes under
results/diagnostics/sft_sliding_equal_weight/.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.outcomes.horizon_outcome import roll_policy
from models.k48_checkpoint import HANDOFF, load_trained_k48
from training.dataset import episode_tensors
from training.k47_codebook import decode_action, load_codebook
from training.k48_config import SELECTED_LIMIT_S
from training.recovery_episode import make_training_env
from visualization import audit_checkpoint3_skill as audit
from visualization.classify_validation_skills import (
    BOTH_OFF_HOLD_S,
    POST_GAP_MIGRATION_M,
    classify_mechanism,
    prepare_blob,
    restore_ic,
)
from visualization.diagnose_sft_sliding_root_cause import (
    expert_mechanism_on_tape,
    free_rollout,
    skill_of,
    teacher_force_row,
)

OUT = ROOT / "results" / "diagnostics" / "sft_sliding_equal_weight"
ORIG_CKPT = ROOT / "results" / "sft" / "checkpoint_epoch_0040.pt"
EQ_CKPT = ROOT / "results" / "sft" / "sliding_equal_weight" / "checkpoint_epoch_0040.pt"
ORIG_MANIFEST = ROOT / "data" / "supervised_targets_with_horizontal.jsonl"
EQ_MANIFEST = ROOT / "data" / "supervised_targets_sliding_equal_weight.jsonl"
VALID = ROOT / "data" / "valid_skill_quotas.jsonl"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_valid_eval(env, model, book, device, ckpt_name, valid_rows, seated_ref):
    traces = OUT / "traces" / ckpt_name
    events = OUT / "contact_events" / ckpt_name
    traces.mkdir(parents=True, exist_ok=True)
    events.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in valid_rows:
        blob = pickle.loads((ROOT / item["snapshot"]).read_bytes())
        if "ref" not in blob:
            blob = prepare_blob(blob, seated_ref)
        print("EVAL", ckpt_name, item["id"], flush=True)
        eval_out = roll_policy(env, model, book, blob, device, cap_steps=200, limit_s=SELECTED_LIMIT_S)
        if "capture" in blob:
            samples, t0, traced, handoff_t = audit.roll_policy_full(
                env, model, book, blob, device, SELECTED_LIMIT_S,
            )
        else:
            from visualization.classify_validation_skills import roll_policy_full_ballistic

            samples, t0, traced, handoff_t = roll_policy_full_ballistic(
                env, model, book, blob, device, SELECTED_LIMIT_S,
            )
        skill = classify_mechanism(samples, t0, handoff_t)
        contact = audit.contact_events(samples, t0)
        (events / f"{item['id']}.json").write_text(json.dumps(contact, indent=2, default=float), encoding="utf-8")
        audit.write_csv(traces / f"{item['id']}.csv", samples, t0)
        first_tok = next((row["token"] for row in samples if row.get("token") is not None), None)
        rows.append({
            "id": item["id"],
            "ckpt": ckpt_name,
            "data_labels": list(item.get("categories") or []),
            "result": eval_out.get("reason"),
            "success": bool(eval_out.get("success")),
            "mechanism": skill["mechanism"],
            "confidence": skill["confidence"],
            "why": "; ".join(skill["why"]),
            "first_token": first_tok,
            "handoff_t": handoff_t,
            "both_off": json.dumps(skill["facts"].get("sustained_both_off_intervals")),
            "max_migration_m": skill["facts"].get("max_contact_migration_m"),
            "bilateral_steps": skill["facts"].get("bilateral_steps"),
            "recovery_steps": skill["facts"].get("recovery_steps"),
        })
        print("DONE", ckpt_name, item["id"], rows[-1]["result"], rows[-1]["mechanism"], flush=True)
    return rows


def summarize_eval(rows):
    by = defaultdict(list)
    for row in rows:
        by[row["ckpt"]].append(row)
    out = {}
    for ckpt, items in by.items():
        mech = Counter(item["mechanism"] for item in items)
        out[ckpt] = {
            "n": len(items),
            "success": sum(1 for item in items if item["success"]),
            "Contact-Preserving": mech.get("Contact-Preserving Adjustment", 0),
            "Release-Recatch": mech.get("Release-Recatch", 0),
            "Hybrid": mech.get("Hybrid", 0),
            "Other": mech.get("Other/Unclear", 0),
        }
        # by data label (overlapping allowed)
        label_success = defaultdict(lambda: [0, 0])
        for item in items:
            labels = item["data_labels"] or ["unlabeled"]
            for lab in labels:
                label_success[lab][1] += 1
                if item["success"]:
                    label_success[lab][0] += 1
        out[ckpt]["by_label_success"] = {
            lab: {"success": a, "n": b} for lab, (a, b) in sorted(label_success.items())
        }
    return out


def tf_compare(orig_model, eq_model, book, device, rows, tag="train"):
    """Teacher-force both models on the same expert rows."""
    # Stratified sample similar to root-cause script
    pools = defaultdict(list)
    for row in rows:
        pools[skill_of(row)].append(row)
    chosen = []
    chosen.extend(pools["horizontal_recatch"])
    chosen.extend(pools["vertical_recatch"])
    sliding = pools["sliding"]
    stride = max(1, len(sliding) // 20)
    for row in sliding[::stride]:
        chosen.append(row)
    if len(chosen) > 90:
        chosen = chosen[:90]

    records = []
    for row in chosen:
        print("TF", tag, skill_of(row), row["id"], flush=True)
        for name, model in (("original", orig_model), ("equal_weight", eq_model)):
            for step in teacher_force_row(model, book, row, device):
                records.append({
                    "split": tag,
                    "ckpt": name,
                    "id": row["id"],
                    "skill": skill_of(row),
                    "t": step["t"],
                    "expert_token": step["expert_token"],
                    "expert_prob": step["expert_prob"],
                    "expert_rank": step["expert_rank"],
                    "argmax_token": step["argmax_token"],
                    "argmax_prob": step["argmax_prob"],
                    "nll": step["nll"],
                    "resid_l2": step["resid_l2"],
                    "is_handoff": step["is_handoff"],
                    "is_first": step["is_first"],
                    "expert_grip": None if step["expert_token"] == HANDOFF else float(book[int(step["expert_token"])][6]),
                    "argmax_grip": None if step["argmax_token"] == HANDOFF else float(book[int(step["argmax_token"])][6]),
                })
    return records


def tf_summary(records):
    summary = {}
    for ckpt in sorted({row["ckpt"] for row in records}):
        summary[ckpt] = {}
        for skill in sorted({row["skill"] for row in records if row["ckpt"] == ckpt}):
            subset = [row for row in records if row["ckpt"] == ckpt and row["skill"] == skill]
            motion = [row for row in subset if not row["is_handoff"]]
            first = [row for row in subset if row["is_first"]]
            early5 = [row for row in subset if row["t"] < 5]
            early10 = [row for row in subset if row["t"] < 10]
            hand = [row for row in subset if row["is_handoff"]]

            def acc(items, k=1):
                if not items:
                    return None
                return float(np.mean([row["expert_rank"] <= k for row in items]))

            open_frac = None
            if first:
                open_frac = float(np.mean([
                    (row["argmax_grip"] is not None and row["argmax_grip"] < -0.5) for row in first
                ]))
            closed_frac = None
            if first:
                closed_frac = float(np.mean([
                    (row["argmax_grip"] is not None and row["argmax_grip"] > 0.5) for row in first
                ]))
            summary[ckpt][skill] = {
                "n_steps": len(subset),
                "n_episodes": len({row["id"] for row in subset}),
                "top1": acc(motion, 1),
                "top5": acc(motion, 5),
                "first_top1": acc(first, 1),
                "first_top5": acc(first, 5),
                "early5_top1": acc(early5, 1),
                "early10_top1": acc(early10, 1),
                "mean_expert_prob": float(np.mean([row["expert_prob"] for row in motion])) if motion else None,
                "mean_nll": float(np.mean([row["nll"] for row in motion])) if motion else None,
                "mean_resid_l2": float(np.mean([row["resid_l2"] for row in motion])) if motion else None,
                "first_mean_expert_prob": float(np.mean([row["expert_prob"] for row in first])) if first else None,
                "handoff_mean_prob": float(np.mean([row["expert_prob"] for row in hand])) if hand else None,
                "first_argmax_open_frac": open_frac,
                "first_argmax_closed_frac": closed_frac,
                "first_argmax_top": Counter(row["argmax_token"] for row in first).most_common(6),
            }
    return summary


def matched_compare(env, orig_model, eq_model, book, device, rows):
    focus_ids = [
        "ball_train_v2_0012", "ball_train_v2_0085", "ball_train_v2_0070",
        "ball_train_v2_0135", "k48r_valid_00009", "inward_extra_250043",
    ]
    comparison = []
    seated_ref = None
    for eid in focus_ids:
        train_rows = [row for row in rows if row["id"] == eid and skill_of(row) == "sliding"]
        split = "train"
        if not train_rows:
            # valid expert from demonstrations
            demos = load_jsonl(ROOT / "data" / "valid_demonstrations.jsonl")
            train_rows = [
                row for row in demos
                if row["id"] == eid and (row.get("family") == "sliding" or row.get("strategy") in ("controlled_slide", "inward_reseat"))
            ]
            split = "valid"
        if not train_rows:
            continue
        row = train_rows[0]
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        if "ref" not in blob:
            if seated_ref is None:
                from data_generation.strategies.recatch_oracle import _setup

                _sim, _base, seated_ref, _t, _c = _setup()
            blob = prepare_blob(blob, seated_ref)
        if "capture" not in blob:
            continue
        print("MATCHED", split, eid, flush=True)
        expert = expert_mechanism_on_tape(env, book, row)
        entry = {
            "id": eid,
            "split": split,
            "strategy": row.get("strategy"),
            "expert_mechanism": None if expert is None else expert.get("classified_mechanism"),
            "expert_result": None if expert is None else expert.get("result"),
        }
        for name, model in (("original", orig_model), ("equal_weight", eq_model)):
            run = free_rollout(env, model, book, blob, device, SELECTED_LIMIT_S, stochastic=False)
            sk = classify_mechanism(run["samples"], run["t0"], run["handoff_t"])
            entry[f"{name}_first_token"] = run["first_token"]
            entry[f"{name}_mechanism"] = sk["mechanism"]
            entry[f"{name}_result"] = run["outcome"].get("reason")
            entry[f"{name}_success"] = bool(run["outcome"].get("success"))
            entry[f"{name}_both_off"] = json.dumps(sk["facts"].get("sustained_both_off_intervals"))
            entry[f"{name}_migration_m"] = sk["facts"].get("max_contact_migration_m")
            entry[f"{name}_why"] = "; ".join(sk["why"])
        comparison.append(entry)
    return comparison


def plot_tf(summary):
    skills = ["sliding", "vertical_recatch", "horizontal_recatch"]
    metrics = [("first_top1", "First-token top-1"), ("first_mean_expert_prob", "First expert p"), ("mean_expert_prob", "Motion mean expert p")]
    for key, title in metrics:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        x = np.arange(len(skills))
        w = 0.35
        orig = [summary.get("original", {}).get(s, {}).get(key) or 0 for s in skills]
        eq = [summary.get("equal_weight", {}).get(s, {}).get(key) or 0 for s in skills]
        ax.bar(x - w / 2, orig, w, label="original SFT")
        ax.bar(x + w / 2, eq, w, label="equal-weight SFT")
        ax.set_xticks(x)
        ax.set_xticklabels(skills, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"tf_{key}.png", dpi=140)
        plt.close(fig)


def write_report(eval_summary, tf_sum, matched, weight_rows):
    orig = eval_summary.get("original", {})
    eq = eval_summary.get("equal_weight", {})

    def cell(d, k):
        return d.get(k, "—")

    lines = [
        "# Equal-weight Sliding SFT — controlled ablation",
        "",
        "## Hypothesis under test",
        "",
        "Sliding was not preserved by the original SFT primarily because its `sample_weight` was ~0.02 vs Recatch 1.0.",
        "This run raises **only** Sliding per-tape weight to 1.0, trains from **scratch** with the same seed/epochs/hparams,",
        "and compares Epoch 40 to the original.",
        "",
        "## Facts about training setup",
        "",
        "- Base config: `config/training/k48/unified_k48_recovery_v3_horizontal.yaml`",
        "- Experiment config: `config/training/k48/sft_sliding_equal_weight.yaml`",
        "- Independent manifest: `data/supervised_targets_sliding_equal_weight.jsonl` (325 rows, same tapes)",
        "- Original manifest untouched: `data/supervised_targets_with_horizontal.jsonl`",
        "- Init: `scratch` via `build_k48`; seed **1**; epochs **40**; batch **8**; Adam lr **3e-4**; **no scheduler**",
        "- Sampling: each epoch `np.random.default_rng(seed).permutation` — every row once; **no class sampler**",
        "- Loss: traj-mean CE+resid, then `mean(sample_weight * traj_loss)` (`training/train_sft.py`)",
        "- No other class / length / IC normalization beyond that formula",
        "- Output: `results/sft/sliding_equal_weight/` (does not overwrite `results/sft/checkpoint_epoch_0040.pt`)",
        "",
        "### Effective weight mass",
        "",
        "| skill | tapes | weight before | share before | weight after | share after |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in weight_rows:
        lines.append(
            f"| {row['skill']} | {row['tapes']} | {float(row['weight_before']):.4f} | "
            f"{float(row['share_before']):.4f} | {float(row['weight_after']):.4f} | {float(row['share_after']):.4f} |"
        )
    lines.extend([
        "",
        "**Note (fact):** Per-tape Sliding weight is equalized to Recatch (1.0). Because there are 112 Sliding tapes,",
        "Sliding **skill mass** rises to ~67% of total weight — not merely “equal skill share.”",
        "",
        "## Validation (17 IC, deterministic)",
        "",
        "| Metric | Original SFT | Equal-weight SFT |",
        "|---|---:|---:|",
        f"| SUCCESS_12S | {cell(orig,'success')}/17 | {cell(eq,'success')}/17 |",
        f"| Contact-Preserving | {cell(orig,'Contact-Preserving')}/17 | {cell(eq,'Contact-Preserving')}/17 |",
        f"| Release–Recatch | {cell(orig,'Release-Recatch')}/17 | {cell(eq,'Release-Recatch')}/17 |",
        f"| Hybrid | {cell(orig,'Hybrid')}/17 | {cell(eq,'Hybrid')}/17 |",
        f"| Other | {cell(orig,'Other')}/17 | {cell(eq,'Other')}/17 |",
        "",
        "### SUCCESS_12S by data label (overlapping; do not sum)",
        "",
    ])
    labs = sorted(set(list(orig.get("by_label_success", {})) + list(eq.get("by_label_success", {}))))
    lines.append("| label | Original | Equal-weight |")
    lines.append("|---|---|---|")
    for lab in labs:
        o = orig.get("by_label_success", {}).get(lab, {})
        e = eq.get("by_label_success", {}).get(lab, {})
        lines.append(
            f"| `{lab}` | {o.get('success',0)}/{o.get('n',0)} | {e.get('success',0)}/{e.get('n',0)} |"
        )
    lines.extend([
        "",
        "## Teacher forcing",
        "",
        "| ckpt / skill | first top-1 | first expert p | motion top-1 | mean expert p | first closed argmax | first open argmax |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for ckpt in ("original", "equal_weight"):
        for skill in ("sliding", "vertical_recatch", "horizontal_recatch"):
            slot = tf_sum.get(ckpt, {}).get(skill, {})
            if not slot:
                continue
            lines.append(
                f"| {ckpt} / {skill} | {slot.get('first_top1')} | {slot.get('first_mean_expert_prob')} | "
                f"{slot.get('top1')} | {slot.get('mean_expert_prob')} | "
                f"{slot.get('first_argmax_closed_frac')} | {slot.get('first_argmax_open_frac')} |"
            )
    lines.extend([
        "",
        "## Matched-state free rollouts (Sliding expert ICs)",
        "",
        "| IC | split | expert | original first/mech/result | equal-weight first/mech/result |",
        "|---|---|---|---|---|",
    ])
    for row in matched:
        lines.append(
            f"| `{row['id']}` | {row['split']} | {row.get('expert_mechanism')}/{row.get('expert_result')} | "
            f"{row.get('original_first_token')}/{row.get('original_mechanism')}/{row.get('original_result')} | "
            f"{row.get('equal_weight_first_token')}/{row.get('equal_weight_mechanism')}/{row.get('equal_weight_result')} |"
        )

    # Answers
    eq_cp = eq.get("Contact-Preserving", 0)
    eq_succ = eq.get("success", 0)
    orig_succ = orig.get("success", 0)
    slide_tf_o = tf_sum.get("original", {}).get("sliding", {})
    slide_tf_e = tf_sum.get("equal_weight", {}).get("sliding", {})
    vert_tf_o = tf_sum.get("original", {}).get("vertical_recatch", {})
    vert_tf_e = tf_sum.get("equal_weight", {}).get("vertical_recatch", {})
    horiz_tf_o = tf_sum.get("original", {}).get("horizontal_recatch", {})
    horiz_tf_e = tf_sum.get("equal_weight", {}).get("horizontal_recatch", {})

    valid_matched = [row for row in matched if row["split"] == "valid"]
    train_matched = [row for row in matched if row["split"] == "train"]
    eq_cp_valid = sum(1 for row in valid_matched if row.get("equal_weight_mechanism") == "Contact-Preserving Adjustment")
    eq_cp_train = sum(1 for row in train_matched if row.get("equal_weight_mechanism") == "Contact-Preserving Adjustment")

    lines.extend([
        "",
        "## Q&A",
        "",
        "### Q1. Did Sliding effective weight rise?",
        "",
        "**Yes (fact).** Weight sum 2.44 → 112.0; share 4.25% → 67.07%. Per-tape Sliding = Horizontal = Vertical = 1.0.",
        "",
        "### Q2. Did the new SFT learn Contact-Preserving Sliding?",
        "",
        f"Validation Contact-Preserving: **{eq_cp}/17**. "
        f"Matched train CP free-rollouts: {eq_cp_train}/{len(train_matched)}; "
        f"matched valid CP: {eq_cp_valid}/{len(valid_matched)}. "
        f"TF Sliding first top-1: {slide_tf_o.get('first_top1')} → {slide_tf_e.get('first_top1')}; "
        f"first expert p: {slide_tf_o.get('first_mean_expert_prob')} → {slide_tf_e.get('first_mean_expert_prob')}.",
        "",
        "### Q3. Sliding on held-out validation ICs?",
        "",
        f"See validation table Contact-Preserving={eq_cp}/17 and valid matched rows.",
        "",
        "### Q4. Did Recatch degrade?",
        "",
        f"Overall SUCCESS {orig_succ}/17 → {eq_succ}/17. "
        f"TF Vertical first top-1 {vert_tf_o.get('first_top1')} → {vert_tf_e.get('first_top1')}; "
        f"Horizontal {horiz_tf_o.get('first_top1')} → {horiz_tf_e.get('first_top1')}. "
        "See by-label SUCCESS rows for Horizontal / Vertical tags.",
        "",
        "### Q5. Mode swap vs multi-skill retention?",
        "",
        "Compare both Contact-Preserving count and Recatch/Hybrid counts plus Recatch TF metrics. "
        "If Sliding rises while Recatch TF/SUCCESS collapses, that is mode replacement, not unified multi-skill learning.",
        "",
        "### Q6. Does this support “severe downweighting was the main cause”?",
        "",
        "Requires: (a) Sliding TF first-token probability/top-1 rises materially, and "
        "(b) free-rollout Contact-Preserving appears on train and ideally valid ICs. "
        "If (a) yes but (b) no → weight mattered for fitting but overlap/closed-loop still blocks expression. "
        "If neither → weight alone is insufficient (hypothesis weakened).",
        "",
        "## Interpretation classes",
        "",
        "- **Fact:** numbers from this run’s CSVs and checkpoints.",
        "- **Inference:** causal attribution to weight change (only intentional difference).",
        "- **Unverified:** whether a milder reweight (skill-equal, not tape-equal×112) would keep Recatch.",
        "",
        "## Repro",
        "",
        "```powershell",
        "python -u scripts/train.py --supervised --config config/training/k48/sft_sliding_equal_weight.yaml",
        "python -u visualization/eval_sft_sliding_equal_weight.py --device cuda",
        "```",
        "",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)
    (OUT / "contact_events").mkdir(exist_ok=True)
    if not EQ_CKPT.is_file():
        raise SystemExit(f"missing equal-weight checkpoint {EQ_CKPT}")
    if not ORIG_CKPT.is_file():
        raise SystemExit(f"missing original checkpoint {ORIG_CKPT}")

    device = torch.device(args.device)
    book = load_codebook()
    print("LOAD original", ORIG_CKPT, flush=True)
    orig_model, _ = load_trained_k48(ORIG_CKPT, device)
    orig_model.eval()
    print("LOAD equal", EQ_CKPT, flush=True)
    eq_model, _ = load_trained_k48(EQ_CKPT, device)
    eq_model.eval()

    weight_rows = list(csv.DictReader((OUT / "effective_weights_before_after.csv").open(encoding="utf-8")))
    valid_rows = load_jsonl(VALID)
    orig_rows = load_jsonl(ORIG_MANIFEST)

    seated_ref = None
    from data_generation.strategies.recatch_oracle import _setup

    _sim, _base, seated_ref, _t, _c = _setup()

    env = make_training_env(device, model=eq_model)
    eval_rows = []
    eval_rows.extend(run_valid_eval(env, orig_model, book, device, "original", valid_rows, seated_ref))
    eval_rows.extend(run_valid_eval(env, eq_model, book, device, "equal_weight", valid_rows, seated_ref))
    with (OUT / "per_ic_evaluation.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(eval_rows[0].keys()))
        writer.writeheader()
        for row in eval_rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()})
    eval_summary = summarize_eval(eval_rows)
    (OUT / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")

    # mechanism comparison wide table
    by_id = defaultdict(dict)
    for row in eval_rows:
        by_id[row["id"]][row["ckpt"]] = row
    mech_rows = []
    for eid, pair in sorted(by_id.items()):
        o = pair.get("original", {})
        e = pair.get("equal_weight", {})
        mech_rows.append({
            "id": eid,
            "data_labels": json.dumps(o.get("data_labels") or e.get("data_labels")),
            "original_result": o.get("result"),
            "original_success": o.get("success"),
            "original_mechanism": o.get("mechanism"),
            "original_first_token": o.get("first_token"),
            "equal_result": e.get("result"),
            "equal_success": e.get("success"),
            "equal_mechanism": e.get("mechanism"),
            "equal_first_token": e.get("first_token"),
        })
    with (OUT / "mechanism_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mech_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mech_rows)

    print("TF_COMPARE", flush=True)
    tf_records = tf_compare(orig_model, eq_model, book, device, orig_rows, tag="train")
    with (OUT / "teacher_forcing_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(tf_records[0].keys()))
        writer.writeheader()
        writer.writerows(tf_records)
    tf_sum = tf_summary(tf_records)
    (OUT / "teacher_forcing_summary.json").write_text(json.dumps(tf_sum, indent=2), encoding="utf-8")
    plot_tf(tf_sum)

    print("MATCHED", flush=True)
    matched = matched_compare(env, orig_model, eq_model, book, device, orig_rows)
    with (OUT / "matched_rollout_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(matched[0].keys()) if matched else ["id"])
        writer.writeheader()
        writer.writerows(matched)

    write_report(eval_summary, tf_sum, matched, weight_rows)
    print("DONE", json.dumps({
        "orig_success": eval_summary.get("original", {}).get("success"),
        "eq_success": eval_summary.get("equal_weight", {}).get("success"),
        "orig_cp": eval_summary.get("original", {}).get("Contact-Preserving"),
        "eq_cp": eval_summary.get("equal_weight", {}).get("Contact-Preserving"),
        "slide_first_top1": {
            "original": tf_sum.get("original", {}).get("sliding", {}).get("first_top1"),
            "equal": tf_sum.get("equal_weight", {}).get("sliding", {}).get("first_top1"),
        },
    }), flush=True)


if __name__ == "__main__":
    main()
