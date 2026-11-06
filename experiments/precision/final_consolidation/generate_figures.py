"""Generate report-ready figures from verified P70/P73/P81/P83 artifacts only."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Figures are result artifacts under results/; this script lives in experiments/.
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "experiments" / "final_consolidation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def fig01() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("Final frozen system (P64 Actor/Critic + P70 V2 + M111)", fontsize=12, pad=8)
    boxes = [
        (0.3, 3.6, 2.2, 1.4, "Observation\n(Actor 39-D)"),
        (3.0, 3.6, 2.4, 1.4, "P64 Actor\nM111 discrete\ntokens"),
        (5.8, 3.6, 2.6, 1.4, "P70 V2\nadaptive pre-contact\n(ready_score)"),
        (8.6, 3.6, 1.2, 1.4, "Env\nstep"),
        (3.0, 1.2, 2.4, 1.4, "P64 Critic\n(value; unused\nfor control)"),
        (5.8, 1.2, 2.6, 1.4, "Terminal metrics\nR3F + SUCCESS_\nRELAXED_12S"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(
            plt.Rectangle((x, y), w, h, fill=True, facecolor="#eef3f8", edgecolor="#1f3a5f", lw=1.4)
        )
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8.5)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#1f3a5f", lw=1.3),
        )

    arrow(2.5, 4.3, 3.0, 4.3)
    arrow(5.4, 4.3, 5.8, 4.3)
    arrow(8.4, 4.3, 8.6, 4.3)
    arrow(4.2, 3.6, 4.2, 2.6)
    arrow(7.1, 3.6, 7.1, 2.6)
    ax.text(
        5.0,
        0.35,
        "Overlays (P73/P81) and obsolete train branches are NOT on this path.",
        fontsize=8,
        ha="center",
        color="#7a2e0b",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig01_final_system_architecture.png", dpi=160)
    fig.savefig(OUT / "fig01_final_system_architecture.pdf")
    plt.close()


def fig02() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.axis("off")
    ax.set_title("Skill-aware discrete action vocabulary (M111)", fontsize=12)
    families = [("APPROACH", "#4c78a8"), ("GRIP", "#f58518"), ("ORIENT", "#54a24b"), ("OTHER", "#b279a2")]
    for i, (name, col) in enumerate(families):
        ax.add_patch(
            plt.Rectangle(
                (0.5 + i * 2.0, 1.6), 1.7, 1.5, facecolor=col, alpha=0.35, edgecolor=col, lw=1.5
            )
        )
        ax.text(1.35 + i * 2.0, 2.35, name, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(
        4.5,
        0.7,
        "Book size = 111 tokens loaded from P64 Actor checkpoint\n"
        "V2 approach pool uses tokens {56, 50, 71}; grip uses skill-book GRIP family",
        ha="center",
        fontsize=9,
    )
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 4)
    fig.tight_layout()
    fig.savefig(OUT / "fig02_m111_action_representation.png", dpi=160)
    fig.savefig(OUT / "fig02_m111_action_representation.pdf")
    plt.close()


def fig03() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    ax.set_title("Training / evaluation pipeline (frozen inference)", fontsize=12)
    stages = [
        "P54 IC pool\n(blocked eval\nexcluded)",
        "P64 PPO\nActor/Critic\ncheckpoints",
        "P70 V2\npre-contact\ncontroller",
        "Canonical rollout\np82.collect.\nrollout_episode",
        "P83 panel\n32 ICs/class\nMSP report",
    ]
    for i, s in enumerate(stages):
        x = 0.3 + i * 1.95
        ax.add_patch(
            plt.Rectangle((x, 1.1), 1.7, 1.8, facecolor="#f4f6f8", edgecolor="#222", lw=1.2)
        )
        ax.text(x + 0.85, 2.0, s, ha="center", va="center", fontsize=8)
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x + 1.85, 2.0),
                xytext=(x + 1.7, 2.0),
                arrowprops=dict(arrowstyle="->", lw=1.2),
            )
    ax.text(
        5.0,
        0.4,
        "No P85; no new training; final-test pool remains untouched.",
        ha="center",
        fontsize=9,
        color="#7a2e0b",
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    fig.tight_layout()
    fig.savefig(OUT / "fig03_train_eval_pipeline.png", dpi=160)
    fig.savefig(OUT / "fig03_train_eval_pipeline.pdf")
    plt.close()


def fig04() -> None:
    classes = ["A1", "B1", "C", "D"]
    succ = [8 / 32, 6 / 32, 4 / 32, 17 / 32]
    counts = [8, 6, 4, 17]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(
        classes, succ, color=["#4c78a8", "#f58518", "#e45756", "#72b7b2"], edgecolor="#222"
    )
    ax.axhline(0.2734375, color="#333", ls="--", lw=1, label="Equal-weight MSP = 0.2734375")
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("SUCCESS_RELAXED rate")
    ax.set_title("P83 validated panel — per-class SUCCESS_RELAXED (n=32/class)")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{c}/32", ha="center", fontsize=9)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(
        0.5,
        -0.18,
        "Source: p83_recoverable_failure_mass/BASELINE_PANEL_VALIDATION.md\n"
        "Not a pristine final-test result.",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        ha="center",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig04_p83_per_class_success.png", dpi=160)
    fig.savefig(OUT / "fig04_p83_per_class_success.pdf")
    plt.close()


def fig05() -> None:
    steps = [
        "all",
        "first_contact",
        "opposing|fc",
        "pad>=4|opp",
        "e<=9|pad",
        "frame|e9",
        "r3|frame",
        "succ|r3",
    ]
    nums = [32, 29, 17, 15, 8, 6, 4, 4]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.plot(range(len(steps)), nums, marker="o", color="#e45756", lw=2)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Surviving episodes (Class C, n=32)")
    ax.set_title("Class C contact-formation funnel (P83 nested conditionals)")
    for i, n in enumerate(nums):
        ax.text(i, n + 0.6, str(n), ha="center", fontsize=8)
    ax.set_ylim(0, 36)
    ax.text(
        0.01,
        -0.22,
        "Source: CROSS_CLASS_FUNNEL.md (class C conditional_nested).",
        transform=ax.transAxes,
        fontsize=7.5,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig05_c_contact_formation_funnel.png", dpi=160)
    fig.savefig(OUT / "fig05_c_contact_formation_funnel.pdf")
    plt.close()


def fig06() -> None:
    modes = [
        "no_first_contact",
        "no_opposing_after_fc",
        "no_min_pad4",
        "no_e_pos_le9",
        "no_frame_ok",
        "no_r3_ready",
        "post_handoff",
    ]
    by = {
        "A1": [0, 1, 4, 13, 1, 5, 0],
        "B1": [2, 2, 5, 9, 4, 3, 1],
        "C": [3, 12, 2, 7, 2, 2, 0],
        "D": [0, 6, 4, 3, 1, 1, 0],
    }
    x = np.arange(len(modes))
    w = 0.2
    fig, ax = plt.subplots(figsize=(10, 4.6))
    cols = {"A1": "#4c78a8", "B1": "#f58518", "C": "#e45756", "D": "#72b7b2"}
    for i, sk in enumerate(["A1", "B1", "C", "D"]):
        ax.bar(x + (i - 1.5) * w, by[sk], width=w, label=sk, color=cols[sk], edgecolor="#222", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Exclusive earliest-decisive count")
    ax.set_title("Cross-class exclusive failure attribution (P83, n_fail=93/128)")
    ax.legend(fontsize=8)
    ax.text(
        0.01,
        -0.24,
        "Source: EXCLUSIVE_FAILURE_ATTRIBUTION.md. One label per failed episode.",
        transform=ax.transAxes,
        fontsize=7.5,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig06_cross_class_failure_attribution.png", dpi=160)
    fig.savefig(OUT / "fig06_cross_class_failure_attribution.pdf")
    plt.close()


def fig07() -> None:
    labels = [
        "P73 soft_approach\nΔSUCCESS (held n=96)",
        "P81 tok_56 matched\nmean ΔSUCCESS",
        "P81 tok_56 full-ep\nMSP Δ (n=32)",
    ]
    vals = [-0.09375, 0.0875, -0.25]
    colors = ["#e45756" if v < 0 else "#54a24b" for v in vals]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(labels, vals, color=colors, edgecolor="#222")
    ax.axhline(0, color="#333", lw=1)
    ax.set_ylabel("ΔSUCCESS / ΔMSP")
    ax.set_title("Matched counterfactual vs full-episode intervention outcomes")
    for i, v in enumerate(vals):
        ax.text(
            i,
            v + (0.015 if v >= 0 else -0.03),
            f"{v:+.4f}",
            ha="center",
            fontsize=9,
            va="bottom" if v >= 0 else "top",
        )
    ax.text(
        0.01,
        -0.22,
        "Sources: p73 P74_DECISION.json; p81 FINAL_REPORT.json / FULL_EPISODE_TRANSFER.md.\n"
        "Panels differ — not a single paired series.",
        transform=ax.transAxes,
        fontsize=7.5,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig07_matched_vs_full_episode.png", dpi=160)
    fig.savefig(OUT / "fig07_matched_vs_full_episode.pdf")
    plt.close()


if __name__ == "__main__":
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    fig07()
    print("wrote", sorted(p.name for p in OUT.glob("*")))
