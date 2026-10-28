import csv
from pathlib import Path

p = Path("results/diagnostics/p43_r3_working_hold_v2/a1_contact_ablation/ROLLOUTS.csv")
rows = list(csv.DictReader(p.open(encoding="utf-8")))
for rule in ("R2", "Rc3", "Rc4", "Rc4F"):
    sub = [r for r in rows if r["rule"] == rule]
    s12 = sum(int(r["SUCCESS_12S"]) for r in sub)
    hand = sum(int(r["handoff"]) for r in sub)
    lost = [r for r in sub if "LOST_GRASP" in r["fail_category"]]
    print(
        rule,
        "12S",
        s12,
        "hand",
        hand,
        "P(12S|hand)",
        round(s12 / hand, 3) if hand else None,
        "lost",
        len(lost),
    )
    if lost:
        pads = []
        for r in lost:
            if r.get("hand_nL") in ("", None) or r.get("hand_nR") in ("", None):
                continue
            pads.append(min(int(r["hand_nL"]), int(r["hand_nR"])))
        print("  lost min_pad", pads)
        print(
            "  lost steps",
            [r["handoff_step"] for r in lost],
            "fb",
            [r.get("used_fallback") for r in lost],
        )

p43 = Path("results/diagnostics/p43_r3_working_hold_v2/EXPERT_ROLLOUTS.csv")
rows = list(csv.DictReader(p43.open(encoding="utf-8")))
for skill in ("A1", "B1", "C", "D"):
    for rule in ("R2", "R3"):
        sub = [r for r in rows if r["skill"] == skill and r["rule"] == rule and int(r["handoff"]) == 1]
        if not sub:
            continue
        s12 = sum(int(r["SUCCESS_12S"]) for r in sub)
        snew = sum(int(r["SUCCESS_NEW"]) for r in sub)
        print(
            f"P43 {skill} {rule}: hand={len(sub)} 12S={s12} P={s12/len(sub):.3f} NEW={snew}"
        )
