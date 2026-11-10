import csv
from pathlib import Path

import numpy as np

OUT = Path("results/experiments/p100_frozen_token_feasibility_audit")
rows = list(csv.DictReader((OUT / "STEP_LEVEL_AUDIT.csv").open(encoding="utf-8")))
for r in rows:
    r["t"] = int(r["t"])
    r["early"] = int(r["early"])
    r["best_l2_under_z_hat"] = float(r["best_l2_under_z_hat"])
    r["a_err_p64_l2"] = float(r["a_err_p64_l2"])


def stats(name, sub):
    n = len(sub) or 1
    cats = {c: sum(1 for r in sub if r["category"] == c) for c in "ABCD"}
    feas = cats["A"] + cats["B"]
    a_err = np.array([r["a_err_p64_l2"] for r in sub if r["category"] == "A"], float)
    best = np.array([r["best_l2_under_z_hat"] for r in sub], float)
    print(f"== {name} n={len(sub)} ==")
    print(
        "  A/B/C/D",
        {k: f"{100 * v / n:.2f}%" for k, v in cats.items()},
        f"feas@zhat={100 * feas / n:.2f}%",
    )
    if len(a_err):
        print(
            "  A a_err_p64 L2: mean %.4f median %.4f p90 %.4f frac<0.05=%.3f <0.1=%.3f <0.2=%.3f"
            % (
                a_err.mean(),
                np.median(a_err),
                np.percentile(a_err, 90),
                (a_err < 0.05).mean(),
                (a_err < 0.1).mean(),
                (a_err < 0.2).mean(),
            )
        )
    print(
        "  best_l2@zhat: mean %.4f median %.4f p90 %.4f"
        % (best.mean(), np.median(best), np.percentile(best, 90))
    )


exp = [r for r in rows if r["source"] in ("b1_train_expert", "c_train_expert")]
stats("expert all", exp)
stats("expert early", [r for r in exp if r["early"] == 1])
stats("expert B1", [r for r in exp if r["skill"] == "B1"])
stats("expert C", [r for r in exp if r["skill"] == "C"])
stats("expert B1 early", [r for r in exp if r["skill"] == "B1" and r["early"] == 1])
stats("expert C early", [r for r in exp if r["skill"] == "C" and r["early"] == 1])
stats("student all", [r for r in rows if str(r["source"]).startswith("student_state")])
stats("student early", [r for r in rows if str(r["source"]).startswith("student_state") and r["early"] == 1])

print("== per-root expert early feas% ==")
roots = sorted({(r["skill"], r["phys_root"]) for r in exp})
for sk, pr in roots:
    sub = [r for r in exp if r["phys_root"] == pr and r["early"] == 1]
    n = len(sub) or 1
    feas = sum(1 for r in sub if r["category"] in "AB")
    cats = "".join(r["category"] for r in sorted(sub, key=lambda x: x["t"]))
    print(f"  {sk} {pr}: early_feas={100 * feas / n:.0f}% seq={cats}")
