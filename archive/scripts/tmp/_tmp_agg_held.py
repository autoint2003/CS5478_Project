import pandas as pd
from pathlib import Path

OUT = Path("results/diagnostics/freefall_recatch_collision_aware_feedback")
h = pd.read_csv(OUT / "HELDOUT_GENERALIZATION.csv")


def as_bool(s):
    if pd.isna(s):
        return False
    if isinstance(s, (bool, int, float)):
        return bool(s) and s != 0
    t = str(s).strip().lower()
    return t in ("1", "true", "yes")


for c in ["open_loop", "orig_P", "orig_O", "corr_P", "corr_O"]:
    sub = h[h.controller == c]
    n = len(sub)
    rates = {}
    for col in ["SUCCESS_NEW", "first_bil", "jaw_entry", "housing_collision", "first_contact"]:
        if col not in sub.columns:
            rates[col] = "n/a"
            continue
        k = int(sub[col].map(as_bool).sum())
        rates[col] = f"{k}/{n}"
    print(c, rates)
