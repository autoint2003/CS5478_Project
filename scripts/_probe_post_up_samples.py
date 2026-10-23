#!/usr/bin/env python
import csv
from pathlib import Path

rows = list(csv.DictReader((Path("results/diagnostics/seq4_post_up_reaccel") / "series.csv").open(encoding="utf-8")))
for tq in [17.76, 18.44, 18.66, 25.0, 30.92, 31.18, 31.42, 32.2, 35.0, 45.9]:
    r = min(rows, key=lambda x: abs(float(x["t"]) - tq))
    print(
        f"t={float(r['t']):6.2f} ph={r['phase']:16s} sub={r['subphase'][:32]:32s} "
        f"a={float(r['alpha_up_deg']):6.1f} wa={float(r['omega_act_norm']):.3f} "
        f"wc={float(r['omega_cmd_norm']):.3f} v={float(r['v_cmd_norm']):.3f}"
    )
