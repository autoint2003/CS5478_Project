#!/usr/bin/env python
import csv
import json
from pathlib import Path

p = Path("results/diagnostics/seq4_to_up_brake_before_vertical")
rows = list(csv.DictReader((p / "to_up_series.csv").open(encoding="utf-8")))
print("n_series", len(rows))
print("--- approach sample ---")
step = max(1, len(rows) // 25)
for r in rows[::step]:
    print(
        f"t={float(r['t']):5.2f} ang={float(r['ang_err_deg']):6.2f} "
        f"a={float(r['alpha_up_deg']):6.2f} rate={float(r['rate']):.4f} "
        f"w={float(r['omega_closing'] or 0):.4f} coast={float(r['coast_deg'] or 0):.2f} "
        f"br={r['braking_latched']} g={r['guard']}"
    )
print("--- last 30 ---")
for r in rows[-30:]:
    print(
        f"t={float(r['t']):5.2f} ang={float(r['ang_err_deg']):6.2f} "
        f"a={float(r['alpha_up_deg']):6.2f} rate={float(r['rate']):.4f} "
        f"w={float(r['omega_closing'] or 0):.4f} coast={float(r['coast_deg'] or 0):.2f} "
        f"br={r['braking_latched']} stop={r['stop_reason']}"
    )
near = [r for r in rows if float(r["ang_err_deg"]) < 40]
print("near40 max_w", max(float(r["omega_closing"] or 0) for r in near) if near else None)
print("near40 min_ang", min(float(r["ang_err_deg"]) for r in near) if near else None)
# first time rate soft-capped below floor while ang>20
for r in rows:
    if float(r["ang_err_deg"]) < 35 and float(r["rate"]) < 0.05:
        print("first_soft_decel", {k: r[k] for k in ("t", "ang_err_deg", "rate", "omega_closing", "coast_deg")})
        break
z = [r for r in rows if float(r["rate"]) <= 1e-9]
if z:
    print("first_zero_rate", {k: z[0][k] for k in ("t", "ang_err_deg", "alpha_up_deg", "omega_closing", "coast_deg", "braking_latched", "stop_reason")})
# max omega while ang in (8, 30)
band = [r for r in rows if 8.0 < float(r["ang_err_deg"]) < 30.0]
if band:
    wmax = max(band, key=lambda r: float(r["omega_closing"] or 0))
    print("max_w_in_8_30", {k: wmax[k] for k in ("t", "ang_err_deg", "rate", "omega_closing", "coast_deg")})
r = json.loads((p / "RESULT.json").read_text(encoding="utf-8"))
print("to_up", json.dumps(r["to_up"], indent=2, default=str)[:2000])
print("acceptance", r["acceptance"])
print("prior_to_up", r.get("prior_to_up"))
