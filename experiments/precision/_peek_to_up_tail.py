import csv
from pathlib import Path

rows = list(csv.DictReader(Path("results/diagnostics/seq4_to_up_overshoot/to_up_series.csv").open()))
ns = [r for r in rows if not r.get("settle")]
print("n", len(ns))
for r in ns[-30:]:
    print(
        f"t={float(r['t']):.2f} ang={float(r['ang_err_deg']):.2f} "
        f"alpha={float(r['alpha_up_deg']):.2f} rate={float(r['rate_cmd']):.3f} "
        f"wcmd={float(r['omega_cmd_norm']):.3f} wact={float(r['omega_act_norm']):.3f}"
    )
for thr in (40, 30, 20, 15, 12):
    hit = next((r for r in ns if float(r["alpha_up_deg"]) <= thr), None)
    if hit:
        print(f"first alpha<={thr}: t={hit['t']} ang={hit['ang_err_deg']} rate={hit['rate_cmd']}")
# time spent with alpha<25 while still commanding
tail = [r for r in ns if float(r["alpha_up_deg"]) < 25]
print("steps with alpha<25:", len(tail), "duration_s", len(tail) * 0.02)
print("mean rate in that band", sum(float(r["rate_cmd"]) for r in tail) / max(len(tail), 1))
