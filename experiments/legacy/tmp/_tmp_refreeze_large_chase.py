import csv
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.expand_k49_large_chase_dynamic import OUT, freeze_large_chase, is_large_chase

success = list(csv.DictReader((OUT / "EXISTING_SUCCESS_MOTION_AUDIT.csv").open(encoding="utf-8")))
thr = freeze_large_chase(success, [])
print("thr", thr["threshold_hand_max_disp_m"], thr["threshold_disp_to_recatch_m"])
for r in success:
    r["large_chase"] = int(is_large_chase(r, thr))
keys = list(success[0].keys())
with (OUT / "EXISTING_SUCCESS_MOTION_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(success)
seven = list(csv.DictReader((OUT / "SEVEN_DYNAMIC_SUCCESS_AUDIT.csv").open(encoding="utf-8")))
for r in seven:
    r["large_chase"] = int(is_large_chase(r, thr))
    r["pursuit_vs_local"] = (
        "ACTIVE_PURSUIT"
        if float(r["disp_to_recatch_m"]) >= 0.03 or float(r["hand_max_disp_m"]) >= 0.04
        else "LOCAL_CORRECTION"
    )
keys = list(seven[0].keys())
with (OUT / "SEVEN_DYNAMIC_SUCCESS_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(seven)
(OUT / "RAW" / "LARGE_CHASE_THRESHOLD_FREEZE.json").write_text(json.dumps(thr, indent=2), encoding="utf-8")
(OUT / "CHASE_DISPLACEMENT_THRESHOLDS.md").write_text(
    "# LARGE_CHASE displacement thresholds (frozen before Phase B search)\n\n"
    "**HIGH_MOTION (unchanged):** v>=0.12 m/s OR w>=2.5 rad/s OR v_lat>=0.1 m/s\n\n"
    f"**LARGE_CHASE thr_max:** {thr['threshold_hand_max_disp_m']:.4f} m\n"
    f"**LARGE_CHASE thr_recatch:** {thr['threshold_disp_to_recatch_m']:.4f} m\n"
    f"**max-disp also requires disp_to_recatch >= {thr['threshold_max_requires_recatch_m']:.3f} m**\n\n"
    f"**Rule:** {thr['rule']}\n\n"
    f"**Interpretation:** {thr['interpretation']}\n\n"
    "## Quantiles (existing SUCCESS_NEW hand motion)\n\n"
    + "\n".join(f"- `{k}`: {v:.4f}" for k, v in thr["quantiles"].items())
    + "\n\nNeither diagnostic changes SUCCESS_NEW.\n",
    encoding="utf-8",
)
fail = list(csv.DictReader((OUT / "HIGH_MOTION_FAILURE_AUDIT.csv").open(encoding="utf-8")))
(OUT / "RAW" / "PHASE_A_AUDIT.pkl").write_bytes(
    pickle.dumps(
        {
            "thr_pack": thr,
            "success_rows": success,
            "seven_rows": seven,
            "fail_rows": fail,
        }
    )
)
print("LC count", sum(int(r["large_chase"]) for r in success))
print("seven LC", sum(int(r["large_chase"]) for r in seven))
