import csv
import json
from collections import Counter
from pathlib import Path

OUT = Path("results/expert_generation/k49_horizontal_endcatch_recatch")
ans = json.load(open(OUT / "RAW" / "answers.json", encoding="utf-8"))
print("ANSWERS keys ok, n3", ans["Q5_n_type3_ics"], "primary", ans["Q6_primary_limit"])

c = list(csv.DictReader((OUT / "ENDCATCH_CANDIDATES.csv").open(encoding="utf-8")))
hits = [
    x for x in c
    if x["eccentric"] == "1" and x["downward"] == "1"
    and x["beneficial_rot"] == "1" and x["bilateral"] == "1"
]
print("ecc+down+ben+bil", len(hits))
for x in hits[:20]:
    print(
        x["ic_id"], x["tag"],
        "dth", x["delta_theta_deg"],
        "ax", x["axial_max_mm"],
        "sup", x["supported"],
        "th_bil", x["theta_bil"],
        "lim", x["limit_factor"],
        "ok", x["ok_prefix"],
    )
print("supported among those", sum(1 for x in hits if x["supported"] == "1"))

partials = []
if (OUT / "PARTIAL_MECHANISM.csv").exists():
    partials = list(csv.DictReader((OUT / "PARTIAL_MECHANISM.csv").open(encoding="utf-8")))
print("partials", len(partials), Counter(p.get("status") for p in partials))
ph = [p for p in partials if p.get("status") == "POSE_OR_HOLD_FAIL"]
print("POSE_OR_HOLD", len(ph))
for p in ph[:12]:
    print(p.get("ic_id"), p.get("tag"), "win_th", p.get("win_theta"), "win_pos", p.get("win_pos"),
          "s12", p.get("s12"), "reason", p.get("reason"), "dth", p.get("delta_theta_deg"))

# failures with eccentric evidence close to NEW
fails = list(csv.DictReader((OUT / "FAILURES.csv").open(encoding="utf-8")))
near = [
    f for f in fails
    if f.get("family") == "type3" and f.get("s12") == "1"
    and f.get("delta_theta") not in (None, "")
]
near.sort(key=lambda f: float(f.get("win_theta") or 99))
print("type3 s12 fails closest theta:")
for f in near[:15]:
    print(f.get("ic_id"), f.get("tag"), "th", f.get("win_theta"), "pos", f.get("win_pos"),
          "dth", f.get("delta_theta"), "m3", f.get("m3_reason"), "lim", f.get("limit"),
          "sup", f.get("supported"))

matched = list(csv.DictReader((OUT / "MATCHED_BASELINES.csv").open(encoding="utf-8")))
print("MATCHED")
for m in matched:
    print(
        m["ic_id"], "best", m["best"],
        "th0", round(float(m["e_theta0"]), 1),
        "A", m["A_NEW"], "B", m["B_NEW"], "C", m["C_TYPE2_NEW"], "D", m["D_TYPE3_NEW"],
    )

# max dth by IC among ecc+down+ben
from collections import defaultdict
best = defaultdict(lambda: -1e9)
for x in c:
    if x["eccentric"] == "1" and x["downward"] == "1" and x["beneficial_rot"] == "1":
        best[x["ic_id"]] = max(best[x["ic_id"]], float(x["delta_theta_deg"]))
print("best dth per IC", dict(best))
