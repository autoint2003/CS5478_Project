import json
from pathlib import Path

OUT = Path("results/experiments/p64_ppo_multiround")
print("BEST", (OUT / "BEST_CHECKPOINT.json").read_text(encoding="utf-8"))
b = json.loads((OUT / "BASELINES.json").read_text(encoding="utf-8"))
print("P62", b["P62_epoch3"]["MSP_RELAXED"], "P63A", b["P63_PPO_A"]["MSP_RELAXED"])
for st in range(1, 5):
    d = json.loads((OUT / f"P64_step{st}.json").read_text(encoding="utf-8"))
    e = d["eval"]
    bc = e.get("by_class") or {}
    print(
        f"step{st}",
        round(e["MSP_RELAXED"], 4),
        "A",
        round((bc.get("A") or {}).get("MSP_RELAXED", 0), 4),
        "B",
        round((bc.get("B") or {}).get("MSP_RELAXED", 0), 4),
        "C",
        round((bc.get("C") or {}).get("MSP_RELAXED", 0), 4),
        "D",
        round((bc.get("D") or {}).get("MSP_RELAXED", 0), 4),
        "z",
        e.get("zero_success_ics"),
        "rec",
        round(e.get("p_recatch") or 0, 4),
        "hand",
        round(e.get("p_handoff") or 0, 4),
    )
print((OUT / "SAMPLE_EFFICIENCY.json").read_text(encoding="utf-8"))
print((OUT / "EPOCH_COMPARISON.md").read_text(encoding="utf-8"))
