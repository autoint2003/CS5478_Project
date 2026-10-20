from pathlib import Path

p = Path("scripts/run_freefall_recatch_pose_feedback.py")
text = p.read_text(encoding="utf-8")
start = text.index("    # ========== Stage B: distal correction from snapshots ==========")
end = text.index("    # ========== Stage D: 39D audit ==========")
new_bc = Path("scripts/_tmp_stage_bc_block.py").read_text(encoding="utf-8")
if not new_bc.endswith("\n"):
    new_bc += "\n"
text2 = text[:start] + new_bc + text[end:]

old = (
    '    best_P = {"gains": {"kp": 1.4, "kz": 0.5, "kw": 0.6, "lift": -0.5, "vz_bias": 0.25}, "score": -1e9}\n'
    '    best_O = {"gains": {"kp": 1.4, "kz": 0.5, "kw": 0.55, "lift": -0.45, "vz_bias": 0.25, "wy": 0.15}, "score": -1e9}'
)
new = (
    '    best_P = {"gains": {"kp": 1.4, "kz": 0.5, "kw": 0.6, "lift": -0.55, "vz_bias": 0.25, '
    '"wx_approach": 0.0, "wy_approach": 0.14, "wz_approach": 0.20, "kw_app": 0.45}, "score": -1e9}\n'
    '    best_O = {"gains": {"kp": 1.4, "kz": 0.5, "kw": 0.55, "lift": -0.5, "vz_bias": 0.25, "wy": 0.15, '
    '"wx_approach": 0.0, "wy_approach": 0.14, "wz_approach": 0.20}, "score": -1e9}'
)
if old not in text2:
    raise SystemExit("Stage E defaults not found")
text2 = text2.replace(old, new, 1)

old2 = """        gP = {
            \"kp\": float(rng.uniform(0.8, 1.8)),
            \"kz\": float(rng.uniform(0.2, 0.8)),
            \"kw\": float(rng.uniform(0.3, 0.9)),
            \"lift\": float(rng.uniform(-0.7, -0.2)),
            \"vz_bias\": float(rng.uniform(0.0, 0.4)),
        }
        gO = {
            \"kp\": float(rng.uniform(0.8, 1.8)),
            \"kz\": float(rng.uniform(0.2, 0.8)),
            \"kw\": float(rng.uniform(0.3, 0.9)),
            \"lift\": float(rng.uniform(-0.7, -0.2)),
            \"vz_bias\": float(rng.uniform(0.0, 0.4)),
            \"wy\": float(rng.uniform(0.0, 0.35)),
        }"""
new2 = """        gP = {
            \"kp\": float(rng.uniform(0.8, 1.8)),
            \"kz\": float(rng.uniform(0.2, 0.8)),
            \"kw\": float(rng.uniform(0.3, 0.9)),
            \"lift\": float(rng.uniform(-0.7, -0.2)),
            \"vz_bias\": float(rng.uniform(0.0, 0.4)),
            \"wx_approach\": float(rng.choice([0.0, -0.05, 0.05])),
            \"wy_approach\": float(rng.choice([0.10, 0.14, 0.18, 0.20])),
            \"wz_approach\": float(rng.choice([0.15, 0.18, 0.20, 0.22])),
            \"kw_app\": float(rng.uniform(0.3, 0.6)),
        }
        gO = {
            \"kp\": float(rng.uniform(0.8, 1.8)),
            \"kz\": float(rng.uniform(0.2, 0.8)),
            \"kw\": float(rng.uniform(0.3, 0.9)),
            \"lift\": float(rng.uniform(-0.7, -0.2)),
            \"vz_bias\": float(rng.uniform(0.0, 0.4)),
            \"wy\": float(rng.uniform(0.0, 0.35)),
            \"wx_approach\": float(rng.choice([0.0, -0.05, 0.05])),
            \"wy_approach\": float(rng.choice([0.10, 0.14, 0.18, 0.20])),
            \"wz_approach\": float(rng.choice([0.15, 0.18, 0.20, 0.22])),
        }"""
if old2 not in text2:
    raise SystemExit("calib sampling not found")
text2 = text2.replace(old2, new2, 1)

# Fix post-bilateral first_theta markers in Stage A block if present as naive scan
# Update conclusions
repls = [
    (
        "1. **Why ~20.5° on distal?** Error present by HANDOFF; onset tracked in DISTAL_POSE_ERROR_TIMELINE / REFERENCE_REPLAY (first >=15/20 steps). Typically established at/after capture geometry rather than only nominal continuation.",
        "1. **Why ~20.5 deg on distal?** Locked at first sustained bilateral capture (~21.7 deg at step 7); remains ~20.2-20.5 through lift/HANDOFF/nominal. Cause is capture approach geometry, not nominal continuation.",
    ),
    (
        "5. **Correction during capture?** Early-cut + seat/axis_align + privileged pivot was the primary Stage C strategy; post-only suffix often loses HANDOFF compatibility.",
        "5. **Correction during capture?** Required. Post-capture pivoting moves orientation by <1 deg; during-capture early wrist (wy~0.14, wz~0.20) sets bil theta near 0 deg and enables SUCCESS_NEW.",
    ),
]
for a, b in repls:
    if a in text2:
        text2 = text2.replace(a, b, 1)
    else:
        print("WARN missing:", a[:60])

# Also add calib distal IC for better shared gains
old3 = "    calib_ics = [roots[\"perp_off01_r15\"][0], roots[\"perp_off01_r100\"][0]]"
new3 = "    calib_ics = [roots[\"perp_off01_r15\"][0], roots[\"perp_off01_r100\"][0], roots[\"distal_off01_r8\"][0]]"
if old3 in text2:
    text2 = text2.replace(old3, new3, 1)

p.write_text(text2, encoding="utf-8")
print("patched ok, lines", text2.count("\n") + 1)
