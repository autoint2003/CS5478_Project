"""Bounded HANDOFF variants after successful vertical 5 mm slip."""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.precision import run_finger_supported_vertical_axial as R  # noqa: E402
from data_generation.oracles.precision.finger_supported_vertical import (  # noqa: E402
    U_OPEN_HAND,
    opening_alignment,
)
from data_generation.oracles.precision.force_slip_control import ForcePController, make_action, sample_state  # noqa: E402
from data_generation.oracles.precision.orientation_planner import (  # noqa: E402
    _height_hold_v_hand,
    execute_restore_plan,
    plan_restore_orientation,
)
from envs.simulation.capture import restore_capture, take_capture  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402

OUT = R.OUT
blob_post = R.load_blob(R.B_POST)
vert = pickle.loads((OUT / "snapshots/best_vertical.pkl").read_bytes())
vert["ref"] = blob_post["ref"]
env = R.make_env()

d_in_h = -U_OPEN_HAND
R.restore_capture(env, vert["capture"])
gate = R.vertical_gate_report(env.sim)
z_ref = float(gate["obj_z"])
tilt = {"capture": vert["capture"], "ref": vert["ref"]}
slip2 = R.run_axial_slip(env, tilt, d_in_h, milestone_mm=2.0, z_ref=z_ref, e_x0=float(gate["e_signed_mm"][0]))
assert slip2["ok"], slip2
tilt5 = {"capture": slip2["capture"], "ref": vert["ref"]}
slip5 = R.run_axial_slip(
    env, tilt5, d_in_h, milestone_mm=5.0, z_ref=z_ref, e_x0=float(slip2["e_signed"][0])
)
assert slip5["ok"], slip5
post5 = {"capture": slip5["capture"], "ref": vert["ref"]}
print("post5 e", slip5["e_signed"], "θ", slip5["theta"], "α", opening_alignment(env.sim)["alpha_open_up_deg"])

variants = {}

# V1: firm hold then HANDOFF in-place (fingertips still up)
R.restore_capture(env, slip5["capture"])
ctl = ForcePController(kp=0.15, da_max=0.2)
ctl.reset(0.9, sample_state(env.sim)["fn_tot"])
z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
for _ in range(int(0.5 / R.DT)):
    s = sample_state(env.sim)
    a_g, _ = ctl.step(s["fn_tot"], 14.0)
    step_legal(env.sim, make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g))
cap_hold = take_capture(env)
h1 = R.handoff_rich(env, cap_hold, vert["ref"])
variants["handoff_inplace_firm"] = {
    k: h1.get(k)
    for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm", "pre_theta", "pre_pos_mm", "reason")
}
print("inplace", variants["handoff_inplace_firm"])

# V2: slow restore with firm grip
R.restore_capture(env, slip5["capture"])
post = {"capture": take_capture(env), "ref": vert["ref"]}
# bump grip before restore
ctl = ForcePController(kp=0.15, da_max=0.2)
ctl.reset(0.95, sample_state(env.sim)["fn_tot"])
z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
for _ in range(int(0.3 / R.DT)):
    s = sample_state(env.sim)
    a_g, _ = ctl.step(s["fn_tot"], 14.0)
    step_legal(env.sim, make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g))
post = {"capture": take_capture(env), "ref": vert["ref"]}
rplan = plan_restore_orientation(env, post, fn_target=14.0)
# execute with lower rate by hacking n_steps via re-plan angles — use execute as-is
rexe = execute_restore_plan(env, post, rplan)
s = sample_state(env.sim)
oa = opening_alignment(env.sim)
variants["restore_then_handoff"] = {
    "restore_ok": rexe.get("ok"),
    "plan": {k: rplan.get(k) for k in ("ok", "axis_name", "angle_deg", "already_upright")},
    "post_e": s["e_signed_mm"],
    "post_theta": s["e_theta_deg"],
    "post_alpha": oa["alpha_open_up_deg"],
}
h2 = R.handoff_rich(env, take_capture(env), vert["ref"])
variants["restore_then_handoff"]["handoff"] = {
    k: h2.get(k)
    for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm", "pre_theta", "pre_pos_mm", "reason")
}
print("restore", variants["restore_then_handoff"])

# V3: HANDOFF after 2mm only (less axial, still vertical)
R.restore_capture(env, slip2["capture"])
ctl = ForcePController(kp=0.15, da_max=0.2)
ctl.reset(0.9, sample_state(env.sim)["fn_tot"])
z_ref = float(env.sim.data.xpos[env.sim.ids.object_body][2])
for _ in range(int(0.5 / R.DT)):
    s = sample_state(env.sim)
    a_g, _ = ctl.step(s["fn_tot"], 14.0)
    step_legal(env.sim, make_action(_height_hold_v_hand(env.sim, z_ref), [0, 0, 0], a_g))
h3 = R.handoff_rich(env, take_capture(env), vert["ref"])
variants["handoff_after_2mm_inplace"] = {
    k: h3.get(k)
    for k in ("SUCCESS_12S", "SUCCESS_NEW", "SUCCESS_AXIS_V1", "pre_e_signed_mm", "pre_theta", "pre_pos_mm", "reason")
}
print("after2", variants["handoff_after_2mm_inplace"])

(OUT / "handoff_variants.json").write_text(json.dumps(R.jsafe(variants), indent=2), encoding="utf-8")
print("wrote handoff_variants.json")
