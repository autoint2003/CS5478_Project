"""Privileged grasp pack built from simulator state.

This is the dict behind the physical observation, not the 39-D policy vector.
"""

from __future__ import annotations

import numpy as np

from controllers.recovery.rule_based import g_hand
from envs.contacts.scene_contacts import cylinder_min_clearance, raw_object_contacts, scene_contacts
from envs.simulation.grasp_quality import body_twist


def obs_from_sim(sim) -> dict:
    Rh = np.array(sim.data.xmat[sim.ids.hand_body].reshape(3, 3), float)
    ph = np.array(sim.data.xpos[sim.ids.hand_body], float)
    po = np.array(sim.data.xpos[sim.ids.object_body], float)
    rh = Rh.T @ (po - ph)
    vo, wo = body_twist(sim.model, sim.data, sim.ids.object_body)
    vh, wh = body_twist(sim.model, sim.data, sim.ids.hand_body)
    cs = raw_object_contacts(sim)
    nL = sum(1 for c in cs if c["kind"] == "object-left_finger")
    nR = sum(1 for c in cs if c["kind"] == "object-right_finger")
    fnL = sum(abs(c["fn"]) for c in cs if c["kind"] == "object-left_finger")
    fnR = sum(abs(c["fn"]) for c in cs if c["kind"] == "object-right_finger")
    sc = scene_contacts(sim)
    return {
        "e_x": float(rh[0]), "rh": rh, "Rh": Rh, "ph": ph, "po": po,
        "nL": nL, "nR": nR, "Fn_L": fnL, "Fn_R": fnR,
        "v_rel": float(np.linalg.norm(vo - vh)),
        "w_rel": float(np.linalg.norm(wo - wh)),
        "obj_z": float(po[2]), "g_w": np.array(sim.model.opt.gravity, float),
        "scene": len(sc),
        "g_hx": float(g_hand(Rh, np.array(sim.model.opt.gravity, float))[0]),
        "clear": cylinder_min_clearance(sim),
    }

