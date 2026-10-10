"""Train only on first bilateral losses that are still on the distal tip.

d_center is the seated-center displacement. A sample enters the natural-loss
set only when the first persistent bilateral loss has d_clearance about 0
against the distal tip boxes, and the v3 intercept then regrasps for 1 s.
Already-separated states are stored as a diagnostic split and are not trained
or scored. Architecture, v3 limits, and the K=32 prototypes stay fixed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_transformer_v3 as v3
from controllers.gripper_controller import finger_opening
from envs.airborne_obs import observe_airborne, relative_state
from envs.observable_obs import ObservableObsState
from envs.physical_recovery import physical_pack
from training.demo_ballistic_impact import assert_noslip
from training.full_3d_airborne_recatch_dataset import (
    TRACK,
    begin,
    drop_supports,
    grasp_settled,
    hold_step,
    restore_to,
)
from training.horizontal_grasp_offset_freefall_recatch import apply_offset
from training.long_context_recovery_pilot import jsonable
from training.natural_loss_recatch_coverage_expansion import oracle_chase
from training.phase_decoupled_recovery_pilot import world
from training.recovery_policy_pretraining_preparation import schedules
from training.recovery_runtime import run_impact
from training.replay_core import geom_name, raw_object_contacts

RAW = ROOT / "results" / "diagnostics" / "raw" / "clean_natural_loss_recatch"
V3_CKPT = (
    ROOT / "results" / "diagnostics" / "raw"
    / "natural_loss_recatch_transformer_v3" / "ckpt_120.pt"
)
CLEAR_TOL = 0.001  # 1 mm. The excluded 5.5 mm case sits well above this.
PERSIST_STEPS = 2  # 40 ms. A one-step brush is not the loss event.

# Hold-out tags are fixed before any oracle result is known.
SPECS = [
    {"tag": "dz13.2", "dz": -13.2, "split": "train"},
    {"tag": "dz13.5", "dz": -13.5, "split": "train"},
    {"tag": "dz13.8", "dz": -13.8, "split": "train"},
    {"tag": "dz14.2", "dz": -14.2, "split": "train"},
    {"tag": "dz14.6", "dz": -14.6, "split": "train"},
    {"tag": "dvz_m", "dz": -13.5, "dvz": -0.01, "split": "train"},
    {"tag": "dvz_p", "dz": -13.5, "dvz": 0.005, "split": "train"},
    {"tag": "dq2", "dz": -13.5, "dq2": 0.001, "split": "train"},
    {"tag": "dx05", "dz": -13.5, "dx": 0.5, "split": "train"},
    {"tag": "dz13.4", "dz": -13.4, "split": "hold"},
    {"tag": "dz13.6", "dz": -13.6, "split": "hold"},
    {"tag": "dz14.0", "dz": -14.0, "split": "hold"},
]


def distal_geoms(sim):
    """Tip boxes at local z = 50 mm. These make the last contact on the way down."""
    out = []
    bodies = {int(sim.ids.left_body), int(sim.ids.right_body)}
    box = int(mujoco.mjtGeom.mjGEOM_BOX)
    for g in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[g]) not in bodies:
            continue
        if int(sim.model.geom_type[g]) != box:
            continue
        pos = np.array(sim.model.geom_pos[g], float)
        size = np.array(sim.model.geom_size[g], float)
        if abs(pos[2] - 0.05) < 0.002 and abs(size[2] - 0.003) < 0.0005:
            out.append(g)
    if len(out) != 4:
        raise RuntimeError(f"expected 4 distal tip boxes, found {len(out)}")
    return out


def clearance_m(sim, distal):
    obj = int(sim.ids.object_geom)
    best = None
    rows = []
    for g in distal:
        fromto = np.zeros(6)
        dist = float(mujoco.mj_geomDistance(sim.model, sim.data, obj, g, 0.2, fromto))
        name = geom_name(sim.model, g) or f"g{g}"
        rows.append({"geom": name, "dist_mm": round(dist * 1e3, 3)})
        if best is None or dist < best[0]:
            best = (dist, name)
    return best[0], best[1], rows


def measure(sim, pinch, distal):
    rel = relative_state(sim.model, sim.data, sim.ids)
    o = physical_pack(sim)
    dist, gname, rows = clearance_m(sim, distal)
    contacts = [
        {"kind": c["kind"], "geom": c["g2"], "fn": round(float(c["fn"]), 3)}
        for c in raw_object_contacts(sim)
        if "finger" in c["kind"]
    ]
    return {
        "d_center_mm": round(float(rel["p_rel_h"][2] - pinch[2]) * 1e3, 2),
        "d_clearance_mm": round(dist * 1e3, 3),
        "v_rel_h": [round(float(x), 4) for x in rel["v_rel_h"]],
        "finger_mm": round(float(finger_opening(sim.data, sim.ids)) * 1e3, 2),
        "n": [int(o["nL"]), int(o["nR"])],
        "nearest_geom": gname,
        "distal": rows,
        "contacts": contacts[:8],
        "p_rel_mm": [round(float(x) * 1e3, 2) for x in rel["p_rel_h"]],
    }


def slip_until_loss(sim, pinch, distal):
    """Closed hold until the first bilateral loss that stays lost for 40 ms."""
    hist = ObservableObsState(v3.DT_POLICY)
    frames = []
    t0 = float(sim.data.time)
    for _ in range(int(8.5 / v3.DT_POLICY)):
        o = physical_pack(sim)
        if int(o["nL"]) == 0 or int(o["nR"]) == 0:
            if not frames:
                return frames, None, "already_separated"
        obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
        frames.append({
            "obs": obs,
            "snap": v3.stamp(sim),
            "meas": measure(sim, pinch, distal),
            "t": float(sim.data.time) - t0,
        })
        v3.step_v3(sim, v3.HOLD, stop_on_loss=True)
        o2 = physical_pack(sim)
        if int(o2["nL"]) > 0 and int(o2["nR"]) > 0:
            continue
        loss_snap = v3.stamp(sim)
        persisted = True
        for _k in range(PERSIST_STEPS):
            v3.step_v3(sim, v3.HOLD)
            o3 = physical_pack(sim)
            if int(o3["nL"]) > 0 and int(o3["nR"]) > 0:
                persisted = False
                break
        if not persisted:
            continue
        return frames, loss_snap, "loss"
    return frames, None, "no_loss"


def restore_loss(sim, snap, nominal, pads, pad_mu):
    info = restore_to(sim, snap, nominal, pads, pad_mu)
    drop_supports(sim)
    return bool(info.get("pose_ok", False))


def windows_from(slip_frames, chase_frames, tag, split, loss_meas):
    """Pre-loss hold plus the oracle chase. The cut is still in bilateral contact."""
    out = []
    prefixes = [(15, 0.3), (50, 1.0), (100, 2.0)]
    for nback, _sec in prefixes:
        if len(slip_frames) < 8:
            start = 0
        else:
            start = max(0, len(slip_frames) - nback)
        pre = slip_frames[start:]
        if not pre:
            continue
        if pre[0]["meas"]["n"][0] == 0 or pre[0]["meas"]["n"][1] == 0:
            continue
        obs = [f["obs"] for f in pre] + [f["obs"] for f in chase_frames]
        act = [v3.HOLD.copy() for _ in pre] + [f["act"] for f in chase_frames]
        obs = np.stack(obs).astype(np.float32)
        act = np.stack(act).astype(np.float32)
        prev = np.zeros_like(act)
        prev[1:] = act[:-1]
        look = round(len(pre) * v3.DT_POLICY, 2)
        out.append({
            "family": f"clean_{tag}_p{look:.1f}",
            "split": split,
            "domain": "natural",
            "tag": tag,
            "obs": obs,
            "act": act,
            "prev": prev,
            "snap": pre[0]["snap"],
            "loss_meas": loss_meas,
        })
        if len(slip_frames) <= nback:
            break
    return out


def load_old(device):
    import torch

    ckpt = torch.load(V3_CKPT, map_location=device, weights_only=False)
    proto = np.asarray(ckpt["proto"], np.float32)
    model = v3.build_model(proto, device)
    return model, proto


def policy_eval(sim, model, snap, nominal, pads, pad_mu, pinch, distal, n_steps=160):
    import torch

    device = next(model.parameters()).device
    if not restore_loss(sim, snap, nominal, pads, pad_mu):
        return {"pose_ok": False}
    hist = ObservableObsState(v3.DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    rows = []
    loss_i = None
    lost_run = 0
    model.eval()
    with torch.no_grad():
        for k in range(n_steps):
            meas = measure(sim, pinch, distal)
            obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            obs_seq.append(obs)
            prev_seq.append(prev.copy())
            pred = model(
                torch.tensor(np.stack(obs_seq), device=device),
                torch.tensor(np.stack(prev_seq), device=device),
            )[2][-1].float().cpu().numpy()
            act = np.clip(pred, -1.0, 1.0).astype(np.float32)
            cmd = v3.step_v3(sim, act)
            o = physical_pack(sim)
            rows.append({
                "act": act,
                "cmd_vz": float(cmd["v_world"][2]),
                "meas": meas,
                "post_n": [int(o["nL"]), int(o["nR"])],
            })
            prev = act
            before_bilateral = meas["n"][0] > 0 and meas["n"][1] > 0
            if not before_bilateral:
                lost_run += 1
                if lost_run >= PERSIST_STEPS and loss_i is None:
                    loss_i = k - PERSIST_STEPS + 1
            else:
                lost_run = 0
            if k > 20 and meas["d_center_mm"] > 80 and int(o["nL"]) == 0 and int(o["nR"]) == 0:
                break
            if float(o["obj_z"]) < -0.05:
                break
    re = None
    re_meas = None
    if loss_i is not None:
        re = next((i for i in range(loss_i, len(rows)) if rows[i]["post_n"][0] > 0 and rows[i]["post_n"][1] > 0), None)
        if re is not None:
            re_meas = rows[re]["meas"]
    held = False
    if re is not None and re + 50 <= len(rows):
        tail = rows[re:re + 50]
        held = float(np.mean([r["post_n"][0] > 0 and r["post_n"][1] > 0 for r in tail])) >= 0.95
        held = held and tail[-1]["post_n"][0] > 0 and tail[-1]["post_n"][1] > 0
    loss = None if loss_i is None else rows[loss_i]["meas"]
    chase = [r for r in (rows[loss_i:] if loss_i is not None else []) if r["act"][2] > 0.4]
    strong = [r for r in rows if r["cmd_vz"] <= -3.0]
    return {
        "pose_ok": True,
        "held_1s": bool(held),
        "recontact": re is not None,
        "d_center_mm": None if loss is None else loss["d_center_mm"],
        "d_clearance_mm": None if loss is None else loss["d_clearance_mm"],
        "v_rel_z": None if loss is None else loss["v_rel_h"][2],
        "finger_mm": None if loss is None else loss["finger_mm"],
        "first_a2": None if not chase else round(float(chase[0]["act"][2]), 3),
        "first_cmd": None if not chase else round(float(chase[0]["cmd_vz"]), 3),
        "strong_s": round(len(strong) * v3.DT_POLICY, 3),
        "loss_contacts": None if loss is None else loss["contacts"],
        "end_n": rows[-1]["post_n"] if rows else None,
        "clean_realized": bool(loss is not None and loss["d_clearance_mm"] <= CLEAR_TOL * 1e3),
        "re_p_mm": None if re_meas is None else re_meas["p_rel_mm"],
        "re_v": None if re_meas is None else re_meas["v_rel_h"],
        "re_d_center_mm": None if re_meas is None else re_meas["d_center_mm"],
        "re_clear_mm": None if re_meas is None else re_meas["d_clearance_mm"],
    }


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    info = v3.cuda_info()
    v3.assert_map()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _old, proto = load_old(device)
    print("PROTO", len(proto), flush=True)

    sim, _g, nominal, pad_ids, pad_mu, _pl, _pr = world()
    assert_noslip(sim)
    seat = grasp_settled(sim, TRACK)
    if seat is None or not begin(sim, seat, nominal, pad_ids, pad_mu, {"dx": 0, "dy": 0, "dz": 0, "pitch": 0}):
        raise RuntimeError("seat failed")
    for _ in range(40):
        hold_step(sim, TRACK)
    pinch = np.array(relative_state(sim.model, sim.data, sim.ids)["p_rel_h"], float)
    distal = distal_geoms(sim)
    print("PINCH_MM", [round(float(x) * 1e3, 2) for x in pinch], "DISTAL", len(distal), flush=True)
    drop_supports(sim)
    base = v3.stamp(sim)

    audit = []
    train_trajs = []
    hold_trajs = []
    for spec in SPECS:
        info_r = restore_to(sim, base, nominal, pad_ids, pad_mu)
        drop_supports(sim)
        if not info_r.get("pose_ok", False):
            audit.append({"tag": spec["tag"], "split": spec["split"], "cls": "RESTORE_FAIL"})
            continue
        apply_offset(sim, {
            "dx": spec.get("dx", 0.0), "dy": 0.0, "dz": spec["dz"], "pitch": 0.0,
        })
        if spec.get("dq2"):
            sim.data.qpos[sim.ids.arm_jnt[1]] += float(spec["dq2"])
        if spec.get("dvz"):
            dadr = int(sim.model.jnt_dofadr[sim.ids.object_jnt])
            sim.data.qvel[dadr + 2] += float(spec["dvz"])
        mujoco.mj_forward(sim.model, sim.data)
        start = measure(sim, pinch, distal)
        if start["n"][0] == 0 or start["n"][1] == 0:
            row = {
                "tag": spec["tag"], "split": spec["split"], "dz": spec["dz"],
                "cls": "INITIAL_ALREADY_SEPARATED",
                "d_center_mm": start["d_center_mm"],
                "d_clearance_mm": start["d_clearance_mm"],
                "train": False, "primary_eval": False, "oracle": None,
            }
            audit.append(row)
            print("SKIP", spec["tag"], row["cls"], "clear", start["d_clearance_mm"], flush=True)
            continue
        slip, loss_snap, why = slip_until_loss(sim, pinch, distal)
        if loss_snap is None:
            audit.append({
                "tag": spec["tag"], "split": spec["split"], "dz": spec["dz"],
                "cls": why, "train": False, "primary_eval": False, "oracle": None,
            })
            print("SKIP", spec["tag"], why, flush=True)
            continue
        if not restore_loss(sim, loss_snap, nominal, pad_ids, pad_mu):
            audit.append({"tag": spec["tag"], "split": spec["split"], "cls": "LOSS_RESTORE_FAIL"})
            continue
        loss = measure(sim, pinch, distal)
        t_loss = round(slip[-1]["t"] + v3.DT_POLICY, 3) if slip else None
        separated = loss["d_clearance_mm"] > CLEAR_TOL * 1e3
        cls = "POST_LOSS_SEPARATED" if separated else "CLEAN_FIRST_LOSS"
        oframes, osum = oracle_chase(sim, loss_snap, nominal, pad_ids, pad_mu, pinch)
        oracle_ok = bool(osum.get("ok"))
        oracle_cls = "ORACLE_RECATChABLE" if oracle_ok else "ORACLE_UNRECATChABLE"
        use_train = (not separated) and oracle_ok and spec["split"] == "train"
        use_eval = (not separated) and oracle_ok and spec["split"] == "hold"
        made = []
        if use_train or use_eval:
            made = windows_from(slip, oframes, spec["tag"], spec["split"], loss)
            if spec["split"] == "train":
                train_trajs.extend(made)
            else:
                hold_trajs.extend(made)
        row = {
            "tag": spec["tag"],
            "split": spec["split"],
            "dz": spec["dz"],
            "dx": spec.get("dx", 0.0),
            "dvz": spec.get("dvz", 0.0),
            "dq2": spec.get("dq2", 0.0),
            "cls": cls,
            "oracle_cls": oracle_cls,
            "t_loss": t_loss,
            "d_center_mm": loss["d_center_mm"],
            "d_clearance_mm": loss["d_clearance_mm"],
            "v_rel_h": loss["v_rel_h"],
            "finger_mm": loss["finger_mm"],
            "nearest_geom": loss["nearest_geom"],
            "n_at_loss": loss["n"],
            "contacts": loss["contacts"],
            "oracle_end": osum.get("end_n"),
            "oracle_min_d_center_mm": osum.get("min_excess_mm"),
            "n_windows": len(made),
            "train": use_train,
            "primary_eval": use_eval,
        }
        audit.append(row)
        print(
            spec["tag"], cls, oracle_cls,
            "dc", loss["d_center_mm"], "clear", loss["d_clearance_mm"],
            "v", loss["v_rel_h"][2], "finger", loss["finger_mm"],
            "wins", len(made),
            flush=True,
        )

    (RAW / "audit.json").write_text(json.dumps(jsonable(audit), indent=2), encoding="utf-8")
    print("TRAIN_WINDOWS", len(train_trajs), "HOLD_WINDOWS", len(hold_trajs), flush=True)
    if not train_trajs:
        print("NO_CLEAN_TRAIN", flush=True)
        (RAW / "summary.json").write_text(json.dumps(jsonable({
            "cuda": info, "audit": audit, "trained": False,
        }), indent=2), encoding="utf-8")
        print("DONE", flush=True)
        return

    sim_l, gains_l, nominal_l, pads_l_ids, pad_mu_l, pads_left, pads_right = world()
    assert_noslip(sim_l)
    local = []
    sched = schedules()
    for name, z in (("zm6", -0.006),):
        impact = run_impact(sim_l, gains_l, 6.5, z, 0.0, pads_left, pads_right, nominal_l, None, None)
        for skill in ("wrist_reseat", "recatch_open20", "gravity_inward"):
            tr = v3.record_local(sim_l, skill, sched[skill], impact["snap"], nominal_l, pads_l_ids, pad_mu_l, None)
            if tr is None:
                print("LOCAL_FAIL", skill, flush=True)
                continue
            tr["split"] = "train"
            tr["ic"] = name
            tr["domain"] = "local"
            local.append(tr)
            print("LOCAL", skill, len(tr["act"]), flush=True)

    train_set = train_trajs + local
    v3.OUT = RAW
    model, loss, elapsed, amp_on = v3.train_model(train_set, proto, device, use_amp=True)
    tf_hold = v3.teacher_forced(model, hold_trajs, device) if hold_trajs else []
    tf_train = v3.teacher_forced(model, train_trajs, device)
    print("TF_TRAIN", [(r["family"], r["mae"], r.get("a2_hat_chase")) for r in tf_train], flush=True)
    print("TF_HOLD", [(r["family"], r["mae"], r.get("a2_hat_chase")) for r in tf_hold], flush=True)

    closed = []
    for t in hold_trajs:
        row = policy_eval(sim, model, t["snap"], nominal, pad_ids, pad_mu, pinch, distal)
        row["family"] = t["family"]
        row["tag"] = t["tag"]
        closed.append(row)
        print(
            "CL", t["family"], row.get("held_1s"),
            "dc", row.get("d_center_mm"), "clear", row.get("d_clearance_mm"),
            "v", row.get("v_rel_z"), "a2", row.get("first_a2"),
            "clean", row.get("clean_realized"),
            flush=True,
        )
    local_cl = []
    for t in local:
        row = v3.closed_loop(sim_l, model, t["snap"], nominal_l, pads_l_ids, pad_mu_l, None, 56, "local")
        row.update({"family": t["family"], "ic": t["ic"]})
        local_cl.append(row)
        print("LOCL", t["family"], row.get("t12"), flush=True)

    n_primary = len(closed)
    n_ok = sum(1 for r in closed if r.get("held_1s"))
    blob = {
        "cuda": info,
        "amp": amp_on,
        "train_s": round(elapsed, 1),
        "epochs": v3.EPOCHS,
        "loss": loss,
        "tokenizer": "frozen_k32_ckpt_120",
        "clear_tol_mm": CLEAR_TOL * 1e3,
        "audit": audit,
        "n_train_windows": len(train_trajs),
        "n_hold_windows": len(hold_trajs),
        "n_local": len(local),
        "teacher_train": tf_train,
        "teacher_hold": tf_hold,
        "closed_primary": closed,
        "primary_success": n_ok,
        "primary_n": n_primary,
        "local_closed": local_cl,
    }
    (RAW / "summary.json").write_text(json.dumps(jsonable(blob), indent=2), encoding="utf-8")
    print("PRIMARY", n_ok, "/", n_primary, flush=True)
    print("DONE", flush=True)


def eval_only():
    """Replay the held-out windows on the saved clean-set checkpoint."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(RAW / "ckpt_120.pt", map_location=device, weights_only=False)
    model = v3.build_model(np.asarray(ckpt["proto"], np.float32), device)
    model.load_state_dict(ckpt["model"])
    sim, _g, nominal, pad_ids, pad_mu, _pl, _pr = world()
    seat = grasp_settled(sim, TRACK)
    begin(sim, seat, nominal, pad_ids, pad_mu, {"dx": 0, "dy": 0, "dz": 0, "pitch": 0})
    for _ in range(40):
        hold_step(sim, TRACK)
    pinch = np.array(relative_state(sim.model, sim.data, sim.ids)["p_rel_h"], float)
    distal = distal_geoms(sim)
    drop_supports(sim)
    base = v3.stamp(sim)
    for spec in [s for s in SPECS if s["tag"] in ("dz13.4", "dz14.0")]:
        restore_to(sim, base, nominal, pad_ids, pad_mu)
        drop_supports(sim)
        apply_offset(sim, {"dx": 0.0, "dy": 0.0, "dz": spec["dz"], "pitch": 0.0})
        mujoco.mj_forward(sim.model, sim.data)
        slip, loss_snap, why = slip_until_loss(sim, pinch, distal)
        if loss_snap is None:
            print("NOLOSS", spec["tag"], why, flush=True)
            continue
        for nback, sec in ((15, 0.3), (50, 1.0), (100, 2.0)):
            start = 0 if len(slip) < 8 else max(0, len(slip) - nback)
            row = policy_eval(sim, model, slip[start]["snap"], nominal, pad_ids, pad_mu, pinch, distal)
            print(
                spec["tag"], sec,
                "held", row.get("held_1s"),
                "dc", row.get("d_center_mm"),
                "clear", row.get("d_clearance_mm"),
                "v", row.get("v_rel_z"),
                "a2", row.get("first_a2"),
                "cmd", row.get("first_cmd"),
                "strong", row.get("strong_s"),
                "re_dc", row.get("re_d_center_mm"),
                "re_clear", row.get("re_clear_mm"),
                "re_p", row.get("re_p_mm"),
                "re_v", row.get("re_v"),
                flush=True,
            )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        eval_only()
    else:
        main()
