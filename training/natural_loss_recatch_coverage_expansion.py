"""Expand natural-loss coverage only where the v3 intercept can still catch.

Failure states are replayed from the saved v3 policy. The privileged intercept
runs under recovery7d_airborne_v3. Unreachable states are not trained on.
Architecture, the K=32 prototypes, and the v3 limits stay as they are.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import training.natural_loss_recatch_transformer_v3 as v3
from controllers.recovery_actions import AIRBORNE_V3, map_action
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
from training.long_context_recovery_pilot import jsonable
from training.phase_decoupled_recovery_pilot import world
from training.recovery_policy_pretraining_preparation import schedules
from training.recovery_runtime import continue_long, physical_of, run_impact
from training.replay_core import park_impact_ball

RAW = ROOT / "results" / "diagnostics" / "raw" / "natural_loss_recatch_coverage_expansion"
V3_CKPT = (
    ROOT / "results" / "diagnostics" / "raw"
    / "natural_loss_recatch_transformer_v3" / "ckpt_120.pt"
)
GATES_MM = (25.0, 28.0, 30.0, 32.0, 34.0, 36.0, 37.0)
PREFIX_S = (0.2, 0.6, 1.2, 2.0)


def r3(x, n=3):
    return [round(float(v), n) for v in np.asarray(x, float).reshape(-1)]


def describe(sim, pinch):
    rel = relative_state(sim.model, sim.data, sim.ids)
    o = physical_pack(sim)
    excess = float(rel["p_rel_h"][2] - pinch[2])
    q = np.array(sim.data.qpos[sim.ids.arm_jnt], float)
    return {
        "p_rel_mm": r3(rel["p_rel_h"] * 1e3, 2),
        "v_rel_mps": r3(rel["v_rel_h"], 4),
        "rot_rel": r3(rel["rot_rel"], 4),
        "w_rel": r3(rel["w_rel_h"], 3),
        "v_hand_mps": r3(rel["v_hand_w"], 4),
        "w_hand": r3(rel["w_hand_h"], 3),
        "p_hand_mm": r3(rel["p_hand_w"] * 1e3, 2),
        "aperture_mm": round(float(o["aperture"]) * 1e3, 2),
        "q_arm": r3(q, 4),
        "excess_mm": round(excess * 1e3, 2),
        "n": [int(o["nL"]), int(o["nR"])],
        "obj_z": round(float(o["obj_z"]), 4),
    }


def load_old(device):
    import torch

    ckpt = torch.load(V3_CKPT, map_location=device, weights_only=False)
    proto = np.asarray(ckpt["proto"], np.float32)
    model = v3.build_model(proto, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, proto


def policy_rollout(sim, model, snap, nominal, pads, pad_mu, pinch, n_steps=160):
    """Replay the saved policy. Stamp the state before each action."""
    import torch

    device = next(model.parameters()).device
    info = restore_to(sim, snap, nominal, pads, pad_mu)
    drop_supports(sim)
    if not info.get("pose_ok", False):
        return []
    hist = ObservableObsState(v3.DT_POLICY)
    prev = np.zeros(7, np.float32)
    obs_seq, prev_seq = [], []
    rows = []
    prev_vz = None
    model.eval()
    with torch.no_grad():
        for k in range(n_steps):
            snapped = v3.stamp(sim)
            desc = describe(sim, pinch)
            obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
            obs_seq.append(obs)
            prev_seq.append(prev.copy())
            pred = model(
                torch.tensor(np.stack(obs_seq), device=device),
                torch.tensor(np.stack(prev_seq), device=device),
            )[2][-1].float().cpu().numpy()
            act = np.clip(pred, -1.0, 1.0).astype(np.float32)
            cmd = v3.step_v3(sim, act)
            rel = relative_state(sim.model, sim.data, sim.ids)
            vh = float(rel["v_hand_w"][2])
            az = 0.0 if prev_vz is None else (vh - prev_vz) / v3.DT_POLICY
            prev_vz = vh
            o = physical_pack(sim)
            rows.append({
                "snap": snapped,
                "obs": obs,
                "act": act,
                "desc": desc,
                "cmd_vz": float(cmd["v_world"][2]),
                "vh_z": vh,
                "az": float(az),
                "vrel_z": float(desc["v_rel_mps"][2]),
                "excess": float(desc["excess_mm"]) / 1e3,
                "nL": int(desc["n"][0]),
                "nR": int(desc["n"][1]),
                "post_n": [int(o["nL"]), int(o["nR"])],
                "k": k,
            })
            prev = act
            post_ex = float(rel["p_rel_h"][2] - pinch[2])
            if k > 20 and post_ex > 0.15 and int(o["nL"]) == 0 and int(o["nR"]) == 0:
                break
            if float(o["obj_z"]) < -0.05:
                break
    return rows


def pick_states(family, rows):
    """Exact decision states. A gate is the both-off frame nearest that gap, not the first frame past it."""
    picked = []
    if not rows:
        return picked
    bilat = next((i for i, r in enumerate(rows) if r["nL"] == 0 or r["nR"] == 0), None)
    both = next((i for i, r in enumerate(rows) if r["nL"] == 0 and r["nR"] == 0), None)
    if bilat is not None:
        picked.append(("bilateral_loss", bilat))
    if both is not None and both != bilat:
        picked.append(("both_off", both))
    separated = [(i, r) for i, r in enumerate(rows) if r["nL"] == 0 and r["nR"] == 0]
    used = {i for _k, i in picked}
    for gate in GATES_MM:
        if not separated:
            break
        i, r = min(separated, key=lambda z: abs(z[1]["desc"]["excess_mm"] - gate))
        if abs(r["desc"]["excess_mm"] - gate) > 2.0 or i in used:
            continue
        picked.append((f"gate_{gate:.0f}", i))
        used.add(i)
    out = []
    for kind, i in picked:
        r = rows[i]
        out.append({
            "family": family,
            "kind": kind,
            "step": i,
            "snap": r["snap"],
            "desc": r["desc"],
            "prefix": rows[:i],
        })
    return out


def dedup(states):
    """Keep every bilateral-loss state. Drop a later gate only when it repeats one."""
    kept = []
    for s in states:
        same = False
        for k in kept:
            if (
                k["family"] == s["family"]
                and abs(k["desc"]["excess_mm"] - s["desc"]["excess_mm"]) < 0.4
                and abs(k["desc"]["v_rel_mps"][2] - s["desc"]["v_rel_mps"][2]) < 0.02
            ):
                same = True
                break
        if not same:
            kept.append(s)
    return kept


def oracle_chase(sim, snap, nominal, pads, pad_mu, pinch):
    """Privileged intercept from an exact snapshot, under the v3 clip only."""
    info = restore_to(sim, snap, nominal, pads, pad_mu)
    drop_supports(sim)
    if not info.get("pose_ok", False):
        return [], {"ok": False, "why": "restore"}
    hist = ObservableObsState(v3.DT_POLICY)
    phase = "chase"
    both_hold = 0.0
    re_t = None
    hold_t = None
    free = True
    prev_vz = None
    frames = []
    t0 = float(sim.data.time)
    for _ in range(int(2.4 / v3.DT_POLICY)):
        o = physical_pack(sim)
        rel = relative_state(sim.model, sim.data, sim.ids)
        excess = float(rel["p_rel_h"][2] - pinch[2])
        bilateral = int(o["nL"]) > 0 and int(o["nR"]) > 0
        aper = float(o["aperture"])
        vrel = rel["v_obj_w"] - rel["v_hand_w"]
        slow = abs(float(vrel[2])) < 0.15 and float(np.linalg.norm(vrel[:2])) < 0.20
        near = (
            abs(excess) < 0.012
            and abs(float(rel["p_rel_h"][0] - pinch[0])) < 0.012
            and abs(float(rel["p_rel_h"][1] - pinch[1])) < 0.012
        )
        if phase == "chase" and aper > 0.036 and near and slow:
            phase = "close"
        if phase == "close":
            if bilateral and free:
                if re_t is None:
                    re_t = float(sim.data.time) - t0
                both_hold += v3.DT_POLICY
                if both_hold >= 0.04 and hold_t is None:
                    hold_t = float(sim.data.time) - t0
                    phase = "hold"
            else:
                both_hold = 0.0
        action, v_world = v3.intercept_action(sim, pinch, phase)
        obs = observe_airborne(sim.model, sim.data, sim.ids, sim.fsm, hist)
        vh = float(rel["v_hand_w"][2])
        az = 0.0 if prev_vz is None else (vh - prev_vz) / v3.DT_POLICY
        prev_vz = vh
        frames.append({
            "obs": obs,
            "act": np.asarray(action, np.float32).copy(),
            "phase": phase,
            "excess": excess,
            "nL": int(o["nL"]),
            "nR": int(o["nR"]),
            "cmd_vz": float(v_world[2]),
            "vh_z": vh,
            "vrel_z": float(vrel[2]),
            "az": float(az),
            "a2": float(action[2]),
            "t": float(sim.data.time) - t0,
        })
        v3.step_v3(sim, action)
        if hold_t is not None and frames[-1]["t"] >= hold_t + 1.02:
            break
        if re_t is None and float(sim.data.xpos[sim.ids.object_body][2]) < -0.15:
            break
        if re_t is None and frames[-1]["t"] > 0.70 and excess > 0.05:
            if len(frames) > 8 and frames[-1]["excess"] > frames[-8]["excess"]:
                break
    end = frames[-1] if frames else None
    window = []
    if re_t is not None:
        window = [f for f in frames if re_t - 1e-9 <= f["t"] <= re_t + 1.0]
    both_frac = float(np.mean([(f["nL"] > 0 and f["nR"] > 0) for f in window])) if window else None
    covered = bool(window) and window[-1]["t"] >= re_t + 0.98
    ok = bool(
        re_t is not None and covered and both_frac is not None and both_frac >= 0.95
        and end is not None and end["nL"] > 0 and end["nR"] > 0 and abs(end["excess"]) < 0.02
    )
    return frames, {
        "ok": ok,
        "why": "ok" if ok else "no 1s hold",
        "re_t": None if re_t is None else round(float(re_t), 3),
        "both_frac": None if both_frac is None else round(both_frac, 3),
        "end_n": None if end is None else [end["nL"], end["nR"]],
        "min_excess_mm": None if not frames else round(float(min(f["excess"] for f in frames)) * 1e3, 2),
        "n_frames": len(frames),
    }


def chase_profile(frames):
    """Backoff is the first command that leaves the near-cap region."""
    if not frames:
        return {}
    cmds = [float(f["cmd_vz"]) for f in frames]
    strong = [c <= -3.0 for c in cmds]
    first = frames[0]
    dur = float(sum(strong) * v3.DT_POLICY)
    back = None
    seen = False
    for f, s in zip(frames, strong):
        if s:
            seen = True
        elif seen:
            back = f
            break
    re = next((f for f in frames if f["nL"] > 0 and f["nR"] > 0 and f["phase"] in ("close", "hold", "chase")), None)
    # Closest approach after the dive has started.
    i_min = int(np.argmin([f["excess"] for f in frames]))
    sl = frames[:i_min + 1]
    return {
        "first_cmd": round(float(first["cmd_vz"]), 3),
        "first_a2": round(float(first["a2"]), 3),
        "strong_s": round(dur, 3),
        "peak_hand_vz": round(float(min(f["vh_z"] for f in sl)), 3),
        "peak_az": round(float(min(f["az"] for f in sl)), 2),
        "backoff_t": None if back is None else round(float(back["t"]), 3),
        "gap_at_backoff_mm": None if back is None else round(float(back["excess"]) * 1e3, 2),
        "vrel_at_backoff": None if back is None else round(float(back["vrel_z"]), 3),
        "cmd_at_backoff": None if back is None else round(float(back["cmd_vz"]), 3),
        "vrel_at_recontact": None if re is None else round(float(re["vrel_z"]), 3),
        "gap_at_recontact_mm": None if re is None else round(float(re["excess"]) * 1e3, 2),
        "min_excess_mm": round(float(frames[i_min]["excess"]) * 1e3, 2),
    }


def policy_backoff(rows):
    """Same backoff definition on the saved policy's replay."""
    if not rows:
        return {}
    loss = next((i for i, r in enumerate(rows) if r["nL"] == 0 or r["nR"] == 0), None)
    strong_i = [i for i, r in enumerate(rows) if r["cmd_vz"] <= -3.0]
    back = None
    if strong_i:
        for r in rows[strong_i[0]:]:
            if r["cmd_vz"] > -3.0:
                back = r
                break
    return {
        "gap_at_loss_mm": None if loss is None else rows[loss]["desc"]["excess_mm"],
        "vrel_at_loss": None if loss is None else rows[loss]["desc"]["v_rel_mps"][2],
        "first_strong_cmd": None if not strong_i else round(float(rows[strong_i[0]]["cmd_vz"]), 3),
        "strong_s": round(float(len(strong_i) * v3.DT_POLICY), 3),
        "gap_at_backoff_mm": None if back is None else back["desc"]["excess_mm"],
        "vrel_at_backoff": None if back is None else round(float(back["vrel_z"]), 3),
        "cmd_at_backoff": None if back is None else round(float(back["cmd_vz"]), 3),
        "opened_before_loss": bool(loss is not None and any(rows[i]["act"][6] < -0.3 for i in range(loss))),
    }


def make_gap_trajs(state, oframes):
    """Contact prefix labeled as a closed hold, then the oracle chase from this state."""
    # Only the bilateral-loss instant is the frame after the contact prefix.
    # A later gap crossing is already airborne; prepending the old contact
    # would splice two states that are not one step apart.
    if state["kind"] == "bilateral_loss":
        prefix = [r for r in state["prefix"] if r["nL"] > 0 and r["nR"] > 0]
    else:
        prefix = []
    chase_obs = [f["obs"] for f in oframes]
    chase_act = [f["act"] for f in oframes]
    if not chase_act:
        return []
    out = []
    cuts = [0]
    for sec in PREFIX_S:
        n = int(round(sec / v3.DT_POLICY))
        if n < len(prefix):
            cuts.append(len(prefix) - n)
    cuts = sorted(set(cuts))
    # A later airborne gate has no contact prefix. Keep the chase itself once.
    if not prefix:
        cuts = [0]
    for c in cuts:
        pre = prefix[c:]
        obs = [r["obs"] for r in pre] + chase_obs
        act = [v3.HOLD.copy() for _ in pre] + chase_act
        obs = np.stack(obs).astype(np.float32)
        act = np.stack(act).astype(np.float32)
        prev = np.zeros_like(act)
        prev[1:] = act[:-1]
        look = round(len(pre) * v3.DT_POLICY, 2)
        gap = state["desc"]["excess_mm"]
        out.append({
            "family": f"gap_{gap:.1f}_{state['family']}_p{look:.1f}",
            "split": "train",
            "domain": "gap",
            "gap_mm": gap,
            "kind": state["kind"],
            "obs": obs,
            "act": act,
            "prev": prev,
            "snap": pre[0]["snap"] if pre else state["snap"],
            "loss_snap": state["snap"],
            "vrel_z": state["desc"]["v_rel_mps"][2],
        })
    return out


def frozen_token_report(proto, actions):
    if len(actions) == 0:
        return {}
    dist = ((actions[:, None, :] - proto[None, :, :]) ** 2).sum(-1)
    lab = dist.argmin(1)
    resid = actions - proto[lab]
    return {
        "n": int(len(actions)),
        "mae": round(float(np.abs(resid).mean()), 4),
        "max_abs_resid": round(float(np.max(np.abs(resid))), 3),
        "frac_beyond_0.25": round(float(np.mean(np.any(np.abs(resid) > 0.25 + 1e-6, axis=1))), 3),
        "high_vz_mae": round(float(np.mean(np.abs(resid[actions[:, 2] > 0.5][:, 2]))), 4) if np.any(actions[:, 2] > 0.5) else None,
    }


def eval_natural(sim, model, snap, nominal, pads, pad_mu, pinch, n_steps=160):
    rows = policy_rollout(sim, model, snap, nominal, pads, pad_mu, pinch, n_steps=n_steps)
    if not rows:
        return {"pose_ok": False}
    loss = next((i for i, r in enumerate(rows) if r["nL"] == 0 or r["nR"] == 0), None)
    re = None
    if loss is not None:
        re = next((i for i in range(loss, len(rows)) if rows[i]["post_n"][0] > 0 and rows[i]["post_n"][1] > 0), None)
    held = False
    if re is not None and re + 50 <= len(rows):
        tail = rows[re:re + 50]
        held = float(np.mean([r["post_n"][0] > 0 and r["post_n"][1] > 0 for r in tail])) >= 0.95
        held = held and tail[-1]["post_n"][0] > 0 and tail[-1]["post_n"][1] > 0
    back = policy_backoff(rows)
    excesses = [r["excess"] for r in rows]
    i_min = int(np.argmin(excesses))
    return {
        "pose_ok": True,
        "steps": len(rows),
        "gap_at_loss_mm": None if loss is None else rows[loss]["desc"]["excess_mm"],
        "vrel_at_loss": None if loss is None else rows[loss]["desc"]["v_rel_mps"][2],
        "first_cmd": back.get("first_strong_cmd"),
        "strong_s": back.get("strong_s"),
        "gap_at_backoff_mm": back.get("gap_at_backoff_mm"),
        "vrel_at_backoff": back.get("vrel_at_backoff"),
        "cmd_at_backoff": back.get("cmd_at_backoff"),
        "min_gap_mm": round(float(rows[i_min]["excess"]) * 1e3, 2),
        "recontact": re is not None,
        "held_1s": bool(held),
        "end_n": rows[-1]["post_n"],
        "peak_cmd": round(float(min(r["cmd_vz"] for r in rows)), 3),
        "max_a2": round(float(max(r["act"][2] for r in rows)), 3),
    }


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    info = v3.cuda_info()
    v3.assert_map()
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old, proto = load_old(device)
    print("LOADED", V3_CKPT.name, "K", len(proto), "device", device, flush=True)

    sim, _gains, nominal, pad_ids, pad_mu, _pl, _pr = world()
    assert_noslip(sim)
    base_seat = grasp_settled(sim, TRACK)
    if base_seat is None or not begin(sim, base_seat, nominal, pad_ids, pad_mu, {"dx": 0, "dy": 0, "dz": 0, "pitch": 0}):
        raise RuntimeError("seat failed")
    for _ in range(40):
        hold_step(sim, TRACK)
    pinch = np.array(relative_state(sim.model, sim.data, sim.ids)["p_rel_h"], float)
    drop_supports(sim)
    base = v3.stamp(sim)

    train_specs = [
        {"tag": "dz13.5", "dz": -13.5},
        {"tag": "dz13.3", "dz": -13.3},
        {"tag": "dz13.8", "dz": -13.8},
        {"tag": "dz14.2", "dz": -14.2},
        {"tag": "dvz", "dz": -13.5, "dvz": -0.01},
    ]
    hold_specs = [
        {"tag": "dz13.6", "dz": -13.6},
        {"tag": "dz14.0", "dz": -14.0},
    ]
    natural = []
    for spec, split in [(s, "train") for s in train_specs] + [(s, "hold") for s in hold_specs]:
        wins, summary, _frames = v3.run_natural(sim, base, nominal, pad_ids, pad_mu, pinch, spec, split)
        if summary["ok"]:
            natural.extend(wins)
    eval_cases = []
    for t in natural:
        if t["split"] == "train" and not t["family"].endswith("w1.0") and "dz13.5" not in t["family"]:
            continue
        eval_cases.append(t)
    print("EVAL_CASES", [t["family"] for t in eval_cases], flush=True)

    raw_states = []
    policy_cmp = []
    for t in eval_cases:
        rows = policy_rollout(sim, old, t["snap"], nominal, pad_ids, pad_mu, pinch)
        states = pick_states(t["family"], rows)
        raw_states.extend(states)
        cmp = policy_backoff(rows)
        cmp["family"] = t["family"]
        policy_cmp.append(cmp)
        gaps = [round(s["desc"]["excess_mm"], 1) for s in states]
        trail = []
        for r in rows:
            if r["nL"] == 0 and r["nR"] == 0:
                g = round(r["desc"]["excess_mm"], 1)
                if not trail or abs(trail[-1] - g) >= 1.5:
                    trail.append(g)
        print("REPLAY", t["family"], "picked", gaps, "both_off", trail, "policy_back", cmp.get("gap_at_backoff_mm"), flush=True)

    chosen = dedup(raw_states)
    print("ORACLE_N", len(chosen), "of", len(raw_states), flush=True)
    audits = []
    reachable = []
    for s in chosen:
        oframes, osum = oracle_chase(sim, s["snap"], nominal, pad_ids, pad_mu, pinch)
        prof = chase_profile(oframes)
        rec = {
            "family": s["family"],
            "kind": s["kind"],
            "state": s["desc"],
            "oracle_ok": bool(osum["ok"]),
            "oracle": osum,
            "profile": prof,
            "cls": "V3_RECATChABLE" if osum["ok"] else "V3_UNRECATChABLE",
        }
        audits.append(rec)
        print(
            rec["cls"], s["family"], s["kind"],
            "gap", s["desc"]["excess_mm"], "vrel", s["desc"]["v_rel_mps"][2],
            "first", prof.get("first_cmd"), "back_gap", prof.get("gap_at_backoff_mm"),
            "min", osum.get("min_excess_mm"), "end", osum.get("end_n"),
            flush=True,
        )
        if osum["ok"]:
            reachable.append((s, oframes, rec))

    (RAW / "basin.json").write_text(json.dumps(jsonable({
        "policy_backoff": policy_cmp,
        "audits": audits,
    }), indent=2), encoding="utf-8")

    gaps_ok = sorted(r["state"]["excess_mm"] for _s, _f, r in reachable)
    print("REACHABLE_GAPS", gaps_ok, flush=True)
    larger = [g for g in gaps_ok if g >= 27.0]
    if not larger:
        print("NO_LARGER_BASIN", flush=True)
        (RAW / "summary.json").write_text(json.dumps(jsonable({
            "cuda": info,
            "reachable_gaps_mm": gaps_ok,
            "n_tested": len(audits),
            "n_recatchable": len(reachable),
            "trained": False,
            "policy_backoff": policy_cmp,
            "audits": [{k: v for k, v in a.items()} for a in audits],
        }), indent=2), encoding="utf-8")
        print("DONE_BASIN_ONLY", flush=True)
        return

    gap_trajs = []
    used_bins = {}
    for s, oframes, rec in sorted(reachable, key=lambda z: z[2]["state"]["excess_mm"]):
        g = rec["state"]["excess_mm"]
        if g < 27.0:
            continue
        b = int(round(g))
        used_bins.setdefault(b, 0)
        if used_bins[b] >= 2:
            continue
        made = make_gap_trajs(s, oframes)
        if not made:
            continue
        used_bins[b] += 1
        gap_trajs.extend(made)
        print("DEMO", made[0]["family"], "nwin", len(made), "steps0", len(made[0]["act"]), flush=True)

    new_acts = np.concatenate([t["act"] for t in gap_trajs]) if gap_trajs else np.zeros((0, 7), np.float32)
    tok = frozen_token_report(proto, new_acts)
    print("FROZEN_TOK", tok, flush=True)

    # Opposite-sign impacts whose existing recovery already retains the object.
    sim_l, gains_l, nominal_l, pads_l_ids, pad_mu_l, pads_left, pads_right = world()
    assert_noslip(sim_l)
    snaps = {}
    for name, z in (("zm6", -0.006), ("zp6", 0.006)):
        impact = run_impact(sim_l, gains_l, 6.5, z, 0.0, pads_left, pads_right, nominal_l, None, None)
        snaps[name] = impact["snap"]
        print("impact", name, flush=True)
    sched = schedules()
    local_jobs = [
        ("wrist_reseat", "zm6", "train"),
        ("recatch_open20", "zm6", "train"),
        ("gravity_inward", "zm6", "train"),
        ("recatch_open20", "zp6", "train"),
        ("gravity_inward", "zp6", "train"),
        ("wrist_reseat", "zp6", "hold"),
    ]
    local = []
    for name, ic, split in local_jobs:
        tr = v3.record_local(sim_l, name, sched[name], snaps[ic], nominal_l, pads_l_ids, pad_mu_l, None)
        if tr is None:
            print("SKIP", name, ic, flush=True)
            continue
        tr["split"] = split
        tr["ic"] = ic
        tr["domain"] = "local"
        print("LOCAL", name, ic, split, "steps", len(tr["act"]), flush=True)
        local.append(tr)

    original = [t for t in natural if t["split"] == "train"]
    for t in original:
        t["domain"] = "natural"
    train_set = original + gap_trajs + [t for t in local if t["split"] == "train"]
    print(
        "TRAIN", len(train_set),
        "orig", len(original), "gap", len(gap_trajs),
        "local", sum(t["domain"] == "local" for t in train_set),
        flush=True,
    )

    v3.OUT = RAW
    model, loss, elapsed, amp_on = v3.train_model(train_set, proto, device, use_amp=True)
    groups = {
        "original_25": [t for t in original],
        "larger_gap": gap_trajs,
        "local_zm6": [t for t in local if t["ic"] == "zm6"],
        "opposite_zp6": [t for t in local if t["ic"] == "zp6"],
    }
    teacher = {}
    for name, rows in groups.items():
        teacher[name] = v3.teacher_forced(model, rows, device) if rows else []
        maes = [r["mae"] for r in teacher[name]]
        print("TF", name, "n", len(maes), "mae", None if not maes else round(float(np.mean(maes)), 4), flush=True)

    closed_old = []
    for t in eval_cases:
        row = eval_natural(sim, model, t["snap"], nominal, pad_ids, pad_mu, pinch)
        row["family"] = t["family"]
        row["split"] = t["split"]
        closed_old.append(row)
        print(
            "CL", t["family"], row.get("held_1s"),
            "loss", row.get("gap_at_loss_mm"), "back", row.get("gap_at_backoff_mm"),
            "min", row.get("min_gap_mm"), "end", row.get("end_n"),
            flush=True,
        )

    closed_gap = []
    seen_gap = set()
    for t in gap_trajs:
        key = (round(t["gap_mm"], 1), t["family"].split("_p")[0])
        if key in seen_gap:
            continue
        if not str(t["family"]).endswith("_p0.2") and "_p0.0" not in t["family"] and abs(t["family"].count("p") ):
            # Prefer the shortest stored prefix, which is listed last or the p0.0 chase-only cut.
            pass
        seen_gap.add(key)
        row = eval_natural(sim, model, t["loss_snap"], nominal, pad_ids, pad_mu, pinch, n_steps=120)
        row["family"] = t["family"]
        row["gap_mm"] = t["gap_mm"]
        closed_gap.append(row)
        print(
            "GAPCL", t["family"], row.get("held_1s"),
            "loss", row.get("gap_at_loss_mm"), "min", row.get("min_gap_mm"),
            "end", row.get("end_n"),
            flush=True,
        )

    closed_local = []
    for t in local:
        row = v3.closed_loop(sim_l, model, t["snap"], nominal_l, pads_l_ids, pad_mu_l, None, 56, "local")
        row.update({"family": t["family"], "split": t["split"], "ic": t["ic"]})
        closed_local.append(row)
        print("LOCL", t["family"], t["ic"], t["split"], row.get("t12"), row.get("max_abs_v"), flush=True)

    blob = {
        "cuda": info,
        "amp": amp_on,
        "train_s": round(elapsed, 1),
        "epochs": v3.EPOCHS,
        "loss": loss,
        "tokenizer": "frozen_k32_from_v3_ckpt_120",
        "frozen_token_on_new": tok,
        "reachable_gaps_mm": gaps_ok,
        "trained_gap_bins": used_bins,
        "n_original": len(original),
        "n_gap_windows": len(gap_trajs),
        "n_local": len(local),
        "policy_backoff_old_model": policy_cmp,
        "audits": audits,
        "teacher": teacher,
        "closed_original_ten": closed_old,
        "closed_from_loss_snap": closed_gap,
        "closed_local": closed_local,
    }
    (RAW / "summary.json").write_text(json.dumps(jsonable(blob), indent=2), encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
