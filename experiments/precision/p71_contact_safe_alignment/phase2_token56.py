"""Phase 2 — Token 56 contact dynamics."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import StabilizeGrpoEnv
from experiments.precision.p60_c_stable_state_grpo.full_state import restore_full_state, take_full_state
from experiments.precision.p61_reproducible_joint_eval.seeding import rollout_seed, seed_all
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor, load_critic
from experiments.precision.p63_transformer_critic_ppo.sampling import act_with_value
from experiments.precision.p64_ppo_multiround.schedule import blocked_eval_ids
from experiments.precision.p66_c_curriculum_ppo.expert_drive import ensure_inner_capture_buffers
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry
from experiments.precision.p71_contact_safe_alignment.align_tokens import decode_token
from experiments.precision.p71_contact_safe_alignment.paths import (
    MAX_EP_STEPS,
    N_TOK56_STATES,
    OUT,
    P64_STEP4_ACTOR,
    P64_STEP4_CRITIC,
    P69_OUT,
    P69_SPLIT,
    SEED,
    STATE_DIR,
    T_MAX,
    TOKEN_56,
)


def _lookup():
    blocked = blocked_eval_ids()
    return {
        s.traj_id: s
        for s in build_p54_dataset(seed=SEED)["schedule"]
        if s.skill == "C" and s.traj_id not in blocked
    }


def _states(env, actor, critic, device, n: int) -> list[dict]:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    lookup = _lookup()
    p69 = sorted((P69_OUT / "states").glob("disc_*.pkl")) if (P69_OUT / "states").exists() else []
    for path in p69:
        if len(out) >= n:
            break
        state = pickle.loads(path.read_bytes())
        tid = (state.get("meta") or {}).get("traj_id")
        if tid not in lookup:
            continue
        dst = STATE_DIR / path.name
        if not dst.exists():
            dst.write_bytes(path.read_bytes())
        out.append({"path": str(dst), "traj_id": tid, "id": path.stem})
    if len(out) >= n // 2:
        return out[:n]

    split = json.loads(Path(P69_SPLIT).read_text(encoding="utf-8"))
    for tid in split["discovery_ics"]:
        if len(out) >= n or tid not in lookup:
            continue
        st = lookup[tid]
        for i in range(3):
            if len(out) >= n:
                break
            seed = rollout_seed(split="train", traj_id=tid, rollout_i=99200 + i)
            seed_all(seed, deterministic_torch=True)
            env.reset(st)
            ensure_inner_capture_buffers(env)
            last = last_m = None
            done = False
            while not done:
                g = relative_geometry(env.env.sim)
                if not g["has_pad_contact"] and env.t <= 14:
                    ensure_inner_capture_buffers(env)
                    last = take_full_state(env, meta={"traj_id": tid, "t": int(env.t), "e_pos_mm": g["e_pos_mm"]})
                    last_m = {"e": float(g["e_pos_mm"]), "t": int(env.t)}
                tok, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device)
                _r, done, _ = env.step(tok, resid)
                if last is not None and relative_geometry(env.env.sim)["has_pad_contact"]:
                    sid = f"tok56_{len(out):04d}"
                    path = STATE_DIR / f"{sid}.pkl"
                    path.write_bytes(pickle.dumps(last))
                    out.append({"path": str(path), "traj_id": tid, "id": sid, "e0": last_m["e"]})
                    break
    return out[:n]


def run_phase2(*, device: str = "cuda") -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    device_t = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    actor, book, rho, _ = load_actor(P64_STEP4_ACTOR, device_t)
    critic = load_critic(P64_STEP4_CRITIC, actor, device_t)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    lookup = _lookup()
    states = _states(env, actor, critic, device_t, N_TOK56_STATES)
    dec = decode_token(book, TOKEN_56)
    print(f"[tok56] n_states={len(states)} decode={dec}", flush=True)

    rows = []
    by_root = defaultdict(list)
    for sm in states:
        start = lookup[sm["traj_id"]]
        state = pickle.loads(Path(sm["path"]).read_bytes())
        restore_full_state(env, state, start)
        ensure_inner_capture_buffers(env)
        g0 = relative_geometry(env.env.sim)
        obj0 = np.asarray(g0["obj_pos"], float)
        hand0 = np.asarray(g0["hand_pos"], float)
        rel0 = obj0 - hand0
        e0 = float(g0["e_pos_mm"])
        # Execute ONLY token 56 once
        _t, resid, _, _ = act_with_value(actor, critic, env.obs_seq, env.prev_seq, device_t)
        env.step(int(TOKEN_56), resid)
        g1 = relative_geometry(env.env.sim)
        obj1 = np.asarray(g1["obj_pos"], float)
        hand1 = np.asarray(g1["hand_pos"], float)
        rel1 = obj1 - hand0
        # Lateral: hand-frame-ish xy of object relative vector change
        lat0 = float(np.linalg.norm(rel0[:2]))
        lat1 = float(np.linalg.norm((obj1 - hand1)[:2]))
        row = {
            "id": sm["id"],
            "traj_id": sm["traj_id"],
            "root": sm["traj_id"].split("__")[1] if "__" in sm["traj_id"] else sm["traj_id"][:20],
            "e0": e0,
            "e1": float(g1["e_pos_mm"]),
            "delta_e": float(g1["e_pos_mm"] - e0),
            "theta0": float(g0["theta_deg"]),
            "theta1": float(g1["theta_deg"]),
            "delta_theta": float(g1["theta_deg"] - g0["theta_deg"]),
            "rel_dist0": float(g0["rel_dist_m"]),
            "rel_dist1": float(g1["rel_dist_m"]),
            "delta_rel_dist": float(g1["rel_dist_m"] - g0["rel_dist_m"]),
            "lat0": lat0,
            "lat1": lat1,
            "delta_lat": lat1 - lat0,
            "obj_disp_m": float(np.linalg.norm(obj1 - obj0)),
            "hand_disp_m": float(np.linalg.norm(hand1 - hand0)),
            "contact0": bool(g0["has_pad_contact"]),
            "contact1": bool(g1["has_pad_contact"]),
            "first_contact": bool(not g0["has_pad_contact"] and g1["has_pad_contact"]),
            "min_pad1": int(g1["min_pad"]),
            "sidewall1": bool(g1["correct_sidewall"]),
            "nL1": int(g1["nL"]),
            "nR1": int(g1["nR"]),
            "vrel1": float(g1.get("vrel") or 0),
        }
        rows.append(row)
        by_root[row["root"]].append(row)
        print(
            f"[tok56] {sm['id']} de={row['delta_e']:+.1f} fc={row['first_contact']} "
            f"drel={row['delta_rel_dist']:+.4f} dlat={row['delta_lat']:+.4f} odisp={row['obj_disp_m']:.4f}",
            flush=True,
        )

    labels = []
    # A: improves e and approaches (rel_dist decreases) without pushing object far
    # B: causes first contact
    # C: object displacement large while e worsens or lateral worsens
    for r in rows:
        tags = []
        if r["delta_e"] < -1 and r["delta_rel_dist"] < 0:
            tags.append("A_align_while_approach")
        if r["first_contact"]:
            tags.append("B_contact_before_aligned")
        if r["obj_disp_m"] > 0.01 and (r["delta_e"] > 0 or r["delta_lat"] > 0.005):
            tags.append("C_pushes_object")
        if not tags:
            tags.append("mixed_or_neutral")
        r["labels"] = tags
        for t in tags:
            labels.append(t)

    from collections import Counter

    cnt = Counter(labels)
    # subtype variation
    subtype = {}
    for root, rs in by_root.items():
        subtype[root] = {
            "n": len(rs),
            "mean_delta_e": float(np.mean([x["delta_e"] for x in rs])),
            "frac_first_contact": float(np.mean([x["first_contact"] for x in rs])),
            "mean_obj_disp": float(np.mean([x["obj_disp_m"] for x in rs])),
        }

    report = {
        "token": TOKEN_56,
        "decode": dec,
        "n_states": len(rows),
        "mean_delta_e": float(np.mean([r["delta_e"] for r in rows])) if rows else None,
        "frac_first_contact": float(np.mean([r["first_contact"] for r in rows])) if rows else None,
        "frac_e_improves": float(np.mean([r["delta_e"] < 0 for r in rows])) if rows else None,
        "mean_obj_disp_m": float(np.mean([r["obj_disp_m"] for r in rows])) if rows else None,
        "mean_delta_rel_dist": float(np.mean([r["delta_rel_dist"] for r in rows])) if rows else None,
        "mean_delta_lat": float(np.mean([r["delta_lat"] for r in rows])) if rows else None,
        "label_counts": dict(cnt),
        "dominant": cnt.most_common(1)[0][0] if cnt else None,
        "by_subtype": subtype,
        "note_grip": "token 56 has grip=-1.0 (opening) with strong +Y/+Z linear — approaches while opening",
        "rows": rows,
    }
    (OUT / "TOKEN56_CONTACT_DYNAMICS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# Token 56 Contact Dynamics (P71 Phase 2)\n",
        f"```json\n{json.dumps({k: report[k] for k in report if k != 'rows'}, indent=2)}\n```\n",
        "Do not assume repeating token 56 is beneficial.\n",
    ]
    (OUT / "TOKEN56_CONTACT_DYNAMICS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("decode", "frac_first_contact", "label_counts", "dominant")}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_phase2()
