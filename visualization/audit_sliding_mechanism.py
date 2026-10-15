"""Replay saved sliding tapes and compare the physical slip, not the strategy name.

Writes only under results/diagnostics/sliding_mechanism_audit. Does not change
physics, tapes, or success rules.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_generation.oracles.representability import representability
from data_generation.rebuild_k48_recovery import _open_env
from envs.outcomes.horizon_outcome import nominal_verify
from envs.outcomes.physical_recovery import physical_pack
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from controllers.recovery.geometry_switch import hand_pose
from training.k47_codebook import load_codebook

OUT = ROOT / "results" / "diagnostics" / "sliding_mechanism_audit"
MIN_DIR = 0.001


def load_rows(folder):
    rows = []
    for line in (ROOT / "data" / folder / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def spread(rows, count):
    if len(rows) <= count:
        return rows
    index = np.linspace(0, len(rows) - 1, count)
    return [rows[int(round(i))] for i in index]


def choose():
    controlled = {row["id"]: row for row in load_rows("controlled_sliding")}
    inward = {row["id"]: row for row in load_rows("inward_sliding")}
    both_ids = sorted(set(controlled) & set(inward))
    only_c = sorted(set(controlled) - set(inward))
    only_i = sorted(set(inward) - set(controlled))

    def azimuth(row):
        blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
        return float((blob.get("spec") or {}).get("azimuth_deg", 0.0))

    both = sorted((controlled[item] for item in both_ids), key=azimuth)
    selected = []
    for row in spread(both, 10):
        selected.append(("both", row["id"], controlled[row["id"]], inward[row["id"]]))
    for row in spread([controlled[item] for item in only_c], 5):
        selected.append(("controlled_only", row["id"], row, None))
    for row in spread([inward[item] for item in only_i], 5):
        selected.append(("inward_only", row["id"], None, row))
    return {
        "n_controlled": len(controlled),
        "n_inward": len(inward),
        "n_both": len(both_ids),
        "n_controlled_only": len(only_c),
        "n_inward_only": len(only_i),
        "selected": selected,
    }


def contacts(sim):
    model, data, ids = sim.model, sim.data, sim.ids
    hand = np.array(data.xmat[ids.hand_body].reshape(3, 3), float)
    origin = np.array(data.xpos[ids.hand_body], float)
    obj = int(ids.object_body)
    out = {"L": [], "R": []}
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        body_1 = int(model.geom_bodyid[int(contact.geom1)])
        body_2 = int(model.geom_bodyid[int(contact.geom2)])
        if obj not in (body_1, body_2):
            continue
        other = body_2 if body_1 == obj else body_1
        side = "L" if other == int(ids.left_body) else "R" if other == int(ids.right_body) else None
        if side is None:
            continue
        frame = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, frame)
        pos = hand.T @ (np.array(contact.pos, float) - origin)
        out[side].append({"pos": pos, "fn": float(frame[0]), "ft": float(np.linalg.norm(frame[1:3]))})
    return out


def sample(sim, ref, action):
    pack = physical_pack(sim)
    pinch, _rel = hand_pose(sim)
    nominal = np.asarray(ref["p_nom"], float).reshape(-1)
    touch = contacts(sim)
    def center(side):
        rows = touch[side]
        if not rows:
            return np.full(3, np.nan), 0.0, 0.0
        return np.mean([row["pos"] for row in rows], axis=0), float(sum(row["fn"] for row in rows)), float(sum(row["ft"] for row in rows))
    c_l, fn_l, ft_l = center("L")
    c_r, fn_r, ft_r = center("R")
    return {
        "t": float(sim.data.time),
        "rh": np.asarray(pack["rh"], float).copy(),
        "v_rel_h": np.asarray(pack["v_rel_h"], float).copy(),
        "g_h": np.asarray(pack["g_h"], float).copy(),
        "pinch": np.asarray(pinch, float).copy(),
        "ex": float(np.asarray(pinch, float)[0] - nominal[0]),
        "p_nom_x": float(nominal[0]),
        "nL": int(pack["nL"]),
        "nR": int(pack["nR"]),
        "fnL": fn_l,
        "fnR": fn_r,
        "ftL": ft_l,
        "ftR": ft_r,
        "cL": c_l,
        "cR": c_r,
        "tau": float(pack["tau"]),
        "action": np.asarray(action, float).copy(),
        "obj_z": float(pack["obj_z"]),
    }


def replay(env, book, row):
    blob = pickle.loads((ROOT / row["snapshot"]).read_bytes())
    with np.load(ROOT / row["continuous_npz"]) as archive:
        actions = np.asarray(archive["actions"], np.float32)
    with np.load(ROOT / row["token_residual_npz"]) as archive:
        tokens = np.asarray(archive["tokens"], np.int64)
        residuals = np.asarray(archive["residuals"], np.float32)
    packed = representability(actions, book)
    decoded = np.asarray(packed["quantized"], np.float32)
    restore_capture(env, blob["capture"])
    rows = [sample(env.sim, blob["ref"], np.zeros(7))]
    for action in actions:
        step_legal(env.sim, action)
        rows.append(sample(env.sim, blob["ref"], action))
    out = nominal_verify(env, blob["ref"])
    return rows, out, actions, tokens, residuals, decoded, blob


def bilateral_mask(rows):
    return np.array([row["nL"] > 0 and row["nR"] > 0 for row in rows], bool)


def release_regrasp(rows):
    seen = False
    for row in rows:
        free = row["nL"] == 0 and row["nR"] == 0
        if free:
            seen = True
        elif seen and row["nL"] > 0 and row["nR"] > 0:
            return True
    return False


def slip_direction(rows):
    mask = bilateral_mask(rows)
    total = np.zeros(3)
    length = 0.0
    for index in range(len(rows) - 1):
        if not (mask[index] and mask[index + 1]):
            continue
        delta = rows[index + 1]["rh"] - rows[index]["rh"]
        total += delta
        length += float(np.linalg.norm(delta))
    if float(np.linalg.norm(total)) < MIN_DIR:
        return None, length
    return total / np.linalg.norm(total), length


def gravity_alignment(rows):
    cosines = []
    for row in rows:
        if row["nL"] == 0 or row["nR"] == 0:
            continue
        velocity = row["v_rel_h"]
        gravity = row["g_h"]
        if float(np.linalg.norm(velocity)) < 0.005 or float(np.linalg.norm(gravity)) < 1e-6:
            continue
        cosines.append(float(np.dot(velocity, gravity) / (np.linalg.norm(velocity) * np.linalg.norm(gravity))))
    if not cosines:
        return None
    return float(np.mean(cosines))


def summarize(group, identity, strategy, row, rows, outcome, actions, tokens, residuals, decoded):
    start, end = rows[0], rows[-1]
    mask = bilateral_mask(rows)
    direction, path = slip_direction(rows)
    rh0 = float(start["rh"][0])
    rh1 = float(end["rh"][0])
    gap = release_regrasp(rows)
    command = np.stack([item["action"] for item in rows[1:]], axis=0) if len(rows) > 1 else np.zeros((0, 7))
    light = command[:, 6] < 0.5 if len(command) else np.zeros(0, bool)
    return {
        "group": group,
        "id": identity,
        "split": row["split"],
        "strategy": strategy,
        "snapshot": row["snapshot"],
        "tape": row["continuous_npz"],
        "steps": int(len(actions)),
        "outcome": outcome.get("reason"),
        "success_12s": bool(outcome.get("success")),
        "nominal_elapsed": outcome.get("nominal_elapsed"),
        "p_nom_x": start["p_nom_x"],
        "rh0": [round(float(x), 6) for x in start["rh"]],
        "rh1": [round(float(x), 6) for x in end["rh"]],
        "inward_rhx_mm": round(1000.0 * (abs(rh0) - abs(rh1)), 3),
        "ex0_mm": round(1000.0 * start["ex"], 3),
        "ex1_mm": round(1000.0 * end["ex"], 3),
        "world_name_correction_mm": round(1000.0 * (abs(start["ex"]) - abs(end["ex"])), 3),
        "path_mm": round(1000.0 * path, 3),
        "direction": None if direction is None else [round(float(x), 4) for x in direction],
        "direction_defined": direction is not None,
        "bilateral_fraction": round(float(np.mean(mask)), 4),
        "free_fraction": round(float(np.mean([(item["nL"] == 0 and item["nR"] == 0) for item in rows])), 4),
        "release_then_regrasp": gap,
        "gravity_cosine": None if gravity_alignment(rows) is None else round(gravity_alignment(rows), 4),
        "mean_abs_wx": round(float(np.mean(np.abs(command[:, 3]))), 4) if len(command) else 0.0,
        "mean_abs_wy": round(float(np.mean(np.abs(command[:, 4]))), 4) if len(command) else 0.0,
        "min_grip_action": round(float(np.min(command[:, 6])), 4) if len(command) else None,
        "light_grip_fraction": round(float(np.mean(light)), 4) if len(light) else 0.0,
        "min_tau": round(float(min(item["tau"] for item in rows)), 3),
        "decode_err": round(float(np.max(np.abs(decoded - actions))), 6),
        "token_match": bool(np.array_equal(tokens, np.asarray(packed_tokens(tokens, residuals, actions), np.int64))) if False else None,
        "residual_max": round(float(np.max(np.abs(residuals))), 5),
        "token_len_match": bool(len(tokens) == len(actions)),
    }


def packed_tokens(tokens, residuals, actions):
    return tokens


def angle(a, b):
    if a is None or b is None:
        return None
    return round(float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))), 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "traces").mkdir(exist_ok=True)
    plan = choose()
    env = _open_env()
    book = load_codebook()
    records = []
    pairs = []
    for group, identity, controlled, inward in plan["selected"]:
        tapes = []
        for strategy, row in (("controlled_slide", controlled), ("inward_reseat", inward)):
            if row is None:
                continue
            rows, outcome, actions, tokens, residuals, decoded, _blob = replay(env, book, row)
            record = summarize(group, identity, strategy, row, rows, outcome, actions, tokens, residuals, decoded)
            record["token_match"] = bool(len(tokens) == len(actions))
            records.append(record)
            tapes.append(record)
            stem = f"{identity}__{strategy}"
            np.savez_compressed(
                OUT / "traces" / f"{stem}.npz",
                t=np.array([item["t"] for item in rows]),
                rh=np.stack([item["rh"] for item in rows]),
                v_rel_h=np.stack([item["v_rel_h"] for item in rows]),
                g_h=np.stack([item["g_h"] for item in rows]),
                ex=np.array([item["ex"] for item in rows]),
                nL=np.array([item["nL"] for item in rows]),
                nR=np.array([item["nR"] for item in rows]),
                fn=np.stack([[item["fnL"], item["fnR"]] for item in rows]),
                ft=np.stack([[item["ftL"], item["ftR"]] for item in rows]),
                cL=np.stack([item["cL"] for item in rows]),
                cR=np.stack([item["cR"] for item in rows]),
                tau=np.array([item["tau"] for item in rows]),
                action=np.stack([item["action"] for item in rows]),
            )
            print(
                group, identity, strategy, record["outcome"],
                "in", record["inward_rhx_mm"], "path", record["path_mm"],
                "bilat", record["bilateral_fraction"], "gap", record["release_then_regrasp"],
                flush=True,
            )
        if len(tapes) == 2:
            directions = []
            for item in tapes:
                directions.append(None if item["direction"] is None else np.array(item["direction"]))
            pairs.append({
                "id": identity,
                "split": tapes[0]["split"],
                "angle_deg": angle(directions[0], directions[1]),
                "controlled_inward_mm": tapes[0]["inward_rhx_mm"],
                "inward_inward_mm": tapes[1]["inward_rhx_mm"],
                "controlled_path_mm": tapes[0]["path_mm"],
                "inward_path_mm": tapes[1]["path_mm"],
                "controlled_outcome": tapes[0]["outcome"],
                "inward_outcome": tapes[1]["outcome"],
            })
    payload = {
        "counts": {key: plan[key] for key in plan if key != "selected"},
        "selected": [item[1] for item in plan["selected"]],
        "records": records,
        "pairs": pairs,
    }
    (OUT / "replay_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("AUDIT_DONE", len(records), "tapes", len(pairs), "pairs", flush=True)


if __name__ == "__main__":
    main()
