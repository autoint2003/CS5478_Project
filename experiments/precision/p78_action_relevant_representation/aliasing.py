"""Phase 5 — state aliasing vs action ambiguity (privileged labels for analysis only)."""

from __future__ import annotations

import json
import pickle

import numpy as np

from experiments.precision.p78_action_relevant_representation.features import engineered_from_obs27_hist, obs27
from experiments.precision.p78_action_relevant_representation.paths import (
    CACHE,
    ENG_HIST,
    FC_WINDOWS,
    OUT,
    P74_AUDIT,
    P77_SENSE,
    SEED,
)


def _z(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X - mu) / sd


def run_aliasing(cf_rows: list[dict] | None = None) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    # Prefer P77 sensing (aligned obs + sidewall) else P74
    rows = []
    if P77_SENSE.exists():
        eps = pickle.loads(P77_SENSE.read_bytes())
        for ep in eps:
            hist = []
            for s in ep["steps"]:
                o = np.asarray(s["obs27"], float)
                hist.append(o)
                if s["window"] not in FC_WINDOWS:
                    continue
                eng = engineered_from_obs27_hist(hist[-ENG_HIST:])
                rows.append(
                    {
                        "x_raw": o,
                        "x_eng": np.concatenate([o, eng]),
                        "sw": int(s["correct_sidewall"]),
                        "traj_id": ep["traj_id"],
                        "opp_ep": int(ep["reached_opposing"]),
                    }
                )
    else:
        raw = json.loads(P74_AUDIT.read_text(encoding="utf-8"))
        for ep in raw["rows"]:
            hist = []
            for s in ep["steps"]:
                o = np.zeros(27)
                v8 = np.asarray(s["obs_vec8"], float)
                for j, idx in enumerate((2, 3, 0, 1, 12, 13, 18, 19)):
                    o[idx] = v8[j]
                hist.append(o)
                if s.get("window") not in FC_WINDOWS:
                    continue
                eng = engineered_from_obs27_hist(hist[-ENG_HIST:])
                rows.append(
                    {
                        "x_raw": o,
                        "x_eng": np.concatenate([o, eng]),
                        "sw": int(s.get("correct_sidewall") or 0),
                        "traj_id": ep["traj_id"],
                        "opp_ep": int(ep.get("reached_opposing") or 0),
                    }
                )

    Xr = _z(np.stack([r["x_raw"] for r in rows]))
    Xe = _z(np.stack([r["x_eng"] for r in rows]))
    sw = np.asarray([r["sw"] for r in rows], np.int8)
    rng = np.random.default_rng(SEED)
    # radius from raw
    n = len(Xr)
    i, j = rng.integers(0, n, 2000), rng.integers(0, n, 2000)
    rad = float(np.quantile(np.linalg.norm(Xr[i] - Xr[j], axis=1), 0.05))

    def pair_stats(X):
        pos = np.where(sw == 1)[0]
        pairs = []
        for a in pos:
            d = np.linalg.norm(X - X[a], axis=1)
            cand = np.where((sw == 0) & (d > 0) & (d <= rad))[0]
            if len(cand) == 0:
                continue
            b = cand[np.argmin(d[cand])]
            pairs.append((a, b, float(d[b])))
        return pairs

    pairs_raw = pair_stats(Xr)
    pairs_eng = pair_stats(Xe)

    # Separation: among raw-matched different-sw pairs, does eng distance increase?
    sep = []
    for a, b, dr in pairs_raw:
        de = float(np.linalg.norm(Xe[a] - Xe[b]))
        sep.append({"d_raw": dr, "d_eng": de, "eng_larger": int(de > dr)})

    # Action evidence from CF rows if present
    action_classes = {"1_same_preferred_action": 0, "2_different_preferred_action": 0, "3_insufficient": 0}
    if cf_rows:
        # For each snap, preferred rule = argmax r3; compare snaps with similar A but different sw0
        by_snap = {}
        for r in cf_rows:
            by_snap.setdefault(r["snap_id"], {"sw0": r.get("sw0"), "rep_A": r["rep_A"], "best": None, "rules": {}})
            by_snap[r["snap_id"]]["rules"][r["rule"]] = r["r3_ready"]
        for sid, blob in by_snap.items():
            best = max(blob["rules"].items(), key=lambda kv: kv[1])[0] if blob["rules"] else None
            blob["best"] = best
        snaps = list(by_snap.values())
        # match different sw0
        for i in range(len(snaps)):
            for j in range(i + 1, len(snaps)):
                if snaps[i]["sw0"] == snaps[j]["sw0"]:
                    continue
                if snaps[i]["best"] is None or snaps[j]["best"] is None:
                    action_classes["3_insufficient"] += 1
                    continue
                da = float(np.linalg.norm(np.asarray(snaps[i]["rep_A"]) - np.asarray(snaps[j]["rep_A"])))
                if da > rad * 3:  # not obs-similar
                    continue
                if snaps[i]["best"] == snaps[j]["best"]:
                    action_classes["1_same_preferred_action"] += 1
                else:
                    action_classes["2_different_preferred_action"] += 1
    else:
        action_classes["3_insufficient"] = -1  # flag missing CF

    report = {
        "n_fc_steps": len(rows),
        "radius_raw_q05": rad,
        "n_raw_diff_sw_pairs": len(pairs_raw),
        "n_eng_diff_sw_pairs": len(pairs_eng),
        "eng_separates_raw_pairs_frac": float(np.mean([s["eng_larger"] for s in sep])) if sep else None,
        "mean_d_raw": float(np.mean([s["d_raw"] for s in sep])) if sep else None,
        "mean_d_eng": float(np.mean([s["d_eng"] for s in sep])) if sep else None,
        "action_ambiguity_classes": action_classes,
        "direct_action_relevant_aliasing": bool(
            action_classes.get("2_different_preferred_action", 0) > action_classes.get("1_same_preferred_action", 0)
            and action_classes.get("2_different_preferred_action", 0) >= 3
        ),
        "note": (
            "Only class 2 (different hidden geometry + different preferred action) supports "
            "action-relevant aliasing. Privileged sw for analysis only."
        ),
    }
    (OUT / "ACTION_RELEVANT_ALIASING.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "ACTION_RELEVANT_ALIASING.md").write_text(
        "# Action-Relevant Aliasing (P78 Phase 5)\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    rows = None
    p = CACHE / "cf_diag_rows.pkl"
    if p.exists():
        rows = pickle.loads(p.read_bytes())
    run_aliasing(rows)
