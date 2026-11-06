"""Phase 2 — existing 39D / causal history information probes (IC-disjoint)."""

from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np

from experiments.precision.p77_contact_observability.metrics import eval_binary_probe
from experiments.precision.p77_contact_observability.paths import (
    FC_WINDOWS,
    OUT,
    P74_AUDIT,
    P75_DATA,
    SEED,
    SHORT_HIST,
    VAL_FRAC,
)


def _ic_split(ids: list[str], val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    ids = sorted(set(ids))
    rng.shuffle(ids)
    n_val = max(1, int(round(len(ids) * val_frac))) if len(ids) > 3 else max(1, len(ids) // 4)
    return set(ids[n_val:]), set(ids[:n_val])


def _pack_hist(vecs: list[np.ndarray], t: int, k: int) -> np.ndarray:
    lo = max(0, t + 1 - k)
    chunk = vecs[lo : t + 1]
    if len(chunk) < k:
        chunk = [chunk[0]] * (k - len(chunk)) + chunk
    return np.concatenate(chunk)


def _engineered_from_vec8_hist(hist: list[np.ndarray]) -> np.ndarray:
    """Causal engineered features from obs_vec8 = L,R,e_hat,valid,ap,ap_dot,vhx,vhy."""
    X = np.stack(hist)
    L, R, eh, vld, ap, apd, vhx, vhy = X.T
    persist_L = float(L.mean())
    persist_R = float(R.mean())
    both = float(np.mean((L > 0.5) & (R > 0.5)))
    # first contact side timing within window
    tL = next((i for i, x in enumerate(L) if x > 0.5), len(L))
    tR = next((i for i, x in enumerate(R) if x > 0.5), len(R))
    return np.asarray(
        [
            eh[-1],
            vld[-1],
            L[-1],
            R[-1],
            ap[-1],
            apd[-1],
            vhx[-1],
            vhy[-1],
            persist_L,
            persist_R,
            both,
            float(tL - tR),
            float(np.std(eh)),
            float(eh[-1] - eh[0]),
            float(np.linalg.norm([vhx[-1], vhy[-1]])),
            float(apd[-1] - apd[0]),
        ],
        dtype=np.float64,
    )


def analyze_p74_sidewall() -> dict:
    raw = json.loads(P74_AUDIT.read_text(encoding="utf-8"))
    # Build per-episode step lists
    rows = []
    for ep in raw["rows"]:
        tid = ep["traj_id"]
        vecs = []
        for s in ep["steps"]:
            vecs.append(np.asarray(s["obs_vec8"], float))
        for ti, s in enumerate(ep["steps"]):
            if s.get("window") not in FC_WINDOWS:
                continue
            # label: current correct sidewall (not future episode outcome)
            y_sw = int(s.get("correct_sidewall") or 0)
            # early loss prediction: will lose contact later in episode (outcome — diagnostic only for loss task)
            y_loss = int(ep.get("lost_contact_after_fc") or 0)
            y_opp = int(ep.get("reached_opposing") or 0)
            cur = vecs[ti]
            short = _pack_hist(vecs, ti, SHORT_HIST)
            # full history summary up to ti
            full = np.concatenate([vecs[ti], np.mean(vecs[: ti + 1], 0), np.std(vecs[: ti + 1], 0)])
            eng = _engineered_from_vec8_hist(vecs[max(0, ti + 1 - 5) : ti + 1])
            rows.append(
                {
                    "traj_id": tid,
                    "t": ti,
                    "y_sw": y_sw,
                    "y_loss": y_loss,
                    "y_opp": y_opp,
                    "A": cur,
                    "B": short,
                    "C": full,
                    "D": eng,
                }
            )
    trajs = [r["traj_id"] for r in rows]
    fit_ics, eval_ics = _ic_split(trajs, VAL_FRAC, SEED)
    fit = np.asarray([r["traj_id"] in fit_ics for r in rows])
    ev = np.asarray([r["traj_id"] in eval_ics for r in rows])

    out = {
        "n_fc_steps": len(rows),
        "n_fit_ics": len(fit_ics),
        "n_eval_ics": len(eval_ics),
        "sidewall_prevalence_eval": float(np.mean([r["y_sw"] for r in rows if r["traj_id"] in eval_ics]))
        if any(r["traj_id"] in eval_ics for r in rows)
        else None,
        "tasks": {},
    }
    for task, ykey in (("correct_sidewall_now", "y_sw"), ("eventual_contact_loss", "y_loss"), ("eventual_opposing", "y_opp")):
        out["tasks"][task] = {}
        for rep in ("A", "B", "C", "D"):
            X = np.stack([r[rep] for r in rows])
            y = np.asarray([r[ykey] for r in rows])
            out["tasks"][task][rep] = eval_binary_probe(X, y, fit, ev)
    return out


def _eng39(frames: np.ndarray, t: int) -> np.ndarray:
    """Engineered causal features from full 39D history prefix."""
    lo = max(0, t + 1 - 8)
    X = frames[lo : t + 1]
    o = frames[t]
    # indices: 0 e_hat, 2/3 L/R, 6-9 uv, 10-11 fn, 12-13 ap, 18-20 vel
    L, R = X[:, 2], X[:, 3]
    eh = X[:, 0]
    fnL, fnR = X[:, 10], X[:, 11]
    uL, vL, uR, vR = X[:, 6], X[:, 7], X[:, 8], X[:, 9]
    return np.concatenate(
        [
            o[:27],  # observable only (exclude GT airborne 27-38 for deployable probe)
            [
                float(L.mean()),
                float(R.mean()),
                float(np.mean((L > 0.5) & (R > 0.5))),
                float(eh.std()),
                float(eh[-1] - eh[0]),
                float(fnL[-1] - fnR[-1]),
                float(uL[-1] - uR[-1]),
                float(vL[-1] - vR[-1]),
                float(np.mean(np.abs(np.diff(L)))) if len(L) > 1 else 0.0,
                float(np.mean(np.abs(np.diff(R)))) if len(R) > 1 else 0.0,
            ],
        ]
    )


def analyze_p75_outcomes() -> dict:
    train = pickle.loads((P75_DATA / "train_episodes.pkl").read_bytes())
    eval_eps = pickle.loads((P75_DATA / "eval_episodes.pkl").read_bytes())
    # Use P75 official IC-disjoint split
    fit_ics = {e["traj_id"] for e in train if e["skill"] == "C"}
    eval_ics = {e["traj_id"] for e in eval_eps if e["skill"] == "C"}

    def build(eps, ics):
        rows = []
        for ep in eps:
            if ep["traj_id"] not in ics:
                continue
            frames = ep["obs_frames"]
            for t in range(ep["n_steps"]):
                if ep["stages"][t] not in FC_WINDOWS:
                    continue
                rows.append(
                    {
                        "traj_id": ep["traj_id"],
                        "A": frames[t, :27].astype(np.float64),  # deployable current
                        "A39": frames[t].astype(np.float64),
                        "B": np.concatenate(
                            [frames[max(0, t - SHORT_HIST + 1) : t + 1, :27].mean(0), frames[t, :27]]
                        ),
                        "C": np.concatenate(
                            [frames[t, :27], frames[: t + 1, :27].mean(0), frames[: t + 1, :27].std(0)]
                        ),
                        "D": _eng39(frames, t),
                        "y_opp": int(ep["reached_opposing"]),
                        "y_loss": int(ep["lost_contact_after_fc"]),
                    }
                )
        return rows

    # Fit on train C, eval on eval C
    fit_rows = build(train, fit_ics)
    eval_rows = build(eval_eps, eval_ics)
    # combine with masks
    all_rows = fit_rows + eval_rows
    fit = np.asarray([True] * len(fit_rows) + [False] * len(eval_rows))
    ev = ~fit
    out = {"n_fit": len(fit_rows), "n_eval": len(eval_rows), "tasks": {}}
    for task, ykey in (("eventual_opposing", "y_opp"), ("eventual_contact_loss", "y_loss")):
        out["tasks"][task] = {}
        for rep in ("A", "A39", "B", "C", "D"):
            X = np.stack([r[rep] for r in all_rows])
            y = np.asarray([r[ykey] for r in all_rows])
            # pad A39 note: includes GT airborne — diagnostic upper bound if used
            blob = eval_binary_probe(X, y, fit, ev)
            if rep == "A39":
                blob["label"] = "includes_GT_airborne_27_38_diagnostic_only"
            if rep == "A":
                blob["label"] = "deployable_observable_27d"
            out["tasks"][task][rep] = blob
    return out


def run_existing_obs() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "p74_sidewall_detection": analyze_p74_sidewall(),
        "p75_outcome_prediction": analyze_p75_outcomes(),
        "representations": {
            "A": "current observation",
            "B": "short causal history",
            "C": "full causal history summary",
            "D": "causal engineered features from existing sensors",
            "A39": "current 39D including GT airborne (diagnostic / not deployable as-is)",
        },
    }
    (OUT / "EXISTING_OBSERVATION_INFORMATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # compact md
    compact = {
        "p74_sidewall": {
            k: {r: v.get("auroc") for r, v in task.items()}
            for k, task in report["p74_sidewall_detection"]["tasks"].items()
        },
        "p75_outcomes": {
            k: {r: v.get("auroc") for r, v in task.items()}
            for k, task in report["p75_outcome_prediction"]["tasks"].items()
        },
        "p74_sw_prevalence": report["p74_sidewall_detection"]["sidewall_prevalence_eval"],
    }
    (OUT / "EXISTING_OBSERVATION_INFORMATION.md").write_text(
        "# Existing Observation Information (P77 Phase 2)\n\nIC-disjoint linear/logistic probes.\n\n```json\n"
        + json.dumps(compact, indent=2)
        + "\n```\n\nFull:\n```json\n"
        + json.dumps(report, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(compact, indent=2), flush=True)
    return report


if __name__ == "__main__":
    run_existing_obs()
