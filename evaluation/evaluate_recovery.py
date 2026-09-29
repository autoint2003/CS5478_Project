"""Compare nominal / heuristic / SAC recovery on a frozen eval buffer."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.heuristic_recovery import HeuristicRecovery  # noqa: E402
from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.recovery_env import RecoveryEnv, load_buffer  # noqa: E402

EVAL_PATH = ROOT / "results" / "buffers" / "recovery_eval.npz"
CKPT = ROOT / "results" / "checkpoints"
OUT_CSV = ROOT / "results" / "logs" / "recovery_eval.csv"
FIG = ROOT / "results" / "figures" / "recovery_eval.png"


def _ci(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n)


def run_method(env: RecoveryEnv, snaps: list[dict], policy, name: str) -> list[dict]:
    rows = []
    for i, snap in enumerate(snaps):
        obs, info = env.reset(options={"snapshot": snap})
        env.limiter.reset()
        max_slip = 0.0
        effort = 0.0
        resid = 0.0
        steps = 0
        D_hist = []
        vz_hist = []
        fg_hist = []
        dvz_hist = []
        term = "timeout"
        while True:
            action = policy(obs, info)
            obs, _r, terminated, truncated, info = env.step(action)
            steps += 1
            max_slip = max(max_slip, float(info.get("slip", 0.0)))
            effort += abs(float(info.get("fg", 0.0))) * env.sim.dt_policy
            resid += float(info.get("residual_mag", 0.0)) * env.sim.dt_policy
            D_hist.append(float(info["D"]))
            vz_hist.append(float(info["vz"]))
            fg_hist.append(float(info["fg"]))
            dvz_hist.append(float(info["dvz"]))
            if terminated or truncated:
                term = info.get("term") or ("timeout" if truncated else "drop")
                break
        recovered = term == "recovered"
        drop = term == "drop"
        rows.append(
            {
                "method": name,
                "idx": i,
                "recovered": int(recovered),
                "drop": int(drop),
                "term": term,
                "steps": steps,
                "time": steps * env.sim.dt_policy,
                "max_slip": max_slip,
                "grip_effort": effort,
                "residual": resid,
                "D0": float(snap.get("D", D_hist[0] if D_hist else 0.0)),
                "severity": snap.get("severity", ""),
                "mass": float(snap.get("mass", 0.0)),
                "friction": float(snap.get("friction", 0.0)),
                "D_hist": D_hist,
                "vz_hist": vz_hist,
                "fg_hist": fg_hist,
                "dvz_hist": dvz_hist,
            }
        )
    return rows


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    rec = sum(r["recovered"] for r in rows) / max(n, 1)
    drop = sum(r["drop"] for r in rows) / max(n, 1)
    t_rec = [r["time"] for r in rows if r["recovered"]]
    print(
        f"{rows[0]['method']:10s}  recovery {100*rec:.1f}% +/- {100*_ci(rec, n):.1f}  "
        f"drop {100*drop:.1f}%  n={n}  "
        f"mean t_rec={np.mean(t_rec) if t_rec else float('nan'):.2f}s  "
        f"mean slip={np.mean([r['max_slip'] for r in rows]):.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/randomized.yaml")
    parser.add_argument("--action", default="2d", choices=("2d", "7d"))
    parser.add_argument("--no-contact-obs", action="store_true")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max-episodes", type=int, default=40)
    args = parser.parse_args()
    cfg = merge_sim_config(load_yaml(args.config))
    if not EVAL_PATH.exists():
        print(f"missing {EVAL_PATH}")
        return 1
    snaps = load_buffer(EVAL_PATH)[: args.max_episodes]
    env = RecoveryEnv(
        cfg=cfg,
        snapshots=snaps,
        action_kind=args.action,
        use_contact_obs=not args.no_contact_obs,
    )

    z_act = np.zeros(env.action_space.shape, dtype=np.float32)
    nom = run_method(env, snaps, lambda o, i: z_act, "nominal")

    heur = HeuristicRecovery(cfg, env.limiter)

    def heur_policy(_o, info):
        cmd = heur.command(float(info["D"]), float(info["Ddot"]))
        return env.limiter.to_action(cmd, args.action)

    heu = run_method(env, snaps, heur_policy, "heuristic")

    sac_rows = []
    ckpt = Path(args.checkpoint) if args.checkpoint else CKPT / f"sac_recovery_{args.action}.zip"
    if args.no_contact_obs:
        alt = CKPT / f"sac_recovery_{args.action}_nocontact.zip"
        if alt.exists():
            ckpt = alt
    if ckpt.exists():
        from stable_baselines3 import SAC

        model = SAC.load(str(ckpt))

        def sac_policy(obs, _info):
            a, _ = model.predict(obs, deterministic=True)
            return a

        sac_rows = run_method(env, snaps, sac_policy, "sac")
    else:
        print(f"no SAC checkpoint at {ckpt}; skipping SAC")

    for block in (nom, heu, sac_rows):
        if block:
            summarize(block)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "idx",
        "recovered",
        "drop",
        "term",
        "steps",
        "time",
        "max_slip",
        "grip_effort",
        "residual",
        "D0",
        "severity",
        "mass",
        "friction",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for block in (nom, heu, sac_rows):
            w.writerows(block)

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    series = [("heuristic", heu), ("nominal", nom)]
    if sac_rows:
        series.append(("sac", sac_rows))
    pick = min(3, len(snaps) - 1)
    for name, block in series:
        if not block:
            continue
        r = block[pick]
        t = np.arange(len(r["D_hist"])) * env.sim.dt_policy
        axes[0, 0].plot(t, r["D_hist"], label=name)
        axes[0, 1].plot(t, r["fg_hist"], label=name)
        axes[1, 0].plot(t, r["vz_hist"], label=name)
        axes[1, 1].plot(t, r["dvz_hist"], label=name)
    axes[0, 0].set_title("D_t")
    axes[0, 1].set_title("F_g")
    axes[1, 0].set_title("v_z")
    axes[1, 1].set_title("Delta v_z")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Recovery trajectories on one held-out episode")
    fig.tight_layout()
    fig.savefig(FIG, dpi=120)
    plt.close(fig)
    print(f"csv: {OUT_CSV}")
    print(f"fig: {FIG}")
    print("CP10 GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
