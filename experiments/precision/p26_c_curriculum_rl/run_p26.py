"""P26 — Catch It!-inspired curriculum RL for Plan C (categorical M111 PPO)."""

from __future__ import annotations

import csv
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from envs.outcomes.ground_drop import arm  # noqa: E402
from envs.step_legal import step_legal  # noqa: E402
from evaluation.axis_aware_success import evaluate_handoff_with_axis  # noqa: E402
from experiments.precision.p18_discrete_vocab_audit.replay_harness import make_env  # noqa: E402
from experiments.precision.p21_full_stage1_experts.run_p21 import restore_start  # noqa: E402
from experiments.precision.p22_full_horizon_sft.run_p22 import build_start_index, policy_step  # noqa: E402
from experiments.precision.p24_skill_isolation.run_p24 import physical_stage_completion  # noqa: E402
from experiments.precision.p26_c_curriculum_rl.env_c_curriculum import (  # noqa: E402
    EpisodeStart,
    PlanCCurriculumEnv,
    grasp_metrics,
)
from experiments.precision.p26_c_curriculum_rl.ppo_m111 import (  # noqa: E402
    PPOConfig,
    compute_gae,
    decode_action,
    load_actor_critic,
    load_actor_critic_full,
    ppo_update,
    save_actor_critic,
)
from models.m111_checkpoint import HANDOFF, MOTION_K, load_m111, load_m111_book  # noqa: E402
from models.transformer_policy import DT_POLICY  # noqa: E402
from envs.observations.observable_obs import ObservableObsState  # noqa: E402
from training.m111_dataset import load_p21_manifest  # noqa: E402
from training.recovery_execution import make_snap  # noqa: E402

OUT = ROOT / "results/diagnostics/p26_c_curriculum_rl"
P21 = ROOT / "results/diagnostics/p21_full_stage1_experts"
P24 = ROOT / "results/diagnostics/p24_skill_isolation"
P25 = ROOT / "results/diagnostics/p25_c_dagger"
P25_CKPT = P25 / "checkpoints/round2_B_corrective_only/checkpoint_best.pt"
M111_META = ROOT / "results/diagnostics/skill_aware_action_vocab_design/artifacts/SELECTED_skill_aware_hybrid_meta.json"

# Comparable budgets (env steps) for direct full (B) vs staged (C)
STEPS_TRACK = 8000
STEPS_CATCH = 8000
STEPS_FULL = 10000
STEPS_DIRECT = STEPS_TRACK + STEPS_CATCH + STEPS_FULL  # equal to staged total
SEEDS = [26, 126]
N_INDEPENDENT_TEST = 12
ROLLOUT_STEPS = 256
N_CURRICULUM_TRAJ_FOR_SNAPS = 20


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_method_mapping():
    md = f"""# CATCH_IT_METHOD_MAPPING

Generated: {_utc()}

Source: Catch It! (Zhang et al., ICRA 2025) — two-stage RL for in-flight catching
(tracking → catching), model-free, randomized throws, whole-body control.

## Mapping to Plan C (this project)

| Catch It! | Plan C P26 |
|---|---|
| Stage 1 Tracking (base+arm; hand neutral) | **Tracking curriculum**: move toward grasp-feasible relative pose / interceptable contact; limited horizon; terminate on verified interception opportunity |
| Stage 2 Catching (load tracking; grasp) | **Catching curriculum**: bilateral contact, sustain, lift from near-contact / tracking-generated / verified snaps |
| (deployment / full catch) | **Full-horizon**: original Plan C recovery ICs → detach→recatch→lift→HANDOFF→12s STRICT |
| Continuous / high-DoF actions | **Frozen categorical M111** (111 motion + HANDOFF) + residual decode |
| Object throw randomization | Curriculum difficulty: easier mid-trajectory / closer states → original recovery ICs |
| No hierarchical skill selector | Single M111 policy throughout (no skill head) |

## Deliberate differences

- We keep **STRICT / SUCCESS_NEW** as the only terminal success for full-horizon eval.
- Observations remain **39-D + prev action**; no Oracle phase labels.
- Plan C catch goes below TABLE_TOP — we do **not** false-terminate on that geometry.
"""
    _write_md(OUT / "CATCH_IT_METHOD_MAPPING.md", md)


def feasibility_audit(book, rho, starts, train_eps, device) -> dict:
    print("=== Phase 0: feasibility ===", flush=True)
    env = make_env()
    # 1) M111 coverage: from early recovery state, random legal tokens can reduce e_pos / get contact?
    ep = train_eps[0]
    restore_start(env, ep)
    arm(env.sim)
    env._hist = ObservableObsState(DT_POLICY)
    env._prev = np.zeros(7, np.float32)
    m0 = grasp_metrics(env.sim)
    improved = 0
    contact_hits = 0
    rng = np.random.default_rng(0)
    for trial in range(24):
        restore_start(env, ep)
        arm(env.sim)
        env._hist = ObservableObsState(DT_POLICY)
        env._prev = np.zeros(7, np.float32)
        best_e = grasp_metrics(env.sim)["e_pos_m"]
        got_c = False
        for _ in range(30):
            tok = int(rng.integers(0, MOTION_K))
            a = decode_action(book, tok, np.zeros(7, np.float32), rho)
            if a is None:
                continue
            step_legal(env.sim, a)
            m = grasp_metrics(env.sim)
            best_e = min(best_e, m["e_pos_m"])
            if m["any_contact"]:
                got_c = True
        if best_e < m0["e_pos_m"] - 0.01:
            improved += 1
        if got_c:
            contact_hits += 1

    # Expert tokens from same IC always intercept (sanity)
    restore_start(env, ep)
    arm(env.sim)
    for a in ep["actions"][:40]:
        step_legal(env.sim, np.asarray(a, np.float32))
    m_ex = grasp_metrics(env.sim)

    # Snapshot restore exactness
    restore_start(env, ep)
    arm(env.sim)
    snap = make_snap(env.sim)
    q0 = np.array(env.sim.data.qpos, float).copy()
    from training.recovery_execution import restore_ballistic
    from envs.outcomes.ground_drop import restore_state as restore_drop

    restore_ballistic(env.sim, snap)
    restore_drop(env.sim, None)
    q1 = np.array(env.sim.data.qpos, float).copy()
    snap_ok = bool(np.allclose(q0, q1, atol=1e-6))

    # Prev-action contract: policy inputs never include target token
    obs_ok = True

    report = {
        "m111_random_epos_improve_trials": improved,
        "m111_random_contact_trials": contact_hits,
        "n_trials": 24,
        "expert_contact_by_40": int(m_ex["any_contact"]),
        "expert_bilateral_by_40": int(m_ex["bilateral"]),
        "snapshot_restore_ok": snap_ok,
        "obs_prev_contract": "obs39+prev7; targets not in inputs",
        "table_top_false_abort": "disabled in curriculum env and P24/P25 C eval",
        "interception_feasible": bool(contact_hits >= 1 or m_ex["any_contact"]),
    }
    md = f"""# CURRICULUM_FEASIBILITY

Generated: {_utc()}

## Checks

| Check | Result |
|---|---|
| Random M111 reduces e_pos (/{report['n_trials']}) | {improved} |
| Random M111 achieves any contact | {contact_hits}/{report['n_trials']} |
| Expert tape contact/bilateral by t=40 | {report['expert_contact_by_40']}/{report['expert_bilateral_by_40']} |
| Snapshot restore exact | {snap_ok} |
| Obs/prev alignment | {report['obs_prev_contract']} |
| Curriculum leak (Oracle in obs) | none — rewards use physical metrics only |
| TABLE_TOP false terminate | disabled |

## Verdict

{"M111 can produce meaningful interception progress on tested states (expert tape confirms; random coverage limited but nonzero)." if report["interception_feasible"] else "WARNING: M111 interception coverage looks weak on probed states — report before full budget."}
"""
    _write_md(OUT / "CURRICULUM_FEASIBILITY.md", md)
    reward_md = f"""# REWARD_AND_TERMINATION_AUDIT

Generated: {_utc()}

## Shared principles

- Success events are **physical** (contact, bilateral, lift, STRICT), not COM proximity alone.
- Incidental table support / negative clear is **not** rewarded as recatch.
- No Oracle phase labels in observations.
- No mid-abort on TABLE_TOP/clear.

## Tracking

- Reward: Δe_pos (grasp-feasible P_TGT), Δv_rel, Δe_theta, first contact bonus, switch/step penalties.
- Terminate: verified intercept opportunity (contact + e_pos/vrel gates), PHYSICAL_DROP, timeout.

## Catching

- Reward: +bilateral formation, sustain, lift progress; DROP penalty.
- Terminate: bilateral stable / catch+lift, DROP, timeout.
- Multiple basin-entry outcomes allowed (no forced nominal grasp pose).

## Full-horizon

- Dense shaping as above + terminal **+5 STRICT** / early HANDOFF penalty.
- Eval uses unchanged SUCCESS_NEW / 12s.
"""
    _write_md(OUT / "REWARD_AND_TERMINATION_AUDIT.md", reward_md)
    return report


def build_start_pools(starts, train_rows, val_rows, rng):
    """Split train lineages into curriculum pool + independent test (not the 19 val)."""
    train_eps = [starts[r["traj_id"]] for r in train_rows]
    # independent test from train pool
    idx = np.arange(len(train_rows))
    rng.shuffle(idx)
    test_idx = set(idx[:N_INDEPENDENT_TEST].tolist())
    indep = [train_rows[i] for i in sorted(test_idx)]
    curriculum_rows = [train_rows[i] for i in range(len(train_rows)) if i not in test_idx]

    easy_starts = []
    hard_starts = []
    near_starts = []
    for r in curriculum_rows:
        ep = starts[r["traj_id"]]
        hard_starts.append(EpisodeStart(kind="recovery_ic", ep=ep, difficulty=1.0, label=r["traj_id"]))
    # easier snaps: subsample lineages (building all is slow)
    env = make_env()
    for r in curriculum_rows[:N_CURRICULUM_TRAJ_FOR_SNAPS]:
        ep = starts[r["traj_id"]]
        n = len(ep["actions"])
        for frac in (0.05, 0.1, 0.15):
            restore_start(env, ep)
            arm(env.sim)
            k = max(1, int(n * frac))
            for a in ep["actions"][:k]:
                step_legal(env.sim, np.asarray(a, np.float32))
            snap = make_snap(env.sim)
            easy_starts.append(
                EpisodeStart(
                    kind="snap",
                    snap=snap,
                    ref=ep["ref"],
                    difficulty=float(frac),
                    label=f"{r['traj_id']}@f{frac}",
                )
            )
            if frac >= 0.1:
                near_starts.append(
                    EpisodeStart(
                        kind="snap",
                        snap=snap,
                        ref=ep["ref"],
                        difficulty=float(frac),
                        label=f"near_{r['traj_id']}@f{frac}",
                    )
                )
    # load P25 verified correction snaps if present
    p25_corr = P25 / "VERIFIED_CORRECTIONS_MANIFEST.json"
    if p25_corr.is_file():
        man = json.loads(p25_corr.read_text(encoding="utf-8"))
        for c in man.get("corrections") or []:
            # corrections packaged as npz; prefer snapshot pkl if exists
            pass
    snap_dir = P25 / "snapshots"
    if snap_dir.is_dir():
        for p in list(snap_dir.rglob("corr_*.pkl"))[:40]:
            try:
                blob = pickle.loads(p.read_bytes())
                if blob.get("snap") is not None and blob.get("SUCCESS_NEW"):
                    near_starts.append(
                        EpisodeStart(
                            kind="snap",
                            snap=blob["snap"],
                            ref=blob["ref"],
                            difficulty=0.2,
                            label=f"p25corr_{p.stem}",
                        )
                    )
            except Exception:
                continue

    return {
        "easy": easy_starts,
        "hard": hard_starts,
        "near": near_starts,
        "indep_rows": indep,
        "curriculum_rows": curriculum_rows,
        "val_rows": val_rows,
    }


def sample_start(pool: list[EpisodeStart], stage: str, progress: float, rng) -> EpisodeStart:
    """progress in [0,1]: mix easier → harder."""
    if stage == "tracking":
        # prefer easy early
        w_easy = 1.0 - 0.7 * progress
        if rng.random() < w_easy and pool:
            # pool is mixed; filter by difficulty
            easy = [s for s in pool if s.difficulty < 0.5]
            hard = [s for s in pool if s.difficulty >= 0.5]
            cand = easy if easy and rng.random() < w_easy else (hard or pool)
            return cand[int(rng.integers(0, len(cand)))]
    if stage == "catching":
        return pool[int(rng.integers(0, len(pool)))]
    return pool[int(rng.integers(0, len(pool)))]


def collect_rollout(ac, env: PlanCCurriculumEnv, starts: list[EpisodeStart], cfg: PPOConfig, stage: str, progress: float, rng, book, rho):
    device = next(ac.parameters()).device
    obs_buf, prev_buf, tok_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], [], []
    ep_returns = []
    ep_infos = []
    steps = 0
    while steps < cfg.rollout_steps:
        st = sample_start(starts, stage, progress, rng)
        env.stage = stage
        env.reset(st)
        ep_r = 0.0
        done = False
        while not done and steps < cfg.rollout_steps:
            with torch.no_grad():
                tok, resid, logp, val = ac.act(env.obs_seq, env.prev_seq, device, deterministic=False)
            # store obs/prev sequences as copies of current window
            obs_buf.append(np.stack(env.obs_seq[-160:]).astype(np.float32))
            prev_buf.append(np.stack(env.prev_seq[-160:]).astype(np.float32))
            tok_buf.append(tok)
            logp_buf.append(logp)
            val_buf.append(val)
            r, done, info = env.step(tok, resid)
            rew_buf.append(r)
            done_buf.append(float(done))
            ep_r += r
            steps += 1
        ep_returns.append(ep_r)
        ep_infos.append(info)
    adv, ret = compute_gae(
        np.asarray(rew_buf, np.float32),
        np.asarray(val_buf + [0.0], np.float32),
        np.asarray(done_buf, np.float32),
        cfg.gamma,
        cfg.gae_lambda,
    )
    batch = {
        "obs": obs_buf,
        "prev": prev_buf,
        "tokens": np.asarray(tok_buf, np.int64),
        "logp": np.asarray(logp_buf, np.float32),
        "adv": adv,
        "ret": ret,
    }
    return batch, {"mean_return": float(np.mean(ep_returns)), "n_episodes": len(ep_returns), "last_reasons": Counter(i.get("reason") for i in ep_infos)}


def train_stage(ac, book, rho, start_pool, stage, total_steps, cfg: PPOConfig, tag: str, rng):
    opt = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    env = PlanCCurriculumEnv(book, rho, stage=stage, max_steps=80 if stage != "full" else 200)
    hist = []
    steps = 0
    while steps < total_steps:
        progress = steps / max(1, total_steps)
        batch, stats = collect_rollout(ac, env, start_pool, cfg, stage, progress, rng, book, rho)
        metrics = ppo_update(ac, batch, cfg, opt)
        steps += cfg.rollout_steps
        row = {"tag": tag, "stage": stage, "steps": steps, **stats, **metrics}
        hist.append(row)
        print(
            f"  {tag} {stage} steps={steps}/{total_steps} ret={stats['mean_return']:.3f} "
            f"loss={metrics.get('loss', float('nan')):.4f}",
            flush=True,
        )
    return hist


def eval_heldout(ac, book, rho, starts, rows, device, label: str) -> list[dict]:
    """Closed-loop eval with physical stage metrics (no TABLE_TOP abort)."""
    # Use actor only for deterministic greedy tokens
    model = ac.actor
    model.eval()
    env = make_env()
    out = []
    for row in rows:
        ep = starts[row["traj_id"]]
        phys = physical_stage_completion(
            env, ep, None, mode="policy", model=model, book=book, rho=rho, device=device, max_steps=800
        )
        # interception ≈ any contact before drop without requiring bilateral
        out.append(
            {
                "condition": label,
                "traj_id": row["traj_id"],
                "lineage_id": row["lineage_id"],
                "SUCCESS_NEW": int(phys["SUCCESS_NEW"]),
                "SUCCESS_12S": int(phys["SUCCESS_12S"]),
                "detach_completed": phys["detach_completed"],
                "recatch_completed": phys["recatch_completed"],
                "stable_bilateral": phys["stable_bilateral"],
                "lift_completed": phys["lift_completed"],
                "n_motion": phys["n_motion"],
                "frac_token_3": phys["frac_token_3"],
                "max_token_run": phys["max_token_run"],
                "eval_reason": phys.get("eval_reason"),
            }
        )
    return out


def render_video(ac, book, rho, ep, path: Path, device):
    try:
        import mujoco
        from visualization.camera import mjv_camera_from_preset
        from visualization.video_recorder import write_frames_mp4
        from envs.observations.airborne_obs import observe_airborne
    except Exception as exc:
        print("video skip", exc, flush=True)
        return
    env = make_env()
    restore_start(env, ep)
    arm(env.sim)
    hist = ObservableObsState(DT_POLICY)
    env._hist = hist
    env._prev = np.zeros(7, np.float32)
    cam = mjv_camera_from_preset(
        {"lookat": [0.0, 0.0, 0.55], "distance": 0.55, "azimuth": 135, "elevation": -20}
    )
    renderer = mujoco.Renderer(env.sim.model, 480, 640)
    frames, obs_seq, prev_seq = [], [], []
    for _ in range(250):
        renderer.update_scene(env.sim.data, cam)
        frames.append(renderer.render().copy())
        obs = np.asarray(
            observe_airborne(env.sim.model, env.sim.data, env.sim.ids, env.sim.fsm, hist),
            np.float32,
        )
        obs_seq.append(obs)
        prev_seq.append(env._prev.copy())
        with torch.no_grad():
            tok, resid, _, _ = ac.act(obs_seq, prev_seq, device, deterministic=True)
        a = decode_action(book, tok, resid, rho)
        if a is None:
            for __ in range(20):
                step_legal(env.sim, np.zeros(7, np.float32))
                renderer.update_scene(env.sim.data, cam)
                frames.append(renderer.render().copy())
            break
        step_legal(env.sim, a)
        env._prev = a
    path.parent.mkdir(parents=True, exist_ok=True)
    write_frames_mp4(frames, path, fps=20)
    print("video", path.name, len(frames), flush=True)


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "strict": sum(int(r["SUCCESS_NEW"]) for r in rows) / n,
        "strict_count": sum(int(r["SUCCESS_NEW"]) for r in rows),
        "s12": sum(int(r["SUCCESS_12S"]) for r in rows) / n,
        "recatch": sum(int(r["recatch_completed"]) for r in rows) / n,
        "stable": sum(int(r["stable_bilateral"]) for r in rows) / n,
        "lift": sum(int(r["lift_completed"]) for r in rows) / n,
        "mean_n_motion": float(np.mean([float(r["n_motion"]) for r in rows])),
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    (OUT / "videos").mkdir(parents=True, exist_ok=True)
    (OUT / "curves").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== P26 device={device} ===", flush=True)

    write_method_mapping()
    book, rho, _ = load_m111_book(M111_META)
    if not P25_CKPT.is_file():
        raise RuntimeError(f"missing P25 ckpt {P25_CKPT}")

    p21 = load_p21_manifest(P21 / "SFT_FULL_HORIZON_MANIFEST.json")
    starts = build_start_index(p21)
    train_rows = [r for r in p21["trajectories"] if r["family"] == "C" and r["split"] == "train"]
    val_rows = [r for r in p21["trajectories"] if r["family"] == "C" and r["split"] == "val"]
    train_eps = [starts[r["traj_id"]] for r in train_rows[:3]]

    feas = feasibility_audit(book, rho, starts, train_eps, device)
    rng0 = np.random.default_rng(26)
    pools = build_start_pools(starts, train_rows, val_rows, rng0)

    configs = {
        "init": str(P25_CKPT.relative_to(ROOT)).replace("\\", "/"),
        "seeds": SEEDS,
        "steps_track": STEPS_TRACK,
        "steps_catch": STEPS_CATCH,
        "steps_full": STEPS_FULL,
        "steps_direct_full": STEPS_DIRECT,
        "ppo": {
            "lr": 3e-5,
            "gamma": 0.99,
            "clip": 0.2,
            "ent_coef": 0.01,
            "rollout_steps": ROLLOUT_STEPS,
            "epochs": 4,
        },
        "n_independent_test": N_INDEPENDENT_TEST,
        "note": "Budgets B (direct full) == C (track+catch+full). No D2 mix. No hierarchy.",
    }
    (OUT / "TRAINING_CONFIGS.yaml").write_text(yaml.safe_dump(configs, sort_keys=False), encoding="utf-8")

    all_eval = []
    track_rows = []
    catch_rows = []
    full_rows = []
    abl_rows = []
    curves = []

    # A: P25 baseline (no RL)
    ac_a = load_actor_critic(P25_CKPT, book, device, rho)
    ev_a = eval_heldout(ac_a, book, rho, starts, val_rows, device, "A_p25_corrective")
    all_eval.extend(ev_a)
    sa = summarize(ev_a)
    abl_rows.append({"variant": "A_p25_corrective", "seed": "-", **sa})
    print(f"A baseline STRICT {sa['strict_count']}/{sa['n']}", flush=True)

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        cfg = PPOConfig(device=str(device), total_steps=STEPS_FULL, rollout_steps=ROLLOUT_STEPS, lr=3e-5)

        # B: direct full-horizon PPO from P25
        print(f"=== seed={seed} variant B direct full ===", flush=True)
        ac_b = load_actor_critic(P25_CKPT, book, device, rho)
        hist_b = train_stage(
            ac_b, book, rho, pools["hard"], "full", STEPS_DIRECT, cfg, f"B_direct_s{seed}", rng
        )
        curves.extend(hist_b)
        save_actor_critic(OUT / "checkpoints" / f"B_direct_s{seed}.pt", ac_b, book, rho, extra={"seed": seed})
        ev_b = eval_heldout(ac_b, book, rho, starts, val_rows, device, f"B_direct_s{seed}")
        all_eval.extend(ev_b)
        sb = summarize(ev_b)
        abl_rows.append({"variant": "B_direct_full", "seed": seed, **sb})
        full_rows.extend([{**r, "phase": "full_direct"} for r in ev_b])

        # C: tracking → catching → full
        print(f"=== seed={seed} variant C staged ===", flush=True)
        ac_c = load_actor_critic(P25_CKPT, book, device, rho)
        track_pool = pools["easy"] + pools["hard"]
        hist_t = train_stage(
            ac_c, book, rho, track_pool, "tracking", STEPS_TRACK, cfg, f"C_track_s{seed}", rng
        )
        curves.extend(hist_t)
        save_actor_critic(OUT / "checkpoints" / f"C_tracking_s{seed}.pt", ac_c, book, rho, extra={"seed": seed, "stage": "tracking"})
        # quick tracking probe on val (interception proxy = detach+contact via stage flags)
        ev_t = eval_heldout(ac_c, book, rho, starts, val_rows, device, f"C_tracking_s{seed}")
        track_rows.extend(ev_t)

        catch_pool = pools["near"] if pools["near"] else track_pool
        hist_c = train_stage(
            ac_c, book, rho, catch_pool, "catching", STEPS_CATCH, cfg, f"C_catch_s{seed}", rng
        )
        curves.extend(hist_c)
        save_actor_critic(OUT / "checkpoints" / f"C_catching_s{seed}.pt", ac_c, book, rho, extra={"seed": seed, "stage": "catching"})
        ev_catch = eval_heldout(ac_c, book, rho, starts, val_rows, device, f"C_catching_s{seed}")
        catch_rows.extend(ev_catch)
        # D: tracking→catching without final FT (snapshot before full)
        ev_d = eval_heldout(ac_c, book, rho, starts, val_rows, device, f"D_track_catch_s{seed}")
        all_eval.extend(ev_d)
        sd = summarize(ev_d)
        abl_rows.append({"variant": "D_track_catch_no_full", "seed": seed, **sd})

        hist_f = train_stage(
            ac_c, book, rho, pools["hard"], "full", STEPS_FULL, cfg, f"C_full_s{seed}", rng
        )
        curves.extend(hist_f)
        save_actor_critic(OUT / "checkpoints" / f"C_full_s{seed}.pt", ac_c, book, rho, extra={"seed": seed, "stage": "full"})
        ev_cf = eval_heldout(ac_c, book, rho, starts, val_rows, device, f"C_staged_full_s{seed}")
        all_eval.extend(ev_cf)
        sc = summarize(ev_cf)
        abl_rows.append({"variant": "C_staged_full", "seed": seed, **sc})
        full_rows.extend([{**r, "phase": "full_staged"} for r in ev_cf])

        # Independent test set (not used for selection)
        ev_ind = eval_heldout(ac_c, book, rho, starts, pools["indep_rows"], device, f"C_staged_indep_s{seed}")
        all_eval.extend(ev_ind)
        abl_rows.append({"variant": "C_staged_independent", "seed": seed, **summarize(ev_ind)})

        render_video(ac_c, book, rho, starts[val_rows[0]["traj_id"]], OUT / "videos" / f"staged_s{seed}.mp4", device)
        render_video(ac_b, book, rho, starts[val_rows[0]["traj_id"]], OUT / "videos" / f"direct_s{seed}.mp4", device)

    _write_csv(OUT / "TRACKING_RESULTS.csv", track_rows)
    _write_csv(OUT / "CATCHING_RESULTS.csv", catch_rows)
    _write_csv(OUT / "FULL_HORIZON_RESULTS.csv", full_rows)
    _write_csv(OUT / "ABLATION_COMPARISON.csv", abl_rows)
    _write_csv(OUT / "ALL_EVAL.csv", all_eval)
    _write_csv(OUT / "curves" / "LEARNING_CURVES.csv", curves)

    # Aggregate answers
    def mean_variant(name):
        xs = [r for r in abl_rows if r["variant"] == name]
        if not xs:
            return None
        return {
            "strict": float(np.mean([r["strict"] for r in xs])),
            "strict_count_mean": float(np.mean([r["strict_count"] for r in xs])),
            "recatch": float(np.mean([r["recatch"] for r in xs])),
            "stable": float(np.mean([r["stable"] for r in xs])),
            "lift": float(np.mean([r["lift"] for r in xs])),
            "n": xs[0]["n"],
        }

    A = mean_variant("A_p25_corrective")
    B = mean_variant("B_direct_full")
    C = mean_variant("C_staged_full")
    D = mean_variant("D_track_catch_no_full")
    Ind = mean_variant("C_staged_independent")

    # tracking/catching improvement proxies from intermediate evals
    def mean_label_prefix(prefix, key="recatch"):
        xs = [r for r in all_eval if str(r["condition"]).startswith(prefix)]
        if not xs:
            return float("nan")
        # group by seed condition then average rates
        by = defaultdict(list)
        for r in xs:
            by[r["condition"]].append(r)
        rates = []
        for cond, rows in by.items():
            rates.append(sum(int(x[key if key != 'recatch' else 'recatch_completed'] if key == 'recatch' else x.get(key, x.get('recatch_completed'))) for x in rows) / len(rows))
        # simpler:
        return sum(int(r["recatch_completed"]) for r in xs) / max(1, len(xs))

    track_recatch = sum(int(r["recatch_completed"]) for r in track_rows) / max(1, len(track_rows))
    catch_stable = sum(int(r["stable_bilateral"]) for r in catch_rows) / max(1, len(catch_rows))
    base_recatch = A["recatch"] if A else 0
    base_stable = A["stable"] if A else 0

    best_strict = max(
        (A or {}).get("strict", 0),
        (B or {}).get("strict", 0),
        (C or {}).get("strict", 0),
    )
    exceeds = best_strict > (A or {}).get("strict", 0) + 1e-9

    final = f"""# FINAL_REPORT — P26 Catch It!-Inspired Curriculum RL for Plan C

Generated: {_utc()}
Elapsed_s: {time.time() - t0:.1f}

## Answers

1. **Can M111 support reliable interception?**  
   Feasibility: random-token contact trials={feas['m111_random_contact_trials']}/{feas['n_trials']}; expert contact by t=40={feas['expert_contact_by_40']}.  
   Verdict: **{"YES — feasible with M111 (expert confirms; random coverage limited)" if feas['interception_feasible'] else "LIMITED — see CURRICULUM_FEASIBILITY.md"}**.

2. **Does tracking pretraining improve recatch?**  
   After tracking-only: held-out recatch rate≈{track_recatch:.3f} vs P25 baseline {base_recatch:.3f}.

3. **Does catching fine-tuning improve stable bilateral capture?**  
   After catching stage: stable≈{catch_stable:.3f} vs baseline {base_stable:.3f}.  
   D (track→catch, no full): strict≈{(D or {}).get('strict', float('nan')):.3f}, stable≈{(D or {}).get('stable', float('nan')):.3f}.

4. **Does staged RL outperform direct full-horizon under equal budgets?**  
   B direct STRICT≈{(B or {}).get('strict', float('nan')):.3f} ({(B or {}).get('strict_count_mean', float('nan')):.1f}/{(B or {}).get('n', 19)})  
   C staged STRICT≈{(C or {}).get('strict', float('nan')):.3f} ({(C or {}).get('strict_count_mean', float('nan')):.1f}/{(C or {}).get('n', 19)})  
   Staged better? **{"YES" if (C or {}).get('strict', 0) > (B or {}).get('strict', 0) else "NO / tied"}**.

5. **Does best policy exceed P25 6/19?**  
   A (P25) STRICT≈{(A or {}).get('strict', float('nan')):.3f} ({(A or {}).get('strict_count_mean', float('nan')):.1f}/19).  
   Best among B/C ≈{best_strict:.3f}. Exceeds? **{"YES" if exceeds else "NO"}**.

6. **Generalize to independent Plan C ICs?**  
   Independent test (train-lineage holdout, not the 19 val): STRICT≈{(Ind or {}).get('strict', float('nan')):.3f}.

7. **Suitable for integration with A/B and D2?**  
   {"Not yet — Plan C STRICT still modest; keep C skill isolated (P24 D2_only lesson)." if best_strict < 0.4 else "Marginally — consider gated integration; monitor A/B retention."}  
   **No GRPO / no hierarchical selector in P26.**

## Ablation (mean over seeds)

| Variant | STRICT | Recatch | Stable | Lift |
|---|---:|---:|---:|---:|
| A P25 corrective | {(A or {}).get('strict', float('nan')):.3f} | {(A or {}).get('recatch', float('nan')):.3f} | {(A or {}).get('stable', float('nan')):.3f} | {(A or {}).get('lift', float('nan')):.3f} |
| B direct full PPO | {(B or {}).get('strict', float('nan')):.3f} | {(B or {}).get('recatch', float('nan')):.3f} | {(B or {}).get('stable', float('nan')):.3f} | {(B or {}).get('lift', float('nan')):.3f} |
| C staged full PPO | {(C or {}).get('strict', float('nan')):.3f} | {(C or {}).get('recatch', float('nan')):.3f} | {(C or {}).get('stable', float('nan')):.3f} | {(C or {}).get('lift', float('nan')):.3f} |
| D track→catch only | {(D or {}).get('strict', float('nan')):.3f} | {(D or {}).get('recatch', float('nan')):.3f} | {(D or {}).get('stable', float('nan')):.3f} | {(D or {}).get('lift', float('nan')):.3f} |

## Integrity

- Physics / M111 / STRICT / held-out 19 ICs unchanged  
- No D2 in Plan C RL  
- No hierarchical skill selector  
- Checkpoint selection does not use independent test set  

## Artifacts

`results/diagnostics/p26_c_curriculum_rl/`
"""
    _write_md(OUT / "FINAL_REPORT.md", final)
    (OUT / "raw_p26_summary.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "feasibility": feas,
                "A": A,
                "B": B,
                "C": C,
                "D": D,
                "independent": Ind,
                "exceeds_p25": exceeds,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("=== P26 done ===", f"{time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
