"""Assemble EPOCH_COMPARISON / SAMPLE_EFFICIENCY / FINAL_REPORT."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.precision.p64_ppo_multiround.paths import OUT, P63_A_CUMULATIVE, TRAJ_PER_STEP


def _load_baselines():
    p = OUT / "BASELINES.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _load_steps():
    rows = []
    for p in sorted(OUT.glob("P64_step*.json")):
        if p.name.endswith("_optim.pt"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append(d)
    rows.sort(key=lambda r: r.get("step", 0))
    return rows


def _msp_row(name: str, e: dict, extra: str = "") -> str:
    bc = e.get("by_class") or {}

    def g(c):
        return float((bc.get(c) or {}).get("MSP_RELAXED") or float("nan"))

    return (
        f"| {name} | {float(e.get('MSP_RELAXED', float('nan'))):.4f} | "
        f"{g('A'):.4f} | {g('B'):.4f} | {g('C'):.4f} | {g('D'):.4f} | "
        f"{e.get('zero_success_ics', '')} | {extra} |"
    )


def write_all_reports(*, best: dict | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bases = _load_baselines()
    steps = _load_steps()
    p62 = bases.get("P62_epoch3") or {}
    p63a = bases.get("P63_PPO_A") or {}
    if best is None and (OUT / "BEST_CHECKPOINT.json").is_file():
        best = json.loads((OUT / "BEST_CHECKPOINT.json").read_text(encoding="utf-8"))
    best = best or {"tag": "P63_PPO_A", "msp": p63a.get("MSP_RELAXED"), "step": 0}

    # EPOCH_COMPARISON
    lines = [
        "# EPOCH_COMPARISON",
        "",
        "Development set = frozen P61 SHA-256 protocol (not final-test).",
        "",
        "| Checkpoint | MSP | A | B | C | D | Zero | Notes |",
        "|------------|----:|--:|--:|--:|--:|-----:|-------|",
    ]
    if p62:
        lines.append(_msp_row("P62_epoch3", p62, "GRPO baseline"))
    if p63a:
        lines.append(_msp_row("P63 PPO-A", p63a, "cum=2048"))
    for s in steps:
        e = s.get("eval") or {}
        note = f"cum={s.get('cumulative_traj_after_p63a')}; ΔvsP63A={((s.get('paired_vs_p63a') or {}).get('mean_delta'))}"
        lines.append(_msp_row(s.get("tag", f"step{s.get('step')}"), e, note))
    lines += ["", f"**Selected best (dev):** `{best.get('tag')}` MSP={best.get('msp')}", ""]
    (OUT / "EPOCH_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")

    # PER_CLASS_DIAGNOSTICS
    dlines = [
        "# PER_CLASS_DIAGNOSTICS",
        "",
        "## Process rates (dev)",
        "",
        "| Checkpoint | p_recatch | p_stable | p_handoff | zeros |",
        "|------------|----------:|---------:|----------:|------:|",
    ]
    for name, e in (("P62_epoch3", p62), ("P63_PPO_A", p63a)):
        if e:
            dlines.append(
                f"| {name} | {e.get('p_recatch', float('nan'))} | {e.get('p_stable_recatch', float('nan'))} | "
                f"{e.get('p_handoff', float('nan'))} | {e.get('zero_success_ics')} |"
            )
    for s in steps:
        e = s.get("eval") or {}
        dlines.append(
            f"| {s.get('tag')} | {e.get('p_recatch', float('nan'))} | {e.get('p_stable_recatch', float('nan'))} | "
            f"{e.get('p_handoff', float('nan'))} | {e.get('zero_success_ics')} |"
        )
    dlines += ["", "## Critic / PPO health by update", ""]
    for s in steps:
        dlines.append(f"### {s.get('tag')}")
        dlines.append("")
        dlines.append("| upd | EV | KL | clip | ent | pg | vf | p_rel_rollout |")
        dlines.append("|----:|---:|---:|-----:|----:|---:|---:|-------------:|")
        for h in s.get("history") or []:
            # history may be absent in slim TRAINING_LOGS; load from step file
            pass
        # Prefer update_rows from TRAINING_LOGS if history stripped
        log = json.loads((OUT / "TRAINING_LOGS.json").read_text(encoding="utf-8")) if (OUT / "TRAINING_LOGS.json").is_file() else {"steps": []}
        match = next((x for x in log["steps"] if x.get("step") == s.get("step")), None)
        rows = (match or {}).get("update_rows") or []
        if not rows and s.get("history"):
            rows = [
                {"update": h["update"], "rollout": h["rollout"], "ppo": h["ppo"]}
                for h in s["history"]
            ]
        for r in rows:
            ro = r.get("rollout") or {}
            ppo = r.get("ppo") or {}
            dlines.append(
                f"| {r.get('update')} | {ro.get('explained_var')} | {ppo.get('approx_kl')} | "
                f"{ppo.get('clipfrac')} | {ppo.get('ent')} | {ppo.get('pg')} | {ppo.get('vf')} | "
                f"{ro.get('overall_p_relaxed')} |"
            )
        dlines.append("")
    (OUT / "PER_CLASS_DIAGNOSTICS.md").write_text("\n".join(dlines), encoding="utf-8")

    # SAMPLE_EFFICIENCY
    total_traj = P63_A_CUMULATIVE + TRAJ_PER_STEP * len(steps)
    total_steps_sim = int(sum(s.get("total_steps") or 0 for s in steps))
    # include rough P63-A steps from its json if present
    p63_steps = 110140  # from P63 PILOT_A
    p63_path = OUT.parent / "p63_transformer_ppo_c_grasp" / "PILOT_A.json"
    if p63_path.is_file():
        p63_steps = int(json.loads(p63_path.read_text(encoding="utf-8")).get("total_steps") or p63_steps)
    wall = float(sum(s.get("wall_s") or 0 for s in steps))
    p63_wall = 4484.8
    if p63_path.is_file():
        p63_wall = float(json.loads(p63_path.read_text(encoding="utf-8")).get("wall_s") or p63_wall)
    total_wall = wall + p63_wall
    msp0 = float(p62.get("MSP_RELAXED") or 0.166)
    msp_best = float(best.get("msp") or msp0)
    gain = msp_best - msp0
    # traj to best
    if best.get("step", 0) == 0:
        traj_to_best = P63_A_CUMULATIVE
        wall_to_best = p63_wall
        steps_to_best = p63_steps
    else:
        traj_to_best = P63_A_CUMULATIVE + TRAJ_PER_STEP * int(best["step"])
        wall_to_best = p63_wall + float(
            sum(s.get("wall_s") or 0 for s in steps if s.get("step", 99) <= best["step"])
        )
        steps_to_best = p63_steps + int(
            sum(s.get("total_steps") or 0 for s in steps if s.get("step", 99) <= best["step"])
        )
    gpu_h = wall_to_best / 3600.0
    eff = {
        "msp_p62": msp0,
        "msp_best": msp_best,
        "gain_from_p62": gain,
        "traj_to_best": traj_to_best,
        "sim_steps_to_best": steps_to_best,
        "wall_s_to_best": wall_to_best,
        "gpu_hours_to_best": gpu_h,
        "msp_gain_per_1k_traj": gain / (traj_to_best / 1000.0) if traj_to_best else None,
        "msp_gain_per_gpu_hour": gain / gpu_h if gpu_h > 0 else None,
        "total_traj_ran": total_traj,
        "total_sim_steps_p64_only": total_steps_sim,
        "note": (
            "Not a matched-budget randomized comparison vs full GRPO multi-epoch training; "
            "GRPO-64 uses 64 traj/IC/collection while PPO here uses 8."
        ),
    }
    (OUT / "SAMPLE_EFFICIENCY.json").write_text(json.dumps(eff, indent=2), encoding="utf-8")
    (OUT / "SAMPLE_EFFICIENCY.md").write_text(
        "\n".join(
            [
                "# SAMPLE_EFFICIENCY",
                "",
                f"- Best checkpoint: **{best.get('tag')}** MSP={msp_best:.4f}",
                f"- Gain vs P62_epoch3: **{gain:.4f}**",
                f"- Trajectories to best (incl. P63-A): **{traj_to_best}**",
                f"- Simulator steps to best (approx): **{steps_to_best}**",
                f"- GPU-hours to best (approx): **{gpu_h:.3f}**",
                f"- MSP gain / 1k traj: **{eff['msp_gain_per_1k_traj']}**",
                f"- MSP gain / GPU-h: **{eff['msp_gain_per_gpu_hour']}**",
                "",
                eff["note"],
                "",
            ]
        ),
        encoding="utf-8",
    )

    # FINAL_REPORT
    sat = "unknown"
    if len(steps) >= 2:
        msps = [float(p63a.get("MSP_RELAXED") or 0)] + [float((s.get("eval") or {}).get("MSP_RELAXED") or 0) for s in steps]
        # saturation = first step where later gains < 0.005
        sat = "not_reached"
        for i in range(1, len(msps)):
            if max(msps[i:]) <= msps[i - 1] + 0.005 and i >= 1:
                # check if peak already behind
                peak_i = int(np.argmax(msps))
                if peak_i <= i:
                    sat = f"around_{['P63-A','P64_step1','P64_step2','P64_step3','P64_step4'][peak_i] if peak_i < 5 else peak_i}"
                    break

    c_msps = []
    if p63a:
        c_msps.append(float((p63a.get("by_class") or {}).get("C", {}).get("MSP_RELAXED") or 0))
    for s in steps:
        c_msps.append(float(((s.get("eval") or {}).get("by_class") or {}).get("C", {}).get("MSP_RELAXED") or 0))

    zeros = []
    if p63a:
        zeros.append(int(p63a.get("zero_success_ics") or 0))
    for s in steps:
        zeros.append(int((s.get("eval") or {}).get("zero_success_ics") or 0))

    final = f"""# FINAL_REPORT — P64 Full-History Transformer PPO Multi-Round

Continuation from **P63 PPO-A** only. No B/C, no C reward, no new C ICs.
Reward / architecture / M111 / R3F / SUCCESS_RELAXED unchanged.

## Integrity

See `P63_INTEGRITY_CHECK.md`. Branches were independent; matched 2048-traj budgets;
PPO-A optimizer state was **not** in the P63 checkpoint (fresh Adam at start of P64_step1;
optimizer saved thereafter).

## Results

See `EPOCH_COMPARISON.md`, `PER_CLASS_DIAGNOSTICS.md`, `SAMPLE_EFFICIENCY.md`, `TRAINING_LOGS.json`.

Best (dev): **{best.get('tag')}** MSP={best.get('msp')}

## Final questions

1. **Continue improving beyond 2048?** {"Yes" if steps and float((steps[-1].get("eval") or {}).get("MSP_RELAXED") or 0) > float(p63a.get("MSP_RELAXED") or 0) + 0.005 else ("Mixed/inconclusive" if steps else "Pending — no P64 step yet")}: latest steps MSP={[float((s.get("eval") or {}).get("MSP_RELAXED") or 0) for s in steps]}; P63-A={p63a.get("MSP_RELAXED")}.
2. **Saturation?** {sat}
3. **Critic EV healthy?** See per-update EV in `PER_CLASS_DIAGNOSTICS.md` (warn if ≪0.1).
4. **C improves naturally?** C MSP series={c_msps}
5. **A/B/D retained?** Compare class columns in `EPOCH_COMPARISON.md`.
6. **Zero-success coverage?** zeros series={zeros}
7. **Strongest checkpoint?** `{best.get("tag")}` at {best.get("ckpt")}
8. **Clearly more efficient than GRPO?** Suggestive at pilot scale ({eff.get("msp_gain_per_1k_traj")} MSP/1k traj; {eff.get("msp_gain_per_gpu_hour")} /GPU-h) but **not** a matched randomized bake-off vs full GRPO-64 multi-epoch.

No reward/data/architecture changes in P64.
"""
    (OUT / "FINAL_REPORT.md").write_text(final, encoding="utf-8")


if __name__ == "__main__":
    write_all_reports()
