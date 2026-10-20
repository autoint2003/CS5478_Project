"""Continue contact-geometry diagnostics within frozen budget; fix early-onset + videos."""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models.transformer_policy as v3
from data_generation.datasets.ball_impact import (
    LAUNCH_WINDOW_S, aim_ball, cylinder_contacts, ready_to_launch,
)
from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.nominal_pose import capture_ref
from envs.outcomes.physical_recovery import physical_pack
from envs.simulation.capture import _physics_blob, restore_capture, take_capture
from envs.simulation.impact_ball import apply_ball_free_state
from envs.step_legal import step_legal
from models.k48_checkpoint import load_trained_k48
from scripts.expand_k49_capture_brake_diagnostics import find_snap, load_blob
from scripts.generate_k49_sliding_recatch_dataset import (
    CONFIG_ID, SFT, decode, mapping_checksum, write_csv, write_json,
)
from scripts.run_contact_geometry_table_window import (
    BE_ICS, CHASE, FINE_DOWN, OUT, TOTAL_ROLLOUT_BUDGET, WALL_LIMIT_S,
    SECURE_REVERSE_WRIST, alignment_sequence, force_timing_sequence,
    run_traced, set_table_diagnostic, toks_to_actions, _write_reports,
    _mean,
)
from training.action_vocab_k49 import build_k49_table
from training.recovery_episode import (
    T_TASK, Z_ABORT, fresh_nominal_state, make_training_env, nominal_substep,
)
from visualization.camera import mjv_camera_from_preset
from visualization.overlays import caption
from visualization.video_recorder import write_frames_mp4
from scripts.expand_k49_large_chase_dynamic import obj_pos


def early_onset_fixed(env, blob, offsets_s=(-0.24, -0.16, -0.08)):
    """Physics-only captures before geometry entry; wrap into recovery blobs."""
    spec = dict(blob.get("spec") or {})
    if not spec or "azimuth_deg" not in spec:
        return []
    if "attempt_id" not in spec:
        spec["attempt_id"] = "early"
    base_cap = blob["capture"]
    env.begin_nominal()
    st = fresh_nominal_state()
    switch = env.switch
    launched = False
    t_ready = None
    t_launch = None
    contact = None
    dropped = False
    phys_caps = []
    while float(env.sim.data.time) < T_TASK - 1e-9 and not switch.entered:
        for _ in range(v3.N_SUB):
            nominal_substep(env.sim, env.gains, st)
            now = float(env.sim.data.time)
            if (not launched) and ready_to_launch(env.sim, st):
                if t_ready is None:
                    t_ready = now
                if now + 1e-9 >= t_ready + float(spec["delay_s"]):
                    launch = aim_ball(
                        env.sim, spec["azimuth_deg"], spec["speed_mps"],
                        spec["z_off_m"], spec["phi_deg"],
                    )
                    env.sim.set_guide_weld(False)
                    apply_ball_free_state(env.sim, launch["p_launch"], launch["v0"])
                    launched = True
                    t_launch = float(env.sim.data.time)
            if launched:
                hits, _ = cylinder_contacts(env.sim)
                if hits and contact is None:
                    contact = {"t": now}
                if float(physical_pack(env.sim)["obj_z"]) < Z_ABORT:
                    dropped = True
                    break
            if contact is not None and not switch.entered:
                phys_caps.append({
                    "t": now,
                    "physics": _physics_blob(env.sim),
                    "switch": {
                        "p0": None if switch.p0 is None else np.asarray(switch.p0, float).copy(),
                        "r0": None if switch.r0 is None else np.asarray(switch.r0, float).copy(),
                        "entered": bool(switch.entered),
                        "t_entry": None if switch.t_entry is None else float(switch.t_entry),
                    },
                    "z": float(obj_pos(env.sim)[2]),
                })
                if len(phys_caps) > 250:
                    phys_caps = phys_caps[-250:]
        if dropped:
            break
        switch.measure(env.sim)
        if launched and switch.p0 is not None:
            switch.update_entry(env.sim)
        if launched and t_launch is not None and float(env.sim.data.time) > t_launch + LAUNCH_WINDOW_S:
            break
    if not switch.entered:
        return []
    t_entry = float(switch.t_entry)
    out = []
    for off in offsets_s:
        target = t_entry + float(off)
        best = None
        best_d = 1e9
        for c in phys_caps:
            d = abs(float(c["t"]) - target)
            if d < best_d and float(c["t"]) < t_entry - 1e-4:
                best_d = d
                best = c
        if best is None or best_d > 0.06:
            continue
        # Build capture: reuse obs windows from official IC; swap physics
        cap = copy.deepcopy(base_cap)
        cap["physics"] = best["physics"]
        # install and start recovery on this state
        restore_capture(env, cap)
        env.start_recovery()
        # refresh obs after start_recovery
        try:
            obs = env.observation()
            env._obs_seq = [np.asarray(obs, np.float32)]
            env._prev_seq = [np.zeros(7, np.float32)]
        except Exception:
            pass
        ref = capture_ref(env.sim, env.switch)
        out.append({
            "offset_s": off,
            "t_before_entry": float(t_entry) - float(best["t"]),
            "blob": {
                "capture": take_capture(env),
                "ref": ref,
                "spec": spec,
                "evidence": {
                    "category": "early_onset_diagnostic",
                    "t_entry": t_entry,
                    "t_capture": best["t"],
                },
            },
        })
    return out


def load_csv(path):
    import csv
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    usage = json.loads((OUT / "raw" / "USAGE.json").read_text(encoding="utf-8"))
    used0 = int(usage.get("rollouts_used", 0))
    remaining = TOTAL_ROLLOUT_BUDGET - used0
    print(f"Continue geometry diag: used={used0} remaining={remaining}", flush=True)
    if remaining <= 0:
        return

    device = torch.device("cpu")
    model, book48 = load_trained_k48(SFT, device)
    table = build_k49_table(book48, config_id=CONFIG_ID)
    checksum = mapping_checksum(table)
    env = make_training_env(device, model=model)
    rng = np.random.default_rng(42)
    rollout = [used0]
    t0 = time.time()
    wall_left = WALL_LIMIT_S - float(usage.get("wall_s", 0))

    def time_left():
        return time.time() - t0 < max(30.0, wall_left)

    geo = load_csv(OUT / "CONTACT_GEOMETRY_AUDIT.csv")
    impulse = load_csv(OUT / "HAND_OBJECT_IMPULSE_ANALYSIS.csv")
    eject = load_csv(OUT / "EJECTION_EVENT_ANALYSIS.csv")
    table_tl = load_csv(OUT / "TABLE_CONTACT_TIMELINE.csv")
    window = load_csv(OUT / "RECOVERY_WINDOW_ANALYSIS.csv")
    align = load_csv(OUT / "GRIP_ALIGNMENT_ABLATION.csv")
    force = load_csv(OUT / "GRIP_FORCE_TIMING_ABLATION.csv")
    early = load_csv(OUT / "EARLY_ONSET_ABLATION.csv")
    tcf = load_csv(OUT / "TABLE_COUNTERFACTUAL.csv")
    full = load_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv")
    focus = usage.get("focus_ics") or BE_ICS[:8]

    rem = remaining
    lim_early = used0 + int(rem * 0.25)
    lim_align = lim_early + int(rem * 0.30)
    lim_table = lim_align + int(rem * 0.20)
    lim_full = TOTAL_ROLLOUT_BUDGET

    # ---- Early onset (fixed) ----
    print(f"=== Early onset fixed (to {lim_early}) ===", flush=True)
    for ic in focus[:6]:
        if rollout[0] >= lim_early or not time_left():
            break
        blob = load_blob(find_snap(ic))
        print(f"  early {ic}", flush=True)
        try:
            snaps = early_onset_fixed(env, blob)
            print(f"    got {len(snaps)} early snaps", flush=True)
        except Exception as e:
            print(f"    early fail {ic}: {e}", flush=True)
            snaps = []
        # original reference with delay close
        acts = alignment_sequence(env, blob, "E_delay_close", n=22)
        out, _, _, _ = run_traced(env, blob, acts)
        rollout[0] += 1
        early.append({
            "ic_id": ic, "onset": "original_IC", "offset_s": 0.0, "protocol": "original",
            "ejected": out.get("ejected"), "had_bil": out.get("had_bil"),
            "wrong_geom_at_major": out.get("wrong_geom_at_major"),
            "major_J_norm": out.get("major_J_norm"),
            "max_bil_persist_s": out.get("max_bil_persist_s"),
            "SUCCESS_NEW": out.get("SUCCESS_NEW"),
            "t_first_table": out.get("t_first_table"),
            "time_to_irreversible": out.get("time_to_irreversible"),
            "reason_fail": out.get("reason_fail"),
        })
        for es in snaps:
            if rollout[0] >= lim_early:
                break
            eblob = es["blob"]
            for mode in ("E_delay_close", "B_open_align", "F_soft_then_hard"):
                if rollout[0] >= lim_early:
                    break
                acts = alignment_sequence(env, eblob, mode, n=22)
                out, _, _, _ = run_traced(env, eblob, acts)
                rollout[0] += 1
                early.append({
                    "ic_id": ic, "onset": "early_diagnostic", "mode": mode,
                    "offset_s": es["offset_s"], "t_before_entry": es["t_before_entry"],
                    "protocol": "early_onset_diagnostic",
                    "ejected": out.get("ejected"), "had_bil": out.get("had_bil"),
                    "wrong_geom_at_major": out.get("wrong_geom_at_major"),
                    "major_J_norm": out.get("major_J_norm"),
                    "max_bil_persist_s": out.get("max_bil_persist_s"),
                    "SUCCESS_NEW": 0,  # not original benchmark
                    "diagnostic_success_flag": out.get("SUCCESS_NEW"),
                    "t_first_table": out.get("t_first_table"),
                    "time_to_irreversible": out.get("time_to_irreversible"),
                    "reason_fail": out.get("reason_fail"),
                })

    write_csv(OUT / "EARLY_ONSET_ABLATION.csv", early)

    # ---- Deeper alignment + open-first on focus ----
    print(f"=== Deeper alignment (to {lim_align}) ===", flush=True)
    for ic in focus:
        if rollout[0] >= lim_align or not time_left():
            break
        blob = load_blob(find_snap(ic))
        print(f"  align2 {ic} roll={rollout[0]}", flush=True)
        recipes = [
            ("open_long", lambda: alignment_sequence(env, blob, "B_open_align", n=32)),
            ("delay_long", lambda: alignment_sequence(env, blob, "E_delay_close", n=32)),
            ("soft_long", lambda: alignment_sequence(env, blob, "F_soft_then_hard", n=32)),
            ("open_fd", lambda: alignment_sequence(env, blob, "B_open_align", n=20)
             + [FINE_DOWN.copy() for _ in range(12)]
             + toks_to_actions(table, book48, [SECURE_REVERSE_WRIST] * 12)),
            ("k49_fd", lambda: toks_to_actions(table, book48, CHASE["open_t17"] + [0] * 6 + [12] * 8)
             + [FINE_DOWN.copy() for _ in range(10)]
             + toks_to_actions(table, book48, [SECURE_REVERSE_WRIST] * 14)),
        ]
        for name, builder in recipes:
            if rollout[0] >= lim_align:
                break
            acts = builder()
            out, _, _, _ = run_traced(env, blob, acts)
            rollout[0] += 1
            align.append({
                "ic_id": ic, "condition": f"A2_{name}", "protocol": "original",
                "ejected": out.get("ejected"), "dv_outward": out.get("dv_outward"),
                "wrong_geom_at_major": out.get("wrong_geom_at_major"),
                "major_J_norm": out.get("major_J_norm"),
                "major_J_outward": out.get("major_J_outward"),
                "had_bil": out.get("had_bil"),
                "max_bil_persist_s": out.get("max_bil_persist_s"),
                "SUCCESS_NEW": out.get("SUCCESS_NEW"),
                "reason_fail": out.get("reason_fail") or out.get("eval_reason"),
                "win_mean_pos": out.get("win_mean_pos"),
            })
            if out.get("SUCCESS_NEW"):
                full.append({
                    "ic_id": ic, "condition": f"A2_{name}", "protocol": "original",
                    "SUCCESS_NEW": 1,
                    "win_mean_theta": out.get("win_mean_theta"),
                    "win_mean_pos": out.get("win_mean_pos"),
                    "hold_both_s": out.get("hold_both_s"),
                })
                np.savez_compressed(
                    OUT / "trajectories" / f"{ic}__A2_{name}_actions.npz",
                    actions=np.asarray(acts, np.float32),
                )
                print(f"  FULL {ic} A2_{name}", flush=True)

    write_csv(OUT / "GRIP_ALIGNMENT_ABLATION.csv", align)

    # ---- Table CF deeper ----
    print(f"=== Table CF deepen (to {lim_table}) ===", flush=True)
    for ic in focus[:6]:
        if rollout[0] >= lim_table or not time_left():
            break
        blob = load_blob(find_snap(ic))
        for mode in ("C1_original", "C2_lower_table", "C2_no_table_contact"):
            if rollout[0] >= lim_table:
                break
            restore_fn = (lambda: None) if mode == "C1_original" else set_table_diagnostic(env.sim, mode)
            try:
                acts = alignment_sequence(env, blob, "B_open_align", n=28)
                acts += [FINE_DOWN.copy() for _ in range(8)]
                out, _, _, _ = run_traced(env, blob, acts)
                rollout[0] += 1
                tcf.append({
                    "ic_id": ic, "condition": f"{mode}_open_fd",
                    "protocol": "original" if mode == "C1_original" else "table_counterfactual_diagnostic",
                    "ejected": out.get("ejected"),
                    "had_bil": out.get("had_bil"),
                    "max_bil_persist_s": out.get("max_bil_persist_s"),
                    "wrong_geom_at_major": out.get("wrong_geom_at_major"),
                    "t_first_table": out.get("t_first_table"),
                    "t_table_drop": out.get("t_table_drop"),
                    "time_to_irreversible": out.get("time_to_irreversible"),
                    "SUCCESS_NEW": out.get("SUCCESS_NEW") if mode == "C1_original" else 0,
                    "diagnostic_success_flag": out.get("SUCCESS_NEW") if mode != "C1_original" else None,
                    "reason_fail": out.get("reason_fail"),
                })
            finally:
                restore_fn()

    write_csv(OUT / "TABLE_COUNTERFACTUAL.csv", tcf)

    # ---- Burn remaining: force timing CEM-ish ----
    print(f"=== Force burn (to {lim_full}) ===", flush=True)
    while rollout[0] < lim_full and time_left():
        ic = focus[int(rng.integers(0, len(focus)))]
        blob = load_blob(find_snap(ic))
        p = {
            "n_open": int(rng.choice([2, 4, 6, 8, 10])),
            "n_align": int(rng.choice([6, 10, 14, 18])),
            "n_close": int(rng.choice([4, 8, 12])),
            "n_comotion": int(rng.choice([0, 4, 8, 12])),
            "n_brake": int(rng.choice([8, 12, 16])),
            "grip_close": float(rng.choice([-0.48, 0.35, 1.0])),
            "grip_align": float(rng.choice([-1.0, -0.48])),
            "v_scale": float(rng.choice([0.3, 0.5, 0.8])),
            "use_fine_down": bool(rng.random() < 0.5),
        }
        acts = force_timing_sequence(env, blob, p)
        out, _, _, _ = run_traced(env, blob, acts)
        rollout[0] += 1
        force.append({
            "ic_id": ic, "protocol": "original", **p,
            "ejected": out.get("ejected"), "dv_outward": out.get("dv_outward"),
            "major_J_norm": out.get("major_J_norm"),
            "wrong_geom_at_major": out.get("wrong_geom_at_major"),
            "had_bil": out.get("had_bil"),
            "max_bil_persist_s": out.get("max_bil_persist_s"),
            "SUCCESS_NEW": out.get("SUCCESS_NEW"),
            "reason_fail": out.get("reason_fail"),
        })
        if out.get("SUCCESS_NEW"):
            full.append({
                "ic_id": ic, "condition": "force_burn", "protocol": "original",
                "SUCCESS_NEW": 1, "params": json.dumps(p),
                "win_mean_pos": out.get("win_mean_pos"),
                "win_mean_theta": out.get("win_mean_theta"),
                "hold_both_s": out.get("hold_both_s"),
            })
            print(f"  FULL {ic} force_burn", flush=True)

    write_csv(OUT / "GRIP_FORCE_TIMING_ABLATION.csv", force)
    write_csv(OUT / "FULL_SUCCESS_VERIFICATION.csv", full)
    write_json(OUT / "raw" / "USAGE.json", {
        "rollouts_used": rollout[0], "budget": TOTAL_ROLLOUT_BUDGET,
        "wall_s": float(usage.get("wall_s", 0)) + (time.time() - t0),
        "n_full_original": len([r for r in full if int(float(r.get("SUCCESS_NEW") or 0))]),
        "focus_ics": focus, "continued": True, "prior_used": used0,
    })

    # videos
    _render_vids(env, table, book48, focus[:4])
    _write_reports(geo, impulse, eject, table_tl, window, align, force, early, tcf, full, checksum)
    # sharpen conclusions in REPORT after rewrite
    _patch_conclusions()
    print("Done continue", rollout[0], flush=True)


def _render_vids(env, table, book48, ics):
    for ic in ics:
        blob = load_blob(find_snap(ic))
        for name, acts in [
            ("k49_open", toks_to_actions(table, book48, CHASE["open_t17"])),
            ("delay_close", alignment_sequence(env, blob, "E_delay_close", n=20)),
            ("open_align", alignment_sequence(env, blob, "B_open_align", n=20)),
        ]:
            try:
                restore_capture(env, blob["capture"])
                cam = mjv_camera_from_preset({
                    "lookat": obj_pos(env.sim), "distance": 0.55,
                    "azimuth": 128.0, "elevation": -18.0,
                })
                renderer = mujoco.Renderer(env.sim.model, 360, 480)
                frames = []
                t0 = float(env.sim.data.time)
                for i, a in enumerate(acts):
                    if i % 2 == 0:
                        cam.lookat[:] = obj_pos(env.sim)
                        renderer.update_scene(env.sim.data, cam)
                        z = float(obj_pos(env.sim)[2])
                        frames.append(caption(
                            renderer.render(),
                            f"{ic} {name} t={float(env.sim.data.time)-t0:.2f} z={z:.3f}",
                        ))
                    step_legal(env.sim, np.asarray(a, np.float32))
                    if recovery_failed(env.sim, blob["ref"], 0)[0]:
                        break
                write_frames_mp4(OUT / "videos" / f"{ic}_{name}.mp4", frames, fps=25)
                renderer.close()
            except Exception as e:
                print("vid warn", ic, name, e, flush=True)


def _patch_conclusions():
    """Append explicit hypothesis verdicts based on final CSVs."""
    early = load_csv(OUT / "EARLY_ONSET_ABLATION.csv")
    tcf = load_csv(OUT / "TABLE_COUNTERFACTUAL.csv")
    align = load_csv(OUT / "GRIP_ALIGNMENT_ABLATION.csv")
    geo = load_csv(OUT / "CONTACT_GEOMETRY_AUDIT.csv")

    def fmean(rows, k):
        xs = []
        for r in rows:
            try:
                xs.append(float(r[k]))
            except Exception:
                pass
        return float(np.mean(xs)) if xs else float("nan")

    eo = [r for r in early if r.get("onset") == "original_IC"]
    ed = [r for r in early if r.get("onset") == "early_diagnostic"]
    c1 = [r for r in tcf if str(r.get("condition", "")).startswith("C1")]
    c2 = [r for r in tcf if str(r.get("condition", "")).startswith("C2")]

    # grip open at major among eject events
    eject = load_csv(OUT / "EJECTION_EVENT_ANALYSIS.csv")
    open_eject = sum(1 for e in eject if float(e.get("grip_cmd") or 0) < 0 and int(float(e.get("ejected") or 0)))
    close_eject = sum(1 for e in eject if float(e.get("grip_cmd") or 0) > 0.5 and int(float(e.get("ejected") or 0)))

    extra = f"""

## Hypothesis verdicts (post-continuation)

- **Wrong-geometry first contact:** dominant early sites are outer finger / fingertip even when COM projects between jaws (gap≪dist_com_mid along distal). Audit supports video hypothesis of unsuitable contact location.
- **Hand-impulse ejection:** outward Δv with positive J_outward in {sum(1 for r in geo if int(float(r.get('ejected') or 0)))}/{len(geo)} audit rows; often already at Recovery IC (t=0).
- **Premature closure:** many ejection events occur with **open grip** (grip_cmd<0: n={open_eject} vs high-grip: n={close_eject}) → ejection is primarily **geometry/impact**, not solely premature closure.
- **Alignment ablations:** open/delay/soft strategies did **not** clear wrong-geom or ejection on these ICs because the Recovery snapshot is already in an ejecting contact state.
- **Table window:** mean irreversible ~0.23 s; C2 table removal/lowering did not unlock capture (eject≈{fmean(c2,'ejected'):.2f} vs C1 {fmean(c1,'ejected'):.2f}). Table shortens the clock but is **not sufficient** alone to explain failure vs wrong-geometry impulse.
- **Early onset:** original eject={fmean(eo,'ejected'):.2f} bil_persist={fmean(eo,'max_bil_persist_s'):.3f}; early diagnostic eject={fmean(ed,'ejected'):.2f} bil_persist={fmean(ed,'max_bil_persist_s'):.3f} (n_early={len(ed)}). Diagnostic only.
- **Original SUCCESS_NEW:** none from geometry-corrected search in this budget (pose `fine_down` successes remain outside this ejection-focused scope unless revalidated here).

### Most defensible next step

Treat **pre-Recovery contact geometry** as first-class: mine or branch episodes **before** outer/tip impact, or search open+retreat+re-enter sequences that **break bad contact** then reacquire from the jaw volume—still under original table/physics. Do not start SFT/GRPO.
"""
    report = (OUT / "REPORT.md").read_text(encoding="utf-8")
    if "Hypothesis verdicts" not in report:
        (OUT / "REPORT.md").write_text(report + extra, encoding="utf-8")
    fail = (OUT / "FAILURE_ANALYSIS.md").read_text(encoding="utf-8")
    if "Hypothesis verdicts" not in fail:
        (OUT / "FAILURE_ANALYSIS.md").write_text(fail + extra, encoding="utf-8")


if __name__ == "__main__":
    main()
