    # ========== Stage B: distal correction from snapshots ==========
    print("Stage B: post-capture correction search", flush=True)
    dic, dacts, dcut = roots["distal_off01_r8"]
    snaps = extract_snapshots(env, dic, dacts, dcut)
    write_json(OUT / "raw" / "distal_snapshots_meta.json", {
        k: {"step": v["step"], "pose": v["pose"], "prefix_len": v["prefix_len"]}
        for k, v in snaps.items()
    })
    modes = ["none", "wrist", "pivot", "slide", "combined", "early"]
    best_corr = None
    n_trials = int(budgets["stage_b_trials_per_snap"])
    for snap_name, snap in snaps.items():
        if not time_left():
            break
        print(f"  snapshot {snap_name} theta0={snap['pose']['e_theta_deg']:.1f}", flush=True)
        for mode in modes:
            for ti in range(max(1, n_trials // len(modes))):
                if not time_left():
                    break
                gains = {
                    "kp": float(rng.choice([0.8, 1.2, 1.6])),
                    "kz": float(rng.choice([0.2, 0.4, 0.6])),
                    "kw": float(rng.choice([0.35, 0.55, 0.75, 0.9])),
                    "wy": float(rng.choice([0.0, 0.2, 0.35, 0.45])),
                }
                n_steps = int(rng.choice([20, 30, 40, 50]))
                if mode == "early" and snap_name != "after_bilateral":
                    continue
                suf, pose_f = run_correction_from_capture(
                    env, dic, snap, mode if mode != "early" else "combined", n_steps, gains, "privileged",
                )
                row = {
                    "snap": snap_name,
                    "mode": mode,
                    "method_family": "post_capture_suffix",
                    "theta0": snap["pose"]["e_theta_deg"],
                    "pos0": snap["pose"]["e_pos_mm"],
                    "theta1": pose_f["e_theta_deg"],
                    "pos1": pose_f["e_pos_mm"],
                    "d_theta": snap["pose"]["e_theta_deg"] - pose_f["e_theta_deg"],
                    "d_pos": snap["pose"]["e_pos_mm"] - pose_f["e_pos_mm"],
                    "bilateral_end": pose_f["bilateral"],
                    "n_suffix": len(suf),
                    "gains": json.dumps(gains),
                    "below_10": int(pose_f["e_theta_deg"] <= WORKING_THETA_DEG),
                    "pos_ok": int(pose_f["e_pos_mm"] <= WORKING_POS_MM),
                }
                corr_rows.append(row)
                sc = (
                    100 * int(row["below_10"] and row["pos_ok"] and row["bilateral_end"])
                    + row["d_theta"] - 0.05 * abs(row["d_pos"])
                )
                if best_corr is None or sc > best_corr["score"]:
                    best_corr = {
                        "score": sc, "snap": snap_name, "mode": mode, "suffix": suf,
                        "prefix_len": snap["prefix_len"], "pose_f": pose_f, "gains": gains,
                        "row": row, "full": None, "cut": None,
                    }
        print(
            f"    best post-capture snap={best_corr['snap'] if best_corr else None} "
            f"th={best_corr['pose_f']['e_theta_deg'] if best_corr else None}",
            flush=True,
        )

    # Stage B2: during-capture early wrist
    print("Stage B2: during-capture early-wrist search", flush=True)
    wrist_grid = [
        (0.0, 0.14, 0.20), (0.0, 0.10, 0.20), (0.0, 0.18, 0.15), (0.0, 0.20, 0.20),
        (-0.1, 0.14, 0.20), (0.05, 0.14, 0.20), (0.0, 0.12, 0.18), (0.0, 0.14, 0.25),
        (0.0, 0.08, 0.22), (-0.05, 0.16, 0.18),
    ]
    if args.smoke:
        wrist_grid = wrist_grid[:4]
    best_early = None
    theta0_ref = float(snaps.get("after_bilateral", {}).get("pose", {}).get("e_theta_deg", 21.65))
    pos0_ref = float(snaps.get("after_bilateral", {}).get("pose", {}).get("e_pos_mm", 8.12))
    for wx, wy, wz in wrist_grid:
        if not time_left():
            break
        for until in (dcut + 1, dcut + 2, dcut + 3):
            pref = early_wrist_prefix(dacts, wx, wy, wz, until)
            restore_ic(env, dic)
            clear_probe_latch(env.sim)
            bil_i = None
            for i, a in enumerate(pref):
                step_legal(env.sim, a)
                if contact_summary(env.sim)["bilateral"] and bil_i is None:
                    bil_i = i
                if latched(env.sim):
                    break
            if bil_i is None or latched(env.sim):
                continue
            pose_bil = pose_now(env.sim)
            c = min(len(pref), bil_i + 3)
            stab_cfgs = ((35, 28, 0.35), (40, 32, 0.25), (30, 24, 0.40))
            if args.smoke:
                stab_cfgs = ((35, 28, 0.35),)
            for n_lift, n_seat, wy_stab in stab_cfgs:
                p = {
                    "n_lift": n_lift, "lift": -0.95, "n_return": 14, "n_seat": n_seat,
                    "n_fine": 0, "wy": wy_stab, "n_wy": 14, "xy_gain": 0.6,
                    "axis_align": True, "use_fine_down": False,
                    "return_vz": -0.18, "seat_vz": 0.08,
                }
                stab = build_stabilization(env, dic, pref[:c], p)
                if stab is None:
                    continue
                restore_ic(env, dic)
                clear_probe_latch(env.sim)
                for a in list(pref[:c]) + stab:
                    step_legal(env.sim, a)
                extra = []
                for _ in range(20):
                    a = privileged_correct_action(
                        env.sim, "combined", {"kp": 1.0, "kz": 0.25, "kw": 0.55, "wy": 0.0}
                    )
                    a[2] = float(np.clip(a[2] - 0.15, -1, 1))
                    step_legal(env.sim, a)
                    extra.append(a.copy())
                    if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                        break
                full = list(pref[:c]) + stab + extra
                pose_f = pose_now(env.sim)
                row = {
                    "snap": "during_capture",
                    "mode": "early_wrist",
                    "method_family": "during_capture_early_wrist",
                    "wx": wx, "wy": wy, "wz": wz, "until": until, "cut": c,
                    "theta0": theta0_ref,
                    "pos0": pos0_ref,
                    "theta_at_bil_mod": pose_bil["e_theta_deg"],
                    "pos_at_bil_mod": pose_bil["e_pos_mm"],
                    "theta1": pose_f["e_theta_deg"],
                    "pos1": pose_f["e_pos_mm"],
                    "d_theta": theta0_ref - pose_f["e_theta_deg"],
                    "d_pos": pos0_ref - pose_f["e_pos_mm"],
                    "bilateral_end": pose_f["bilateral"],
                    "n_suffix": len(stab) + len(extra),
                    "gains": json.dumps({"wx": wx, "wy": wy, "wz": wz}),
                    "below_10": int(pose_f["e_theta_deg"] <= WORKING_THETA_DEG),
                    "pos_ok": int(pose_f["e_pos_mm"] <= WORKING_POS_MM),
                }
                corr_rows.append(row)
                sc = (
                    100 * int(row["below_10"] and row["pos_ok"] and row["bilateral_end"])
                    + row["d_theta"] - 0.05 * abs(row["d_pos"])
                )
                payload = {
                    "score": sc, "full": full, "cut": c, "pose_f": pose_f,
                    "row": row, "wx": wx, "wy": wy, "wz": wz, "until": until,
                    "snap": "during_capture", "mode": "early_wrist",
                    "suffix": stab + extra, "prefix_len": c, "gains": {"wx": wx, "wy": wy, "wz": wz},
                }
                if best_early is None or sc > best_early["score"]:
                    best_early = payload
                if best_corr is None or sc > best_corr["score"]:
                    best_corr = payload
            print(
                f"  early wx={wx:.2f} wy={wy:.2f} wz={wz:.2f} until={until} "
                f"th_bil={pose_bil['e_theta_deg']:.2f} best_th="
                f"{best_corr['pose_f']['e_theta_deg'] if best_corr else None}",
                flush=True,
            )

    # ========== Stage C: full SUCCESS_NEW validation ==========
    print("Stage C: full SUCCESS_NEW validation of corrections", flush=True)
    distal_sn = False
    c_candidates = []
    if best_early is not None and best_early.get("full") is not None:
        c_candidates.append(("early_wrist_best", best_early["full"], best_early["cut"]))
    if best_corr is not None and best_corr.get("full") is not None:
        c_candidates.append(("best_corr_full", best_corr["full"], best_corr["cut"]))
    for wx, wy, wz in ((0.0, 0.14, 0.20), (0.0, 0.10, 0.20), (0.0, 0.12, 0.18)):
        for until in (dcut + 1, dcut + 2):
            pref = early_wrist_prefix(dacts, wx, wy, wz, until)
            restore_ic(env, dic)
            clear_probe_latch(env.sim)
            bil_i = None
            for i, a in enumerate(pref):
                step_legal(env.sim, a)
                if contact_summary(env.sim)["bilateral"] and bil_i is None:
                    bil_i = i
                if latched(env.sim):
                    break
            if bil_i is None:
                continue
            c = min(len(pref), bil_i + 3)
            p = {
                "n_lift": 35, "lift": -0.95, "n_return": 14, "n_seat": 28,
                "n_fine": 0, "wy": 0.35, "n_wy": 14, "xy_gain": 0.6,
                "axis_align": True, "use_fine_down": False,
                "return_vz": -0.18, "seat_vz": 0.08,
            }
            stab = build_stabilization(env, dic, pref[:c], p)
            if stab is None:
                continue
            restore_ic(env, dic)
            clear_probe_latch(env.sim)
            for a in list(pref[:c]) + stab:
                step_legal(env.sim, a)
            extra = []
            for _ in range(20):
                a = privileged_correct_action(
                    env.sim, "combined", {"kp": 1.0, "kz": 0.25, "kw": 0.55}
                )
                a[2] = float(np.clip(a[2] - 0.15, -1, 1))
                step_legal(env.sim, a)
                extra.append(a.copy())
                if latched(env.sim) or not contact_summary(env.sim)["bilateral"]:
                    break
            c_candidates.append((
                f"early_wrist_{wx}_{wy}_{wz}_u{until}",
                list(pref[:c]) + stab + extra, c,
            ))

    for method, full, c in c_candidates:
        if not time_left() or distal_sn:
            break
        out = eval_full_sn(env, dic, full, c, nominal_s=float(NOMINAL_S))
        handoff_rows.append({
            "method": method,
            "cut": c,
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SN_mean_theta": out.get("SN_mean_theta"),
            "SN_mean_pos": out.get("SN_mean_pos"),
            "held_s": out.get("held_s"),
            "fail_class": out.get("fail_class"),
        })
        print(
            f"  C {method} SN={out['SUCCESS_NEW']} CH={out['CATCH_HANDOFF_SUCCESS']} "
            f"th={out.get('SN_mean_theta')} pos={out.get('SN_mean_pos')}",
            flush=True,
        )
        if out["SUCCESS_NEW"]:
            ok = sum(
                int(eval_full_sn(env, dic, full, c, nominal_s=float(NOMINAL_S))["SUCCESS_NEW"])
                for _ in range(3)
            )
            if ok == 3:
                distal_sn = True
                np.savez_compressed(
                    OUT / "trajectories" / "distal_off01_r8__SUCCESS_NEW.npz",
                    actions=np.asarray(full, np.float32), cut=c, method=method,
                )
                new_experts.append({
                    "ic_id": "distal_off01_r8",
                    "provenance": "open_loop_traj_opt_during_capture_early_wrist",
                    "actions": full, "cut": c, "ic": dic,
                })
                cand_rows.append({
                    "ic_id": "distal_off01_r8", "SUCCESS_NEW": 1,
                    "provenance": "open_loop_traj_opt_during_capture_early_wrist",
                    "SN_mean_theta": out.get("SN_mean_theta"),
                    "SN_mean_pos": out.get("SN_mean_pos"),
                    "method": method,
                })
                if not args.skip_videos:
                    render_expert_video(
                        env, dic, full,
                        OUT / "videos" / "distal_off01_r8__SUCCESS_NEW.mp4",
                        "distal_corrected",
                    )
                print("  DISTAL SUCCESS_NEW verified 3/3", flush=True)

    if best_corr is not None and best_corr.get("full") is None and best_corr.get("suffix") is not None:
        pref = dacts[: best_corr["prefix_len"]]
        out = full_prefix_plus_suffix(env, dic, pref, best_corr["suffix"])
        handoff_rows.append({
            "method": f"post_only_{best_corr['snap']}_{best_corr['mode']}",
            "cut": best_corr["prefix_len"],
            "SUCCESS_NEW": out["SUCCESS_NEW"],
            "CATCH_HANDOFF_SUCCESS": out["CATCH_HANDOFF_SUCCESS"],
            "SN_mean_theta": out.get("SN_mean_theta"),
            "SN_mean_pos": out.get("SN_mean_pos"),
            "held_s": out.get("held_s"),
            "theta_local": best_corr["pose_f"]["e_theta_deg"],
        })
        print(
            f"  post_only SN={out['SUCCESS_NEW']} th={out.get('SN_mean_theta')} "
            f"local_th={best_corr['pose_f']['e_theta_deg']}",
            flush=True,
        )
        if not args.skip_videos:
            render_expert_video(
                env, dic, list(pref) + list(best_corr["suffix"]),
                OUT / "videos" / "distal_off01_r8__best_correction.mp4",
                "distal_best_corr",
            )
    elif best_corr is not None and best_corr.get("full") is not None and not args.skip_videos:
        render_expert_video(
            env, dic, best_corr["full"],
            OUT / "videos" / "distal_off01_r8__best_correction.mp4",
            "distal_best_corr",
        )

    if not args.skip_videos:
        render_expert_video(
            env, dic, dacts,
            OUT / "videos" / "distal_off01_r8__CATCH_HANDOFF_fail.mp4",
            "distal_fail",
        )

