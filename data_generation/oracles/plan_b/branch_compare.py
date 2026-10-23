"""Plan B branch comparison: B0 early-HANDOFF vs B1/B2 in-Recovery active recatch.

B0: short release → legal HANDOFF → nominal regrasp (nominal may restore contact)
B1: short release → active recatch inside Recovery → brake → HANDOFF
B2: B1 + optional Plan-A-style slide/pivot before HANDOFF

Does not change physics, SUCCESS_NEW, or the nominal controller.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from envs.outcomes.horizon_outcome import recovery_failed
from envs.outcomes.support_drop import latched, terminal_reason
from envs.simulation.capture import restore_capture
from envs.step_legal import step_legal
from evaluation.grasp_geometry import measure
from evaluation.success_criteria import P_TGT, U_TGT, contact_summary
from evaluation.handoff_from_state import evaluate_handoff_from_state
from training.action_vocab_k49 import build_k49_table, decode_action as decode_k49
from training.k47_codebook import decode_action as decode_k48
from training.k47_codebook import load_codebook

CONFIG_ID = "K49_tok48_wy0p25"
HANDOFF = 47


def load_vocab():
    book48 = load_codebook()
    table = build_k49_table(book48, config_id=CONFIG_ID)
    return table, book48


def decode_tok(table, book48, tok: int) -> np.ndarray:
    tok = int(tok)
    if tok <= 46:
        return np.asarray(decode_k48(book48, tok, np.zeros(7, np.float32))[1], np.float32)
    return np.asarray(decode_k49(table, tok, np.zeros(7, np.float32))[1], np.float32)


def sample_contact(sim) -> dict:
    cs = contact_summary(sim)
    m = measure(sim, U_TGT, P_TGT)
    return {
        "t": float(sim.data.time),
        "nL": int(cs["nL"]),
        "nR": int(cs["nR"]),
        "bilateral": bool(cs["bilateral"]),
        "obj_z": float(m["obj_z"]),
        "e_pos_mm": float(m["e_pos_mm"]),
        "e_theta_deg": float(m["e_theta_deg"]),
        "ground_drop": bool(latched(sim)),
    }


def run_tokens(env, blob, table, book48, tokens, *, trace=None) -> tuple[bool, str | None]:
    lost = 0
    for tok in tokens:
        if trace is not None:
            row = sample_contact(env.sim)
            row["tok"] = int(tok)
            trace.append(row)
        step_legal(env.sim, decode_tok(table, book48, tok))
        failed, lost = recovery_failed(env.sim, blob["ref"], lost)
        if failed:
            return False, terminal_reason(env.sim) or "DROP_DURING_RECOVERY"
    return True, None


def first_both_off(trace: list[dict], start_i: int = 0) -> int | None:
    for i in range(start_i, len(trace)):
        if int(trace[i]["nL"]) == 0 and int(trace[i]["nR"]) == 0:
            return i
    return None


def first_bilateral_after(trace: list[dict], start_i: int = 0, persist: int = 3) -> int | None:
    run = 0
    first = None
    for i in range(start_i, len(trace)):
        if trace[i]["bilateral"]:
            run += 1
            if run >= persist:
                return i - persist + 1 if first is None else first
            if first is None:
                first = i
        else:
            run = 0
            first = None
    return None


def classify_recatch_controller(trace_recovery: list[dict], prefix_len: int, post_handoff_bilateral_t) -> str:
    """Return who restored bilateral grasp: recovery | nominal | never | retained."""
    off = first_both_off(trace_recovery, 0)
    bil = first_bilateral_after(trace_recovery, prefix_len)
    if off is None and any(r["bilateral"] for r in trace_recovery):
        return "retained_or_partial"
    if bil is not None:
        return "recovery"
    if post_handoff_bilateral_t is not None:
        return "nominal"
    return "never"


def regrasp_candidates(rng: np.random.Generator, n: int = 32) -> list[list[int]]:
    """Legal K49 sequences for in-Recovery close/track after a short open prefix."""
    seqs: list[list[int]] = []
    for n0 in (5, 8, 12, 16, 20, 24, 30):
        seqs.append([0] * n0)
        seqs.append([12] * min(4, n0 // 2) + [0] * n0)
    for t in (0, 1, 5, 12, 13, 17, 33):
        for n_c in (8, 12, 16, 20):
            seqs.append([t] * n_c + [0] * 4)
            seqs.append([12] * 3 + [t] * n_c + [0] * 4)
    for o in (30, 36, 39, 44, 41, 32):
        for n_o in (1, 2, 3):
            seqs.append([o] * n_o + [0] * 12)
            seqs.append([o] * n_o + [12] * 3 + [0] * 12)
            seqs.append([o] * n_o + [1] * 12 + [0] * 5)
    for piv in ([4] * 10, [4] * 16, [6] * 12, []):
        seqs.append(list(piv) + [12] * 3 + [0] * 12)
        seqs.append(list(piv) + [0] * 16)
    seqs.append([36] * 5 + [0] * 8)
    seqs.append([44] * 1 + [0] * 16)
    seqs.append([30] * 2 + [0] * 16)
    seen, uniq = set(), []
    for s in seqs:
        k = tuple(s)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(list(s))
    head, tail = uniq[:10], uniq[10:]
    rng.shuffle(tail)
    return (head + tail)[:n]


def pose_family(pose: list[int]) -> str:
    """Coarse contact-geometry family for diversity-aware B2 pruning."""
    if not pose:
        return "none"
    # Dominant non-brake token decides family (K49: 4/6/10 pivot, 1/5/13/17 translate/slide, 48 wrist).
    counts: dict[str, int] = {"pivot": 0, "slide": 0, "wrist48": 0, "other": 0}
    for t in pose:
        t = int(t)
        if t in (4, 6, 10):
            counts["pivot"] += 1
        elif t in (1, 5, 13, 17, 12):
            counts["slide"] += 1
        elif t == 48:
            counts["wrist48"] += 1
        elif t != 0:
            counts["other"] += 1
    # Prefer explicit mechanism tokens over grip/brake padding.
    ranked = sorted(
        (("pivot", counts["pivot"]), ("slide", counts["slide"]), ("wrist48", counts["wrist48"]), ("other", counts["other"])),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked[0][0] if ranked[0][1] > 0 else "other"


def pose_candidates(rng: np.random.Generator, n: int = 12) -> list[list[int]]:
    """Diverse slide / pivot / wrist sequences; stratified head so pruning keeps family coverage."""
    by_fam: dict[str, list[list[int]]] = {"pivot": [], "slide": [], "wrist48": [], "other": []}
    for piv in ([4] * 20 + [12] * 5 + [0] * 3, [4] * 24 + [0] * 4, [6] * 16 + [0] * 4, [10] * 12 + [0] * 4):
        by_fam["pivot"].append(list(piv))
    for t in (1, 5, 13, 17):
        for n_tr in (8, 12, 16):
            by_fam["slide"].append([12] * 4 + [t] * n_tr + [0] * 4)
            by_fam["slide"].append([t] * n_tr + [0] * 4)
    for n48 in (8, 16, 24):
        by_fam["wrist48"].append([48] * n48 + [0] * 3)
    # Mixed pivot+translate for contact-geometry coverage
    by_fam["other"].append([4] * 12 + [1] * 8 + [0] * 4)
    by_fam["other"].append([6] * 10 + [5] * 10 + [0] * 4)

    for fam in by_fam:
        rng.shuffle(by_fam[fam])

    # Round-robin families so early budget slots are not all one mechanism.
    uniq: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    fam_order = ["pivot", "slide", "wrist48", "other"]
    idx = {f: 0 for f in fam_order}
    while len(uniq) < max(0, n - 1):
        progressed = False
        for fam in fam_order:
            if idx[fam] >= len(by_fam[fam]):
                continue
            s = by_fam[fam][idx[fam]]
            idx[fam] += 1
            k = tuple(s)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(list(s))
            progressed = True
            if len(uniq) >= n - 1:
                break
        if not progressed:
            break
    return uniq[:n]


def eval_branch(
    env,
    blob,
    table,
    book48,
    tokens: list[int],
    *,
    branch: str,
    prefix_len: int,
) -> dict[str, Any]:
    """Run Recovery tokens then HANDOFF+nominal; classify recatch controller."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    ic = sample_contact(env.sim)
    trace: list[dict] = []
    ok, fail_reason = run_tokens(env, blob, table, book48, tokens, trace=trace)
    if not ok:
        return {
            "branch": branch,
            "ok_recovery": False,
            "tokens": list(tokens),
            "prefix_len": int(prefix_len),
            "terminal_reason": fail_reason,
            "SUCCESS_12S": False,
            "SUCCESS_NEW": False,
            "recatch_controller": "never",
            "t_grasp_loss": None,
            "t_active_recatch": None,
            "t_handoff": None,
            "ground_contact": bool(latched(env.sim)),
            "ic": ic,
            "pre_handoff": sample_contact(env.sim),
            "trace_recovery": trace,
        }

    off_i = first_both_off(trace, 0)
    bil_i = first_bilateral_after(trace, prefix_len)
    pre = sample_contact(env.sim)
    t_handoff = float(env.sim.data.time)
    # Nominal after HANDOFF
    meta = evaluate_handoff_from_state(env.sim, blob["ref"], stabilize_n=0)
    post = sample_contact(env.sim)
    # Infer post-handoff first bilateral from meta / final state only
    post_bil_t = None
    if bil_i is None and post["bilateral"]:
        # Nominal restored contact sometime during 12s (exact time not in meta)
        post_bil_t = float(post["t"] - t0)
    controller = classify_recatch_controller(trace, prefix_len, post_bil_t)
    # Strict B1/B2: must have genuine both-off and pre-HANDOFF bilateral
    genuine_release = off_i is not None
    active_recatch = bil_i is not None and genuine_release
    if branch in ("B1", "B2") and not active_recatch:
        # Do not credit nominal-only regrasp as active
        return {
            "branch": branch,
            "ok_recovery": True,
            "tokens": list(tokens),
            "prefix_len": int(prefix_len),
            "terminal_reason": meta["reason"],
            "SUCCESS_12S": bool(meta["reason"] == "SUCCESS_12S"),
            "SUCCESS_NEW": False,
            "rejected_as": "NO_ACTIVE_RECATCH_BEFORE_HANDOFF",
            "recatch_controller": controller,
            "t_grasp_loss": None if off_i is None else float(trace[off_i]["t"] - t0),
            "t_active_recatch": None,
            "t_handoff": float(t_handoff - t0),
            "ground_contact": bool(latched(env.sim)) or bool(pre["ground_drop"]),
            "ic": ic,
            "pre_handoff": pre,
            "post_nominal": post,
            "handoff_meta": {
                "reason": meta["reason"],
                "SUCCESS_NEW": bool(meta["SUCCESS_NEW"]),
                "pre_pos_mm": meta.get("pre_pos_mm"),
                "pre_theta": meta.get("pre_theta"),
            },
            "trace_recovery": trace,
            "e_pos_final_mm": post["e_pos_mm"],
            "e_theta_final_deg": post["e_theta_deg"],
        }

    success_new = bool(meta["SUCCESS_NEW"])
    # B1/B2 require Recovery-authored recatch; B0 may credit nominal regrasp.
    if branch in ("B1", "B2") and controller != "recovery":
        success_new = False
    if branch == "B0":
        # Baseline: SUCCESS_NEW under early HANDOFF, regardless of who recatches.
        success_new = bool(meta["SUCCESS_NEW"])

    return {
        "branch": branch,
        "ok_recovery": True,
        "tokens": list(tokens),
        "prefix_len": int(prefix_len),
        "terminal_reason": meta["reason"],
        "SUCCESS_12S": bool(meta["reason"] == "SUCCESS_12S"),
        "SUCCESS_NEW": success_new,
        "recatch_controller": controller,
        "genuine_release": bool(genuine_release),
        "active_recatch_before_handoff": bool(active_recatch),
        "t_grasp_loss": None if off_i is None else float(trace[off_i]["t"] - t0),
        "t_active_recatch": None if bil_i is None else float(trace[bil_i]["t"] - t0),
        "t_handoff": float(t_handoff - t0),
        "ground_contact": bool(latched(env.sim)) or bool(pre["ground_drop"]),
        "ic": ic,
        "pre_handoff": pre,
        "post_nominal": post,
        "handoff_meta": {
            "reason": meta["reason"],
            "SUCCESS_NEW": bool(meta["SUCCESS_NEW"]),
            "pre_pos_mm": meta.get("pre_pos_mm"),
            "pre_theta": meta.get("pre_theta"),
        },
        "trace_recovery": trace,
        "e_pos_final_mm": post["e_pos_mm"],
        "e_theta_final_deg": post["e_theta_deg"],
    }


def run_b0(env, blob, table, book48, prefix: list[int]) -> dict:
    return eval_branch(env, blob, table, book48, list(prefix), branch="B0", prefix_len=len(prefix))


# Cheap pre-HANDOFF band: skip full 12 s nominal unless near SUCCESS_NEW working targets.
# Working SUCCESS_NEW uses θ≤10°, e_pos≤12 mm; keep a modest outer screen only.
SCREEN_THETA_DEG = 22.0
SCREEN_POS_MM = 28.0


def _recovery_only(env, blob, table, book48, tokens: list[int], prefix_len: int) -> dict | None:
    """Stage B cheap filter: Recovery tokens only; require release+bilateral before HANDOFF."""
    restore_capture(env, blob["capture"])
    t0 = float(env.sim.data.time)
    trace: list[dict] = []
    ok, fail = run_tokens(env, blob, table, book48, tokens, trace=trace)
    if not ok:
        return None
    off_i = first_both_off(trace, 0)
    bil_i = first_bilateral_after(trace, prefix_len)
    if off_i is None or bil_i is None:
        return None
    pre = sample_contact(env.sim)
    return {
        "tokens": list(tokens),
        "trace": trace,
        "t_grasp_loss": float(trace[off_i]["t"] - t0),
        "t_active_recatch": float(trace[bil_i]["t"] - t0),
        "pre": pre,
    }


def _in_screen_band(pre: dict) -> bool:
    return float(pre["e_theta_deg"]) <= SCREEN_THETA_DEG and float(pre["e_pos_mm"]) <= SCREEN_POS_MM


def _empty_stats() -> dict:
    return {
        "n_candidates": 0,
        "n_release_ok": 0,
        "n_active_recatch": 0,
        "n_pose_tried": 0,
        "n_full_nominal": 0,
        "n_screen_reject": 0,
        "best_theta": None,
        "best_pos_mm": None,
        "t_search_s": 0.0,
        "t_nominal_s": 0.0,
        "pose_families_tried": {},
        "pose_families_full_nominal": {},
    }


def _bump_fam(stats: dict, key: str, fam: str) -> None:
    bag = stats.setdefault(key, {})
    bag[fam] = int(bag.get(fam, 0)) + 1


def _track_best_pose(stats: dict, pre: dict) -> None:
    th = float(pre["e_theta_deg"])
    pos = float(pre["e_pos_mm"])
    if stats["best_theta"] is None or th < float(stats["best_theta"]):
        stats["best_theta"] = th
    if stats["best_pos_mm"] is None or pos < float(stats["best_pos_mm"]):
        stats["best_pos_mm"] = pos


def search_b1(
    env,
    blob,
    table,
    book48,
    prefix: list[int],
    rng: np.random.Generator,
    *,
    budget: int = 32,
    max_full_nominal: int = 6,
) -> dict:
    """Stage B→D: cheap active-recatch filter, then capped full SUCCESS_NEW evals."""
    import time

    t_stage = time.time()
    stats = _empty_stats()
    best_fail = None
    bodies = regrasp_candidates(rng, n=budget)
    stats["n_candidates"] = len(bodies)
    for body in bodies:
        tokens = list(prefix) + list(body)
        t_s0 = time.time()
        filt = _recovery_only(env, blob, table, book48, tokens, len(prefix))
        stats["t_search_s"] = round(float(stats["t_search_s"]) + (time.time() - t_s0), 3)
        if filt is None:
            continue
        stats["n_release_ok"] += 1
        stats["n_active_recatch"] += 1
        _track_best_pose(stats, filt["pre"])
        # Stage C screen: skip full 12 s unless near band or we still have budget for exploration
        if not _in_screen_band(filt["pre"]) and stats["n_full_nominal"] >= 2:
            stats["n_screen_reject"] += 1
            continue
        if stats["n_full_nominal"] >= int(max_full_nominal):
            continue
        stats["n_full_nominal"] += 1
        t_n0 = time.time()
        out = eval_branch(env, blob, table, book48, tokens, branch="B1", prefix_len=len(prefix))
        stats["t_nominal_s"] = round(float(stats["t_nominal_s"]) + (time.time() - t_n0), 3)
        if out.get("SUCCESS_NEW") and out.get("active_recatch_before_handoff"):
            out["correction"] = "none"
            out["search_stats"] = {**stats, "wall_s": round(time.time() - t_stage, 2)}
            return out
        if out.get("active_recatch_before_handoff") and out.get("SUCCESS_12S"):
            if best_fail is None or float(out["pre_handoff"]["e_pos_mm"]) < float(
                best_fail["pre_handoff"]["e_pos_mm"]
            ):
                best_fail = out
    result = best_fail or {
        "branch": "B1",
        "SUCCESS_NEW": False,
        "SUCCESS_12S": False,
        "terminal_reason": "NO_CANDIDATE",
        "recatch_controller": "never",
        "tokens": list(prefix),
        "prefix_len": len(prefix),
    }
    result["search_stats"] = {**stats, "wall_s": round(time.time() - t_stage, 2)}
    return result


def search_b2(
    env,
    blob,
    table,
    book48,
    prefix: list[int],
    rng: np.random.Generator,
    *,
    budget_regrasp: int = 24,
    budget_pose: int = 10,
    max_full_nominal: int = 8,
) -> dict:
    """Stage B→C→D: active recatch, optional pose tokens, capped full SUCCESS_NEW."""
    import time

    t_stage = time.time()
    stats = _empty_stats()
    best = None
    bodies = regrasp_candidates(rng, n=budget_regrasp)
    poses = [p for p in pose_candidates(rng, n=budget_pose) if p]
    stats["n_candidates"] = len(bodies)
    # Reserve at least one full-nominal slot per pose family when budget ≥ 4.
    fam_seen_full: set[str] = set()
    for body in bodies:
        if stats["n_full_nominal"] >= int(max_full_nominal):
            break
        base = list(prefix) + list(body)
        t_s0 = time.time()
        filt = _recovery_only(env, blob, table, book48, base, len(prefix))
        stats["t_search_s"] = round(float(stats["t_search_s"]) + (time.time() - t_s0), 3)
        if filt is None:
            continue
        stats["n_release_ok"] += 1
        stats["n_active_recatch"] += 1
        _track_best_pose(stats, filt["pre"])
        # Direct HANDOFF after recatch (no pose): only if in screen band
        if _in_screen_band(filt["pre"]) and stats["n_full_nominal"] < int(max_full_nominal):
            stats["n_full_nominal"] += 1
            _bump_fam(stats, "pose_families_full_nominal", "none")
            fam_seen_full.add("none")
            t_n0 = time.time()
            out1 = eval_branch(env, blob, table, book48, base, branch="B1", prefix_len=len(prefix))
            stats["t_nominal_s"] = round(float(stats["t_nominal_s"]) + (time.time() - t_n0), 3)
            if out1.get("SUCCESS_NEW") and out1.get("active_recatch_before_handoff"):
                out1["branch"] = "B1"
                out1["correction"] = "none"
                out1["search_stats"] = {**stats, "wall_s": round(time.time() - t_stage, 2)}
                return out1
            if out1.get("active_recatch_before_handoff"):
                if best is None or (
                    out1.get("SUCCESS_12S")
                    and float(out1["pre_handoff"]["e_pos_mm"])
                    < float((best.get("pre_handoff") or {}).get("e_pos_mm", 1e9))
                ):
                    best = out1
                    best["correction"] = "none_partial"
        for pose in poses:
            if stats["n_full_nominal"] >= int(max_full_nominal):
                break
            fam = pose_family(pose)
            tokens = base + list(pose)
            stats["n_pose_tried"] += 1
            _bump_fam(stats, "pose_families_tried", fam)
            t_s1 = time.time()
            filt2 = _recovery_only(env, blob, table, book48, tokens, len(prefix))
            stats["t_search_s"] = round(float(stats["t_search_s"]) + (time.time() - t_s1), 3)
            if filt2 is None:
                continue
            _track_best_pose(stats, filt2["pre"])
            in_band = _in_screen_band(filt2["pre"])
            # Keep diversity: allow one out-of-band full eval per unseen family if slots remain.
            diversify = fam not in fam_seen_full and stats["n_full_nominal"] + 1 < int(max_full_nominal)
            if not in_band and not diversify:
                stats["n_screen_reject"] += 1
                continue
            if not in_band:
                stats["n_screen_reject"] += 1  # counted, but still evaluated for diversity
            stats["n_full_nominal"] += 1
            _bump_fam(stats, "pose_families_full_nominal", fam)
            fam_seen_full.add(fam)
            t_n1 = time.time()
            out = eval_branch(env, blob, table, book48, tokens, branch="B2", prefix_len=len(prefix))
            stats["t_nominal_s"] = round(float(stats["t_nominal_s"]) + (time.time() - t_n1), 3)
            corr = "pivot" if fam == "pivot" else ("slide" if fam == "slide" else ("wrist48" if fam == "wrist48" else "slide_or_pivot"))
            if out.get("SUCCESS_NEW") and out.get("active_recatch_before_handoff"):
                out["correction"] = corr
                out["pose_family"] = fam
                out["pose_tokens"] = list(pose)
                out["search_stats"] = {**stats, "wall_s": round(time.time() - t_stage, 2)}
                return out
            if out.get("active_recatch_before_handoff"):
                if best is None or (
                    out.get("SUCCESS_12S")
                    and float(out["pre_handoff"]["e_pos_mm"])
                    < float((best.get("pre_handoff") or {}).get("e_pos_mm", 1e9))
                ):
                    best = out
                    best["correction"] = f"{corr}_partial"
                    best["pose_family"] = fam
    result = best or {
        "branch": "B2",
        "SUCCESS_NEW": False,
        "SUCCESS_12S": False,
        "terminal_reason": "NO_CANDIDATE",
        "recatch_controller": "never",
        "tokens": list(prefix),
        "prefix_len": len(prefix),
    }
    result["search_stats"] = {**stats, "wall_s": round(time.time() - t_stage, 2)}
    return result


def exact_replay_3x(env, blob, table, book48, tokens: list[int], *, branch: str, prefix_len: int) -> dict:
    rows = []
    for i in range(3):
        out = eval_branch(env, blob, table, book48, tokens, branch=branch, prefix_len=prefix_len)
        rows.append(
            {
                "replay_i": i,
                "SUCCESS_NEW": int(bool(out.get("SUCCESS_NEW"))),
                "SUCCESS_12S": int(bool(out.get("SUCCESS_12S"))),
                "reason": out.get("terminal_reason"),
                "recatch_controller": out.get("recatch_controller"),
                "active_recatch_before_handoff": bool(out.get("active_recatch_before_handoff")),
            }
        )
    n_ok = sum(r["SUCCESS_NEW"] for r in rows)
    return {"n": 3, "n_pass": n_ok, "rows": rows, "ok": n_ok == 3}
