"""P93 — Frozen dual-head router: independent generalization and C-retention validation.

Evaluation-only. Does not train, retune, or modify the P92 router, residual heads,
Transformer backbone, M111 token policy, or official R3F gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.precision.final_independent_success_rate.paths import (  # noqa: E402
    MAX_EP_STEPS,
    P64_ACTOR,
    T_MAX,
)
from experiments.precision.p51_four_class_mixed_ppo.ic_pools import root_id  # noqa: E402
from experiments.precision.p53_independent_grpo64.policy_sample import mask_handoff_logits  # noqa: E402
from experiments.precision.p53_independent_grpo64.reward import PoseRewardWeights  # noqa: E402
from experiments.precision.p54_rebalanced_grpo64.build_dataset import build_p54_dataset  # noqa: E402
from experiments.precision.p57_post_recatch_stabilization.env_stabilize import (  # noqa: E402
    StabilizeGrpoEnv,
)
from experiments.precision.p61_reproducible_joint_eval.seeding import seed_all  # noqa: E402
from experiments.precision.p63_transformer_critic_ppo.checkpointing import load_actor  # noqa: E402
from experiments.precision.p66_c_curriculum_ppo.expert_drive import (  # noqa: E402
    ensure_inner_capture_buffers,
)
from experiments.precision.p67_r3f_gate_diagnosis.gate_predicates import (  # noqa: E402
    evaluate_predicates,
)
from experiments.precision.p68_precontact_geometry.geometry import relative_geometry  # noqa: E402
from experiments.precision.p70_geometry_to_r3f.obs_proxy import last_obs  # noqa: E402
from experiments.precision.p73_first_contact_bilateral.controller import (  # noqa: E402
    WrapState,
    select_token,
    v2_spec,
)
from experiments.precision.p73_first_contact_bilateral.rollout import _r3_terminal  # noqa: E402
from experiments.precision.p86_b1_gate_counterfactual.gate_ablation import make_monitor  # noqa: E402
from experiments.precision.p92_dual_residual_head_routing.run_p92 import (  # noqa: E402
    CONF_THRESH,
    DECIDE_T,
    extract_features,
    forward_parts,
    make_router,
)

P90_B = ROOT / "results/experiments/p90_b1_residual_head_transfer/checkpoints/P90_stageB_p64_residual.pt"
P92_OUT = ROOT / "results/experiments/p92_dual_residual_head_routing"
P92_ROUTER = P92_OUT / "router_weights.npz"
P92_PAIRED = P92_OUT / "PAIRED_HEAD_OUTCOMES.csv"
P90_PAIRED = ROOT / "results/experiments/p90_b1_residual_head_transfer/PAIRED_PHYSICAL_EVALUATION.csv"
P90_XCLASS = ROOT / "results/experiments/p90_b1_residual_head_transfer/CROSS_CLASS_ROWS.csv"
P91_ABL = ROOT / "results/experiments/p91_c_preserving_residual/LOCALIZED_RESIDUAL_ABLATION.csv"
P88_CAND = ROOT / "results/experiments/p88_b1_short_sequence_learnability/PHASE0_CANDIDATES.csv"
P86_PANEL = ROOT / "results/experiments/p86_b1_gate_counterfactual/PANEL.json"
P86_XCLASS = ROOT / "results/experiments/p86_b1_gate_counterfactual/CROSS_CLASS_ABLATION.csv"
FINAL_OUT = ROOT / "results/experiments/final_independent_success_rate"
FROZEN_MANIFEST = FINAL_OUT / "FROZEN_TEST_MANIFEST.json"

OUT = ROOT / "results/experiments/p93_frozen_router_independent_validation"
SEED = 93
PANEL_SEED = 93

# Predefined sampling targets (fixed before seeing outcomes).
N_B1 = 20
N_C2 = 20
N_A1 = 12
N_D = 12
N_C1_TARGET = 10
N_C_SCREEN_CAP = 53  # all available root-disjoint C if needed
PERTURB_EPS = 1e-3
PERTURB_REPEATS = 5


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path, n: int = 16) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:n]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
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


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = (z / den) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, center - half), min(1.0, center + half)


def lookup_starts() -> dict[str, Any]:
    return {s.traj_id: s for s in build_p54_dataset(seed=5478)["schedule"]}


def load_frozen_router():
    z = np.load(P92_ROUTER)
    w = np.asarray(z["w"], dtype=np.float64)
    b = float(z["b"])
    thresh = float(z["thresh"])
    assert abs(thresh - CONF_THRESH) < 1e-12, f"threshold changed: {thresh} vs {CONF_THRESH}"
    return make_router(w, b, thresh), w, b, thresh


def contaminated_roots_and_tids() -> tuple[set[str], set[str], dict[str, list[str]]]:
    """Roots/tids used in P86–P92 development or exposed independent final test."""
    tids: set[str] = set()
    sources: dict[str, list[str]] = {}

    def add(label: str, xs: list[str]) -> None:
        sources[label] = sorted(set(xs))
        tids.update(xs)

    add("p92_panel", [r["traj_id"] for r in csv.DictReader(P92_PAIRED.open(encoding="utf-8"))])
    add("p90_paired", [r["traj_id"] for r in csv.DictReader(P90_PAIRED.open(encoding="utf-8"))])
    add("p90_xclass", [r["traj_id"] for r in csv.DictReader(P90_XCLASS.open(encoding="utf-8"))])
    add("p91_ablation", [r["traj_id"] for r in csv.DictReader(P91_ABL.open(encoding="utf-8"))])
    add("p88_candidates", [r["traj_id"] for r in csv.DictReader(P88_CAND.open(encoding="utf-8"))])
    p86 = json.loads(P86_PANEL.read_text(encoding="utf-8"))
    add("p86_b1", list(p86["b1_traj_ids"]))
    add("p86_xclass", [r["traj_id"] for r in csv.DictReader(P86_XCLASS.open(encoding="utf-8"))])
    man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    add("frozen_independent", [e["traj_id"] for e in man["episodes"]])
    add("exposed_b1", [e["traj_id"] for e in man["episodes"] if e["skill"] == "B1"])

    roots = {root_id(t) for t in tids}
    return roots, tids, sources


def available_by_skill(starts: dict[str, Any], ban_roots: set[str]) -> dict[str, list[str]]:
    by: dict[str, list[str]] = {"A1": [], "B1": [], "C": [], "D": []}
    for tid, s in starts.items():
        sk = s.skill
        if sk not in by:
            continue
        rid = root_id(tid)
        if rid in ban_roots:
            continue
        # substring overlap with banned roots (plan_b__ vs raw, etc.)
        if any(rid in br or br in rid for br in ban_roots):
            continue
        by[sk].append(tid)
    for sk in by:
        by[sk] = sorted(set(by[sk]))
    return by


def sample_fixed(pool: list[str], n: int, rng: np.random.Generator) -> list[str]:
    if not pool:
        return []
    order = list(pool)
    rng.shuffle(order)
    return order[: min(n, len(order))]


@torch.no_grad()
def closed_loop(
    token_actor,
    resid_actor,
    start,
    book,
    rho,
    device,
    *,
    seed: int = SEED,
    router_fn=None,
    collect_feat: bool = False,
    resid_trace_n: int = 4,
) -> dict:
    """Paired closed-loop. If router_fn is set, decide once at DECIDE_T and lock head."""
    seed_all(seed, deterministic_torch=True)
    env = StabilizeGrpoEnv(
        book, rho, max_steps=MAX_EP_STEPS, t_max=T_MAX, weights=PoseRewardWeights(), use_stabilization=False
    )
    env.reset(start)
    env.mon = make_monitor("G0", t_max=int(env.t_max))
    ensure_inner_capture_buffers(env)
    st = WrapState()
    v2 = v2_spec()
    done = False
    info: dict = {}
    tokens: list[int] = []
    resid_trace: list[list[float]] = []
    ever = {k: False for k in ("frame9", "frame12", "bil")}
    chosen = None
    conf = None
    route_t = None
    feat0 = None
    score0 = None
    while not done:
        t = int(env.t)
        obs = torch.as_tensor(np.stack(env.obs_seq[-160:]), device=device, dtype=torch.float32)
        prev = torch.as_tensor(np.stack(env.prev_seq[-160:]), device=device, dtype=torch.float32)
        ctx, logits, _, _ = forward_parts(token_actor, obs, prev)
        logit = mask_handoff_logits(logits[-1].float())
        tok_a = int(torch.argmax(logit).item())
        tok, st, _ = select_token(env_t=t, obs=last_obs(env), actor_tok=tok_a, st=st, v2=v2, rule=None)
        g = relative_geometry(env.env.sim)
        tr = getattr(env.mon, "tracker", None)
        ok_s = float(getattr(tr, "ok_s", 0.0) or 0.0)
        preds = evaluate_predicates(env.env.sim, ok_s=ok_s)
        pack = {"t": t, "ctx": ctx, "obs": obs, "prev": prev, "geom": g, "preds": preds}
        if router_fn is not None:
            if chosen is None and t == DECIDE_T:
                feat = extract_features(pack)
                feat0 = feat
                chosen, conf = router_fn(feat)
                # raw P90 score for margin audit
                # recover from conf + chosen
                route_t = t
            head = resid_actor["P90"] if chosen == "P90" else resid_actor["P64"]
        else:
            head = resid_actor
            if collect_feat and t == DECIDE_T and feat0 is None:
                feat0 = extract_features(pack)
        _, _, res, _ = forward_parts(head, obs, prev)
        resid = res[-1].detach().cpu().numpy()
        tokens.append(int(tok))
        if len(resid_trace) < resid_trace_n:
            resid_trace.append([float(x) for x in resid.tolist()])
        bil = bool(g["bilateral"])
        pad4 = int(g.get("min_pad") or 0) >= 4
        wk = bool(preds.get("working_ok"))
        e, th = float(g["e_pos_mm"]), float(g["theta_deg"])
        if bil:
            ever["bil"] = True
        if bil and pad4 and wk and e <= 9 + 1e-9 and th <= 8 + 1e-9:
            ever["frame9"] = True
        if bil and pad4 and wk and e <= 12 + 1e-9 and th <= 8 + 1e-9:
            ever["frame12"] = True
        _r, done, info = env.step(int(tok), resid)
        ensure_inner_capture_buffers(env)
    term = _r3_terminal(env, info)
    out = {
        "SUCCESS_RELAXED_12S": int(term.get("SUCCESS_RELAXED") or info.get("SUCCESS_RELAXED_12S") or 0),
        "SUCCESS_12S": int(info.get("SUCCESS_12S") or 0),
        "handoff": int(info.get("handoff") or 0),
        "r3_ready": int(term.get("r3_ready") or 0),
        "reach12": int(ever["frame12"]),
        "reach9": int(ever["frame9"]),
        "ep_len": int(env.t),
        "fail_reason": info.get("fail_reason"),
        "tokens": tokens,
        "resid_trace": resid_trace,
        "feat0": feat0,
        "chosen_head": chosen,
        "route_conf": conf,
        "route_t": route_t,
        "score0": score0,
    }
    return out


def route_score(w: np.ndarray, b: float, feat: np.ndarray) -> float:
    z = float(np.dot(feat, w) + b)
    return 1.0 / (1.0 + np.exp(-z))


def phase0_freeze(w: np.ndarray, b: float, thresh: float) -> None:
    lines = [
        f"# FROZEN_CONFIGURATION\n\nGenerated: {_utc()}\n\n",
        "Evaluation-only. No weights, thresholds, or controllers were modified after this freeze.\n\n",
        "## Artifacts and hashes (sha256[:16])\n\n",
        f"| Artifact | Path | sha256[:16] |\n|---|---|---|\n",
        f"| P64 actor (backbone+token+P64 residual) | `{P64_ACTOR}` | `{_sha256(P64_ACTOR)}` |\n",
        f"| P90 residual checkpoint | `{P90_B}` | `{_sha256(P90_B)}` |\n",
        f"| P92 router weights | `{P92_ROUTER}` | `{_sha256(P92_ROUTER)}` |\n",
        "\n## Router\n\n",
        f"- Feature dim: {w.shape[0]} (obs[:16] + ctx64 + prev7 + geo7)\n",
        f"- Bias b: {b}\n",
        f"- Threshold (P90 activation): **{thresh}** (frozen; not retuned)\n",
        f"- Decision time: t={DECIDE_T}; head locked for episode\n",
        f"- Default: P64 when P(P90) < threshold\n",
        "- Runtime inputs: inference-available t=0 obs/hidden/geometry/contact only\n",
        "- Excluded at runtime: task-class labels, IC/root IDs, teacher actions, future obs, oracle outcomes\n",
        "\n## Controllers\n\n",
        "- M111 token policy via frozen P64 logits + HANDOFF mask\n",
        "- Official R3F / G0 monitor unchanged\n",
        "- No residual interpolation; hard head selection only\n",
    ]
    _write(OUT / "FROZEN_CONFIGURATION.md", "".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[P93] device={device}", flush=True)

    router_fn, w, b, thresh = load_frozen_router()
    phase0_freeze(w, b, thresh)

    starts = lookup_starts()
    ban_roots, ban_tids, ban_sources = contaminated_roots_and_tids()
    avail = available_by_skill(starts, ban_roots)
    rng = np.random.default_rng(PANEL_SEED)

    # ------------------------------------------------------------------
    # Phase 1 — independent panels (sampling fixed before eval outcomes)
    # ------------------------------------------------------------------
    panel_b1 = sample_fixed(avail["B1"], N_B1, rng)
    panel_c2 = sample_fixed(avail["C"], N_C2, rng)
    panel_a1 = sample_fixed(avail["A1"], N_A1, rng)
    panel_d = sample_fixed(avail["D"], N_D, rng)

    # C1 candidates: remaining C not in C2, deterministic shuffle
    c_screen_pool = [t for t in avail["C"] if t not in set(panel_c2)]
    rng2 = np.random.default_rng(PANEL_SEED + 1)
    rng2.shuffle(c_screen_pool)
    c_screen_pool = c_screen_pool[:N_C_SCREEN_CAP]
    # Also screen C2 with P64 as part of representative eval; C1 filled from any P64 successes
    # among screened independent C (C2 + screen pool), excluding P92 roots already banned.

    man = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    exposed_b1 = sorted(e["traj_id"] for e in man["episodes"] if e["skill"] == "B1")

    manifest_rows = []
    for panel, tids, note in [
        ("B1", panel_b1, "root-disjoint B1 generalization; not conditioned on P90 benefit"),
        ("C2", panel_c2, "representative C; not conditioned on P64 success"),
        ("A1", panel_a1, "A1 regression control"),
        ("D", panel_d, "D regression control"),
        ("C_SCREEN", c_screen_pool, "C pool for P64-success screening toward C1"),
        ("DIAG_EXPOSED_B1", exposed_b1, "diagnostic replay only; NOT fresh independent validation"),
    ]:
        for tid in tids:
            manifest_rows.append(
                {
                    "panel": panel,
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "skill": "B1" if panel in ("B1", "DIAG_EXPOSED_B1") else (
                        "C" if panel in ("C2", "C_SCREEN", "C1") else panel
                    ),
                    "fresh_independent": int(panel not in ("DIAG_EXPOSED_B1",)),
                    "note": note,
                }
            )
    _write_csv(OUT / "EVALUATION_SPLIT_MANIFEST.csv", manifest_rows)
    _write(
        OUT / "PANEL_SAMPLING.json",
        json.dumps(
            {
                "panel_seed": PANEL_SEED,
                "ban_sources": {k: len(v) for k, v in ban_sources.items()},
                "available": {k: len(v) for k, v in avail.items()},
                "targets": {"B1": N_B1, "C2": N_C2, "A1": N_A1, "D": N_D, "C1": N_C1_TARGET},
                "selected": {
                    "B1": panel_b1,
                    "C2": panel_c2,
                    "A1": panel_a1,
                    "D": panel_d,
                    "C_SCREEN": c_screen_pool,
                },
            },
            indent=2,
        ),
    )

    p64, book, rho, _ = load_actor(P64_ACTOR, device)
    p90, _, _, _ = load_actor(P90_B, device)
    for p in list(p64.parameters()) + list(p90.parameters()):
        p.requires_grad_(False)
    p64.eval()
    p90.eval()
    heads = {"P64": p64, "P90": p90}

    # ------------------------------------------------------------------
    # Phase 2 — paired physical evaluation
    # ------------------------------------------------------------------
    # Union of ICs to evaluate (fresh panels). Screen C separately then merge C1.
    eval_plan: list[tuple[str, str]] = []  # (skill, tid) with panel tags applied later
    for tid in panel_b1:
        eval_plan.append(("B1", tid))
    for tid in panel_a1:
        eval_plan.append(("A1", tid))
    for tid in panel_d:
        eval_plan.append(("D", tid))
    for tid in panel_c2:
        eval_plan.append(("C", tid))

    # Cache results by tid
    cache: dict[str, dict] = {}

    def ensure_triple(skill: str, tid: str) -> dict:
        if tid in cache:
            return cache[tid]
        print(f"  eval {skill} {tid}", flush=True)
        start = starts[tid]
        r64 = closed_loop(p64, p64, start, book, rho, device, collect_feat=True)
        r90 = closed_loop(p64, p90, start, book, rho, device)
        rr = closed_loop(p64, heads, start, book, rho, device, router_fn=router_fn, collect_feat=True)
        # fill score/margin from features
        feat = rr.get("feat0") if rr.get("feat0") is not None else r64.get("feat0")
        p90_score = route_score(w, b, feat) if feat is not None else float("nan")
        margin = p90_score - thresh if feat is not None else float("nan")
        # token agreement P64 vs router path (same tokens expected — token head shared)
        tok_agree = float(
            np.mean([int(a == b) for a, b in zip(r64["tokens"], rr["tokens"])]) if r64["tokens"] and rr["tokens"] else float("nan")
        )
        row = {
            "skill": skill,
            "traj_id": tid,
            "root_id": root_id(tid),
            "P64_SUCCESS_RELAXED_12S": r64["SUCCESS_RELAXED_12S"],
            "P90_SUCCESS_RELAXED_12S": r90["SUCCESS_RELAXED_12S"],
            "ROUTER_SUCCESS_RELAXED_12S": rr["SUCCESS_RELAXED_12S"],
            "P64_SUCCESS_12S": r64["SUCCESS_12S"],
            "P90_SUCCESS_12S": r90["SUCCESS_12S"],
            "ROUTER_SUCCESS_12S": rr["SUCCESS_12S"],
            "chosen_head": rr["chosen_head"],
            "route_conf": rr["route_conf"],
            "p90_score": p90_score,
            "threshold": thresh,
            "margin": margin,
            "route_t": rr["route_t"],
            "P64_reach12": r64["reach12"],
            "P90_reach12": r90["reach12"],
            "ROUTER_reach12": rr["reach12"],
            "P64_r3_ready": r64["r3_ready"],
            "ROUTER_r3_ready": rr["r3_ready"],
            "P64_handoff": r64["handoff"],
            "ROUTER_handoff": rr["handoff"],
            "P64_ep_len": r64["ep_len"],
            "P90_ep_len": r90["ep_len"],
            "ROUTER_ep_len": rr["ep_len"],
            "P64_fail": r64.get("fail_reason"),
            "P90_fail": r90.get("fail_reason"),
            "ROUTER_fail": rr.get("fail_reason"),
            "token_agree_p64_router": tok_agree,
            "resid0_p64": json.dumps(r64["resid_trace"][0]) if r64["resid_trace"] else "",
            "resid0_p90": json.dumps(r90["resid_trace"][0]) if r90["resid_trace"] else "",
            "resid0_router": json.dumps(rr["resid_trace"][0]) if rr["resid_trace"] else "",
            "resid_trace_router": json.dumps(rr["resid_trace"]),
            "feat0": feat,
        }
        # paired class vs P64/P90
        s64, s90 = int(r64["SUCCESS_RELAXED_12S"]), int(r90["SUCCESS_RELAXED_12S"])
        if s90 and not s64:
            row["oracle_class"] = "P90_BETTER"
        elif s64 and not s90:
            row["oracle_class"] = "P64_BETTER"
        elif s64 and s90:
            row["oracle_class"] = "BOTH_SUCCESS"
        else:
            row["oracle_class"] = "BOTH_FAIL"
        cache[tid] = row
        return row

    print("[P93] Phase 2 paired eval (B1/A1/D/C2)", flush=True)
    for skill, tid in eval_plan:
        ensure_triple(skill, tid)

    # C screening for C1: P64-success retention candidates
    print("[P93] Phase 2/3 C1 screening", flush=True)
    panel_c1: list[str] = []
    # Prefer successes found already in C2
    for tid in panel_c2:
        row = cache[tid]
        if int(row["P64_SUCCESS_RELAXED_12S"]) == 1:
            panel_c1.append(tid)
    for tid in c_screen_pool:
        if len(panel_c1) >= N_C1_TARGET:
            break
        if tid in cache:
            row = cache[tid]
        else:
            # P64-only probe first; only full-triple if success (saves compute on fails)
            print(f"  screen C {tid}", flush=True)
            r64 = closed_loop(p64, p64, starts[tid], book, rho, device, collect_feat=True)
            if int(r64["SUCCESS_RELAXED_12S"]) != 1:
                # record fail-only screen row lightly
                cache[tid] = {
                    "skill": "C",
                    "traj_id": tid,
                    "root_id": root_id(tid),
                    "P64_SUCCESS_RELAXED_12S": 0,
                    "P90_SUCCESS_RELAXED_12S": "",
                    "ROUTER_SUCCESS_RELAXED_12S": "",
                    "screen_only_fail": 1,
                    "P64_fail": r64.get("fail_reason"),
                    "feat0": r64.get("feat0"),
                }
                continue
            # success → full triple
            row = ensure_triple("C", tid)
        if int(row.get("P64_SUCCESS_RELAXED_12S") or 0) == 1 and tid not in panel_c1:
            panel_c1.append(tid)

    # Ensure every C1 has full triple
    for tid in panel_c1:
        ensure_triple("C", tid)

    # Append C1 to manifest
    for tid in panel_c1:
        manifest_rows.append(
            {
                "panel": "C1",
                "traj_id": tid,
                "root_id": root_id(tid),
                "skill": "C",
                "fresh_independent": 1,
                "note": "P64-successful C retention stress test; conditioned on P64 success",
            }
        )
    _write_csv(OUT / "EVALUATION_SPLIT_MANIFEST.csv", manifest_rows)

    # Diagnostic exposed B1 (optional, labeled not-fresh). Map plan_b__* → schedule ids by root.
    print("[P93] Diagnostic exposed B1 replay (not fresh)", flush=True)
    diag_rows = []
    starts_by_root: dict[str, str] = {}
    for tid0, s0 in starts.items():
        if s0.skill == "B1":
            starts_by_root.setdefault(root_id(tid0), tid0)
    for tid in exposed_b1:
        mapped = tid if tid in starts else starts_by_root.get(root_id(tid))
        if mapped is None:
            # strip plan_b__ prefix
            alt = tid.replace("plan_b__", "", 1)
            mapped = alt if alt in starts else starts_by_root.get(root_id(alt))
        if mapped is None:
            diag_rows.append(
                {
                    "traj_id": tid,
                    "mapped_tid": "",
                    "error": "not_in_schedule",
                    "panel": "DIAG_EXPOSED_B1",
                    "fresh": 0,
                }
            )
            continue
        try:
            row = ensure_triple("B1", mapped)
            diag_rows.append(
                {
                    **{k: v for k, v in row.items() if k != "feat0"},
                    "exposed_traj_id": tid,
                    "mapped_tid": mapped,
                    "panel": "DIAG_EXPOSED_B1",
                    "fresh": 0,
                }
            )
        except Exception as e:
            diag_rows.append(
                {
                    "traj_id": tid,
                    "mapped_tid": mapped,
                    "error": repr(e),
                    "panel": "DIAG_EXPOSED_B1",
                    "fresh": 0,
                }
            )

    # ------------------------------------------------------------------
    # Emit primary CSVs
    # ------------------------------------------------------------------
    def strip_feat(row: dict) -> dict:
        return {k: v for k, v in row.items() if k != "feat0"}

    b1_rows = []
    for tid in panel_b1:
        r = cache[tid]
        s64 = int(r["P64_SUCCESS_RELAXED_12S"])
        sr = int(r["ROUTER_SUCCESS_RELAXED_12S"])
        if sr and not s64:
            trans = "P64_FAIL_ROUTER_SUCCESS"
        elif s64 and not sr:
            trans = "P64_SUCCESS_ROUTER_FAIL"
        elif s64 and sr:
            trans = "BOTH_SUCCESS"
        else:
            trans = "BOTH_FAIL"
        b1_rows.append({**strip_feat(r), "panel": "B1", "paired_transition": trans})
    _write_csv(OUT / "B1_PAIRED_OUTCOMES.csv", b1_rows)

    c1_rows = []
    for tid in panel_c1:
        r = cache[tid]
        s64 = int(r["P64_SUCCESS_RELAXED_12S"])
        sr = int(r["ROUTER_SUCCESS_RELAXED_12S"])
        status = "PRESERVED" if (s64 and sr) else ("REGRESSED" if (s64 and not sr) else "N/A")
        c1_rows.append(
            {
                **strip_feat(r),
                "panel": "C1",
                "retention_status": status,
                "bad_p90_activation": int(s64 == 1 and r["chosen_head"] == "P90"),
            }
        )
    _write_csv(OUT / "C_SUCCESS_RETENTION.csv", c1_rows)

    c2_rows = []
    for tid in panel_c2:
        r = cache[tid]
        c2_rows.append({**strip_feat(r), "panel": "C2"})
    _write_csv(OUT / "C_REPRESENTATIVE_EVALUATION.csv", c2_rows)

    a1d_rows = []
    for skill, tids in (("A1", panel_a1), ("D", panel_d)):
        for tid in tids:
            r = cache[tid]
            s64 = int(r["P64_SUCCESS_RELAXED_12S"])
            sr = int(r["ROUTER_SUCCESS_RELAXED_12S"])
            a1d_rows.append(
                {
                    **strip_feat(r),
                    "panel": skill,
                    "regression": int(s64 == 1 and sr == 0),
                    "gain": int(s64 == 0 and sr == 1),
                }
            )
    _write_csv(OUT / "A1_D_REGRESSION.csv", a1d_rows)

    # Router decision audit across all fresh triples
    audit = []
    for tid, r in cache.items():
        if r.get("screen_only_fail"):
            continue
        if "ROUTER_SUCCESS_RELAXED_12S" not in r or r["ROUTER_SUCCESS_RELAXED_12S"] == "":
            continue
        panels = []
        if tid in panel_b1:
            panels.append("B1")
        if tid in panel_c1:
            panels.append("C1")
        if tid in panel_c2:
            panels.append("C2")
        if tid in panel_a1:
            panels.append("A1")
        if tid in panel_d:
            panels.append("D")
        if tid in exposed_b1:
            panels.append("DIAG_EXPOSED_B1")
        audit.append(
            {
                "traj_id": tid,
                "root_id": r["root_id"],
                "skill": r["skill"],
                "panels": "|".join(panels),
                "fresh": int(tid not in set(exposed_b1)),
                "chosen_head": r["chosen_head"],
                "route_conf": r["route_conf"],
                "p90_score": r["p90_score"],
                "margin": r["margin"],
                "route_t": r["route_t"],
                "oracle_class": r.get("oracle_class"),
                "P64_SUCCESS": r["P64_SUCCESS_RELAXED_12S"],
                "P90_SUCCESS": r["P90_SUCCESS_RELAXED_12S"],
                "ROUTER_SUCCESS": r["ROUTER_SUCCESS_RELAXED_12S"],
                "resid0_p64": r.get("resid0_p64"),
                "resid0_router": r.get("resid0_router"),
                "token_agree_p64_router": r.get("token_agree_p64_router"),
            }
        )
    _write_csv(OUT / "ROUTER_DECISION_AUDIT.csv", audit)
    if diag_rows:
        _write_csv(OUT / "DIAG_EXPOSED_B1.csv", [{k: v for k, v in r.items() if k != "feat0"} for r in diag_rows])

    # ------------------------------------------------------------------
    # Phase 5 — reliability (feature shift + perturbation)
    # ------------------------------------------------------------------
    print("[P93] Phase 5 reliability", flush=True)
    # P92 train features: recompute on P92 disagreement ICs at t=0 via P64 path
    p92_rows = list(csv.DictReader(P92_PAIRED.open(encoding="utf-8")))
    p92_feats = []
    for r in p92_rows:
        tid = r["traj_id"]
        if tid not in starts:
            continue
        out = closed_loop(p64, p64, starts[tid], book, rho, device, collect_feat=True)
        if out.get("feat0") is not None:
            p92_feats.append(out["feat0"])
    X_p92 = np.stack(p92_feats) if p92_feats else np.zeros((0, w.shape[0]))

    fresh_feats = []
    for tid in panel_b1 + panel_c2 + panel_a1 + panel_d + panel_c1:
        feat = cache.get(tid, {}).get("feat0")
        if feat is not None:
            fresh_feats.append(np.asarray(feat, dtype=np.float64))
    X_fresh = np.stack(fresh_feats) if fresh_feats else np.zeros((0, w.shape[0]))

    shift_l2 = float("nan")
    if len(X_p92) and len(X_fresh):
        shift_l2 = float(np.linalg.norm(X_p92.mean(0) - X_fresh.mean(0)))

    # Perturbation stability on fresh B1+C1
    pert_rows = []
    for tid in panel_b1 + panel_c1:
        feat = cache.get(tid, {}).get("feat0")
        if feat is None:
            continue
        base_head, _ = router_fn(np.asarray(feat, dtype=np.float64))
        flips = 0
        for i in range(PERTURB_REPEATS):
            noise = np.random.default_rng(SEED * 1000 + i).normal(0.0, PERTURB_EPS, size=feat.shape)
            h, _ = router_fn(np.asarray(feat, dtype=np.float64) + noise)
            if h != base_head:
                flips += 1
        pert_rows.append(
            {
                "traj_id": tid,
                "base_head": base_head,
                "flips": flips,
                "repeats": PERTURB_REPEATS,
                "eps": PERTURB_EPS,
                "stable": int(flips == 0),
            }
        )
    _write_csv(OUT / "PERTURBATION_STABILITY.csv", pert_rows)
    n_stable = sum(int(r["stable"]) for r in pert_rows)
    n_pert = len(pert_rows)

    # ------------------------------------------------------------------
    # Phases 3–4 metrics + CASE
    # ------------------------------------------------------------------
    # B1 paired
    both_s = sum(1 for r in b1_rows if r["paired_transition"] == "BOTH_SUCCESS")
    gain = sum(1 for r in b1_rows if r["paired_transition"] == "P64_FAIL_ROUTER_SUCCESS")
    loss = sum(1 for r in b1_rows if r["paired_transition"] == "P64_SUCCESS_ROUTER_FAIL")
    both_f = sum(1 for r in b1_rows if r["paired_transition"] == "BOTH_FAIL")
    n_b1 = len(b1_rows)
    p64_b1 = sum(int(r["P64_SUCCESS_RELAXED_12S"]) for r in b1_rows)
    rou_b1 = sum(int(r["ROUTER_SUCCESS_RELAXED_12S"]) for r in b1_rows)
    p90_b1 = sum(int(r["P90_SUCCESS_RELAXED_12S"]) for r in b1_rows)
    net = gain - loss
    frac_p90 = (sum(1 for r in b1_rows if r["chosen_head"] == "P90") / n_b1) if n_b1 else float("nan")
    # precision among measurable benefit (P90_BETTER)
    p90_better = [r for r in b1_rows if r.get("oracle_class") == "P90_BETTER"]
    prec = (
        sum(1 for r in p90_better if r["chosen_head"] == "P90") / len(p90_better) if p90_better else float("nan")
    )

    # C1 retention
    n_c1 = len(c1_rows)
    n_c1_pres = sum(1 for r in c1_rows if r["retention_status"] == "PRESERVED")
    n_c1_reg = sum(1 for r in c1_rows if r["retention_status"] == "REGRESSED")
    bad_p90_c = sum(int(r["bad_p90_activation"]) for r in c1_rows)
    ret_rate, ret_lo, ret_hi = wilson_ci(n_c1_pres, n_c1) if n_c1 else (float("nan"),) * 3

    # C2 representative
    n_c2 = len(c2_rows)
    p64_c2 = sum(int(r["P64_SUCCESS_RELAXED_12S"]) for r in c2_rows)
    rou_c2 = sum(int(r["ROUTER_SUCCESS_RELAXED_12S"]) for r in c2_rows)
    p90_c2 = sum(int(r["P90_SUCCESS_RELAXED_12S"]) for r in c2_rows)

    # A1/D
    a1_reg = sum(int(r["regression"]) for r in a1d_rows if r["panel"] == "A1")
    d_reg = sum(int(r["regression"]) for r in a1d_rows if r["panel"] == "D")
    a1_gain = sum(int(r["gain"]) for r in a1d_rows if r["panel"] == "A1")
    d_gain = sum(int(r["gain"]) for r in a1d_rows if r["panel"] == "D")
    p64_a1 = sum(int(r["P64_SUCCESS_RELAXED_12S"]) for r in a1d_rows if r["panel"] == "A1")
    rou_a1 = sum(int(r["ROUTER_SUCCESS_RELAXED_12S"]) for r in a1d_rows if r["panel"] == "A1")
    p64_d = sum(int(r["P64_SUCCESS_RELAXED_12S"]) for r in a1d_rows if r["panel"] == "D")
    rou_d = sum(int(r["ROUTER_SUCCESS_RELAXED_12S"]) for r in a1d_rows if r["panel"] == "D")

    b1_p, b1_lo, b1_hi = wilson_ci(rou_b1, n_b1) if n_b1 else (float("nan"),) * 3
    gain_p, gain_lo, gain_hi = wilson_ci(gain, n_b1) if n_b1 else (float("nan"),) * 3

    # C1 target for a meaningful retention screen is 10; <5 is too thin for CASE A.
    insufficient_c = n_c1 < 5
    insufficient_b1 = n_b1 < 10
    unstable = (n_pert > 0 and (n_stable / n_pert) < 0.8) or (
        shift_l2 == shift_l2 and shift_l2 > 20.0
    )

    b1_improved = net >= 1 and rou_b1 > p64_b1
    c_retained = (n_c1_reg == 0) and (n_c1 >= 1)
    a1d_ok = a1_reg == 0 and d_reg == 0

    # Aggressive P90 rate plus P64_BETTER→P90 mistakes on A1/D indicate development-biased routing.
    dev_biased_routing = (
        (isinstance(frac_p90, float) and frac_p90 >= 0.9)
        and (a1_reg + d_reg) >= 1
        and c_retained
    )

    if insufficient_b1 or insufficient_c:
        case, title = "E", "Insufficient independent data for a reliable conclusion"
    elif b1_improved and n_c1_reg > 0:
        case, title = "B", "B1 improvement generalizes, but C retention fails"
    elif b1_improved and c_retained and a1d_ok and not unstable and not dev_biased_routing:
        case, title = "A", "Independent B1 improvement with no observed C/A1/D regression"
    elif b1_improved and c_retained and (not a1d_ok or unstable or dev_biased_routing):
        # C held, but A1/D regressions / unstable or development-biased P90 activation.
        case, title = "D", "Routing decisions are unstable or depend on development-specific features"
    elif (not b1_improved) and c_retained and n_c1 >= 5 and a1d_ok:
        case, title = "C", "C retention generalizes, but B1 improvement disappears"
    elif unstable or dev_biased_routing:
        case, title = "D", "Routing decisions are unstable or depend on development-specific features"
    elif not b1_improved:
        case, title = "F", "No improvement over frozen P64"
    else:
        case, title = "E", "Insufficient independent data for a reliable conclusion"

    unc = f"""# UNCERTAINTY_ANALYSIS

Generated: {_utc()}

Wilson score intervals (z=1.96). Finite-panel preservation is not a population guarantee.

## B1 (fresh independent, n={n_b1})

| Metric | k/n | rate | Wilson 95% CI |
|---|---:|---:|---|
| Router SUCCESS_RELAXED_12S | {rou_b1}/{n_b1} | {b1_p:.3f} | [{b1_lo:.3f}, {b1_hi:.3f}] |
| Paired gain (P64 fail→router succ) | {gain}/{n_b1} | {gain_p:.3f} | [{gain_lo:.3f}, {gain_hi:.3f}] |
| Paired loss | {loss}/{n_b1} | | |
| Net paired gain | {net} | | |
| P64 / P90 baselines | {p64_b1} / {p90_b1} | | |

## C1 retention (P64-successful independent C, n={n_c1})

| Metric | k/n | rate | Wilson 95% CI |
|---|---:|---:|---|
| PRESERVED | {n_c1_pres}/{n_c1} | {ret_rate:.3f} | [{ret_lo:.3f}, {ret_hi:.3f}] |
| REGRESSED | {n_c1_reg}/{n_c1} | | |
| Bad P90 on protected | {bad_p90_c} | | |

Target was ≥{N_C1_TARGET} baseline-successful C ICs; obtained **{n_c1}** after screening {len(c_screen_pool)} + C2 pool.
Zero observed regressions does **not** establish zero population regression risk.

## C2 representative (n={n_c2})

| Policy | SUCCESS_RELAXED_12S |
|---|---:|
| P64 | {p64_c2}/{n_c2} |
| P90 | {p90_c2}/{n_c2} |
| Router | {rou_c2}/{n_c2} |

C2 estimates overall C rate under routing; it is **not** the conditional retention rate.

## A1 / D

| Skill | P64 | Router | regressions | gains |
|---|---:|---:|---:|---:|
| A1 | {p64_a1}/{len(panel_a1)} | {rou_a1}/{len(panel_a1)} | {a1_reg} | {a1_gain} |
| D | {p64_d}/{len(panel_d)} | {rou_d}/{len(panel_d)} | {d_reg} | {d_gain} |

## Reliability

| Item | Value |
|---|---|
| Feature mean L2 shift (P92 panel → fresh) | {shift_l2:.4f} |
| Perturbation stable (eps={PERTURB_EPS}) | {n_stable}/{n_pert} |
| Frac B1 routed to P90 | {frac_p90:.3f} |
| P90 selection precision on P90_BETTER | {prec} |

Frozen independent P64 B1 **0/8** is preserved and not counted as fresh validation.
"""
    _write(OUT / "UNCERTAINTY_ANALYSIS.md", unc)

    root_md = f"""# ROOT_DISJOINT_VALIDATION

Generated: {_utc()}

## Exclusion

Banned roots from P86/P88/P90/P91/P92 development and frozen independent final-test episodes.
Available after exclusion: B1={len(avail['B1'])}, C={len(avail['C'])}, A1={len(avail['A1'])}, D={len(avail['D'])}.

## Fresh panels

| Panel | n | note |
|---|---:|---|
| B1 | {len(panel_b1)} | generalization |
| C1 | {len(panel_c1)} | P64-success retention stress |
| C2 | {len(panel_c2)} | representative C |
| A1 | {len(panel_a1)} | regression control |
| D | {len(panel_d)} | regression control |

Exposed independent B1 (0/8 under P64) replayed only as diagnostics — not fresh.
"""
    _write(OUT / "ROOT_DISJOINT_VALIDATION.md", root_md)

    final = f"""# FINAL_REPORT — P93 Frozen Dual-Head Router Independent Validation

Generated: {_utc()}

## CASE {case} — {title}

### Integrity

- Evaluation-only: router / residual heads / backbone / M111 / R3F **unchanged**
- Threshold frozen at {thresh}; not retuned on evaluation data
- Routing from actual t={DECIDE_T} causal features with head lock
- No privileged class/IC labels at runtime
- P64 **not** replaced
- Frozen independent P64 B1 **0/8** preserved (diagnostic replay only)

### Phase 0 — freeze

See `FROZEN_CONFIGURATION.md` (P64 `{_sha256(P64_ACTOR)}`, P90 `{_sha256(P90_B)}`, router `{_sha256(P92_ROUTER)}`).

### Phase 1 — panels

| Panel | n | Available pool |
|---|---:|---:|
| B1 | {len(panel_b1)} | {len(avail['B1'])} |
| C1 (P64-success) | {len(panel_c1)} (target {N_C1_TARGET}) | screened from {len(avail['C'])} |
| C2 (representative) | {len(panel_c2)} | {len(avail['C'])} |
| A1 | {len(panel_a1)} | {len(avail['A1'])} |
| D | {len(panel_d)} | {len(avail['D'])} |

### Phase 2–4 — paired outcomes

**B1:** P64 {p64_b1}/{n_b1} → Router {rou_b1}/{n_b1} (P90 {p90_b1}/{n_b1}); transitions both_succ={both_s}, gain={gain}, loss={loss}, both_fail={both_f}; **net={net}**.

**C1 retention:** {n_c1_pres}/{n_c1} PRESERVED, {n_c1_reg} REGRESSED; bad P90 activations={bad_p90_c}. Wilson [{ret_lo:.3f}, {ret_hi:.3f}].

**C2 representative:** P64 {p64_c2}/{n_c2}, Router {rou_c2}/{n_c2}, P90 {p90_c2}/{n_c2}.

**A1/D:** regressions A1={a1_reg}, D={d_reg}; gains A1={a1_gain}, D={d_gain}.

### Phase 5 — reliability

Feature mean L2 shift={shift_l2:.4f}; perturbation stability={n_stable}/{n_pert}; B1→P90 fraction={frac_p90:.3f}.

### Caveats

- Conditional C1 retention ≠ general C population success.
- n_C1={n_c1} after independent screening; {"below target — retention evidence limited." if n_c1 < N_C1_TARGET else "target met."}
- CASE A supports further consideration, not automatic deployment.
"""
    _write(OUT / "FINAL_REPORT.md", final)
    (OUT / "SUMMARY.json").write_text(
        json.dumps(
            {
                "case": case,
                "title": title,
                "b1": {
                    "n": n_b1,
                    "p64": p64_b1,
                    "p90": p90_b1,
                    "router": rou_b1,
                    "gain": gain,
                    "loss": loss,
                    "net": net,
                    "frac_p90": frac_p90,
                },
                "c1": {
                    "n": n_c1,
                    "preserved": n_c1_pres,
                    "regressed": n_c1_reg,
                    "bad_p90": bad_p90_c,
                    "wilson": [ret_rate, ret_lo, ret_hi],
                },
                "c2": {"n": n_c2, "p64": p64_c2, "p90": p90_c2, "router": rou_c2},
                "a1d": {
                    "a1_reg": a1_reg,
                    "d_reg": d_reg,
                    "a1_gain": a1_gain,
                    "d_gain": d_gain,
                },
                "reliability": {
                    "shift_l2": shift_l2,
                    "perturb_stable": [n_stable, n_pert],
                },
                "frozen_independent_b1_0_8_preserved": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[P93] DONE CASE {case} — {title}", flush=True)


if __name__ == "__main__":
    main()
