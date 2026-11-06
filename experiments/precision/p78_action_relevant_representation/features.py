"""Phase 1 — causal engineered feature definitions (existing 39D only)."""

from __future__ import annotations

import json

import numpy as np

from experiments.precision.p78_action_relevant_representation.paths import ENG_HIST, OUT, SHORT_HIST

# Deployable obs indices 0..26 (observable_obs); 27..38 are GT airborne (privileged stand-in).
OBS27_NAMES = [
    "e_hat_x",
    "estimate_valid",
    "contact_present_L",
    "contact_present_R",
    "valid_L",
    "valid_R",
    "u_L",
    "v_L",
    "u_R",
    "v_R",
    "fn_L",
    "fn_R",
    "aperture",
    "ap_dot",
    "tau_from_secure",
    "g_hx",
    "g_hy",
    "g_hz",
    "v_hx",
    "v_hy",
    "v_z_world",
    "w_hx",
    "w_hy",
    "w_hz",
    "hand_z_off_table",
    "track_z",
    "e_hat_dot",
]

# Compact vec8 used in P74 (subset of 39D)
VEC8_IDX = (2, 3, 0, 1, 12, 13, 18, 19)  # L,R,e_hat,valid,ap,ap_dot,vhx,vhy
VEC8_NAMES = ["contact_L", "contact_R", "e_hat_x", "estimate_valid", "aperture", "ap_dot", "v_hx", "v_hy"]

ENG_NAMES = [
    "eh_last",
    "vld_last",
    "L_last",
    "R_last",
    "ap_last",
    "apd_last",
    "vhx_last",
    "vhy_last",
    "persist_L",
    "persist_R",
    "both_contact_frac",
    "tL_minus_tR",
    "eh_std",
    "eh_delta",
    "hand_speed_xy",
    "apd_delta",
]


def feature_catalog() -> dict:
    rows = []
    for name, src in zip(ENG_NAMES, [
        "e_hat_x[t]",
        "estimate_valid[t]",
        "contact_L[t]",
        "contact_R[t]",
        "aperture[t]",
        "ap_dot[t]",
        "v_hx[t]",
        "v_hy[t]",
        f"mean(contact_L[t-{ENG_HIST}+1:t+1])",
        f"mean(contact_R[t-{ENG_HIST}+1:t+1])",
        f"mean(L∧R over hist {ENG_HIST})",
        f"argfirst(L>0.5)-argfirst(R>0.5) in hist",
        f"std(e_hat over hist {ENG_HIST})",
        "e_hat[t]-e_hat[t0]",
        "hypot(v_hx,v_hy)[t]",
        "ap_dot[t]-ap_dot[t0]",
    ]):
        rows.append(
            {
                "name": name,
                "definition": src,
                "obs_components": "subset of obs indices {0,1,2,3,12,13,18,19} (+ history)",
                "history_length": ENG_HIST,
                "available_at_decision_t": True,
                "uses_future": False,
                "uses_privileged_geometry": False,
                "requires_new_sensing": False,
                "deployable": True,
            }
        )
    return {
        "engineered_features": rows,
        "targets_separated": {
            "1_current_sidewall_t": "correct_sidewall at decision state t (aligned with obs_t)",
            "2_sidewall_t_plus_1": "correct_sidewall after action at t (requires next-state label; features still pre-action)",
            "3_eventual_opposing": "episode reaches opposing bilateral (outcome; not current-state)",
            "4_eventual_success": "SUCCESS_RELAXED / r3_ready (terminal outcome)",
        },
        "ineligible": {
            "gt_airborne_27_38": "privileged stand-in — diagnostic only, not Actor input expansion basis",
            "Ft_shear": "P77 showed negligible lift; excluded (no new sensing)",
            "correct_sidewall_label": "privileged D — analysis only",
        },
        "history_params": {"SHORT_HIST": SHORT_HIST, "ENG_HIST": ENG_HIST},
    }


def obs27(obs39: np.ndarray) -> np.ndarray:
    o = np.asarray(obs39, float).reshape(-1)
    return o[:27].copy()


def vec8_from_obs27(o27: np.ndarray) -> np.ndarray:
    # Map full 27D → vec8 using absolute 39D indices that lie in 0..26
    # idx 2,3,0,1,12,13,18,19
    o = np.asarray(o27, float)
    return np.asarray([o[i] for i in VEC8_IDX], float)


def engineered_from_vec8_hist(hist_vec8: list[np.ndarray]) -> np.ndarray:
    X = np.stack(hist_vec8)
    L, R, eh, vld, ap, apd, vhx, vhy = X.T
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
            float(L.mean()),
            float(R.mean()),
            float(np.mean((L > 0.5) & (R > 0.5))),
            float(tL - tR),
            float(np.std(eh)),
            float(eh[-1] - eh[0]),
            float(np.hypot(vhx[-1], vhy[-1])),
            float(apd[-1] - apd[0]),
        ],
        dtype=np.float64,
    )


def engineered_from_obs27_hist(hist27: list[np.ndarray]) -> np.ndarray:
    return engineered_from_vec8_hist([vec8_from_obs27(h) for h in hist27])


def rep_A_current(o27: np.ndarray) -> np.ndarray:
    return np.asarray(o27, float)


def rep_B_current_plus_eng(hist27: list[np.ndarray]) -> np.ndarray:
    o = hist27[-1]
    eng = engineered_from_obs27_hist(hist27[-ENG_HIST:])
    return np.concatenate([o, eng])


def rep_C_history(hist27: list[np.ndarray]) -> np.ndarray:
    X = np.stack(hist27)
    return np.concatenate([X[-1], X.mean(0), X.std(0)])


def rep_D_history_plus_eng(hist27: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([rep_C_history(hist27), engineered_from_obs27_hist(hist27[-ENG_HIST:])])


def write_feature_defs() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    cat = feature_catalog()
    (OUT / "FEATURE_DEFINITIONS.json").write_text(json.dumps(cat, indent=2), encoding="utf-8")
    lines = [
        "# Feature Definitions (P78 Phase 1)\n",
        "Only **causal** features from existing 39D observations (indices 0–26). "
        "No Ft/shear, no privileged sidewall as inputs.\n",
        "## Engineered features\n",
        "| name | definition | hist | future? | privileged? | new sensing? |\n|---|---|---:|---|---|---|\n",
    ]
    for r in cat["engineered_features"]:
        lines.append(
            f"| `{r['name']}` | `{r['definition']}` | {r['history_length']} | "
            f"{r['uses_future']} | {r['uses_privileged_geometry']} | {r['requires_new_sensing']} |\n"
        )
    lines.append("\n## Separated targets\n")
    for k, v in cat["targets_separated"].items():
        lines.append(f"- **{k}**: {v}\n")
    lines.append("\n## Ineligible\n")
    for k, v in cat["ineligible"].items():
        lines.append(f"- **{k}**: {v}\n")
    (OUT / "FEATURE_DEFINITIONS.md").write_text("".join(lines), encoding="utf-8")
    return cat


if __name__ == "__main__":
    write_feature_defs()
    print("ok")
