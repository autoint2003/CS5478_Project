"""recovery4d interface tests. No SAC training."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controllers.residual import (  # noqa: E402
    RECOVERY4D_TAU_OPEN,
    RECOVERY4D_TAU_SECURE,
    map_recovery4d,
)
from controllers.gripper_controller import clip_fg  # noqa: E402
from envs.config_util import load_yaml, merge_sim_config  # noqa: E402
from envs.physical_recovery import (  # noqa: E402
    RH_Z_NOM,
    SCALE_AP,
    SCALE_CLEAR,
    SCALE_EX,
    SCALE_EY,
    SCALE_F,
    SCALE_G,
    SCALE_RH_Z,
    SCALE_TAU,
    SCALE_V,
    SCALE_W,
    SCALE_Z,
    TABLE_TOP,
    TAU_SECURE,
    V_REL_TOL,
    W_REL_TOL,
    mat6,
    observe_recovery4d,
    physical_pack,
)
from envs.recovery_env import RecoveryEnv, load_buffer  # noqa: E402
from training.offset_construct import make_one_training_ic  # noqa: E402
from training.replay_core import FROZEN_YAML, CTRL_PY, sha256_file  # noqa: E402
from training.replay_core import EXPECTED_YAML, EXPECTED_PY  # noqa: E402

OFFSET_NPZ = ROOT / "results" / "eval_sets" / "airborne_offset_eval.npz"
IMPACT_NPZ = ROOT / "results" / "eval_sets" / "impact_severity_four.npz"


def _ic():
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    return cfg, make_one_training_ic(cfg, np.random.default_rng(3))


def test_frozen_and_eval_untouched():
    assert sha256_file(FROZEN_YAML) == EXPECTED_YAML
    assert sha256_file(CTRL_PY) == EXPECTED_PY
    assert OFFSET_NPZ.is_file() and IMPACT_NPZ.is_file()


def test_zero_no_hidden_lift(cfg, snap):
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d")
    env.reset(options={"snapshot": snap})
    z0 = float(env.sim.fsm.p_des[2])
    vz = []
    for _ in range(15):
        _, _, _, _, info = env.step(np.zeros(4, np.float32))
        vz.append(float(info["v_cmd"][2]))
    z1 = float(env.sim.fsm.p_des[2])
    assert max(abs(v) for v in vz) < 1e-9, vz[:3]
    assert abs(z1 - z0) < 1e-6, (z0, z1)


def test_hand_x_signs(cfg, snap):
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d")
    env.reset(options={"snapshot": snap})
    p0 = np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    R0 = np.array(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
    for _ in range(20):
        env.step(np.array([0.0, 1.0, 0.0, 1.0], np.float32))
    p1 = np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    d_plus = float((R0.T @ (p1 - p0))[0])
    env.reset(options={"snapshot": snap})
    p0 = np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    R0 = np.array(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
    for _ in range(20):
        env.step(np.array([0.0, -1.0, 0.0, 1.0], np.float32))
    p1 = np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    d_minus = float((R0.T @ (p1 - p0))[0])
    assert d_plus > 0.002, d_plus
    assert d_minus < -0.002, d_minus


def _rot_axis(r0, r1):
    d = r1 @ r0.T
    sk = np.array([d[2, 1] - d[1, 2], d[0, 2] - d[2, 0], d[1, 0] - d[0, 1]])
    n = np.linalg.norm(sk)
    return sk / max(n, 1e-12), n


def test_wrist_hand_y(cfg, snap):
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d")
    env.reset(options={"snapshot": snap})
    r0 = np.array(env.sim.fsm.r_des, float).reshape(3, 3)
    hy = r0[:, 1].copy()
    for _ in range(10):
        env.step(np.array([1.0, 0.0, 0.0, 1.0], np.float32))
    r1 = np.array(env.sim.fsm.r_des, float).reshape(3, 3)
    ax, mag = _rot_axis(r0, r1)
    assert mag > 0.05, mag
    assert abs(float(np.dot(ax, hy))) > 0.85, (ax, hy)

    env.reset(options={"snapshot": snap})
    r0 = np.array(env.sim.fsm.r_des, float).reshape(3, 3)
    hy = r0[:, 1].copy()
    for _ in range(10):
        env.step(np.array([-1.0, 0.0, 0.0, 1.0], np.float32))
    r1 = np.array(env.sim.fsm.r_des, float).reshape(3, 3)
    ax, mag = _rot_axis(r0, r1)
    assert mag > 0.05
    assert abs(float(np.dot(ax, hy))) > 0.85
    # opposite sense vs +a[0] relative to +hand-y
    env.reset(options={"snapshot": snap})
    r0 = np.array(env.sim.fsm.r_des, float).reshape(3, 3)
    env.step(np.array([1.0, 0.0, 0.0, 1.0], np.float32))
    rp = np.array(env.sim.fsm.r_des, float).reshape(3, 3)
    env.reset(options={"snapshot": snap})
    env.step(np.array([-1.0, 0.0, 0.0, 1.0], np.float32))
    rm = np.array(env.sim.fsm.r_des, float).reshape(3, 3)
    assert float(np.linalg.norm(rp - rm)) > 1e-3


def test_grip_mapping_and_no_fg_min():
    r = np.eye(3)
    m18 = map_recovery4d(np.array([0, 0, 0, 1.0]), r, r)
    m1 = map_recovery4d(np.array([0, 0, 0, -1.0]), r, r)
    a_slip = 2.0 * ((-2.0) - RECOVERY4D_TAU_OPEN) / (RECOVERY4D_TAU_SECURE - RECOVERY4D_TAU_OPEN) - 1.0
    m2 = map_recovery4d(np.array([0, 0, 0, a_slip]), r, r)
    assert abs(m18["tau"] - (-18.0)) < 1e-9
    assert abs(m1["tau"] - RECOVERY4D_TAU_OPEN) < 1e-9
    assert abs(m2["tau"] - (-2.0)) < 1e-6
    cfg = {"gripper": {"fg_min": 8.0, "fg_max": 45.0}}
    # clip_fg(2) would be 8; recovery4d must not use that path
    assert clip_fg(2.0, cfg) == 8.0
    assert m2["tau"] > -8.0  # more open than fg_min=8 => tau=-8


def test_grip_env_ctrl(cfg, snap):
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d")
    env.reset(options={"snapshot": snap})
    env.step(np.array([0.0, 0.0, 0.0, 1.0], np.float32))
    assert abs(float(env.sim.data.ctrl[7]) - (-18.0)) < 1e-6
    env.reset(options={"snapshot": snap})
    env.step(np.array([0.0, 0.0, 0.0, -1.0], np.float32))
    assert abs(float(env.sim.data.ctrl[7]) - RECOVERY4D_TAU_OPEN) < 1e-6
    a_slip = 2.0 * ((-2.0) - RECOVERY4D_TAU_OPEN) / (RECOVERY4D_TAU_SECURE - RECOVERY4D_TAU_OPEN) - 1.0
    env.reset(options={"snapshot": snap})
    env.step(np.array([0.0, 0.0, 0.0, a_slip], np.float32))
    assert abs(float(env.sim.data.ctrl[7]) - (-2.0)) < 1e-5


def test_snapshot_roundtrip(cfg, snap):
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d")
    env.reset(options={"snapshot": snap})
    q = np.array(env.sim.data.qpos, float).copy()
    v = np.array(env.sim.data.qvel, float).copy()
    p = np.array(env.sim.fsm.p_des, float).copy()
    r = np.array(env.sim.fsm.r_des, float).copy()
    vc = np.array(env.sim.fsm.v_cmd, float).copy()
    c7 = float(env.sim.data.ctrl[7])
    packed = env.sim.snapshot()
    env.sim.reset(0.08, 1.0, np.zeros(3))
    env.reset(options={"snapshot": packed})
    assert float(np.max(np.abs(np.array(env.sim.data.qpos, float) - q))) < 1e-8
    assert float(np.max(np.abs(np.array(env.sim.data.qvel, float) - v))) < 1e-8
    assert float(np.max(np.abs(np.array(env.sim.fsm.p_des, float) - p))) < 1e-8
    assert float(np.max(np.abs(np.array(env.sim.fsm.r_des, float) - r))) < 1e-8
    assert float(np.max(np.abs(np.array(env.sim.fsm.v_cmd, float) - vc))) < 1e-8
    assert abs(float(env.sim.data.ctrl[7]) - c7) < 1e-8


def test_no_d_authority(cfg, snap):
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d")
    env.reset(options={"snapshot": snap})
    env.sim.meter.D_enter = -1.0
    env.sim.meter.D_exit = 1e9
    env.sim.meter.N_stable = 1
    # Must not succeed immediately from D_exit; ZERO hold is not centered.
    terms = []
    for _ in range(12):
        _, _, term, trunc, info = env.step(np.zeros(4, np.float32))
        terms.append(info["term"])
        if term or trunc:
            break
    assert "recovered" not in terms
    assert env._ok_s < 0.20 or abs(float(info["e_x"])) > 0.003


def smoke_authority(cfg, snap):
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d")
    env.reset(options={"snapshot": snap})
    Rh0 = np.array(env.sim.data.xmat[env.sim.ids.hand_body], float).reshape(3, 3)
    g0 = float((Rh0.T @ np.array([0.0, 0.0, -9.81]))[0])
    rows = {}
    env.reset(options={"snapshot": snap})
    for _ in range(8):
        env.step(np.array([0.0, 0.0, 0.0, 1.0], np.float32))
    rows["secure_tau"] = float(env.sim.data.ctrl[7])
    env.reset(options={"snapshot": snap})
    for _ in range(25):
        env.step(np.array([1.0, 0.0, 0.0, 1.0], np.float32))
    g1 = float(
        (np.array(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float).T @ np.array([0.0, 0.0, -9.81]))[0]
    )
    rows["g_hx_before"] = g0
    rows["g_hx_after_wrist"] = g1
    rows["g_hx_changed"] = abs(g1 - g0) > 2.0
    a_slip = 2.0 * ((-2.0) - RECOVERY4D_TAU_OPEN) / (RECOVERY4D_TAU_SECURE - RECOVERY4D_TAU_OPEN) - 1.0
    env.reset(options={"snapshot": snap})
    for _ in range(10):
        env.step(np.array([0.0, 0.0, 0.0, a_slip], np.float32))
    rows["slip_tau"] = float(env.sim.data.ctrl[7])
    env.reset(options={"snapshot": snap})
    for _ in range(8):
        env.step(np.array([0.0, 0.0, 0.0, 1.0], np.float32))
    rows["brake_tau"] = float(env.sim.data.ctrl[7])
    env.reset(options={"snapshot": snap})
    p0 = np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    R0 = np.array(env.sim.data.xmat[env.sim.ids.hand_body].reshape(3, 3), float)
    for _ in range(12):
        env.step(np.array([0.0, 1.0, 0.0, -1.0], np.float32))
    p1 = np.array(env.sim.data.xpos[env.sim.ids.hand_body], float)
    rows["open_tau"] = float(env.sim.data.ctrl[7])
    rows["open_hx"] = float((R0.T @ (p1 - p0))[0])
    env.reset(options={"snapshot": snap})
    for _ in range(8):
        env.step(np.array([0.0, 0.0, 0.0, -1.0], np.float32))
    for _ in range(8):
        env.step(np.array([0.0, 0.0, 0.0, 1.0], np.float32))
    rows["reclose_tau"] = float(env.sim.data.ctrl[7])
    assert abs(rows["secure_tau"] + 18.0) < 1e-5
    assert rows["g_hx_changed"]
    assert abs(rows["slip_tau"] + 2.0) < 1e-4
    assert abs(rows["brake_tau"] + 18.0) < 1e-5
    assert abs(rows["open_tau"] - RECOVERY4D_TAU_OPEN) < 1e-5
    assert rows["open_hx"] > 0.001
    assert abs(rows["reclose_tau"] + 18.0) < 1e-5
    return rows


def test_obs_sat_n20():
    buf = ROOT / "results" / "buffers" / "recovery_train_smoke.npz"
    assert buf.is_file(), buf
    snaps = load_buffer(buf)
    assert len(snaps) == 20
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    env = RecoveryEnv(cfg=cfg, snapshots=snaps, action_kind="recovery4d")
    names = [
        "rh_x/e_x", "rh_y", "rh_z-nom",
        "vrel_hx", "vrel_hy", "vrel_hz",
        "Rrel0", "Rrel1", "Rrel2", "Rrel3", "Rrel4", "Rrel5",
        "wrel_hx", "wrel_hy", "wrel_hz",
        "gh_x", "gh_y", "gh_z",
        "aperture", "tau_from_secure", "FnL", "FnR", "obj_z_off", "clear",
    ]
    raws = []
    nrms = []
    for s in snaps:
        env.reset(options={"snapshot": s})
        p = physical_pack(env.sim)
        rh = np.asarray(p["rh"], float)
        raw = np.concatenate([
            [rh[0] / SCALE_EX, rh[1] / SCALE_EY, (rh[2] - RH_Z_NOM) / SCALE_RH_Z],
            np.asarray(p["v_rel_h"], float) / SCALE_V,
            mat6(p["R_rel"]),
            np.asarray(p["w_rel_h"], float) / SCALE_W,
            np.asarray(p["g_h"], float) / SCALE_G,
            [p["aperture"] / SCALE_AP],
            [(float(p["tau"]) - TAU_SECURE) / SCALE_TAU],
            [float(p["Fn_L"]) / SCALE_F, float(p["Fn_R"]) / SCALE_F],
            [(float(p["obj_z"]) - TABLE_TOP) / SCALE_Z],
            [float(p["clear"]) / SCALE_CLEAR],
        ])
        raws.append(raw)
        nrms.append(observe_recovery4d(env.sim))
    raws = np.vstack(raws)
    nrms = np.vstack(nrms)
    print("OBS_SAT N=20  SCALE_EX", SCALE_EX, "RH_Z_NOM", RH_Z_NOM, "SCALE_RH_Z", SCALE_RH_Z)
    print(f"{'i':>2} {'name':16s} {'raw_min':>12} {'raw_max':>12} {'nmin':>8} {'nmax':>8} {'c-1':>6} {'c+1':>6}")
    for i, name in enumerate(names):
        lo = float(np.mean(nrms[:, i] <= -1.0 + 1e-9))
        hi = float(np.mean(nrms[:, i] >= 1.0 - 1e-9))
        print(
            f"{i:2d} {name:16s} {raws[:, i].min():12.6f} {raws[:, i].max():12.6f} "
            f"{nrms[:, i].min():8.4f} {nrms[:, i].max():8.4f} {lo:6.2f} {hi:6.2f}"
        )
        assert lo < 1.0 and hi < 1.0, (name, lo, hi)
    assert abs(float(snaps[0]["e_x"])) / SCALE_EX < 1.0
    assert V_REL_TOL == 0.02
    assert W_REL_TOL == 0.50


def test_action_penalty_no_a3(cfg, snap):
    env = RecoveryEnv(cfg=cfg, snapshots=[snap], action_kind="recovery4d")
    env.reset(options={"snapshot": snap})
    o = physical_pack(env.sim)
    r_sec = env._reward(o, np.array([0.0, 0.0, 0.0, 1.0]), None)
    r_mid = env._reward(o, np.array([0.0, 0.0, 0.0, 0.0]), None)
    r_open = env._reward(o, np.array([0.0, 0.0, 0.0, -1.0]), None)
    assert abs(r_sec - r_mid) < 1e-12
    assert abs(r_open - r_mid) < 1e-12
    r_move = env._reward(o, np.array([1.0, 0.0, 0.0, 1.0]), None)
    assert r_move < r_sec - 0.005


def test_impact_ball_park_order():
    from training.replay_core import (
        BALL_PARK_POS,
        ball_is_parked,
        load_impact_ic,
        make_impact_sim,
        prepare_replay,
        restore_replay,
        rollout_from_replay,
        params_from_frozen,
    )
    from controllers.jacobian_controller import gains_from_cfg
    cfg = merge_sim_config(load_yaml(ROOT / "config" / "nominal.yaml"))
    sim = make_impact_sim(cfg)
    ic = load_impact_ic(8.5)
    restore_replay(sim, ic)
    q = int(sim.ball_qadr)
    flying = np.array(sim.data.qpos[q : q + 3], float)
    assert float(np.linalg.norm(flying - BALL_PARK_POS)) > 0.1
    prepare_replay(sim, ic)
    assert ball_is_parked(sim)
    gains = gains_from_cfg(cfg)
    params, _ = params_from_frozen()
    _, z = rollout_from_replay(sim, ic, gains, params, True, 0.2, restore=False)
    assert ball_is_parked(sim)
    assert z["grasp_retention"] in (0, 1)


def main() -> int:
    test_frozen_and_eval_untouched()
    h0 = hashlib.sha256(OFFSET_NPZ.read_bytes()).hexdigest()
    h_imp0 = hashlib.sha256(IMPACT_NPZ.read_bytes()).hexdigest()
    cfg, snap = _ic()
    test_zero_no_hidden_lift(cfg, snap)
    test_hand_x_signs(cfg, snap)
    test_wrist_hand_y(cfg, snap)
    test_grip_mapping_and_no_fg_min()
    test_grip_env_ctrl(cfg, snap)
    test_snapshot_roundtrip(cfg, snap)
    test_no_d_authority(cfg, snap)
    test_action_penalty_no_a3(cfg, snap)
    test_obs_sat_n20()
    test_impact_ball_park_order()
    smoke = smoke_authority(cfg, snap)
    print("SMOKE", smoke)
    h1 = hashlib.sha256(OFFSET_NPZ.read_bytes()).hexdigest()
    h_imp1 = hashlib.sha256(IMPACT_NPZ.read_bytes()).hexdigest()
    assert h0 == h1 and h_imp0 == h_imp1
    print("INTERFACE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
