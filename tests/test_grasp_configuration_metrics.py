"""Unit tests for grasp-configuration metrics (no MuJoCo required)."""

import math

import numpy as np

from evaluation.grasp_geometry import (
    calibrate_from_samples,
    orientation_error_rad,
    position_error_m,
    success_new_components,
)


def test_axis_symmetric_orientation():
    u = np.array([0.0, 0.0, 1.0])
    assert orientation_error_rad(u, u, True) < 1e-9
    assert orientation_error_rad(u, -u, True) < 1e-9
    assert orientation_error_rad(u, -u, False) > math.pi / 2 - 1e-6


def test_orthogonal_is_90deg():
    e = orientation_error_rad(np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), True)
    assert abs(e - math.pi / 2) < 1e-6


def test_position_error():
    assert abs(position_error_m([0, 0, 0.1], [0, 0, 0.1])) < 1e-12
    assert abs(position_error_m([0, 0, 0], [0, 0, 0.01]) - 0.01) < 1e-12


def test_calibrate_and_success_new():
    us = [np.array([0.0, 0.0, 1.0]), np.array([0.01, 0.0, 0.99995]), np.array([0.0, 0.0, -1.0])]
    ps = [np.array([0.0, 0.0, 0.099]), np.array([0.001, 0.0, 0.098]), np.array([0.0, -0.001, 0.100])]
    cal = calibrate_from_samples(us, ps)
    assert cal["n_samples"] == 3
    assert cal["tol_theta_rad_proposed"] >= 0
    assert cal["tol_pos_m_proposed"] >= 0
    c = success_new_components(
        True, 0.0, 0.0, cal["tol_theta_rad_proposed"], cal["tol_pos_m_proposed"],
        pose_ok_duration_s=1.0, pos_ok_duration_s=1.0, hold_required_s=0.5,
    )
    assert c["SUCCESS_NEW"] is True
    c2 = success_new_components(
        True, 1.0, 0.0, 0.1, 0.01,
        pose_ok_duration_s=1.0, pos_ok_duration_s=1.0, hold_required_s=0.5,
    )
    assert c2["SUCCESS_NEW"] is False
    assert c2["SUCCESS_12S"] is True
