"""Fail if privileged GT is passed into the tactile sensor / e_x estimator API."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_NAMES = {
    "e_x",
    "e_x_GT",
    "EVAL_ONLY_GT_e_x",
    "object_qpos",
    "object_qvel",
    "object_pose",
}

FORBIDDEN_PARAM_SUBSTR = (
    "e_x",
    "object_qpos",
    "object_qvel",
    "object_pose",
    "privileged",
)


def _py_files():
    return [
        ROOT / "sensors" / "spatial_tactile.py",
        ROOT / "sensors" / "ex_estimator.py",
    ]


def test_no_forbidden_api_parameters():
    from sensors.ex_estimator import geometric_from_reading
    from sensors.spatial_tactile import measure_spatial_tactile

    for fn in (measure_spatial_tactile, geometric_from_reading):
        names = list(inspect.signature(fn).parameters)
        for n in names:
            low = n.lower()
            for s in FORBIDDEN_PARAM_SUBSTR:
                assert s not in low, f"{fn.__name__} has forbidden param {n}"


def test_measure_return_keys_allowlist():
    from sensors.spatial_tactile import SENSOR_KEYS, SpatialTactileReading, FingerTactile

    r = SpatialTactileReading(
        left=FingerTactile(False, False, float("nan"), float("nan"), 0.0, 0),
        right=FingerTactile(False, False, float("nan"), float("nan"), 0.0, 0),
    )
    d = r.to_array()
    assert set(d.keys()) == set(SENSOR_KEYS)
    for k in FORBIDDEN_NAMES:
        assert k not in d


def test_ast_sensor_estimator_do_not_read_object_pose():
    """Contact filtering may mention object_body; pose arrays must not be indexed by it."""
    forbidden_attr_chains = {
        ("ids", "object_body"),  # allowed only in compare for geom filter — checked separately
    }
    pose_reads = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        src = path.read_text(encoding="utf-8")
        # Hard fail if GT labels appear as identifiers in these files.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                pose_reads.append(f"{path.name}: Name {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
                pose_reads.append(f"{path.name}: attr {node.attr}")
        # data.qpos / data.qvel / data.xpos[ids.object_body]
        if "data.qpos" in src or "data.qvel" in src:
            pose_reads.append(f"{path.name}: data.qpos/qvel")
        if "xpos[ids.object" in src.replace(" ", "") or "xmat[ids.object" in src.replace(" ", ""):
            pose_reads.append(f"{path.name}: object xpos/xmat")
        if "physical_pack" in src or "obs_from_sim" in src:
            pose_reads.append(f"{path.name}: privileged obs helper")
    assert pose_reads == [], pose_reads


def test_estimator_rejects_extra_gt_kwargs():
    from sensors.ex_estimator import geometric_from_reading
    from sensors.spatial_tactile import FingerTactile, SpatialTactileReading

    r = SpatialTactileReading(
        left=FingerTactile(True, True, 0.002, 0.0, 1.0, 1),
        right=FingerTactile(True, True, -0.002, 0.0, 1.0, 1),
    )
    with pytest.raises(TypeError):
        geometric_from_reading(r, e_x_GT=0.002)  # type: ignore[call-arg]
