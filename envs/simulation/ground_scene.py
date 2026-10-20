"""GROUND_SCENE: Panda + cylinder + impact ball on a collidable ground plane.

Replaces the elevated table with the physical ground plane at world z = 0.
Robot base (link0) remains at the origin. Cylinder mass/inertia/friction and
the diagnostic ball match the ballistic elevated-table scene.

This module writes ``assets/scene_ground.xml``. Elevated-table scenes remain
available via ``assets/scene.xml`` / ``scene_ballistic_impact.xml``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "assets" / "scene_ground.xml"

SCENE_ID = "GROUND_SCENE"

# Support plane (MuJoCo infinite plane through the origin).
GROUND_Z = 0.0

# Cylinder: identical to ballistic elevated-table scene.
CYL_M = 0.20
CYL_R = 0.018
CYL_H = 0.06
CYL_HALF = 0.03
CYL_IZZ = 0.5 * CYL_M * CYL_R * CYL_R
CYL_IXX = (1.0 / 12.0) * CYL_M * (3.0 * CYL_R * CYL_R + CYL_H * CYL_H)

# Resting COM height when the cylinder stands on the ground.
CYL_REST_Z = GROUND_Z + CYL_HALF

# Ball: identical to ballistic scene.
BALL_M = 0.05
BALL_R = 0.015
BALL_I = 0.4 * BALL_M * BALL_R * BALL_R

# Geometric height guard used only as a secondary abort after GROUND_DROP is armed.
# Not the old table top (0.40) + margin. Half-height + 1 cm numerical margin.
GROUND_DROP_Z = CYL_REST_Z + 0.01  # 0.04 m

# Default keyframe places the cylinder resting on the ground for scene setup.
# Airborne end-grasps are installed from snapshots; they are not required to
# be produced by lifting from this rest pose.
HOME_OBJ_XY = (0.5, 0.0)
HOME_BALL = (0.50, -0.40, 0.55)

XML = f"""<mujoco model="panda GROUND_SCENE noslip1">
  <include file="panda_torque.xml"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="0 0 0.05" pos="0 0 {GROUND_Z}"
      material="groundplane" friction="1.0 0.005 0.0001"/>
    <body name="object" pos="{HOME_OBJ_XY[0]} {HOME_OBJ_XY[1]} {CYL_REST_Z}">
      <freejoint name="object_joint"/>
      <inertial mass="{CYL_M}" pos="0 0 0" diaginertia="{CYL_IXX:.8e} {CYL_IXX:.8e} {CYL_IZZ:.8e}"/>
      <geom name="object" type="cylinder" size="{CYL_R} {CYL_HALF}" condim="4"
        friction="1.0 0.005 0.0001" rgba="0.15 0.75 0.35 1" solref="0.01 1"
        contype="3" conaffinity="3"/>
    </body>
    <body name="impact_ball" pos="{HOME_BALL[0]} {HOME_BALL[1]} {HOME_BALL[2]}">
      <freejoint name="ball_joint"/>
      <inertial mass="{BALL_M}" pos="0 0 0" diaginertia="{BALL_I:.8e} {BALL_I:.8e} {BALL_I:.8e}"/>
      <geom name="impact_ball" type="sphere" size="{BALL_R}" condim="3"
        friction="0.40 0.005 0.0001" rgba="0.95 0.35 0.08 1" solref="0.004 1"
        contype="2" conaffinity="2"/>
    </body>
    <body name="ball_guide" mocap="true" pos="{HOME_BALL[0]} {HOME_BALL[1]} {HOME_BALL[2]}">
      <geom name="ball_guide_geom" type="sphere" size="0.010" contype="0" conaffinity="0"
        group="3" rgba="0.15 0.40 0.95 0.08"/>
    </body>
  </worldbody>

  <equality>
    <weld name="ball_guide_weld" body1="impact_ball" body2="ball_guide"
      solimp="0.99 0.999 0.0001" solref="0.004 1"/>
  </equality>

  <keyframe>
    <key name="home"
      qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04  {HOME_OBJ_XY[0]} {HOME_OBJ_XY[1]} {CYL_REST_Z}  1 0 0 0  {HOME_BALL[0]} {HOME_BALL[1]} {HOME_BALL[2]}  1 0 0 0"
      ctrl="0 0 0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def write_ground_scene() -> Path:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(XML, encoding="utf-8")
    return DEST


if __name__ == "__main__":
    print(write_ground_scene())
