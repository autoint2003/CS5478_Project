"""Diagnostic ballistic-impact scene. Not used by RecoveryEnv / official scene.xml.

Ball mass and cylinder mass are defined in XML before mjModel construction.
Ball collides with the cylinder only (contype/conaffinity bit 1).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "assets" / "scene_ballistic_impact.xml"

# Cylinder: m=0.20 kg, r=0.018, half-height 0.03.
# Izz = 1/2 m r^2; Ixx=Iyy = 1/12 m (3 r^2 + h^2), h=0.06.
CYL_M = 0.20
CYL_R = 0.018
CYL_H = 0.06
CYL_IZZ = 0.5 * CYL_M * CYL_R * CYL_R
CYL_IXX = (1.0 / 12.0) * CYL_M * (3.0 * CYL_R * CYL_R + CYL_H * CYL_H)

# Ball: m=0.05 kg (moderate diagnostic mass), r=0.015, I = 2/5 m r^2.
BALL_M = 0.05
BALL_R = 0.015
BALL_I = 0.4 * BALL_M * BALL_R * BALL_R

XML = f"""<mujoco model="panda ballistic impact diagnostic noslip1">
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
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    <geom name="table" type="box" size="0.12 0.12 0.02" pos="0.5 0.0 0.38"
      friction="1.0 0.005 0.0001" rgba="0.45 0.32 0.18 1"/>
    <body name="object" pos="0.5 0.0 0.43">
      <freejoint name="object_joint"/>
      <inertial mass="{CYL_M}" pos="0 0 0" diaginertia="{CYL_IXX:.8e} {CYL_IXX:.8e} {CYL_IZZ:.8e}"/>
      <geom name="object" type="cylinder" size="{CYL_R} 0.03" condim="4"
        friction="1.0 0.005 0.0001" rgba="0.15 0.75 0.35 1" solref="0.01 1"
        contype="3" conaffinity="3"/>
    </body>
    <body name="impact_ball" pos="0.50 -0.40 0.55">
      <freejoint name="ball_joint"/>
      <inertial mass="{BALL_M}" pos="0 0 0" diaginertia="{BALL_I:.8e} {BALL_I:.8e} {BALL_I:.8e}"/>
      <geom name="impact_ball" type="sphere" size="{BALL_R}" condim="3"
        friction="0.40 0.005 0.0001" rgba="0.95 0.35 0.08 1" solref="0.004 1"
        contype="2" conaffinity="2"/>
    </body>
    <body name="ball_guide" mocap="true" pos="0.50 -0.40 0.55">
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
      qpos="0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04  0.5 0 0.43  1 0 0 0  0.50 -0.40 0.55  1 0 0 0"
      ctrl="0 0 0 0 0 0 0 0"/>
  </keyframe>
</mujoco>
"""


def write_ballistic_scene() -> Path:
    DEST.write_text(XML, encoding="utf-8")
    return DEST


if __name__ == "__main__":
    print(write_ballistic_scene())
