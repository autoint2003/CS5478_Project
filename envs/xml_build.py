"""Build a torque-actuated Panda MJCF without editing MuJoCo Menagerie."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_PANDA = ROOT / "mujoco_menagerie" / "franka_emika_panda" / "panda.xml"
ASSETS = ROOT / "assets"


def write_panda_torque(dest: Path | None = None) -> Path:
    dest = dest or (ASSETS / "panda_torque.xml")
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = MENAGERIE_PANDA.read_text(encoding="utf-8")

    # panda_torque.xml lives in assets/; meshes stay in Menagerie.
    text = text.replace(
        'meshdir="assets"',
        'meshdir="../mujoco_menagerie/franka_emika_panda/assets"',
        1,
    )
    text = text.replace(
        '<option integrator="implicitfast"/>',
        '<option integrator="implicitfast" timestep="0.002" gravity="0 0 -9.81"/>',
        1,
    )

    left_site = """
                        <site name="touch_left" pos="0 0.006 0.045" size="0.01" group="4" rgba="1 0 0 0.35"/>"""
    right_site = """
                        <site name="touch_right" pos="0 0.006 0.045" size="0.01" group="4" rgba="1 0 0 0.35"/>"""
    text = text.replace(
        '<body name="left_finger" pos="0 0 0.0584">',
        '<body name="left_finger" pos="0 0 0.0584">' + left_site,
        1,
    )
    text = text.replace(
        '<body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">',
        '<body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">' + right_site,
        1,
    )

    actuator = """
  <actuator>
    <motor name="actuator1" joint="joint1" gear="1" ctrlrange="-87 87" forcerange="-87 87"/>
    <motor name="actuator2" joint="joint2" gear="1" ctrlrange="-87 87" forcerange="-87 87"/>
    <motor name="actuator3" joint="joint3" gear="1" ctrlrange="-87 87" forcerange="-87 87"/>
    <motor name="actuator4" joint="joint4" gear="1" ctrlrange="-87 87" forcerange="-87 87"/>
    <motor name="actuator5" joint="joint5" gear="1" ctrlrange="-12 12" forcerange="-12 12"/>
    <motor name="actuator6" joint="joint6" gear="1" ctrlrange="-12 12" forcerange="-12 12"/>
    <motor name="actuator7" joint="joint7" gear="1" ctrlrange="-12 12" forcerange="-12 12"/>
    <motor name="actuator8" tendon="split" gear="1" ctrlrange="-50 50" forcerange="-50 50"/>
  </actuator>
"""
    text = re.sub(
        r"  <actuator>.*?</actuator>\n",
        actuator,
        text,
        count=1,
        flags=re.S,
    )

    sensors = """
  <sensor>
    <touch name="touch_left" site="touch_left"/>
    <touch name="touch_right" site="touch_right"/>
  </sensor>
"""
    text = text.replace("  <keyframe>", sensors + "  <keyframe>", 1)
    # Scene.xml owns the 16-DoF home key (arm + object). Drop the 9-DoF
    # Menagerie key so include does not register a duplicate name="home".
    text = re.sub(
        r"\n  <keyframe>.*?</keyframe>\n",
        "\n",
        text,
        count=1,
        flags=re.S,
    )

    dest.write_text(text, encoding="utf-8")
    return dest
