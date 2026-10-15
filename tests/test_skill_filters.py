"""Regression tests for recovery acceptance. No simulator."""
from __future__ import annotations

import unittest

from data_generation.skill_filters import (
    admit_horizontal_search,
    blocks_task_success,
    observed_mechanism,
    release_and_regrip,
    scene_category,
    task_success,
)


class SkillFilterTests(unittest.TestCase):
    def test_success_without_slip_is_kept(self):
        self.assertTrue(task_success(False, True, True))
        self.assertEqual(observed_mechanism("no_in_hand_slip"), "stabilization_without_slip")
        self.assertEqual(observed_mechanism("weak_net_correction"), "stabilization_without_slip")
        self.assertFalse(blocks_task_success("no_in_hand_slip"))
        self.assertFalse(blocks_task_success("weak_net_correction"))

    def test_tilted_horizontal_scene_stays_horizontal(self):
        self.assertTrue(admit_horizontal_search("horizontal_cylinder", 60.0))
        self.assertEqual(scene_category("horizontal_cylinder", 60.0), "horizontal_recatch")
        self.assertEqual(scene_category("horizontal_cylinder", 80.0), "horizontal_recatch")

    def test_later_rotation_does_not_swap_scenes(self):
        self.assertEqual(scene_category("vertical_cylinder", 5.0), "vertical_recatch")
        self.assertEqual(scene_category("horizontal_cylinder", 85.0), "horizontal_recatch")
        self.assertNotEqual(
            scene_category("horizontal_cylinder", 85.0),
            scene_category("vertical_cylinder", 85.0),
        )

    def test_nominal_solvable_is_excluded(self):
        self.assertFalse(task_success(True, True, True))

    def test_single_step_contact_change_is_not_recatch(self):
        n_left = [2, 0, 2, 2]
        n_right = [2, 0, 2, 2]
        self.assertFalse(release_and_regrip(n_left, n_right))
        self.assertTrue(release_and_regrip([2, 0, 0, 2, 2], [2, 0, 0, 2, 2]))
        self.assertFalse(release_and_regrip([2, 2, 2, 2], [2, 2, 2, 2]))

    def test_mechanism_does_not_make_a_failure_succeed(self):
        self.assertFalse(task_success(False, True, False))
        self.assertEqual(observed_mechanism("confirmed"), "in_hand_slip")
        self.assertFalse(task_success(False, False, True))


if __name__ == "__main__":
    unittest.main()
