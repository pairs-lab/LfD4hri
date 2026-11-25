from __future__ import annotations

import unittest

from VLA.common.spatial import FIXED_DESTINATION


class SpatialSemanticsTest(unittest.TestCase):
    def test_named_workspace_regions_are_unique(self) -> None:
        self.assertEqual(len(set(FIXED_DESTINATION.values())), len(FIXED_DESTINATION))

    def test_right_side_is_not_bottom_right_corner(self) -> None:
        self.assertEqual(FIXED_DESTINATION["right side"], (0.25, -0.5, 0.0))
        self.assertEqual(FIXED_DESTINATION["bottom right corner"], (0.25, -0.75, 0.0))
        self.assertNotEqual(
            FIXED_DESTINATION["right side"], FIXED_DESTINATION["bottom right corner"]
        )


if __name__ == "__main__":
    unittest.main()
