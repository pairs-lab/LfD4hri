import unittest
import numpy as np

from VLA.central_section_clearance_teacher import central_section_options
from VLA.ray_refined_clearance_teacher import RayRefinedClearanceTeacher


class CentralSectionClearanceTests(unittest.TestCase):
    def mesh(self, width=.04):
        return np.array([[x, y, z] for y in np.linspace(-.71, -.55, 31)
                         for x in (-width/2, width/2) for z in (.005, .045)])

    def test_five_central_body_sections_retain_physical_width(self):
        rows = central_section_options(self.mesh(), .088595606)
        self.assertEqual([r["central_section_quantile"] for r in rows], [.5, .4, .6, .3, .7])
        self.assertTrue(all(r["full_local_section_support_m"] < .0846 for r in rows))
        self.assertTrue(all(.012 <= r["cross_section_width_m"] <= .077 for r in rows))
        self.assertTrue(all(abs(r["jaw_axis_world"][0]) == 1 for r in rows))

    def test_wide_or_round_footprints_not_given_fake_section(self):
        self.assertEqual(central_section_options(self.mesh(.10), .088595606), [])
        cube = np.array([[x, y, z] for x in (-.02, .02) for y in (-.65, -.61)
                         for z in np.linspace(.005, .045, 3)])
        self.assertEqual(central_section_options(cube, .088595606), [])

    def test_bad_geometry_rejected(self):
        for mesh, gap in ((self.mesh()*np.nan, .0886), (self.mesh(), 0.), (np.ones((5, 3)), .0886)):
            with self.assertRaises(ValueError):
                central_section_options(mesh, gap)

    def test_original_teacher_extension_still_rejects_by_default(self):
        teacher = object.__new__(RayRefinedClearanceTeacher)
        self.assertIsNone(teacher._alternate_clearance_plan(None, None, None, None, None, None))


if __name__ == "__main__":
    unittest.main()
