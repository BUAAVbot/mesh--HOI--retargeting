import unittest
from common import physics_substeps


class PhysicsClockTests(unittest.TestCase):
    def test_legacy_gallery_and_n21_clocks(self):
        self.assertEqual(physics_substeps(1/50,.005),4)
        self.assertEqual(physics_substeps(1/30,1/300),10)
        self.assertEqual(208*physics_substeps(1/30,1/300),2080)

    def test_rejects_silent_time_warp(self):
        for source,sim in [(1/30,.005),(0,.005),(.02,float('nan')),(.02,-1)]:
            with self.assertRaises(ValueError):physics_substeps(source,sim)


if __name__=='__main__':unittest.main()
