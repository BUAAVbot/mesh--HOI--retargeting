import unittest
import numpy as np
from scipy import sparse
from repair_n21_contacts import support_height,collision_residual


class N21SupportTests(unittest.TestCase):
    def test_collision_residual_targets_recovery_without_hard_conflicts(self):
        g=sparse.csr_matrix([[1.,0.],[-1.,0.],[0.,1.]])
        a,b=collision_residual(g,np.array([.02,.01,-.003]))
        np.testing.assert_allclose(a.toarray(),[[-1000.,0.],[1000.,0.]])
        np.testing.assert_allclose(b,[-20.,-10.])
        step=np.linalg.lstsq(a.toarray(),b,rcond=None)[0]
        self.assertLess(np.linalg.norm(a@step-b),np.linalg.norm(b))

    def test_keeps_steps_and_unknown_space(self):
        triangles=np.array([[[0,0,1],[1,0,1],[0,1,1]],[[1,0,.9],[2,0,.9],[1,1,.9]]],float)
        heights=support_height(np.array([[.2,.2],[1.2,.2],[3.,3.],[1.,0.]]),triangles)
        np.testing.assert_allclose(heights[[0,1,3]],[1.,.9,1.])
        self.assertTrue(np.isnan(heights[2]))


if __name__=='__main__':unittest.main()
