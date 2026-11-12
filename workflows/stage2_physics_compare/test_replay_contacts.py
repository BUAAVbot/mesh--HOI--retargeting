import unittest
import mujoco as mj
import numpy as np
from replay_validate import contact_loads


class ReplayContactTests(unittest.TestCase):
    def test_world_vertical_support_is_positive(self):
        m=mj.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="2 2 .1"/>'
            '<body pos="0 0 .099"><freejoint/><geom type="box" size=".1 .1 .1" mass="10"/></body>'
            '</worldbody></mujoco>')
        d=mj.MjData(m);mj.mj_forward(m,d)
        depth,force=contact_loads(m,d,{1},[1])
        self.assertEqual(depth,0.)
        self.assertGreater(force[0,2],90.)
        np.testing.assert_allclose(force[0],d.qfrc_constraint[:3],atol=1e-10)

    def test_self_contact_is_not_support(self):
        m=mj.MjModel.from_xml_string('<mujoco><worldbody>'
            '<body pos="0 0 .2"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>'
            '<body pos=".19 0 .2"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>'
            '</worldbody></mujoco>')
        d=mj.MjData(m);mj.mj_forward(m,d)
        depth,force=contact_loads(m,d,{1,2},[1,2])
        self.assertGreater(depth,.009)
        np.testing.assert_array_equal(force,np.zeros((2,3)))


if __name__=='__main__':unittest.main()
