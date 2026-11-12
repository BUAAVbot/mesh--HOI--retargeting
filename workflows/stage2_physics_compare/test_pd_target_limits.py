import unittest
import mujoco as mj
import numpy as np
from common import configure_position_pd


class PDTargetLimitsTests(unittest.TestCase):
    def test_declared_servo_target_range_survives_reconfiguration(self):
        m=mj.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
          <body><joint name="left_ankle_pitch_joint" range="-.5 .5"/>
          <geom type="sphere" size=".1" mass="1"/></body></worldbody>
          <actuator><motor joint="left_ankle_pitch_joint"/></actuator></mujoco>''')
        configure_position_pd(m,mj)
        np.testing.assert_array_equal(m.actuator_ctrlrange[0],[-.5,.5])
        original_joints=m.jnt_range.copy();original_caps=m.actuator_forcerange.copy()
        m.actuator_ctrlrange[0]=[-4.5,4.5]
        configure_position_pd(m,mj)
        np.testing.assert_array_equal(m.actuator_ctrlrange[0],[-4.5,4.5])
        np.testing.assert_array_equal(m.jnt_range,original_joints)
        np.testing.assert_array_equal(m.actuator_forcerange,original_caps)
        d=mj.MjData(m);d.qpos[0]=-.4;d.ctrl[0]=1.2;mj.mj_forward(m,d)
        self.assertAlmostEqual(d.actuator_force[0],64.)
        d.ctrl[0]=4.5;mj.mj_forward(m,d)
        self.assertAlmostEqual(d.actuator_force[0],141.)


if __name__=='__main__':unittest.main()
