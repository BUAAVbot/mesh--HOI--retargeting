import unittest
import mujoco as mj
import numpy as np
from prepare_n21_dynamics import canonical


class InitialVelocityTests(unittest.TestCase):
    def test_preserves_source_velocity_and_uses_it_in_fk(self):
        m=mj.MjModel.from_xml_string('''<mujoco><worldbody><geom type="plane" size="2 2 .1"/>
          <body name="base_link" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>
          <body name="left_ankle_roll_link" pos="0 .1 -.2"><geom name="foot_l_box" type="box" size=".1 .04 .02" mass=".1"/></body>
          <body name="right_ankle_roll_link" pos="0 -.1 -.2"><geom name="foot_r_box" type="box" size=".1 .04 .02" mass=".1"/></body>
          </body></worldbody></mujoco>''')
        q=np.tile(m.qpos0,(2,1));times=np.array([0.,1/30]);phase=np.full((2,2),'flat')
        velocity=np.zeros((2,m.nv));velocity[:,0]=.2;velocity[:,2]=-.03
        result=canonical(m,q,times,phase,velocities=velocity)
        np.testing.assert_array_equal(result['qvel_ref'],velocity)
        np.testing.assert_allclose(result['body_lin_vel_w'][0,:,0],.2)
        np.testing.assert_allclose(result['body_lin_vel_w'][0,:,2],-.03)
        legacy=canonical(m,q,times,phase)
        np.testing.assert_array_equal(legacy['qvel_ref'][0],np.zeros(m.nv))
        with self.assertRaises(ValueError):canonical(m,q,times,phase,velocities=velocity[:1])


if __name__=='__main__':unittest.main()
