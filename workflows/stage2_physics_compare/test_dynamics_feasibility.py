"""Regression checks for force demand and finite-area support."""
import unittest

import mujoco as mj
import numpy as np

from audit_dynamics_feasibility import contact_rays, fit_forces, required_force


class DynamicsAuditTests(unittest.TestCase):
    def setUp(self):
        self.model = mj.MjModel.from_xml_string('''<mujoco><option timestep=".002"/>
          <worldbody><geom type="plane" size="2 2 .1"/>
          <body pos="0 0 .099"><freejoint/><geom type="box" size=".2 .1 .1" mass="10"/>
          </body></worldbody></mujoco>''')
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)

    def test_demand_does_not_subtract_contact(self):
        m, d = self.model, self.data
        acceleration = np.array([.3, -.1, .2, .1, .2, .3])
        demand = required_force(m, d, acceleration)
        d.qacc[:] = acceleration
        mj.mj_inverse(m, d)
        np.testing.assert_allclose(demand, d.qfrc_inverse + d.qfrc_constraint, atol=1e-10)
        self.assertGreater(np.linalg.norm(d.qfrc_constraint), 1.)
        self.assertFalse(np.allclose(demand, d.qfrc_inverse))

    def test_patch_balances_pitch_moment_single_point_cannot(self):
        m, d = self.model, self.data
        demand = required_force(m, d, np.zeros(m.nv))
        demand[4] = 5.
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        columns = []
        for contact in d.contact:
            mj.mj_jac(m, d, jacp, jacr, contact.pos, 1)
            columns.append(jacp.T @ contact_rays(contact.frame.reshape(3, 3), [.8, .8]))
        _, residual = fit_forces(demand, np.zeros((6, 0)), np.zeros(0), columns, np.ones(6))
        self.assertLess(np.linalg.norm(residual), 1e-8)
        mj.mj_jac(m, d, jacp, jacr, [0, 0, 0], 1)
        _, residual = fit_forces(demand, np.zeros((6, 0)), np.zeros(0),
                                 [jacp.T @ contact_rays(np.eye(3)[[2, 0, 1]], [.8, .8])], np.ones(6))
        self.assertGreater(np.linalg.norm(residual), 4.)

    def test_no_support_cannot_balance_gravity(self):
        demand = required_force(self.model, self.data, np.zeros(6))
        _, residual = fit_forces(demand, np.zeros((6, 0)), np.zeros(0), [], np.ones(6))
        self.assertGreater(residual[2], 98.)

    def test_force_fit_enforces_motor_limit(self):
        force, residual = fit_forces(np.array([3.]), np.ones((1, 1)),
                                    np.array([1.]), [], np.ones(1))
        np.testing.assert_allclose(force, [1.])
        np.testing.assert_allclose(residual, [2.])

    def test_friction_cone_rejects_excessive_tangential_force(self):
        rays = contact_rays(np.eye(3)[[2, 0, 1]], [.8, .8])
        _, residual = fit_forces(np.array([100., 0., 98.1]), np.zeros((3, 0)),
                                 np.zeros(0), [rays], np.ones(3))
        self.assertGreater(np.linalg.norm(residual), 10.)


if __name__ == '__main__':
    unittest.main()
