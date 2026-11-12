"""Free-root stance inverse statics via genuine actuator torques only."""
import numpy as np
import mujoco as mj


class GroundReactionPD:
    def __init__(self, model, initial_qpos, actuator_ids, qadr, vadr):
        self.m = model
        self.q0 = np.asarray(initial_qpos).copy()
        self.aids = np.asarray(actuator_ids)
        self.qa = np.asarray(qadr)
        self.va = np.asarray(vadr)
        self.kin = mj.MjData(model)
        self.fullM = np.zeros((model.nv, model.nv))
        self.points = []
        for geom in range(model.ngeom):
            name = model.geom(geom).name or ''
            if not name.startswith(('collision_foot_l_capsule_', 'collision_foot_r_capsule_')):
                continue
            if model.geom_type[geom] != mj.mjtGeom.mjGEOM_CAPSULE:
                raise ValueError('Ground support contract expects original shoe capsules')
            self.points.extend([(geom, -1.), (geom, 1.)])
        if len(self.points) != 28:
            raise ValueError('Expected two endpoints on each of original fourteen shoe capsules')
        self.rays = np.array([[.5, 0., 1.], [-.5, 0., 1.], [0., .5, 1.], [0., -.5, 1.]]).T
        self.next_update = -1.
        self.feedforward = np.zeros(len(self.aids))
        self.logs = []

    def action(self, data, elapsed):
        m = self.m
        if data.time+1e-9 >= self.next_update:
            kin = self.kin
            kin.qpos[:] = data.qpos
            kin.qvel[:] = 0.
            mj.mj_forward(m, kin)
            mj.mj_fullM(m, self.fullM, kin.qM)
            error = np.zeros(m.nv)
            mj.mj_differentiatePos(m, error, 1., data.qpos, self.q0)
            accel = np.zeros(m.nv)
            accel[:3] = np.clip(np.array([35., 35., 70.])*error[:3]-np.array([10., 10., 16.])*data.qvel[:3], -2., 2.)
            accel[3:6] = np.clip(60.*error[3:6]-12.*data.qvel[3:6], -6., 6.)
            columns = []
            jp = np.zeros((3, m.nv))
            for geom, sign in self.points:
                G = kin.geom_xmat[geom].reshape(3, 3)
                center = kin.geom_xpos[geom]
                point = center+sign*m.geom_size[geom, 1]*G[:, 2]-np.array([0., 0., m.geom_size[geom, 0]])
                mj.mj_jac(m, kin, jp, None, point, int(m.geom_bodyid[geom]))
                columns.extend((jp.T@self.rays).T)
            B = np.asarray(columns).T
            demand = kin.qfrc_bias+self.fullM@accel
            A = B[:6]
            b = demand[:6]
            available = np.ones(B.shape[1], bool)
            weights = np.zeros(B.shape[1])
            for _ in range(B.shape[1]):
                indices = np.flatnonzero(available)
                if len(indices) < 6:
                    break
                active = A[:, indices]
                solution = active.T@np.linalg.solve(active@active.T+1e-10*np.eye(6), b)
                bad = solution < -1e-8
                if not np.any(bad):
                    weights[indices] = np.maximum(solution, 0.)
                    break
                available[indices[bad]] = False
            residual = A@weights-b
            torque = demand[self.va]-B[self.va]@weights
            limits = m.actuator_forcerange[self.aids]
            self.feedforward = np.clip(torque, limits[:, 0]*.95, limits[:, 1]*.95)
            self.logs.append({'time_s': float(data.time),
                              'root_wrench_residual_max': float(np.max(abs(residual))),
                              'minimum_nonnegative_ray_weight_N': float(weights.min()),
                              'predicted_ground_vertical_force_N': float(weights.sum()),
                              'native_torque_goal_exceeded': bool(np.any(abs(torque-self.feedforward) > 1e-6)),
                              'root_position_error_m': error[:3].tolist(),
                              'base_acceleration_goal': accel[:6].tolist(),
                              'feedforward_joint_torques_Nm': torque.tolist()})
            self.next_update = float(data.time)+.02
        ctrl = np.zeros(m.nu)
        ctrl[self.aids] = self.q0[self.qa]+self.feedforward/m.actuator_gainprm[self.aids, 0]
        return ctrl

    def report(self):
        return {'method': 'PD plus floating-root ground-reaction-aware inverse statics; no learned policy, base weld, applied force, or hand geometry change.',
                'ground_cone': 'Four friction-pyramid rays, mu=.5, on original shoe capsule endpoints; force variables are a control prediction, never external applied forces.',
                'maximum_root_wrench_residual': max((r['root_wrench_residual_max'] for r in self.logs), default=0.),
                'records': self.logs}
