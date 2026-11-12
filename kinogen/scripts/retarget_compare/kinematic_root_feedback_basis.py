"""Private nominal FK basis for real joint-target root-error feedback.

No state writes to a live plant; root motion is allocated to leg targets with
the nominal foot Jacobian. The six root-error channels are actual measurements.
"""
import numpy as np
import mujoco as mj


def rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rotation_error(goal, actual):
    if np.array_equal(goal, actual):
        return np.zeros(3)
    a, b = goal/np.linalg.norm(goal), actual/np.linalg.norm(actual)
    w = a[0]*b[0]+np.dot(a[1:], b[1:])
    xyz = -a[0]*b[1:]+b[0]*a[1:]-np.cross(a[1:], b[1:])
    if w < 0:
        w, xyz = -w, -xyz
    length = np.linalg.norm(xyz)
    return np.zeros(3) if length < 1e-15 else 2*np.arctan2(length, w)*xyz/length


class KinematicRootFeedback:
    def __init__(self, model, planq, planv, rootq, rootv, body_aids):
        self.planq, self.planv = planq, planv
        self.rq, self.rv = rootq, rootv
        self.legcols = []
        scratch = mj.MjData(model)
        names = [side+'_'+name+'_joint' for side in ('left', 'right')
                 for name in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll')]
        jointids, feet = [], []
        for prefix in ('p1_', 'p2_'):
            jids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, prefix+name) for name in names]
            aids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, prefix+'motor_'+name) for name in names]
            self.legcols.append(np.array([np.flatnonzero(body_aids == aid)[0] for aid in aids]))
            jointids.append(jids)
            feet.append([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, prefix+side+'_ankle_roll_link') for side in ('left', 'right')])
        self.basis = np.empty(((len(planq)+3)//4, 2, 12, 6))
        for index, k in enumerate(range(0, len(planq), 4)):
            scratch.qpos[:] = planq[k]
            mj.mj_kinematics(model, scratch); mj.mj_comPos(model, scratch)
            for actor in range(2):
                rows, targets = [], []
                for foot in feet[actor]:
                    jp = np.zeros((3, model.nv)); jr = np.zeros_like(jp)
                    mj.mj_jacBody(model, scratch, jp, jr, foot)
                    rows.append(np.r_[jp[:, model.jnt_dofadr[jointids[actor]]], .15*jr[:, model.jnt_dofadr[jointids[actor]]]])
                    x, y, z = scratch.xpos[foot]-planq[k, self.rq[actor]:self.rq[actor]+3]
                    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                    targets.append(np.block([[-np.eye(3), skew], [np.zeros((3, 3)), -.15*np.eye(3)]]))
                jac, target = np.concatenate(rows), np.concatenate(targets)
                self.basis[index, actor] = np.linalg.solve(jac.T@jac+.003*np.eye(12), jac.T@target)

    def feedback(self, k, q, v, kp, kd):
        output = np.zeros(56)
        for actor, (rq, rv) in enumerate(zip(self.rq, self.rv)):
            # Exact zero is essential for this numerically sensitive plan.
            if np.array_equal(q[rq:rq+7], self.planq[k, rq:rq+7]) and np.array_equal(v[rv:rv+6], self.planv[k, rv:rv+6]):
                continue
            position = self.planq[k, rq:rq+3]-q[rq:rq+3]
            angle = rotation_error(self.planq[k, rq+3:rq+7], q[rq+3:rq+7])
            velocity = self.planv[k, rv:rv+3]-v[rv:rv+3]
            omega = rotation(self.planq[k, rq+3:rq+7])@self.planv[k, rv+3:rv+6]-rotation(q[rq+3:rq+7])@v[rv+3:rv+6]
            output[self.legcols[actor]] = self.basis[k//4, actor]@(kp*np.r_[position, angle]+kd*np.r_[velocity, omega])
        return output
