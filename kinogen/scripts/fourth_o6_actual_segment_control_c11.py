"""Control the segment carrying real load; original physical hand is untouched."""
import numpy as np
import mujoco as mj
from fourth_o6_load_hysteresis_cached_c10 import O6CoupledReferenceContactFeedback as Base


class O6CoupledReferenceContactFeedback(Base):
    def __init__(self, model, reference, config=None):
        super().__init__(model, reference, config)
        self._c11_actual_data = None
        self._c11_geometry_cache = {}
        self._c11_actual_segment_uses = {}
        for hand in self.hands:
            for digit in hand['digits']:
                digit['tip_geoms'] = list(digit['all_segment_geoms'])
        self._c10_geometry_cache = {}

    def apply(self, data, ctrl, t, contact_plan=None):
        self._c11_actual_data = data
        return super().apply(data, ctrl, t, contact_plan)

    def _geometry(self, hand, object_index, jacobian=False):
        gaps, J = super()._geometry(hand, object_index, True)
        actual = getattr(self, '_c11_actual_data', None)
        if actual is None:
            return gaps, J
        target = int(self.object_bodies[object_index])
        digit_names = [hand['prefix']+digit['name'] for digit in hand['digits']]
        groups = [[] for _ in digit_names]
        for ci, contact in enumerate(actual.contact[:actual.ncon]):
            g1, g2 = map(int, contact.geom)
            b1, b2 = map(int, self.model.geom_bodyid[[g1, g2]])
            if target not in (b1, b2):
                continue
            target_is_second = b2 == target
            other = b1 if target_is_second else b2
            body_name = self.model.body(other).name or ''
            row = next((k for k, prefix in enumerate(digit_names) if body_name.startswith(prefix)), None)
            if row is None or contact.dist > .001:
                continue
            force = np.zeros(6)
            mj.mj_contactForce(self.model, actual, ci, force)
            if force[0] <= .02:
                continue
            hand_geom = g1 if target_is_second else g2
            normal = np.asarray(contact.frame).reshape(3, 3)[0]*(1 if target_is_second else -1)
            groups[row].append((float(force[0]), float(contact.dist), other,
                                np.asarray(contact.pos).copy(), normal, hand_geom))
        jp = np.zeros((3, self.model.nv))
        for row, contacts in enumerate(groups):
            if not contacts:
                continue
            total = sum(c[0] for c in contacts)
            gap = 0.
            derivative = np.zeros(len(hand['va']))
            for force, dist, body, point, normal, geom in contacts:
                weight = force/total
                mj.mj_jac(self.model, self._kin, jp, None, point, body)
                derivative += weight*(-normal@jp[:, hand['va']])
                gap += weight*dist
                name = self.model.geom(geom).name or str(geom)
                self._c11_actual_segment_uses[name] = self._c11_actual_segment_uses.get(name, 0)+1
            gaps[row] = gap
            J[row] = derivative
        return gaps, J

    def report(self):
        result = super().report()
        result['actual_segment_admittance'] = {
            'physical_hand_collision_change': False,
            'unloaded_geometry': 'Minimum signed SAT separation over original proximal and distal segment meshes.',
            'loaded_geometry': 'Force-weighted actual contact signed gaps and contact normal/point Jacobians on the actual carrying links.',
            'actual_segment_query_counts': self._c11_actual_segment_uses,
            'load_hysteresis_s': .1}
        return result
