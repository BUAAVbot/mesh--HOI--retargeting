"""Same C7 loading hysteresis, caching exact identical-state SAT queries."""
from fourth_o6_load_hysteresis_c7 import O6CoupledReferenceContactFeedback as Base


class O6CoupledReferenceContactFeedback(Base):
    def _geometry(self, hand, object_index, jacobian=False):
        if not hasattr(self, '_c10_geometry_cache'):
            self._c10_geometry_cache = {}
        key = (hand['prefix'], object_index)
        signature = self._kin.qpos.tobytes()
        cached = self._c10_geometry_cache.get(key)
        if cached is None or cached[0] != signature:
            distance, J = super()._geometry(hand, object_index, True)
            cached = (signature, distance.copy(), J.copy())
            self._c10_geometry_cache[key] = cached
        return cached[1].copy(), cached[2].copy()
