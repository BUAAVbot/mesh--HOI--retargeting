"""Load-stage contact hysteresis only; physical contacts and proof stay raw."""
import numpy as np
from fourth_contact_feedback_o6_coupled import O6CoupledReferenceContactFeedback as Base


class O6CoupledReferenceContactFeedback(Base):
    def _ramp_load_targets(self, side, oi, mask, desired, required, forces, now):
        state = self.states[side]
        oid = int(self.object_ids[oi]) if oi >= 0 else -1
        if state.get('c7_oid') != oid:
            state['c7_oid'] = oid
            state['c7_seen'] = np.full(5, -np.inf)
            state['c7_load_since'] = None
            state['c7_loaded'] = False
        seen = state['c7_seen']
        seen[forces > float(self.config['contact_detection_force_N'])] = now
        recent = (now-seen) <= .100000001
        if oi >= 0:
            gaps, _ = self._geometry(self.hands[side], oi)
            recent &= np.isfinite(gaps) & (gaps < .002)
        else:
            recent[:] = False
        present = bool(oi >= 0 and np.all(recent[required]))
        loaded = bool(desired.max(initial=0) > self.load_preload_N+.05)
        if loaded and not state['c7_loaded']:
            state['c7_load_since'] = now if present else None
        state['c7_loaded'] = loaded
        if present and loaded:
            if state['c7_load_since'] is None:
                state['c7_load_since'] = now
        elif not present:
            state['c7_load_since'] = None
        elapsed = 0. if state['c7_load_since'] is None else max(now-state['c7_load_since'], 0.)
        fraction = float(np.clip(elapsed/self.load_ramp_s, 0, 1)) if loaded else 0.
        blend = fraction*fraction*(3-2*fraction)
        preload = np.where(mask, np.minimum(desired, self.load_preload_N), 0.)
        return preload+(desired-preload)*blend, present, fraction
