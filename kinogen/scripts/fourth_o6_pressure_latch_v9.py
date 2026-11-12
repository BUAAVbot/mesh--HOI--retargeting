"""Maintain previously established, bounded grasp pressure during lift pauses."""
import numpy as np
from fourth_o6_load_hysteresis_cached_c10 import O6CoupledReferenceContactFeedback as Base


class O6CoupledReferenceContactFeedback(Base):
    def set_pressure_hold(self, side, enabled):
        state=self.states[side]
        enabled=bool(enabled)
        if enabled and not state.get('v9_hold_enabled',False):
            state['v9_hold_since']=None
            state['v9_hold_seed']=state.get('v9_last_returned_force',np.zeros(5)).copy()
        state['v9_hold_enabled']=enabled

    def reseed_pressure_hold(self, side):
        state=self.states[side]
        state['v9_hold_since']=None
        state['v9_hold_seed']=state.get('v9_last_returned_force',np.zeros(5)).copy()

    def _ramp_load_targets(self,side,oi,mask,desired,required,forces,now):
        state=self.states[side]
        if state.get('v9_hold_enabled',False):
            if state['v9_hold_since'] is None:state['v9_hold_since']=float(now)
            elapsed=max(0.,float(now)-state['v9_hold_since'])
            s=float(np.clip(elapsed/self.load_ramp_s,0,1));s=s*s*(3-2*s)
            seed=np.where(mask,state['v9_hold_seed'],0.)
            value=seed+(desired-seed)*s
            present=bool(oi>=0 and np.all(forces[required]>float(self.config['contact_detection_force_N'])))
            state['v9_last_returned_force']=value.copy()
            return value,present,s
        result=super()._ramp_load_targets(side,oi,mask,desired,required,forces,now)
        state['v9_last_returned_force']=result[0].copy()
        return result

    def report(self):
        result=super().report()
        result['formal_lift_pressure_memory']='Keep last certified bounded pressure through individual-digit or wrench-certificate loss; new certified allocation crossfades over original0.8s ramp; raw presence remains separately reported.'
        return result
