"""Unit fixture for pressure memory; no physical experiment or geometry edits."""
from pathlib import Path
import json
import numpy as np
from fourth_o6_pressure_latch_v9 import O6CoupledReferenceContactFeedback

controller=object.__new__(O6CoupledReferenceContactFeedback)
controller.states=[{},{}];controller.load_ramp_s=.8
controller.config={'contact_detection_force_N':.03}
desired=np.array([2.55,1.52,2.55,.54,.05]);mask=np.ones(5,bool);required=np.array([True,True,True,True,False])
controller.states[1]['v9_last_returned_force']=desired.copy();controller.set_pressure_hold(1,True)
forces=desired.copy();controller._ramp_load_targets(1,0,mask,desired,required,forces,0.)
forces[3]=0.
for t in [.02,.04,.1,.15]:
    result,present,_=controller._ramp_load_targets(1,0,mask,desired,required,forces,t)
    assert not present
    assert np.array_equal(result,desired)
new_target=np.array([2.6,1.7,2.6,.05,.05]);controller.reseed_pressure_hold(1)
first=controller._ramp_load_targets(1,0,mask,new_target,required,forces,.2)[0]
assert np.array_equal(first,desired)
end=controller._ramp_load_targets(1,0,mask,new_target,required,forces,1.)[0]
assert np.allclose(end,new_target,rtol=0.,atol=1e-12)
report={'success':True,'missing_ring_contact_reported_false':True,'other_pressure_not_unloaded_during150ms_grace':True,
        'new_certified_target_first_step_jump_N':0.,'crossfade_duration_s':.8}
path=Path(__file__).resolve().parents[1]/'reports/pressure_latch_v9_fixture.json'
path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
