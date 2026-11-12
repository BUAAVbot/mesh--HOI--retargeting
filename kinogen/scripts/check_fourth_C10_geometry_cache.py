from pathlib import Path
import json,time
import numpy as np
import mujoco as mj
from fourth_o6_load_hysteresis_c7 import O6CoupledReferenceContactFeedback as Uncached
from fourth_o6_load_hysteresis_cached_c10 import O6CoupledReferenceContactFeedback as Cached
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'outputs/fourth_full_sequence/physics_o6_C6_upperbodyhold_v1'
m=mj.MjModel.from_binary_path(str(p/'effective_model.mjb'))
ref=dict(np.load(p/'prefix/push/source_reference.npz'))
cfg=json.loads((p/'finger_config.json').read_text())
a=Uncached(m,ref,cfg);b=Cached(m,ref,cfg);z=np.load(p/'physical_trace.npz');rows=[]
for requested in [59.5,63.5,70.87,81.4]:
 k=int(np.argmin(abs(z['time_s']-requested)))
 for helper in [a,b]:
  helper._kin.qpos[:]=z['qpos'][k];helper._kin.qvel[:]=z['qvel'][k];mj.mj_forward(m,helper._kin)
 da,_=a._geometry(a.hands[1],0,False);db,_=b._geometry(b.hands[1],0,False)
 daj,ja=a._geometry(a.hands[1],0,True);dbj,jb=b._geometry(b.hands[1],0,True)
 before=b._sat.calls
 start=time.monotonic()
 for _ in range(10):b._geometry(b.hands[1],0,False)
 row={'time_s':float(z['time_s'][k]),'distance_max_error_m':float(max(np.max(abs(da-db)),np.max(abs(daj-dbj)))),
      'jacobian_max_error':float(np.max(abs(ja-jb))),'SAT_calls_for_ten_identical_queries':int(b._sat.calls-before),
      'ten_cached_queries_wall_s':time.monotonic()-start};rows.append(row)
 assert row['distance_max_error_m']==0 and row['jacobian_max_error']==0 and row['SAT_calls_for_ten_identical_queries']==0
out={'scope':'Private FK/SAT numerical equivalence; no dynamics, no physical success claim.','rows':rows}
(ROOT/'reports/fourth_C10_geometry_cache_equivalence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out),flush=True)
