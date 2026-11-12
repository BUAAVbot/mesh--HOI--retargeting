"""Prepare only; no simulated success is claimed by model construction."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco as mj
R=Path(__file__).resolve().parents[1]
planpath=R/'reports/fourth_o6_stiff_eq_push_plan_v1.json';plan=json.loads(planpath.read_text())
base=Path(plan['model']);m=mj.MjModel.from_binary_path(str(base))
out=R/'outputs/fourth_full_sequence/assets/vita_six_objects_o6_microstep_v1.mjb'
if out.exists():raise FileExistsError(out)
source=json.loads(base.with_suffix('.manifest.json').read_text())
preserved=[k for k in source['preserved_exactly']if k!='eq_solref']
before={k:getattr(m,k).copy()for k in preserved};oldref=m.eq_solref.copy();olddt=float(m.opt.timestep)
ids=np.flatnonzero(m.eq_type==mj.mjtEq.mjEQ_JOINT)
m.eq_solref[ids]=[.002,1.];m.opt.timestep=.001
mj.mj_saveModel(m,str(out),None);n=mj.MjModel.from_binary_path(str(out))
assert all(np.array_equal(before[k],getattr(n,k))for k in preserved)
report={'scope':'Explicit numerical transmission/integration variant, no hand or object geometry changes.',
        'source_model':str(base),'model':str(out),'model_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
        'changed_eq_solref_ids':ids.tolist(),'old_eq_solref':oldref[ids].tolist(),
        'new_eq_solref':n.eq_solref[ids].tolist(),'old_physical_dt_s':olddt,'new_physical_dt_s':.001,
        'control_packet_dt_s':.005,'control_substeps':5,'nominal_actuator_delay_s':.015,
        'policy_dt_s':.02,'finger_feedback_dt_s':.02,'preserved_exactly':preserved,
        'refsafe_disabled':False,'source':'https://mujoco.readthedocs.io/en/stable/modeling.html#solver-parameters',
        'hardware_stiffness_claim':False,'actual_rollout_performed':False}
out.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n')
plan['model']=str(out);plan['numerical_variant_provenance']=report
(R/'reports/fourth_o6_microstep_push_plan_v1.json').write_text(json.dumps(plan,indent=2)+'\n')
s=(R/'scripts/run_fourth_o6_preslide_extended_v4.py').read_text()
s=s.replace('from fourth_sonic_controller import Controller','from fourth_o6_microstep_clock_v1 import Controller,CONTROL_DT,step_packet,packet_mimic_peak,numerical_report')
s=s.replace('dt=float(m.opt.timestep)','dt=CONTROL_DT').replace('mj.mj_step(m,d)','step_packet(m,d)')
s=s.replace('error=max(abs(v)for v in errors);','error=max(max(abs(v)for v in errors),packet_mimic_peak(m,d));')
s=s.replace("(a.output/'report.json').write_text", "report['numerical_integration']=numerical_report(m,d);\n(a.output/'report.json').write_text")
s=s.replace("'vita_policy_tracking.py']:","'vita_policy_tracking.py','fourth_o6_microstep_clock_v1.py']:")
runner=R/'scripts/run_fourth_o6_microstep_push_v1.py'
compile(s,str(runner),'exec');runner.write_text(s)
print(json.dumps({'model':str(out),'runner':str(runner),'preserved_all_checked_arrays':True}))
