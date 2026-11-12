"""Check arm feedback state immutability and control-channel boundaries."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import mujoco as mj
from fourth_arm_object_feedback import ArmObjectFeedback
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gravity-compensation',action='store_true');p.add_argument('--opposed-grasp',action='store_true');a=p.parse_args()
r=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(r['model']['path']);z=np.load(a.trial/'physical_trace.npz');ref=dict(np.load(a.reference,allow_pickle=False));d=mj.MjData(m);i=int(np.argmin(abs(z['time']-4.5)));d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i];d.time=float(z['time'][i]);mj.mj_forward(m,d)
helper=ArmObjectFeedback(m,ref,{'gravity_compensation':a.gravity_compensation,'require_opposed_grasp':a.opposed_grasp});keys=['qpos','qvel','ctrl','qfrc_applied','xfrc_applied','geom_xpos','geom_xmat'];before={k:getattr(d,k).copy()for k in keys};now=float(d.time);input_ctrl=d.ctrl.copy();out=helper.apply(d,input_ctrl,d.time)
checks={k+'_unchanged':np.array_equal(getattr(d,k),before[k])for k in keys};checks['time_unchanged']=d.time==now;checks['input_ctrl_unchanged']=np.array_equal(input_ctrl,before['ctrl'])
aids=np.r_[helper.chains[0]['aids'],helper.chains[1]['aids']];mask=np.ones(m.nu,dtype=bool);mask[aids]=False;checks['all_non_arm_actuator_channels_unchanged']=np.array_equal(out[mask],input_ctrl[mask]);checks['all_targets_finite']=bool(np.isfinite(out).all())
for side,chain in enumerate(helper.chains):checks[f'hand_{side}_arm_target_native_ranges']=bool(np.all(out[chain['aids']]>=chain['lower']-1e-7)and np.all(out[chain['aids']]<=chain['upper']+1e-7))
record=helper.records[-1];offset=record['actuator_gravity_offset_rad']
checks['gravity_motor_offset_within_cap']=bool(np.all(abs(offset)<=.25+1e-7))
checks['actual_motor_targets_equal_geometric_plus_logged_gravity_offset']=bool(np.allclose(record['applied_joint_targets'],record['geometric_joint_targets']+offset))
if a.gravity_compensation:checks['active_arm_gravity_compensation_nonzero']=bool(np.max(abs(offset))>1e-5)
report={'status':'passed'if all(checks.values())else'failed','checks':checks,'gravity_compensation_enabled':a.gravity_compensation,'require_opposed_grasp':a.opposed_grasp,'maximum_logged_motor_gravity_offset_rad':float(np.max(abs(offset))),'helper_sha256':hashlib.sha256(Path(__file__).with_name('fourth_arm_object_feedback.py').read_bytes()).hexdigest(),'actual_simulation_steps':0,'scope':'One active acquire state from a frozen physical trace; no claim of successful manipulation.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if not all(checks.values()):raise SystemExit(2)
