"""Default-off shadow regression against the still-frozen execution module."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from fourth_arm_object_feedback import ArmObjectFeedback as Old
from fourth_arm_object_feedback_absolute_hold_candidate import ArmObjectFeedback as New
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();report=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(report['model']['path']);d=mj.MjData(m);z=dict(np.load(a.trial/'physical_trace.npz'));ref=dict(np.load(a.reference));config=json.loads(a.config.read_text())['arm_feedback'];old=Old(m,ref,config);new=New(m,ref,config);maximum=0.;goalerr=0.;calls=0;readonly=True
for k in range(0,len(z['ctrl']),4):
    d.qpos[:]=z['qpos'][k];d.qvel[:]=z['qvel'][k];d.ctrl[:]=z['ctrl'][k];d.time=float(z['time'][k]);mj.mj_forward(m,d);before=(d.qpos.copy(),d.qvel.copy(),d.qfrc_applied.copy(),d.xfrc_applied.copy());co=old.apply(d,z['ctrl'][k],d.time);cn=new.apply(d,z['ctrl'][k],d.time);maximum=max(maximum,float(np.max(abs(cn-co))));goalerr=max(goalerr,float(np.max(abs(new.records[-1]['goal']-old.records[-1]['goal']))));readonly &=all(np.array_equal(x,y)for x,y in zip(before,[d.qpos,d.qvel,d.qfrc_applied,d.xfrc_applied]));calls+=1
out={'status':'pass'if maximum==0 and goalerr==0 and readonly else'fail','scope':'Default-off shadow packets on identical recorded actual states, not another rollout','calls':calls,'maximum_output_packet_error':maximum,'maximum_planned_goal_error':goalerr,'actual_state_and_external_forces_unchanged':bool(readonly),'default_absolute_hold_disabled':not config.get('absolute_reference_hold',False)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
if out['status']!='pass':raise SystemExit(1)
