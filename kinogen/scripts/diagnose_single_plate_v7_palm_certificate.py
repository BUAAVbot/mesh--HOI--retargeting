"""Read-only selected recorded packet contacts: finger-only vs whole-hand certificate."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from fourth_contact_wrench import contact_columns
from fourth_contact_wrench_balanced_c9 import wrench_lp
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=mj.MjModel.from_binary_path(str(a.trial/'effective_model.mjb'));z=dict(np.load(a.trial/'physical_trace.npz'));c=dict(np.load(a.trial/'physical_contacts.npz'));d=mj.MjData(m);obj=m.body('target_object_01').id;digits=['thumb','index','middle','ring','pinky'];prefix=tuple('rh_'+v for v in digits)+('rh_hand_base_link',);names=[m.joint(i).name or''for i in range(m.njnt)];masters=[names.index('rh_'+n)for n in['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];arm=[names.index(n)for n in['right_shoulder_pitch_joint','right_shoulder_roll_joint','right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_yaw_joint','right_wrist_pitch_joint']];P=np.zeros((m.nv,6));
for i,j in enumerate(masters):P[m.jnt_dofadr[j],i]=1
for e in range(m.neq):
 if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT and int(m.eq_obj2id[e])in masters:P[m.jnt_dofadr[int(m.eq_obj1id[e])],masters.index(int(m.eq_obj2id[e]))]=m.eq_data[e,1]
Q=np.c_[P,np.eye(m.nv)[:,m.jnt_dofadr[arm]]];arm_aids=[m.actuator('motor_'+names[j]).id for j in arm];caps=np.r_[np.full(6,.4),m.actuator_forcerange[arm_aids,1]];allsteps=np.unique(c['step']);rows=[]
for t in[18.,20.,22.,24.,24.5,25.,25.2,25.4,25.6,25.8,26.,26.2,26.5,27.]:
 st=int(allsteps[np.argmin(abs(z['time_s'][allsteps+1]-t))]);mask=c['step']==st;indices=np.flatnonzero(mask);normal=np.zeros(6);up=np.zeros(6);full=np.zeros(3);groups={};d.qpos[:]=z['qpos'][st];d.qvel[:]=0.;d.ctrl[:]=z['ctrl'][st];mj.mj_forward(m,d)
 for j in indices:
  bs=list(map(int,c['body'][j]));gs=list(map(int,c['geom'][j]));
  if obj not in bs:continue
  oi=bs.index(obj);ii=1-oi;name=m.body(bs[ii]).name or'world';F=c['force'][j];wf=(1 if oi==1 else-1)*c['frame'][j].reshape(3,3).T@F[:3];full+=wf;di=next((i for i,pre in enumerate(prefix)if name.startswith(pre)),None)
  if di is not None:normal[di]+=max(float(F[0]),0.);up[di]+=wf[2]
  key='|'.join([name,m.geom(gs[ii]).name,m.geom(gs[oi]).name]);groups[key]={'normal_N':float(F[0]),'world_force_N':wf.tolist()}
 fi=contact_columns(m,d,obj,prefix[:5],P,max_gap_m=0.);whole=contact_columns(m,d,obj,prefix,Q,max_gap_m=0.);gravity=np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)];flp=wrench_lp(fi['columns'],fi['normal_rows'],gravity,normal_cap_N=.95*normal[:5],motor_torque_matrix=fi['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@d.qfrc_bias,motor_cap_Nm=.4);wlp=wrench_lp(whole['columns'],whole['normal_rows'],gravity,normal_cap_N=.95*normal,motor_torque_matrix=whole['motor_torque_matrix'],motor_gravity_bias_Nm=Q.T@d.qfrc_bias,motor_cap_Nm=caps);row={'time_s':float(z['time_s'][st+1]),'recorded_native_step':st,'recorded_all_contact_force_N':full.tolist(),'recorded_RH_digit_and_palm_normal_N':normal.tolist(),'recorded_RH_digit_and_palm_up_N':up.tolist(),'finger_only_measured95pct_LP':flp,'whole_RH_hand_measured95pct_LP_including_original7arm_torque_caps':wlp,'contact_groups':groups};rows.append(row);print(json.dumps({'t':row['time_s'],'finger_LP':flp['feasible'],'whole_hand_LP':wlp['feasible'],'palm_up_N':up[5],'all_hand_up_N':float(up.sum())}),flush=True)
result={'scope':'Recorded native packet-end forces, selected times only; no new rollout or state/model modification. Private FK reconstructs exact pre-step collision geometry for contact Jacobians; measured caps from recorded actual contacts. Whole-hand certificate includes original RH seven-arm plus six finger motor caps; excludes floating-root/stance proof and is diagnostic only. Does not alter original finger-only gate or raw3cm success result.','trial':str(a.trial),'group_order':digits+['palm'],'whole_hand_motor_caps_Nm':caps.tolist(),'samples':rows};a.output.write_text(json.dumps(result,indent=2,default=lambda v:v.tolist())+'\n')
