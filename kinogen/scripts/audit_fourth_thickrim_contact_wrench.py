"""Static wrench feasibility at actual recorded hand-dish contact poses."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from fourth_contact_wrench import contact_columns,wrench_lp
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--times',type=float,nargs='+',default=[11.005,12.005,12.505]);a=p.parse_args()
m=mj.MjModel.from_binary_path(str((a.trial/'effective_model.mjb').resolve()));z=dict(np.load(a.trial/'physical_trace.npz'));c=dict(np.load(a.trial/'physical_contacts.npz'));t=z['time_s']if'time_s'in z else z['time'];d=mj.MjData(m);digits=['thumb','index','middle','ring','pinky'];prefixes=tuple('rh_'+x for x in digits);master=[m.joint('rh_'+x).id for x in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];P=np.zeros((m.nv,6))
for col,j in enumerate(master):P[m.jnt_dofadr[j],col]=1
for e in range(m.neq):
    if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT and int(m.eq_obj2id[e])in master:P[m.jnt_dofadr[int(m.eq_obj1id[e])],master.index(int(m.eq_obj2id[e]))]=m.eq_data[e,1]
obj=m.body('target_object_01').id;rows=[]
for requested in a.times:
    k=int(np.argmin(abs(t[:-1]-requested)));d.qpos[:]=z['qpos'][k];d.qvel[:]=0;mj.mj_forward(m,d);w=contact_columns(m,d,obj,digit_prefixes=prefixes,projection=P);bias=P.T@d.qfrc_bias;gravity=np.r_[m.body_mass[obj]*m.opt.gravity,[0.,0.,0.]];lp={str(cap):wrench_lp(w['columns'],w['normal_rows'],gravity,normal_cap_N=cap,motor_torque_matrix=w['motor_torque_matrix'],motor_gravity_bias_Nm=bias,motor_cap_Nm=.4)for cap in[3.,6.,12.]};recorded=[];tableforce=0.;handforce=np.zeros(3)
    for ci in np.flatnonzero(c['step']==k):
        g1,g2=map(int,c['geom'][ci]);b1,b2=map(int,c['body'][ci])
        if obj not in(b1,b2):continue
        other=b2 if b1==obj else b1;og=g2 if b1==obj else g1;name=m.body(other).name or'world';sign=1. if b2==obj else-1.;frame=c['frame'][ci].reshape(3,3);F=sign*(frame.T@c['force'][ci,:3]);normal=sign*frame[0]
        if(m.geom(og).name or'').startswith('table_support'):tableforce+=float(F[2])
        if name.startswith(prefixes):
            handforce+=F;recorded.append({'hand_body':name,'hand_geom':m.geom(og).name,'object_geom':m.geom(g1 if b1==obj else g2).name,'point_world_m':c['point'][ci].tolist(),'normal_on_object':normal.tolist(),'actual_force_world_N':F.tolist(),'actual_normal_N':float(c['force'][ci,0]),'distance_m':float(c['distance'][ci])})
    geometry_only=wrench_lp(w['columns'],w['normal_rows'],gravity,normal_cap_N=None)
    rows.append({'requested_time_s':requested,'actual_pre_step_time_s':float(t[k]),'recorded_step':k,'actual_hand_contacts':recorded,'actual_table_up_N':tableforce,'actual_hand_force_world_N':handforce.tolist(),'static_gravity_motor_bias_Nm':bias.tolist(),'recomputed_static_patches_at_recorded_qpos':w['patches'],'LP_with_hard_motor_0p4Nm':lp,'LP_geometry_only_no_motor_or_normal_caps':geometry_only})
out={'scope':'Actual recorded geometry/contact forces plus static gravity wrench LP with exact original O6 coupling and engineering hard0.4Nm cap. LP uses mj_forward at recorded pre-step qpos and zero velocity; no physical rollout or lift success inferred. Per-digit normal caps3/6/12N are engineering search budgets, not hardware ratings.','trial':str(a.trial),'digit_order':digits,'master_motor_order':[m.joint(j).name for j in master],'samples':rows};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
