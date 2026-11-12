"""Inspect a prospective support pose in private FK; no physics or mutation."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from fourth_convex_separation import ConvexSeparationOracle
from fourth_contact_wrench import contact_columns,wrench_lp
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();info=json.loads(a.candidate.read_text());row=info['best'];m=mj.MjModel.from_xml_path(info['model']);d=mj.MjData(m);d.qpos[:]=row['qpos'];d.qvel[:]=0;mj.mj_forward(m,d);obj=m.body('target_object_01').id;R=d.xmat[obj].reshape(3,3);pos=d.xpos[obj];oracle=ConvexSeparationOracle(m);bottom=next(g for g in range(m.ngeom)if m.geom_bodyid[g]==obj and m.geom_type[g]==mj.mjtGeom.mjGEOM_CYLINDER);geometry=[];selected=[]
for digit in ['thumb','index','middle']:
    g=next(g for g in range(m.ngeom)if m.geom_contype[g]and m.body(int(m.geom_bodyid[g])).name=='rh_'+digit+'_distal');selected.append(g);v=oracle._shape(g)['vertices']@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];local=(v-pos)@R;z=local[:,2];point=local[z<=z.min()+1e-9].mean(0)if digit=='thumb'else local[z>=z.max()-1e-9].mean(0);fromto=np.zeros(6);distance=mj.mj_geomDistance(m,d,g,bottom,.1,fromto)
    geometry.append({'digit':digit,'bottommost_z_object_m':float(z.min()),'topmost_z_object_m':float(z.max()),'selected_support_point_object':point.tolist(),'support_radius_m':float(np.linalg.norm(point[:2])),'bottom_disc_actual_distance_m':float(distance),'bottom_disc_closest_points_world':fromto.tolist()})
names=[m.joint(i).name for i in range(m.njnt)];master=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];P=np.zeros((m.nv,6))
for k,j in enumerate(master):P[m.jnt_dofadr[j],k]=1
for e in range(m.neq):
    if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
        child,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e])
        if parent in master:P[m.jnt_dofadr[child],master.index(parent)]=m.eq_data[e,1]
lp=[]
for margin in [.0015,.003,.005,.01]:
    m.geom_margin[selected]=margin;mj.mj_forward(m,d);c=contact_columns(m,d,obj,('rh_thumb','rh_index','rh_middle'),projection=P,max_gap_m=margin);result=wrench_lp(c['columns'],c['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=None,motor_torque_matrix=c['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@d.qfrc_bias,motor_cap_Nm=.4);sweep=[]
    if result['feasible']:
        for cap in [.35,.30,.25,.20]:sweep.append({'motor_planning_cap_Nm':cap,'result':wrench_lp(c['columns'],c['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=3.,motor_torque_matrix=c['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@d.qfrc_bias,motor_cap_Nm=cap)})
    lp.append({'query_margin_m':margin,'contact_patches':c['patches'],'wrench':result,'motor_headroom_sweep':sweep})
original=d.qpos.copy();curve={}
for digit in ['index','middle']:
    j=names.index('rh_'+digit+'_mcp_pitch');child=names.index('rh_'+digit+'_dip');g=selected[['thumb','index','middle'].index(digit)];values=[]
    for value in [.1,.2,.3,.4,.5,.6,.7]:
        d.qpos[:]=original;d.qpos[m.jnt_qposadr[j]]=value;d.qpos[m.jnt_qposadr[child]]=.904318245922221*value;mj.mj_kinematics(m,d);v=oracle._shape(g)['vertices']@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];values.append([value,float(np.max(((v-pos)@R)[:,2]))])
    curve[digit]=values
d.qpos[:]=original;mj.mj_kinematics(m,d)
out={'scope':'Private geometry/query margin sweep; prospective LP is not physical contact success. Original physical model not modified.','candidate':str(a.candidate),'geometry':geometry,'selected_joint_values':{names[j]:{'q_rad':float(d.qpos[m.jnt_qposadr[j]]),'range':m.jnt_range[j].tolist()}for j in master},'support_height_by_MCP':curve,'margin_query_wrench':lp};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'geometry':geometry,'support_height_by_MCP':curve,'query_3mm_result':lp[1]},indent=2),flush=True)
