"""Private one-dimensional closure sweep from a load-bearing pregrasp seed."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_convex_separation import ConvexSeparationOracle
from fourth_contact_wrench import contact_columns,wrench_lp
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();info=json.loads(a.candidate.read_text());base=info['best'];m=mj.MjModel.from_xml_path(info['model']);d=mj.MjData(m);oracle=ConvexSeparationOracle(m);seed=np.asarray(base['qpos']);names=[m.joint(i).name for i in range(m.njnt)];master=[names.index('rh_'+n)for n in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];index=master[2];child=names.index('rh_index_dip');hand=[g for g in range(m.ngeom)if m.geom_contype[g]and(m.body(int(m.geom_bodyid[g])).name or'').startswith('rh_')];objectgeoms=[g for g in range(m.ngeom)if m.geom_contype[g]and(m.body(int(m.geom_bodyid[g])).name or'').startswith('target_object')];table=m.geom('table_support').id;obj=m.body('target_object_01').id;P=np.zeros((m.nv,6))
for k,j in enumerate(master):P[m.jnt_dofadr[j],k]=1
for e in range(m.neq):
    if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
        c,parent=int(m.eq_obj1id[e]),int(m.eq_obj2id[e])
        if parent in master:P[m.jnt_dofadr[c],master.index(parent)]=m.eq_data[e,1]
original_margin=m.geom_margin.copy();results=[]
for delta in [0,.005,.01,.015,.018,.02,.022,.025]:
    d.qpos[:]=seed;d.qpos[m.jnt_qposadr[index]]+=delta;d.qpos[m.jnt_qposadr[child]]=.904318245922221*d.qpos[m.jnt_qposadr[index]];d.qvel[:]=0;m.geom_margin[:]=original_margin;mj.mj_kinematics(m,d);mj.mj_comPos(m,d);closest=min((oracle.query(d,g,og)[0],g,og)for g in hand for og in objectgeoms);tablegap=min(oracle.query(d,g,table)[0]for g in hand);m.geom_margin[hand]=.0015;mj.mj_forward(m,d);cols=contact_columns(m,d,obj,('rh_thumb','rh_index','rh_middle'),projection=P,max_gap_m=.0015);lp=wrench_lp(cols['columns'],cols['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=3.,motor_torque_matrix=cols['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@d.qfrc_bias,motor_cap_Nm=.4);m.geom_margin[:]=original_margin;row={'index_delta_rad':delta,'minimum_mesh_objects_sat_m':float(closest[0]),'closest_pair':[m.geom(closest[1]).name,m.geom(closest[2]).name],'minimum_hand_table_sat_m':float(tablegap),'wrench':lp,'patches':cols['patches'],'qpos':d.qpos.tolist(),'template':dict(base['template'])};row['template']['finger_joint_targets']={names[j]:float(d.qpos[m.jnt_qposadr[j]])for j,n in enumerate(names)if n and n.startswith('rh_')};row['pass']=bool(closest[0]>=.0004 and tablegap>=.0018 and lp['feasible']);results.append(row);print(json.dumps({k:v for k,v in row.items()if k not in('qpos','template','patches')}),flush=True)
    if row['pass']:break
out={'scope':'Pure private FK closure sweep; actual physical model and object states are unchanged. Query-only margins restored.','model':info['model'],'model_sha256':hashlib.sha256(Path(info['model']).read_bytes()).hexdigest(),'source_candidate':str(a.candidate),'pre_slide_planning_translation_m':info['pre_slide_planning_translation_m'],'best':next((x for x in results if x['pass']),min(results,key=lambda x:x['wrench']['residual_force_equivalent_L1'])),'attempts':results};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n')
