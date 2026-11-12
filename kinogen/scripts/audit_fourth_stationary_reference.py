"""Audit all source-reference robot/table contacts without stepping dynamics."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);z=np.load(a.reference,allow_pickle=False)
names=list(map(str,z['dof_names']));jids=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]);qa=m.jnt_qposadr[jids]
table={g for g in range(m.ngeom)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('table_support')}
objects={mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,f'target_object_{i:02d}')for i in z['object_ids']}
rows=[]
for f,q in enumerate(z['qpos']):
    d.qpos[:7]=q[:7];d.qpos[qa]=q[7:];d.qpos[59:]=z['object_target_pose'][f].reshape(-1);mj.mj_forward(m,d)
    for c in d.contact:
        g1,g2=int(c.geom1),int(c.geom2)
        if g1 not in table and g2 not in table:continue
        robot=g2 if g1 in table else g1;bid=int(m.geom_bodyid[robot])
        if bid<=0 or bid in objects or c.dist>=0:continue
        rows.append({'frame':f,'time_s':float(z['time_s'][f]-z['time_s'][0]),'robot_geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,robot),'robot_body':mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,bid),'penetration_m':float(-c.dist)})
worst=sorted(rows,key=lambda x:x['penetration_m'],reverse=True)
report={'scope':'All923 reference-pose FK contact audit; zero dynamics steps, not physical rollout','frames':len(z['qpos']),'frames_with_robot_table_penetration':len(set(r['frame']for r in rows)),'maximum_robot_table_penetration_m':max((r['penetration_m']for r in rows),default=0.),'by_robot_body':{b:max(r['penetration_m']for r in rows if r['robot_body']==b)for b in sorted(set(r['robot_body']for r in rows))},'worst_contacts':worst[:12]}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
