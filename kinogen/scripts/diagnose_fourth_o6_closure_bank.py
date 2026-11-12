from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--sweep',type=Path,required=True);p.add_argument('--row',type=int,default=4);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);d.qpos[:]=json.loads(a.sweep.read_text())['rows'][a.row]['qpos'];mj.mj_forward(m,d);obj=m.body('target_object_01').id;OR=d.xmat[obj].reshape(3,3);op=d.xpos[obj];og=[g for g in range(m.ngeom)if m.geom_bodyid[g]==obj and m.geom_contype[g]]
for digit in ['thumb','index','middle','ring','pinky']:
 hg=[g for g in range(m.ngeom)if m.geom_contype[g]and(m.body(int(m.geom_bodyid[g])).name or'').startswith('rh_'+digit)];rows=[]
 for g in hg:
  for o in og:
   w=np.zeros(6);gap=mj.mj_geomDistance(m,d,g,o,.1,w);rows.append((float(gap),g,o,w.copy()))
 gap,g,o,w=min(rows,key=lambda row:row[0]);print(json.dumps({'digit':digit,'gap_m':gap,'hand_geom':m.geom(g).name,'object_geom':m.geom(o).name,'closest_hand_objectlocal':(OR.T@(w[:3]-op)).tolist(),'closest_object_objectlocal':(OR.T@(w[3:]-op)).tolist()}))
