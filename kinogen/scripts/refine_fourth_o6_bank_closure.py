"""Close each of four native MCPs to the actual curved rim, with thumb fixed."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from scipy.optimize import brentq
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--sweep',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);z=json.loads(a.sweep.read_text());d.qpos[:]=z['rows'][4]['qpos'];base=d.qpos.copy();obj=m.body('target_object_01').id;og=[g for g in range(m.ngeom)if m.geom_bodyid[g]==obj and m.geom_contype[g]]
def forward():
 for e in range(m.neq):
  if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
   c,p=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);d.qpos[m.jnt_qposadr[c]]=m.eq_data[e,1]*d.qpos[m.jnt_qposadr[p]]
 mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
entries=[]
for digit in ['thumb','index','middle','ring','pinky']:
 j=m.joint('rh_'+('thumb_cmc_pitch'if digit=='thumb'else digit+'_mcp_pitch')).id;qa=int(m.jnt_qposadr[j]);hg=[g for g in range(m.ngeom)if m.geom_contype[g]and(m.body(int(m.geom_bodyid[g])).name or'').startswith('rh_'+digit)]
 def gap(q):
  d.qpos[qa]=q;forward();return min(float(mj.mj_geomDistance(m,d,g,o,.05,None))for g in hg for o in og)-.0005
 center=float(d.qpos[qa]);grid=np.linspace(max(m.jnt_range[j,0],center-.15),min(m.jnt_range[j,1],center+.25),31);vals=[gap(q)for q in grid];roots=[]
 for k in range(len(grid)-1):
  if vals[k]*vals[k+1]<0:roots.append(brentq(gap,grid[k],grid[k+1],xtol=1e-10))
 q=min(roots,key=lambda q:abs(q-center))if roots else grid[int(np.argmin(abs(np.asarray(vals))))];dist=gap(q)+.0005
 entries.append({'digit':digit,'master_command_rad':float(q),'gap_m':float(dist),'root_found':bool(roots)})
 print(json.dumps(entries[-1]),flush=True)
forward();out={'scope':'private exact collision gap based per-digit closure, not physical motion','rows':[{'qpos':d.qpos.tolist(),'residual_norm':0.}],'best':{'qpos':d.qpos.tolist(),'residual_norm':0.},'digits':entries};a.output.write_text(json.dumps(out,indent=2)+'\n')
