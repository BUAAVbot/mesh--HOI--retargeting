"""Compare real capsule projected support to coarse AABB support at RESET."""
from pathlib import Path
import json
import numpy as np
import mujoco as mj
from scipy.spatial import ConvexHull
R=Path(__file__).resolve().parents[1];r=json.loads((R/'reports/single_plate_explicit_edge_reset_benchmark_v3.json').read_text());m=mj.MjModel.from_xml_path(r['model']);d=mj.MjData(m);d.qpos[:]=r['best']['initial_qpos'];mj.mj_forward(m,d);xy=[];heights=[];theta=np.arange(720)*2*np.pi/720;circle=np.column_stack([np.cos(theta),np.sin(theta)])
for g in range(m.ngeom):
 if(m.geom(g).name or'').startswith('collision_foot'):
  axis=d.geom_xmat[g].reshape(3,3)[:,2];c=d.geom_xpos[g];s=m.geom_size[g]
  for sign in[-1,1]:xy.append(c[:2]+sign*s[1]*axis[:2]+s[0]*circle)
  heights.append({'geom':m.geom(g).name,'lowest_z_m':float(c[2]-s[1]*abs(axis[2])-s[0])})
hull=ConvexHull(np.concatenate(xy));b=[i for i in range(1,m.nbody)if not(m.body(i).name or'').startswith('target_object')];com=np.sum(d.xipos[b]*m.body_mass[b,None],axis=0)/np.sum(m.body_mass[b]);margin=float(np.min(-(hull.equations[:,:2]@com[:2]+hull.equations[:,2])));report={'scope':'Static projected support only; dynamic standing still requires physical settle validation. Capsule XY outline, not loose geom AABBs.','capsule_projected_support_COM_margin_m':margin,'robot_COM_world_m':com.tolist(),'foot_lowest_z_m':heights,'nominal_floor_z_m':-1.79097073316,'minimum_foot_floor_gap_m':min(t['lowest_z_m']for t in heights)+1.79097073316};p=R/'reports/single_plate_reset_support_capsule_audit.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
