"""Read-only actual positive-gap contact directions for partial-tilt candidates."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from fourth_arm_object_feedback import ArmObjectFeedback
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--search',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);h=ArmObjectFeedback(m,dict(np.load(a.reference,allow_pickle=False)));source=json.loads(a.search.read_text());rows=[]
for candidate in source['attempts']:
    d.qpos[:]=candidate['qpos'];mj.mj_kinematics(m,d);mj.mj_comPos(m,d);parts=[]
    for g in h.hand_geoms[1]:
        best=None
        for og in h.object_geoms[0]:
            w=np.zeros(6);gap=float(mj.mj_geomDistance(m,d,g,og,.2,w));delta=w[:3]-w[3:]
            if gap<=1e-6 or abs(np.linalg.norm(delta)-gap)>1e-4:continue
            row={'hand_geom':m.geom(g).name,'object_geom':m.geom(og).name,'gap_m':gap,'normal_force_on_object_world':(-delta/gap).tolist(),'hand_witness_world_m':w[:3].tolist(),'object_witness_world_m':w[3:].tolist()}
            if best is None or gap<best['gap_m']:best=row
        if best is not None:parts.append(best)
    rows.append({'attempt':candidate['attempt'],'base_xy_yaw':candidate['base_xy_yaw'],'closest_parts':sorted(parts,key=lambda item:item['gap_m']),'qpos':candidate['qpos']})
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'scope':'Positive-gap direction audit only; no force or executed motion claim','rows':rows},indent=2)+'\n')
print(json.dumps([{'attempt':row['attempt'],'closest':row['closest_parts'][:5]}for row in rows]),flush=True)
