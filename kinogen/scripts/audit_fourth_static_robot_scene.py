"""Whole-robot collision audit at private static candidate poses."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();m=mj.MjModel.from_xml_path(str(a.model.resolve()));r=json.loads(a.candidate.read_text());best=r.get('best',r);q=np.asarray(best['qpos']);d=mj.MjData(m);d.qpos[:]=q;d.qvel[:]=0;mj.mj_forward(m,d);rows=[]
for c in d.contact[:d.ncon]:
    g1,g2=map(int,c.geom);b1,b2=map(int,m.geom_bodyid[[g1,g2]]);n1,n2=m.body(b1).name or'world',m.body(b2).name or'world';robot1=b1!=0 and not n1.startswith('target_object');robot2=b2!=0 and not n2.startswith('target_object')
    if robot1==robot2:continue
    rg,og=(g1,g2)if robot1 else(g2,g1);rb,ob=(b1,b2)if robot1 else(b2,b1);gn=m.geom(og).name or'unnamed';rn=m.body(rb).name;known_floor=gn=='assumed_floor'and'ankle'in rn
    rows.append({'robot_body':rn,'robot_geom':m.geom(rg).name,'other_body':m.body(ob).name or'world','other_geom':gn,'signed_gap_m':float(c.dist),'penetration_m':float(max(0,-c.dist)),'point_world_m':c.pos.tolist(),'known_foot_floor_contact':known_floor})
rows.sort(key=lambda r:-r['penetration_m']);unexpected=[x for x in rows if not x['known_foot_floor_contact']and x['penetration_m']>1e-6];out={'schema':'fourth.static_whole_robot_scene_collision_audit.v1','scope':'Private static FK/collision only. Checks every robot collider against physical table/floor/all six objects. Existing foot-floor contact is listed separately. No physics/force/lift claim.','model':str(a.model),'candidate':str(a.candidate),'contact_rows':rows,'unexpected_robot_scene_penetrations':unexpected,'unexpected_penetration_max_m':max([x['penetration_m']for x in unexpected],default=0.),'known_foot_floor_penetration_max_m':max([x['penetration_m']for x in rows if x['known_foot_floor_contact']],default=0.),'static_nonfoot_scene_collision_free':not unexpected};a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
