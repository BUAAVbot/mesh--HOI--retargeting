"""Read-only initial-pose support audit for the fourth Vita trial candidates."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1]
DEMO=ROOT/'outputs/fourth_vita_demo'

def hull(points):
    pts=sorted(set((float(x),float(y))for x,y in points))
    def cross(o,a,b):return(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    low=[]
    for p in pts:
        while len(low)>=2 and cross(low[-2],low[-1],p)<=0:low.pop()
        low.append(p)
    high=[]
    for p in reversed(pts):
        while len(high)>=2 and cross(high[-2],high[-1],p)<=0:high.pop()
        high.append(p)
    return low[:-1]+high[:-1]

def audit(model_path,ref_path):
    m=mujoco.MjModel.from_xml_path(str(model_path));d=mujoco.MjData(m)
    r=np.load(ref_path,allow_pickle=False)
    q=np.asarray(r['qpos'][0],dtype=float)
    if q.shape!=(59,):raise ValueError(f'{ref_path} qpos expected 59')
    d.qpos[:59]=q;d.qpos[59:66]=r['object_target_pose'][0]
    mujoco.mj_forward(m,d)
    floor_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'assumed_floor')
    floor_z=float(m.geom_pos[floor_id,2])
    foot={}
    hull_points=[]
    for side,part in [('left','collision_foot_l_'),('right','collision_foot_r_')]:
        ids=[i for i in range(m.ngeom)if m.geom(i).name.startswith(part)]
        vals=[]
        for i in ids:
            c=np.asarray(d.geom_xpos[i]);rad=float(m.geom_size[i,0]);vals.append(float(c[2]-rad-floor_z))
            for v in ([rad,0],[-rad,0],[0,rad],[0,-rad]):hull_points.append(c[:2]+v)
        foot[side]={'geom_count':len(ids),'min_clearance_to_floor_m':min(vals),
                    'max_clearance_to_floor_m':max(vals)}
    polygon=hull(hull_points)
    baseid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'base_link')
    com=np.asarray(d.subtree_com[baseid],dtype=float)
    signed=min(((b[0]-a[0])*(com[1]-a[1])-(b[1]-a[1])*(com[0]-a[0]))/
               np.linalg.norm(np.asarray(b)-a)for a,b in zip(polygon,polygon[1:]+polygon[:1]))
    contact_names=[];floor_contacts=[];table_contacts=[]
    for i in range(d.ncon):
        c=d.contact[i];a=m.geom(int(c.geom1)).name;b=m.geom(int(c.geom2)).name
        row={'geom1':a,'geom2':b,'penetration_m':float(min(0.,c.dist)),
             'position_m':np.asarray(c.pos).tolist()}
        contact_names.append(row)
        if 'assumed_floor' in (a,b):floor_contacts.append(row)
        if 'table_support' in (a,b):table_contacts.append(row)
    return {'model':str(model_path),'reference':str(ref_path),
            'source_frame':int(r['frame_ids'][0]),
            'robot_root_xyz_m':q[:3].tolist(),'robot_com_xyz_m':com.tolist(),
            'floor_z_m':floor_z,'feet':foot,'support_hull_vertices_xy_m':polygon,
            'com_signed_distance_inside_foot_hull_m':float(signed),
            'floor_contacts':floor_contacts,'table_contacts':table_contacts,
            'initial_contact_count':len(contact_names),
            'object_target_pose_matches_model_qpos0':bool(np.allclose(r['object_target_pose'][0],m.qpos0[59:66],atol=1e-10)),
            'physics_steps':0}

def main():
    paths=[DEMO/'retarget_reference.npz',DEMO/'retarget_reference_status1.npz',
           DEMO/'retarget_reference_floor_contact_v3.npz']
    models=[DEMO/'assets/vita_object_05_dynamic.xml',DEMO/'assets/vita_object_05_dynamic_v2.xml']
    rows=[audit(m,p)for m in models for p in paths if m.is_file() and p.is_file()]
    output=DEMO/'assets/initial_pose_support_audit.json'
    output.write_text(json.dumps({'schema':'fourth_vita_initial_support.v1','results':rows},indent=2)+'\n',encoding='utf-8')
    for row in rows:
        print(json.dumps({'model':Path(row['model']).name,'ref':Path(row['reference']).name,
                          'foot_min_z':[row['feet'][s]['min_clearance_to_floor_m']for s in ('left','right')],
                          'COM_hull_signed_m':row['com_signed_distance_inside_foot_hull_m'],
                          'floor_contacts':len(row['floor_contacts']),'table_contacts':len(row['table_contacts']),
                          'table_min_penetration_m':min([x['penetration_m']for x in row['table_contacts']],default=0.)}))
    print(output)

if __name__=='__main__':main()
