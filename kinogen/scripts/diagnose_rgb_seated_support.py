"""Report actual initial COM and contact support, plus waist sensitivity."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--reference',default='outputs/rgb_near_pipeline/reference_v17_source_hipcyl_prefix47.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v5_source_hipcyl.xml')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    m=mj.MjModel.from_xml_path(str(a.root/a.scene));d=mj.MjData(m);z=np.load(a.root/a.reference,allow_pickle=False)
    maps=[]
    for ac in range(2):
        pre=f'p{ac+1}_';js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+str(n))for n in z['dof_names']])
        r=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root');rq=int(m.jnt_qposadr[r])
        d.qpos[rq:rq+7]=z['qpos'][0,ac,:7];d.qpos[m.jnt_qposadr[js]]=z['qpos'][0,ac,7:];maps.append((pre,js))
    sj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free');sq=int(m.jnt_qposadr[sj]);d.qpos[sq:sq+7]=z['object_target_pose'][0]
    mj.mj_forward(m,d);original=d.qpos.copy()
    def gn(g):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,int(g))or''
    report=[]
    for pre,js in maps:
        base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,pre+'base_link');d.qpos[:]=original;mj.mj_forward(m,d)
        com=d.subtree_com[base].copy();contacts=[{'geom1':gn(c.geom1),'geom2':gn(c.geom2),'signed_distance_m':float(c.dist),'position_world':c.pos.tolist(),'normal_world':c.frame[:3].tolist()}for c in d.contact if pre in gn(c.geom1)+gn(c.geom2)]
        seat=[g for g in range(m.ngeom)if gn(g)=='bench_seat_proxy'];hip=[g for g in range(m.ngeom)if gn(g).startswith(pre+'collision_')and'hip_roll'in gn(g)]
        hipdist=[{'hip':gn(g),'seat_gap_m':float(mj.mj_geomDistance(m,d,g,seat[0],1.,np.zeros(6)))}for g in hip]
        floor=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'ground_proxy')
        feet=[g for g in range(m.ngeom)if gn(g).startswith(pre+'collision_foot_')and m.geom_contype[g]]
        footdist=[{'foot':gn(g),'floor_gap_m':float(mj.mj_geomDistance(m,d,g,floor,1.,np.zeros(6)))}for g in feet]
        sensitivity=[]
        for joint in ('waist_pitch_joint','left_shoulder_pitch_joint','left_shoulder_roll_joint','left_elbow_joint'):
            j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+joint);qa=int(m.jnt_qposadr[j]);lo,hi=m.jnt_range[j]
            for delta in(-.15,.15):
                d.qpos[:]=original;d.qpos[qa]=np.clip(original[qa]+delta,lo,hi);mj.mj_forward(m,d)
                sensitivity.append({'joint':joint,'original_angle_rad':float(original[qa]),'new_angle_rad':float(d.qpos[qa]),'com_delta_world_m':(d.subtree_com[base]-com).tolist()})
        report.append({'actor':pre,'initial_COM_world_m':com.tolist(),'contacts':contacts,'hip_distances':hipdist,'foot_floor_distances':footdist,'sensitivity':sensitivity})
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
