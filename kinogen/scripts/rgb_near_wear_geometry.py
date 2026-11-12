"""Read-only geometric fit of the inferred shoe shell around Vita's foot."""
from pathlib import Path
import argparse
import json
import itertools


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m)
    ref=np.load(a.reference,allow_pickle=False)
    d.qpos[:59]=ref['qpos'][0,0];d.qpos[59:118]=ref['qpos'][0,1]
    shoe=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'shoe')
    sj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free');sq=int(m.jnt_qposadr[sj])
    d.qpos[sq:sq+7]=ref['object_target_pose'][0]
    mj.mj_forward(m,d)
    bid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_left_ankle_roll_link')
    P=d.xpos[bid].copy();R=d.xmat[bid].reshape(3,3).copy()
    foot_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==bid and m.geom_contype[g]>0]
    clouds=[]
    signs=np.asarray(list(itertools.product([-1.,1.],repeat=3)))
    for g in foot_geoms:
        centre=d.geom_xpos[g];rotation=d.geom_xmat[g].reshape(3,3);kind=int(m.geom_type[g])
        if kind==mj.mjtGeom.mjGEOM_MESH:
            mid=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid])
            world=m.mesh_vert[start:start+n]@rotation.T+centre
        elif kind==mj.mjtGeom.mjGEOM_SPHERE:
            world=signs*m.geom_size[g,0]+centre
        elif kind==mj.mjtGeom.mjGEOM_BOX:
            world=(signs*m.geom_size[g])@rotation.T+centre
        elif kind==mj.mjtGeom.mjGEOM_CAPSULE:
            axis=rotation[:,2]*m.geom_size[g,1]
            world=np.concatenate([signs*m.geom_size[g,0]+centre+axis,signs*m.geom_size[g,0]+centre-axis])
        else:raise ValueError('Unsupported foot collision primitive')
        clouds.append(world)
    world=np.concatenate(clouds);foot=(world-P)@R
    low,high=foot.min(0),foot.max(0)
    sole=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'shoe_sole_proxy')
    Q=np.empty(9);mj.mju_quat2Mat(Q,m.geom_quat[sole]);Q=Q.reshape(3,3)
    C=Q.T@m.geom_pos[sole];half=m.geom_size[sole]
    shell_low=C-half;shell_high=C+half
    # Move the sole inner top 1 mm beneath the existing robot-foot surface;
    # place toe wall 1 mm ahead of the robot toe. Open heel allows overhang.
    t=np.asarray([high[0]+.001-(shell_high[0]-.004),
                  (low[1]+high[1])/2-(shell_low[1]+shell_high[1])/2,
                  low[2]-.001-shell_high[2]])
    shoeR=R@Q.T
    shoeP=P+R@t
    quat=np.empty(4);mj.mju_mat2Quat(quat,shoeR.ravel())
    d.qpos[sq:sq+7]=np.r_[shoeP,quat];mj.mj_forward(m,d)
    shoe_geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==shoe and m.geom_contype[g]>0]
    bottoms=[float(d.geom_xpos[g,2]-abs(d.geom_xmat[g].reshape(3,3)[2])@m.geom_size[g])for g in shoe_geoms]
    minimum=float(min(bottoms));lift=max(.002-minimum,0.)
    distances=[]
    for fg in foot_geoms:
        for sg in shoe_geoms:
            dist=float(mj.mj_geomDistance(m,d,fg,sg,1.,np.zeros(6)))
            distances.append({'foot_geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,fg),
                              'shoe_geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,sg),'distance_m':dist})
    distances.sort(key=lambda x:x['distance_m'])
    whole_robot=[]
    for g in range(m.ngeom):
        bodyname=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or''
        if not bodyname.startswith(('p1_','p2_')) or not m.geom_contype[g]:continue
        for sg in shoe_geoms:
            dist=float(mj.mj_geomDistance(m,d,g,sg,.5,np.zeros(6)))
            whole_robot.append({'robot_geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),
                                'robot_body':bodyname,'shoe_geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,sg),'distance_m':dist})
    whole_robot.sort(key=lambda x:x['distance_m'])
    report={'foot_local_collision_bounds':[low.tolist(),high.tolist()],
            'foot_world_min_z_m':float(world[:,2].min()),
            'foot_rotation_world':R.tolist(),'shell_box_rotation_local':Q.tolist(),
            'shell_sole_centre_canonical':C.tolist(),'shell_sole_halfsize':half.tolist(),
            'shoe_rotation_relative_to_foot':Q.T.tolist(),
            'shoe_origin_in_foot_frame_m':t.tolist(),
            'initial_shoe_pose_world_xyz_wxyz':np.r_[shoeP,quat].tolist(),
            'initial_shoe_min_z_m':minimum,'required_foot_and_shoe_floor_lift_m':lift,
            'recommended_foot_min_z_m':float(world[:,2].min()+lift),
            'minimum_foot_shoe_distance_m':distances[0]['distance_m'],
            'closest_foot_shoe_pairs':distances[:12],
            'all_robot_shoe_min_distance_m':whole_robot[0]['distance_m'],
            'closest_all_robot_shoe_pairs':whole_robot[:12],
            'required_cavity_width_m':float(high[1]-low[1]+.002),
            'current_cavity_width_m':float(2*half[1]-.008),
            'note':'Diagnostic FK/geometry only. No active trial states or frozen assets modified.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
