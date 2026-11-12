"""Freeze the latest N21 reference and replace the old supports with the stair candidate."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from common import classify_robot_static_bodies,configure_position_pd,sha256,write_json


BASE=Path('/data/yihang.bai/training/hdmi_vita_mjlab')
SOURCE=BASE/'outputs/umr_n21_descent_20261105.BvuV0o/smooth_targets_strong_r3/alpha_1/reference.npz'
SCENE=BASE/'outputs/n21_23_takeover_20261103/N21_source_yaw_standing_scene/scene_model.xml'
GEOMETRY=BASE/'outputs/n21_planar_asset_20261105.0DeGWk'


def foot_distances(model,data,feet,scene):
    distances=np.full(2,.3);points=np.zeros((2,3))
    for side,geoms in enumerate(feet):
        for g in geoms:
            for h in scene:
                ends=np.zeros(6)
                gap=mj.mj_geomDistance(model,data,g,h,.3,ends)
                if gap<distances[side]:
                    distances[side]=gap;points[side]=(ends[:3]+ends[3:])/2
    return distances,points


def canonical(model,poses,times,phase,velocities=None):
    robot,static,bodies=classify_robot_static_bodies(model,mj)
    feet=[[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in robot
        and ('foot' in (model.geom(g).name or '') or 'toe' in (model.geom(g).name or ''))
        and token in (model.geom(g).name or '')] for token in ('_l_','_r_')]
    assert all(feet)
    scene=[g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in static]
    n=len(poses);v=np.zeros((n,model.nv))
    for i in range(1,n):
        first,last=max(0,i-1),min(n-1,i+1)
        mj.mj_differentiatePos(model,v[i],times[last]-times[first],poses[first],poses[last])
    if velocities is not None:
        v=np.asarray(velocities,dtype=float).copy()
        if v.shape!=(n,model.nv) or not np.isfinite(v).all():
            raise ValueError('Source velocities must be finite and match the model/clock')
    pos=[];quat=[];linear=[];angular=[];gap=[];anchors=[];depth=[]
    data=mj.MjData(model)
    for q,velocity in zip(poses,v):
        data.qpos[:]=q;data.qvel[:]=velocity;mj.mj_forward(model,data)
        pos.append(data.xpos[bodies].copy());quat.append(data.xquat[bodies].copy())
        linear.append(data.cvel[bodies,3:].copy());angular.append(data.cvel[bodies,:3].copy())
        distance,point=foot_distances(model,data,feet,scene)
        gap.append(distance);anchors.append(point)
        depth.append(max([0.]+[-float(c.dist) for c in data.contact
            if (int(model.geom_bodyid[c.geom1]) in robot)!=(int(model.geom_bodyid[c.geom2]) in robot)]))
    gap=np.array(gap)
    names=np.array([model.body(b).name.split('/')[-1] for b in bodies])
    jointnames=np.array([model.joint(i).name for i in range(1,model.njnt)])
    adrs=model.jnt_qposadr[model.actuator_trnid[:,0]]
    return dict(qpos_ref=poses,qvel_ref=v,ctrl_ref=poses[:,adrs],
        qpos=poses,qvel=v,ctrl=poses[:,adrs],
        contact_ref=np.isin(phase,['flat','heel','toe']).astype(float),
        contact_pos_ref=np.array(anchors),foot_phase=phase,
        joint_pos=poses[:,7:],joint_vel=v[:,6:],joint_names=jointnames,body_names=names,
        body_pos_w=np.array(pos),body_quat_w=np.array(quat),
        body_lin_vel_w=np.array(linear),body_ang_vel_w=np.array(angular),
        foot_min_distance_m=gap,scene_penetration_m=np.array(depth),
        fps=np.array([30.]),dt=np.array([1/30]),source_times_s=times,
        source_frame_ids=np.arange(n),source_valid=np.ones(n,bool))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=True)
    assert sha256(SOURCE)=='19d71482c52c3717d33bdeb69e95c4568937a0710d8c7cd1f0ac93cd5367719b'
    with np.load(SOURCE,allow_pickle=False) as source:
        poses=source['qpos'].copy();times=source['source_times_s'].copy();phase=source['foot_phase'].copy()
        stationary=source['stationary_mask'].copy()
        assert poses.shape==(209,35) and np.all(source['source_valid'])
        assert np.allclose(np.diff(times),1/30) and np.array_equal(source['source_frame_ids'],np.arange(209))
    old=mj.MjModel.from_xml_path(str(SCENE))
    tree=ET.parse(SCENE);root=tree.getroot();world=root.find('worldbody')
    robot=next(e for e in world if e.tag=='body' and e.get('name')=='base_link')
    for element in list(world):
        if element is not robot:world.remove(element)
    # Retain only assets the robot actually uses; old scene collision parts are removed.
    asset=root.find('asset')
    if asset is None:asset=ET.SubElement(root,'asset')
    mesh_names={g.get('mesh') for g in robot.iter('geom') if g.get('mesh')}
    for element in list(asset):
        if element.tag=='mesh' and element.get('name') not in mesh_names:asset.remove(element)
    candidate=ET.parse(GEOMETRY/'candidate_collision.xml').getroot()
    for element in candidate.find('asset'):
        copied=deepcopy(element);copied.set('name','n21_new_'+element.get('name'));asset.append(copied)
    for i,element in enumerate(candidate.find('worldbody')):
        copied=deepcopy(element);copied.set('name',f'n21_new_stair_{i:03d}')
        copied.set('mesh','n21_new_'+element.get('mesh'))
        copied.set('friction','.8 .005 .0001');world.append(copied)
    option=root.find('option');option.set('timestep',str(1/300));option.set('integrator','implicitfast')
    actuator=root.find('actuator')
    if actuator is None:actuator=ET.SubElement(root,'actuator')
    assert len(actuator)==0
    for joint in list(robot.iter('joint')):
        ET.SubElement(actuator,'motor',name='motor_'+joint.get('name'),joint=joint.get('name'),gear='1')
    xml=out/'scene_model.xml';tree.write(xml,encoding='utf-8',xml_declaration=True)
    model=mj.MjModel.from_xml_path(str(xml))
    assert (model.nq,model.nv,model.nu)==(35,34,28)
    assert np.array_equal(old.body_mass,model.body_mass)
    assert np.allclose(old.body_inertia,model.body_inertia)
    a,b=mj.MjData(old),mj.MjData(model)
    for q in poses[[0,50,100,150,208]]:
        a.qpos[:]=b.qpos[:]=q;mj.mj_forward(old,a);mj.mj_forward(model,b)
        np.testing.assert_allclose(a.xpos,b.xpos,atol=1e-12)
    act=configure_position_pd(model,mj)
    mj.mj_saveModel(model,str(out/'compiled_model.mjb'),None)
    arrays=canonical(model,poses,times,phase)
    arrays['stationary_mask']=stationary
    np.testing.assert_array_equal(arrays['qpos_ref'],poses)
    np.savez_compressed(out/'reference_raw.npz',**arrays)
    shutil.copy2(SOURCE,out/'source_reference.npz')
    shutil.copy2(GEOMETRY/'candidate_geometry.npz',out/'candidate_geometry.npz')
    report=dict(source=str(SOURCE),source_sha256=sha256(SOURCE),source_frames=209,source_fps=30.,
        duration_s=float(times[-1]-times[0]),geometry=str(GEOMETRY/'candidate_collision.xml'),
        geometry_sha256=sha256(GEOMETRY/'candidate_collision.xml'),
        model_sha256=sha256(out/'compiled_model.mjb'),reference_sha256=sha256(out/'reference_raw.npz'),
        removed_old_static_geometry=True,collision_parts=len(candidate.find('worldbody')),
        robot_kinematics_mass_inertia_preserved=True,actuation=act,
        physics_dt=1/300,zero_initial_velocity=True,source_poses_unchanged=True,
        raw_scene_penetration_max_m=float(np.max(arrays['scene_penetration_m'])),
        raw_foot_gap_initial_m=arrays['foot_min_distance_m'][0].tolist(),
        raw_stance_abs_gap_p95_m=float(np.quantile(np.abs(arrays['foot_min_distance_m'][arrays['contact_ref']>.5]),.95)),
        geometry_warnings=json.loads((GEOMETRY/'report.json').read_text())['warnings'])
    write_json(out/'input_manifest.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
