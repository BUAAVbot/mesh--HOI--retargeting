"""Freeze an explicitly identified N21 retarget and its paired collision scene."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from common import configure_position_pd,sha256,write_json
from prepare_n21_dynamics import canonical


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--full-effort-targets',action='store_true')
    args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False)
    if sha256(args.reference)!=args.source_sha256:raise ValueError('Source changed before freezing')
    qa=json.loads(args.reference.with_name('qa.json').read_text())
    assert qa['reference_sha256']==args.source_sha256
    assert qa['scene_model_sha256']==sha256(args.scene)
    with np.load(args.reference,allow_pickle=False) as z:
        q=z['qpos'].copy();v=z['qvel'].copy();times=z['source_times_s'].copy()
        phase=z['foot_phase'].copy();stationary=z['stationary_mask'].copy()
        assert q.shape==(209,35) and v.shape==(209,34)
        assert np.isfinite(q).all() and np.isfinite(v).all()
        assert np.all(z['source_valid']) and np.array_equal(z['source_frame_ids'],np.arange(209))
        np.testing.assert_allclose(np.diff(times),1/30,rtol=0,atol=1e-12)
    tree=ET.parse(args.scene);xml=tree.getroot();option=xml.find('option')
    option.set('timestep',str(1/300));option.set('integrator','implicitfast')
    before=mj.MjModel.from_xml_path(str(args.scene))
    actuator=xml.find('actuator')
    if actuator is None:actuator=ET.SubElement(xml,'actuator')
    assert len(actuator)==0 and before.nu==0
    for i in range(1,before.njnt):
        name=before.joint(i).name
        ET.SubElement(actuator,'motor',name='motor_'+name,joint=name,gear='1')
    tree.write(out/'scene_model.xml',encoding='utf-8',xml_declaration=True)
    model=mj.MjModel.from_xml_path(str(out/'scene_model.xml'))
    assert (model.nq,model.nv,model.nu)==(35,34,28)
    for name in ('body_mass','body_inertia','geom_type','geom_bodyid','geom_size','geom_pos',
                 'geom_quat','geom_friction','geom_contype','geom_conaffinity'):
        np.testing.assert_array_equal(getattr(model,name),getattr(before,name))
    a,b=mj.MjData(model),mj.MjData(before)
    for pose in q[[0,50,100,150,208]]:
        a.qpos[:]=b.qpos[:]=pose;mj.mj_forward(model,a);mj.mj_forward(before,b)
        np.testing.assert_allclose(a.xpos,b.xpos,atol=1e-12)
    actuation=configure_position_pd(model,mj)
    if args.full_effort_targets:
        for entry in actuation['actuators']:
            i=entry['actuator_id'];name=entry['joint_name']
            vmax=7.85 if 'ankle_' in name or 'waist_' in name else 14.66 if 'hip_' in name or 'knee_' in name else 15.71
            margin=(entry['effort_limit']+entry['kd']*vmax)/entry['kp']
            model.actuator_ctrlrange[i]=model.jnt_range[entry['joint_id']]+np.array([-margin,margin])
        expected=model.actuator_ctrlrange.copy();configure_position_pd(model,mj)
        np.testing.assert_array_equal(model.actuator_ctrlrange,expected)
    mj.mj_saveModel(model,str(out/'compiled_model.mjb'),None)
    arrays=canonical(model,q,times,phase,velocities=v)
    arrays['stationary_mask']=stationary
    np.testing.assert_array_equal(arrays['qpos_ref'],q)
    np.testing.assert_array_equal(arrays['qvel_ref'],v)
    np.savez_compressed(out/'reference_raw.npz',**arrays)
    shutil.copy2(args.reference,out/'source_reference.npz')
    shutil.copy2(args.reference.with_name('qa.json'),out/'source_qa.json')
    shutil.copy2(args.scene,out/'source_scene.xml')
    assert sha256(out/'source_reference.npz')==args.source_sha256
    write_json(out/'input_manifest.json',dict(source=str(args.reference),source_sha256=args.source_sha256,
        source_scene=str(args.scene),source_scene_sha256=sha256(args.scene),
        source_qa_passed=qa['passed_gates'],reference_sha256=sha256(out/'reference_raw.npz'),
        model_sha256=sha256(out/'compiled_model.mjb'),source_frames=209,source_fps=30.,physics_dt=1/300,
        source_poses_preserved=True,source_velocities_preserved=True,initial_qvel=v[0].tolist(),
        source_geometry_friction_and_robot_preserved=True,actuation=actuation,
        pd_target_range='full effort plus damping envelope' if args.full_effort_targets else 'joint position range',
        actual_joint_limits_unchanged=True,actual_motor_effort_caps_unchanged=True,
        initial_foot_gap_m=arrays['foot_min_distance_m'][0].tolist(),
        scene_penetration_max_m=float(arrays['scene_penetration_m'].max()),
        geometry_status='inherited paired N21 candidate; metric dimensions remain provisional'))
    print(json.dumps(json.loads((out/'input_manifest.json').read_text()),indent=2),flush=True)


if __name__=='__main__':main()
