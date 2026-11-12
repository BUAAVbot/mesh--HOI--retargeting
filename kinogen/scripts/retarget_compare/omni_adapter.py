"""Vita52 adapter for the official Holosoma/OmniRetarget constrained solver.

The algorithm is imported from a pinned official source checkout. Only unused
viewer imports are removed in a separate headless source copy. The dataset and
embodiment adapter bypasses the stock loader's unconditional object-qpos tail
write for a static scene, and corrects named-free-joint limit indexing.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types
import xml.etree.ElementTree as ET


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shared', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=172)
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--initial-iterations', type=int, default=50)
    parser.add_argument('--foot-speed-threshold', type=float, default=.01)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    begun = time.perf_counter()
    # Preserve the isolated CVXPY/Clarabel packages ahead of existing runtimes.
    sys.path.append('/data/grail-envs/retarget/lib/python3.10/site-packages')
    import numpy as np
    import mujoco as mj
    from scipy.spatial.transform import Rotation

    source_root = args.source / 'src/holosoma_retargeting'
    core_path = source_root / 'holosoma_retargeting/src/interaction_mesh_retargeter.py'
    core_text = core_path.read_text()
    removals = ['import viser ', 'import yourdfpy ', 'from viser.extras import ViserUrdf',
                'from viser_utils import create_motion_control_sliders']
    headless_lines = [line for line in core_text.splitlines()
                      if not any(line.startswith(prefix) for prefix in removals)]
    adapted_dir = args.output / 'headless_source/src'
    adapted_dir.mkdir(parents=True)
    adapted_path = adapted_dir / core_path.name
    adapted_path.write_text('\n'.join(headless_lines) + '\n')
    for name in ['utils.py', 'mujoco_utils.py']:
        (adapted_dir/name).write_bytes((core_path.parent/name).read_bytes())
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location('official_omni_headless', adapted_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = json.loads((args.shared/'source_manifest.json').read_text())
    observations = np.load(args.shared/'common_observations.npz', allow_pickle=False)
    times = observations['time_s'][:args.frames]
    joints = observations['repaired_source_joints_zup'][:len(times)]
    original_model = Path(manifest['model'])
    support = json.loads(Path(manifest['support_geometry']).read_text())
    tree = ET.parse(original_model)
    renamed = {}
    for geom in tree.getroot().find('worldbody').findall('geom'):
        name = geom.get('name', '')
        if 'seat' in name:
            renamed[name] = 'multi_boxes__' + name
            geom.set('name', renamed[name])
    native_model_path = args.output/'vita52_static_scene.xml'
    tree.write(native_model_path, encoding='unicode')
    original = mj.MjModel.from_xml_path(str(original_model))
    native = mj.MjModel.from_xml_path(str(native_model_path))
    geometry_arrays = ['geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_bodyid',
                       'geom_contype', 'geom_conaffinity', 'jnt_range', 'body_mass',
                       'body_pos', 'body_quat']
    geometry_checks = {key: bool(np.array_equal(getattr(original,key), getattr(native,key)))
                       for key in geometry_arrays}
    if not all(geometry_checks.values()):
        raise ValueError('Embodiment adapter changed shared geometry')
    names = [mj.mj_id2name(native, mj.mjtObj.mjOBJ_JOINT, i)
             for i in range(native.njnt) if native.jnt_type[i] == mj.mjtJoint.mjJNT_HINGE]
    if names != manifest['dof_names'] or native.nq != 59:
        raise ValueError('Shared Vita52 joint contract mismatch')
    mapping = {'Pelvis':'base_link', 'L_Hip':'left_hip_roll_link', 'R_Hip':'right_hip_roll_link',
               'L_Knee':'left_knee_link', 'R_Knee':'right_knee_link',
               'L_Ankle':'left_ankle_roll_link', 'R_Ankle':'right_ankle_roll_link',
               'Neck':'neck_yaw_link', 'L_Shoulder':'left_shoulder_roll_link',
               'R_Shoulder':'right_shoulder_roll_link', 'L_Elbow':'left_elbow_link',
               'R_Elbow':'right_elbow_link', 'L_Wrist':'left_wrist_pitch_link',
               'R_Wrist':'right_wrist_pitch_link'}
    source_indices = [0,1,2,4,5,7,8,12,16,17,18,19,20,21]
    constants = types.SimpleNamespace(ROBOT_URDF_FILE=str(native_model_path.with_suffix('.urdf')),
        SCENE_XML_FILE=str(native_model_path), OBJECT_NAME='multi_boxes', ROBOT_DOF=52,
        FOOT_STICKING_LINKS=['left_ankle_roll_link','right_ankle_roll_link'],
        DEMO_JOINTS=list(mapping), JOINTS_MAPPING=mapping, MANUAL_LB={}, MANUAL_UB={},
        MANUAL_COST={}, NOMINAL_TRACKING_INDICES=np.arange(7))
    # Static terrain/seat points participate in the original interaction mesh.
    cloud=[]
    a,b,c=support['ground']['z_eq_ax_by_c']
    for x in np.linspace(-1.9,1.4,6):
        for y in np.linspace(.5,2.3,5): cloud.append([x,y,a*x+b*y+c])
    for seat in support['seats']:
        yaw=seat['yaw_rad'];rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                xy=np.asarray(seat['center_xy_m'])+rot@(np.array([x,y])*seat['half_size_xy_m'])
                cloud.append([*xy,seat['top_z_m']])
    cloud=np.asarray(cloud)
    outputs=[];costs=[];frame_seconds=[];solver_rows=[]
    for actor in range(2):
        solver=module.InteractionMeshRetargeter(constants,'',visualize=False,debug=False,
             activate_foot_sticking=True,activate_obj_non_penetration=True,
             activate_joint_limits=True,foot_sticking_tolerance=.001,
             w_nominal_tracking_init=5.)
        # The stock constructor includes a named free joint in its hinge-limit
        # list. Rebuild limits from the canonical qpos addresses without
        # weakening any joint range or changing solver constraints.
        low=np.full(native.nq,-1e6);high=np.full(native.nq,1e6)
        low[3:7]=-1.;high[3:7]=1.
        for j in range(native.njnt):
            if native.jnt_type[j]==mj.mjtJoint.mjJNT_HINGE:
                qa=int(native.jnt_qposadr[j]);low[qa],high[qa]=native.jnt_range[j]
        solver.q_a_lb=low[solver.q_a_indices];solver.q_a_ub=high[solver.q_a_indices]
        target=joints[:,actor,source_indices]
        nominal=np.tile(native.qpos0,(len(times),1))
        nominal[:,:3]=joints[:,actor,0]
        lateral=joints[:,actor,1]-joints[:,actor,2]
        forward=np.cross(lateral,np.array([0.,0.,1.]))
        heading=np.arctan2(forward[:,1],forward[:,0])
        nominal[:,3:7]=Rotation.from_euler('z',heading).as_quat()[:,[3,0,1,2]]
        nominal[:,7:]=np.clip(nominal[:,7:],low[7:],high[7:])
        for j,n in enumerate(names):
            if 'hip_pitch' in n:nominal[:,7+j]=-.35
            elif 'knee_joint' in n:nominal[:,7+j]=.7
            elif 'ankle_pitch' in n:nominal[:,7+j]=-.35
            elif 'shoulder_roll' in n:nominal[:,7+j]=.2 if n.startswith('left') else -.2
        q=nominal[0].copy();rows=[];actor_cost=[];actor_seconds=[]
        foot_speed=np.linalg.norm(np.gradient(joints[:,actor,[7,8],:2],times,axis=0),axis=-1)
        sticking=foot_speed<args.foot_speed_threshold
        for frame in range(len(times)):
            start=time.perf_counter()
            vertices,tetra=module.create_interaction_mesh(np.vstack([target[frame],cloud]))
            adjacency=module.get_adjacency_list(tetra,len(vertices))
            laplacian=module.calculate_laplacian_coordinates(vertices,adjacency)
            q,cost=solver.iterate(q_locked=nominal[frame],q_n=q,
                q_t_last=rows[-1] if rows else q.copy(),target_laplacian=laplacian,
                adj_list=adjacency,obj_pts_local=cloud,
                foot_sticking={'left':bool(sticking[frame,0]),'right':bool(sticking[frame,1])},
                w_nominal_tracking=5.,q_a_nominal=nominal[frame],init_t=frame==0,
                n_iter=args.initial_iterations if frame==0 else args.iterations,frame_idx=frame)
            rows.append(q.copy());actor_cost.append(float(cost));actor_seconds.append(time.perf_counter()-start)
            if frame%10==0:print(json.dumps({'actor':actor+1,'frame':frame,'cost':float(cost)}),flush=True)
            np.savez_compressed(args.output/f'person_{actor+1:02d}_partial.npz',qpos=np.asarray(rows),time_s=times[:len(rows)])
        outputs.append(np.asarray(rows));costs.append(actor_cost);frame_seconds.append(actor_seconds)
        solver_rows.append({'actor':actor+1,'frames':len(rows),'foot_sticking_frames':sticking.sum(axis=0).tolist()})
    result=args.output/'reference.npz'
    np.savez_compressed(result,qpos=np.stack(outputs,axis=1),time_s=times,dof_names=np.asarray(names),
        frame_ids=observations['frame_ids'][:len(times)],source_repaired_frame_mask=observations['source_repaired_frame_mask'][:len(times)],
        loss=np.asarray(costs).T,frame_wall_seconds=np.asarray(frame_seconds).T)
    record={'method':'OmniRetarget official constrained interaction-mesh solver, Vita52 adapter',
        'status':'kinematic_retarget_complete_not_physical_success','official_repository':'https://github.com/amazon-far/holosoma',
        'commit':json.loads((args.source/'SOURCE_MANIFEST.json').read_text())['commit'],
        'official_core_sha256':sha(core_path),'headless_core_sha256':sha(adapted_path),
        'model_sha256':sha(original_model),'native_geometry_arrays_identical':geometry_checks,
        'support_geometry_sha256':sha(Path(manifest['support_geometry'])),'output_sha256':sha(result),
        'actors':2,'frames':len(times),'elapsed_seconds':time.perf_counter()-begun,'solver_runs':solver_rows,
        'source_shared_manifest_sha256':sha(args.shared/'source_manifest.json'),
        'adaptations':['Vita52 body-name correspondence and canonical joint bounds; no scene/world scaling',
          'Static terrain and seats keep shared geometry; seat geom names prefixed for official collision selector',
          'Headless import removes unused viewer imports only; original CVXPY/Clarabel solve/iterate untouched',
          'Dataset wrapper calls official iterate directly, avoiding stock unconditional dynamic-object qpos tail write',
          'Named free-joint bound-indexing corrected from shared compiled model; joint ranges unchanged',
          'Shared decoded/repaired joints replace stock dataset decoder; hands have no observed ground truth'],
        'limitations':['Two actors solved separately; inter-robot collisions audited downstream, not coupled optimization',
          'No dynamics or SONIC evaluation in this retargeting run','Morphology/source keypoint matches are explicit local adaptation',
          'Root heading initialized from hip axis; no independent measured metric scale']}
    (args.output/'report.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    main()
