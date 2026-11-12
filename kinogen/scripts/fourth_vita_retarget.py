"""Estimate Vita02a reference targets from the fourth handoff's baked SMPL-X mesh.

The bundle is data, not executable code. This reader never evaluates scripts or
uses NumPy pickle loading. The only legacy pickle is the repository's SMPL-X
triangle segmentation; it is decoded with the inert RestrictedReader.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/fourth_sim_handoff'
SEGMENTATION = ROOT / 'dependencies/dno-reference/visualize/joints2smpl/smpl_models/smplx_parts_segm.pkl'


def read_segments(np):
    class Array:
        def __setstate__(self, state):
            self.version, self.shape, self.dtype, self.fortran, self.data = state
            if self.shape != (20908,):
                raise ValueError('Unexpected segmented array size')

    class DType:
        def __init__(self, code, *unused):
            self.code = code
        def __setstate__(self, state):
            self.state = state

    class RestrictedReader(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'numpy.core.multiarray' and name == '_reconstruct':
                return lambda *unused: Array()
            if module == 'numpy' and name == 'ndarray':
                return Array
            if module == 'numpy' and name == 'dtype':
                return DType
            raise ValueError(f'Unsafe pickle global: {module}.{name}')
        def persistent_load(self, pid):
            raise ValueError('Persistent pickle IDs disabled')
    with SEGMENTATION.open('rb') as stream:
        item = RestrictedReader(stream, encoding='latin1').load()
    raw = item['segm'].data
    values = np.frombuffer(raw.encode('latin1') if isinstance(raw, str) else raw, dtype='<i8')
    if values.shape != (20908,) or np.min(values) < 0 or np.max(values) > 54:
        raise ValueError('SMPL-X segmentation does not match the supplied mesh')
    return values


def segment_vertices(faces, segments, ids, np):
    selected = np.isin(segments, ids)
    return np.unique(faces[selected].reshape(-1))


def probe(data, start, stop):
    import numpy as np
    from scipy.spatial import cKDTree
    human = np.load(data/'human/mesh_sequence_sim.npz', allow_pickle=False)
    objects = np.load(data/'objects/poses_sim.npz', allow_pickle=False)
    mesh = np.load(data/'objects/object_05/mesh.npz', allow_pickle=False)
    roots = np.load(data/'human/root_poses_sim.npz', allow_pickle=False)
    seg = read_segments(np)
    faces = human['faces']
    if not np.array_equal(faces, np.load(data/'human/smplx_params_sim.npz', allow_pickle=False)['faces']):
        raise ValueError('Mesh topology differs from SMPL-X parameters')
    hand_idx = [segment_vertices(faces, seg, range(25,40), np),
                segment_vertices(faces, seg, range(40,55), np)]
    wrist_idx = [segment_vertices(faces, seg, [20], np),
                 segment_vertices(faces, seg, [21], np)]
    print('OBJECT_MESH_KEYS', {k: (mesh[k].shape, str(mesh[k].dtype)) for k in mesh.files})
    mv = mesh['vertices']
    object_i = 4
    rows=[]
    for frame in range(start, stop):
        valid = bool(human['valid'][frame] and objects['valid'][frame,object_i])
        hands = human['hand_valid_full'][frame].tolist()
        if not valid or frame >= human['vertices_sim'].shape[0]:
            rows.append({'frame':frame,'valid':False,'hands':hands});continue
        verts = human['vertices_sim'][frame]
        T = objects['sim_from_object'][frame,object_i]
        world = mv @ T[:3,:3].T + T[:3,3]
        tree = cKDTree(world)
        d=[];near=[];wrist=[]
        for j in range(2):
            hv=verts[hand_idx[j]]
            ds,_=tree.query(hv,k=1,workers=1)
            d.append([float(np.min(ds)),float(np.quantile(ds,.1)),float(np.median(ds))])
            near.append(hv[np.argmin(ds)].tolist())
            wrist.append(np.mean(verts[wrist_idx[j]],axis=0).tolist())
        rows.append({'frame':frame,'valid':True,'hands':hands,'distance_min_q10_med_m':d,
                     'nearest_hand_vertices_world':near,'wrist_centroid_world':wrist,
                     'object_position_world':T[:3,3].tolist(),
                     'pelvis_world':roots['position_m'][frame].tolist(),
                     'object_status':int(objects['status'][frame,object_i])})
    print(json.dumps({'start':start,'stop':stop,'object_id':5,'rows':rows},allow_nan=False))


def compile_vita(np):
    import xml.etree.ElementTree as ET
    import mujoco
    urdf=ROOT/'data/vita_demo_20261026/robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf'
    xml=ET.parse(urdf).getroot()
    names=[j.get('name') for j in xml.findall('joint') if j.get('type')!='fixed']
    if len(names)!=52:raise ValueError('Expected the supplied full Vita52 URDF')
    limits={j.get('name'):j.find('limit') for j in xml.findall('joint') if j.get('type')!='fixed'}
    for mesh in xml.findall('.//mesh'):
        name=mesh.get('filename')
        path=(urdf.parent/name).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        mesh.set('filename',str(path))
    extension=xml.find('mujoco')
    if extension is None:extension=ET.SubElement(xml,'mujoco')
    compiler=extension.find('compiler')
    if compiler is None:compiler=ET.SubElement(extension,'compiler')
    compiler.attrib.update(discardvisual='false',fusestatic='false',strippath='false')
    spec=mujoco.MjSpec.from_string(ET.tostring(xml,encoding='unicode'))
    spec.body('base_link').add_freejoint(name='vita_ik_free_root')
    model=spec.compile()
    if model.nq!=59 or model.nv!=58:raise ValueError((model.nq,model.nv))
    qa=[];va=[];lo=[];hi=[];speed=[]
    for name in names:
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        qa.append(int(model.jnt_qposadr[jid]));va.append(int(model.jnt_dofadr[jid]))
        lim=limits[name]
        lo.append(float(lim.get('lower')));hi.append(float(lim.get('upper')))
        speed.append(float(lim.get('velocity')))
    bodies={n:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,n)for n in
            ('lh_hand_base_link','rh_hand_base_link','left_elbow_link','right_elbow_link',
             'left_ankle_roll_link','right_ankle_roll_link')}
    if min(bodies.values())<0:raise ValueError('Missing IK body')
    return mujoco,model,names,np.asarray(qa),np.asarray(va),np.asarray(lo),np.asarray(hi),np.asarray(speed),bodies,urdf


def fit_reference(args):
    import hashlib
    import numpy as np
    from scipy.spatial import cKDTree
    import mujoco
    data=args.data
    mesh=np.load(data/'human/mesh_sequence_sim.npz',allow_pickle=False)
    obj=np.load(data/'objects/poses_sim.npz',allow_pickle=False)
    objmesh=np.load(data/'objects/object_05/mesh.npz',allow_pickle=False)
    roots=np.load(data/'human/root_poses_sim.npz',allow_pickle=False)
    timeline=np.load(data/'timeline.npz',allow_pickle=False)
    params=np.load(data/'human/smplx_params_sim.npz',allow_pickle=False)
    seg=read_segments(np);faces=mesh['faces'];verts=mesh['vertices_sim'];omesh=objmesh['vertices']
    if args.stop>len(verts) or args.start<0:raise ValueError('Selected frame is missing human mesh')
    ids={
        'left_palm':segment_vertices(faces,seg,range(25,40),np),
        'right_palm':segment_vertices(faces,seg,range(40,55),np),
        'left_wrist':segment_vertices(faces,seg,[20],np),
        'right_wrist':segment_vertices(faces,seg,[21],np),
        'left_elbow':segment_vertices(faces,seg,[18],np),
        'right_elbow':segment_vertices(faces,seg,[19],np),
        'left_shoulder':segment_vertices(faces,seg,[16],np),
        'right_shoulder':segment_vertices(faces,seg,[17],np),
        'left_foot':segment_vertices(faces,seg,[7,10],np),
        'right_foot':segment_vertices(faces,seg,[8,11],np),
    }
    # A standing tabletop segment should have planted feet. The mesh foot
    # centroids jitter independently of real support; use one fixed support
    # target per foot and keep the raw targets for audit.
    source_foot_samples=np.asarray([[np.median(verts[f,ids['left_foot']],axis=0),
                                     np.median(verts[f,ids['right_foot']],axis=0)]
                                    for f in range(args.start,args.stop)])
    support_foot=np.median(source_foot_samples,axis=0)
    source_support_foot=support_foot.copy()
    mj,model,names,qa,va,lo,hi,speed,bodies,urdf=compile_vita(np)
    dat=mj.MjData(model)
    import xml.etree.ElementTree as ET
    scene_xml=ROOT/'outputs/fourth_vita_demo/assets/vita_object_05_dynamic.xml'
    floor_geom=ET.parse(scene_xml).getroot().find(".//geom[@name='assumed_floor']")
    if floor_geom is None:raise ValueError('Actual physics scene floor missing')
    floor_z=float(floor_geom.get('pos').split()[2])
    foot_geoms={side:[g for g in range(model.ngeom)if
                         (mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g) or '').startswith('collision_foot_'+short+'_')]
                for side,short in [('left','l'),('right','r')]}
    if min(map(len,foot_geoms.values()))<8:raise ValueError('Supplied foot collision spheres missing')
    def sole_z(side):
        return float(min(dat.geom_xpos[g,2]-model.geom_size[g,0] for g in foot_geoms[side]))
    moving_prefix=('left_shoulder','left_elbow','left_wrist','waist_') if args.right_safe_static else (
                   'left_shoulder','right_shoulder','left_elbow','right_elbow',
                   'left_wrist','right_wrist','waist_') if args.stand_seed else (
                   'left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle',
                   'left_shoulder','right_shoulder','left_elbow','right_elbow',
                   'left_wrist','right_wrist','waist_')
    arm_leg=[i for i,n in enumerate(names)if n.startswith(moving_prefix)]
    select_q=qa[arm_leg];select_v=va[arm_leg]
    collision_model=None;collision_data=None;contact_joint_qa=None;contact_joint_va=None
    collision_finger_geoms={};collision_wall_geoms=[]
    if args.finger_collision_ik:
        collision_model=mj.MjModel.from_xml_path(str(scene_xml))
        collision_data=mj.MjData(collision_model)
        if collision_model.nq!=66:raise ValueError('Expected full Vita+one free object scene')
        contact_names=[n for n in names if n.startswith(('waist_', 'left_shoulder','left_elbow','left_wrist',
            'lh_thumb','lh_index','lh_middle'))]
        contact_joint_qa=np.asarray([collision_model.jnt_qposadr[mj.mj_name2id(collision_model,mj.mjtObj.mjOBJ_JOINT,n)]
                                     for n in contact_names],dtype=int)
        contact_joint_va=np.asarray([collision_model.jnt_dofadr[mj.mj_name2id(collision_model,mj.mjtObj.mjOBJ_JOINT,n)]
                                     for n in contact_names],dtype=int)
        collision_joint_indices=np.asarray([names.index(n) for n in contact_names],dtype=int)
        for tag,name in [('palm','collision_lh_palm_mesh'),('thumb','collision_lh_thumb_distal_mesh'),
                         ('index','collision_lh_index_distal_mesh'),('middle','collision_lh_middle_distal_mesh')]:
            gid=mj.mj_name2id(collision_model,mj.mjtObj.mjOBJ_GEOM,name)
            if gid<0:raise ValueError('Missing contact geom '+name)
            collision_finger_geoms[tag]=gid
        collision_wall_geoms=[mj.mj_name2id(collision_model,mj.mjtObj.mjOBJ_GEOM,
                              f'target_object_05_wall_{i:02d}')for i in range(16)]
        if min(collision_wall_geoms)<0:raise ValueError('Missing bowl wall proxies')
    def collision_distances(robot_q,object_pose,return_witness=False):
        collision_data.qpos[:59]=robot_q;collision_data.qpos[59:66]=object_pose
        mj.mj_forward(collision_model,collision_data)
        rows=[]
        for tag,gid in collision_finger_geoms.items():
            best=None
            for wall in collision_wall_geoms:
                witness=np.zeros(6)
                distance=float(mj.mj_geomDistance(collision_model,collision_data,gid,wall,.3,witness))
                if best is None or distance<best[0]:best=(distance,witness,gid,wall)
            rows.append(best)
        return rows if return_witness else np.asarray([r[0] for r in rows])
    qout=[];rawhands=[];projhands=[];objposes=[];handdist=[];contact=[];projection=[]
    handvalid=[];objstatus=[];fkhand=[];fkfoot=[];humanfoot=[];humanelbow=[]
    ikerror=[];yaw_values=[];root_source=[];support_targets=[];sole_clearance=[]
    finger_distance_before=[];finger_distance_after=[];contact_ik_delta=[]
    palm_support_site_fk=[];palm_support_target=[];palm_support_weight=[];palm_orientation_error_deg=[]
    root_z_offset=None
    stand_qpos=None
    if args.stand_seed:
        stand=np.load(args.stand_seed,allow_pickle=False)
        stand_qpos=np.asarray(stand['qpos'][0],dtype=float).copy()
        if stand_qpos.shape!=(59,):raise ValueError('Stationary stand seed must be full Vita52+free root')
        stand_qpos[2]+=args.stand_floor_adjust_m
        if args.right_safe_static:
            # A static non-task-arm pose found by actual whole-hand/table
            # collision-geometry search: 23.7 cm initial table clearance.
            for name,value in {'right_shoulder_pitch_joint':-.5,
                               'right_shoulder_roll_joint':-.4,
                               'right_shoulder_yaw_joint':0.,
                               'right_elbow_joint':0.}.items():
                stand_qpos[qa[names.index(name)]]=value
        # Keep legs, free root and support orientation fixed. Only waist/arms
        # move; this is disclosed as a stationary-base tabletop adaptation.
        dat.qpos[:]=stand_qpos;mj.mj_forward(model,dat)
        support_foot[:]=[dat.xpos[bodies['left_ankle_roll_link']].copy(),
                         dat.xpos[bodies['right_ankle_roll_link']].copy()]
    prev=np.asarray(stand_qpos[qa]).copy() if stand_qpos is not None else np.clip(np.zeros(52),lo,hi)
    for frame in range(args.start,args.stop):
        if not bool(mesh['valid'][frame] and obj['valid'][frame,4] and np.all(mesh['hand_valid_full'][frame])):
            raise ValueError(f'Frame {frame} lacks human/object/hand observation')
        v=verts[frame]
        cent={name:np.median(v[index],axis=0)for name,index in ids.items()}
        # SMPL-X triangle group 16/17 names identify anatomical left/right.
        lateral=cent['left_shoulder']-cent['right_shoulder'];lateral[2]=0
        lateral/=max(np.linalg.norm(lateral),1e-8)
        forward=np.cross(lateral,np.array([0.,0.,1.]))
        yaw=float(np.arctan2(forward[1],forward[0]))
        cy,sy=np.cos(yaw/2),np.sin(yaw/2)
        yaw_values.append(yaw)
        T=obj['sim_from_object'][frame,4]
        ov=omesh@T[:3,:3].T+T[:3,3];tree=cKDTree(ov)
        raw=[];projected=[];dists=[];candidates=[];shift=[]
        for side in ('left','right'):
            hv=v[ids[side+'_palm']]
            ds,nearest=tree.query(hv,k=1,workers=1)
            i=int(np.argmin(ds));distance=float(ds[i])
            target=cent[side+'_palm'].copy()
            nearest_surface=ov[nearest[i]]
            delta=nearest_surface-hv[i]
            # This synthetic correction is separate from the raw observed target.
            # It is bounded and only applied to proximity-based CONTACT CANDIDATES.
            candidate=distance<=args.contact_threshold_m
            correction=delta if candidate else np.zeros(3)
            correction*=min(1.,args.projection_cap_m/max(np.linalg.norm(correction),1e-9))
            raw.append(target);projected.append(target+correction)
            dists.append(distance);candidates.append(candidate)
            shift.append(float(np.linalg.norm(correction)))
        # Root follows the pelvis and observed shoulder yaw. This is reference
        # construction, not active simulation state writing.
        root=roots['position_m'][frame].astype(float).copy()
        root_source.append(root.copy())
        if stand_qpos is None:
            root[1]+=args.root_away_y_m
            dat.qpos[:3]=root;dat.qpos[3:7]=[cy,0.,0.,sy]
        else:
            dat.qpos[:7]=stand_qpos[:7]
        dat.qpos[qa]=prev
        if root_z_offset is None:
            mj.mj_forward(model,dat)
            if stand_qpos is None:
                soles=np.asarray([sole_z('left'),sole_z('right')])
                root_z_offset=float(floor_z+.002-np.min(soles))
                # Desired ankle height is derived from the actual collision
                # sole, not the human mesh foot centroid.
                support_foot[:,2]=floor_z+.002+np.asarray([
                    dat.xpos[bodies['left_ankle_roll_link'],2]-soles[0],
                    dat.xpos[bodies['right_ankle_roll_link'],2]-soles[1]])
            else:
                root_z_offset=float(stand_qpos[2]-roots['position_m'][frame,2])
        if stand_qpos is None:dat.qpos[2]+=root_z_offset
        object_pose=np.r_[obj['position_m'][frame,4],obj['quaternion_wxyz'][frame,4]]
        support_blend=float(np.clip((args.support_release_frame-frame)/4.,0.,1.)) if args.palm_support else 0.
        left_weight=1.4*(1-support_blend)+.05*support_blend
        right_weight=.05 if args.right_safe_static else 1.4
        goals=[('lh_hand_base_link',projected[0],left_weight),('rh_hand_base_link',projected[1],right_weight),
               ('left_elbow_link',cent['left_elbow'],.16),('right_elbow_link',cent['right_elbow'],.16),
               ('left_ankle_roll_link',support_foot[0],2.5),
               ('right_ankle_roll_link',support_foot[1],2.5)]
        palm_site=np.asarray([.015,0.,.055])
        support_target=object_pose[:3]+np.asarray([0.,0.,-.001])
        palm_support_target.append(support_target)
        palm_support_weight.append(support_blend)
        for iteration in range(35):
            mj.mj_forward(model,dat)
            jac=[];errors=[]
            for bname,target,weight in goals:
                jid=bodies[bname]
                J=np.zeros((3,model.nv));R=np.zeros((3,model.nv))
                mj.mj_jacBody(model,dat,J,R,jid)
                jac.append(np.sqrt(weight)*J[:,select_v]);errors.append(np.sqrt(weight)*(target-dat.xpos[jid]))
            if support_blend>0:
                palm_id=bodies['lh_hand_base_link']
                palm_R=dat.xmat[palm_id].reshape(3,3)
                palm_world=dat.xpos[palm_id]+palm_R@palm_site
                J=np.zeros((3,model.nv));R=np.zeros((3,model.nv))
                mj.mj_jac(model,dat,J,R,palm_world,palm_id)
                jac.append(np.sqrt(8*support_blend)*J[:,select_v])
                errors.append(np.sqrt(8*support_blend)*(support_target-palm_world))
                from scipy.spatial.transform import Rotation
                palm_forward=object_pose[:3]-dat.qpos[:3];palm_forward[2]=0
                palm_forward/=max(np.linalg.norm(palm_forward),1e-8)
                desired_x=np.asarray([0.,0.,1.]);desired_z=palm_forward
                desired_y=np.cross(desired_z,desired_x)
                desired_R=np.column_stack((desired_x,desired_y,desired_z))
                orientation_error=Rotation.from_matrix(desired_R@palm_R.T).as_rotvec()
                jac.append(np.sqrt(1.2*support_blend)*R[:,select_v])
                errors.append(np.sqrt(1.2*support_blend)*orientation_error)
            A=np.vstack(jac);e=np.concatenate(errors)
            if np.max(np.abs(e))<.008:break
            H=A.T@A+.015*np.eye(len(select_v))
            step=np.linalg.solve(H,A.T@e)
            step=np.clip(step,-.16,.16)
            dat.qpos[select_q]=np.clip(dat.qpos[select_q]+step,lo[arm_leg],hi[arm_leg])
        mj.mj_forward(model,dat)
        palm_id=bodies['lh_hand_base_link']
        palm_R=dat.xmat[palm_id].reshape(3,3)
        palm_support_site_fk.append(dat.xpos[palm_id]+palm_R@palm_site)
        palm_orientation_error_deg.append(float(np.degrees(np.arccos(np.clip(palm_R[:,0]@[0.,0.,1.],-1,1)))))
        if collision_model is not None:
            before=collision_distances(dat.qpos,object_pose)
            initial_contact_q=np.asarray(dat.qpos).copy()
            # The source hand/object gap is not a contact annotation. This
            # deliberately assumed pregrasp seeks surface contact for a
            # limited early phase, with every applied joint correction saved.
            if candidates[0] and frame<args.contact_release_frame:
                for _ in range(28):
                    pairs=collision_distances(dat.qpos,object_pose,return_witness=True)
                    jac=[];errors=[]
                    for tag,(distance,witness,gid,wall) in zip(collision_finger_geoms,pairs):
                        if tag=='palm':continue  # keep the broad palm clear of the bowl wall
                        if not .003<distance<.25:continue
                        # Do not pull a penetrated geom further through the
                        # wall; physical QC decides whether it is acceptable.
                        robot_point=witness[:3];wall_point=witness[3:]
                        delta=(wall_point-robot_point)*((distance-.003)/max(distance,1e-9))
                        J=np.zeros((3,collision_model.nv));R=np.zeros((3,collision_model.nv))
                        bid=int(collision_model.geom_bodyid[gid])
                        mj.mj_jac(collision_model,collision_data,J,R,robot_point,bid)
                        weight=.65
                        jac.append(np.sqrt(weight)*J[:,contact_joint_va])
                        errors.append(np.sqrt(weight)*delta)
                    if not jac:break
                    A=np.vstack(jac);e=np.concatenate(errors)
                    H=A.T@A+.025*np.eye(len(contact_joint_qa))
                    dq=np.clip(np.linalg.solve(H,A.T@e),-.09,.09)
                    original=np.asarray(collision_data.qpos[contact_joint_qa]).copy()
                    accepted=False
                    for alpha in (1.,.5,.25,.125,.0625):
                        proposed=np.clip(original+alpha*dq,lo[collision_joint_indices],hi[collision_joint_indices])
                        collision_data.qpos[contact_joint_qa]=proposed
                        trial=collision_distances(collision_data.qpos[:59],object_pose)
                        if trial[0]>=.003 and np.min(trial[1:])>=-.002:
                            dat.qpos[qa[collision_joint_indices]]=proposed
                            accepted=True
                            break
                    if not accepted:
                        collision_data.qpos[contact_joint_qa]=original
                        break
            after=collision_distances(dat.qpos,object_pose)
            finger_distance_before.append(before);finger_distance_after.append(after)
            contact_ik_delta.append(np.asarray(dat.qpos[qa])-initial_contact_q[qa])
            mj.mj_forward(model,dat)
        err=np.asarray([np.linalg.norm(dat.xpos[bodies[b]]-t)for b,t,w in goals])
        # Preserve all 52 joints, including ten independently articulated
        # fingers per hand and the neck. Only body IK is solved here; fingers
        # remain at the manufacturer zero/reference pose rather than inventing
        # a confident grasp from weak visual data.
        current=np.asarray(dat.qpos[qa]).copy()
        if len(qout):
            # Respect per-joint URDF velocity specifications at source PTS.
            dt=float(timeline['pts_seconds'][frame]-timeline['pts_seconds'][frame-1])
            current=np.clip(current,prev-speed*dt,prev+speed*dt)
            dat.qpos[qa]=current;mj.mj_forward(model,dat)
        sole_clearance.append([sole_z('left')-floor_z,sole_z('right')-floor_z])
        prev=current
        qout.append(np.asarray(dat.qpos).copy())
        rawhands.append(raw);projhands.append(projected)
        objposes.append(np.r_[obj['position_m'][frame,4],obj['quaternion_wxyz'][frame,4]])
        handdist.append(dists);contact.append(candidates);projection.append(shift)
        handvalid.append(mesh['hand_valid_full'][frame]);objstatus.append(int(obj['status'][frame,4]))
        fkhand.append([dat.xpos[bodies['lh_hand_base_link']].copy(),dat.xpos[bodies['rh_hand_base_link']].copy()])
        fkfoot.append([dat.xpos[bodies['left_ankle_roll_link']].copy(),dat.xpos[bodies['right_ankle_roll_link']].copy()])
        humanfoot.append([cent['left_foot'],cent['right_foot']]);humanelbow.append([cent['left_elbow'],cent['right_elbow']])
        support_targets.append(support_foot.copy())
        ikerror.append(err)
    qout=np.asarray(qout);handdist=np.asarray(handdist);ikerror=np.asarray(ikerror)
    frame_ids=np.arange(args.start,args.stop,dtype=np.int32)
    time=np.asarray(timeline['pts_seconds'][frame_ids],dtype=np.float64)
    target=args.output;target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,frame_ids=frame_ids,time_s=time-time[0],source_pts_seconds=time,
        qpos=qout,root_pos=qout[:,:3],root_rot_wxyz=qout[:,3:7],dof_pos=qout[:,qa],
        dof_names=np.asarray(names),object_id=np.array(5),object_target_pose=np.asarray(objposes),
        human_hand_target_world_raw=np.asarray(rawhands),
        robot_hand_target_world_projected=np.asarray(projhands),
        left_hand_target_world=np.asarray(projhands)[:,0],
        right_hand_target_world=np.asarray(projhands)[:,1],
        source_hand_object_min_distance_m=handdist,
        contact_estimated=np.asarray(contact),contact_projection_m=np.asarray(projection),
        hand_valid=np.asarray(handvalid),object_status=np.asarray(objstatus),
        robot_hand_fk_world=np.asarray(fkhand),robot_foot_fk_world=np.asarray(fkfoot),
        human_foot_target_world=np.asarray(humanfoot),human_elbow_target_world=np.asarray(humanelbow),
        foot_support_target_world=np.asarray(support_targets),
        source_pelvis_world=np.asarray(root_source),
        root_alignment_offset_xyz_m=qout[:,:3]-np.asarray(root_source),
        root_contact_aware_alignment_xyz_m=qout[:,:3]-np.asarray(root_source),
        foot_sole_clearance_m=np.asarray(sole_clearance),physics_floor_z_m=np.asarray(floor_z),
        ik_error_m=np.asarray(ikerror),root_yaw_rad=np.asarray(yaw_values),
        finger_collision_distance_before_m=np.asarray(finger_distance_before),
        finger_collision_distance_after_m=np.asarray(finger_distance_after),
        contact_ik_joint_delta_rad=np.asarray(contact_ik_delta),
        palm_support_target_world=np.asarray(palm_support_target),
        palm_support_site_fk_world=np.asarray(palm_support_site_fk),
        palm_support_blend=np.asarray(palm_support_weight),
        palm_orientation_up_error_deg=np.asarray(palm_orientation_error_deg),
        source_left_hand_to_synthetic_palm_target_m=np.linalg.norm(np.asarray(palm_support_target)-np.asarray(rawhands)[:,0],axis=1),
        initial_pregrasp_assumed=np.asarray(bool(args.finger_collision_ik or args.palm_support)),
        target_semantics=np.asarray('SMPL-X baked mesh anatomical segment centroids; provisional contact projection; not measured contact'))
    report={'schema':'fourth_vita_retarget.v1','output':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'frames':len(frame_ids),'frame_start':args.start,'frame_stop_exclusive':args.stop,'object_id':5,
        'human_valid_all':True,'hands_valid_all':True,'object_valid_all':True,
        'object_status_codes':{str(int(k)):int(v) for k,v in zip(*np.unique(objstatus,return_counts=True))},
        'raw_hand_object_distance_min_m':np.min(handdist,axis=0).tolist(),
        'raw_hand_object_distance_median_m':np.median(handdist,axis=0).tolist(),
        'contact_candidate_frame_count':np.sum(contact,axis=0).tolist(),
        'contact_source':f'mesh proximity under {args.contact_threshold_m:g} m; synthetic target only; no measured force or verified contact',
        'contact_threshold_m':args.contact_threshold_m,
        'contact_projection_cap_m':args.projection_cap_m,
        'contact_projection_max_m':np.max(projection,axis=0).tolist(),
        'ik_hand_error_median_m':np.median(ikerror[:,:2],axis=0).tolist(),
        'ik_foot_error_median_m':np.median(ikerror[:,4:6],axis=0).tolist(),
        'root_alignment_offset_first_xyz_m':(qout[0,:3]-np.asarray(root_source)[0]).tolist(),
        'root_alignment_offset_max_norm_m':float(np.max(np.linalg.norm(qout[:,:3]-np.asarray(root_source),axis=1))),
        'foot_support_policy':'stationary seeded stand with frozen lower body and root, foot collision sole at physics floor' if stand_qpos is not None else
                              'per-foot fixed XY median observed; actual foot collision sphere sole set to physics floor plus 2mm initial clearance',
        'stationary_stand_seed':str(args.stand_seed) if args.stand_seed else None,
        'finger_collision_ik':bool(args.finger_collision_ik),
        'initial_pregrasp_assumed':bool(args.finger_collision_ik or args.palm_support),
        'palm_bottom_support_assumed':bool(args.palm_support),
        'support_release_frame':args.support_release_frame if args.palm_support else None,
        'palm_support_site_error_median_m':float(np.median(np.linalg.norm(np.asarray(palm_support_site_fk)-np.asarray(palm_support_target),axis=1))) if args.palm_support else None,
        'palm_up_orientation_error_initial_deg':palm_orientation_error_deg[0] if args.palm_support else None,
        'right_safe_static_pose_assumed':bool(args.right_safe_static),
        'contact_release_frame':args.contact_release_frame,
        'finger_collision_initial_m':np.asarray(finger_distance_before)[0].tolist() if finger_distance_before else None,
        'finger_collision_final_m':np.asarray(finger_distance_after)[0].tolist() if finger_distance_after else None,
        'contact_ik_joint_delta_max_rad':float(np.max(np.abs(contact_ik_delta))) if contact_ik_delta else 0.,
        'physics_floor_z_m':floor_z,
        'foot_sole_clearance_min_m':np.min(sole_clearance,axis=0).tolist(),
        'raw_to_support_foot_target_median_m':np.median(np.linalg.norm(source_foot_samples-support_foot[None,:,:],axis=2),axis=0).tolist(),
        'source_foot_median_world':source_support_foot.tolist(),
        'qpos_shape':list(qout.shape),'max_joint_velocity_limit_excess_rad_s':float(np.max(np.maximum(0,np.abs(np.diff(qout[:,qa],axis=0)/np.diff(time)[:,None])-speed))) if len(qout)>1 else 0.,
        'source_urdf':str(urdf),'source_data_root':str(data),'visual_mesh_segmentation':'repo smplx_parts_segm.pkl safely decoded with RestrictedReader',
        'physics_simulation':False,'fingers_fit_from_source':False,
        'limitations':['Segment centroids are proxy anatomical targets, not exact SMPL-X joints',
                       'Distance-threshold contact is a candidate, not ground truth',
                       'Frames with object status5 include partial-visibility prior assistance',
                       'Full-body reference does not by itself establish dynamic balance']}
    rep=args.report;rep.parent.mkdir(parents=True,exist_ok=True)
    rep.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA)
    p.add_argument('--probe',action='store_true')
    p.add_argument('--start',type=int,default=567)
    p.add_argument('--stop',type=int,default=627)
    p.add_argument('--root-away-y-m',type=float,default=0.)
    p.add_argument('--contact-threshold-m',type=float,default=.08)
    p.add_argument('--projection-cap-m',type=float,default=.06)
    p.add_argument('--stand-seed',type=Path)
    p.add_argument('--stand-floor-adjust-m',type=float,default=-.0025)
    p.add_argument('--finger-collision-ik',action='store_true')
    p.add_argument('--contact-release-frame',type=int,default=580)
    p.add_argument('--palm-support',action='store_true')
    p.add_argument('--support-release-frame',type=int,default=581)
    p.add_argument('--right-safe-static',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/fourth_vita_demo/retarget_reference.npz')
    p.add_argument('--report',type=Path,default=ROOT/'reports/fourth_vita_retarget.json')
    args=p.parse_args()
    if args.probe:probe(args.data,args.start,args.stop)
    else:fit_reference(args)


if __name__=='__main__':main()
