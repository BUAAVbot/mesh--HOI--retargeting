#!/usr/bin/env python3
"""CPU audit of source bindings, mixed-unit linkage, independent FK and decoded exports."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import hashlib
import json
import os
import traceback
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/user_reconstruction_v2'
VIDEO_SHA='035c5852f9580cbbf5d612e2ad93a7f80065426a34ce355fb87db43876b20d6b'
URDF_SHA='71bc895263e495a3ea01bbb9626458ac5a15a81387bf6f9a8afaa1e700bff7b6'
CAMERA_SHA='813f64ca29c7199cb033c020e6d9b8ada1bc2c77313946baf53585fe50247b5a'
BOX_SHA='a9c5b1f274997f86382f24ba09a67abe84ae4231370888f05f254a2d49c415fa'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def record(path):
    return {'path':str(Path(path).resolve().relative_to(ROOT.resolve())).replace('\\','/'),'bytes':Path(path).stat().st_size,'sha256':sha(path)}


def require(condition,message):
    if not condition:raise RuntimeError(message)


def verify(report,args):
    import numpy as np
    from scipy.spatial.transform import Rotation
    import av
    import trimesh
    from PIL import Image,ImageDraw
    from user_rohand_coupling import ROHandCoupling
    contact=args.run_variant=='contact'
    names=('camera_refinement','body_trajectory','box_trajectory','hand_trajectory_wrist_refined','full_trajectory')
    paths={name:OUT/(name+'.npz') for name in names}
    paths['body_trajectory']=args.body_file or OUT/('body_trajectory_contact.npz' if contact else 'body_trajectory.npz')
    paths['hand_trajectory_wrist_refined']=args.hand_file or OUT/('hand_trajectory_contact.npz' if contact else 'hand_trajectory_wrist_refined.npz')
    paths['full_trajectory']=OUT/('full_trajectory_contact.npz' if contact else 'full_trajectory.npz')
    body_report_name='user-arm-contact.json' if contact else 'user-body-optimization.json'
    hand_report_name='user-hand-optimization_contact.json' if contact else 'user-hand-optimization.json'
    scene_report_name='user-full-scene-contact.json' if contact else 'user-full-scene.json'
    require(all(p.exists() for p in paths.values()),'Incomplete final exports: '+str([str(p) for p in paths.values() if not p.exists()]))
    def load(p):
        with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
    camera,body,box,hand,full=[load(paths[n]) for n in names]
    hashes={n:sha(p) for n,p in paths.items()}
    video=ROOT/'data/video_generation/candidate_scene/generated.mp4'
    scene_path=ROOT/'data/video_generation/candidate_scene/scene_rohand.json'
    scene=json.loads(scene_path.read_text());urdf=ROOT/scene['robot']['urdf'];xml=ET.parse(urdf).getroot()
    require(sha(video)==VIDEO_SHA,'Source video changed')
    require(sha(urdf)==URDF_SHA==scene['robot']['sha256'],'Unexpected five-finger model')
    require(hashes['camera_refinement']==CAMERA_SHA,'Frozen camera was replaced')
    require(hashes['box_trajectory']==BOX_SHA,'Active box must remain the frozen v2, not candidate v3')
    require(np.array_equal(box['box_size_xyz_m'],np.array([.28,.34,.24])) and np.array_equal(box['box_size_xyz_m'],scene['box']['size_xyz_m']),'Known box size changed')
    require(bool(body['finger_joint_positions_included'])==contact,'Body finger-state inclusion semantics mismatch')
    with np.load(ROOT/'outputs/user_reconstruction/input.npz') as z:
        times=z['timestamps'];require(str(z['video_sha256'])==VIDEO_SHA,'Input/video binding mismatch')
    require(times.shape==(121,) and np.allclose(times,np.arange(121)/24,atol=1e-10),'Input time axis')
    for name,data in [('body',body),('box',box),('hand',hand),('full',full)]:
        require(np.array_equal(data['timestamps'],times),name+' time axis mismatch')
    require(np.array_equal(hand['frame_indices'],np.arange(121)),'Hand contains sparse debug frames')
    for name,data in [('body',body),('hand',hand),('box',box)]:
        require(str(data['camera_sha256'])==CAMERA_SHA,name+' camera SHA mismatch')
    require(np.array_equal(body['camera_P_video'],camera['P_refined']) and np.array_equal(box['projection_video'],camera['P_refined']),'Component camera matrices differ')
    require(np.array_equal(camera['P_video'],camera['P_refined']),'Camera aliases differ')
    camera_numeric_checks=[]
    for v in range(2):
        R,t,K,H=[camera[k][v] for k in ('R','t','K','H')]
        # Authored PyTorch3D camera entries originated as float32. This is a
        # serialization check, separate from unchanged physical quality gates.
        numeric=8*np.finfo(np.float32).eps
        require(np.allclose(R.T@R,np.eye(3),atol=numeric,rtol=0) and abs(np.linalg.det(R)-1)<numeric,'Improper camera rotation beyond source precision')
        require(np.allclose(H@K@np.column_stack((R,t)),camera['P_refined'][v],atol=1e-9),'Camera decomposition mismatch')
        camera_numeric_checks.append({'view':v,'orthogonality_max_abs_error':float(np.abs(R.T@R-np.eye(3)).max()),'determinant_error':float(abs(np.linalg.det(R)-1)),'source_precision_tolerance':float(numeric)})
    require(camera['frame_camera_valid'].shape==(121,2),'Camera validity axis')
    for data in (body,hand):require(str(data['source_video_sha256'])==VIDEO_SHA,'Component source video SHA')
    require(str(hand['source_body_sha256'])==hashes['body_trajectory']==str(hand['body_sha256']),'Stale wrist refinement source body')
    require(str(hand['box_sha256'])==BOX_SHA,'Stale hand contact box')
    require(str(hand['urdf_sha256'])==URDF_SHA,'Hand fitted a different robot model')
    coupling=ROHandCoupling(ROOT)
    require(str(hand['coupling_math_sha256'])==sha(coupling.math_path),'Hand coupling math source mismatch')
    bn=list(map(str,body['joint_names']));hn=list(map(str,hand['joint_names']));fn=list(map(str,full['joint_names']))
    require(len(bn)==29 and len(set(bn))==29 and len(hn)==50 and len(set(hn))==50,'Joint name/cardinality mismatch')
    require(full['joint_positions'].shape==(121,79),'Full mixed-unit state must have 79 entries')
    require(len(fn)==79 and set(fn)==set(bn+hn),'Full joint name mismatch')
    bq=body['joint_positions_rad'];require(bq.shape==(121,29) and np.isfinite(bq).all(),'Body angles incomplete')
    parent_body_finger=None;parent_records=[];resolved_parents={}
    def hand_parent(checksum):
        if checksum in resolved_parents:return resolved_parents[checksum]
        for path in (OUT/'hand_trajectory_velocity_refined.npz',OUT/'hand_trajectory_wrist_refined.npz'):
            if path.exists() and sha(path)==checksum:
                d=load(path);require(str(d['camera_sha256'])==CAMERA_SHA and str(d['source_video_sha256'])==VIDEO_SHA,'Parent hand source mismatch');require(np.array_equal(d['timestamps'],times),'Parent hand timeline mismatch');require(str(d['urdf_sha256'])==URDF_SHA and str(d['coupling_math_sha256'])==sha(coupling.math_path),'Parent hand model/math mismatch');parent_records.append(record(path));resolved_parents[checksum]=d
                ancestor=str(d.get('source_hand_sha256',''))
                if ancestor:
                    require(ancestor!=checksum,'Self-referencing hand parent');hand_parent(ancestor)
                return d
        raise RuntimeError('Cannot resolve declared extra parent hand SHA '+checksum)
    if contact:
        baseline_path=OUT/'body_trajectory.npz';baseline=load(baseline_path);base_names=list(map(str,baseline['joint_names']))
        require(str(body['source_body_sha256'])==sha(baseline_path),'Contact body parent D mismatch')
        require(str(body['source_box_sha256'])==BOX_SHA and str(body['source_camera_sha256'])==CAMERA_SHA,'Contact extra source bindings mismatch')
        unchanged=[i for i,n in enumerate(bn) if not any(k in n for k in ('shoulder_','elbow_','wrist_'))]
        require(len(unchanged)==15,'Expected only14 arm coordinates mutable in C')
        require(np.array_equal(bq[:,unchanged],baseline['joint_positions_rad'][:,[base_names.index(bn[i]) for i in unchanged]]),'Contact changed legs/waist')
        for k in ('base_xyz_m','base_quaternion_xyzw'):require(np.array_equal(body[k],baseline[k]),'Contact changed floating base')
        parent_body_finger=hand_parent(str(body['source_hand_sha256']))
        require(np.array_equal(body['finger_actuator_positions'],parent_body_finger['actuator_positions']),'C silently changed fixed parent hand controls')
        require(np.array_equal(body['finger_joint_positions'],parent_body_finger['joint_positions']) and np.array_equal(body['finger_joint_units'],parent_body_finger['joint_units']),'C did not preserve explicit coupled parent finger states')
        require(np.array_equal(body['frame_indices'],np.arange(121)),'Contact body is a sparse debug artifact')
        require(str(parent_body_finger['source_body_sha256'])==sha(baseline_path),'Parent hand was fitted to another source body')
        hand_parent(str(hand['source_hand_sha256']))
        require(str(hand['source_hand_sha256'])==str(body['source_hand_sha256']),'C/newH used different velocity-hand parents')
        parent_records.append(record(baseline_path))
    refined=hand['body_joint_positions_rad_refined'][:,[list(map(str,hand['body_joint_names'])).index(n) for n in bn]]
    nonyaw=[i for i,n in enumerate(bn) if n not in ('left_wrist_yaw_joint','right_wrist_yaw_joint')]
    require(np.array_equal(refined[:,nonyaw],bq[:,nonyaw]),'Hand silently changed non-yaw body angles')
    for sideindex,side in enumerate(('left','right')):
        k=bn.index(side+'_wrist_yaw_joint');unaccepted=~hand['wrist_refinement_accepted'][:,sideindex].astype(bool)
        require(np.array_equal(refined[unaccepted,k],bq[unaccepted,k]),'Unaccepted wrist yaw changed')
    if contact:
        require(hand['new_hand_candidate_accepted'].shape==(121,2),'Contact acceptance axis')
        pnames=list(map(str,parent_body_finger['actuator_names']))
        for sideindex,side in enumerate(('left','right')):
            rejected=~hand['new_hand_candidate_accepted'][:,sideindex].astype(bool)
            controls_side=[i for i,n in enumerate(map(str,hand['actuator_names'])) if n.startswith(side+'_')]
            source_columns=[pnames.index(str(hand['actuator_names'][i])) for i in controls_side]
            require(np.array_equal(hand['actuator_positions'][rejected][:,controls_side],parent_body_finger['actuator_positions'][rejected][:,source_columns]),'Rejected contact side changed source actuator controls')
            k=bn.index(side+'_wrist_yaw_joint');require(np.array_equal(refined[rejected,k],bq[rejected,k]),'Rejected contact side changed selected-body yaw')
        before_dense,after_dense=hand['dense_collision_metrics_before'],hand['dense_collision_metrics_after']
        require(before_dense.shape==after_dense.shape==(121,2,3) and np.isfinite(before_dense).all() and np.isfinite(after_dense).all(),'Contact dense sample metrics incomplete')
        dense_numeric=np.array([1e-6,1e-7,1e-7])  # Explicit source optimizer's numerical non-regression contract.
        require(np.all(after_dense<=before_dense+dense_numeric+1e-12),'Contact candidate degraded a declared dense collision metric')
    require(bool(hand['body_wrist_origins_fixed']),'Yaw stage allowed free wrist translation')
    require(np.nanmax(np.abs(hand['wrist_origin_world_shift_m']))<1e-9,'Yaw-only hand stage moved wrist origins')
    require(np.array_equal(hand['wrist_keypoint_error_before_px'],hand['wrist_keypoint_error_after_px'],equal_nan=True),'Yaw-only stage changed positional wrist-keypoint errors')
    accepted=hand['wrist_refinement_accepted'].astype(bool)
    before,after=hand['heldout_edge_mean_before_px'],hand['heldout_edge_mean_after_px']
    require(np.isfinite(before[accepted]).all() and np.isfinite(after[accepted]).all(),'Accepted yaw has missing image validation')
    require(np.all(after[accepted]<=before[accepted]+.250001),'Accepted yaw violates image validation contract')
    require(np.array_equal(full['body_joint_positions_rad_before_wrist_refinement'],bq[:,[bn.index(n) for n in fn[:29]]]),'Exporter original-body binding')
    require(np.array_equal(full['body_joint_positions_rad'],refined[:,[bn.index(n) for n in fn[:29]]]),'Exporter did not use explicit refined body')
    require(bool(full['wrist_refinement_used']),'Wrist refinement was silently omitted')
    actuator_names=list(map(str,hand['actuator_names']));controls=hand['actuator_positions'][:,[actuator_names.index(n) for n in coupling.actuator_names]]
    require(controls.shape==(121,12) and np.isfinite(controls).all(),'Missing physical hand controls')
    require(np.all(controls>=np.array(coupling.lower)-1e-12) and np.all(controls<=np.array(coupling.upper)+1e-12),'Official actuator bounds violated')
    expanded=coupling.expand_array(controls)
    require(np.allclose(hand['joint_positions'][:,[hn.index(n) for n in coupling.joint_names]],expanded,atol=1e-10,rtol=0),'Official nonlinear coupling mismatch')
    expected_units=dict(zip(coupling.joint_names,coupling.joint_units));expected_units.update({n:'rad' for n in bn})
    require(list(map(str,full['joint_units']))==[expected_units[n] for n in fn],'Mixed joint units mislabelled')
    require(list(map(str,hand['joint_units']))==[expected_units[n] for n in hn],'Hand mixed units mismatch')
    require(list(map(str,hand['actuator_units']))==[dict(zip(coupling.actuator_names,coupling.actuator_units))[n] for n in actuator_names],'Actuator units mismatch')
    require(np.array_equal(full['hand_actuator_positions'],controls),'Full controls mismatch')
    require(list(map(str,full['hand_actuator_names']))==coupling.actuator_names,'Full actuator order')
    require(list(map(str,full['hand_actuator_units']))==coupling.actuator_units,'Full actuator units')
    fullq=full['joint_positions'];require(np.isfinite(fullq).all(),'Nonfinite full state')
    for n in bn:require(np.array_equal(fullq[:,fn.index(n)],refined[:,bn.index(n)]),'Body state combination mismatch: '+n)
    for n in hn:require(np.array_equal(fullq[:,fn.index(n)],hand['joint_positions'][:,hn.index(n)]),'Finger state combination mismatch: '+n)
    joints={j.attrib['name']:j for j in xml.findall('joint')}
    unbounded=[]
    for n in fn:
        limit=joints[n].find('limit')
        if limit is None:
            require(joints[n].attrib['type']=='continuous' and n in hn,'Missing required model joint bounds: '+n)
            unbounded.append(n);continue
        lo,hi=float(limit.attrib['lower']),float(limit.attrib['upper']);q=fullq[:,fn.index(n)]
        require(np.all(q>=lo-1e-8) and np.all(q<=hi+1e-8),'Joint bounds: '+n)
    # Independent XML FK; no call to the exporter or G1Kinematics FK routine.
    links=[x.attrib['name'] for x in xml.findall('link')];ordered=[];todo=list(joints.values());known={'pelvis'}
    while todo:
        done=[]
        for j in todo:
            if j.find('parent').attrib['link'] in known:ordered.append(j);known.add(j.find('child').attrib['link']);done.append(j)
        require(bool(done),'URDF has disconnected or cyclic kinematics');todo=[j for j in todo if j not in done]
    def origin(j):
        o=j.find('origin');T=np.eye(4)
        if o is not None:T[:3,3]=np.fromstring(o.attrib.get('xyz','0 0 0'),sep=' ');T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
        return T
    fixed={j.attrib['name']:origin(j) for j in ordered}
    def fk(base,quat,values):
        result={'pelvis':np.eye(4)};result['pelvis'][:3,:3]=Rotation.from_quat(quat).as_matrix();result['pelvis'][:3,3]=base
        for j in ordered:
            motion=np.eye(4);kind=j.attrib['type'];name=j.attrib['name']
            if kind!='fixed':
                axis=np.fromstring(j.find('axis').attrib.get('xyz','1 0 0'),sep=' ')
                require(np.linalg.norm(axis)>0,'Zero URDF joint axis');axis=axis/np.linalg.norm(axis)
                if kind in ('revolute','continuous'):motion[:3,:3]=Rotation.from_rotvec(axis*values[name]).as_matrix()
                elif kind=='prismatic':motion[:3,3]=axis*values[name]
                else:raise RuntimeError('Unknown URDF motion')
            result[j.find('child').attrib['link']]=result[j.find('parent').attrib['link']]@fixed[name]@motion
        return result
    full_links=list(map(str,full['link_names']));require(set(full_links)==set(links),'Full mesh skeleton differs from URDF')
    require(full['link_world_transforms'].shape==(121,len(links),4,4),'Incomplete full skeleton transforms')
    require(np.array_equal(full['base_xyz_m'],body['base_xyz_m']) and np.array_equal(full['base_quaternion_xyzw'],body['base_quaternion_xyzw']),'Base drift introduced by hand/export')
    require(np.allclose(np.linalg.norm(full['base_quaternion_xyzw'],axis=1),1,atol=1e-8),'Invalid base quaternions')
    maximum=0.;body_maximum=0.;body_links=list(map(str,body['link_names']))
    q0=scene['robot'].get('joint_positions',scene['robot']['joint_positions_rad'])
    for f in range(121):
        calc=fk(full['base_xyz_m'][f],full['base_quaternion_xyzw'][f],dict(zip(fn,fullq[f])))
        expected=np.stack([calc[n] for n in full_links]);maximum=max(maximum,float(np.abs(expected-full['link_world_transforms'][f]).max()))
        if parent_body_finger is None:vals={n:float(q0.get(n,0)) for n in hn}
        else:vals=dict(zip(map(str,parent_body_finger['joint_names']),parent_body_finger['joint_positions'][f]))
        vals.update(dict(zip(bn,bq[f])))
        calc_body=fk(body['base_xyz_m'][f],body['base_quaternion_xyzw'][f],vals)
        body_maximum=max(body_maximum,float(np.abs(np.stack([calc_body[n] for n in body_links])-body['link_world_transforms'][f]).max()))
    require(maximum<1e-7 and body_maximum<1e-7,'Independent XML FK differs from saved combined/body skeleton')
    require(np.array_equal(full['box_center_xyz_m'],box['center_xyz_m']) and np.array_equal(full['box_quaternion_xyzw'],box['quaternion_xyzw']),'Export changed box poses')
    for key,shape in [('joint_image_supported',(121,29)),('candidate_valid',(121,))]:require(body[key].shape==shape,'Body evidence axis '+key)
    require(not body['joint_image_supported'][:,[i for i,n in enumerate(bn) if 'wrist_' in n]].any(),'Body prior-only wrist angles promoted to measured support')
    for key,shape in [('observed',(121,50)),('uncertain',(121,50)),('actuator_observed',(121,12)),('actuator_uncertain',(121,12)),('side_valid',(121,2)),('valid',(121,))]:require(hand[key].shape==shape,'Hand evidence axis '+key)
    require(np.array_equal(hand['uncertain'],~hand['observed']) and np.array_equal(hand['actuator_uncertain'],~hand['actuator_observed']),'Uncertainty was erased')
    require(np.array_equal(hand['prior_only'],~hand['observed']),'Finite prior values promoted to observation')
    require(np.array_equal(hand['valid'],hand['side_valid'].all(axis=1)),'Hand side/frame validity mismatch')
    require(np.array_equal(hand['contact_geometry_uncertain'],~box['valid']),'Contact uncertainty ignores failed box geometry')
    for target,src in [('body_valid',body['candidate_valid']),('hand_valid',hand['valid']),('box_valid',box['valid']),('body_observed',body['joint_image_supported']),('hand_joint_observed',hand['observed']),('hand_actuator_observed',hand['actuator_observed'])]:require(np.array_equal(full[target],src),'Exporter promoted evidence: '+target)
    require(not bool(full['strict_metric_reconstruction_valid']),'Strict physical reconstruction cannot be declared valid')
    require(np.isnan(box['center_valid_only_xyz_m'][~box['valid']]).all(),'Invalid box positions were relabelled valid')
    # Inspect actual source mesh assets and five separate finger branches per side.
    meshfiles={};mesh_cache={};fingers={};collision_parts=[];ground_parts=[]
    for side in ('left','right'):
        fingers[side]={}
        for code in ('th','if','mf','rf','lf'):
            group=[n for n in links if n.startswith(side+'_hand_ro_'+code+'_')]
            require(len(group)>=4,'Missing physical finger chain: '+side+code);fingers[side][code]=len(group)
    for link in xml.findall('link'):
        for visual in link.findall('visual'):
            mesh=visual.find('geometry/mesh')
            if mesh is None:continue
            filename=mesh.attrib['filename']
            if filename.startswith('package://'):
                package,relative=filename[10:].split('/',1);bases={'g1_description':ROOT/'dependencies/unitree-ros/robots/g1_description','rohand_urdf':ROOT/'dependencies/rohand-urdf'};p=bases[package]/relative
            else:p=urdf.parent/filename
            if str(p) not in meshfiles:
                m=trimesh.load_mesh(p,process=False);require(len(m.vertices)>0 and len(m.faces)>0 and np.isfinite(m.vertices).all(),'Invalid actual source mesh')
                meshfiles[str(p)]=record(p);mesh_cache[str(p)]=m
            m=mesh_cache[str(p)];scale=np.fromstring(mesh.attrib.get('scale','1 1 1'),sep=' ');V=np.asarray(m.vertices,dtype=float)*scale
            unique=np.unique(V,axis=0);M=origin(visual);ground_parts.append((full_links.index(link.attrib['name']),unique@M[:3,:3].T+M[:3,3]))
            if '_hand_ro_' in link.attrib['name']:
                centers=V[np.asarray(m.faces)].mean(axis=1)
                points=np.vstack((unique,centers))@M[:3,:3].T+M[:3,3]
                collision_parts.append((full_links.index(link.attrib['name']),points,len(unique),len(centers),0 if link.attrib['name'].startswith('left_') else 1))
    reports={n:json.loads((ROOT/'reports'/n).read_text()) for n in (body_report_name,hand_report_name,scene_report_name,'user-stereo-refinement.json','user-box-optimization.json')}
    scene_report=reports[scene_report_name];require(scene_report['strict_metric_reconstruction_valid'] is False,'Scene report falsely promotes physical validity')
    require(scene_report['urdf_sha256']==URDF_SHA,'Scene report robot mismatch')
    for name,p in paths.items():
        if name=='full_trajectory':continue
        key='hand_trajectory' if name=='hand_trajectory_wrist_refined' else name
        binding=scene_report['inputs'][key];require(sha(ROOT/binding['path'])==binding['sha256']==hashes[name],'Stale scene component binding '+name)
    for binding in scene_report['outputs']:require(sha(ROOT/binding['path'])==binding['sha256'],'Scene output changed after renderer audit')
    animation_name='scene_animated_contact.glb' if contact else 'scene_animated.glb'
    animation_records=[record(ROOT/x['path']) for x in scene_report['outputs'] if Path(x['path']).name==animation_name]
    if contact:require(len(animation_records)==1,'Contact export lacks its SHA-bound animated GLB')
    animation_audit=scene_report.get('animated_glb_audit')
    if animation_records:
        require(animation_audit is not None and animation_audit['frames']==121 and animation_audit['animated_nodes']>0,'Animated scene audit incomplete')
        require(animation_audit['serialized_transform_max_abs_error']<=2e-6 and animation_audit['independent_trimesh_load_passed'] is True,'Animated scene serialized TF audit failed')
    for p,s in reports[body_report_name]['inputs'].items():require(sha(ROOT/p)==s,'Stale body input '+p)
    for p,s in reports[hand_report_name]['files'].items():require(sha(ROOT/p)==s,'Stale hand input '+p)
    require(reports[body_report_name]['output']['sha256']==hashes['body_trajectory'],'Body report is stale')
    glbs=[]
    for f in (0,60,120):
        path=OUT/(f'scene_contact_frame_{f:03d}.glb' if contact else f'scene_frame_{f:03d}.glb');s=trimesh.load(path,force='scene',process=False);require(len(s.geometry)>0,'Empty GLB')
        facecount=sum(len(m.faces) for m in s.geometry.values());require(facecount==scene_report['renderer_reference_check']['render_faces'],'GLB mesh skeleton/triangle export incomplete')
        require(all(np.isfinite(m.vertices).all() for m in s.geometry.values()),'GLB nonfinite vertices');glbs.append({**record(path),'faces':facecount})
    sheet=Image.new('RGB',(1440,2*388),'#101827');draw=ImageDraw.Draw(sheet);videos=[]
    prefix='user_v2_contact_full' if contact else 'user_v2_full'
    for row,(name,size) in enumerate([(prefix+'_mesh.mp4',(960,720)),(prefix+'_overlay.mp4',(1280,800))]):
        path=ROOT/'outputs/previews'/name;count=0;decoded_times=[]
        with av.open(str(path)) as c:
            require(c.streams.video[0].average_rate==24,'Unexpected preview FPS')
            for count,frame in enumerate(c.decode(video=0),1):
                require((frame.width,frame.height)==size,'Video resolution mismatch')
                if frame.time is not None:decoded_times.append(frame.time)
                if count-1 in (0,60,120):
                    col=(0,60,120).index(count-1);img=frame.to_image();img.thumbnail((480,350));sheet.paste(img,(480*col,row*388+28));draw.text((480*col+6,row*388+8),f'{name} | frame {count-1:03d}',fill='white')
        require(count==121 and len(decoded_times)==121 and np.allclose(np.diff(decoded_times),1/24,atol=1e-6),'Truncated/irregular preview')
        videos.append({**record(path),'decoded_frames':count,'width':size[0],'height':size[1]})
    review=ROOT/'outputs/previews'/('user_reconstruction_v2_contact_review.jpg' if contact else 'user_reconstruction_v2_review.jpg');sheet.save(review,quality=91)
    mesh_audit=scene_report['independent_full_mesh_audit']
    tip=hand['contact_tip_penetration_beyond_4mm_m'];tip=tip[np.isfinite(tip)]
    tip_max=float(tip.max()) if len(tip) else None
    report.update(status='cpu_contract_audit_passed_conditional_articulated_candidates',source_video_sha256=VIDEO_SHA,frames=121,fps=24,views=2,urdf_sha256=URDF_SHA,camera_sha256=CAMERA_SHA,active_box_sha256=BOX_SHA,source_bindings=hashes,joint_state_entries=79,body_angle_entries=29,coupled_hand_state_entries=50,independent_hand_controls=12,independent_hand_control_units=coupling.actuator_units,all_joint_limits_passed=True,official_coupling_all_values_passed=True,independent_xml_fk_max_abs_error=maximum,independent_body_xml_fk_max_abs_error=body_maximum,non_yaw_body_angles_unchanged=True,fingers_per_hand=5,finger_link_counts=fingers,source_mesh_files=list(meshfiles.values()),body={'candidate_valid_frames':int(body['candidate_valid'].sum()),'image_supported_joint_entries':int(body['joint_image_supported'].sum()),'unobserved_joint_entries':int((~body['joint_image_supported']).sum())},hand={'strict_valid_frames':int(hand['valid'].sum()),'strict_valid_sides':hand['side_valid'].sum(axis=0).tolist(),'observed_joint_entries':int(hand['observed'].sum()),'uncertain_joint_entries':int(hand['uncertain'].sum()),'observed_actuator_entries':int(hand['actuator_observed'].sum()),'uncertain_actuator_entries':int(hand['actuator_uncertain'].sum()),'wrist_yaw_refinements_accepted':int(hand['wrist_refinement_accepted'].sum()),'contact_conflicts':int(hand['contact_image_conflict'].sum()),'tip_penetration_beyond_4mm_max_m':tip_max},box={'image_fit_valid_frames':int(box['image_fit_valid'].sum()),'strict_valid_frames':int(box['valid'].sum()),'strict_failed_frames':np.flatnonzero(~box['valid']).tolist(),'v3_candidate_adopted':False},mesh_audit=mesh_audit,artifacts=[record(p) for p in paths.values()]+glbs,videos=videos,review=record(review),strict_metric_reconstruction_valid=False,physical_hardware_validated=False,full_kinogen_pipeline=False,training=False,gpu_used_for_this_audit=False,limitations=['All finite states are conditional model candidates; observation/uncertainty flags are retained and checked separately.','Box strict geometry failures remain; alternate v3 is diagnostic and is not the active box.','Occluded fingers and axial/wrist ambiguity remain; a five-finger model is not proof of uniquely observed hand motion.','Mesh/tip penetration and image conflicts are reported rather than hidden. No dynamics, torque feasibility, collision-free motion, hardware or original full KinoGen pipeline was validated.','Camera refinement uses real static image evidence with spatial holdout, but does not establish physical camera calibration.'])
    report['body']['temporal_diagnostics']=reports[body_report_name].get('temporal_diagnostics')
    report['body']['unknown_local_uncertainty_entries']=int((~np.isfinite(body['joint_approx_uncertainty_rad'])).sum())
    report['camera_numeric_checks']=camera_numeric_checks
    report['artifacts'].extend(animation_records)
    if animation_audit:report['animated_glb_audit']={'source':'SHA-bound scene exporter report; CPU verifier rechecks output SHA, not an extra optimization',**animation_audit}
    if contact:report['contact_stage']={'body_accepted_sides':body['contact_accepted'].sum(axis=0).tolist(),'body_collision_valid_sides':body['contact_collision_valid'].sum(axis=0).tolist(),'new_hand_candidate_accepted_sides':hand['new_hand_candidate_accepted'].sum(axis=0).tolist(),'body_base_legs_waist_unchanged':True,'rejected_hand_controls_and_yaw_preserved':True,'dense_sample_metrics_nonregressing':True,'declared_metric_numeric_tolerance':[1e-6,1e-7,1e-7],'independent_raw_geometry_still_checked':True}
    report['provenance_files']=[record(p) for p in (video,scene_path,urdf,coupling.math_path,ROOT/'reports'/body_report_name,ROOT/'reports'/hand_report_name,ROOT/'reports'/scene_report_name)]+parent_records
    require(reports[hand_report_name]['physical_hardware_validated'] is False,'Hand report falsely claims physical validation')
    report['urdf_continuous_coupled_joints_without_position_limits']=unbounded
    velocities=np.abs(np.diff(fullq,axis=0)/np.diff(times)[:,None]);velocity_rows=[]
    for k,n in enumerate(fn):
        lim=joints[n].find('limit');raw=float(lim.attrib['velocity']) if lim is not None and 'velocity' in lim.attrib else None
        limit=raw if n in bn and raw is not None and raw>0 else None;actual=velocities[:,k]
        velocity_rows.append({'joint':n,'unit':expected_units[n]+'/s','urdf_authored_value':raw,'urdf_limit':limit,'max_abs_velocity':float(actual.max()),'check_status':'positive_body_velocity_limit_checked' if limit is not None else 'not_specified_zero_or_missing_coupled_hand_velocity','failed_intervals_ending_at_frame':(np.flatnonzero(actual>limit+1e-8)+1).tolist() if limit is not None else []})
    wrist_rows=[x for x in velocity_rows if 'wrist_' in x['joint']]
    require(len(collision_parts)==52,'Expected actual52 ROHand visual meshes for collision audit')
    raw_collision=[];raw_ground=[]
    for f in range(121):
        raw_ground.append(min(float((points@full['link_world_transforms'][f,linkid,2,:3]+full['link_world_transforms'][f,linkid,2,3]).min()) for linkid,points in ground_parts))
        sv,st=[],[];by_side=[[],[]];boxR=Rotation.from_quat(box['quaternion_xyzw'][f]).as_matrix()
        for linkid,points,nv,nt,side in collision_parts:
            T=full['link_world_transforms'][f,linkid];world=points@T[:3,:3].T+T[:3,3]
            local=(world-box['center_xyz_m'][f])@boxR;axes=np.abs(local)-box['box_size_xyz_m']/2
            sdf=np.linalg.norm(np.maximum(axes,0),axis=1)+np.minimum(axes.max(axis=1),0)
            sv.append(sdf[:nv]);st.append(sdf[nv:])
            by_side[side].append(sdf)
        sv,st=np.concatenate(sv),np.concatenate(st)
        raw_collision.append({'frame':f,'vertex_count':len(sv),'trianglecenter_count':len(st),'max_vertex_penetration_m':float(max(0,-sv.min())),'max_trianglecenter_penetration_m':float(max(0,-st.min())),'vertex_fraction_beyond_5mm':float(np.mean(sv<-.005)),'trianglecenter_fraction_beyond_5mm':float(np.mean(st<-.005)),'minimum_signed_distance_m_per_side':[float(np.concatenate(x).min()) for x in by_side],'nearest_sample_to_box_surface_m_per_side':[float(np.abs(np.concatenate(x)).min()) for x in by_side]})
    fractions=np.array([x['vertex_fraction_beyond_5mm'] for x in raw_collision]);triangle_fractions=np.array([x['trianglecenter_fraction_beyond_5mm'] for x in raw_collision]);depths=np.array([max(x['max_vertex_penetration_m'],x['max_trianglecenter_penetration_m']) for x in raw_collision])
    speed_ok=all(not x['failed_intervals_ending_at_frame'] for x in velocity_rows)
    collision_ok=bool(np.max(depths)<=.005 and np.max(fractions)==0 and np.max(triangle_fractions)==0)
    raw_ground=np.array(raw_ground);simplified_ground=np.array([x['robot_mesh_min_world_z_m'] for x in scene_report['per_frame_mesh_audit']]);ground_ok=bool(np.min(raw_ground)>=-.005)
    ground={'authored_floor_z_m':0.,'physical_floor_calibrated':False,'threshold_m':.005,'raw_visual_method':'All original robot visual unique vertices with visual scale/origin and independently verified full FK; plane minimum occurs at a mesh vertex','raw_min_world_z_range_m':[float(raw_ground.min()),float(raw_ground.max())],'raw_max_penetration_m':float(max(0,-raw_ground.min())),'raw_failed_frames_gt5mm':np.flatnonzero(raw_ground<-.005).tolist(),'raw_per_frame_min_world_z_m':raw_ground.tolist(),'renderer_simplified_min_world_z_range_m':[float(simplified_ground.min()),float(simplified_ground.max())],'renderer_simplified_max_penetration_m':float(max(0,-simplified_ground.min())),'renderer_simplified_failed_frames_gt5mm':np.flatnonzero(simplified_ground<-.005).tolist(),'passed':ground_ok,'note':'No feet/floor adaptation or calibrated physical floor recovery was performed. Contact C preserves parent base/legs/waist; this limitation is retained rather than hidden.'}
    report['physical_quality_checks']={'passed':bool(speed_ok and collision_ok),'all_urdf_joint_velocity_limits_passed':speed_ok,'wrist_urdf_velocity_limits_passed':all(not x['failed_intervals_ending_at_frame'] for x in wrist_rows),'wrist_joints':wrist_rows,'all_joint_velocity_audit':velocity_rows,'box_penetration_threshold_m':.005,'max_hand_box_vertex_penetration_m':float(depths.max()),'median_hand_vertices_inside_box_beyond_5mm_fraction':float(np.median(fractions)),'max_hand_vertices_inside_box_beyond_5mm_fraction':float(fractions.max()),'penetration_failed_frames':np.flatnonzero((depths>.005)|(fractions>0)).tolist(),'hand_box_penetration_check_passed':collision_ok,'note':'These are necessary geometric/URDF speed checks on the exported candidate, not dynamics, collision-free hardware feasibility or real-world validation.'}
    report['ground_mesh_audit']=ground
    report['physical_quality_checks']['ground_mesh_check_passed']=ground_ok
    report['physical_quality_checks']['passed']=bool(speed_ok and collision_ok and ground_ok)
    report['physical_quality_checks']['scope']='Necessary authored-URDF speed, raw hand-box sample nonpenetration and authored-floor mesh checks only; contact, self-collision, dynamics and hardware feasibility remain unverified.'
    report['physical_quality_passed']=bool(speed_ok and collision_ok and ground_ok)
    report['contract_audit_passed']=True
    report['physical_quality_checks']['velocity_limits_not_authored_for_joints']=[x['joint'] for x in velocity_rows if x['urdf_limit'] is None]
    report['physical_quality_checks'].update(collision_geometry_method='Independent CPU raw52 URDF hand visuals: scale+visual origin+verified XML-FK transforms; per-visual exact unique vertices and every triangle center, no simplification/subsampling',median_trianglecenter_fraction_beyond_5mm=float(np.median(triangle_fractions)),max_trianglecenter_fraction_beyond_5mm=float(triangle_fractions.max()),raw_hand_collision_per_frame=raw_collision,renderer_collision_method=scene_report.get('hand_collision_geometry','baseline simplified renderer mesh; retained as separate comparison'))
    report['physical_quality_checks']['bilateral_contact_restoration_proven']=False
    report['physical_quality_checks']['contact_note']='Nonpenetration does not establish grasp/contact. Per-side surface sample gaps are retained; these vertex/triangle-center samples are not an exact continuous mesh intersection proof or a physical contact measurement.'
    report['physical_quality_summary']={'audited_necessary_checks_passed':report['physical_quality_passed'],'speed_passed':speed_ok,'hand_box_sample_nonpenetration_passed':collision_ok,'ground_mesh_passed':ground_ok,'max_raw_sample_hand_box_penetration_m':float(depths.max()),'max_raw_vertex_hand_box_penetration_m':max(x['max_vertex_penetration_m'] for x in raw_collision),'max_raw_trianglecenter_hand_box_penetration_m':max(x['max_trianglecenter_penetration_m'] for x in raw_collision),'max_raw_ground_penetration_m':ground['raw_max_penetration_m'],'ground_failed_frames':len(ground['raw_failed_frames_gt5mm']),'velocity_failed_joints':[x['joint'] for x in velocity_rows if x['failed_intervals_ending_at_frame']],'bilateral_contact_restoration_proven':False}
    report['status']='cpu_contract_audit_passed_conditional_quality_checks_passed' if report['physical_quality_passed'] else 'cpu_contract_audit_passed_physical_quality_failed'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true');parser.add_argument('--run-variant',choices=['baseline','contact'],default='baseline');parser.add_argument('--body-file',type=Path);parser.add_argument('--hand-file',type=Path);args=parser.parse_args()
    for key in ('body_file','hand_file'):
        value=getattr(args,key)
        if value is not None:
            value=(ROOT/value).resolve() if not value.is_absolute() else value.resolve()
            require(value.is_relative_to(ROOT.resolve()),'Selected artifact must stay within project root');setattr(args,key,value)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    report={'verified_at_utc':datetime.now(timezone.utc).isoformat(),'status':'pending_final_exports','strict_metric_reconstruction_valid':False,'run_variant':args.run_variant}
    if args.preflight:
        for path,expected_sha in [(ROOT/'data/video_generation/candidate_scene/generated.mp4',VIDEO_SHA),(ROOT/'data/robots/g1_rohand_five_finger/g1_rohand.urdf',URDF_SHA),(OUT/'camera_refinement.npz',CAMERA_SHA),(OUT/'box_trajectory.npz',BOX_SHA)]:
            if path.exists():require(sha(path)==expected_sha,'Frozen delivery pin mismatch: '+str(path))
        contact=args.run_variant=='contact'
        expected=[OUT/n for n in ['camera_refinement.npz','box_trajectory.npz']]
        expected += [args.body_file or OUT/('body_trajectory_contact.npz' if contact else 'body_trajectory.npz'),args.hand_file or OUT/('hand_trajectory_contact.npz' if contact else 'hand_trajectory_wrist_refined.npz'),OUT/('full_trajectory_contact.npz' if contact else 'full_trajectory.npz')]
        expected += [OUT/(f'scene_contact_frame_{f:03d}.glb' if contact else f'scene_frame_{f:03d}.glb') for f in (0,60,120)]
        prefix='user_v2_contact_full' if contact else 'user_v2_full'
        expected += [ROOT/'outputs/previews'/(prefix+s+'.mp4') for s in ('_mesh','_overlay')]
        print(json.dumps({'status':'ready_for_cpu_audit' if all(p.exists() for p in expected) else 'waiting_for_final_exports','missing':[str(p.relative_to(ROOT)) for p in expected if not p.exists()]},indent=2));return
    try:verify(report,args)
    except Exception as exc:
        report.update(status='cpu_contract_audit_failed',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        path=ROOT/'reports'/('user-reconstruction-v2-summary-contact.json' if args.run_variant=='contact' else 'user-reconstruction-v2-summary.json');path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
        compact={k:v for k,v in report.items() if k in ('status','error','run_variant','frames','body','hand','physical_quality_summary','strict_metric_reconstruction_valid')}
        if 'box' in report:compact['box']={k:v for k,v in report['box'].items() if k!='strict_failed_frames'}
        print(json.dumps(compact,ensure_ascii=False,indent=2),flush=True)
    return 0 if report.get('physical_quality_passed') else 3


if __name__=='__main__':raise SystemExit(main())
