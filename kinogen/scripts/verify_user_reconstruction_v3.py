#!/usr/bin/env python3
"""Independent CPU candidate comparison; never modifies frozen v2 artifacts."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
V2=ROOT/'outputs/user_reconstruction_v2'
VIDEO_SHA='035c5852f9580cbbf5d612e2ad93a7f80065426a34ce355fb87db43876b20d6b'
CAMERA_SHA='813f64ca29c7199cb033c020e6d9b8ada1bc2c77313946baf53585fe50247b5a'
BOX_SHA='a9c5b1f274997f86382f24ba09a67abe84ae4231370888f05f254a2d49c415fa'
URDF_SHA='71bc895263e495a3ea01bbb9626458ac5a15a81387bf6f9a8afaa1e700bff7b6'

def require(test,message):
    if not test:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def record(path):
    path=Path(path).resolve()
    return {'path':path.relative_to(ROOT.resolve()).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size}

def load(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}

def check_provenance(path):
    d=json.loads(path.read_text());checked=[];seen={}
    def one(p,h):
        p=Path(p);p=(ROOT/p).resolve()if not p.is_absolute()else p.resolve()
        require(p.is_relative_to(ROOT.resolve()),'Provenance target outside project')
        if p in seen:require(seen[p]==h,'Inconsistent duplicate provenance binding');return
        require(sha(p)==h,'Stale provenance binding '+str(p));checked.append(record(p))
        seen[p]=h
    for field in ('inputs','outputs','files','parent_roles','source_bindings','sources'):
        block=d.get(field,{})
        for key,value in (block.items()if isinstance(block,dict)else enumerate(block)):
            if isinstance(value,str)and len(value)==64:one(key,value)
            elif isinstance(value,dict)and 'path'in value and 'sha256'in value:one(value['path'],value['sha256'])
    for field in ('initial_hand_parent','source_hand_comparison_parent','source_trial_report'):
        value=d.get(field)
        if isinstance(value,dict)and 'path'in value and 'sha256'in value:one(value['path'],value['sha256'])
    value=d.get('output')
    if isinstance(value,dict)and 'path'in value and 'sha256'in value:one(value['path'],value['sha256'])
    def nested(value):
        if isinstance(value,dict):
            if 'path'in value and 'sha256'in value:one(value['path'],value['sha256'])
            for v in value.values():nested(v)
        elif isinstance(value,list):
            for v in value:nested(v)
    nested(d)
    require(checked,'Provided provenance report has no supported SHA bindings')
    return {'report':record(path),'checked_bindings':checked}

class XMLModel:
    def __init__(self,urdf):
        import numpy as np
        import trimesh
        self.np=np
        from scipy.spatial.transform import Rotation
        self.Rotation=Rotation
        self.xml=ET.parse(urdf).getroot();self.links=[l.attrib['name'] for l in self.xml.findall('link')]
        self.joints={j.attrib['name']:j for j in self.xml.findall('joint')};todo=list(self.joints.values());known={'pelvis'};self.ordered=[]
        while todo:
            done=[j for j in todo if j.find('parent').attrib['link'] in known]
            require(done,'Invalid XML kinematic tree')
            for j in done:self.ordered.append(j);known.add(j.find('child').attrib['link']);todo.remove(j)
        self.parts=[];cache={};self.mesh_files=[]
        for link in self.xml.findall('link'):
            name=link.attrib['name']
            for visual in link.findall('visual'):
                mesh=visual.find('geometry/mesh')
                if mesh is None:continue
                filename=mesh.attrib['filename']
                if filename.startswith('package://'):
                    package,rel=filename[10:].split('/',1)
                    p={'g1_description':ROOT/'dependencies/unitree-ros/robots/g1_description','rohand_urdf':ROOT/'dependencies/rohand-urdf'}[package]/rel
                else:p=urdf.parent/filename
                if p not in cache:
                    cache[p]=trimesh.load_mesh(p,process=False);self.mesh_files.append(record(p))
                m=cache[p];v=np.asarray(m.vertices)*np.fromstring(mesh.attrib.get('scale','1 1 1'),sep=' ')
                unique=np.unique(v,axis=0);centers=v[np.asarray(m.faces)].mean(axis=1)
                T=self.origin(visual);points=np.vstack((unique,centers))@T[:3,:3].T+T[:3,3]
                self.parts.append((name,points,len(unique),'_hand_ro_' in name))
        require(sum(x[3] for x in self.parts)==52,'Expected52 original hand visuals')

    def origin(self,element):
        np=self.np;o=element.find('origin');T=np.eye(4)
        if o is not None:
            T[:3,3]=np.fromstring(o.attrib.get('xyz','0 0 0'),sep=' ')
            T[:3,:3]=self.Rotation.from_euler('xyz',np.fromstring(o.attrib.get('rpy','0 0 0'),sep=' ')).as_matrix()
        return T

    def fk(self,xyz,quat,q):
        np=self.np;out={'pelvis':np.eye(4)};out['pelvis'][:3,3]=xyz;out['pelvis'][:3,:3]=self.Rotation.from_quat(quat).as_matrix()
        for j in self.ordered:
            motion=np.eye(4);kind=j.attrib['type'];name=j.attrib['name']
            if kind!='fixed':
                axis=np.fromstring(j.find('axis').attrib.get('xyz','1 0 0'),sep=' ');axis/=np.linalg.norm(axis)
                if kind=='prismatic':motion[:3,3]=axis*q[name]
                else:motion[:3,:3]=self.Rotation.from_rotvec(axis*q[name]).as_matrix()
            out[j.find('child').attrib['link']]=out[j.find('parent').attrib['link']]@self.origin(j)@motion
        return out

    def audit(self,body,hand,box,full=None):
        np=self.np
        from user_rohand_coupling import ROHandCoupling
        c=ROHandCoupling(ROOT);bn=list(map(str,body['joint_names']));hn=list(map(str,hand['joint_names']))
        controls=hand['actuator_positions'][:,[list(map(str,hand['actuator_names'])).index(n) for n in c.actuator_names]]
        unit_by_name=dict(zip(c.actuator_names,c.actuator_units));require(list(map(str,hand['actuator_units']))==[unit_by_name[str(n)]for n in hand['actuator_names']],'Actuator mixed units differ')
        joint_unit_by_name=dict(zip(c.joint_names,c.joint_units));require(list(map(str,hand['joint_units']))==[joint_unit_by_name[n]for n in hn],'Coupled joint mixed units differ')
        require(np.isfinite(controls).all() and np.all(controls>=np.array(c.lower)-1e-12) and np.all(controls<=np.array(c.upper)+1e-12),'Original actuator position bound')
        require(np.allclose(hand['joint_positions'][:,[hn.index(n)for n in c.joint_names]],c.expand_array(controls),atol=1e-10,rtol=0),'Official coupling differs')
        require(str(hand['urdf_sha256'])==URDF_SHA and str(hand['coupling_math_sha256'])==sha(c.math_path),'Model/coupling source changed')
        q=body['joint_positions_rad'];hbody=hand.get('body_joint_positions_rad_refined',q)
        hbn=list(map(str,hand.get('body_joint_names',body['joint_names'])))
        actual=hbody[:,[hbn.index(n)for n in bn]]
        require(actual.shape==(121,29) and np.isfinite(actual).all(),'Incomplete body state')
        require(np.isfinite(hand['joint_positions']).all(),'Incomplete coupled state')
        require(np.array_equal(hand['uncertain'],~hand['observed']) and np.array_equal(hand['actuator_uncertain'],~hand['actuator_observed']),'Uncertainty lost')
        require(np.array_equal(hand['contact_geometry_uncertain'],~box['valid']),'Box uncertainty lost')
        qfull=np.column_stack((actual,hand['joint_positions']));names=bn+hn;times=body['timestamps']
        require(len(bn)==29 and len(hn)==50 and len(set(names))==79,'Mixed-state cardinality mismatch')
        require(np.isfinite(body['base_xyz_m']).all()and np.isfinite(body['base_quaternion_xyzw']).all(),'Nonfinite base')
        require(np.allclose(np.linalg.norm(body['base_quaternion_xyzw'],axis=1),1,atol=1e-8),'Base quaternions not normalized')
        require(np.array_equal(times,hand['timestamps']) and np.allclose(times,np.arange(121)/24,atol=1e-10),'Timeline differs')
        if full is not None:
            fn=list(map(str,full['joint_names']));units={**dict.fromkeys(bn,'rad'),**joint_unit_by_name}
            require(len(fn)==79 and len(set(fn))==79 and set(fn)==set(names),'Export is not same79 mixed states')
            require(list(map(str,full['joint_units']))==[units[n]for n in fn],'Full mixed units differ')
        bounds=[];speeds=[]
        for k,n in enumerate(names):
            j=self.joints[n];limit=j.find('limit')
            if limit is not None:
                require(np.all(qfull[:,k]>=float(limit.attrib['lower'])-1e-8) and np.all(qfull[:,k]<=float(limit.attrib['upper'])+1e-8),'Position limit '+n)
            raw=float(limit.attrib['velocity']) if limit is not None and 'velocity' in limit.attrib else None
            vmax=float(np.abs(np.diff(qfull[:,k])/np.diff(times)).max())
            authored=raw if n in bn and raw is not None and raw>0 else None
            speeds.append({'joint':n,'actual_max_per_s':vmax,'positive_original_body_limit':authored,'passed':None if authored is None else vmax<=authored+1e-8})
        rows=[];fk_error=0.;body_fk_error=0.
        for f in range(121):
            tf=self.fk(body['base_xyz_m'][f],body['base_quaternion_xyzw'][f],dict(zip(names,qfull[f])))
            if full is not None:
                require(np.array_equal(full['base_xyz_m'],body['base_xyz_m']),'Export base differs')
                for k,n in enumerate(map(str,full['joint_names'])):require(full['joint_positions'][f,k]==qfull[f,names.index(n)],'Export state differs')
                fk_error=max(fk_error,float(np.abs(np.stack([tf[str(n)]for n in full['link_names']])-full['link_world_transforms'][f]).max()))
            low=1e9;sole=[1e9,1e9];sv=[];st=[];side_samples=[[],[]];rot=self.Rotation.from_quat(box['quaternion_xyzw'][f]).as_matrix()
            for link,points,nv,handpart in self.parts:
                T=tf[link];world=points@T[:3,:3].T+T[:3,3];low=min(low,float(world[:nv,2].min()))
                if 'ankle_roll_link' in link:sole[0 if link.startswith('left_')else 1]=min(sole[0 if link.startswith('left_')else 1],float(world[:nv,2].min()))
                if handpart:
                    local=(world-box['center_xyz_m'][f])@rot;axes=np.abs(local)-box['box_size_xyz_m']/2
                    sdf=np.linalg.norm(np.maximum(axes,0),axis=1)+np.minimum(axes.max(axis=1),0)
                    sv.append(sdf[:nv]);st.append(sdf[nv:]);side_samples[0 if link.startswith('left_')else 1].append(sdf)
            sv=np.concatenate(sv);st=np.concatenate(st)
            rows.append({'frame':f,'robot_visual_min_z_m':low,'sole_min_z_m':sole,'hand_vertex_depth_m':float(max(0,-sv.min())),'hand_trianglecenter_depth_m':float(max(0,-st.min())),'hand_vertex_fraction_gt5mm':float(np.mean(sv<-.005)),'hand_trianglecenter_fraction_gt5mm':float(np.mean(st<-.005)),'side_minimum_signed_distance_m':[float(np.concatenate(x).min())for x in side_samples],'side_nearest_sample_surface_gap_m':[float(np.abs(np.concatenate(x)).min())for x in side_samples],'side_all_sample_fraction_gt5mm':[float(np.mean(np.concatenate(x)<-.005))for x in side_samples],'side_mean_depth_beyond5mm_m':[float(np.mean(np.maximum(-np.concatenate(x)-.005,0)))for x in side_samples]})
        require(fk_error<1e-7,'Full FK mismatch')
        floor=np.array([r['robot_visual_min_z_m']for r in rows]);depth=np.array([max(r['hand_vertex_depth_m'],r['hand_trianglecenter_depth_m'])for r in rows])
        return {'frames':121,'official_coupling_all_values_passed':True,'original_position_limits_passed':True,'full_xml_fk_max_abs_error':fk_error if full is not None else None,'velocity':speeds,'all_positive_original_body_velocity_limits_passed':all(x['passed'] is not False for x in speeds),'ground_max_penetration_m':float(max(0,-floor.min())),'ground_failed_frames_gt5mm':np.flatnonzero(floor<-.005).tolist(),'hand_max_sample_penetration_m':float(depth.max()),'hand_failed_frames_gt5mm':np.flatnonzero(depth>.005).tolist(),'hand_vertex_fraction_gt5mm_median':float(np.median([r['hand_vertex_fraction_gt5mm']for r in rows])),'hand_trianglecenter_fraction_gt5mm_median':float(np.median([r['hand_trianglecenter_fraction_gt5mm']for r in rows])),'body_supported_joint_entries':int(body['joint_image_supported'].sum()),'body_unknown_uncertainty_entries':int((~np.isfinite(body['joint_approx_uncertainty_rad'])).sum()),'hand_observed_joint_entries':int(hand['observed'].sum()),'hand_uncertain_joint_entries':int(hand['uncertain'].sum()),'hand_image_fit_valid_sides':hand['image_fit_valid'].sum(axis=0).tolist(),'hand_strict_valid_sides':hand['side_valid'].sum(axis=0).tolist(),'per_frame':rows}

def source_evidence(model,scene):
    import numpy as np
    source=ROOT/'data/video_generation/candidate_scene/scene.json';s=json.loads(source.read_text())
    q=scene['robot']['joint_positions'];tf=model.fk(scene['robot']['base_position_xyz_m'],scene['robot']['base_quaternion_xyzw'],q)
    sole={}
    for side in ('left','right'):
        sole[side]=min(float((p[:nv]@tf[n][:3,:3].T+tf[n][:3,3])[:,2].min())for n,p,nv,h in model.parts if n==side+'_ankle_roll_link')
    stereo=json.loads((ROOT/'reports/user-stereo-refinement.json').read_text());ind=json.loads((ROOT/'reports/user-stereo-independent-audit.json').read_text())
    return {'authored_source':record(source),'original_floor':s['floor'],'original_box':s['box'],'original_table':s['table'],'five_finger_q0_sole_min_z_m':sole,'floor_evidence':'scene.json explicitly authors floor.surface_z_m=0 and visual-foot clearance=.002m. render_candidate_scene.py independently grounds q0 from minimum visual z. This is source geometry, not physical ground measured in AI video.','metric_validity':False,'calibration':{'P_fixed_sha256':CAMERA_SHA,'static_refinement':stereo['views'],'holdout_scope':'Incremental refinement only: original H was SIFT-estimated on the same frame; not an independently acquired calibration. Right camera unchanged after insufficient16/8 spatial count.','independent_foot_track_audit':ind,'foot_warning':'37 foot/ankle material tracks are not sole-contact points; approximately40mm median world z must not be used to translate the floor.'},'model_mismatch':'Original reference/video used integrated three-finger G1; current five-finger ROHand is a subsequent explicit model assembly. Source body chain unchanged; finger image fit cannot be treated as original-model ground truth.','remaining_blind_spots':['AI generation can preserve static background yet alter dynamic object/body multi-view shape. Static reprojection is not metric truth.','Unknown floor/box physical accuracy does not justify shifting P, box size, box poses or floor to remove an optimization failure.','A raw vertex/triangle-center sample audit is not exact continuous triangle intersection or physical contact measurement.','Positive body velocity checks exclude zero/absent ROHand limits; coupled actuator dynamics remain unspecified.','Ground-adapting legs/base can change wrist positions; final combined arms/hands must be rechecked rather than merely concatenated.','Same box/camera candidates remain conditionally calibrated; strict box valid flags and uncertain joints must survive rendering.']}

def declared_body_fk_audit(model,body,scene):
    import numpy as np
    from user_rohand_coupling import ROHandCoupling
    c=ROHandCoupling(ROOT);bn=list(map(str,body['joint_names']));ln=list(map(str,body['link_names']))
    require(len(ln)==89 and set(ln)==set(model.links),'Body model skeleton changed')
    included=bool(body['finger_joint_positions_included']);maximum=0.
    if included:
        hn=list(map(str,body['finger_joint_names']));actnames=list(map(str,body['finger_actuator_names']))
        controls=body['finger_actuator_positions'][:,[actnames.index(n)for n in c.actuator_names]]
        expanded=c.expand_array(controls)
        require(np.allclose(body['finger_joint_positions'][:,[hn.index(n)for n in c.joint_names]],expanded,atol=1e-10,rtol=0),'Declared body fixed fingers do not obey official coupling')
    for f in range(121):
        q=dict(zip(bn,body['joint_positions_rad'][f]))
        q.update(dict(zip(hn,body['finger_joint_positions'][f]))if included else {n:scene['robot']['joint_positions'][n]for n in c.joint_names})
        tf=model.fk(body['base_xyz_m'][f],body['base_quaternion_xyzw'][f],q)
        maximum=max(maximum,float(np.abs(np.stack([tf[n]for n in ln])-body['link_world_transforms'][f]).max()))
    require(maximum<1e-7,'Saved body FK differs from its explicitly declared fixed fingers/body states')
    return {'max_abs_error':maximum,'finger_positions_included':included,'policy':'Body saved transforms use that file declared finger states, which can be fixed parent-hand priors. Final full transforms separately use the actual selected hand; these are not silently equated.'}

def foot_track_comparison(model,scene,baseline_body,baseline_hand,body,hand):
    """Freeze original q0 link-local attachments; never reattach to candidate."""
    import numpy as np
    gp=ROOT/'outputs/user_reconstruction/geometry.npz';tp=ROOT/'outputs/user_reconstruction/tracking.npz'
    g=load(gp);tr=load(tp);P=load(V2/'camera_refinement.npz')['P_refined']
    names=[str(g['link_names'][int(i)])for i in g['shared_link_index']]
    require(len(names)==37,'Expected37 fixed shared material IDs')
    q0=model.fk(scene['robot']['base_position_xyz_m'],scene['robot']['base_quaternion_xyzw'],scene['robot']['joint_positions'])
    local=np.stack([(point-q0[name][:3,3])@q0[name][:3,:3]for name,point in zip(names,g['shared_robot_points_world'])])
    observed=tr['shared_positions_xy'];valid=tr['shared_observation_valid'].astype(bool)
    require(observed.shape==(121,2,37,2) and valid.shape==(121,2,37),'Shared foot observations incomplete')
    held=np.arange(37)%5==0;shoe=np.array(['ankle_roll_link'in n for n in names])
    def evaluate(b,h):
        bn=list(map(str,b['joint_names']));hn=list(map(str,h['joint_names']));hbn=list(map(str,h.get('body_joint_names',b['joint_names'])))
        q=h.get('body_joint_positions_rad_refined',b['joint_positions_rad'])[:,[hbn.index(n)for n in bn]]
        errors=np.full((121,2,37),np.nan);uvs=np.empty((121,2,37,2))
        for f in range(121):
            tf=model.fk(b['base_xyz_m'][f],b['base_quaternion_xyzw'][f],dict(zip(bn+hn,np.r_[q[f],h['joint_positions'][f]])))
            world=np.stack([local[k]@tf[n][:3,:3].T+tf[n][:3,3]for k,n in enumerate(names)])
            homogeneous=np.column_stack((world,np.ones(37)))
            for v in range(2):
                xyz=homogeneous@P[v].T;uvs[f,v]=xyz[:,:2]/xyz[:,2:]
                good=valid[f,v];errors[f,v,good]=np.linalg.norm(uvs[f,v,good]-observed[f,v,good],axis=1)
        return errors,uvs
    a,au=evaluate(baseline_body,baseline_hand);b,bu=evaluate(body,hand)
    def stats(x):
        x=x[np.isfinite(x)]
        return {'count':len(x),'median_px':float(np.median(x))if len(x)else None,'p90_px':float(np.quantile(x,.9))if len(x)else None,'max_px':float(x.max())if len(x)else None}
    per=[]
    for f in range(121):
        per.append({'frame':f,'views':[{'view':v,'baseline_all':stats(a[f,v]),'candidate_all':stats(b[f,v]),'baseline_shoe_only':stats(a[f,v,shoe]),'candidate_shoe_only':stats(b[f,v,shoe]),'baseline_fixed_heldout_ids':stats(a[f,v,held]),'candidate_fixed_heldout_ids':stats(b[f,v,held]),'max_same_point_projection_shift_px':float(np.linalg.norm(bu[f,v]-au[f,v],axis=1).max())}for v in range(2)]})
    return {'bindings':[record(gp),record(tp)],'point_link_names':names,'fixed_local_xyz_m':local.tolist(),'fixed_heldout_ids':np.flatnonzero(held).tolist(),'baseline_all':stats(a),'candidate_all':stats(b),'baseline_fixed_heldout_ids':stats(a[:,:,held]),'candidate_fixed_heldout_ids':stats(b[:,:,held]),'per_frame':per,'policy':'Exact original q0 link-local attachments and unchanged observed visibility for both models. No reassociation, no residual rejection based on candidate, no worldZ=0 targets. Every fifth ID is the original body-fit heldout; reuse during model selection is diagnostic validation, not fresh confirmatory evidence.'}

def frame54_image_comparison(model,baseline_body,baseline_hand,body,hand,training_domain='sample72',frame=54,side='right'):
    """Independent XML poses, fixed source RGB observation domains, no refit."""
    import numpy as np
    from scipy.spatial import cKDTree
    from hand_geometry import ROHandGeometry,polygon_samples
    from user_hand_observations_v3 import make_observation
    from optimize_user_hands import observation
    from scipy.ndimage import map_coordinates
    vp=ROOT/'outputs/user_reconstruction/input.npz';sp=ROOT/'outputs/user_reconstruction/segmentation.npz';lp=V2/'body_landmark_observations.npz'
    view=0 if side=='right'else 1;pi=3+4*view;wi=2+4*view
    v=load(vp);s=load(sp);l=load(lp);P=load(V2/'camera_refinement.npz')['P_refined'][view];f=frame
    old=observation(v['rgb'][f,view],s['robot'][f,view],s['box'][f,view],s['observation_valid'][f,view],l['positions_xy'][f,view,pi],1 if side=='right'else -1)
    new=make_observation(v['rgb'][f,view],s['robot'][f,view],s['box'][f,view],s['observation_valid'][f,view],l['positions_xy'][f,view,pi],l['positions_xy'][f,view,wi])
    if old is None or new is None:
        return {'frame':f,'view':view,'side':side,'observation_available':False,'old_domain_available':old is not None,'new_domain_available':new is not None,'new_domain_heldout_count':len(new['heldout_points'])if new is not None else 0,'old_domain_heldout_count':len(old['heldout_points'])if old is not None else 0,'baseline':None,'candidate':{'hand_image_necessary_gates_passed':False,'new_domain_heldout_mean_px':None,'legacy12_new_domain_heldout_mean_px':None,'new_domain_original4_5px_gate_passed':None,'legacy12_original4_5px_gate_passed':None},'source_bindings':[record(p)for p in (vp,sp,lp,ROOT/'scripts/user_hand_observations_v3.py')],'validation_scope':'The unchanged source image-domain policy yielded insufficient observations. Missing hand evidence is unknown, not a passed image gate; no model projection supplies replacement pixels.'}
    proxy=ROHandGeometry(ROOT,side)
    def evaluate(b,h):
        bn=list(map(str,b['joint_names']));hn=list(map(str,h['joint_names']));hbn=list(map(str,h.get('body_joint_names',b['joint_names'])))
        q=h.get('body_joint_positions_rad_refined',b['joint_positions_rad'])[f,[hbn.index(n)for n in bn]]
        tf=model.fk(b['base_xyz_m'][f],b['base_quaternion_xyzw'][f],dict(zip(bn+hn,np.r_[q,h['joint_positions'][f]])))
        projected=[]
        for link,points in proxy.vertices.items():
            T=tf[side+'_hand_ro_'+link];world=points@T[:3,:3].T+T[:3,3];xy=np.column_stack((world,np.ones(len(world))))@P.T
            projected.append(xy[:,:2]/xy[:,2:])
        boundaries={n:np.concatenate([polygon_samples(x,n)for x in projected])for n in (12,16)}
        tree=cKDTree(boundaries[16]);a=float(tree.query(old['heldout_points'])[0].mean());b=float(tree.query(new['heldout_points'])[0].mean())
        legacy_tree=cKDTree(boundaries[12]);legacy_old=float(legacy_tree.query(old['heldout_points'])[0].mean());legacy_new=float(legacy_tree.query(new['heldout_points'])[0].mean())
        def sample(image,points,cval=25):return map_coordinates(image,[points[:,1],points[:,0]],order=1,mode='constant',cval=cval)
        matrix=[]
        for n,points in boundaries.items():
            for domain,target in [('sample72',new['points']),('dense',new['training_all_points'])]:
                visible=sample(new['valid'].astype(float),points,cval=0)>.5
                occluded=(sample(new['box'].astype(float),points,cval=0)>.5)&(sample(new['robot'].astype(float),points,cval=0)<.5)
                weights=visible&~occluded
                forward=np.clip(sample(new['distance'],points),0,18)
                outside=np.clip(sample(new['outside'],points),0,18)
                backward=cKDTree(points).query(target)[0]
                weighted=np.r_[forward*weights/3,outside*weights*.4/3,np.clip(backward,0,18)*.3/3]
                score=float(np.sqrt(np.mean(weighted**2))*3)
                raw=float(np.sqrt(np.mean(np.r_[sample(new['distance'],points,cval=40),backward]**2)))
                matrix.append({'proxy_polygon_samples_per_link':n,'training_domain':domain,'forward_sample_count':len(points),'weighted_visible_sample_count':int(weights.sum()),'backward_training_pixel_count':len(target),'weighted_residual_scalar_count':len(weighted),'original_weighted_rms_px':score,'original4px_gate_passed':score<4,'raw_forward_backward_concat_rms_px':raw,'raw_is_original4px_gate':False})
        primary=next(x for x in matrix if x['proxy_polygon_samples_per_link']==12 and x['training_domain']==training_domain)
        return {'old_domain_heldout_mean_px':a,'new_domain_heldout_mean_px':b,'new_domain_original4_5px_gate_passed':b<4.5,'legacy12_old_domain_heldout_mean_px':legacy_old,'legacy12_new_domain_heldout_mean_px':legacy_new,'legacy12_original4_5px_gate_passed':legacy_new<4.5,'weighted_metric_matrix':matrix,'primary_original_weighted4px_metric':primary,'primary_training_recipe':training_domain,'hand_image_necessary_gates_passed':b<4.5 and legacy_new<4.5 and primary['original4px_gate_passed']}
    return {'frame':f,'view':view,'side':side,'observation_available':True,'baseline':evaluate(baseline_body,baseline_hand),'candidate':evaluate(body,hand),'new_domain_heldout_count':len(new['heldout_points']),'old_domain_heldout_count':len(old['heldout_points']),'metric_definition':'Original4px criterion is sqrt(mean(image_residual**2))*3 with12 polygon samples/link, clipped18px forward/outside/backward, outside weight.4, backward weight.3 and original visibility/box-ownership. Raw unweighted concat is a different diagnostic. Both12/16 and72/dense recipes are listed, not merged. Both legacy12 and search16 heldout proxies use the same fixed observed pixels and unchanged4.5px gate.','source_bindings':[record(p)for p in (vp,sp,lp,ROOT/'scripts/user_hand_observations_v3.py',ROOT/'scripts/hand_geometry.py',ROOT/'scripts/optimize_user_hands.py')],'pose_method':'Independent normalized-axis assembled-URDF XML FK; fixed per-link source visual hull samples. No observation or pose refitting.','validation_scope':'Old/new masks are image-only and common to both poses. Spatial holdout was used for candidate selection, hence diagnostic validation; not untouched confirmatory evidence or all-finger observability.'}

def main():
    import numpy as np
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--body-file',type=Path);ap.add_argument('--hand-file',type=Path);ap.add_argument('--full-file',type=Path);ap.add_argument('--scene-report',type=Path);ap.add_argument('--provenance-report',type=Path,action='append',default=[]);ap.add_argument('--hand-training-domain',choices=['sample72','dense'],default='dense');ap.add_argument('--output',type=Path,default=Path('reports/user-v3-independent-audit.json'));args=ap.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ('body_file','hand_file','full_file','scene_report','output'):
        p=getattr(args,key)
        if p is not None:
            p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve();require(p.is_relative_to(ROOT.resolve()),'Artifact outside project');setattr(args,key,p)
    require('user_reconstruction_v2' not in args.output.as_posix() and not args.output.name.startswith('user-reconstruction-v2-'),'Must preserve v2 report')
    report_paths=[]
    for p in args.provenance_report:
        p=(ROOT/p).resolve()if not p.is_absolute()else p.resolve();require(p.is_relative_to(ROOT.resolve()),'Provenance outside project');report_paths.append(p)
    require(sha(V2/'camera_refinement.npz')==CAMERA_SHA and sha(V2/'box_trajectory.npz')==BOX_SHA,'Frozen geometry changed')
    require(sha(ROOT/'data/video_generation/candidate_scene/generated.mp4')==VIDEO_SHA,'Source video changed')
    scene=json.loads((ROOT/'data/video_generation/candidate_scene/scene_rohand.json').read_text());urdf=ROOT/scene['robot']['urdf'];require(sha(urdf)==URDF_SHA,'Robot model changed');model=XMLModel(urdf)
    report={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'verification_script':record(Path(__file__)),'source_evidence':source_evidence(model,scene),'strict_metric_reconstruction_valid':False,'physical_hardware_validated':False,'gpu_used':False,'raw_geometry_method':'Independent normalized-axis XML FK, original scaled visual meshes with visual origins, every unique vertex and every hand triangle center; no rendered simplification','source_meshes':model.mesh_files}
    report['provenance_audits']=[check_provenance(p)for p in report_paths]
    if args.body_file or args.hand_file:
        require(args.body_file is not None and args.hand_file is not None,'Both body and hand required')
        body=load(args.body_file);hand=load(args.hand_file);box=load(V2/'box_trajectory.npz')
        require(str(hand['source_body_sha256'])==sha(args.body_file),'Hand not bound to selected candidate body')
        for d in (body,hand):require(str(d['camera_sha256'])==CAMERA_SHA and str(d['source_video_sha256'])==VIDEO_SHA,'Candidate source binding differs')
        for key in ('base_xyz_m','base_quaternion_xyzw'):
            if key in hand:require(np.array_equal(hand[key],body[key]),'Hand explicit base differs from body')
        require(str(hand['box_sha256'])==BOX_SHA,'Hand contact used a different box')
        if 'observation_module_sha256'in hand:require(str(hand['observation_module_sha256'])==sha(ROOT/'scripts/user_hand_observations_v3.py'),'Hand observation implementation changed')
        import numpy as np
        require(np.array_equal(box['box_size_xyz_m'],np.array([.28,.34,.24])),'Known box dimensions changed')
        report['box_strict_valid_frames']=int(box['valid'].sum())
        report['box_strict_invalid_frames']=np.flatnonzero(~box['valid']).tolist()
        report['box_strict_invalid_frame_count']=len(report['box_strict_invalid_frames'])
        report['camera_sha256']=CAMERA_SHA;report['active_box_sha256']=BOX_SHA;report['source_video_sha256']=VIDEO_SHA;report['urdf_sha256']=URDF_SHA
        report['joint_state_entries']=79;report['independent_body_plus_hand_controls']=41;report['floating_base_parameters']=6
        require(np.array_equal(body['camera_P_video'],load(V2/'camera_refinement.npz')['P_refined']),'Body projection differs from frozen camera')
        bn=list(map(str,body['joint_names']));hbn=list(map(str,hand.get('body_joint_names',body['joint_names'])))
        refined=hand.get('body_joint_positions_rad_refined',body['joint_positions_rad'])[:,[hbn.index(n)for n in bn]]
        nonwrist=[i for i,n in enumerate(bn)if '_wrist_'not in n]
        require(len(nonwrist)==23 and np.array_equal(refined[:,nonwrist],body['joint_positions_rad'][:,nonwrist]),'Hand stage changed non-wrist body angles')
        report['hand_body_refinement_contract']={'only_six_authored_wrist_axes_mutable':True,'other23_body_angles_exact':True,'joint_names':[n for n in bn if '_wrist_'in n]}
        full=load(args.full_file)if args.full_file else None
        report['inputs']=[record(p)for p in (args.body_file,args.hand_file,V2/'camera_refinement.npz',V2/'box_trajectory.npz')]
        baseline_body=load(V2/'body_trajectory_contact.npz');baseline_hand=load(V2/'hand_trajectory_contact.npz')
        report['baseline_v2_contact']=model.audit(baseline_body,baseline_hand,box)
        report['candidate']=model.audit(body,hand,box,full)
        report['candidate']['body_candidate_valid_frames']=int(body['candidate_valid'].sum())
        report['candidate']['body_unobserved_joint_entries']=int((~body['joint_image_supported']).sum())
        report['candidate']['hand_observed_actuator_entries']=int(hand['actuator_observed'].sum())
        report['candidate']['hand_uncertain_actuator_entries']=int(hand['actuator_uncertain'].sum())
        report['candidate']['hand_strict_valid_frames']=int(hand['valid'].sum())
        report['independent_declared_body_saved_fk_audit']=declared_body_fk_audit(model,body,scene)
        report['audited_necessary_physical_checks_passed']=bool(report['candidate']['all_positive_original_body_velocity_limits_passed'] and not report['candidate']['ground_failed_frames_gt5mm'] and not report['candidate']['hand_failed_frames_gt5mm'])
        report['necessary_check_scope']='Authored-floor original-mesh samples, authored positive body velocity limits and hand-box surface samples only. Not dynamics, full continuous self-collision, actuator dynamics, actual grasp or physical metric calibration.'
        report['fixed_material_foot_track_comparison']=foot_track_comparison(model,scene,baseline_body,baseline_hand,body,hand)
        foot_failures=[]
        for row in report['fixed_material_foot_track_comparison']['per_frame']:
            for view in row['views']:
                old=view['baseline_fixed_heldout_ids'];new=view['candidate_fixed_heldout_ids']
                if not new['count']:foot_failures.append({'frame':row['frame'],'view':view['view'],'reason':'no original heldout material observations'});continue
                if new['median_px']>3 or new['p90_px']>6 or new['median_px']>old['median_px']+1.5 or new['p90_px']>old['p90_px']+2.5:
                    foot_failures.append({'frame':row['frame'],'view':view['view'],'baseline':old,'candidate':new})
        report['independent_fixed_foot_projection_checks']={'passed':not foot_failures,'failures':foot_failures,'thresholds':{'median_max_px':3,'p90_max_px':6,'median_degradation_max_px':1.5,'p90_degradation_max_px':2.5},'threshold_source':'Unchanged optimize_user_ground.py original candidate acceptance; not loosened for this audit. Shoe-only all-ID diagnostics separately reported so knee points cannot silently substitute for shoe evidence.'}
        if (ROOT/'scripts/user_hand_observations_v3.py').exists():
            report['frame54_fixed_rgb_hand_image_comparison']=frame54_image_comparison(model,baseline_body,baseline_hand,body,hand,args.hand_training_domain)
        import numpy as np
        def actual_body(b,h):
            n=list(map(str,b['joint_names']));hn=list(map(str,h.get('body_joint_names',b['joint_names'])))
            return h.get('body_joint_positions_rad_refined',b['joint_positions_rad'])[:,[hn.index(x)for x in n]]
        oldq=actual_body(baseline_body,baseline_hand);newq=actual_body(body,hand)
        oldnames=list(map(str,baseline_body['joint_names']));newnames=list(map(str,body['joint_names']));oldq=oldq[:,[oldnames.index(n)for n in newnames]]
        changed=oldq!=newq;control_changed=baseline_hand['actuator_positions']!=hand['actuator_positions']
        report['changes_from_frozen_v2_contact']={'body_frames':np.flatnonzero(changed.any(axis=1)).tolist(),'body_joints':{n:np.flatnonzero(changed[:,k]).tolist()for k,n in enumerate(newnames)if changed[:,k].any()},'actuator_frames':np.flatnonzero(control_changed.any(axis=1)).tolist(),'base_translation_frames':np.flatnonzero((body['base_xyz_m']!=baseline_body['base_xyz_m']).any(axis=1)).tolist(),'base_rotation_frames':np.flatnonzero((body['base_quaternion_xyzw']!=baseline_body['base_quaternion_xyzw']).any(axis=1)).tolist(),'changed_body_entries_without_image_support':int((changed&~body['joint_image_supported']).sum()),'max_body_angle_delta_rad':float(np.abs(oldq-newq).max())}
        if full is not None:
            import numpy as np
            require(not bool(full['strict_metric_reconstruction_valid']),'Export falsely claims strict reconstruction')
            for k,v in [('body_observed',body['joint_image_supported']),('body_valid',body['candidate_valid']),('hand_joint_observed',hand['observed']),('hand_actuator_observed',hand['actuator_observed']),('hand_valid',hand['valid']),('box_valid',box['valid'])]:require(np.array_equal(full[k],v),'Export promoted '+k)
            require(np.array_equal(full['box_center_xyz_m'],box['center_xyz_m'])and np.array_equal(full['box_quaternion_xyzw'],box['quaternion_xyzw']),'Export box changed')
            require(np.array_equal(full['timestamps'],body['timestamps']),'Export timeline differs')
            report['inputs'].append(record(args.full_file))
        if args.scene_report:
            sr=json.loads(args.scene_report.read_text());require(sr['strict_metric_reconstruction_valid']is False,'Scene promoted physical validity')
            expected={'body_trajectory':sha(args.body_file),'hand_trajectory':sha(args.hand_file),'camera_refinement':CAMERA_SHA,'box_trajectory':BOX_SHA}
            for key,h in expected.items():
                binding=sr['inputs'][key];require(binding['sha256']==h and sha(ROOT/binding['path'])==h,'Renderer used different '+key)
            for binding in sr['outputs']:require(sha(ROOT/binding['path'])==binding['sha256'],'Scene output hash differs')
            for p,h in sr.get('implementation_files',{}).items():require(sha(ROOT/p)==h,'Renderer implementation changed after export')
            for binding in sr.get('visible_hand_audit_inputs',{}).values():require(sha(ROOT/binding['path'])==binding['sha256'],'Visible hand reference changed')
            import av
            import trimesh
            decoded=[];glbs=[]
            for binding in sr['outputs']:
                p=ROOT/binding['path']
                if p.suffix=='.mp4':
                    times=[];count=0
                    with av.open(str(p))as container:
                        require(container.streams.video[0].average_rate==24,'Unexpected video rate')
                        for count,frame in enumerate(container.decode(video=0),1):
                            require(frame.time is not None,'Video timestamp missing');times.append(frame.time)
                    require(count==121 and np.allclose(times,np.arange(121)/24,atol=1e-6),'Video incomplete or irregular')
                    decoded.append({**record(p),'decoded_frames':count})
                if p.suffix=='.glb':
                    mesh=trimesh.load(p,force='scene',process=False);require(bool(mesh.geometry),'Empty GLB')
                    require(all(np.isfinite(x.vertices).all()for x in mesh.geometry.values()),'GLB nonfinite mesh')
                    glbs.append({**record(p),'geometry_count':len(mesh.geometry)})
            require(len(decoded)==2,'Final scene must have both complete videos')
            aa=sr.get('animated_glb_audit');require(aa is not None and aa['frames']==121 and aa['serialized_transform_max_abs_error']<2e-6 and aa['independent_trimesh_load_passed'],'Animation source audit incomplete')
            report['independent_cpu_decoded_videos']=decoded;report['independent_cpu_loaded_glbs']=glbs
            report['scene_exporter_diagnostics']={'source':record(args.scene_report),'independent_full_mesh_audit':sr.get('independent_full_mesh_audit'),'visible_hand_comparison_policy':sr.get('visible_hand_comparison_policy'),'visible_hand_comparison_summary':sr.get('visible_hand_comparison_summary'),'animation':sr.get('animated_glb_audit'),'scope':'SHA-bound full-scene z-buffer external-hand image/animation diagnostics from exporter, distinct from this script independent raw CPU geometry and fixed-material projection audit. A visible contour fit is not proof of all-finger3D observation.'}
        f54=report.get('frame54_fixed_rgb_hand_image_comparison')
        report['all_audited_image_gates_passed']=report['independent_fixed_foot_projection_checks']['passed']and (f54 is None or f54['candidate']['hand_image_necessary_gates_passed'])
        report['candidate_acceptance_checks_passed']=report['audited_necessary_physical_checks_passed']and report['all_audited_image_gates_passed']
        report['status']='independent_candidate_comparison_necessary_checks_passed'if report['candidate_acceptance_checks_passed']else'independent_candidate_comparison_quality_checks_failed'
    else:report['status']='source_geometry_evidence_audited_waiting_for_candidates'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    compact={k:v for k,v in report.get('candidate',{}).items()if k not in ('per_frame','velocity','ground_failed_frames_gt5mm','hand_failed_frames_gt5mm')}
    if compact:
        compact['ground_failed_frame_count']=len(report['candidate']['ground_failed_frames_gt5mm']);compact['hand_failed_frames_gt5mm']=report['candidate']['hand_failed_frames_gt5mm']
    print(json.dumps({'output':record(args.output),'status':report['status'],'candidate':compact},ensure_ascii=False,indent=2))
    return 3 if report.get('candidate_acceptance_checks_passed')is False else 0

if __name__=='__main__':raise SystemExit(main())
