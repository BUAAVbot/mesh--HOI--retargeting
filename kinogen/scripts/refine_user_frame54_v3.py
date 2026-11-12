#!/usr/bin/env python3
"""Finite image-driven F54 arm7 + coupled-hand6 search; immutable v2 parents."""
from pathlib import Path
import argparse,hashlib,json,os,time
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
 import math
 if isinstance(v,dict):return {k:clean(x)for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [clean(x)for x in v]
 if hasattr(v,'tolist'):return clean(v.tolist())
 return None if isinstance(v,float)and not math.isfinite(v)else v

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--body-file',type=Path);ap.add_argument('--max-nfev',type=int,default=90);ap.add_argument('--starts',type=int,default=7);ap.add_argument('--suffix',default='joint');ap.add_argument('--dense-train',action='store_true');ap.add_argument('--seed-report',type=Path);ap.add_argument('--seed-indices',default='4,6');ap.add_argument('--preserve-landmarks',action='store_true');ap.add_argument('--require-near-surface',action='store_true');args=ap.parse_args()
 import numpy as np,cv2,open3d as o3d,trimesh,xml.etree.ElementTree as ET
 from scipy.optimize import least_squares
 from scipy.spatial import cKDTree
 from scipy.spatial.transform import Rotation
 from user_body_kinematics import G1Kinematics,project_points
 from user_rohand_coupling import ROHandCoupling
 from hand_geometry import ROHandGeometry,box_signed_distance,polygon_samples
 from optimize_user_hands import observation,sample
 from user_hand_observations_v3 import make_observation
 cv2.setNumThreads(3);started=time.monotonic();v2=ROOT/'outputs/user_reconstruction_v2';out=ROOT/'outputs/user_reconstruction_v3';out.mkdir(exist_ok=True)
 paths={k:v2/v for k,v in {'body':'body_trajectory_contact.npz','hand':'hand_trajectory_contact.npz','camera':'camera_refinement.npz','box':'box_trajectory.npz','landmarks':'body_landmark_observations.npz'}.items()}
 if args.body_file:paths['body']=args.body_file if args.body_file.is_absolute()else ROOT/args.body_file
 d={k:np.load(p)for k,p in paths.items()};H=d['hand'];B=d['box'];L=d['landmarks'];V=np.load(ROOT/'outputs/user_reconstruction/input.npz');S=np.load(ROOT/'outputs/user_reconstruction/segmentation.npz');f=54;P=d['camera']['P_video'];rgb=V['rgb'];valid=S['observation_valid'];robot=S['robot'];boxes=S['box'];coupling=ROHandCoupling(ROOT)
 kin=G1Kinematics(ROOT,scene_path='data/video_generation/candidate_scene/scene_rohand.json');model=ROHandGeometry(ROOT,'right');body_q=(d['body']['joint_positions_rad']if args.body_file else H['body_joint_positions_rad_refined']).copy();base=d['body']['base_xyz_m'].copy();quat=d['body']['base_quaternion_xyzw'].copy()
 if body_q.shape!=(121,29):raise ValueError('Explicit body candidate must carry the complete original121 frame grid')
 state0=np.r_[base[f],Rotation.from_quat(quat[f]).as_rotvec(),body_q[f]];names=kin.body_joint_names;arm=np.asarray([kin.body_index['right_'+n+'_joint']for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]);wristidx=kin.link_index['right_wrist_yaw_link'];mount=np.linalg.inv(np.asarray(kin.scene['robot']['link_world_transforms']['right_wrist_yaw_link']))@np.asarray(kin.scene['robot']['link_world_transforms']['right_hand_ro_base_link'])
 xml=ET.parse(kin.urdf_path).getroot();speed={j.attrib['name']:float(j.find('limit').attrib['velocity'])for j in xml.findall('joint')if j.attrib['name']in names};vel=np.asarray([speed[names[j]]for j in arm]);dt=V['timestamps'];lo=np.maximum.reduce([kin.body_limits[arm,0],body_q[53,arm]-vel*(dt[54]-dt[53]),body_q[55,arm]-vel*(dt[55]-dt[54])]);hi=np.minimum.reduce([kin.body_limits[arm,1],body_q[53,arm]+vel*(dt[54]-dt[53]),body_q[55,arm]+vel*(dt[55]-dt[54])])
 lower=np.r_[lo,np.zeros(6)];upper=np.r_[hi,np.ones(6)];initial_u=(H['actuator_positions'][f,6:]-model.lower)/(model.upper-model.lower);x0=np.r_[body_q[f,arm],initial_u]
 old=observation(rgb[f,0],robot[f,0],boxes[f,0],valid[f,0],L['positions_xy'][f,0,3],1);obs=make_observation(rgb[f,0],robot[f,0],boxes[f,0],valid[f,0],L['positions_xy'][f,0,3],L['positions_xy'][f,0,2]);assert obs is not None
 if args.dense_train:obs['points']=obs['training_all_points']
 fixedfingers=coupling.expand_one(H['actuator_positions'][f]);pointlinks=np.asarray([kin.link_index[n]for n in ('right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link','right_wrist_yaw_link','left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link','left_wrist_yaw_link')]);zero=np.zeros((8,3))
 def decode(x):
  state=state0.copy();state[6+arm]=x[:7];act=model.lower+x[7:]*(model.upper-model.lower);T=kin.fk(state);g,tips,tf=model.world_geometry(act,T[wristidx]@mount);return state,act,g,tf
 def surface(tf,local):return np.concatenate([p@tf['right_hand_ro_'+n][:3,:3].T+tf['right_hand_ro_'+n][:3,3]for n,p in local.items()])
 def hand_boundary(g,view=0):return np.concatenate([polygon_samples(project_points(P[view],p),16)for p in g.values()])
 def hand_error(g,target):return float(cKDTree(hand_boundary(g)).query(target['heldout_points'])[0].mean())
 # Identical image-space body domain and spatial heldout policy to frozen v2.
 h,w=240,213;sx,sy=640/w,720/h;xx,yy=np.meshgrid((np.arange(w)+.5)*sx,(np.arange(h)+.5)*sy);rays=[];targets=[]
 def domain(points,view):
  good=np.ones(len(points),bool)
  for k,radius in ((2+4*view,55.),(3+4*view,35.)):
   if L['valid'][f,view,k]:good &=np.linalg.norm(points-L['positions_xy'][f,view,k],axis=1)>radius
  return good
 for view in range(2):
  M=P[view,:,:3];center=-np.linalg.solve(M,P[view,:,3]);direction=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(M).T;direction/=np.linalg.norm(direction,axis=-1,keepdims=True);rays.append(o3d.core.Tensor(np.concatenate((np.broadcast_to(center,direction.shape),direction),axis=-1).astype(np.float32)))
  mask=robot[f,view]&valid[f,view];contours=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0];points=np.concatenate([c[:,0]for c in contours if len(c)>4]).astype(float)+.5;points=points[domain(points,view)]
  if len(points)>650:points=points[np.linspace(0,len(points)-1,650).astype(int)]
  held=((points[:,0]//16).astype(int)+64*(points[:,1]//16).astype(int))%5==0;dom=domain(np.column_stack((xx.ravel(),yy.ravel())),view).reshape(h,w);targets.append({'train':points[~held],'heldout':points[held],'domain':dom,'mask':(cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0)&dom})
 bodyfaces=np.asarray(['_hand_ro_'not in kin.link_names[i]for i in kin.proxy_face_links]);rf=kin.proxy_faces[bodyfaces];rl=kin.proxy_face_links[bodyfaces];table=kin.scene['table'];parts=[]
 for size,center in [(table['top_size_xyz_m'],table['top_center_xyz_m'])]+[([.035,.035,.735],[x,y,.3675])for x in (.19,.77)for y in (-.355,.355)]:
  m=trimesh.creation.box(extents=size);m.apply_translation(center);parts.append(m)
 tablemesh=trimesh.util.concatenate(parts);tv=o3d.core.Tensor(np.asarray(tablemesh.vertices,dtype=np.float32));tfaces=o3d.core.Tensor(np.asarray(tablemesh.faces,dtype=np.uint32))
 def render(state):
  scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.core.Tensor(kin.proxy_world(state).astype(np.float32)),o3d.core.Tensor(rf));scene.add_triangles(tv,tfaces);answer=[]
  for view in range(2):
   cast=scene.cast_rays(rays[view],nthreads=4);hit=(cast['geometry_ids'].numpy()==0)&~(cv2.resize(boxes[f,view].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0);boundary=hit&~cv2.erode(hit.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);py,px=np.nonzero(boundary);keep=domain(np.column_stack(((px+.5)*sx,(py+.5)*sy)),view);py,px=py[keep],px[keep]
   if len(px)>500:take=np.linspace(0,len(px)-1,500).astype(int);py,px=py[take],px[take]
   tri=cast['primitive_ids'].numpy()[py,px].astype(int);uv=cast['primitive_uvs'].numpy()[py,px];local=kin.proxy_vertices[rf[tri,0]]*(1-uv.sum(1))[:,None]+kin.proxy_vertices[rf[tri,1]]*uv[:,0,None]+kin.proxy_vertices[rf[tri,2]]*uv[:,1,None];answer.append({'mask':hit,'xy':np.column_stack(((px+.5)*sx,(py+.5)*sy)),'local':local,'links':rl[tri]})
  return answer
 def metrics(x):
  state,act,g,tf=decode(x);pred=render(state);landmarks=kin.points(state,pointlinks,zero);body=[]
  for view in range(2):
   target=targets[view];mask=pred[view]['mask']&target['domain'];dist=cKDTree(pred[view]['xy']).query(target['heldout'])[0];good=L['valid'][f,view].copy();good[[3,7]]=False;errors=np.linalg.norm(project_points(P[view],landmarks)[good]-L['positions_xy'][f,view,good],axis=1);wi=2+4*view
   body.append({'body_mask_iou':float((mask&target['mask']).sum()/max((mask|target['mask']).sum(),1)),'heldout_body_contour_median_px':float(np.median(dist)),'heldout_body_contour_p90_px':float(np.percentile(dist,90)),'actual_landmark_median_px':float(np.median(errors))if len(errors)else None,'actual_landmark_p90_px':float(np.percentile(errors,90))if len(errors)else None,'wrist_keypoint_error_px':float(np.linalg.norm(project_points(P[view],landmarks[[wi]])[0]-L['positions_xy'][f,view,wi]))if L['valid'][f,view,wi]else None})
  sd=box_signed_distance(surface(tf,model.dense_contact_vertices),B['box_world_transform'][f],B['box_size_xyz_m']);boundary=hand_boundary(g);forward=sample(obs['distance'],boundary,cval=40);backward=cKDTree(boundary).query(obs['points'])[0];return {'body':body,'hand_new_heldout_px':hand_error(g,obs),'hand_old_heldout_px':hand_error(g,old),'hand_new_training_rms_px':float(np.sqrt(np.mean(np.r_[forward,backward]**2))),'max_penetration_m':float(max(0,-sd.min())),'fraction_penetration_gt5mm':float(np.mean(sd<-.005)),'min_sdf_m':float(sd.min())}
 baseline=metrics(x0)
 def accepts(m):
  reasons=[]
  if m['max_penetration_m']>.004:reasons.append('penetration')
  if args.require_near_surface and m['min_sdf_m']>.004:reasons.append('not_near_box')
  if m['hand_new_training_rms_px']>=4:reasons.append('hand_train')
  if m['hand_new_heldout_px']>=4.5:reasons.append('hand_new_heldout')
  if m['hand_old_heldout_px']>baseline['hand_old_heldout_px']+2:reasons.append('hand_old_domain_degradation')
  for a,b in zip(m['body'],baseline['body']):
   if a['body_mask_iou']<.55 or a['body_mask_iou']<b['body_mask_iou']-.04:reasons.append('body_iou')
   if a['heldout_body_contour_median_px']>6 or a['heldout_body_contour_median_px']>b['heldout_body_contour_median_px']+2:reasons.append('body_heldout_median')
   if a['heldout_body_contour_p90_px']>15 or a['heldout_body_contour_p90_px']>b['heldout_body_contour_p90_px']+3:reasons.append('body_heldout_p90')
   for key,limit,extra in [('actual_landmark_median_px',8,2),('actual_landmark_p90_px',16,3),('wrist_keypoint_error_px',8,3)]:
    if a[key]is not None and(a[key]>limit or b[key]is not None and a[key]>b[key]+extra):reasons.append(key)
  return sorted(set(reasons))
 seeds=[x0]
 for ff in (53,55):seeds.append(np.r_[body_q[ff,arm],(H['actuator_positions'][ff,6:]-model.lower)/(model.upper-model.lower)])
 for pitch,yaw in ((lo[5],lo[6]),(hi[5],hi[6]),((lo[5]+hi[5])/2,lo[6]),((lo[5]+hi[5])/2,hi[6])):
  seed=x0.copy();seed[4]=np.clip(-1.2,lo[4],hi[4]);seed[5]=pitch;seed[6]=yaw;seed[7:]=.05;seed[-1]=.3;seeds.append(seed)
 if args.seed_report:
  p=args.seed_report if args.seed_report.is_absolute()else ROOT/args.seed_report;sr=json.loads(p.read_text());seeds=[np.asarray(sr['trials'][int(i)]['x'])for i in args.seed_indices.split(',')]
 trials=[]
 for si,seed in enumerate(seeds[:args.starts]):
  x=np.clip(seed,lower+1e-8,upper-1e-8);iterations=[]
  for outer in range(3):
   state,act,g,tf=decode(x);pred=render(state);pairs=[]
   for view in range(2):
    target=targets[view]['train'];nn=cKDTree(target).query(pred[view]['xy'])[1];pairs.append((view,pred[view]['links'],pred[view]['local'],target[nn]))
   active={}
   for name,full in model.dense_contact_vertices.items():
    matrix=tf['right_hand_ro_'+name];sd=box_signed_distance(full@matrix[:3,:3].T+matrix[:3,3],B['box_world_transform'][f],B['box_size_xyz_m']);take=np.argpartition(sd,min(23,len(sd)-1))[:min(24,len(sd))];active[name]=np.concatenate((model.contact_vertices[name],full[take]))
   def residual(z):
    state,act,g,tf=decode(z);boundary=hand_boundary(g);distance=sample(obs['distance'],boundary,cval=40);outside=sample(obs['outside'],boundary,cval=40);back=cKDTree(boundary).query(obs['points'])[0];result=[distance/3,outside*.2/3,back*(1. if args.dense_train else .6)/3]
    for view,links,local,target in pairs:result.append(((project_points(P[view],kin.points(state,links,local))-target)*.45/3).ravel())
    landmarks=kin.points(state,pointlinks,zero)
    for view in range(2):
     good=L['valid'][f,view].copy();good[[3,7]]=False;weights=np.asarray([1.,2.,4.,0.,1.,2.,4.,0.]);result.append(((project_points(P[view],landmarks)[good]-L['positions_xy'][f,view,good])*weights[good,None]/3).ravel())
     if args.preserve_landmarks:
      errors=np.linalg.norm(project_points(P[view],landmarks)[good]-L['positions_xy'][f,view,good],axis=1)
      wi=2+4*view;werr=np.linalg.norm(project_points(P[view],landmarks[[wi]])[0]-L['positions_xy'][f,view,wi])
      budgets=[min(8,baseline['body'][view]['wrist_keypoint_error_px']+3)-.1,min(8,baseline['body'][view]['actual_landmark_median_px']+2)-.1,min(16,baseline['body'][view]['actual_landmark_p90_px']+3)-.1]
      result.append(np.maximum(np.asarray([werr,np.median(errors),np.percentile(errors,90)])-budgets,0)*25)
    sd=box_signed_distance(surface(tf,active),B['box_world_transform'][f],B['box_size_xyz_m']);pen=np.maximum(.001-sd,0)/.003;result.extend((pen*np.sqrt(2000/len(pen)),np.asarray([pen.max()*60]),(z[:7]-x0[:7])*.08,(z[7:]-initial_u)*.03))
    if args.require_near_surface:result.append(np.asarray([max(float(sd.min())-.003,0)/.005*25]))
    return np.concatenate(result)
   fit=least_squares(residual,x,bounds=(lower,upper),diff_step=.002,max_nfev=args.max_nfev,loss='soft_l1',f_scale=3,ftol=3e-5,xtol=3e-5,gtol=3e-4);x=fit.x;iterations.append({'outer':outer,'nfev':fit.nfev,'cost':float(fit.cost),'success':bool(fit.success)})
  m=metrics(x);reasons=accepts(m);record={'seed':si,'x':x,'metrics':m,'rejections':reasons,'accepted':not reasons,'iterations':iterations};trials.append(record);print(json.dumps(clean({'seed':si,'metrics':m,'rejections':reasons,'elapsed_seconds':time.monotonic()-started})),flush=True)
  (ROOT/f'reports/user-frame54-refinement-v3_{args.suffix}.json').write_text(json.dumps(clean({'status':'finite_search_running','baseline':baseline,'trials':trials}),indent=2)+'\n')
 feasible=[t for t in trials if t['accepted']];chosen=min(feasible,key=lambda t:t['metrics']['hand_new_heldout_px'])if feasible else min(trials,key=lambda t:t['metrics']['max_penetration_m']+t['metrics']['hand_new_heldout_px']*.001)
 accepted=bool(feasible);q=body_q.copy();actuators=H['actuator_positions'].copy();selection=np.zeros((121,2),bool)
 if accepted:q[f,arm]=chosen['x'][:7];actuators[f,6:]=model.lower+chosen['x'][7:]*(model.upper-model.lower);selection[f,1]=True
 trial_q=body_q[f].copy();trial_q[arm]=chosen['x'][:7];trial_act=H['actuator_positions'][f].copy();trial_act[6:]=model.lower+chosen['x'][7:]*(model.upper-model.lower)
 path=out/f'frame54_{args.suffix}_candidate.npz';np.savez_compressed(path,body_joint_names=np.asarray(names),body_joint_positions_rad_refined=q,base_xyz_m=base,base_quaternion_xyzw=quat,actuator_names=H['actuator_names'],actuator_positions=actuators,actuator_units=H['actuator_units'],joint_names=H['joint_names'],joint_positions=coupling.expand_array(actuators),joint_units=H['joint_units'],timestamps=H['timestamps'],frame_indices=np.arange(121),refinement_accepted=selection,source_body_sha256=np.asarray(sha(paths['body'])),source_hand_sha256=np.asarray(sha(paths['hand'])),source_camera_sha256=np.asarray(sha(paths['camera'])),source_box_sha256=np.asarray(sha(paths['box'])),candidate_is_metric_ground_truth=np.asarray(False),observed=np.zeros((121,50),bool),uncertain=np.ones((121,50),bool),trial_frame_index=np.asarray(54),best_trial_body_joint_positions_rad=trial_q,best_trial_actuator_positions=trial_act)
 report={'status':'finite_search_completed','frame':54,'accepted':accepted,'source_bindings':{k:{'path':str(p),'sha256':sha(p)}for k,p in paths.items()},'baseline':baseline,'trials':trials,'chosen_seed':chosen['seed'],'chosen_candidate_metrics':chosen['metrics'],'selected_candidate_is_baseline_if_rejected':not accepted,'joint_names':np.asarray(names)[arm],'single_frame_joint_bounds_rad':np.column_stack((lo,hi)),'observation_policy':'Image-only gray-metal/source-robot connected hand ROI, fixed6px spatialmod5 split; old/new cross-domain reported; no projection-derived target. Selection holdout is diagnostic, not untouched confirmation.','no_interpolation':True,'only_allowed_changes':'F54 rightarm7 and right6coupled controls; base/otherframes/camera/box/mount unchanged','output':{'path':str(path),'sha256':sha(path)},'elapsed_seconds':time.monotonic()-started};(ROOT/f'reports/user-frame54-refinement-v3_{args.suffix}.json').write_text(json.dumps(clean(report),indent=2)+'\n');print(json.dumps(clean({k:v for k,v in report.items()if k not in ('trials','source_bindings')})),flush=True)
 # Candidate overlays use the actual full mesh, with no source pixel edits.
 for label,x in [('baseline',x0),('trial',chosen['x'])]:
  state,act,g,tf=decode(x);canvas=cv2.cvtColor(rgb[f,0],cv2.COLOR_RGB2BGR)
  for p in g.values():
   for uv in project_points(P[0],p):
    u,v=np.round(uv).astype(int)
    if 0<=u<640 and 0<=v<720:cv2.circle(canvas,(u,v),1,(30,230,60),-1)
  cv2.imwrite(str(out/f'frame54_{args.suffix}_{label}.jpg'),cv2.resize(canvas[240:385,220:365],(580,580)))

if __name__=='__main__':main()
