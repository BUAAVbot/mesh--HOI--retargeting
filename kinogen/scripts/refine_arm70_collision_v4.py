"""One finite F70 arm-shell/hand box repair; immutable canonical V3 parents."""
from pathlib import Path
import os,time,json,argparse
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
from refine_hand_visibility_32_33_v4 import VisibilityFrame,body_metrics,clean,sha

class Collision70(VisibilityFrame):
 def __init__(self):
  import numpy as np,cv2,open3d as o3d,xml.etree.ElementTree as ET
  from scipy.spatial.transform import Rotation
  from user_body_kinematics import G1Kinematics,project_points
  from user_rohand_coupling import ROHandCoupling
  self.np=np;self.cv2=cv2;self.o3d=o3d;self.project=project_points;self.f=70
  v3=ROOT/'outputs/user_reconstruction_v3';v2=ROOT/'outputs/user_reconstruction_v2'
  self.paths={'body':v3/'body_trajectory.npz','hand':v3/'hand_trajectory.npz','camera':v2/'camera_refinement.npz','box':v2/'box_trajectory.npz','landmarks':v2/'body_landmark_observations.npz','ground':v3/'ground_trajectory_retry.npz','input':ROOT/'outputs/user_reconstruction/input.npz','segmentation':ROOT/'outputs/user_reconstruction/segmentation.npz'}
  assert sha(self.paths['body'])=='8119f59170d210e50db53a767113e2938ec801c89a0e94c40500c0f53cccaf3c'
  assert sha(self.paths['hand'])=='6e131767e11e7263e5593007e80a92a8827b8e86114503c20a897467a475c3c1'
  self.d={k:np.load(p)for k,p in self.paths.items()if k not in ('input','segmentation')};B=self.d['body'];H=self.d['hand'];V=np.load(self.paths['input']);S=np.load(self.paths['segmentation']);L=self.d['landmarks'];C=self.d['camera'];self.rgb=V['rgb'][70].copy();self.robot=S['robot'][70].copy();self.boxmask=S['box'][70].copy();self.valid=S['observation_valid'][70].copy();self.P=C['P_video'];self.times=H['timestamps'];self.kin=G1Kinematics(ROOT,scene_path='data/video_generation/candidate_scene/scene_rohand.json');self.coupling=ROHandCoupling(ROOT);self.q=B['joint_positions_rad'].copy();self.act=H['actuator_positions'].copy();self.state=np.r_[B['base_xyz_m'][70],Rotation.from_quat(B['base_quaternion_xyzw'][70]).as_rotvec(),self.q[70]];self.arm=np.asarray([self.kin.body_index['right_'+n+'_joint']for n in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]);self.boxT=self.d['box']['box_world_transform'][70];self.boxsize=self.d['box']['box_size_xyz_m'];self.point_links=np.asarray([self.kin.link_index[n]for n in ('right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link','right_wrist_yaw_link','left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link','left_wrist_yaw_link')]);self.zero=np.zeros((8,3));G=self.d['ground'];gs=np.r_[G['base_xyz_m'][70],Rotation.from_quat(G['base_quaternion_xyzw'][70]).as_rotvec(),G['joint_positions_rad'][70]];self.ground_errors=self.landmarks(gs)
  xml=ET.parse(self.kin.urdf_path).getroot();vs={j.attrib['name']:float(j.find('limit').attrib['velocity'])for j in xml.findall('joint')if j.attrib['name']in self.kin.body_index};self.vel=np.asarray([vs[n]for n in self.kin.body_joint_names]);speed=self.vel[self.arm];self.lower=np.maximum.reduce([self.kin.body_limits[self.arm,0],self.q[69,self.arm]-speed*(self.times[70]-self.times[69]),self.q[71,self.arm]-speed*(self.times[71]-self.times[70])]);self.upper=np.minimum.reduce([self.kin.body_limits[self.arm,1],self.q[69,self.arm]+speed*(self.times[70]-self.times[69]),self.q[71,self.arm]+speed*(self.times[71]-self.times[70])]);self.x0=self.q[70,self.arm].copy()
  # Missing source wrist prevents the frozen RGB-only hand domain. The source
  # palm ROI is never converted into a fabricated wrist measurement.
  self.hand_observation_available=bool(L['valid'][70,0,2]and L['valid'][70,0,3])
  if self.hand_observation_available:raise RuntimeError('This one-shot script freezes controls because the reviewed F70 wrist is missing; evidence changed')
  actual=self.kin.fk(self.state,self.coupling.expand_one(self.act[70]));self.parts=[];self.raw_visuals=self.kin.visuals
  for visual in self.kin.visuals:
   name=visual['link_name'];link=visual['link_index'];is_hand='_hand_ro_'in name
   if not is_hand and not self.kin.ancestor[link,self.arm].any():continue
   vertices=visual['vertices_local'];full=np.concatenate((np.unique(np.round(vertices,10),axis=0),vertices[visual['faces']].mean(1)));attach=link
   if is_hand:
    side='left'if name.startswith('left')else'right';attach=self.kin.link_index[side+'_wrist_yaw_link'];parent=actual[attach];R=parent[:3,:3].T@actual[link,:3,:3];t=parent[:3,:3].T@(actual[link,:3,3]-parent[:3,3]);full=full@R.T+t
   take=np.linspace(0,len(full)-1,min(256,len(full))).astype(int);self.parts.append({'name':name,'link':attach,'full':full,'sample':full[take],'hand':is_hand,'mesh_sha256':visual['mesh_sha256']})
 def decode7(self,x):
  state=self.state.copy();state[6+self.arm]=x;return state,self.kin.fk(state,self.coupling.expand_one(self.act[70]))
 def dense(self,x):
  import numpy as np
  from hand_geometry import box_signed_distance
  state,T=self.decode7(x);result=[]
  for part in self.parts:
   tf=T[part['link']];world=part['full']@tf[:3,:3].T+tf[:3,3];sdf=box_signed_distance(world,self.boxT,self.boxsize);result.append({'link':part['name'],'kind':'hand'if part['hand']else'arm_shell','samples':len(sdf),'min_sdf_m':float(sdf.min()),'max_penetration_m':float(max(0,-sdf.min())),'fraction_deeper_than5mm':float((sdf<-.005).mean())})
  rawmin=min(float((visual['vertices_local']@T[visual['link_index'],2,:3]+T[visual['link_index'],2,3]).min())for visual in self.raw_visuals)
  return {'per_visual':result,'max_penetration_m':max(r['max_penetration_m']for r in result),'deeper_than5mm':[r['link']for r in result if r['max_penetration_m']>.005],'raw_all_robot_min_z_m':rawmin}

def main():
 import numpy as np,cv2,trimesh
 from scipy.optimize import minimize
 from scipy.spatial import cKDTree
 from hand_geometry import box_signed_distance
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--budget-seconds',type=float,default=180);ap.add_argument('--seeds',type=int,default=2);a=ap.parse_args();started=time.monotonic();v=Collision70();deadline=time.monotonic()+min(180,a.budget_seconds);out=ROOT/'outputs/user_reconstruction_v4';out.mkdir(exist_ok=True);baseline_geo=v.dense(v.x0);baseline_body=body_metrics(v,v.state);L=v.d['landmarks'];target=[];hh,ww=240,213;xx,yy=np.meshgrid((np.arange(ww)+.5)*640/ww,(np.arange(hh)+.5)*720/hh);rays=[]
 def domain(view,points):
  keep=np.ones(len(points),bool)
  for k,radius in ((2+4*view,55),(3+4*view,35)):
   if L['valid'][70,view,k]:keep&=np.linalg.norm(points-L['positions_xy'][70,view,k],axis=1)>radius
  return keep
 for view in range(2):
  mask=v.robot[view]&~v.boxmask[view]&v.valid[view];contours=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0];pixels=np.concatenate([c[:,0]for c in contours if len(c)>4]).astype(float)+.5;train=((pixels[:,0]//16).astype(int)+64*(pixels[:,1]//16).astype(int))%5!=0;pixels=pixels[train&domain(view,pixels)&(pixels[:,1]<500)];pixels=pixels[np.linspace(0,len(pixels)-1,min(550,len(pixels))).astype(int)];target.append(pixels);M=v.P[view,:,:3];camera=-np.linalg.solve(M,v.P[view,:,3]);direction=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(M).T;direction/=np.linalg.norm(direction,axis=-1,keepdims=True);rays.append(v.o3d.core.Tensor(np.concatenate((np.broadcast_to(camera,direction.shape),direction),axis=-1).astype(np.float32)))
 table=v.kin.scene['table'];static=[]
 for size,center in [(table['top_size_xyz_m'],table['top_center_xyz_m'])]+[([.035,.035,.735],[x,y,.3675])for x in (.19,.77)for y in (-.355,.355)]:
  mesh=trimesh.creation.box(extents=size);mesh.apply_translation(center);static.append(mesh)
 boxmesh=trimesh.creation.box(extents=v.boxsize);boxmesh.vertices=np.asarray(boxmesh.vertices)@v.boxT[:3,:3].T+v.boxT[:3,3];static.append(boxmesh);static=trimesh.util.concatenate(static);keep=np.asarray(['_hand_ro_'not in v.kin.link_names[i]for i in v.kin.proxy_face_links]);faces=v.kin.proxy_faces[keep];face_links=v.kin.proxy_face_links[keep]
 def correspondences(state):
  scene=v.o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(v.o3d.core.Tensor(v.kin.proxy_world(state).astype(np.float32)),v.o3d.core.Tensor(faces));scene.add_triangles(v.o3d.core.Tensor(np.asarray(static.vertices,np.float32)),v.o3d.core.Tensor(np.asarray(static.faces,np.uint32)));answer=[]
  for view in range(2):
   cast=scene.cast_rays(rays[view],nthreads=4);hit=cast['geometry_ids'].numpy()==0;boundary=hit&~cv2.erode(hit.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);iy,ix=np.nonzero(boundary);pixels=np.column_stack(((ix+.5)*640/ww,(iy+.5)*720/hh));take=domain(view,pixels)&(pixels[:,1]<500);iy,ix,pixels=iy[take],ix[take],pixels[take]
   if len(pixels)>550:take=np.linspace(0,len(pixels)-1,550).astype(int);iy,ix,pixels=iy[take],ix[take],pixels[take]
   tri=cast['primitive_ids'].numpy()[iy,ix].astype(int);uv=cast['primitive_uvs'].numpy()[iy,ix];local=v.kin.proxy_vertices[faces[tri,0]]*(1-uv.sum(1))[:,None]+v.kin.proxy_vertices[faces[tri,1]]*uv[:,0,None]+v.kin.proxy_vertices[faces[tri,2]]*uv[:,1,None];_,match=cKDTree(target[view]).query(pixels);answer.append((view,face_links[tri],local,target[view][match],.65));_,match=cKDTree(pixels).query(target[view]);answer.append((view,face_links[tri][match],local[match],target[view],.65));good=L['valid'][70,view].copy();good[[3,7]]=False;answer.append((view,v.point_links[good],v.zero[good],L['positions_xy'][70,view,good],np.asarray([4,6,10,0,4,6,10,0])[good]))
  return answer
 def assess(x):
  state,T=v.decode7(x);geo=v.dense(x);body=body_metrics(v,state);land=v.landmarks(state);fail=[]
  if geo['max_penetration_m']>1e-6:fail.append('raw_arm_and_all52hand_mesh_penetration')
  if geo['raw_all_robot_min_z_m']<-.005:fail.append('raw_all_robot_floor')
  for view in range(2):
   for key,absolute,extra in [('wrist_error_px',8,3),('median_px',8,2),('p90_px',16,3)]:
    if land[view][key]is not None and land[view][key]>min(absolute,v.ground_errors[view][key]+extra)+1e-6:fail.append(f'view{view}_original_ground_{key}')
   for key,absolute,extra in [('body_heldout_median_px',6,2),('body_heldout_p90_px',15,3)]:
    if body[view][key]>min(absolute,baseline_body[view][key]+extra)+1e-6:fail.append(f'view{view}_{key}')
   if body[view]['body_mask_iou']<max(.55,baseline_body[view]['body_mask_iou']-.04):fail.append(f'view{view}_body_iou')
  return {'geometry':geo,'body':body,'landmarks':land,'rejections':fail}
 seeds=[v.x0.copy(),v.x0.copy()];seeds[1][4]+=.25;seeds[1][5]-=.25;trials=[]
 for si,seed in enumerate(seeds[:min(2,a.seeds)]):
  if time.monotonic()>deadline:break
  x=np.clip(seed,v.lower,v.upper);history=[]
  for outer in range(4):
   if time.monotonic()>deadline:break
   state,T=v.decode7(x);images=correspondences(state);links=[];points=[];image_groups=[];collision_groups=[];offset=0
   def add(ids,local):
    nonlocal offset
    sl=slice(offset,offset+len(local));offset+=len(local);links.append(np.asarray(ids,int));points.append(local);return sl
   for view,ids,local,pixels,weight in images:image_groups.append((view,add(ids,local),pixels,weight))
   for part in v.parts:
    tf=T[part['link']];world=part['full']@tf[:3,:3].T+tf[:3,3];sdf=box_signed_distance(world,v.boxT,v.boxsize);deep=np.argpartition(sdf,min(63,len(sdf)-1))[:min(64,len(sdf))];local=np.concatenate((part['sample'],part['full'][deep]));collision_groups.append(add(np.full(len(local),part['link']),local))
   ids=np.concatenate(links);local=np.concatenate(points)
   def objective(value):
    s=v.state.copy();s[6+v.arm]=value;world,J=v.kin.points(s,ids,local,True);J=J[:,:,6+v.arm];loss=0.;gradient=np.zeros(7)
    for view,sl,pixels,weight in image_groups:
     uv,pj=v.project(v.P[view],world[sl],True);jac=np.einsum('nij,njk->nik',pj,J[sl]);weight=np.asarray(weight);weight=weight[:,None]if weight.ndim else weight;res=(uv-pixels)*weight;root=np.sqrt(1+(res/4)**2);loss+=float((32*(root-1)).sum());gradient+=np.einsum('ni,nik->k',2*res/root*weight,jac)
    for sl in collision_groups:
     world_box=(world[sl]-v.boxT[:3,3])@v.boxT[:3,:3];q=np.abs(world_box)-v.boxsize/2;outside=np.maximum(q,0);length=np.linalg.norm(outside,axis=1);sdf=length+np.minimum(q.max(1),0);g=np.zeros_like(world_box);ext=length>1e-12;g[ext]=outside[ext]/length[ext,None]*np.sign(world_box[ext]);ix=np.flatnonzero(~ext);axis=q.argmax(1);g[ix,axis[ix]]=np.sign(world_box[ix,axis[ix]]);g=g@v.boxT[:3,:3].T;pen=np.maximum(.0008-sdf,0)/.003;loss+=float(3000*np.mean(pen**2)+12000*pen.max()**2);der=-6000*pen/(.003*len(pen));der[int(pen.argmax())]-=24000*pen.max()/.003;gradient+=np.einsum('n,ni,nik->k',der,g,J[sl])
    regular=(value-v.x0)*2.;loss+=float(regular@regular);gradient+=4*regular;return loss,gradient
   def callback(value):
    if time.monotonic()>deadline:raise StopIteration
   fit=minimize(objective,x,jac=True,method='L-BFGS-B',bounds=list(zip(v.lower,v.upper)),callback=callback,options={'maxiter':150,'maxls':30,'ftol':1e-8,'gtol':1e-4});x=fit.x;m=assess(x);history.append({'outer':outer,'nit':int(fit.nit),'cost':float(fit.fun),'rejections':m['rejections']});print(json.dumps(clean({'seed':si,'outer':outer,'max_penetration_m':m['geometry']['max_penetration_m'],'rejections':m['rejections'],'elapsed_seconds':time.monotonic()-started})),flush=True)
   if not m['rejections']:break
  m=assess(x);trials.append({'seed':si,'x':x.copy(),'metrics':m,'history':history})
  if not m['rejections']:break
 passing=[t for t in trials if not t['metrics']['rejections']];chosen=min(passing,key=lambda t:t['metrics']['geometry']['max_penetration_m'])if passing else None;x=chosen['x']if chosen else v.x0;final=assess(x);state,T=v.decode7(x);B={k:v.d['body'][k].copy()for k in v.d['body'].files};H={k:v.d['hand'][k].copy()for k in v.d['hand'].files};q=B['joint_positions_rad'].copy();q[70]=state[6:];B['joint_positions_rad']=q;
 if chosen:B['link_world_transforms'][70]=T
 B['finger_actuator_positions']=v.act.copy();B['finger_joint_positions']=v.coupling.expand_array(v.act);B.update(source_body_sha256=np.asarray(sha(v.paths['body'])),source_hand_sha256=np.asarray(sha(v.paths['hand'])),source_canonical_v3_body_sha256=np.asarray(sha(v.paths['body'])),source_canonical_v3_hand_sha256=np.asarray(sha(v.paths['hand'])),candidate_is_metric_ground_truth=np.asarray(False))
 if chosen:
  changed=np.abs(q[70]-v.q[70])>1e-7
  if 'joint_image_supported'in B:B['joint_image_supported'][70,changed]=False
  if 'joint_approx_uncertainty_rad'in B:B['joint_approx_uncertainty_rad'][70,changed]=np.nan
  for key in ('candidate_valid','ground_image_valid','image_fit_valid'):
   if key in B:B[key][70]=False
 stale=('contact_tip','hand_surface','contact_nonpenetration','wrist_keypoint_error','wrist_origin_world_shift','dense_collision','heldout_edge','legacy_observation','v3_proxy16');H={k:value for k,value in H.items()if not any(token in k for token in stale)};H['body_joint_positions_rad_refined']=q;H['source_hand_sha256']=np.asarray(sha(v.paths['hand']));H['source_canonical_v3_body_sha256']=np.asarray(sha(v.paths['body']));H['source_canonical_v3_hand_sha256']=np.asarray(sha(v.paths['hand']))
 if chosen:
  for key in ('actuator_observed','observed','wrist_observed'):
   if key in H:H[key][70]=False
  for key in ('actuator_uncertain','uncertain','prior_only','wrist_uncertain'):
   if key in H:H[key][70]=True
  for key in ('side_valid','valid','image_fit_valid'):
   if key in H:H[key][70]=False
 if 'wrist_joint_names'in H:
  wq=q[:,[v.kin.body_index[n]for n in H['wrist_joint_names']]];H['wrist_joint_positions_rad']=wq;H['all_wrist_joint_positions_rad']=wq
 if 'wrist_yaw_positions_rad'in H:H['wrist_yaw_positions_rad']=q[:,[v.kin.body_index[s+'_wrist_yaw_joint']for s in ('left','right')]]
 for f in range(121):
  if f!=70:assert np.array_equal(q[f],v.q[f])
 assert np.array_equal(H['actuator_positions'],v.act)
 assert np.max(np.abs(np.diff(q,axis=0))/np.diff(v.times)[:,None]-v.vel)<1e-5
 assert np.all(q>=v.kin.body_limits[:,0]-1e-8)and np.all(q<=v.kin.body_limits[:,1]+1e-8)
 bp=out/'stage70_body.npz';np.savez_compressed(bp,**B);H['source_body_sha256']=np.asarray(sha(bp));H['body_sha256']=np.asarray(sha(bp));hp=out/'stage70_hand.npz';np.savez_compressed(hp,**H);report={'status':'single_finite_arm70_box_repair_completed','selected':chosen is not None,'selected_is_conditional':chosen is not None,'body_hand_image_flags_false_for_missing_right_wrist_and_hand_domain':True,'actual_landmark_valid':L['valid'][70],'hand_observation_available':False,'source_controls_all121_exact':True,'original_q_and_positive_body_velocity_satisfied':True,'other120_actual_body_and_all_base_feet_exact':True,'inputs':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for k,p in v.paths.items()},'outputs':{'body':{'path':str(bp.relative_to(ROOT)),'sha256':sha(bp)},'hand':{'path':str(hp.relative_to(ROOT)),'sha256':sha(hp)}},'baseline_geometry':baseline_geo,'baseline_body':baseline_body,'ground_landmark_budgets':v.ground_errors,'final':final,'trials':trials,'original_mesh_visuals_audited':[{'link':p['name'],'samples':len(p['full']),'mesh_sha256':p['mesh_sha256']}for p in v.parts],'stale_hand_derived_fields_removed_matching_tokens':stale,'pose_dependent_tips_not_inherited':True,'strict_metric_valid':False,'hardware_validated':False,'elapsed_seconds':time.monotonic()-started,'budget_seconds':min(180,a.budget_seconds),'script_sha256':sha(Path(__file__))};(ROOT/'reports/user-v4-arm70-collision.json').write_text(json.dumps(clean(report),indent=2)+'\n');print(json.dumps(clean({k:report[k]for k in ('status','selected','outputs','elapsed_seconds')}),indent=2),flush=True)

if __name__=='__main__':main()
