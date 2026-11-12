"""Create an explicitly approximate physics-training derivative, never edit CAD sources."""
from pathlib import Path
import argparse,hashlib,json,time,copy,os,xml.etree.ElementTree as ET
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--samples',type=int,default=1025);ap.add_argument('--max-hull-vertices',type=int,default=240);a=ap.parse_args()
 import numpy as np,trimesh
 from scipy.optimize import least_squares
 from scipy.spatial import ConvexHull
 from scipy.spatial.transform import Rotation
 from user_body_kinematics import G1Kinematics,origin
 from user_rohand_coupling import ROHandCoupling,GROUPS
 from hand_geometry import ROHandGeometry,box_signed_distance
 started=time.monotonic();out=ROOT/'data/robots/g1_box_training';out.mkdir(parents=True,exist_ok=True);(out/'collision_meshes').mkdir(exist_ok=True);source=ROOT/'data/robots/g1_rohand_five_finger/g1_rohand.urdf';source_sha=sha(source);assert source_sha=='71bc895263e495a3ea01bbb9626458ac5a15a81387bf6f9a8afaa1e700bff7b6';kin=G1Kinematics(ROOT,scene_path='data/video_generation/candidate_scene/scene_rohand.json');coupling=ROHandCoupling(ROOT);tree=ET.parse(source);xml=tree.getroot();xml.attrib['name']='g1_rohand_box_training_approximation';links={x.attrib['name']:x for x in xml.findall('link')};mass_updates=[];collision_rows=[];collision_local={};source_total=0.;derived_total=0.
 # These are numerical regularizers for the experiment, not corrected CAD.
 for name,link in links.items():
  inert=link.find('inertial');added=inert is None
  if added:
   inert=ET.SubElement(link,'inertial');ET.SubElement(inert,'origin',xyz='0 0 0',rpy='0 0 0');ET.SubElement(inert,'mass',value='0.001');ET.SubElement(inert,'inertia',ixx='1e-7',iyy='1e-7',izz='1e-7',ixy='0',ixz='0',iyz='0')
  mass_el=inert.find('mass');im=inert.find('inertia');m0=float(mass_el.attrib['value']);source_total+=0 if added else m0;keys=('ixx','iyy','izz','ixy','ixz','iyz');d={k:float(im.attrib.get(k,0))for k in keys};I0=np.asarray([[d['ixx'],d['ixy'],d['ixz']],[d['ixy'],d['iyy'],d['iyz']],[d['ixz'],d['iyz'],d['izz']]]);hand='_hand_ro_'in name;m1=max(m0,.002 if hand else .001);I=I0*(m1/m0);w,U=np.linalg.eigh(I);w=np.maximum(w,1e-7 if hand or added else 1e-8)
  if w[2]>w[0]+w[1]:w[:2]+=(w[2]-w[0]-w[1]+1e-10)/2
  I=U@np.diag(w)@U.T;mass_el.set('value',format(m1,'.17g'))
  for key,val in zip(keys,(I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2])):im.set(key,format(float(val),'.17g'))
  derived_total+=m1
  if added or m1!=m0 or np.max(np.abs(I-I0))>1e-15:mass_updates.append({'link':name,'added_fixed_sensor_frame_inertial':added,'original_mass_kg':None if added else m0,'simulation_mass_kg':m1,'original_inertia_kg_m2':None if added else I0.tolist(),'simulation_inertia_kg_m2':I.tolist(),'original_principal_inertias':None if added else np.linalg.eigvalsh(I0).tolist(),'simulation_principal_inertias':np.linalg.eigvalsh(I).tolist()})
 for link in links.values():
  for old in list(link.findall('collision')):link.remove(old)
 for vi,v in enumerate(kin.visuals):
  name=v['link_name'];internal='_hand_ro_'in name and('slider'in name or 'connecting'in name)
  if internal:collision_rows.append({'link':name,'removed_internal_collision':True,'reason':'Actuated slider/connecting/internal abpart remains in79joint dynamics, excluded from external collision.'});continue
  points=np.unique(v['vertices_local'],axis=0);hull=trimesh.convex.convex_hull(points);original_hull_vertices=np.asarray(hull.vertices).copy();limited=False
  if len(hull.vertices)>a.max_hull_vertices:
   p=np.asarray(hull.vertices);chosen=list(dict.fromkeys(np.r_[np.argmin(p,axis=0),np.argmax(p,axis=0)].tolist()));distance=np.min(np.sum((p[:,None]-p[chosen][None])**2,axis=2),axis=1)
   while len(chosen)<a.max_hull_vertices:
    k=int(np.argmax(distance));chosen.append(k);distance=np.minimum(distance,np.sum((p-p[k])**2,axis=1))
   hull=trimesh.convex.convex_hull(p[chosen]);limited=True
  filename=f'collision_meshes/{name}_{vi}.stl';hull.export(out/filename);collision=ET.SubElement(links[name],'collision',name='training_convex_'+str(vi));ET.SubElement(collision,'origin',xyz='0 0 0',rpy='0 0 0');geom=ET.SubElement(collision,'geometry');ET.SubElement(geom,'mesh',filename=filename);vertices=np.asarray(hull.vertices);faces=np.asarray(hull.faces);collision_local.setdefault(name,[]).append(np.concatenate((vertices,vertices[faces].mean(1))));planes=ConvexHull(vertices).equations;under=float(max(0,np.max(original_hull_vertices@planes[:,:3].T+planes[:,3])));collision_rows.append({'link':name,'path':filename,'sha256':sha(out/filename),'vertices':len(vertices),'triangles':len(faces),'limited_vertices':limited,'maximum_original_hull_undercoverage_m':under,'source_mesh_sha256':v['mesh_sha256']})
 # Visual filenames remain relative to the same data/robots/<asset> depth.
 for node in xml.findall('mujoco'):xml.remove(node)
 urdf=out/'g1_box.urdf';tree.write(urdf,encoding='utf-8',xml_declaration=True)
 # Each expanded motion depends on exactly one independent actuator.
 joint_control=[]
 for n in coupling.joint_names:
  side=0 if n.startswith('left_')else 1;bare=n.split('_hand_ro_',1)[1];j=next(i for i,g in enumerate(GROUPS)if bare in g);joint_control.append(side*6+j)
 lower=np.asarray(coupling.lower);upper=np.asarray(coupling.upper);grid=lower[:,None]+(upper-lower)[:,None]*np.linspace(0,1,a.samples)[None];values=np.empty((50,a.samples),np.float32)
 for control in range(12):
  use=np.flatnonzero(np.asarray(joint_control)==control)
  for k in range(a.samples):
   x=lower.copy();x[control]=grid[control,k];d=coupling.expand_one(x)
   for j in use:values[j,k]=d[coupling.joint_names[j]]
 midpoint_max=np.zeros(50)
 for control in range(12):
  use=np.flatnonzero(np.asarray(joint_control)==control)
  for k in np.linspace(0,a.samples-2,min(257,a.samples-1)).astype(int):
   x=lower.copy();x[control]=(grid[control,k]+grid[control,k+1])/2;d=coupling.expand_one(x)
   for j in use:midpoint_max[j]=max(midpoint_max[j],abs((float(values[j,k])+float(values[j,k+1]))/2-d[coupling.joint_names[j]]))
 lut=out/'coupling_lut.npz';np.savez_compressed(lut,actuator_names=np.asarray(coupling.actuator_names),actuator_units=np.asarray(coupling.actuator_units),actuator_lower=lower,actuator_upper=upper,actuator_grid=grid.astype(np.float32),joint_names=np.asarray(coupling.joint_names),joint_units=np.asarray(coupling.joint_units),joint_control_indices=np.asarray(joint_control,np.int64),joint_values=values,official_math_sha256=np.asarray(sha(coupling.math_path)),source_urdf_sha256=np.asarray(source_sha),midpoint_interpolation_max_abs_error=midpoint_max)
 joint_effort={};joint_velocity={};source_limits={};runtime_drives={}
 for j in xml.findall('joint'):
  n=j.attrib['name'];kind=j.attrib['type']
  if kind=='fixed':continue
  lim=j.find('limit');isbody=n in kin.body_index
  if isbody:e=float(lim.attrib['effort']);speed=float(lim.attrib['velocity']);kp=100 if any(s in n for s in ('hip_','knee','ankle_','waist_'))else(20 if 'wrist_'in n else 60);kd=3 if kp>=60 else 1.5
  else:e=15 if kind=='prismatic'else .8;speed=.06 if kind=='prismatic'else(4 if 'th_root'in n else 8);kp=1500 if kind=='prismatic'else 2.;kd=20 if kind=='prismatic'else .10
  joint_effort[n]=e;joint_velocity[n]=speed;source_limits[n]=None if lim is None else dict(lim.attrib);runtime_drives[n]={'effort_limit_sim':e,'velocity_limit_sim':speed,'stiffness':kp,'damping':kd,'source':'G1 original effort/velocity; experiment PD gains'if isbody else 'simulation assumption: expanded independent PD targets, not6motor hardware dynamics'}
 # A floating, flat-foot nominal stance. The base is only written at reset.
 q=kin.initial_state[6:].copy()
 for side in ('left','right'):
  for stem,val in [('hip_pitch',-.10),('hip_roll',0),('hip_yaw',0),('knee',.20),('ankle_pitch',-.10),('ankle_roll',0)]:q[kin.body_index[side+'_'+stem+'_joint']]=val
 for axis in ('yaw','roll','pitch'):q[kin.body_index['waist_'+axis+'_joint']]=0
 base=np.asarray([0.,0.,.79]);quat=np.asarray([0.,0.,0.,1.]);opened=np.zeros(12);closed=upper*.6;closed[[5,11]]=.8;foot_names=['left_ankle_roll_link','right_ankle_roll_link'];T=kin.forward(base,quat,q,coupling.expand_one(opened));foot_min=[min((p@T[kin.link_index[n],:3,:3].T+T[kin.link_index[n],:3,3])[:,2].min()for p in collision_local[n])for n in foot_names];base[2]+=.003-min(foot_min);palm_names=['left_hand_ro_base_link','right_hand_ro_base_link'];offsets=[];Rtargets=[]
 for side in ('left','right'):
  m=ROHandGeometry(ROOT,side);p=m.dense_contact_vertices['base_link'];offsets.append([-.075,0,float(p[:,2].min())]);s=1 if side=='left'else -1;Rtargets.append(np.column_stack(([-1.,0,0],[0,0,s],[0,s,0])))
 offsets=np.asarray(offsets);box_size=np.asarray([.28,.34,.24]);box_center=np.asarray([.30,0,.93]);lift_center=box_center+np.asarray([-.04,0,.12]);boxT=np.eye(4);boxT[:3,3]=box_center;ik_rows=[]
 def solve_pose(seed,center,gap,label,exclude_hand_collision=False):
  qq=seed.copy()
  for si,side in enumerate(('left','right')):
   arm=np.asarray([kin.body_index[side+'_'+stem+'_joint']for stem in ('shoulder_pitch','shoulder_roll','shoulder_yaw','elbow','wrist_roll','wrist_pitch','wrist_yaw')]);handi=kin.link_index[palm_names[si]];goal=center+np.asarray([-.02,(1 if side=='left'else -1)*(box_size[1]/2+gap),0]);rotation=Rtargets[si];part=[]
   for name,geom in collision_local.items():
    li=kin.link_index[name]
    if not kin.ancestor[li,arm].any()or exclude_hand_collision and '_hand_ro_'in name:continue
    for p in geom:part.append((li,p[np.linspace(0,len(p)-1,min(120,len(p))).astype(int)]))
   def evaluate(x):
    qtrial=qq.copy();qtrial[arm]=x;T=kin.forward(base,quat,qtrial,coupling.expand_one(opened));palm=T[handi,:3,:3]@offsets[si]+T[handi,:3,3];rot=Rotation.from_matrix(rotation.T@T[handi,:3,:3]).as_rotvec();BT=np.eye(4);BT[:3,3]=center;points=np.concatenate([p@T[li,:3,:3].T+T[li,:3,3]for li,p in part]);sdf=box_signed_distance(points,BT,box_size);return T,palm,rot,sdf
   def residual(x):
    T,palm,rot,sdf=evaluate(x);pen=np.maximum(.002-sdf,0)/.003;return np.r_[(palm-goal)/.004,rot/.06,pen*np.sqrt(1000/len(pen)),pen.max()*20,(x-seed[arm])*.02]
   seeds=[qq[arm].copy(),np.asarray([-.5,(.4 if side=='left'else -.4),0,1.2,0,0,0]),np.asarray([-.8,(.7 if side=='left'else -.7),(.8 if side=='left'else -.8),1.5,(1 if side=='left'else -1),-.3,0])];trials=[]
   for ii,x in enumerate(seeds):
    fit=least_squares(residual,np.clip(x,kin.body_limits[arm,0]+1e-6,kin.body_limits[arm,1]-1e-6),bounds=(kin.body_limits[arm,0],kin.body_limits[arm,1]),max_nfev=240,ftol=1e-9,xtol=1e-9,gtol=1e-8);T,palm,rot,sdf=evaluate(fit.x);pos=float(np.linalg.norm(palm-goal));ang=float(np.linalg.norm(rot));pen=float(max(0,-sdf.min()));trials.append({'seed':ii,'q':fit.x,'position_error_m':pos,'orientation_error_rad':ang,'collision_penetration_m':pen,'cost':float(fit.cost),'nfev':fit.nfev});
    if pos<.002 and ang<.04 and pen<.0005:break
   best=min(trials,key=lambda r:r['position_error_m']+r['orientation_error_rad']*.05+r['collision_penetration_m']*10);qq[arm]=best['q'];ik_rows.append({'phase':label,'side':side,'target_palm_point_world_m':goal.tolist(),'target_palm_rotation':rotation.tolist(),'best':{k:(v.tolist()if hasattr(v,'tolist')else v)for k,v in best.items()},'trials':[{k:(v.tolist()if hasattr(v,'tolist')else v)for k,v in t.items()}for t in trials]});print(json.dumps({'phase':label,'side':side,'position_error_m':best['position_error_m'],'orientation_error_rad':best['orientation_error_rad'],'collision_penetration_m':best['collision_penetration_m']}),flush=True)
  return qq
 # Write importer-ready assets immediately; JSON is atomic only when IK done.
 print(json.dumps({'urdf_ready':str(urdf.relative_to(ROOT)),'urdf_sha256':sha(urdf),'lut_ready':str(lut.relative_to(ROOT)),'lut_sha256':sha(lut)}),flush=True)
 resetq=solve_pose(q,box_center,.025,'pregrasp_reset');liftq=solve_pose(resetq,lift_center,.002,'lift_reference',exclude_hand_collision=True)
 def quality(qq,center,actuators):
  T=kin.forward(base,quat,qq,coupling.expand_one(actuators));BT=np.eye(4);BT[:3,3]=center;rows=[];minz=[];COM=[];masses=[]
  for name,geom in collision_local.items():
   li=kin.link_index[name];world=np.concatenate([p@T[li,:3,:3].T+T[li,:3,3]for p in geom]);sdf=box_signed_distance(world,BT,box_size);rows.append({'link':name,'minimum_box_sdf_m':float(sdf.min()),'maximum_box_penetration_m':float(max(0,-sdf.min()))});minz.append(float(world[:,2].min()))
  for name,link in links.items():
   inert=link.find('inertial');m=float(inert.find('mass').attrib['value']);p=origin(inert.find('origin'))[:3,3];li=kin.link_index[name];COM.append(p@T[li,:3,:3].T+T[li,:3,3]);masses.append(m)
  feet=[];support=[]
  for n in foot_names:
   li=kin.link_index[n];p=np.concatenate(collision_local[n])@T[li,:3,:3].T+T[li,:3,3];z=float(p[:,2].min());feet.append(z);support.append(p[p[:,2]<z+.01,:2])
  com=np.average(np.asarray(COM),axis=0,weights=masses);hull=ConvexHull(np.concatenate(support));margin=float(-np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2]));held_com=(com*sum(masses)+center)/(sum(masses)+1);held_margin=float(-np.max(hull.equations[:,:2]@held_com[:2]+hull.equations[:,2]));return {'foot_min_z_m':feet,'all_collision_min_z_m':min(minz),'maximum_robot_box_penetration_m':max(x['maximum_box_penetration_m']for x in rows),'penetrating_links':[x for x in rows if x['maximum_box_penetration_m']>1e-6],'robot_com_xyz_m':com.tolist(),'robot_com_support_margin_m':margin,'robot_plus1kg_box_com_xyz_m':held_com.tolist(),'robot_plus1kg_box_support_margin_m':held_margin,'palm_points_world_m':[(T[kin.link_index[n],:3,:3]@offsets[i]+T[kin.link_index[n],:3,3]).tolist()for i,n in enumerate(palm_names)]}
 reset_quality=quality(resetq,box_center,opened);lift_quality=quality(liftq,lift_center,closed);np.savez_compressed(out/'kinematic_references.npz',body_joint_names=np.asarray(kin.body_joint_names),reset_body_q=resetq,lift_body_q=liftq,reset_base_xyz=base,base_quaternion_xyzw=quat,open_actuators=opened,closed_actuators=closed,palm_link_names=np.asarray(palm_names),palm_local_offsets=offsets,box_center_xyz_m=box_center,lift_box_center_xyz_m=lift_center,box_size_xyz_m=box_size)
 config={'schema':'kinogen.g1_box_training_asset.1','purpose':'Floating G1 five-finger box lifting physics training; approximate synthetic50joint hand drives','urdf':'data/robots/g1_box_training/g1_box.urdf','urdf_sha256':sha(urdf),'source_urdf':str(source.relative_to(ROOT)),'source_urdf_sha256':source_sha,'source_audit':{'path':'reports/g1-training-asset-audit.json','sha256':sha(ROOT/'reports/g1-training-asset-audit.json')if(ROOT/'reports/g1-training-asset-audit.json').exists()else None},'body_joint_names':kin.body_joint_names,'hand_joint_names':coupling.joint_names,'joint_names':kin.body_joint_names+coupling.joint_names,'actuator_names':coupling.actuator_names,'actuator_units':coupling.actuator_units,'actuator_lower':lower.tolist(),'actuator_upper':upper.tolist(),'body_joint_lower':kin.body_limits[:,0].tolist(),'body_joint_upper':kin.body_limits[:,1].tolist(),'joint_effort_limits':joint_effort,'joint_velocity_limits':joint_velocity,'original_urdf_limits':source_limits,'runtime_drive_parameters':runtime_drives,'urdf_hand_zero_or_absent_effort_velocity_preserved':True,'runtime_hand_drive_override_required_before_sim_step':True,'coupling_lut':'data/robots/g1_box_training/coupling_lut.npz','coupling_lut_sha256':sha(lut),'coupling_samples':a.samples,'coupling_joint_control_indices':joint_control,'coupling_midpoint_max_error_by_joint':midpoint_max.tolist(),'official_coupling_math_sha256':sha(coupling.math_path),'reset_base_xyz':base.tolist(),'reset_base_quaternion_xyzw':quat.tolist(),'reset_body_q':resetq.tolist(),'lift_body_q':liftq.tolist(),'open_actuators':opened.tolist(),'closed_actuators':closed.tolist(),'box_center_xyz_m':box_center.tolist(),'box_size_xyz_m':box_size.tolist(),'box_mass_kg':1.,'lift_box_center_xyz_m':lift_center.tolist(),'lift_height_m':.12,'table_top_z_m':float(box_center[2]-box_size[2]/2),'palm_link_names':palm_names,'palm_local_offsets':offsets.tolist(),'palm_forward_local':[-1,0,0],'palm_inward_normal_local':[0,0,-1],'foot_link_names':foot_names,'reset_kinematic_quality':reset_quality,'lift_command_kinematic_quality':lift_quality,'ik_diagnostics':ik_rows,'source_total_mass_kg':source_total,'simulation_total_mass_kg':derived_total,'inertia_regularization':{'minimum_rohand_mass_kg':.002,'minimum_rohand_principal_inertia_kg_m2':1e-7,'fixed_sensor_assumed_mass_kg':.001,'fixed_sensor_assumed_inertia_kg_m2':1e-7,'changes':mass_updates},'collision_derivation':collision_rows,'simulation_assumptions':{'self_collisions':False,'floating_base':True,'merge_fixed_joints':False,'box_dynamic':True,'table_static':True,'synthetic_50_independent_PD_drives':True,'hand_12_inputs_only_generate_targets':True,'physical_state_need_not_exactly_follow_target_coupling_during_contact':True,'not_hardware_calibration':True,'no_state_writes_to_lift_or_hold':True,'closed_pose_can_geometrically_intersect_box_because_PD_targets_are_blocked_by_contact':True,'source_CAD_mass_asymmetry_preserved_except_documented_floor':True,'collision_hulls_are_training_approximations':True},'physical_stability_verified':False,'elapsed_seconds':time.monotonic()-started};p=out/'training_asset.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(config,indent=2));tmp.replace(p);report=ROOT/'reports/g1-box-training-asset-preparation.json';report.write_text(json.dumps({'outputs':{str(x.relative_to(ROOT)):sha(x)for x in (urdf,lut,p,out/'kinematic_references.npz')},'source_urdf_unchanged':sha(source)==source_sha,'source_total_mass_kg':source_total,'simulation_total_mass_kg':derived_total,'collision_count':sum(not x.get('removed_internal_collision')for x in collision_rows),'removed_internal_collision_count':sum(bool(x.get('removed_internal_collision'))for x in collision_rows),'mass_regularization_count':len(mass_updates),'body_joint_count':29,'hand_motion_joint_count':50,'independent_control_count':41,'reset_quality':reset_quality,'lift_command_quality':lift_quality,'ik_diagnostics':ik_rows,'physical_stability_verified':False,'elapsed_seconds':time.monotonic()-started},indent=2));print(json.dumps({'complete':True,'config':str(p.relative_to(ROOT)),'config_sha256':sha(p),'reset_quality':reset_quality,'lift_command_quality':lift_quality,'elapsed_seconds':time.monotonic()-started}),flush=True)
if __name__=='__main__':main()
