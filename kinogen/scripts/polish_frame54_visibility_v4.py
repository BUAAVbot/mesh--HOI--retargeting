"""One finite constrained polish of the same third raw-visible F54 seed."""
import os,time,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from user_visibility_v4 import Visibility54,ROOT,sha
from refine_frame54_visibility_v4 import clean
def main():
 import numpy as np,cv2
 from scipy.optimize import minimize
 from scipy.spatial import cKDTree
 from hand_geometry import box_signed_distance
 from optimize_user_hands import sample
 started=time.monotonic();v=Visibility54();out=ROOT/'outputs/user_reconstruction_v4';path=ROOT/'reports/user-v4-frame54-visibility.json';previous=json.loads(path.read_text());trial=[t for t in previous['trials']if t['seed']==2][-1];x0=np.asarray(trial['x']);fixed=v.train_gray;fixed=fixed[np.isfinite(v.box_depth[fixed])];target=v.camera+v.direction[fixed]*(v.box_depth[fixed]-.0015)[:,None];L=v.d['landmarks'];cache={}
 def evaluate(x):
  if 'x'in cache and np.array_equal(cache['x'],x):return cache
  state,act,T,g,tf=v.decode(x);surf=v.model.contact_surface(tf,dense=False);dense=v.full_surface(tf);sdf=box_signed_distance(dense,v.boxT,v.boxsize);world=v.kin.points(state,v.point_links,v.zero);uv=v.project(v.P[0],world);good=L['valid'][54,0].copy();good[[3,7]]=False;err=np.linalg.norm(uv[good]-L['positions_xy'][54,0,good],axis=1);werr=np.linalg.norm(uv[2]-L['positions_xy'][54,0,2]);cache.clear();cache.update(x=x.copy(),state=state,act=act,T=T,g=g,tf=tf,surf=surf,sdf=sdf,err=err,werr=werr);return cache
 def objective(x):
  c=evaluate(x);surf=c['surf'];distance=cKDTree(surf).query(target)[0];boundary=v.boundary(c['g'],16);back=cKDTree(boundary).query(v.obs['training_all_points'])[0];forward=sample(v.obs['distance'],boundary,cval=40);outside=sample(v.obs['outside'],boundary,cval=40);hand,arm=v.ray_depths(c['T']);hd=hand[fixed];occ=np.minimum(v.static_depth[fixed],arm[fixed]);gap=np.zeros(len(hd));hit=np.isfinite(hd);gap[hit]=np.maximum(hd[hit]-occ[hit]+.0005,0);gap[~hit]=distance[~hit]
  return float(np.mean((distance/.009)**2)+np.mean((back/3)**2)+.08*np.mean((forward/3)**2)+.08*np.mean((outside/3)**2)+np.mean((gap/.009)**2)+.003*np.mean((x[:7]-v.x0[:7])**2))
 def constraints(x):
  c=evaluate(x);s=float(c['sdf'].min());g=v.ground_errors[0];return np.asarray([(s-.0002)*1000,(.004-s)*1000,v.wrist_budget-.015-c['werr'],min(8,g['median_px']+2)-.015-np.median(c['err']),min(16,g['p90_px']+3)-.015-np.percentile(c['err'],90)])
 fit=minimize(objective,x0,method='SLSQP',bounds=list(zip(v.lower,v.upper)),constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':80,'ftol':1e-7,'eps':.00015,'disp':True});x=fit.x;m,pred=v.metrics(x);c=evaluate(x);q=v.q.copy();q[54]=c['state'][6:];act=v.act.copy();act[54]=c['act'];B=v.d['body'];H=v.d['hand'];expanded=v.coupling.expand_array(act);body={k:B[k].copy()for k in B.files};body['joint_positions_rad']=q;body['link_world_transforms']=B['link_world_transforms'].copy();body['link_world_transforms'][54]=c['T'];body.update(finger_actuator_positions=act,finger_joint_positions=expanded,source_body_sha256=np.asarray(sha(v.paths['body'])),source_hand_sha256=np.asarray(sha(v.paths['hand'])),source_canonical_v3_body_sha256=np.asarray(sha(v.paths['body'])),source_canonical_v3_hand_sha256=np.asarray(sha(v.paths['hand'])),source_parent_body_path=np.asarray(str(v.paths['body'].relative_to(ROOT))),source_parent_body_sha256=np.asarray(sha(v.paths['body'])))
 for key in ['joint_image_supported']:body[key][54,v.arm]=False
 for key in ['joint_approx_uncertainty_rad']:body[key][54,v.arm]=np.nan
 body['hand_max_penetration_m'][54,1]=m['max_hand_box_penetration_m'];bp=out/'stage_body.npz';np.savez_compressed(bp,**body)
 hand={k:H[k].copy()for k in H.files};hand.update(actuator_positions=act,joint_positions=expanded,body_joint_positions_rad_refined=q,base_xyz_m=body['base_xyz_m'],base_quaternion_xyzw=body['base_quaternion_xyzw'],body_sha256=np.asarray(sha(bp)),source_body_sha256=np.asarray(sha(bp)),source_hand_sha256=np.asarray(sha(v.paths['hand'])),source_canonical_v3_body_sha256=np.asarray(sha(v.paths['body'])),source_canonical_v3_hand_sha256=np.asarray(sha(v.paths['hand'])))
 for key in ['observed','actuator_observed']:hand[key][54,len(hand[key][54])//2:]=False
 for key in ['uncertain','actuator_uncertain']:hand[key][54,len(hand[key][54])//2:]=True
 hand['actuator_std'][54,6:]=np.nan
 for key in ['all_wrist_joint_positions_rad','wrist_joint_positions_rad']:
  names=H['all_wrist_joint_names']if key.startswith('all_')else H['wrist_joint_names'];hand[key]=q[:,[v.kin.body_index[str(n)]for n in names]]
 hand['wrist_yaw_positions_rad']=q[:,[v.kin.body_index[s+'_wrist_yaw_joint']for s in ('left','right')]];hand['wrist_observed'][54,3:]=False;hand['wrist_uncertain'][54,3:]=True
 hand['hand_surface_min_signed_distance_m'][54,1]=m['min_hand_box_sdf_m'];hand['hand_surface_fraction_beyond_5mm'][54,1]=float(np.mean(c['sdf']<-.005));hand['contact_nonpenetration_sample_valid'][54,1]=m['max_hand_box_penetration_m']<=1e-6
 imagepass=m['legacy_weighted_training_rms_px']<4 and m['legacy_proxy12_heldout_px']<4.5;hand['image_fit_valid'][54,1]=imagepass;hand['side_valid'][54,1]=False;hand['valid'][54]=False;hand['contact_geometry_uncertain'][54]=True;hp=out/'stage_hand.npz';np.savez_compressed(hp,**hand)
 unchanged=[i for i in range(121)if i!=54];assert np.array_equal(q[unchanged],v.q[unchanged])and np.array_equal(act[unchanged],v.act[unchanged]);other=[j for j in range(29)if j not in v.arm];assert np.array_equal(q[:,other],v.q[:,other]);assert np.array_equal(act[:,:6],v.act[:,:6]);assert np.max(np.abs(np.diff(q,axis=0))/np.diff(v.times)[:,None]-v.vel)<1e-5
 report={'schema':'kinogen.v4.frame54.constrained_polish.1','status':'finite_search_completed','seed':'same_third_seed_once_80iter_SLSQP','inputs':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for k,p in v.paths.items()},'source_seed_report':{'path':str(path.relative_to(ROOT)),'sha256':sha(path)},'outputs':{'body':{'path':str(bp.relative_to(ROOT)),'sha256':sha(bp)},'hand':{'path':str(hp.relative_to(ROOT)),'sha256':sha(hp)}},'optimizer':{'success':fit.success,'message':fit.message,'iterations':fit.nit,'nfev':fit.nfev},'constraint_values_nonnegative':constraints(x),'x':x,'metrics':m,'image_fit_valid':imagepass,'strict_valid':False,'side_valid':False,'ground_anchored_wrist_budget_px':v.wrist_budget,'source_other120_exact':True,'source_other22_body_joints_exact':True,'no_model_selected_observations':True,'conditional_front_depth_targets_not_measurements':True,'elapsed_seconds':time.monotonic()-started};rp=ROOT/'reports/user-v4-frame54-polish.json';rp.write_text(json.dumps(clean(report),indent=2));canvas=cv2.cvtColor(v.rgb[0],cv2.COLOR_RGB2BGR);edge=pred&~cv2.erode(pred.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);canvas[edge]=(255,255,0);cv2.imwrite(str(out/'stage_visible_054.jpg'),cv2.resize(canvas[270:375,235:355],(600,525)));print(json.dumps(clean(report)),flush=True)
if __name__=='__main__':main()
