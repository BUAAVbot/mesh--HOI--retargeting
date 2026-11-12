"""One fixed pass over six right-hand missing-visible frames; no V3/54 edits."""
import os,time,json,argparse
os.environ['CUDA_VISIBLE_DEVICES']=''
from user_visibility_window_v4 import VisibilityWindow,ROOT,sha
from refine_frame54_visibility_v4 import clean
from user_body_contour_window_v4 import body_audit
class BudgetEnd(Exception):pass
def main():
 import numpy as np
 from scipy.optimize import least_squares,minimize
 from scipy.spatial import cKDTree
 from hand_geometry import box_signed_distance
 from optimize_user_hands import sample
 started=time.monotonic();deadline=started+570;out=ROOT/'outputs/user_reconstruction_v4';bp=out/'stage_body.npz';hp=out/'stage_hand.npz';B={k:v.copy()for k,v in np.load(bp).items()};H={k:v.copy()for k,v in np.load(hp).items()};q=B['joint_positions_rad'].copy();act=H['actuator_positions'].copy();Tf=B['link_world_transforms'].copy();records=[];frames=[53,56,57,59,61,62];lastv=None
 def write_outputs():
  body={k:v.copy()for k,v in B.items()};hand={k:v.copy()for k,v in H.items()};body.update(joint_positions_rad=q.copy(),link_world_transforms=Tf.copy(),finger_actuator_positions=act.copy(),finger_joint_positions=lastv.coupling.expand_array(act),source_body_sha256=np.asarray(sha(bp)),source_hand_sha256=np.asarray(sha(hp)),source_visibility_stage54_body_sha256=np.asarray(sha(bp)),source_visibility_stage54_hand_sha256=np.asarray(sha(hp)),source_visibility_stage54_body_path=np.asarray(str(bp.relative_to(ROOT))),source_visibility_stage54_hand_path=np.asarray(str(hp.relative_to(ROOT))));hand.update(actuator_positions=act.copy(),joint_positions=body['finger_joint_positions'].copy(),body_joint_positions_rad_refined=q.copy(),source_hand_sha256=np.asarray(sha(hp)),source_visibility_stage54_body_sha256=np.asarray(sha(bp)),source_visibility_stage54_hand_sha256=np.asarray(sha(hp)),source_visibility_stage54_body_path=np.asarray(str(bp.relative_to(ROOT))),source_visibility_stage54_hand_path=np.asarray(str(hp.relative_to(ROOT))))
  for r in records:
   f=r['frame'];passed=r.get('accepted',False);hand['side_valid'][f,1]=False;hand['valid'][f]=False;hand['observed'][f,25:]=False;hand['uncertain'][f,25:]=True;hand['actuator_observed'][f,6:]=False;hand['actuator_uncertain'][f,6:]=True;hand['actuator_std'][f,6:]=np.nan;hand['image_fit_valid'][f,1]=passed;hand['new_hand_candidate_selected'][f,1]=passed;hand['new_hand_candidate_accepted'][f,1]=False;hand['refinement_selected_conditional'][f,1]=passed;hand['wrist_refinement_accepted'][f,1]=passed;hand['wrist_observed'][f,3:]=False;hand['wrist_uncertain'][f,3:]=True;hand['contact_geometry_uncertain'][f]=True
   body['parent_hand_image_fit_valid'][f,1]=passed;body['parent_hand_side_valid'][f,1]=False
   if passed:
    body['joint_image_supported'][f,lastv.arm]=False;body['joint_approx_uncertainty_rad'][f,lastv.arm]=np.nan;body['hand_max_penetration_m'][f,1]=r['selected_metrics']['max_hand_box_penetration_m'];hand['hand_surface_min_signed_distance_m'][f,1]=r['selected_metrics']['min_hand_box_sdf_m'];hand['hand_surface_fraction_beyond_5mm'][f,1]=0;hand['contact_nonpenetration_sample_valid'][f,1]=True
  for key in ['all_wrist_joint_positions_rad','wrist_joint_positions_rad']:
   names=H['all_wrist_joint_names']if key.startswith('all_')else H['wrist_joint_names'];hand[key]=q[:,[lastv.kin.body_index[str(n)]for n in names]]
  hand['wrist_yaw_positions_rad']=q[:,[lastv.kin.body_index[s+'_wrist_yaw_joint']for s in ('left','right')]];ob=out/'stage_window_body.npz';oh=out/'stage_window_hand.npz';body['source_parent_body_path']=np.asarray(str(bp.relative_to(ROOT)));body['source_parent_body_sha256']=np.asarray(sha(bp));np.savez_compressed(ob,**body);hand['source_body_sha256']=np.asarray(sha(ob));hand['body_sha256']=np.asarray(sha(ob));hand['source_body_path']=np.asarray(str(ob.relative_to(ROOT)));np.savez_compressed(oh,**hand)
  return {'body':{'path':str(ob.relative_to(ROOT)),'sha256':sha(ob)},'hand':{'path':str(oh.relative_to(ROOT)),'sha256':sha(oh)}}
 for f in frames:
  frame_start=time.monotonic();record={'frame':f,'accepted':False,'trials':[]};frame_deadline=min(deadline,frame_start+86)
  if time.monotonic()>deadline-10:record['rejections']=['global_finite_time_budget'];records.append(record);continue
  try:v=VisibilityWindow(f,q);lastv=v
  except AssertionError as e:record['rejections']=['missing_rgb_hand_observation'];records.append(record);continue
  fixed=v.train_gray;fixed=fixed[np.isfinite(v.box_depth[fixed])];target=v.camera+v.direction[fixed]*(v.box_depth[fixed]-.0015)[:,None];seed=np.r_[B['joint_positions_rad'][54,v.arm],(H['actuator_positions'][54,6:]-v.model.lower)/(v.model.upper-v.model.lower)];x=np.clip(seed,v.lower+1e-7,v.upper-1e-7);L=v.d['landmarks'];record['baseline_metrics']=v.metrics(v.x0)[0];record['ground_reference_landmarks']=v.ground_errors;record['ground_anchored_wrist_budget_px']=v.wrist_budget;record['bounds']=np.column_stack((v.lower,v.upper));record['seed_from_actual_frozen54']=seed
  def budget():
   if time.monotonic()>frame_deadline:raise BudgetEnd()
  def sample_surface(tf):return v.model.contact_surface(tf,dense=False)
  def evaluate(z):
   state,ac,T,g,tf=v.decode(z);surf=sample_surface(tf);sd=box_signed_distance(v.full_surface(tf),v.boxT,v.boxsize);lm=v.landmarks(state);return state,ac,T,g,tf,surf,sd,lm
  def train_terms(z,raw=False):
   budget();state,ac,T,g,tf=v.decode(z);surf=sample_surface(tf);dist=cKDTree(surf).query(target)[0];boundary=v.boundary(g,16);back=cKDTree(boundary).query(v.obs['training_all_points'])[0];forward=sample(v.obs['distance'],boundary,cval=40);outside=sample(v.obs['outside'],boundary,cval=40);parts=[dist/.009,back/3,forward*.3/3,outside*.25/3]
   if raw:
    hand,arm=v.ray_depths(T);hd=hand[fixed];occ=np.minimum(v.static_depth[fixed],arm[fixed]);hit=np.isfinite(hd);gap=dist.copy();gap[hit]=np.maximum(hd[hit]-occ[hit]+.0005,0);parts.append(gap/.009)
   return parts,state,tf
  def constraints(z):
   budget();state,ac,T,g,tf,surf,sd,lm=evaluate(z);s=float(sd.min());result=[(s-.00015)*1000,(.004-s)*1000]
   for a,b in zip(lm,v.ground_errors):result.extend((min(8,b['wrist_error_px']+3)-.015-a['wrist_error_px'],min(8,b['median_px']+2)-.015-a['median_px'],min(16,b['p90_px']+3)-.015-a['p90_px']))
   return np.asarray(result)
  def check(z,label):
   m,pred=v.metrics(z);state,ac,T,g,tf=v.decode(z);bm,br=body_audit(v,state);why=list(br)
   if m['max_hand_box_penetration_m']>1e-6:why.append('raw_mesh_penetration')
   if m['min_hand_box_sdf_m']>.004:why.append('not_near_box')
   if m['legacy_weighted_training_rms_px']>=4:why.append('legacy_weighted_4px')
   if m['legacy_proxy12_heldout_px']>=4.5:why.append('legacy_proxy12_heldout_4.5px')
   vis=m['visibility']
   if vis['visible_gray_recall']<.25 or vis['visible_mask_iou']<.2:why.append('actual_visibility_guard_recall.25_iou.2')
   for i,(a,b)in enumerate(zip(m['landmarks'],v.ground_errors)):
    for key,lim,extra in [('wrist_error_px',8,3),('median_px',8,2),('p90_px',16,3)]:
     if a[key]>min(lim,b[key]+extra)+1e-6:why.append(f'view{i}_ground_{key}')
   trial={'phase':label,'x':z.copy(),'metrics':m,'body_contour_metrics':bm,'rejections':why};record['trials'].append(trial);return trial
  for phase,maxeval in [('front_init',22),('raw_rays',12)]:
   old=x.copy()
   def residual(z):
    parts,state,tf=train_terms(z,phase=='raw_rays');sd=box_signed_distance(sample_surface(tf),v.boxT,v.boxsize);pen=np.maximum(.001-sd,0)/.003;parts.extend((pen*np.sqrt(1000/len(pen)),np.asarray([pen.max()*60])));lm=v.landmarks(state)
    for a,b in zip(lm,v.ground_errors):parts.append(np.maximum([a['wrist_error_px']-min(8,b['wrist_error_px']+3)+.05,a['median_px']-min(8,b['median_px']+2)+.05,a['p90_px']-min(16,b['p90_px']+3)+.05],0)*60)
    return np.concatenate(parts)
   try:fit=least_squares(residual,x,bounds=(v.lower,v.upper),diff_step=.002,max_nfev=maxeval,loss='soft_l1',f_scale=3,ftol=1e-4,xtol=1e-4,gtol=1e-3);x=fit.x
   except BudgetEnd:record['budget_exhausted']=True;x=old;break
   check(x,phase)
  feasible=[t for t in record['trials']if not t['rejections']]
  if not feasible and time.monotonic()<frame_deadline-8:
   latest=[x.copy()]
   def objective(z):
    parts,_,_=train_terms(z,True);return float(sum(np.mean(p*p)for p in parts))
   try:
    def callback(z):latest[0]=z.copy()
    fit=minimize(objective,x,method='SLSQP',bounds=list(zip(v.lower,v.upper)),constraints=[{'type':'ineq','fun':constraints}],callback=callback,options={'maxiter':17,'eps':.00015,'ftol':2e-6,'disp':False});x=fit.x;record['polish_optimizer']={'iterations':fit.nit,'nfev':fit.nfev,'success':fit.success,'message':fit.message}
   except BudgetEnd:x=latest[0];record['budget_exhausted']=True
   check(x,'hard_polish')
  feasible=[t for t in record['trials']if not t['rejections']]
  if feasible:
   chosen=max(feasible,key=lambda t:t['metrics']['visibility']['visible_mask_iou']);state,ac,T,_,_=v.decode(chosen['x']);q[f]=state[6:];act[f]=ac;Tf[f]=T;record.update(accepted=True,selected_phase=chosen['phase'],selected_metrics=chosen['metrics'])
  record['elapsed_seconds']=time.monotonic()-frame_start;records.append(record);outputs=write_outputs();report={'schema':'kinogen.v4.missing_visibility.fixed_pass.1','status':'fixed_pass_running','inputs':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for k,p in v.paths.items()},'frames':frames,'records':records,'outputs':outputs,'source_initializer_body_path':str(bp.relative_to(ROOT)),'source_initializer_body_sha256':sha(bp),'source_initializer_hand_path':str(hp.relative_to(ROOT)),'source_initializer_hand_sha256':sha(hp),'strict_valid':False,'visibility_guard':'New diagnostic recall>=.25 and IoU>=.2; original hand4/4.5, body ground budget, zero raw penetration and positive URDF speed unchanged.','elapsed_seconds':time.monotonic()-started};(ROOT/'reports/user-v4-missing-window.json').write_text(json.dumps(clean(report),indent=2));print(json.dumps(clean({'frame':f,'accepted':record['accepted'],'trials':[{k:t[k]for k in ('phase','metrics','rejections')}for t in record['trials']],'elapsed_seconds':time.monotonic()-started})),flush=True)
 unchanged=[i for i in range(121)if i not in frames];assert np.array_equal(q[unchanged],B['joint_positions_rad'][unchanged]);assert np.array_equal(act[unchanged],H['actuator_positions'][unchanged]);assert np.array_equal(Tf[unchanged],B['link_world_transforms'][unchanged]);assert np.max(np.abs(np.diff(q,axis=0))/np.diff(lastv.times)[:,None]-lastv.vel)<1e-5
 outputs=write_outputs();report.update(status='fixed_pass_completed',records=records,outputs=outputs,elapsed_seconds=time.monotonic()-started,accepted_frames=[r['frame']for r in records if r['accepted']],other_frames_exact=True);(ROOT/'reports/user-v4-missing-window.json').write_text(json.dumps(clean(report),indent=2));print(json.dumps(clean({'complete':True,'outputs':outputs,'accepted_frames':report['accepted_frames'],'elapsed_seconds':report['elapsed_seconds']})),flush=True)
if __name__=='__main__':main()
