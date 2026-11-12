"""Finite F54 raw-visible ray-ordering repair, immutable canonical V3 parents."""
from pathlib import Path
import os,time,json,argparse
os.environ['CUDA_VISIBLE_DEVICES']=''
from user_visibility_v4 import Visibility54,ROOT,sha
def clean(v):
 import math
 if isinstance(v,dict):return {k:clean(x)for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [clean(x)for x in v]
 if hasattr(v,'tolist'):return clean(v.tolist())
 return None if isinstance(v,float)and not math.isfinite(v)else v
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seeds',type=int,default=3);ap.add_argument('--init-nfev',type=int,default=70);ap.add_argument('--ray-nfev',type=int,default=45);a=ap.parse_args()
 import numpy as np,cv2
 from scipy.optimize import least_squares
 from scipy.spatial import cKDTree
 from hand_geometry import box_signed_distance
 from optimize_user_hands import sample
 started=time.monotonic();v=Visibility54();out=ROOT/'outputs/user_reconstruction_v4';out.mkdir(exist_ok=True);baseline,_=v.metrics(v.x0);L=v.d['landmarks'];fixed_indices=v.train_gray;hasbox=np.isfinite(v.box_depth[fixed_indices]);fixed_indices=fixed_indices[hasbox];target=v.camera+v.direction[fixed_indices]*(v.box_depth[fixed_indices]-.002)[:,None]
 # These front-surface points are conditional initialization/ordering targets,
 # never called measured 3D. Pixel selection is immutable source RGB only.
 target_tree=cKDTree(target);source_ground=v.ground_errors;trials=[]
 def surface(tf,local):return np.concatenate([p@tf['right_hand_ro_'+n][:3,:3].T+tf['right_hand_ro_'+n][:3,3]for n,p in local.items()])
 def criteria(m):
  why=[];vis=m['visibility'];land=m['landmarks'][0]
  if m['max_hand_box_penetration_m']>1e-6:why.append('raw_mesh_penetration')
  if m['min_hand_box_sdf_m']>.004:why.append('hand_not_near_box')
  if land['wrist_error_px']>v.wrist_budget+1e-6:why.append('ground_anchored_wrist_budget')
  if land['median_px']>min(8,source_ground[0]['median_px']+2)+1e-6:why.append('ground_landmark_median_budget')
  if land['p90_px']>min(16,source_ground[0]['p90_px']+3)+1e-6:why.append('ground_landmark_p90_budget')
  if vis['visible_gray_pixels']<100:why.append('missing_actual_visible_gray_hand')
  if m['legacy_weighted_training_rms_px']>=4:why.append('legacy_weighted_training_4px')
  if m['legacy_proxy12_heldout_px']>=4.5:why.append('legacy_proxy12_heldout_4.5px')
  return why
 def save(x,index,m,phase):
  state,act,T,g,tf=v.decode(x);q=v.q.copy();q[54]=state[6:];acts=v.act.copy();acts[54]=act;B=v.d['body'];H=v.d['hand'];body={k:B[k].copy()for k in B.files};body['joint_positions_rad']=q;body['link_world_transforms']=B['link_world_transforms'].copy();body['link_world_transforms'][54]=T;body.update(source_body_sha256=np.asarray(sha(v.paths['body'])),source_hand_sha256=np.asarray(sha(v.paths['hand'])),source_canonical_v3_body_sha256=np.asarray(sha(v.paths['body'])),source_canonical_v3_hand_sha256=np.asarray(sha(v.paths['hand'])),candidate_is_metric_ground_truth=np.asarray(False))
  bp=out/f'candidate_{index}_{phase}_body.npz';np.savez_compressed(bp,**body);hand={k:H[k].copy()for k in H.files};hand.update(actuator_positions=acts,joint_positions=v.coupling.expand_array(acts),body_joint_positions_rad_refined=q,body_joint_names=np.asarray(v.kin.body_joint_names),source_body_sha256=np.asarray(sha(bp)),source_hand_sha256=np.asarray(sha(v.paths['hand'])),source_canonical_v3_body_sha256=np.asarray(sha(v.paths['body'])),source_canonical_v3_hand_sha256=np.asarray(sha(v.paths['hand'])),observed=np.zeros((121,50),bool),uncertain=np.ones((121,50),bool),candidate_is_metric_ground_truth=np.asarray(False));hp=out/f'candidate_{index}_{phase}_hand.npz';np.savez_compressed(hp,**hand)
  for k in range(121):
   if k!=54:assert np.array_equal(q[k],v.q[k])and np.array_equal(acts[k],v.act[k])
  assert np.max(np.abs(np.diff(q,axis=0))/np.diff(v.times)[:,None]-v.vel)<1e-5
  return {'body':{'path':str(bp.relative_to(ROOT)),'sha256':sha(bp)},'hand':{'path':str(hp.relative_to(ROOT)),'sha256':sha(hp)}}
 seeds=[v.x0.copy()]
 for f in (53,55):
  x=v.x0.copy();x[:7]=v.q[f,v.arm];x[7:12]=.12;x[12]=.45;seeds.append(x)
 for si,seed in enumerate(seeds[:min(a.seeds,3)]):
  x=np.clip(seed,v.lower+1e-7,v.upper-1e-7);history=[]
  for phase,nfev,outer_count in [('front_init',a.init_nfev,2),('raw_rays',a.ray_nfev,2)]:
   for outer in range(outer_count):
    _,_,_,_,tf=v.decode(x);active={}
    for name,p in v.model.dense_contact_vertices.items():
     t=tf['right_hand_ro_'+name];sdf=box_signed_distance(p@t[:3,:3].T+t[:3,3],v.boxT,v.boxsize);ii=np.argpartition(sdf,min(31,len(p)-1))[:min(32,len(p))];active[name]=np.concatenate((v.model.contact_vertices[name],p[ii]))
    def residual(z):
     state,act,T,g,tf=v.decode(z);surf=surface(tf,v.model.contact_vertices);nn=cKDTree(surf).query(target)[1];delta=surf[nn]-target
     boundary=v.boundary(g,12);back=cKDTree(boundary).query(v.obs['training_all_points'])[0];forward=sample(v.obs['distance'],boundary,cval=40);outside=sample(v.obs['outside'],boundary,cval=40)
     result=[delta.ravel()/.008*np.sqrt(250/len(target)),forward*.25/3,outside*.2/3,back*.7/3]
     sd=box_signed_distance(surface(tf,active),v.boxT,v.boxsize);pen=np.maximum(.001-sd,0)/.003;result.extend((pen*np.sqrt(1500/len(pen)),np.asarray([pen.max()*70]),np.asarray([max(float(sd.min())-.003,0)/.005*20])))
     world=v.kin.points(state,v.point_links,v.zero)
     for view in range(2):
      good=L['valid'][54,view].copy();good[[3,7]]=False;uv=v.project(v.P[view],world);diff=uv[good]-L['positions_xy'][54,view,good];weights=np.asarray([1,1.5,3,0,1,1.5,3,0]);result.append((diff*weights[good,None]/3).ravel());err=np.linalg.norm(diff,axis=1);wi=2+4*view;werr=np.linalg.norm(uv[wi]-L['positions_xy'][54,view,wi]);budgets=np.asarray([min(8,source_ground[view]['wrist_error_px']+3)-.1,min(8,source_ground[view]['median_px']+2)-.1,min(16,source_ground[view]['p90_px']+3)-.1]);result.append(np.maximum(np.asarray([werr,np.median(err),np.percentile(err,90)])-budgets,0)*50)
     if phase=='raw_rays':
      hand,arm=v.ray_depths(T);depth=hand[fixed_indices];occluder=np.minimum(v.static_depth[fixed_indices],arm[fixed_indices]);hit=np.isfinite(depth);gap=np.zeros(len(depth));gap[hit]=np.maximum(depth[hit]-occluder[hit]+.0005,0);gap[~hit]=np.minimum(np.linalg.norm(delta[~hit],axis=1),.3)
      result.append(gap/.006*np.sqrt(150/len(gap)))
     result.extend(((z[:7]-v.x0[:7])*.02,(z[7:]-v.x0[7:])*.01));return np.concatenate(result)
    fit=least_squares(residual,x,bounds=(v.lower,v.upper),diff_step=.002,max_nfev=nfev,loss='soft_l1',f_scale=3,ftol=3e-5,xtol=3e-5,gtol=3e-4);x=fit.x;history.append({'phase':phase,'outer':outer,'nfev':fit.nfev,'cost':float(fit.cost)})
   m,pred=v.metrics(x);paths=save(x,si,m,phase);record={'seed':si,'phase':phase,'x':x,'metrics':m,'rejections':criteria(m),'outputs':paths,'history':history.copy()};trials.append(record);print(json.dumps(clean({'seed':si,'phase':phase,'metrics':m,'rejections':record['rejections'],'outputs':paths,'elapsed':time.monotonic()-started})),flush=True)
   report={'schema':'kinogen.v4.frame54.raw_visibility.1','status':'finite_search_running','inputs':{k:{'path':str(p.relative_to(ROOT)),'sha256':sha(p)}for k,p in v.paths.items()},'baseline':baseline,'trials':trials,'ground_anchored_wrist_budget_px':v.wrist_budget,'elapsed_seconds':time.monotonic()-started,'strict_metric_valid':False,'observation_source':'Frozen RGB-only gray hand/domain6619; fixed spatial6px mod5 heldout; no model-selected observation pixels. Conditional front-box points only initialization/weak prior, not measured3D.'};(ROOT/'reports/user-v4-frame54-visibility.json').write_text(json.dumps(clean(report),indent=2));canvas=cv2.cvtColor(v.rgb[0],cv2.COLOR_RGB2BGR);edge=pred&~cv2.erode(pred.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);canvas[edge]=(255,255,0);cv2.imwrite(str(out/f'candidate_{si}_{phase}_visible.jpg'),cv2.resize(canvas[270:375,235:355],(600,525)))
   if not record['rejections']:break
  if not trials[-1]['rejections']:break
 report['status']='finite_search_completed';report['elapsed_seconds']=time.monotonic()-started;(ROOT/'reports/user-v4-frame54-visibility.json').write_text(json.dumps(clean(report),indent=2));print(json.dumps(clean({'complete':True,'elapsed_seconds':report['elapsed_seconds']})),flush=True)
if __name__=='__main__':main()
