"""Read-only original body contour/landmark gates for source and V4 F54."""
import os,json
os.environ['CUDA_VISIBLE_DEVICES']=''
from user_visibility_v4 import Visibility54,ROOT,sha
from refine_frame54_visibility_v4 import clean
def main():
 import numpy as np,cv2,open3d as o3d,trimesh
 from scipy.spatial import cKDTree
 from scipy.spatial.transform import Rotation
 v=Visibility54();L=v.d['landmarks'];h,w=240,213;sx,sy=640/w,720/h;xx,yy=np.meshgrid((np.arange(w)+.5)*sx,(np.arange(h)+.5)*sy);rays=[];targets=[]
 def domain(points,view):
  good=np.ones(len(points),bool)
  for k,radius in ((2+4*view,55.),(3+4*view,35.)):
   if L['valid'][54,view,k]:good &=np.linalg.norm(points-L['positions_xy'][54,view,k],axis=1)>radius
  return good
 for view in range(2):
  M=v.P[view,:,:3];center=-np.linalg.solve(M,v.P[view,:,3]);direction=np.stack((xx,yy,np.ones_like(xx)),axis=-1)@np.linalg.inv(M).T;direction/=np.linalg.norm(direction,axis=-1,keepdims=True);rays.append(o3d.core.Tensor(np.concatenate((np.broadcast_to(center,direction.shape),direction),axis=-1).astype(np.float32)));mask=v.robot[view]&v.valid[view];contours=cv2.findContours(mask.astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0];points=np.concatenate([c[:,0]for c in contours if len(c)>4]).astype(float)+.5;points=points[domain(points,view)]
  if len(points)>650:points=points[np.linspace(0,len(points)-1,650).astype(int)]
  held=((points[:,0]//16).astype(int)+64*(points[:,1]//16).astype(int))%5==0;dom=domain(np.column_stack((xx.ravel(),yy.ravel())),view).reshape(h,w);targets.append({'heldout':points[held],'domain':dom,'mask':(cv2.resize(mask.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0)&dom})
 keep=np.asarray(['_hand_ro_'not in v.kin.link_names[i]for i in v.kin.proxy_face_links]);faces=v.kin.proxy_faces[keep];table=v.kin.scene['table'];parts=[]
 for size,center in [(table['top_size_xyz_m'],table['top_center_xyz_m'])]+[([.035,.035,.735],[x,y,.3675])for x in (.19,.77)for y in (-.355,.355)]:
  m=trimesh.creation.box(extents=size);m.apply_translation(center);parts.append(m)
 tm=trimesh.util.concatenate(parts)
 def metric(state):
  scene=o3d.t.geometry.RaycastingScene(nthreads=4);scene.add_triangles(o3d.core.Tensor(v.kin.proxy_world(state).astype(np.float32)),o3d.core.Tensor(faces));scene.add_triangles(o3d.core.Tensor(np.asarray(tm.vertices,np.float32)),o3d.core.Tensor(np.asarray(tm.faces,np.uint32)));answer=[]
  for view in range(2):
   cast=scene.cast_rays(rays[view],nthreads=4);hit=(cast['geometry_ids'].numpy()==0)&~(cv2.resize(v.boxmask[view].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0);edge=hit&~cv2.erode(hit.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool);py,px=np.nonzero(edge);xy=np.column_stack(((px+.5)*sx,(py+.5)*sy));xy=xy[domain(xy,view)]
   if len(xy)>500:xy=xy[np.linspace(0,len(xy)-1,500).astype(int)]
   target=targets[view];mask=hit&target['domain'];d=cKDTree(xy).query(target['heldout'])[0];answer.append({'body_mask_iou':float((mask&target['mask']).sum()/max((mask|target['mask']).sum(),1)),'heldout_body_contour_median_px':float(np.median(d)),'heldout_body_contour_p90_px':float(np.percentile(d,90))})
  return answer
 G=v.d['ground'];ground=np.r_[G['base_xyz_m'][54],Rotation.from_quat(G['base_quaternion_xyzw'][54]).as_rotvec(),G['joint_positions_rad'][54]];bp=ROOT/'outputs/user_reconstruction_v4/stage_body.npz';B=np.load(bp);stage=np.r_[B['base_xyz_m'][54],Rotation.from_quat(B['base_quaternion_xyzw'][54]).as_rotvec(),B['joint_positions_rad'][54]];metrics={'ground_reference':metric(ground),'source_v3':metric(v.state),'stage_v4':metric(stage)};reasons=[]
 for i,(a,b)in enumerate(zip(metrics['stage_v4'],metrics['ground_reference'])):
  for key,limit,delta in [('body_mask_iou',.55,-.04),('heldout_body_contour_median_px',6,2),('heldout_body_contour_p90_px',15,3)]:
   fail=a[key]<max(limit,b[key]+delta)if key=='body_mask_iou'else a[key]>min(limit,b[key]+delta)
   if fail:reasons.append(f'view{i}_{key}')
 report={'schema':'kinogen.v4.body_contour_audit.1','inputs':{'stage_body':{'path':str(bp.relative_to(ROOT)),'sha256':sha(bp)},'ground':{'path':str(v.paths['ground'].relative_to(ROOT)),'sha256':sha(v.paths['ground'])}},'frame':54,'metrics':metrics,'gate_rejections':reasons,'body_contour_gates_passed':not reasons,'policy':'Original proxy body heldout, hand wrist/palm exclusion55/35px; fixed source SAM and block16 spatial split, max650 sourcepoints/max500 predictionpoints; same ground anchor.'};p=ROOT/'reports/user-v4-body-contour-audit.json';p.write_text(json.dumps(clean(report),indent=2));print(json.dumps(clean(report)),flush=True)
if __name__=='__main__':main()
