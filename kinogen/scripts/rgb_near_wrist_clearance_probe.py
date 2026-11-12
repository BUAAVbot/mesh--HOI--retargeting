"""Read-only SAT wrist-retreat probe; outputs hypotheses, never a new reference."""
from pathlib import Path
import argparse
import json
import hashlib


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v26_geometry_repaired_full398.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v5_source_hipcyl.xml')
    p.add_argument('--frame',type=int,default=360);p.add_argument('--output',default='outputs/rgb_near_pipeline/wrist_clearance_probe_v26.json')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    from rgb_near_convex_distance import ShoeHandConvexDistance
    z=np.load(a.root/a.input,allow_pickle=False);m=mj.MjModel.from_xml_path(str(a.root/a.scene));d=mj.MjData(m);names=[str(v)for v in z['dof_names']];f=a.frame
    for ac,pre in enumerate(('p1_','p2_')):
        r=int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root')]);js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+n)for n in names]);d.qpos[r:r+7]=z['qpos'][f,ac,:7];d.qpos[m.jnt_qposadr[js]]=z['qpos'][f,ac,7:]
    sq=int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free')]);d.qpos[sq:sq+7]=z['object_target_pose'][f];mj.mj_forward(m,d);fixed=d.qpos.copy();hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link');hp=d.xpos[hand].copy();hR=d.xmat[hand].reshape(3,3).copy()
    js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_'+n)for n in names]);sel=np.array([k for k,n in enumerate(names)if n.startswith('right_')and any(s in n for s in('shoulder','elbow','wrist'))]);j=js[sel];qa=m.jnt_qposadr[j];va=m.jnt_dofadr[j];lim=m.jnt_range[j];original=d.qpos[qa].copy();jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);cache=ShoeHandConvexDistance(m)
    hg=[g for g in range(m.ngeom)if m.geom_contype[g]and(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith('p1_rh_')];sg=[g for g in range(m.ngeom)if m.geom_contype[g]and(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('shoe_')]
    axis=np.array([.249795,-.364251,-.897175]);axis/=np.linalg.norm(axis);rows=[]
    for sign in(-1,1):
      for step in(.005,.008):
        delta=sign*step*axis;target=hp+delta;d.qpos[:]=fixed
        def fun(x):d.qpos[qa]=x;mj.mj_kinematics(m,d);mj.mj_comPos(m,d);return np.r_[10*(d.xpos[hand]-target),.3*(d.xmat[hand].reshape(3,3)-hR).T.ravel()]
        def jac(x):
            fun(x);mj.mj_jacBody(m,d,jp,jr,hand);R=d.xmat[hand].reshape(3,3);return np.vstack([10*jp[:,va],.3*np.concatenate([np.cross(jr[:,va].T,R[:,k]).T for k in range(3)],axis=0)])
        fit=least_squares(fun,original,jac=jac,bounds=(lim[:,0]+1e-9,lim[:,1]-1e-9),max_nfev=60);fun(fit.x);mj.mj_forward(m,d);pairs=[]
        for g in hg:
            for s in sg:
                value=cache.query(d,g,s,precise=False);pairs.append((value['signed_gap_m'],mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,s)))
        closest=min(pairs);row={'sign':sign,'translation_m':step,'hand_local_retreat_m':(hR.T@delta).tolist(),'wrist_position_error_m':float(np.linalg.norm(d.xpos[hand]-target)),'SAT_min_signed_gap_m':float(closest[0]),'closest_pair':closest[1:],'right_arm_q_rad':fit.x.tolist()};print(json.dumps(row),flush=True);rows.append(row)
    out=a.root/a.output;out.write_text(json.dumps({'schema':'wrist_retreat_static_probe_v1','reference_sha256':hashlib.sha256((a.root/a.input).read_bytes()).hexdigest(),'frame':f,'candidates':rows,'not_a_reference_or_dynamic_claim':True},indent=2))


if __name__=='__main__':main()
