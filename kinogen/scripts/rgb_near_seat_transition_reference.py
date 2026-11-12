"""Keyframe-only seated lift feasibility probe; no dynamics certification."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v21_grounded_full398.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v5_source_hipcyl.xml')
    p.add_argument('--frames',type=int,nargs='+',default=[198,205,210,215,220])
    p.add_argument('--output',default='outputs/rgb_near_pipeline/seat_transition_keyframe_probe.json')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    from scipy.spatial import ConvexHull
    z=np.load(a.root/a.input,allow_pickle=False);q=z['qpos'];names=list(z['dof_names']);m=mj.MjModel.from_xml_path(str(a.root/a.scene));d=mj.MjData(m)
    rj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_vita_demo_free_root');rq=int(m.jnt_qposadr[rj]);js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_'+str(n))for n in names]);qa=m.jnt_qposadr[js]
    ankles=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_'+s+'_ankle_roll_link')for s in('left','right')]
    hips=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'p1_collision_'+s+'_hip_roll_cylinder')for s in('l','r')];seat=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'bench_seat_proxy')
    def gn(g):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or''
    rfeet=[g for g in range(m.ngeom)if gn(g).startswith('p1_collision_foot_r_')and m.geom_contype[g]]
    def footz():return min(d.geom_xpos[g,2]-m.geom_size[g,0]-m.geom_size[g,1]*abs(d.geom_xmat[g].reshape(3,3)[2,2])for g in rfeet)
    results=[]
    for f in a.frames:
        d.qpos[rq:rq+7]=q[f,0,:7];d.qpos[qa]=q[f,0,7:];mj.mj_forward(m,d);tp=d.xpos[ankles].copy();tR=d.xmat[ankles].reshape(2,3,3).copy();rootq=q[f,0,3:7].copy();rootp=q[f,0,:3].copy();original=q[f,0,7:19].copy();lim=m.jnt_range[js[:12]]
        def bind(x):
            d.qpos[rq:rq+3]=rootp+[0,x[0],x[1]];roll=np.array([np.cos(x[2]/2),np.sin(x[2]/2),0,0]);mj.mju_mulQuat(d.qpos[rq+3:rq+7],rootq,roll);d.qpos[qa[:12]]=x[3:];mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
        def fun(x):
            bind(x);gaps=np.array([mj.mj_geomDistance(m,d,g,seat,1.,np.zeros(6))for g in hips]);rows=[]
            for k,b in enumerate(ankles):rows.extend([8*(d.xpos[b]-tp[k]),.25*(d.xmat[b].reshape(3,3)-tR[k]).ravel()])
            return np.concatenate(rows+[12*(gaps+.0008),np.array([30*(footz()+.0002)]),.01*x[:3],.0001*(x[3:]-original)])
        lo=np.r_[0.,-.05,-.55,lim[:,0]+1e-9];hi=np.r_[.10,.03,.55,lim[:,1]-1e-9]
        best=None
        for seed in ([.04,-.01,-.15],[.03,-.02,-.3],[0.,0.,.15]):
            x0=np.r_[seed,original];fit=least_squares(fun,np.clip(x0,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=200,diff_step=1e-5,ftol=1e-9,xtol=1e-9,gtol=1e-9)
            if best is None or fit.cost<best.cost:best=fit
        bind(best.x);mj.mj_forward(m,d)
        base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_base_link');support=[]
        for c in d.contact:
            pair=gn(int(c.geom1))+gn(int(c.geom2))
            if'p1_'in pair and('bench_seat_proxy'in pair or'ground_proxy'in pair):support.append(c.pos[:2].copy())
        margin=None
        if len(support)>=3:
            hull=ConvexHull(np.array(support));margin=float(-np.max(hull.equations[:,:2]@d.subtree_com[base,:2]+hull.equations[:,2]))
        row={'frame':f,'time_s':float(z['time_s'][f]),'dy_dz_roll':best.x[:3].tolist(),'qpos_p1':np.r_[d.qpos[rq:rq+7],d.qpos[qa]].tolist(),'COM_world':d.subtree_com[base].tolist(),'COM_support_polygon_margin_m':margin,'foot_position_error_m':np.linalg.norm(d.xpos[ankles]-tp,axis=1).tolist(),'foot_rotation_matrix_norm_error':np.linalg.norm(d.xmat[ankles].reshape(2,3,3)-tR,axis=(1,2)).tolist(),'hip_seat_gap_m':[float(mj.mj_geomDistance(m,d,g,seat,1.,np.zeros(6)))for g in hips],'right_foot_floor_z_m':float(footz()),'cost':float(best.cost),'nfev':best.nfev}
        print(json.dumps({k:v for k,v in row.items()if k!='qpos_p1'}),flush=True);results.append(row)
    (a.root/a.output).write_text(json.dumps({'schema':'seated_lift_keyframe_probe','kinematic_only':True,'results':results},indent=2))


if __name__=='__main__':main()
