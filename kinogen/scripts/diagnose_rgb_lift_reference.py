"""Read-only reference lift-transition velocities and support geometry."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--reference',default='outputs/rgb_near_pipeline/reference_v21_grounded_full398.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v5_source_hipcyl.xml')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/lift_reference_v21_diagnosis.json')
    p.add_argument('--time-start',type=float,default=8.3);p.add_argument('--time-end',type=float,default=10.)
    p.add_argument('--actor',choices=['p1','p2'],default='p1')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.spatial import ConvexHull
    from scipy.optimize import linprog
    z=np.load(a.root/a.reference,allow_pickle=False);m=mj.MjModel.from_xml_path(str(a.root/a.scene));d=mj.MjData(m);names=list(z['dof_names']);q=z['qpos'];t=z['time_s'];N=len(t)
    ac=0 if a.actor=='p1'else 1;pre=a.actor+'_'
    r=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root');js=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+str(s))for s in names]);rq=int(m.jnt_qposadr[r]);qa=m.jnt_qposadr[js]
    sq=int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free')]);base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,pre+'base_link');ankles=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,pre+s+'_ankle_roll_link')for s in('left','right')]
    foot=[];com=[];geometry=[]
    def gn(g):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,int(g))or''
    for f in range(N):
        d.qpos[rq:rq+7]=q[f,ac,:7];d.qpos[qa]=q[f,ac,7:];d.qpos[sq:sq+7]=z['object_target_pose'][f];mj.mj_forward(m,d)
        foot.append(d.xpos[ankles].copy());com.append(d.subtree_com[base].copy())
        if a.time_start<=t[f]<=a.time_end:
            support=[];contact=[];wrenches=[]
            for c in d.contact:
                n1,n2=gn(c.geom1),gn(c.geom2)
                if ((pre in n1+n2)and('bench_seat_proxy'in n1+n2 or'ground_proxy'in n1+n2))or(a.actor=='p1'and'shoe_'in n1+n2 and'ground_proxy'in n1+n2):
                    contact.append({'pair':[n1,n2],'gap_m':float(c.dist),'position_world':c.pos.tolist()});support.append(c.pos[:2].copy())
                    # Contact forces at seat and floor have different heights;
                    # projected COM polygon alone is not an equilibrium proof.
                    normal=c.frame[:3].copy()
                    if n1.startswith(pre)or n1.startswith('shoe_'):normal=-normal
                    tang1=c.frame[3:6];tang2=c.frame[6:9];mu=float(min(c.friction[:2]))
                    for theta in np.linspace(0,2*np.pi,8,endpoint=False):
                        force=normal+mu*(np.cos(theta)*tang1+np.sin(theta)*tang2)
                        wrenches.append(np.r_[force,np.cross(c.pos-d.subtree_com[base],force)])
            margin=None
            if len(support)>=3:
                try:
                    hull=ConvexHull(np.array(support));signed=hull.equations[:,:2]@d.subtree_com[base,:2]+hull.equations[:,2];margin=float(-signed.max())
                except Exception:pass
            feasibility=None
            if wrenches:
                mass=float(m.body_subtreemass[base]);A=np.array(wrenches).T;b=np.r_[0,0,mass*9.81,0,0,0];sol=linprog(np.ones(A.shape[1]),A_eq=A,b_eq=b,bounds=(0,None),method='highs')
                feasibility={'feasible':bool(sol.success),'scope':'8-ray inscribed friction cones, gravity equilibrium only, no actuator effort or dynamic certification','residual_norm':float(np.linalg.norm(A@sol.x-b))if sol.success else None}
            geometry.append({'frame':f,'time_s':float(t[f]),'root_world':q[f,ac,:3].tolist(),'COM_world':d.subtree_com[base].tolist(),'COM_support_polygon_margin_m':margin,'static_multilevel_contact_wrench_feasibility':feasibility,'contacts':contact})
    foot=np.array(foot);com=np.array(com);vf=np.gradient(foot,t,axis=0);vc=np.gradient(com,t,axis=0);vq=np.gradient(q[:,ac,7:19],t,axis=0)
    for row in geometry:
        f=row['frame'];row['foot_ankle_world']=foot[f].tolist();row['foot_velocity_world_m_s']=vf[f].tolist();row['COM_velocity_world_m_s']=vc[f].tolist();row['leg_joint_velocity_rad_s']=vq[f].tolist()
    out=a.root/a.output;out.write_text(json.dumps({'reference':a.reference,'rows':geometry,'max_lift_foot_speed_m_s':np.linalg.norm(vf[(t>=8.3)&(t<=10)],axis=-1).max(axis=0).tolist(),'max_lift_leg_joint_speed_rad_s':np.abs(vq[(t>=8.3)&(t<=10)]).max(axis=0).tolist()},indent=2))
    margins=[r['COM_support_polygon_margin_m']for r in geometry if r['COM_support_polygon_margin_m']is not None]
    print(json.dumps({'output':str(out),'actor':a.actor,'samples':[r for r in geometry if r['frame']in(198,203,205,210,215,220)],'min_COM_polygon_margin_m':min(margins)if margins else None,'max_lift_foot_speed_m_s':np.linalg.norm(vf[(t>=8.3)&(t<=10)],axis=-1).max(axis=0).tolist(),'max_lift_leg_joint_speed_rad_s':np.abs(vq[(t>=8.3)&(t<=10)]).max(axis=0).tolist()}),flush=True)


if __name__=='__main__':main()
