"""Repair hybrid world-contact continuity with offline IK; never a physics result."""
from pathlib import Path
import argparse,hashlib,json,sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=91);p.add_argument('--end-frame',type=int,default=140);p.add_argument('--max-nfev',type=int,default=100)
    a=p.parse_args()
    if sys.platform!='linux':p.error('Offline FK/IK is run on the server; no local simulation')
    if a.output.exists()or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    import numpy as np,mujoco as mj
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from rgb145_contact_plan import convex_hull,halfspaces
    from rgb145_knee_load import KneeLoadProxy
    z=dict(np.load(a.reference,allow_pickle=False));q0=np.asarray(z['qpos'],float);q=q0.copy();expected=np.asarray(z['foot_support_expected'],bool)
    if q.shape!=(350,59):raise ValueError('Requires all350 source frames')
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m)
    names=list(map(str,z['dof_names']));jids=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]);qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    leg=np.asarray([i for i,n in enumerate(names)if n.startswith(('left_','right_'))and any(s in n for s in ('hip_','knee_','ankle_'))])
    if len(leg)!=12:raise ValueError('Expected 12 leg joints')
    ankles=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in ('left','right')]);base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'base_link')
    fk=[];mat=[];com=[]
    for row in q0:
        d.qpos[:]=row;mj.mj_fwdPosition(m,d);fk.append(d.xpos[ankles].copy());mat.append(d.xmat[ankles].reshape(2,3,3).copy());com.append(d.subtree_com[base].copy())
    fk=np.asarray(fk);mat=np.asarray(mat);com=np.asarray(com)
    # Use the last valid prefix support, never the displaced suffix support.
    anchor_frame=a.start_frame-1;anchor=fk[anchor_frame,0].copy();anchor_R=mat[anchor_frame,0].copy()
    release=a.start_frame
    while release<350 and expected[release,0]:release+=1
    targets=fk.copy();rotations=mat.copy();targets[a.start_frame:release,0]=anchor;rotations[a.start_frame:release,0]=anchor_R
    # Free swing returns to the suffix endpoint only after the genuine release.
    delta=anchor-fk[release-1,0]
    for f in range(release,a.end_frame):
        u=np.clip((f-release+1)/(a.end_frame-release+1),0,1);w=1-u*u*(3-2*u);targets[f,0]+=w*delta
    # The new right support on the seat is a world anchor too.
    right_start=next(f for f in range(a.start_frame,350)if expected[f,1])
    right_end=right_start
    while right_end<350 and expected[right_end,1]:right_end+=1
    right_anchor=fk[right_start,1].copy();right_R=mat[right_start,1].copy()
    targets[right_start:right_end,1]=right_anchor;rotations[right_start:right_end,1]=right_R
    n=17;H=np.zeros((m.nv,n));H[:3,:3]=np.eye(3);H[va[leg],np.arange(5,n)]=1
    T=np.c_[np.eye(3),np.zeros((3,n-3))];P=np.c_[np.zeros((12,5)),np.eye(12)]
    robot=[g for g in range(m.ngeom)if m.geom_bodyid[g]>0 and m.geom_contype[g]];ri={g:i for i,g in enumerate(robot)};support={g for g in range(m.ngeom)if m.geom_bodyid[g]==0 and m.geom_contype[g]}
    # Derive sole envelope from unchanged collider geometry at each anchor orientation.
    foot_geoms=[[g for g in robot if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('collision_foot_'+s+'_')]for s in ('l','r')]
    def cloud(frame,side):
        d.qpos[:]=q0[frame];mj.mj_kinematics(m,d);points=[]
        for g in foot_geoms[side]:
            radius=m.geom_size[g,0];center=d.geom_xpos[g];axis=d.geom_xmat[g].reshape(3,3)[:,2]
            ends=[center]if m.geom_type[g]==mj.mjtGeom.mjGEOM_SPHERE else[center-axis*m.geom_size[g,1],center+axis*m.geom_size[g,1]]
            for c in ends:
                th=np.arange(16)*2*np.pi/16;points.extend(c[:2]+radius*np.c_[np.cos(th),np.sin(th)])
        return np.asarray(points)-fk[frame,side,:2]
    footprints=[cloud(anchor_frame,0),cloud(right_start,1)];proxy=KneeLoadProxy(m);rows=[];previous=None
    def skew(v):return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    for f in range(a.start_frame,a.end_frame):
        euler=Rotation.from_quat(np.roll(q0[f,3:7],-1)).as_euler('ZYX');yaw,pitch,roll=euler
        fixed=q0[f].copy();seed=np.r_[fixed[:3],roll,pitch,fixed[qa[leg]]]
        if f<release:seed[:3]+=targets[f,0]-fk[f,0]
        elif previous is not None:seed[:3]=previous[:3]+q0[f,:3]-q0[f-1,:3]
        low=np.r_[q0[f,:3]-[.8,.8,.45],-.65,-.65,m.jnt_range[jids[leg],0]+1e-6];high=np.r_[q0[f,:3]+[.8,.8,.35],.65,.65,m.jnt_range[jids[leg],1]-1e-6];low[2]=max(.40,low[2])
        pts=[targets[f,s,:2]+footprints[s]for s in range(2)if expected[f,s]];hull=convex_hull(np.concatenate(pts));A,b=halfspaces(hull)
        cache={}
        def evaluate(x):
            if 'x'in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
            d.qpos[:]=fixed;d.qpos[:3]=x[:3];d.qpos[3:7]=np.roll(Rotation.from_euler('ZYX',[yaw,x[4],x[3]]).as_quat(),1);d.qpos[qa[leg]]=x[5:]
            H[3:6,3]=[1,0,0];H[3:6,4]=[0,np.cos(x[3]),-np.sin(x[3])]
            mj.mj_kinematics(m,d);mj.mj_comPos(m,d);r=[];J=[]
            def add(v,j,w):r.extend(np.asarray(v).ravel()*w);J.extend(np.asarray(j).reshape(-1,n)*w)
            def point(body,pos):
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,pos,body);return jp@H,jr@H
            add(x[:3]-seed[:3],T,.5);add(x[5:]-fixed[qa[leg]],P,.08)
            add(x[3:5],np.c_[np.zeros((2,3)),np.eye(2),np.zeros((2,12))],.4)
            if previous is not None:add(x[5:]-previous[5:],P,.05)
            for s in range(2):
                body=int(ankles[s]);jp,jr=point(body,d.xpos[body]);add(d.xpos[body]-targets[f,s],jp,300. if expected[f,s]else 15.)
                R=d.xmat[body].reshape(3,3);wanted=rotations[f,s]
                for axis in (0,2):add(R[:,axis]-wanted[:,axis],-skew(R[:,axis])@jr,8. if expected[f,s]else .5)
            jc=np.zeros((3,m.nv));mj.mj_jacSubtreeCom(m,d,jc,base);violation=b+.015-A@d.subtree_com[base,:2];Jc=-A@(jc@H)[:2];Jc[violation<=0]=0;add(np.maximum(violation,0),Jc,30.)
            kr,kj,_=proxy.single_stance_cost(d,expected[f],seated=False,velocity_map=H,ratio_threshold=.8);add(kr,kj,6.)
            mj.mj_collision(m,d);cr=np.zeros(len(robot));cj=np.zeros((len(robot),n))
            for c in d.contact[:d.ncon]:
                g1,g2=int(c.geom1),int(c.geom2)
                if g1 in ri and g2 in support:g=g1;normal=-np.asarray(c.frame[:3])
                elif g2 in ri and g1 in support:g=g2;normal=np.asarray(c.frame[:3])
                else:continue
                k=ri[g]
                if c.dist<-.001 and c.dist+.001<cr[k]:cr[k]=c.dist+.001;cj[k]=normal@point(int(m.geom_bodyid[g]),np.asarray(c.pos))[0]
            add(cr,cj,150.);cache.update(x=x.copy(),r=np.asarray(r),J=np.asarray(J));return cache['r'],cache['J']
        fit=least_squares(lambda x:evaluate(x)[0],np.clip(seed,low+1e-7,high-1e-7),jac=lambda x:evaluate(x)[1],bounds=(low,high),max_nfev=a.max_nfev,ftol=1e-7,gtol=1e-7,xtol=1e-7)
        evaluate(fit.x);q[f]=d.qpos.copy();previous=fit.x.copy();rows.append({'frame':f,'cost':float(fit.cost),'nfev':int(fit.nfev),'ankle_target_error_m':np.linalg.norm(d.xpos[ankles]-targets[f],axis=1).tolist(),'com_support_violation_m':float(np.maximum(b+.015-A@d.subtree_com[base,:2],0).max()),'penetration_m':max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.])})
        if f%5==0:print(json.dumps(rows[-1]),flush=True)
    actual=[];newcom=[];penetration=[]
    for row in q:
        d.qpos[:]=row;mj.mj_fwdPosition(m,d);actual.append(d.xpos[ankles].copy());newcom.append(d.subtree_com[base].copy());penetration.append(max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.]))
    actual=np.asarray(actual);newcom=np.asarray(newcom);penetration=np.asarray(penetration);limited=m.jnt_limited[jids].astype(bool);limits=m.jnt_range[jids];violation=np.maximum(np.maximum(limits[:,0]-q[:,qa],q[:,qa]-limits[:,1]),0);violation[:,~limited]=0
    modified=np.any(q!=q0,axis=1);preserved=[i for i in range(52)if i not in leg];assert np.array_equal(q[:,qa[preserved]],q0[:,qa[preserved]])
    # Keep original arrays under a parent namespace and publish only newly recomputed geometry.
    out={k:v for k,v in z.items()if k.startswith('source_')or k in ('frame_ids','time_s','dof_names','foot_support_expected','foot_support_geom_name','foot_support_confidence','wrist_reference_joint_names','wrist_reference_joint_targets_rad')}
    out.update(qpos=q,foot_target_ankle_world=targets,foot_actual_ankle_world=actual,reference_com_world=newcom,root_target_world=q[:,:3],contact_repair_version=np.asarray('v18_world_anchor_ik'),contact_repair_modified_mask=modified,contact_repair_parent_qpos=q0,contact_repair_parent_fk=fk,static_penetration_m=penetration,static_joint_limit_violation_rad=violation,static_geometry_passed=np.asarray(penetration.max()<=.02 and violation.max()<=.05),physical_joint_limit_model_sha256=np.asarray(sha(a.model)))
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**out)
    report={'version':'v18_world_anchor_ik','physics_passed':False,'physics_steps':0,'input':{'reference':str(a.reference),'reference_sha256':sha(a.reference),'model':str(a.model),'model_sha256':sha(a.model)},'generator_sha256':sha(__file__),'source_frames_preserved':350,'source_fields_preserved':True,'hands_and_upper_body_joints_unchanged':True,'model_parameters_modified':False,'modified_frames':np.flatnonzero(modified).tolist(),'left_anchor_source_frame':anchor_frame,'left_anchor_world_m':anchor.tolist(),'left_release_frame':release,'right_seat_anchor_frame':right_start,'right_seat_anchor_world_m':right_anchor.tolist(),'left_stance_parent_slide_m':float(np.linalg.norm(fk[release-1,0]-fk[anchor_frame,0])),'left_stance_repaired_slide_m':float(np.linalg.norm(actual[release-1,0]-actual[anchor_frame,0])),'left_stance_max_anchor_error_m':float(np.linalg.norm(actual[a.start_frame:release,0]-anchor,axis=1).max()),'root_translation_change_max_m':float(np.linalg.norm(q[:,:3]-q0[:,:3],axis=1).max()),'ankle_target_change_max_m':float(np.linalg.norm(targets-fk,axis=2).max()),'penetration_max_m':float(penetration.max()),'joint_limit_violation_max_rad':float(violation.max()),'static_geometry_passed':bool(penetration.max()<=.02 and violation.max()<=.05),'continuous_dynamic_feasibility_claimed':False,'rows':rows}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True)


if __name__=='__main__':main()
