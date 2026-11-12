"""Offline late-contact transition repair with fixed world stance anchors.

Root and swing trajectories are planned continuously across contact changes.
COM may leave the static support polygon; full dynamics must independently pass.
"""
from pathlib import Path
import argparse,hashlib,json,sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-nfev',type=int,default=100);a=p.parse_args()
    if sys.platform!='linux':p.error('Run offline FK/IK on the server')
    if a.output.exists()or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    import numpy as np,mujoco as mj
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation,Slerp
    from scipy.interpolate import PchipInterpolator
    from rgb145_contact_plan import convex_hull,halfspaces
    z=dict(np.load(a.reference,allow_pickle=False));q0=np.asarray(z['qpos'],float);q=q0.copy();on=np.asarray(z['foot_support_expected'],bool)
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);names=list(map(str,z['dof_names']));jids=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]);qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    leg=np.asarray([i for i,n in enumerate(names)if n.startswith(('left_','right_'))and any(k in n for k in('hip_','knee_','ankle_'))]);feet=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in('left','right')]);base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'base_link')
    if q.shape!=(350,59)or len(leg)!=12:raise ValueError('Requires current350frame Vita reference')
    fk=[];R=[]
    for row in q0:
        d.qpos[:]=row;mj.mj_fwdPosition(m,d);fk.append(d.xpos[feet].copy());R.append(d.xmat[feet].reshape(2,3,3).copy())
    fk=np.asarray(fk);R=np.asarray(R);target=fk.copy();target_R=R.copy();root=q0[:,:3].copy();rootquat=q0[:,3:7].copy();windows=[(190,205),(225,251)];swing_specs=[(1,190,201),(0,225,236),(1,237,246)]
    def smooth(u):return u*u*u*(10-15*u+6*u*u)
    # Continuous source-relative swing adaptation, with delayed lowering to clear
    # the unchanged step edge before approaching the lower support surface.
    for side,start,land in swing_specs:
        begin=fk[start,side].copy();end=fk[land,side].copy();top=max(begin[2],end[2])+.045
        orient=Slerp([0.,1.],Rotation.from_matrix(R[[start,land],side]))
        for f in range(start,land+1):
            u=(f-start)/(land-start);w=smooth(u);target[f,side]=(1-w)*begin+w*end
            if u<.28:target[f,side,2]=(1-smooth(u/.28))*begin[2]+smooth(u/.28)*top
            else:target[f,side,2]=(1-smooth((u-.28)/.72))*top+smooth((u-.28)/.72)*end[2]
            target_R[f,side]=orient([w]).as_matrix()[0]
    for lo,hi in windows:
        idx=np.arange(lo,hi+1);knots=np.asarray([190,201,205]if lo==190 else[225,236,246,251]);spline=PchipInterpolator(knots,q0[knots,:3])
        root[idx]=spline(idx);orient=Slerp(knots,Rotation.from_quat(np.roll(q0[knots,3:7],-1,axis=1)));rootquat[idx]=np.roll(orient(idx).as_quat(),1,axis=1)
        # Exactly preserve every established stance position/orientation. A
        # contact mode never authorizes sliding the support to satisfy reach.
        for s in range(2):
            edges=np.diff(np.r_[0,on[:,s].astype(int),0])
            for st,en in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
                use=np.arange(max(lo,st),min(hi+1,en))
                if len(use):target[use,s]=np.asarray(z['foot_target_ankle_world'])[st,s];target_R[use,s]=R[min(st+2,en-1),s]
    robot=[g for g in range(m.ngeom)if m.geom_bodyid[g]>0 and m.geom_contype[g]];ri={g:i for i,g in enumerate(robot)};support={g for g in range(m.ngeom)if m.geom_bodyid[g]==0 and m.geom_contype[g]}
    foot_geoms=[[g for g in robot if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('collision_foot_'+s+'_')]for s in ('l','r')]
    footprint=[]
    for s in range(2):
        d.qpos[:]=q0[0];mj.mj_kinematics(m,d);points=[];ori=d.xmat[feet[s]].reshape(3,3)
        for g in foot_geoms[s]:
            radius=m.geom_size[g,0];center=d.geom_xpos[g];axis=d.geom_xmat[g].reshape(3,3)[:,2];ends=[center]if m.geom_type[g]==mj.mjtGeom.mjGEOM_SPHERE else[center-axis*m.geom_size[g,1],center+axis*m.geom_size[g,1]]
            for c in ends:
                th=np.arange(16)*2*np.pi/16;points.extend((c-d.xpos[feet[s]])@ori+radius*np.c_[np.cos(th),np.sin(th),np.zeros(16)])
        footprint.append(np.asarray(points))
    n=17;H=np.zeros((m.nv,n));H[:3,:3]=np.eye(3);H[va[leg],np.arange(5,n)]=1;T=np.c_[np.eye(3),np.zeros((3,14))];P=np.c_[np.zeros((12,5)),np.eye(12)];O=np.c_[np.zeros((2,3)),np.eye(2),np.zeros((2,12))];rows=[]
    def skew(v):return np.asarray([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    for lo,hi in windows:
        previous=q0[lo,qa[leg]].copy();previous_root=q0[lo,:3].copy()
        for f in range(lo+1,hi):
            fixed=q0[f].copy();fixed[3:7]=rootquat[f];yaw,pitch,roll=Rotation.from_quat(np.roll(rootquat[f],-1)).as_euler('ZYX');seed=np.r_[root[f],roll,pitch,previous]
            lower=np.r_[root[f]-[.12,.12,.12],-.65,-.65,m.jnt_range[jids[leg],0]+1e-6];upper=np.r_[root[f]+[.12,.12,.12],.65,.65,m.jnt_range[jids[leg],1]-1e-6];lower[2]=max(.4,lower[2]);cache={}
            root_carry=previous_root+(root[f]-root[f-1])-(previous_root-root[f-1])/max(1,hi-f)
            cloud=[target[f,s,:2]+(footprint[s]@target_R[f,s].T)[:,:2]for s in range(2)if on[f,s]];hull=convex_hull(np.concatenate(cloud));A,b=halfspaces(hull)
            def evaluate(x):
                if 'x'in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
                d.qpos[:]=fixed;d.qpos[:3]=x[:3];d.qpos[3:7]=np.roll(Rotation.from_euler('ZYX',[yaw,x[4],x[3]]).as_quat(),1);d.qpos[qa[leg]]=x[5:]
                H[3:6,3]=[1,0,0];H[3:6,4]=[0,np.cos(x[3]),-np.sin(x[3])]
                mj.mj_kinematics(m,d);mj.mj_comPos(m,d);rr=[];JJ=[]
                def add(v,j,w):rr.extend(np.asarray(v).ravel()*w);JJ.extend(np.asarray(j).reshape(-1,n)*w)
                def point(body,pos):
                    jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,pos,body);return jp@H,jr@H
                add(x[:3]-root[f],T,15.);add(x[:3]-root_carry,T,30.);add(x[3:5]-[roll,pitch],O,1.);add(x[5:]-q0[f,qa[leg]],P,.035);add(x[5:]-previous,P,.05)
                for s in range(2):
                    body=int(feet[s]);jp,jr=point(body,d.xpos[body]);add(d.xpos[body]-target[f,s],jp,400. if on[f,s]else 60.)
                    ori=d.xmat[body].reshape(3,3)
                    for axis in (0,2):add(ori[:,axis]-target_R[f,s,:,axis],-skew(ori[:,axis])@jr,8. if on[f,s]else .5)
                jc=np.zeros((3,m.nv));mj.mj_jacSubtreeCom(m,d,jc,base);violation=b-A@d.subtree_com[base,:2];cj=-A@(jc@H)[:2];cj[violation<=0]=0.;add(np.maximum(violation,0.),cj,.5)
                mj.mj_collision(m,d);cr=np.zeros(len(robot));cj=np.zeros((len(robot),n))
                for c in d.contact[:d.ncon]:
                    g1,g2=int(c.geom1),int(c.geom2)
                    if g1 in ri and g2 in support:g=g1;normal=-np.asarray(c.frame[:3])
                    elif g2 in ri and g1 in support:g=g2;normal=np.asarray(c.frame[:3])
                    else:continue
                    k=ri[g]
                    if c.dist<-.001 and c.dist+.001<cr[k]:cr[k]=c.dist+.001;cj[k]=normal@point(int(m.geom_bodyid[g]),np.asarray(c.pos))[0]
                add(cr,cj,200.);cache.update(x=x.copy(),r=np.asarray(rr),J=np.asarray(JJ));return cache['r'],cache['J']
            fit=least_squares(lambda x:evaluate(x)[0],np.clip(seed,lower+1e-7,upper-1e-7),jac=lambda x:evaluate(x)[1],bounds=(lower,upper),max_nfev=a.max_nfev,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            evaluate(fit.x);q[f]=d.qpos.copy();previous=fit.x[5:].copy();previous_root=fit.x[:3].copy();rows.append({'frame':f,'cost':float(fit.cost),'nfev':int(fit.nfev),'ankle_target_error_m':np.linalg.norm(d.xpos[feet]-target[f],axis=1).tolist(),'root_plan_error_m':float(np.linalg.norm(fit.x[:3]-root[f])),'root_roll_pitch_plan_error_rad':(fit.x[3:5]-[roll,pitch]).tolist(),'static_com_support_violation_m':float(np.maximum(b-A@d.subtree_com[base,:2],0).max())})
            if f%5==0:print(json.dumps(rows[-1]),flush=True)
    quaternion_sign_correction_frames=[]
    for f in range(1,len(q)):
        if np.dot(q[f-1,3:7],q[f,3:7])<0:
            q[f,3:7]*=-1;quaternion_sign_correction_frames.append(f)
    actual=[];com=[];pen=[];joint=[]
    for row in q:
        d.qpos[:]=row;mj.mj_fwdPosition(m,d);actual.append(d.xpos[feet].copy());com.append(d.subtree_com[base].copy());pen.append(max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.]));v=np.maximum(np.maximum(m.jnt_range[jids,0]-row[qa],row[qa]-m.jnt_range[jids,1]),0);v[~m.jnt_limited[jids].astype(bool)]=0;joint.append(v)
    actual=np.asarray(actual);com=np.asarray(com);pen=np.asarray(pen);joint=np.asarray(joint);modified=np.any(q!=q0,axis=1);unchanged=np.ones(59,bool);unchanged[:7]=False;unchanged[qa[leg]]=False;assert np.array_equal(q[:,unchanged],q0[:,unchanged])
    out={k:v for k,v in z.items()if k.startswith('source_')or k in('frame_ids','time_s','dof_names','foot_support_expected','foot_support_confidence','foot_support_geom_name','wrist_reference_joint_names','wrist_reference_joint_targets_rad')};out.update({'parent_reference_'+k:v for k,v in z.items()if not k.startswith('source_')});out.update(qpos=q,foot_target_ankle_world=target,foot_actual_ankle_world=actual,reference_com_world=com,root_target_world=root,static_penetration_m=pen,static_joint_limit_violation_rad=joint,static_geometry_passed=np.asarray(pen.max()<=.02 and joint.max()<=.05),physical_joint_limit_model_sha256=np.asarray(sha(a.model)),transition_repair_version=np.asarray('v20_continuous_late_contact_plans'))
    boundaries=[]
    for f,s in [(201,1),(236,0),(238,1),(246,1)]:boundaries.append({'frame':f,'side':['left','right'][s],'parent_ankle_step_m':float(np.linalg.norm(fk[f,s]-fk[f-1,s])),'new_ankle_step_m':float(np.linalg.norm(actual[f,s]-actual[f-1,s])),'parent_root_step_m':float(np.linalg.norm(q0[f,:3]-q0[f-1,:3])),'new_root_step_m':float(np.linalg.norm(q[f,:3]-q[f-1,:3]))})
    report={'version':'v20d_continuous_late_contact_plans_hemisphere','physics_passed':False,'model_parameters_modified':False,'source_frames_preserved':350,'source_fields_preserved':True,'upper_body_hands_unchanged':True,'modified_frames':np.flatnonzero(modified).tolist(),'quaternion_hemisphere_sign_correction_frames':quaternion_sign_correction_frames,'quaternion_adjacent_dot_min':float(np.sum(q[:-1,3:7]*q[1:,3:7],axis=1).min()),'root_and_leg_planning_windows':windows,'swing_plans':swing_specs,'contact_labels_modified':False,'static_com_hinge_weight':.5,'dynamic_COM_departure_allowed':True,'static_com_support_violation_max_m':max(r['static_com_support_violation_m']for r in rows),'foot_target_error_max_m':max(max(r['ankle_target_error_m'])for r in rows),'root_source_reference_offset_max_m':float(np.linalg.norm(q[:,:3]-q0[:,:3],axis=1).max()),'foot_source_reference_offset_max_m':float(np.linalg.norm(actual-fk,axis=2).max()),'penetration_max_m':float(pen.max()),'joint_limit_violation_max_rad':float(joint.max()),'static_geometry_passed':bool(pen.max()<=.02 and joint.max()<=.05),'boundaries':boundaries,'input_reference':str(a.reference),'input_reference_sha256':sha(a.reference),'model_sha256':sha(a.model),'generator_sha256':sha(__file__),'rows':rows,'limitations':['Continuous geometric plans and IK are not full motion tracking or force feasibility.','COM can leave the static support polygon; actual closed-loop dynamics and contact validation are required.','Adjacent30Hz pose steps are interpolated by declared adaptive retiming; they are not physics-state teleports.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**out);a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True)


if __name__=='__main__':main()
