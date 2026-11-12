"""Prepare a real left swing step before the right-foot stair lift.

Offline kinematic adaptation with unchanged physical model. The known v2 ground
prefix is retained through74; incompatible anchors are crossed during swing.
"""
from pathlib import Path
import argparse,hashlib,json,sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--suffix',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-nfev',type=int,default=100);a=p.parse_args()
    if sys.platform!='linux':p.error('Offline FK/IK runs on the server')
    if a.output.exists()or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    import numpy as np,mujoco as mj
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation,Slerp
    from scipy.interpolate import PchipInterpolator
    from rgb145_contact_plan import convex_hull,halfspaces
    from rgb145_knee_load import KneeLoadProxy
    pre=dict(np.load(a.prefix,allow_pickle=False));suf=dict(np.load(a.suffix,allow_pickle=False));qpre=np.asarray(pre['qpos'],float);q0=np.asarray(suf['qpos'],float);q=q0.copy()
    if q.shape!=(350,59)or qpre.shape!=q.shape:raise ValueError('Expected all350Vita poses')
    names=list(map(str,suf['dof_names']));pre_names=list(map(str,pre['dof_names']));qpre=qpre.copy();qpre[:,7:]=pre['qpos'][:,7+np.asarray([pre_names.index(n)for n in names])]
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);jids=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]);qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    leg=np.asarray([i for i,n in enumerate(names)if n.startswith(('left_','right_'))and any(k in n for k in('hip_','knee_','ankle_'))]);aux=np.asarray([i for i,n in enumerate(names)if n.startswith(('lh_','rh_','neck_'))]);wrists=np.asarray([i for i,n in enumerate(names)if '_wrist_'in n]);upper=np.asarray([i for i in range(52)if i not in leg and i not in aux and i not in wrists]);body22=np.asarray([i for i in range(52)if i not in aux and i not in wrists])
    q[:75,:7]=qpre[:75,:7];q[:75,qa[body22]]=qpre[:75,qa[body22]];q[:,qa[aux]]=q0[0,qa[aux]];q[:,qa[wrists]]=q0[0,qa[wrists]]
    def smooth(u):return u*u*u*(10-15*u+6*u*u)
    for f in range(75,91):
        w=smooth((f-74)/16);q[f,qa[upper]]=(1-w)*qpre[f,qa[upper]]+w*q0[f,qa[upper]]
    feet=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in('left','right')]);base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'base_link')
    fk=[];R=[]
    for row in q:
        d.qpos[:]=row;mj.mj_fwdPosition(m,d);fk.append(d.xpos[feet].copy());R.append(d.xmat[feet].reshape(2,3,3).copy())
    fk=np.asarray(fk);R=np.asarray(R);target=fk.copy();target_R=R.copy();on=np.asarray(suf['foot_support_expected'],bool).copy();conf=np.asarray(suf['foot_support_confidence'],float).copy();surface=np.asarray(suf['foot_support_geom_name']).copy()
    left_start=fk[74,0].copy();right_anchor=fk[74,1].copy();left_end=np.asarray(suf['foot_target_ankle_world'])[85,0].copy();right_end=np.asarray(suf['foot_target_ankle_world'])[116,1].copy()
    #75..76double support: weight transfer precedes the new physical left swing.
    on[75:77]=True;on[77:85]=[False,True];on[85:91]=True;on[91:116]=[True,False];on[116:118]=True
    target[75:77,0]=left_start;target_R[75:77,0]=R[74,0];target[75:91,1]=right_anchor;target_R[75:91,1]=R[74,1];target[85:118,0]=left_end;target_R[85:118,0]=R[87,0]
    left_orient=Slerp([0.,1.],Rotation.from_matrix([R[74,0],R[87,0]]))
    for f in range(77,85):
        u=(f-76)/9;w=smooth(u);target[f,0]=(1-w)*left_start+w*left_end;target[f,0,2]+=.14*np.sin(np.pi*u)**2;target_R[f,0]=left_orient([w]).as_matrix()[0]
    right_orient=Slerp([0.,1.],Rotation.from_matrix([R[74,1],R[116,1]]))
    for f in range(91,116):
        u=(f-90)/26;w=smooth(u);target[f,1]=(1-w)*right_anchor+w*right_end;target[f,1,2]+=.22*np.sin(np.pi*u)**2;target_R[f,1]=right_orient([w]).as_matrix()[0]
    target[116:140,1]=right_end;target_R[116:140,1]=R[117,1]
    for f in range(75,130):
        for s in range(2):
            if on[f,s]and f>=118:
                e=np.flatnonzero(~on[:f+1,s]);st=int(e[-1]+1)if len(e)else 0;target[f,s]=np.asarray(suf['foot_target_ankle_world'])[st,s]
    root=q[:,:3].copy();knots=np.asarray([74,76,80,85,90,96,108,116,120,125,130]);points=np.asarray([qpre[74,:3],[.49,-.26,.86],[.52,-.26,.87],[.45,-.12,.86],[.31,.15,.87],[.28,.20,.90],[.27,.24,.90],[.25,.30,.88],q0[120,:3],q0[125,:3],q0[130,:3]])
    root[75:130]=PchipInterpolator(knots,points)(np.arange(75,130));rootquat=q[:,3:7].copy();yaw0=Rotation.from_quat(np.roll(qpre[74,3:7],-1)).as_euler('ZYX')[0];end_yaw,end_pitch,end_roll=Rotation.from_quat(np.roll(q0[130,3:7],-1)).as_euler('ZYX');end_yaw=yaw0+np.arctan2(np.sin(end_yaw-yaw0),np.cos(end_yaw-yaw0))
    _,seat_pitch,seat_roll=Rotation.from_quat(np.roll(q0[120,3:7],-1)).as_euler('ZYX')
    for f in range(75,130):
        yaw=yaw0+(end_yaw-yaw0)*smooth((f-74)/56)
        if f<=120:
            u=np.clip((f-116)/4,0,1);w=smooth(u);pitch=w*seat_pitch;roll=w*seat_roll
        else:
            w=smooth((f-120)/10);pitch=(1-w)*seat_pitch+w*end_pitch;roll=(1-w)*seat_roll+w*end_roll
        rootquat[f]=np.roll(Rotation.from_euler('ZYX',[yaw,pitch,roll]).as_quat(),1)
    robot=[g for g in range(m.ngeom)if m.geom_bodyid[g]>0 and m.geom_contype[g]];ri={g:i for i,g in enumerate(robot)};support={g for g in range(m.ngeom)if m.geom_bodyid[g]==0 and m.geom_contype[g]};foot_geoms=[[g for g in robot if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('collision_foot_'+s+'_')]for s in('l','r')]
    footprint=[]
    for s in range(2):
        d.qpos[:]=q[74];mj.mj_kinematics(m,d);points=[];ori=d.xmat[feet[s]].reshape(3,3)
        for g in foot_geoms[s]:
            center=d.geom_xpos[g];axis=d.geom_xmat[g].reshape(3,3)[:,2];radius=m.geom_size[g,0];ends=[center]if m.geom_type[g]==mj.mjtGeom.mjGEOM_SPHERE else[center-axis*m.geom_size[g,1],center+axis*m.geom_size[g,1]]
            for c in ends:
                th=np.arange(16)*2*np.pi/16;points.extend((c-d.xpos[feet[s]])@ori+radius*np.c_[np.cos(th),np.sin(th),np.zeros(16)])
        footprint.append(np.asarray(points))
    n=15;H=np.zeros((m.nv,n));H[:3,:3]=np.eye(3);H[va[leg],np.arange(3,n)]=1;T=np.c_[np.eye(3),np.zeros((3,12))];P=np.c_[np.zeros((12,3)),np.eye(12)];previous=q[74,qa[leg]].copy();previous_root=q[74,:3].copy();proxy=KneeLoadProxy(m);rows=[]
    def skew(v):return np.asarray([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    for f in range(75,130):
        fixed=q[f].copy();fixed[3:7]=rootquat[f];seed=np.r_[root[f],previous];lower=np.r_[root[f]-[.35,.35,.35],m.jnt_range[jids[leg],0]+1e-6];upperbound=np.r_[root[f]+[.35,.35,.30],m.jnt_range[jids[leg],1]-1e-6];step_limit=.05 if f<116 else .07;lower[:3]=np.maximum(lower[:3],previous_root-step_limit);upperbound[:3]=np.minimum(upperbound[:3],previous_root+step_limit);lower[2]=max(.4,lower[2]);cache={};carry=previous_root+root[f]-root[f-1]-(previous_root-root[f-1])/max(1,130-f)
        cloud=[target[f,s,:2]+(footprint[s]@target_R[f,s].T)[:,:2]for s in range(2)if on[f,s]];A,b=halfspaces(convex_hull(np.concatenate(cloud)))
        def evaluate(x):
            if 'x'in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
            d.qpos[:]=fixed;d.qpos[:3]=x[:3];d.qpos[qa[leg]]=x[3:];mj.mj_kinematics(m,d);mj.mj_comPos(m,d);rr=[];JJ=[]
            def add(v,j,w):rr.extend(np.asarray(v).ravel()*w);JJ.extend(np.asarray(j).reshape(-1,n)*w)
            def point(body,pos):
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,pos,body);return jp@H,jr@H
            add(x[:3]-root[f],T,3.);add(x[:3]-carry,T,5.);add(x[3:]-q0[f,qa[leg]],P,.02);add(x[3:]-previous,P,.05)
            for s in range(2):
                body=int(feet[s]);jp,jr=point(body,d.xpos[body]);add(d.xpos[body]-target[f,s],jp,400. if on[f,s]else 40.);ori=d.xmat[body].reshape(3,3)
                for axis in(0,2):add(ori[:,axis]-target_R[f,s,:,axis],-skew(ori[:,axis])@jr,8. if on[f,s]else .5)
            jc=np.zeros((3,m.nv));mj.mj_jacSubtreeCom(m,d,jc,base);v=b+.01-A@d.subtree_com[base,:2];cj=-A@(jc@H)[:2];cj[v<=0]=0.;add(np.maximum(v,0.),cj,4. if 77<=f<85 else 12. if f<116 else 1.)
            kr,kj,_=proxy.single_stance_cost(d,on[f],seated=False,velocity_map=H,ratio_threshold=.8);add(kr,kj,1. if f<116 else 6.)
            mj.mj_collision(m,d);cr=np.zeros(len(robot));cj=np.zeros((len(robot),n))
            for c in d.contact[:d.ncon]:
                g1,g2=int(c.geom1),int(c.geom2)
                if g1 in ri and g2 in support:g=g1;normal=-np.asarray(c.frame[:3])
                elif g2 in ri and g1 in support:g=g2;normal=np.asarray(c.frame[:3])
                else:continue
                k=ri[g]
                if c.dist<-.001 and c.dist+.001<cr[k]:cr[k]=c.dist+.001;cj[k]=normal@point(int(m.geom_bodyid[g]),np.asarray(c.pos))[0]
            add(cr,cj,200.);cache.update(x=x.copy(),r=np.asarray(rr),J=np.asarray(JJ));return cache['r'],cache['J']
        fit=least_squares(lambda x:evaluate(x)[0],np.clip(seed,lower+1e-7,upperbound-1e-7),jac=lambda x:evaluate(x)[1],bounds=(lower,upperbound),max_nfev=a.max_nfev,ftol=1e-8,gtol=1e-8,xtol=1e-8)
        evaluate(fit.x);q[f]=d.qpos.copy();previous=fit.x[3:].copy();previous_root=fit.x[:3].copy();rows.append({'frame':f,'cost':float(fit.cost),'nfev':int(fit.nfev),'ankle_target_error_m':np.linalg.norm(d.xpos[feet]-target[f],axis=1).tolist(),'com_support_margin_violation_m':float(np.maximum(b+.01-A@d.subtree_com[base,:2],0).max())})
        if f%5==0:print(json.dumps(rows[-1]),flush=True)
    signs=[]
    for f in range(1,350):
        if np.dot(q[f-1,3:7],q[f,3:7])<0:q[f,3:7]*=-1;signs.append(f)
    actual=[];com=[];pen=[];joint=[]
    for row in q:
        d.qpos[:]=row;mj.mj_fwdPosition(m,d);actual.append(d.xpos[feet].copy());com.append(d.subtree_com[base].copy());pen.append(max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.]));v=np.maximum(np.maximum(m.jnt_range[jids,0]-row[qa],row[qa]-m.jnt_range[jids,1]),0);v[~m.jnt_limited[jids].astype(bool)]=0;joint.append(v)
    actual=np.asarray(actual);com=np.asarray(com);pen=np.asarray(pen);joint=np.asarray(joint);euler=Rotation.from_quat(np.roll(q[:,3:7],-1,axis=1)).as_euler('ZYX');root_step=np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1);stance=[]
    for s in range(2):
        edges=np.diff(np.r_[0,on[:,s].astype(int),0])
        for st,en in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if en<=75 or st>=130:continue
            use=np.arange(max(75,st),min(130,en));stance.append({'side':['left','right'][s],'first_frame':int(use[0]),'last_frame':int(use[-1]),'max_anchor_drift_m':float(np.linalg.norm(actual[use,s]-actual[use[0],s],axis=1).max())})
    out={k:v for k,v in suf.items()if k.startswith('source_')or k in('frame_ids','time_s','dof_names','wrist_reference_joint_names','wrist_reference_joint_targets_rad')};out.update(qpos=q,foot_support_expected=on,foot_support_confidence=conf,foot_support_geom_name=surface,foot_actual_ankle_world=actual,foot_target_ankle_world=target,reference_com_world=com,root_target_world=root,static_penetration_m=pen,static_joint_limit_violation_rad=joint,static_geometry_passed=np.asarray(pen.max()<=.02 and joint.max()<=.05),physical_joint_limit_model_sha256=np.asarray(sha(a.model)),prepare_step_version=np.asarray('v21_real_left_prepare_swing'),parent_prefix_qpos=qpre,parent_suffix_qpos=q0)
    report={'version':'v21_real_left_prepare_swing','physics_passed':False,'model_parameters_modified':False,'source_frames_preserved':350,'source_fields_preserved':True,'prefix_root_body22_frames0_through74_preserved':bool(np.array_equal(q[:75,:3],qpre[:75,:3])and np.array_equal(q[:75,qa[body22]],qpre[:75,qa[body22]])),'hands_aux24_and_wrists6':'Fixed to relaxed suffix first-frame targets across350frames','prepared_swing_frames':[77,84],'left_landing_frame':85,'left_start_world_m':left_start.tolist(),'left_end_world_m':left_end.tolist(),'new_left_prepare_step_horizontal_m':float(np.linalg.norm(left_end[:2]-left_start[:2])),'left_support_anchor_shift_during_stance_avoided':True,'root_rp_max_abs_frames75_through115_rad':float(abs(euler[75:116,1:]).max()),'root_step_max_frames74_through130_m':float(root_step[74:130].max()),'root_step_worst_frame':int(75+root_step[74:130].argmax()),'stance_intervals_touched':stance,'ankle_target_error_max_m':max(max(v['ankle_target_error_m'])for v in rows),'root_suffix_offset_max_m':float(np.linalg.norm(q[:,:3]-q0[:,:3],axis=1).max()),'ankle_suffix_offset_max_m':float(np.linalg.norm(actual-np.asarray(suf['foot_actual_ankle_world']),axis=2).max()),'penetration_max_m':float(pen.max()),'joint_limit_violation_max_rad':float(joint.max()),'static_geometry_passed':bool(pen.max()<=.02 and joint.max()<=.05),'quaternion_sign_corrections':signs,'quaternion_adjacent_dot_min':float(np.sum(q[:-1,3:7]*q[1:,3:7],axis=1).min()),'input_prefix_sha256':sha(a.prefix),'input_suffix_sha256':sha(a.suffix),'model_sha256':sha(a.model),'generator_sha256':sha(__file__),'limitations':['Adds an explicit robotic preparatory step within existing350source frames; original human source data remains unchanged.','No runtime root writes, force assistance, physics-model changes or dynamic success claim.','Offline stance geometry and joint limits require independent contact-force/effort and full closed-loop validation.'],'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**out);a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items()if k!='rows'}),flush=True)


if __name__=='__main__':main()
