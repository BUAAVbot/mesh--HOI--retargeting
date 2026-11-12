"""Bounded offline root/leg transition; hard endpoint-distance audit included.

Keeps source poses0..115 and141..349 unchanged. A30mm source-frame root
bound is mathematically impossible for those endpoints; use a declared bound.
"""
from pathlib import Path
import argparse,hashlib,json,sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--max-root-step-m',type=float,default=.06);p.add_argument('--max-iterations',type=int,default=150);p.add_argument('--hard-knee-load',action='store_true');a=p.parse_args()
    if sys.platform!='linux':p.error('Offline FK/IK runs on the server')
    if a.output.exists()or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    import numpy as np,mujoco as mj
    from scipy.optimize import minimize
    from rgb145_contact_plan import convex_hull,halfspaces
    from rgb145_knee_load import KneeLoadProxy
    z=dict(np.load(a.reference,allow_pickle=False));q0=np.asarray(z['qpos'],float);q=q0.copy();on=np.asarray(z['foot_support_expected'],bool);names=list(map(str,z['dof_names']));m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m)
    jids=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]);qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids];leg=np.asarray([i for i,n in enumerate(names)if n.startswith(('left_','right_'))and any(s in n for s in('hip_','knee_','ankle_'))]);feet=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in('left','right')]);base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'base_link')
    first,last=115,141;delta=(q0[last,:3]-q0[first,:3])/(last-first);distance=float(np.linalg.norm(q0[last,:3]-q0[first,:3]));lowerbound=distance/(last-first)
    if a.max_root_step_m<lowerbound-1e-10:raise ValueError(json.dumps({'reason':'Fixed endpoint distance exceeds available source-frame displacement budget','endpoint_distance_m':distance,'steps':last-first,'max_root_step_m_requested':a.max_root_step_m,'minimum_possible_max_step_m':lowerbound}))
    target=np.asarray(z['foot_target_ankle_world']).copy();target_R=[]
    for row in q0:
        d.qpos[:]=row;mj.mj_fwdPosition(m,d);target_R.append(d.xmat[feet].reshape(2,3,3).copy())
    target_R=np.asarray(target_R)
    for s in range(2):
        edges=np.diff(np.r_[0,on[:,s].astype(int),0])
        for st,en in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            use=np.arange(max(116,st),min(141,en))
            if len(use):target[use,s]=np.asarray(z['foot_target_ankle_world'])[st,s];target_R[use,s]=target_R[min(st+2,en-1),s]
    robot=[g for g in range(m.ngeom)if m.geom_bodyid[g]>0 and m.geom_contype[g]];ri={g:i for i,g in enumerate(robot)};support={g for g in range(m.ngeom)if m.geom_bodyid[g]==0 and m.geom_contype[g]};foot_geoms=[[g for g in robot if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('collision_foot_'+s+'_')]for s in('l','r')];footprint=[]
    for s in range(2):
        d.qpos[:]=q0[first];mj.mj_kinematics(m,d);points=[];ori=d.xmat[feet[s]].reshape(3,3)
        for g in foot_geoms[s]:
            center=d.geom_xpos[g];axis=d.geom_xmat[g].reshape(3,3)[:,2];radius=m.geom_size[g,0];ends=[center]if m.geom_type[g]==mj.mjtGeom.mjGEOM_SPHERE else[center-axis*m.geom_size[g,1],center+axis*m.geom_size[g,1]]
            for c in ends:
                th=np.arange(16)*2*np.pi/16;points.extend((c-d.xpos[feet[s]])@ori+radius*np.c_[np.cos(th),np.sin(th),np.zeros(16)])
        footprint.append(np.asarray(points))
    n=15;H=np.zeros((m.nv,n));H[:3,:3]=np.eye(3);H[va[leg],np.arange(3,n)]=1;T=np.c_[np.eye(3),np.zeros((3,12))];P=np.c_[np.zeros((12,3)),np.eye(12)];proxy=KneeLoadProxy(m);rows=[]
    def skew(v):return np.asarray([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    for f in range(116,141):
        fixed=q0[f].copy();root_target=q0[first,:3]+(f-first)*delta;previous=q[f-1].copy();seed=np.r_[previous[:3]+(q0[last,:3]-previous[:3])/(last-f+1),previous[qa[leg]]];lower=np.r_[root_target-[.3,.3,.3],m.jnt_range[jids[leg],0]+1e-6];upper=np.r_[root_target+[.3,.3,.3],m.jnt_range[jids[leg],1]-1e-6];lower[2]=max(.4,lower[2]);cache={}
        cloud=[target[f,s,:2]+(footprint[s]@target_R[f,s].T)[:,:2]for s in range(2)if on[f,s]];A,b=halfspaces(convex_hull(np.concatenate(cloud)))
        def evaluate(x):
            if 'x'in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
            d.qpos[:]=fixed;d.qpos[:3]=x[:3];d.qpos[qa[leg]]=x[3:];mj.mj_kinematics(m,d);mj.mj_comPos(m,d);rr=[];JJ=[]
            def add(v,j,w):rr.extend(np.asarray(v).ravel()*w);JJ.extend(np.asarray(j).reshape(-1,n)*w)
            def point(body,pos):
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,pos,body);return jp@H,jr@H
            add(x[:3]-root_target,T,10.);add(x[:3]-(previous[:3]+delta),T,10.);add(x[3:]-q0[f,qa[leg]],P,.025);add(x[3:]-previous[qa[leg]],P,.05)
            for s in range(2):
                body=int(feet[s]);jp,jr=point(body,d.xpos[body]);add(d.xpos[body]-target[f,s],jp,400. if on[f,s]else 20.);ori=d.xmat[body].reshape(3,3)
                for axis in(0,2):add(ori[:,axis]-target_R[f,s,:,axis],-skew(ori[:,axis])@jr,8. if on[f,s]else .3)
            jc=np.zeros((3,m.nv));mj.mj_jacSubtreeCom(m,d,jc,base);v=b-A@d.subtree_com[base,:2];cj=-A@(jc@H)[:2];cj[v<=0]=0;add(np.maximum(v,0),cj,.5)
            kr,kj,load_info=proxy.single_stance_cost(d,on[f],seated=False,velocity_map=H,ratio_threshold=.8);add(kr,kj,6.)
            mj.mj_collision(m,d);cr=np.zeros(len(robot));cj=np.zeros((len(robot),n))
            for c in d.contact[:d.ncon]:
                g1,g2=int(c.geom1),int(c.geom2)
                if g1 in ri and g2 in support:g=g1;normal=-np.asarray(c.frame[:3])
                elif g2 in ri and g1 in support:g=g2;normal=np.asarray(c.frame[:3])
                else:continue
                k=ri[g]
                if c.dist<-.001 and c.dist+.001<cr[k]:cr[k]=c.dist+.001;cj[k]=normal@point(int(m.geom_bodyid[g]),np.asarray(c.pos))[0]
            add(cr,cj,200.);cache.update(x=x.copy(),r=np.asarray(rr),J=np.asarray(JJ),load_info=load_info);return cache['r'],cache['J']
        def constraints(x):
            prev=x[:3]-previous[:3];remain=x[:3]-q0[last,:3];values=[a.max_root_step_m**2-prev@prev,((last-f)*a.max_root_step_m)**2-remain@remain]
            if a.hard_knee_load:
                evaluate(x);info=cache['load_info'];values.append(1. if info is None else .98-info['effort_ratio'])
            return np.asarray(values)
        def constraint_jac(x):
            J=np.zeros((3 if a.hard_knee_load else 2,n));J[0,:3]=-2*(x[:3]-previous[:3]);J[1,:3]=-2*(x[:3]-q0[last,:3])
            if a.hard_knee_load:
                evaluate(x);info=cache['load_info']
                if info is not None:J[2]=-info['jacobian_effort_ratio']@H
            return J
        def fun(x):r,J=evaluate(x);return .5*r@r
        def jac(x):r,J=evaluate(x);return J.T@r
        fit=minimize(fun,np.clip(seed,lower+1e-7,upper-1e-7),jac=jac,bounds=list(zip(lower,upper)),constraints=[{'type':'ineq','fun':constraints,'jac':constraint_jac}],method='SLSQP',options={'maxiter':a.max_iterations,'ftol':1e-10,'disp':False})
        evaluate(fit.x);q[f]=d.qpos.copy();rows.append({'frame':f,'solver_success':bool(fit.success),'solver_message':str(fit.message),'iterations':int(fit.nit),'cost':float(fit.fun),'root_step_m':float(np.linalg.norm(q[f,:3]-q[f-1,:3])),'root_xyz_m':q[f,:3].tolist(),'leg_position_rad':q[f,qa[leg]].tolist(),'leg_source_step_max_rad':float(np.max(abs(q[f,qa[leg]]-q[f-1,qa[leg]]))),'com_world_m':d.subtree_com[base].tolist(),'com_support_violation_m':float(np.maximum(b-A@d.subtree_com[base,:2],0).max()),'ankle_target_error_m':np.linalg.norm(d.xpos[feet]-target[f],axis=1).tolist(),'root_constraint_slack_m2':constraints(fit.x).tolist()})
        if f%5==0:print(json.dumps(rows[-1]),flush=True)
    actual=[];com=[];pen=[];joint=[];loads=[]
    for f,row in enumerate(q):
        d.qpos[:]=row;mj.mj_fwdPosition(m,d);actual.append(d.xpos[feet].copy());com.append(d.subtree_com[base].copy());pen.append(max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.]));v=np.maximum(np.maximum(m.jnt_range[jids,0]-row[qa],row[qa]-m.jnt_range[jids,1]),0);v[~m.jnt_limited[jids].astype(bool)]=0;joint.append(v)
        if 116<=f<=140:
            _,_,info=proxy.single_stance_cost(d,on[f],seated=False)
            loads.append(None if info is None else{'frame':f,'side':info['side'],'torque_Nm':float(info['joint_torque_Nm']),'effort_ratio':float(info['effort_ratio'])})
    actual=np.asarray(actual);com=np.asarray(com);pen=np.asarray(pen);joint=np.asarray(joint);steps=np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1);error=np.linalg.norm(actual-target,axis=2);valid=[v for v in loads if v is not None];stance=[]
    for s in range(2):
        edges=np.diff(np.r_[0,on[:,s].astype(int),0])
        for st,en in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if en<=116 or st>=141:continue
            use=np.arange(max(116,st),min(141,en));stance.append({'side':['left','right'][s],'first_frame':int(use[0]),'last_frame':int(use[-1]),'anchor_drift_m_max':float(np.linalg.norm(actual[use,s]-actual[use[0],s],axis=1).max())})
    assert np.array_equal(q[:116],q0[:116])and np.array_equal(q[141:],q0[141:])
    out={k:v for k,v in z.items()if k.startswith('source_')or k in('frame_ids','time_s','dof_names','foot_support_expected','foot_support_confidence','foot_support_geom_name','wrist_reference_joint_names','wrist_reference_joint_targets_rad')};out.update(qpos=q,foot_target_ankle_world=target,foot_actual_ankle_world=actual,reference_com_world=com,root_target_world=q[:,:3].copy(),static_penetration_m=pen,static_joint_limit_violation_rad=joint,static_geometry_passed=np.asarray(pen.max()<=.02 and joint.max()<=.05),physical_joint_limit_model_sha256=np.asarray(sha(a.model)),root_transition_version=np.asarray('v22_bounded_offline_root_legs'),parent_reference_qpos=q0)
    report={'version':'v22_bounded_offline_root_legs','physics_passed':False,'model_parameters_modified':False,'first116poses_exactly_preserved':True,'poses141_to349_exactly_preserved':True,'modified_frames':list(range(116,141)),'source_fields_and_clock_preserved':True,'fixed_endpoint_distance_m':distance,'steps_between_fixed_endpoints':last-first,'mathematical_minimum_possible_max_root_step_m':lowerbound,'requested_30mm_source_step_impossible':bool(distance>.03*(last-first)),'declared_max_root_step_m':a.max_root_step_m,'root_step_max_m':float(steps[115:141].max()),'root_step_bound_passed':bool(steps[115:141].max()<=a.max_root_step_m+1e-5),'root_join_141_step_m':float(steps[140]),'stance_target_error_max_m':float(error[116:141][on[116:141]].max()),'swing_target_error_max_m':float(error[116:141][~on[116:141]].max()),'stance_anchor_drift':stance,'static_knee_effort_ratio_max':max([v['effort_ratio']for v in valid]or[0.]),'static_knee_over_capacity_frames':[v['frame']for v in valid if v['effort_ratio']>1.],'static_knee_loads':loads,'penetration_max_m':float(pen.max()),'joint_limit_violation_max_rad':float(joint.max()),'static_geometry_passed':bool(pen.max()<=.02 and joint.max()<=.05),'all_solver_steps_succeeded':all(v['solver_success']for v in rows),'input_reference_sha256':sha(a.reference),'model_sha256':sha(a.model),'generator_sha256':sha(__file__),'rows':rows,'limitations':['30mm source-frame root steps are impossible within the fixed116..140window by endpoint distance alone.','Dynamic COM departure is allowed; this is no proof of closed-loop support or feasible actuator effort.','Static knee proxy uses actual unchanged120Nmcapacity; fullcontact inverse dynamics must check all joints.']}
    report['optimizer_constraints']={'hard_knee_load_enabled':a.hard_knee_load,'hard_knee_effort_ratio_max':.98 if a.hard_knee_load else None,'max_root_step_m':a.max_root_step_m,'terminal_reachability_ball':True,'max_iterations_per_pose':a.max_iterations}
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**out);a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items()if k not in('rows','static_knee_loads')}),flush=True)


if __name__=='__main__':main()
