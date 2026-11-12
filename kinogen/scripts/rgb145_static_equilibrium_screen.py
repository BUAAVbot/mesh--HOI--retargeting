"""Per-pose static actuator/contact equilibrium necessary-condition screen.

No integration and no controller changes. Passing this point-force LP does not
establish actual supporting loads, dynamic feasibility or policy tracking.
"""
import argparse,hashlib,json,sys
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--support',type=Path,default=Path('outputs/rgb145_pipeline/support_geometry.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--relaxed-gap-m',type=float,default=.002)
    p.add_argument('--friction-facets',type=int,default=8);p.add_argument('--planned-stance-only',action='store_true')
    p.add_argument('--exclude-joint-static-friction',action='store_true');a=p.parse_args()
    if sys.platform!='linux':p.error('Run model calculations only on the authorized server')
    if a.output.exists():raise FileExistsError(a.output)
    if a.relaxed_gap_m<0 or a.friction_facets<4:raise ValueError('Invalid relaxed gap or friction approximation')
    import numpy as np,mujoco as mj
    from scipy.optimize import linprog
    from rgb145_support import inside_convex_polygon
    from rgb145_reference_contact_audit import convex_hull,hull_measure,pack_ragged
    z=dict(np.load(a.reference,allow_pickle=False));q=np.asarray(z['qpos']);n=len(q)
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m)
    if (m.nq,m.nv,m.nu)!=(59,58,52)or q.shape!=(n,59):raise ValueError('Expected single Vita 59/58/52')
    names=list(map(str,z['dof_names']));jids=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,name)for name in names]);qa=m.jnt_qposadr[jids]
    root=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'vita_demo_free_root');rootqa=int(m.jnt_qposadr[root]);rootva=int(m.jnt_dofadr[root]);base=int(m.jnt_bodyid[root])
    if np.any(m.dof_frictionloss[rootva:rootva+6]):raise ValueError('Unexpected artificial friction on the free root')
    powered=np.flatnonzero(np.any(m.actuator_gear!=0,axis=1)&(m.actuator_gainprm[:,0]!=0));nt=len(powered)
    if nt!=42 or np.any(~m.actuator_forcelimited[powered].astype(bool)):raise ValueError('Expected42 powered actuators with finite original force limits')
    if np.any(m.actuator_gaintype[powered])or np.any(m.actuator_biastype[powered]!=1):raise ValueError('Expected fixed-gain affine position actuators')
    if np.any(m.actuator_trntype[powered]!=int(mj.mjtTrn.mjTRN_JOINT)):raise ValueError('Expected real joint transmissions')
    B=np.zeros((m.nv,nt));B[m.jnt_dofadr[m.actuator_trnid[powered,0]],np.arange(nt)]=m.actuator_gear[powered,0]
    limits=m.actuator_forcerange[powered].copy();capacity=np.max(abs(limits),axis=1)
    if np.any(capacity<=0):raise ValueError('Invalid motor force limits')
    friction_dofs=np.flatnonzero(m.dof_frictionloss>0)if not a.exclude_joint_static_friction else np.array([],int)
    Sf=np.eye(m.nv)[:,friction_dofs];nf=len(friction_dofs)
    gnames=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or str(g)for g in range(m.ngeom)]
    footgeom=[set(g for g,name in enumerate(gnames)if name.startswith('collision_foot_'+s+'_'))for s in ('l','r')]
    if any(not g for g in footgeom):raise ValueError('Missing real foot collision capsules')
    spec=json.loads(a.support.read_text());patch={s['name']:s for s in spec['supports']};ground_name=spec['ground']['name']
    ground=np.asarray(spec['ground']['z_eq_ax_by_c']);roi=np.asarray(spec['collision_roi_xy_m'])
    ground_poly=np.array([[roi[0,0],roi[0,1]],[roi[1,0],roi[0,1]],[roi[1,0],roi[1,1]],[roi[0,0],roi[1,1]]])
    support_ids={g for g,name in enumerate(gnames)if name in patch or name==ground_name}
    expected=np.asarray(z['foot_support_expected'],bool);pnames=np.asarray(z['foot_support_geom_name']).astype(str)
    pids=np.array([[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,name)for name in row]for row in pnames])
    if expected.shape!=(n,2)or np.any(pids<0):raise ValueError('Invalid planned supports')
    def surface(name):
        beta=ground if name==ground_name else np.asarray(patch[name]['top_z_eq_ax_by_c'])
        polygon=ground_poly if name==ground_name else np.asarray(patch[name]['xy_polygon_m'])
        normal=np.r_[-beta[:2],1.];length=np.linalg.norm(normal)
        return normal/length,float(beta[2]/length),polygon
    def jac(body,pos):
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,pos,body);return jp
    def pair_mu(g1,g2):
        for i in range(m.npair):
            if {int(m.pair_geom1[i]),int(m.pair_geom2[i])}=={g1,g2}:return np.asarray(m.pair_friction[i,:2]).copy()
        if m.geom_priority[g1]>m.geom_priority[g2]:mu=float(m.geom_friction[g1,0])
        elif m.geom_priority[g2]>m.geom_priority[g1]:mu=float(m.geom_friction[g2,0])
        else:mu=float(max(m.geom_friction[g1,0],m.geom_friction[g2,0]))
        return np.array([mu,mu])
    def mimic_jacobian():
        rows=[];pose_error=[]
        for eq in range(m.neq):
            if not d.eq_active[eq]:continue
            if m.eq_type[eq]!=int(mj.mjtEq.mjEQ_JOINT):raise ValueError('Only original joint mimic equalities allowed')
            j1,j2=int(m.eq_obj1id[eq]),int(m.eq_obj2id[eq]);x=float(d.qpos[m.jnt_qposadr[j2]]-m.qpos0[m.jnt_qposadr[j2]])if j2>=0 else 0.
            coef=m.eq_data[eq,:5];poly=sum(coef[k]*x**k for k in range(5));der=sum(k*coef[k]*x**(k-1)for k in range(1,5))
            row=np.zeros(m.nv);row[m.jnt_dofadr[j1]]=1.
            if j2>=0:row[m.jnt_dofadr[j2]]=-der
            rows.append(row);pose_error.append(float(d.qpos[m.jnt_qposadr[j1]]-m.qpos0[m.jnt_qposadr[j1]]-poly))
        return np.asarray(rows).reshape(-1,m.nv),max(abs(x)for x in pose_error)if pose_error else 0.
    def force_bounds():
        gain=m.actuator_gainprm[powered,0];bias=m.actuator_biasprm[powered,0]+m.actuator_biasprm[powered,1]*d.actuator_length[powered]
        rawlo=np.full(nt,-np.inf);rawhi=np.full(nt,np.inf)
        for i,aid in enumerate(powered):
            if m.actuator_ctrllimited[aid]:
                ends=gain[i]*m.actuator_ctrlrange[aid]+bias[i];rawlo[i],rawhi[i]=min(ends),max(ends)
        return np.clip(rawlo,limits[:,0],limits[:,1]),np.clip(rawhi,limits[:,0],limits[:,1]),rawlo,rawhi
    def solve(contacts,Je,rhs,physical_bounds,remove_effort=False):
        nc=len(contacts);ne=len(Je);lstart=nt;estart=lstart+3*nc;fstart=estart+ne;rho=fstart+nf;nx=rho+1
        A=np.zeros((m.nv,nx));A[:,:nt]=B
        for k,c in enumerate(contacts):A[:,lstart+3*k:lstart+3*k+3]=c['J'].T@c['W']
        A[:,estart:fstart]=Je.T;A[:,fstart:rho]=Sf
        objective=np.zeros(nx);objective[rho]=1.;G=[];h=[]
        tau_lo,tau_hi,rawlo,rawhi=physical_bounds
        bounds=[(float(lo),float(hi))for lo,hi in zip(rawlo if remove_effort else tau_lo,rawhi if remove_effort else tau_hi)]
        for k,c in enumerate(contacts):
            bounds.extend([(0.,None),(None,None)if c['mu'][0]>1e-12 else(0.,0.),(None,None)if c['mu'][1]>1e-12 else(0.,0.)]);objective[lstart+3*k]=1e-8
            for theta in np.arange(a.friction_facets)*2*np.pi/a.friction_facets:
                row=np.zeros(nx);row[lstart+3*k]=-1.
                if c['mu'][0]>1e-12:row[lstart+3*k+1]=np.cos(theta)/c['mu'][0]
                if c['mu'][1]>1e-12:row[lstart+3*k+2]=np.sin(theta)/c['mu'][1]
                G.append(row);h.append(0.)
        bounds.extend([(None,None)]*ne)
        bounds.extend([(-float(m.dof_frictionloss[v]),float(m.dof_frictionloss[v]))for v in friction_dofs]);bounds.append((0.,None if remove_effort else 1.))
        for i in range(nt):
            row=np.zeros(nx);row[i]=1.;row[rho]=-capacity[i];G.append(row);h.append(0.)
            row=np.zeros(nx);row[i]=-1.;row[rho]=-capacity[i];G.append(row);h.append(0.)
        row_scale=np.maximum(np.linalg.norm(A,axis=1),1.)
        lp=linprog(objective,A_ub=np.asarray(G),b_ub=np.asarray(h),A_eq=A/row_scale[:,None],b_eq=rhs/row_scale,bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-8,'dual_feasibility_tolerance':1e-8})
        if not lp.success:return {'status':int(lp.status),'message':lp.message,'success':False}
        residual=A@lp.x-rhs;tau=lp.x[:nt]
        cone_ratio=[]
        for k,c in enumerate(contacts):
            lam=lp.x[lstart+3*k:lstart+3*k+3]
            cone_ratio.append(float(np.linalg.norm(lam[1:]/np.maximum(c['mu'],1e-15))/lam[0])if lam[0]>1e-8 else 0.)
        return {'status':int(lp.status),'success':True,'tau':tau,'peak_ratio':float(np.max(abs(tau)/capacity)),
            'root_residual':residual[rootva:rootva+6],'max_residual':float(np.max(abs(residual))),
            'max_circular_cone_ratio':max(cone_ratio)if cone_ratio else 0.,
            'contact_force_world':np.asarray([c['W']@lp.x[lstart+3*k:lstart+3*k+3]for k,c in enumerate(contacts)]).reshape(-1,3)}
    modes=('strict_actual_contacts','relaxed_near_contact_candidates');shape=(n,2)
    feasible=np.zeros(shape,bool);status=np.full(shape,-99,int);ratios=np.full(shape,np.nan);counter_ratio=np.full(shape,np.nan)
    root_res=np.full(shape+(6,),np.nan);residual=np.full(shape,np.nan);count=np.zeros(shape,int);unplanned_count=np.zeros(shape,int);cone=np.full(shape,np.nan)
    torques=np.full(shape+(nt,),np.nan);counter_torques=np.full_like(torques,np.nan);com=np.zeros((n,3));hull_distance=np.full(shape,np.nan);pose_eq_error=np.zeros(n);penetration=np.zeros(n)
    cpoints=[];cnormals=[];origins=[];lp_forces=[];counterfactual_forces=[]
    for f in range(n):
        d.qpos[:]=m.qpos0;d.qpos[rootqa:rootqa+7]=q[f,:7];d.qpos[qa]=q[f,7:];d.qvel[:]=0.;d.ctrl[:]=0.;mj.mj_forward(m,d);mj.mj_collision(m,d)
        rhs=d.qfrc_bias-d.qfrc_passive;Je,pose_eq_error[f]=mimic_jacobian();physical_bounds=force_bounds();com[f]=d.subtree_com[base]
        penetration[f]=max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.]);strict=[]
        for c in d.contact[:d.ncon]:
            if c.dist>1e-6:continue
            gs=(int(c.geom1),int(c.geom2));entry=None
            for k in range(2):
                if a.planned_stance_only and not expected[f,k]:continue
                if gs[0]in footgeom[k]and gs[1]==pids[f,k]:g,other,sign=gs[0],gs[1],-1.
                elif gs[1]in footgeom[k]and gs[0]==pids[f,k]:g,other,sign=gs[1],gs[0],1.
                else:continue
                normal,_,polygon=surface(pnames[f,k])
                if sign*np.asarray(c.frame[:3])@normal<.7 or not inside_convex_polygon(c.pos[:2],polygon):continue
                entry={'g':g,'support':other,'body':int(m.geom_bodyid[g]),'point':np.asarray(c.pos).copy(),'W':sign*np.asarray(c.frame).reshape(3,3).T,'mu':np.asarray(c.friction[:2]).copy(),'origin':0,'unplanned':not bool(expected[f,k])}
            if entry is None:
                for robotidx in (0,1):
                    g,other=gs[robotidx],gs[1-robotidx];bname=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or''
                    if other not in support_ids or not gnames[other].startswith('seat_support_')or not('hip_'in bname or bname=='base_link'):continue
                    normal,_,polygon=surface(gnames[other]);sign=1. if robotidx==1 else-1.
                    if sign*np.asarray(c.frame[:3])@normal<.7 or not inside_convex_polygon(c.pos[:2],polygon):continue
                    entry={'g':g,'support':other,'body':int(m.geom_bodyid[g]),'point':np.asarray(c.pos).copy(),'W':sign*np.asarray(c.frame).reshape(3,3).T,'mu':np.asarray(c.friction[:2]).copy(),'origin':0,'unplanned':False}
            if entry is not None:entry['J']=jac(entry['body'],entry['point']);strict.append(entry)
        relaxed=list(strict)
        for k in range(2):
            if a.planned_stance_only and not expected[f,k]:continue
            normal,offset,polygon=surface(pnames[f,k]);t1=np.array([1.,0.,0.])-normal[0]*normal;t1/=np.linalg.norm(t1);W=np.column_stack([normal,t1,np.cross(normal,t1)])
            for g in sorted(footgeom[k]):
                center=d.geom_xpos[g];axis=d.geom_xmat[g].reshape(3,3)[:,2];radius=float(m.geom_size[g,0])
                if m.geom_type[g]==int(mj.mjtGeom.mjGEOM_CAPSULE):ends=center+np.array([-1.,1.])[:,None]*float(m.geom_size[g,1])*axis
                elif m.geom_type[g]==int(mj.mjtGeom.mjGEOM_SPHERE):ends=np.array([center])
                else:raise ValueError('Expected capsule/sphere sole geometry')
                for point in ends-radius*normal:
                    gap=float(point@normal-offset)
                    if abs(gap)>a.relaxed_gap_m or not inside_convex_polygon(point[:2],polygon):continue
                    if any(c['g']==g and c['support']==pids[f,k]and np.linalg.norm(point-c['point'])<1e-6 for c in relaxed):continue
                    body=int(m.geom_bodyid[g]);relaxed.append({'g':g,'support':int(pids[f,k]),'body':body,'point':point.copy(),'W':W,'mu':pair_mu(g,int(pids[f,k])),'J':jac(body,point),'origin':1,'unplanned':not bool(expected[f,k])})
        for mode,contacts in enumerate((strict,relaxed)):
            count[f,mode]=len(contacts);unplanned_count[f,mode]=sum(c['unplanned']for c in contacts)
            pts=np.asarray([c['point']for c in contacts]).reshape(-1,3);cpoints.append(pts);cnormals.append(np.asarray([c['W'][:,0]for c in contacts]).reshape(-1,3));origins.append(np.asarray([c['origin']for c in contacts],int)[:,None])
            _,hull_distance[f,mode],_=hull_measure(convex_hull(pts[:,:2]),com[f,:2])
            result=solve(contacts,Je,rhs,physical_bounds);status[f,mode]=result['status'];feasible[f,mode]=result['success']
            if result['success']:
                ratios[f,mode]=result['peak_ratio'];root_res[f,mode]=result['root_residual'];residual[f,mode]=result['max_residual'];torques[f,mode]=result['tau'];cone[f,mode]=result['max_circular_cone_ratio']
                lp_forces.append(result['contact_force_world']);counterfactual_forces.append(np.full((len(contacts),3),np.nan))
            else:
                diagnostic=solve(contacts,Je,rhs,physical_bounds,remove_effort=True)
                if diagnostic['success']:counter_ratio[f,mode]=diagnostic['peak_ratio'];counter_torques[f,mode]=diagnostic['tau']
                lp_forces.append(np.full((len(contacts),3),np.nan));counterfactual_forces.append(diagnostic['contact_force_world']if diagnostic['success']else np.full((len(contacts),3),np.nan))
        if f%50==0:print(json.dumps({'frame':f,'strict_feasible':bool(feasible[f,0]),'relaxed_feasible':bool(feasible[f,1])}),flush=True)
    phase_intervals=[('initial_turn_approach',0.,3.3),('first_43cm_step',3.3,4.3),('upper_58cm_step',4.3,5.2),('turn_on_top',5.2,6.2),('descend',6.2,8.7),('sit',8.7,12.)]
    summary={};worst=[]
    for k,mode in enumerate(modes):
        good=feasible[:,k];bad=~good;counter=np.isfinite(counter_ratio[:,k])
        summary[mode]={'feasible_frames':int(good.sum()),'infeasible_or_solver_failed_frames':int(bad.sum()),'frames':n,
            'infeasible_but_finite_effort_relaxation_solution':int(np.count_nonzero(bad&counter)),
            'still_infeasible_after_removing_effort_caps':int(np.count_nonzero(bad&~counter)),
            'max_minimized_peak_original_effort_ratio':float(np.max(ratios[good,k]))if good.any()else None,
            'max_root_equilibrium_residual':float(np.nanmax(abs(root_res[:,k])))if good.any()else None,
            'max_original_equation_residual':float(np.nanmax(residual[:,k]))if good.any()else None,
            'max_circular_friction_cone_ratio_of_outer_relaxation_solution':float(np.nanmax(cone[:,k]))if good.any()else None,
            'phase_groups':{name:{'frames':int(mask.sum()),'feasible':int(np.count_nonzero(mask&good)),'infeasible':int(np.count_nonzero(mask&bad))}for name,lo,hi in phase_intervals for mask in[(z['source_time_s']>=lo)&(z['source_time_s']<hi)]}}
        ranked=sorted(range(n),key=lambda f:(not good[f],float(counter_ratio[f,k])if counter[f]else 1000. if bad[f]else float(ratios[f,k])),reverse=True)[:20]
        for f in ranked:
            tau=counter_torques[f,k]if counter[f]and bad[f]else torques[f,k];j=int(np.nanargmax(abs(tau)/capacity))if np.isfinite(tau).any()else None
            worst.append({'mode':mode,'source_frame_id':int(z['frame_ids'][f]),'source_time_s':float(z['source_time_s'][f]),'feasible_with_original_limits':bool(good[f]),'contact_points':int(count[f,k]),'min_peak_effort_ratio':float(ratios[f,k])if good[f]else None,'counterfactual_required_peak_original_effort_ratio':float(counter_ratio[f,k])if counter[f]else None,'peak_actuator':mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,int(powered[j]))if j is not None else None,'geometric_penetration_mm':float(1000*penetration[f])})
    points,offsets=pack_ragged(cpoints,3);normals,_=pack_ragged(cnormals,3);origin,_=pack_ragged(origins,1);forces,_=pack_ragged(lp_forces,3);counterforces,_=pack_ragged(counterfactual_forces,3)
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'arrays.npz',feasible=feasible,lp_status=status,min_peak_original_effort_ratio=ratios,counterfactual_required_peak_effort_ratio=counter_ratio,
        actuator_force=torques,counterfactual_actuator_force=counter_torques,root_equilibrium_residual=root_res,max_equation_residual=residual,contact_count=count,unplanned_foot_contact_count=unplanned_count,
        contact_points_world=points,contact_normals_world=normals,contact_origin=origin,contact_offsets=offsets,lp_predicted_contact_force_world_N=forces,counterfactual_predicted_contact_force_world_N=counterforces,modes=np.asarray(modes),contact_hull_com_outside_distance_xy_m=hull_distance,circular_friction_cone_ratio_max=cone,
        max_geometric_penetration_m=penetration,mimic_pose_residual_max=pose_eq_error,com_world=com,source_time_s=z['source_time_s'],execution_time_s=z['time_s'],source_frame_ids=z['frame_ids'],
        powered_actuator_names=np.asarray([mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,int(i))for i in powered]),original_actuator_force_limits=limits,joint_static_friction_dof_ids=friction_dofs)
    report={'schema':'rgb145.static_equilibrium_necessary_screen.v1','simulation_passed':False,'mode':'per_pose_v0_ddq0_no_mj_step','model_sha256':sha(a.model),'reference_sha256':sha(a.reference),'support_sha256':sha(a.support),
        'summary':summary,'top_worst_pose_modes':worst,'relaxed_gap_m':a.relaxed_gap_m,'friction_approximation':str(a.friction_facets)+'-facet outer approximation of the circular/elliptic cone',
        'planned_stance_only':a.planned_stance_only,'joint_static_friction_included':not a.exclude_joint_static_friction,
        'max_mimic_pose_residual':float(pose_eq_error.max()),'model_parameters_modified':False,'actuated_free_root_dofs':0,
        'array_layout_notes':{'contact_offsets':'frame-major then strict/relaxed, length2*T+1','contact_origin':'0 actual strict geometric contact,1 near-plane real capsule endpoint candidate'},
        'limitations':['This is a static point-force equilibrium screen, not dynamic feasibility, actual load measurement, a continuous rollout or policy success.',
            'The relaxed mode explicitly permits candidate support at true sole points within plus/minus2mm of the correct surface; it is not actual contact or proof that settling preserves every constraint.',
            'Outer friction cones can return a false positive versus the exact circular cone; inspect reported circular-cone ratios. Spinning/rolling contact torques are omitted, so an infeasible result is relative to this point-force model.',
            'Original actuator ctrl ranges and real effort caps constrain the primary LP. Effort-unbounded LPs are counterfactual diagnostics only and never physical success.',
            'Bounded internal joint static friction uses the unchanged model.dof_frictionloss; root friction must be zero.',
            'Nonzero mimic position residuals or large geometric penetrations make the pose itself suspect even if force balance has a solution.']}
    (a.output/'screen.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'summary':summary,'top_worst_pose_modes':worst},allow_nan=False),flush=True)


if __name__=='__main__':main()
