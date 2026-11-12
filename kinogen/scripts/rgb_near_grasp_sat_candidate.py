"""Independent wrist/O6 fingertip keyframe search with final full SAT/QP audit."""
from pathlib import Path
import argparse,json,hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--frame',type=int,default=260);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--starts',type=int,default=16);p.add_argument('--seed-report',type=Path);p.add_argument('--hard-collision',action='store_true');p.add_argument('--audit-seed-only',type=int);p.add_argument('--hard-seed-trial',type=int,default=10);p.add_argument('--structural-clearance',action='store_true');p.add_argument('--wide-search',action='store_true');a=p.parse_args()
    if a.output_dir.exists():raise FileExistsError('Use a fresh output directory')
    import numpy as np,mujoco as mj
    from scipy.optimize import least_squares,minimize
    from scipy.spatial.transform import Rotation
    from types import SimpleNamespace
    from rgb_near_convex_distance import ShoeHandConvexDistance
    m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);ref=np.load(a.reference,allow_pickle=False);frame=a.frame
    q0=np.r_[ref['qpos'][frame,0],ref['qpos'][frame,1],ref['object_target_pose'][frame]];d.qpos[:]=q0;mj.mj_forward(m,d)
    body=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link');P0=d.xpos[body].copy();R0=d.xmat[body].reshape(3,3).copy()
    digits=['thumb_cmc_pitch','thumb_cmc_yaw','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch'];js=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_rh_'+n)for n in digits];qas=m.jnt_qposadr[js]
    lo=np.r_[[-.065]*3,[-.55]*3,m.jnt_range[js,0]+1e-6];hi=np.r_[[.065]*3,[.55]*3,m.jnt_range[js,1]-1e-6]
    translation_limit=.120 if a.wide_search else .085;rotation_limit=1.3 if a.wide_search else .7
    if a.wide_search:lo[:3]=-.12;hi[:3]=.12;lo[3:6]=-1.3;hi[3:6]=1.3
    couplings=[(int(m.jnt_qposadr[int(m.eq_obj1id[e])]),int(m.jnt_qposadr[int(m.eq_obj2id[e])]),m.eq_data[e,:5].copy())for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT]
    hand=[g for g in range(m.ngeom)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith('p1_rh_')and m.geom_contype[g]]
    shoes=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,n)for n in ['shoe_sole_proxy','shoe_left_wall_proxy','shoe_right_wall_proxy','shoe_toe_wall_proxy']]
    oracle=ShoeHandConvexDistance(m);shapes={g:oracle._shape(g)for g in hand+shoes};shoeR={g:d.geom_xmat[g].reshape(3,3).copy()for g in shoes};shoeP={g:d.geom_xpos[g].copy()for g in shoes}
    thumb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'p1_collision_rh_thumb_distal_0');idx=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'p1_collision_rh_index_distal_0');mid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'p1_collision_rh_middle_distal_0')
    selfpairs=[]
    for ii,g in enumerate(hand):
        b=int(m.geom_bodyid[g]);wb=int(m.body_weldid[b])
        for h in hand[ii+1:]:
            c=int(m.geom_bodyid[h]);wc=int(m.body_weldid[c])
            if wb==wc or int(m.body_weldid[int(m.body_parentid[wb])])==wc or int(m.body_weldid[int(m.body_parentid[wc])])==wb:continue
            selfpairs.append((g,h))
    shadow=SimpleNamespace(geom_xpos=d.geom_xpos.copy(),geom_xmat=d.geom_xmat.copy())
    def state(x):
        d.qpos[:]=q0;d.qpos[qas]=x[6:]
        for dep,drv,c in couplings:d.qpos[dep]=np.polynomial.polynomial.polyval(d.qpos[drv],c)
        mj.mj_kinematics(m,d);R=Rotation.from_rotvec(x[3:6]).as_matrix()@R0;P=P0+x[:3];delta=R@R0.T
        shadow.geom_xpos[:]=d.geom_xpos;shadow.geom_xmat[:]=d.geom_xmat
        for g in hand:
            shadow.geom_xpos[g]=(d.geom_xpos[g]-P0)@delta.T+P;shadow.geom_xmat[g]=(delta@d.geom_xmat[g].reshape(3,3)).ravel()
        return P,R
    def verts(g):return shapes[g]['vertices']@shadow.geom_xmat[g].reshape(3,3).T+shadow.geom_xpos[g]
    def interval_gap(v,g):
        local=(v-shoeP[g])@shoeR[g];sz=m.geom_size[g];return float(np.max(np.r_[local.min(0)-sz,-sz-local.max(0)]))
    def self_gap(g,h):
        # Do not materialize million-axis by thousand-vertex matrices. A
        # positive separating plane certifies safety immediately; overlap
        # requires checking every face and edge-cross axis in bounded batches.
        A,B=shapes[g],shapes[h];RA=shadow.geom_xmat[g].reshape(3,3);RB=shadow.geom_xmat[h].reshape(3,3)
        va,vb=verts(g),verts(h);best=-np.inf
        def scan(axes):
            nonlocal best
            for start in range(0,len(axes),256):
                nn=axes[start:start+256];ll=np.linalg.norm(nn,axis=1);nn=nn[ll>1e-9]/ll[ll>1e-9,None]
                if not len(nn):continue
                pa=nn@va.T;pb=nn@vb.T;best=max(best,float(np.max(np.maximum(pa.min(1)-pb.max(1),pb.min(1)-pa.max(1)))))
                if best>1e-7:return True
            return False
        if scan(np.r_[np.eye(3),A['normals']@RA.T,B['normals']@RB.T]):return best
        ea=A['edges']@RA.T;eb=B['edges']@RB.T
        for start in range(0,len(ea),8):
            cross=np.cross(ea[start:start+8,None,:],eb[None,:,:]).reshape(-1,3)
            if scan(cross):return best
        return best
    def validate(x,digit):
        P,R=state(x);contact=[];penetrations=[]
        bydigit=[]
        for g in [thumb,digit]:
            vals=[]
            for h in shoes:
                v=oracle.query(shadow,g,h,precise=True);vals.append({'hand_geom':m.geom(g).name,'shoe_geom':m.geom(h).name,**v})
            bydigit.append(vals)
        def normal(c):
            if c.get('qp_witness_world')and c['signed_gap_m']>1e-7:
                w=np.asarray(c['qp_witness_world']);n=w[:3]-w[3:];return n/np.linalg.norm(n)
            return None
        near=[[c for c in vals if c['precise_distance_success']and 1e-7<c['signed_gap_m']<=.003]for vals in bydigit]
        pairs=[(float(normal(t)@normal(i)),t,i)for t in near[0]for i in near[1]]
        if pairs:
            opposition,t,i=min(pairs,key=lambda z:z[0]);contact=[t,i]
        else:
            contact=[min(vals,key=lambda c:c['signed_gap_m'])for vals in bydigit];ns=[normal(c)for c in contact];opposition=float(ns[0]@ns[1])if all(n is not None for n in ns)else None
        if not all(c['precise_distance_success']and 0<c['signed_gap_m']<=.003 for c in contact)or opposition is None or opposition>=-.5:
            return{'feasible':False,'collision_audit_complete':False,'hand_position_world_m':P.tolist(),'hand_rotation_world':R.tolist(),'driver_angles_rad':x[6:].tolist(),'wrist_translation_delta_m':x[:3].tolist(),'wrist_rotation_vector_delta_rad':x[3:6].tolist(),'contact_evidence':contact,'all_fingertip_queries':bydigit,'opposition_normal_dot':opposition,'shoe_penetrations':[],'hand_self_penetrations':[],'rejection_reason':'Two opposing precise fingertip contact distances not achieved; full collision audit skipped, not passed.'}
        for g in hand:
            for h in shoes:
                sat=oracle.query(shadow,g,h,precise=False)
                if sat['signed_gap_m']<-1e-7:penetrations.append({'g1':m.geom(g).name,'g2':m.geom(h).name,'gap_m':sat['signed_gap_m']})
        own=[]
        for g,h in selfpairs:
            vg,vh=verts(g),verts(h)
            sep=np.max(np.r_[vg.min(0)-vh.max(0),vh.min(0)-vg.max(0)])
            if sep>0:continue
            gap=self_gap(g,h)
            if gap<-1e-7:own.append({'g1':m.geom(g).name,'g2':m.geom(h).name,'gap_m':gap})
        okay=not penetrations and not own and all(c['precise_distance_success']and 0<c['signed_gap_m']<=.003 for c in contact)and opposition is not None and opposition<-.5
        return{'feasible':okay,'collision_audit_complete':True,'hand_position_world_m':P.tolist(),'hand_rotation_world':R.tolist(),'driver_angles_rad':x[6:].tolist(),'wrist_translation_delta_m':x[:3].tolist(),'wrist_rotation_vector_delta_rad':x[3:6].tolist(),'contact_evidence':contact,'all_fingertip_queries':bydigit,'opposition_normal_dot':opposition,'shoe_penetrations':penetrations,'hand_self_penetrations':own}
    rng=np.random.default_rng(260);trials=[];best=None
    seeds=json.loads(a.seed_report.read_text())['trials']if a.seed_report else None
    for trial in range(a.starts):
        wall=shoes[1];digit=idx if trial%2==0 else mid;sgn=1 if(trial//2)%2==0 else-1
        # Pinch a thin shoe wall near the heel rim, not the entire14cm width.
        patch=np.asarray([(.100 if(trial//4)%2==0 else .070),0.,float(m.geom_size[wall,2]-.007)])
        y=float(m.geom_size[wall,1]+.0013)
        targets=[shoeP[wall]+shoeR[wall]@(patch+np.asarray([0,sgn*y,0])),shoeP[wall]+shoeR[wall]@(patch+np.asarray([0,-sgn*y,0]))]
        x=np.r_[np.zeros(6),[.3,.7,.7,.25,.1,.1]]
        if trial>=8:x[:6]=rng.uniform(-.025,.025,6);x[6:]=rng.uniform(lo[6:],hi[6:])
        if seeds:x=np.asarray(seeds[trial%len(seeds)]['x'])
        if a.hard_collision:
            x=np.asarray(seeds[a.hard_seed_trial]['x']).copy();digit=idx;sgn=-1
            if trial%4 in[1,3]:x[:3]-=.020*shoeR[wall][:,1]
            if trial%4 in[2,3]:x[:3]+=.020*shoeR[wall][:,2]
            if not a.structural_clearance:x[9:]=.02
            targets=[shoeP[wall]+shoeR[wall]@(patch+np.asarray([0,sgn*y,0])),shoeP[wall]+shoeR[wall]@(patch+np.asarray([0,-sgn*y,0]))]
            if a.structural_clearance:
                targets=[]
                for c in seeds[a.hard_seed_trial]['contact_evidence']:
                    w=np.asarray(c['qp_witness_world']);n=w[:3]-w[3:];n/=np.linalg.norm(n);targets.append(w[3:]+.0015*n)
                if a.wide_search:targets=[t-.035*shoeR[wall][:,0]for t in targets]
        x=np.clip(x,lo+1e-7,hi-1e-7)
        if a.audit_seed_only is not None:
            x=np.asarray(seeds[a.audit_seed_only]['x']);digit=mid if'middle'in seeds[a.audit_seed_only]['pinch_digit']else idx
        shell_axis_local=None
        palm=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'p1_collision_rh_hand_base_link_0');shell=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'p1_collision_rh_thumb_metacarpals_0')
        if a.structural_clearance:
            _,RR=state(x);vp,vt=verts(palm),verts(shell)
            axes=np.r_[shapes[palm]['normals']@shadow.geom_xmat[palm].reshape(3,3).T,shapes[shell]['normals']@shadow.geom_xmat[shell].reshape(3,3).T]
            best_gap=-np.inf;best_axis=None
            for start in range(0,len(axes),256):
                nn=axes[start:start+256];pa=nn@vp.T;pb=nn@vt.T;gg=np.maximum(pa.min(1)-pb.max(1),pb.min(1)-pa.max(1));ii=int(np.argmax(gg))
                if gg[ii]>best_gap:best_gap=float(gg[ii]);best_axis=nn[ii]
            shell_axis_local=RR.T@best_axis
        def shell_gap(R,vv):
            nn=R@shell_axis_local;pa=vv[palm]@nn;pb=vv[shell]@nn
            return max(float(pa.min()-pb.max()),float(pb.min()-pa.max()))
        def residual(z):
            state(z);vv={g:verts(g)for g in hand};out=[]
            for g,t in zip([thumb,digit],targets):
                v=vv[g];k=int(np.argmin(np.linalg.norm(v-t,axis=1)));out.extend(700*(v[k]-t))
            # Constrain entire fingertip convex body to opposite sides of
            # the thin wall. Merely touching the same edge from above is
            # not an opposing pinch, even when both distances are tiny.
            for g,sign in [(thumb,sgn),(digit,-sgn)]:
                yy=((vv[g]-shoeP[wall])@shoeR[wall])[:,1]
                signed=sign*yy
                out.append(2400*max(0,m.geom_size[wall,1]+.0007-float(signed.min())))
            for g in hand:
                for h in shoes:out.append(1800*max(0,.0007-interval_gap(vv[g],h)))
            out.append(100*max(0,float(np.linalg.norm(z[:3]))-translation_limit));out.append(2*max(0,float(np.linalg.norm(z[3:6]))-rotation_limit))
            out.extend(2*z[:3]);out.extend(.08*z[3:6]);out.extend(.002*z[6:]);return np.asarray(out)
        if a.audit_seed_only is not None:sol=SimpleNamespace(x=x,cost=0.)
        elif a.hard_collision:
            def cons(z):
                _,RR=state(z);vv={g:verts(g)for g in hand}
                values=[interval_gap(vv[g],h)-.00015 for g in hand for h in shoes]
                if shell_axis_local is not None:values.append(shell_gap(RR,vv)-.0001)
                if a.wide_search:
                    for g,sign in[(thumb,sgn),(digit,-sgn)]:
                        yy=((vv[g]-shoeP[wall])@shoeR[wall])[:,1];values.append(float((sign*yy).min())-m.geom_size[wall,1]-.0003)
                return np.r_[values,translation_limit**2-np.sum(z[:3]**2),rotation_limit**2-np.sum(z[3:6]**2)]
            opt=minimize(lambda z:float(np.sum(residual(z)**2))/10000,x,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[{'type':'ineq','fun':cons}],options={'maxiter':260,'ftol':1e-10,'eps':2e-6})
            sol=SimpleNamespace(x=opt.x,cost=float(opt.fun*5000));print(json.dumps({'trial':trial,'hard_status':str(opt.message),'minimum_constraint':float(cons(opt.x).min())}),flush=True)
        else:sol=least_squares(residual,x,bounds=(lo,hi),max_nfev=160,ftol=1e-7,xtol=1e-7,gtol=1e-7,diff_step=2e-5)
        print(json.dumps({'trial':trial,'optimizer_cost':float(sol.cost),'phase':'full_SAT_QP_validation'}),flush=True)
        detail=validate(sol.x,digit);detail.update(trial=trial,objective_cost=float(sol.cost),x=sol.x.tolist(),pinch_digit=m.geom(digit).name,target_patch_world=[v.tolist()for v in targets])
        trials.append(detail)
        score=sum(abs(c['signed_gap_m'])for c in detail['contact_evidence'])+.1*(len(detail['shoe_penetrations'])+len(detail['hand_self_penetrations']))
        if best is None or(detail['feasible']and not best['feasible'])or(detail['feasible']==best['feasible']and score<best['score']):best={**detail,'score':score}
        print(json.dumps({'trial':trial,'cost':float(sol.cost),'feasible':detail['feasible'],'collision_audit_complete':detail['collision_audit_complete'],'contact_gaps_m':[c['signed_gap_m']for c in detail['contact_evidence']],'shoe_penetrations':len(detail['shoe_penetrations']),'self_penetrations':len(detail['hand_self_penetrations']),'opposition':detail['opposition_normal_dot']}),flush=True)
        if detail['feasible']:break
    a.output_dir.mkdir(parents=True)
    report={'model':str(a.model),'model_sha256':sha(a.model),'reference':str(a.reference),'reference_sha256':sha(a.reference),'frame':frame,'generator_sha256':sha(__file__),'hard_collision':a.hard_collision,'wide_search':a.wide_search,'translation_norm_bound_m':translation_limit,'rotation_norm_bound_rad':rotation_limit,'driver_names':['p1_rh_'+n for n in digits],
            'original_hand_position_world_m':P0.tolist(),'original_hand_rotation_world':R0.tolist(),'best':best,'trials':trials,
            'limitations':['Independent static hand search; no arm IK/whole-body reachability/temporal transition or physical grasp force proof.','Source model and reference unchanged. Wrist-target and6driver output are candidates only until retarget integrates and validates.','Optimization seeds use conservative box-axis separation certificates; every reported feasible candidate is then rechecked by full convex SAT plus precise convex-QP distances.']}
    (a.output_dir/'candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output_dir/'generator.py').write_bytes(Path(__file__).read_bytes())
    np.savez_compressed(a.output_dir/'keyframe_targets.npz',frame=np.asarray(frame),right_hand_base_position_world_m=np.asarray(best['hand_position_world_m']),right_hand_base_rotation_world=np.asarray(best['hand_rotation_world']),driver_names=np.asarray(report['driver_names']),driver_angles_rad=np.asarray(best['driver_angles_rad']),geometric_grasp_quality=np.asarray(best['feasible']))
    print(json.dumps({'output':str(a.output_dir),'feasible':best['feasible'],'best_score':best['score']}),flush=True)
if __name__=='__main__':main()
