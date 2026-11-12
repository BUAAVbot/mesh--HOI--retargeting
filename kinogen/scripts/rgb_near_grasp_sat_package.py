"""Package static grasp transforms and audit explicit O6 closure paths read-only."""
from pathlib import Path
import argparse,json,hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--samples',type=int,default=32);a=p.parse_args()
    import numpy as np,mujoco as mj
    from scipy.spatial.transform import Rotation
    from types import SimpleNamespace
    from rgb_near_convex_distance import ShoeHandConvexDistance
    r=json.loads(a.candidate.read_text());best=r['best']
    if not best['feasible']:raise ValueError('Only package a statically accepted candidate')
    m=mj.MjModel.from_xml_path(r['model']);d=mj.MjData(m);ref=np.load(r['reference'],allow_pickle=False);frame=r['frame'];out=a.candidate.parent
    q0=np.r_[ref['qpos'][frame,0],ref['qpos'][frame,1],ref['object_target_pose'][frame]];d.qpos[:]=q0;mj.mj_forward(m,d)
    hb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link');sb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'shoe')
    P0=d.xpos[hb].copy();R0=d.xmat[hb].reshape(3,3).copy();Ps=d.xpos[sb].copy();Rs=d.xmat[sb].reshape(3,3).copy();Ph=np.asarray(best['hand_position_world_m']);Rh=np.asarray(best['hand_rotation_world'])
    driver_names=r['driver_names'];js=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in driver_names]);qa=m.jnt_qposadr[js];target=np.asarray(best['driver_angles_rad'])
    couplings=[(int(m.jnt_qposadr[int(m.eq_obj1id[e])]),int(m.jnt_qposadr[int(m.eq_obj2id[e])]),m.eq_data[e,:5].copy())for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT]
    sg=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'shoe_sole_proxy');up=d.geom_xmat[sg].reshape(3,3)[:,2].copy()
    contract={'candidate':str(a.candidate),'candidate_sha256':sha(a.candidate),'model':r['model'],'model_sha256':r['model_sha256'],'reference':r['reference'],'reference_sha256':r['reference_sha256'],'frame':frame,
              'right_hand_base_position_world_m':Ph.tolist(),'right_hand_base_rotation_world':Rh.tolist(),'inferred_grasp_rel_position':(Rh.T@(Ps-Ph)).tolist(),'inferred_grasp_rel_rotation':(Rh.T@Rs).tolist(),
              'hand_position_in_shoe_local_m':(Rs.T@(Ph-Ps)).tolist(),'hand_rotation_in_shoe_local':(Rs.T@Rh).tolist(),'shoe_pose_world_xyz_wxyz':ref['object_target_pose'][frame].tolist(),
              'driver_names':driver_names,'driver_angles_rad':target.tolist(),'contact_witnesses':[],'shoe_up_world':up.tolist(),
              'static_geometric_grasp_quality':True,'arm_IK_and_full_body_collision_validated':False,'temporal_transition_validated':False,'dynamic_force_grasp_validated':False}
    for c in best['contact_evidence']:
        w=np.asarray(c['qp_witness_world']);n=(w[:3]-w[3:])/c['signed_gap_m']
        contract['contact_witnesses'].append({'hand_geom':c['hand_geom'],'shoe_geom':c['shoe_geom'],'gap_m':c['signed_gap_m'],'hand_surface_point_in_hand_local_m':(Rh.T@(w[:3]-Ph)).tolist(),
                                             'shoe_surface_point_in_shoe_local_m':(Rs.T@(w[3:]-Ps)).tolist(),'outward_normal_world_from_shoe_to_hand':n.tolist(),'outward_normal_shoe_local':(Rs.T@n).tolist()})
    (out/'keyframe_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    np.savez_compressed(out/'keyframe_contract.npz',**{k:np.asarray(v)for k,v in contract.items()if k in ['right_hand_base_position_world_m','right_hand_base_rotation_world','inferred_grasp_rel_position','inferred_grasp_rel_rotation','hand_position_in_shoe_local_m','hand_rotation_in_shoe_local','shoe_pose_world_xyz_wxyz','driver_names','driver_angles_rad']})
    print(json.dumps({'contract':str(out/'keyframe_contract.json'),'grasp_rel_position':contract['inferred_grasp_rel_position'],'grasp_rel_rotation':contract['inferred_grasp_rel_rotation']}),flush=True)
    oracle=ShoeHandConvexDistance(m);hand=[g for g in range(m.ngeom)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith('p1_rh_')and m.geom_contype[g]]
    shoe=[g for g in range(m.ngeom)if m.geom_bodyid[g]==sb and m.geom_contype[g]];shapes={g:oracle._shape(g)for g in hand+shoe}
    shoeR={g:d.geom_xmat[g].reshape(3,3).copy()for g in shoe};shoeP={g:d.geom_xpos[g].copy()for g in shoe};shadow=SimpleNamespace(geom_xpos=d.geom_xpos.copy(),geom_xmat=d.geom_xmat.copy())
    selfpairs=[]
    for ii,g in enumerate(hand):
        wb=int(m.body_weldid[int(m.geom_bodyid[g])])
        for h in hand[ii+1:]:
            wc=int(m.body_weldid[int(m.geom_bodyid[h])])
            if wb==wc or int(m.body_weldid[int(m.body_parentid[wb])])==wc or int(m.body_weldid[int(m.body_parentid[wc])])==wb:continue
            selfpairs.append((g,h))
    def state(theta,offset):
        d.qpos[:]=q0;d.qpos[qa]=theta
        for dep,drv,c in couplings:d.qpos[dep]=np.polynomial.polynomial.polyval(d.qpos[drv],c)
        mj.mj_kinematics(m,d);delta=Rh@R0.T;shadow.geom_xpos[:]=d.geom_xpos;shadow.geom_xmat[:]=d.geom_xmat
        for g in hand:shadow.geom_xpos[g]=(d.geom_xpos[g]-P0)@delta.T+Ph+offset;shadow.geom_xmat[g]=(delta@d.geom_xmat[g].reshape(3,3)).ravel()
    def vertices(g):return shapes[g]['vertices']@shadow.geom_xmat[g].reshape(3,3).T+shadow.geom_xpos[g]
    def selfgap(g,h,vg,vh):
        A,B=shapes[g],shapes[h];RA=shadow.geom_xmat[g].reshape(3,3);RB=shadow.geom_xmat[h].reshape(3,3);best=-np.inf
        def scan(axes):
            nonlocal best
            for start in range(0,len(axes),256):
                nn=axes[start:start+256];ll=np.linalg.norm(nn,axis=1);nn=nn[ll>1e-9]/ll[ll>1e-9,None]
                if not len(nn):continue
                pa=nn@vg.T;pb=nn@vh.T;best=max(best,float(np.max(np.maximum(pa.min(1)-pb.max(1),pb.min(1)-pa.max(1)))))
                if best>1e-7:return True
            return False
        if scan(np.r_[np.eye(3),A['normals']@RA.T,B['normals']@RB.T]):return best
        ea=A['edges']@RA.T;eb=B['edges']@RB.T
        for start in range(0,len(ea),8):
            if scan(np.cross(ea[start:start+8,None,:],eb[None,:,:]).reshape(-1,3)):return best
        return best
    def point_audit(theta,offset):
        state(theta,offset);vv={g:vertices(g)for g in hand};bad=[];certmin=1e9
        for g in hand:
            for h in shoe:
                local=(vv[g]-shoeP[h])@shoeR[h];sz=m.geom_size[h];cert=float(np.max(np.r_[local.min(0)-sz,-sz-local.max(0)]))
                gap=cert if cert>1e-7 else oracle.query(shadow,g,h,precise=False)['sat_gap_m'];certmin=min(certmin,gap)
                if gap<-1e-7:bad.append({'kind':'hand_shoe','geom1':m.geom(g).name,'geom2':m.geom(h).name,'sat_gap_m':gap})
        if bad:return{'passed':False,'shoe_min_separation_certificate_m':certmin,'violations':bad,'hand_self_audit_complete':False}
        for g,h in selfpairs:
            vg,vh=vv[g],vv[h];bb=float(np.max(np.r_[vg.min(0)-vh.max(0),vh.min(0)-vg.max(0)]))
            if bb>1e-7:continue
            gap=selfgap(g,h,vg,vh)
            if gap<-1e-7:bad.append({'kind':'hand_self','geom1':m.geom(g).name,'geom2':m.geom(h).name,'sat_gap_m':gap})
        return{'passed':not bad,'shoe_min_separation_certificate_m':certmin,'violations':bad,'hand_self_audit_complete':True}
    ss=np.linspace(0,1,a.samples);smooth=ss*ss*(3-2*ss);dense=np.linspace(0,1,2001);h=dense*dense*(3-2*dense);hdot=6*dense*(1-dense)
    speed=[float(np.max(abs(hdot*t)))for t in target]
    for dep,drv,c in couplings:
        if drv not in qa:continue
        t=target[list(qa).index(drv)];der=np.polynomial.polynomial.polyval(h*t,np.polynomial.polynomial.polyder(c));speed.append(float(np.max(abs(der*hdot*t))))
    closure_duration=max(speed)*1.25;paths=[];accepted=None
    for height in [0.,.05,.08,.10]:
        samples=[];failed=False
        for i,s in enumerate(smooth):
            audit=point_audit(target*s,up*height);samples.append({'stage':'closure','i':i,'time_s':float(ss[i]*closure_duration),'offset_world_m':(up*height).tolist(),'driver_angles_rad':(target*s).tolist(),**audit})
            if not audit['passed']:failed=True;break
        if not failed and height>0:
            for i,s in enumerate(smooth):
                audit=point_audit(target,up*height*(1-s));samples.append({'stage':'closed_hand_approach','i':i,'time_s':float(closure_duration+ss[i]),'offset_world_m':(up*height*(1-s)).tolist(),'driver_angles_rad':target.tolist(),**audit})
                if not audit['passed']:failed=True;break
        record={'pregrasp_shoe_up_offset_m':height,'closure_duration_s':closure_duration,'closure_sample_count':a.samples,'requested_approach_sample_count':a.samples if height else 0,'passed':not failed,'audited_points':len(samples),'samples':samples}
        paths.append(record);print(json.dumps({'path_height_m':height,'passed':not failed,'audited_points':len(samples),'first_violation':next((s['violations']for s in samples if not s['passed']),None)}),flush=True)
        if not failed:accepted=record;break
    result={'candidate_sha256':sha(a.candidate),'generator_sha256':sha(__file__),'max_all11_joint_velocity_rad_s':max(speed)/closure_duration,'velocity_limit_assumed_rad_s':1.,'safety_factor':1.25,'cubic_closure_duration_s':closure_duration,
            'paths':paths,'accepted_path':accepted,'limitations':['Discrete static geometry at requested points, not continuous collision detection or a dynamic force-bearing grasp test.','Only hand-shoe and enabled nonadjacent hand-internal collision geometry checked; arm/robot/scene IK and collision and velocity validation remain with retarget/physics.','No existing model/reference was modified.']}
    (out/'closure_path_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    if accepted:
        # The stage boundary is audited twice but exported once so consumers
        # receive strictly increasing time stamps for interpolation.
        samples=[]
        for s in accepted['samples']:
            if samples and abs(s['time_s']-samples[-1]['time_s'])<1e-9:continue
            samples.append(s)
        np.savez_compressed(out/'closure_path_targets.npz',time_s=np.asarray([s['time_s']for s in samples]),right_hand_base_position_world_m=np.asarray([Ph+np.asarray(s['offset_world_m'])for s in samples]),right_hand_base_rotation_world=np.repeat(Rh[None],len(samples),axis=0),driver_names=np.asarray(driver_names),driver_angles_rad=np.asarray([s['driver_angles_rad']for s in samples]),geometric_path_quality=np.asarray(True))
        contract.update(hand_path_discrete_geometry_validated=True,hand_path_audited_points=len(accepted['samples']),hand_path_exported_unique_points=len(samples),hand_path_duration_s=float(samples[-1]['time_s']))
        (out/'keyframe_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    print(json.dumps({'path_found':accepted is not None,'closure_duration_s':closure_duration,'max_joint_velocity':max(speed)/closure_duration}),flush=True)
if __name__=='__main__':main()
