"""Hold the final ground-supported right foot through the final display interval.

Offline leg IK only. Original raw human fields and source clock are retained.
This explicit end-condition adaptation does not certify physical feasibility.
"""
from pathlib import Path
import argparse,hashlib,json,sys


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if sys.platform!='linux':p.error('Run offline FK/IK on the server')
    if a.output.exists()or a.output.with_suffix('.json').exists():raise FileExistsError(a.output)
    import numpy as np,mujoco as mj
    from scipy.optimize import least_squares
    z=dict(np.load(a.reference,allow_pickle=False));q0=np.asarray(z['qpos'],float);q=q0.copy();on=np.asarray(z['foot_support_expected'],bool).copy();confidence=np.asarray(z['foot_support_confidence'],float).copy()
    if q.shape!=(350,59)or not np.all(on[346]):raise ValueError('Expected last valid double support at346 and350source frames')
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m)
    names=list(map(str,z['dof_names']));jids=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]);qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids];leg=np.asarray([i for i,n in enumerate(names)if n.startswith('right_')and any(k in n for k in('hip_','knee_','ankle_'))])
    if len(leg)!=6:raise ValueError('Expected right-leg6')
    feet=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in('left','right')]);base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'base_link')
    d.qpos[:]=q0[346];mj.mj_fwdPosition(m,d);anchor=d.xpos[feet[1]].copy();wanted=d.xmat[feet[1]].reshape(3,3).copy();rows=[]
    def skew(v):return np.asarray([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    for f in range(347,350):
        cache={}
        def evaluate(x):
            if 'x'in cache and np.array_equal(x,cache['x']):return cache['r'],cache['J']
            d.qpos[:]=q0[f];d.qpos[qa[leg]]=x;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
            jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mj.mj_jac(m,d,jp,jr,d.xpos[feet[1]],int(feet[1]));jp=jp[:,va[leg]];jr=jr[:,va[leg]];R=d.xmat[feet[1]].reshape(3,3)
            r=np.r_[300*(d.xpos[feet[1]]-anchor),5*(R[:,0]-wanted[:,0]),5*(R[:,2]-wanted[:,2]),.02*(x-q0[346,qa[leg]])]
            J=np.vstack([300*jp,-5*skew(R[:,0])@jr,-5*skew(R[:,2])@jr,.02*np.eye(6)]);cache.update(x=x.copy(),r=r,J=J);return r,J
        fit=least_squares(lambda x:evaluate(x)[0],q[f-1,qa[leg]],jac=lambda x:evaluate(x)[1],bounds=(m.jnt_range[jids[leg],0]+1e-6,m.jnt_range[jids[leg],1]-1e-6),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        evaluate(fit.x);q[f]=d.qpos.copy();on[f,1]=True;confidence[f,1]=confidence[346,1];rows.append({'frame':f,'right_anchor_error_m':float(np.linalg.norm(d.xpos[feet[1]]-anchor)),'nfev':int(fit.nfev)})
    actual=[];com=[];pen=[];joint=[];gap=[]
    for f,row in enumerate(q):
        d.qpos[:]=row;mj.mj_fwdPosition(m,d);actual.append(d.xpos[feet].copy());com.append(d.subtree_com[base].copy());pen.append(max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.]));v=np.maximum(np.maximum(m.jnt_range[jids,0]-row[qa],row[qa]-m.jnt_range[jids,1]),0);v[~m.jnt_limited[jids].astype(bool)]=0;joint.append(v)
    actual=np.asarray(actual);com=np.asarray(com);pen=np.asarray(pen);joint=np.asarray(joint);parent_actual=np.asarray(z['foot_actual_ankle_world'])
    # Preserve every parent's numerical array under a namespace, then publish only
    # frame-dependent diagnostics independently recomputed for the repaired poses.
    out={k:v for k,v in z.items()if k.startswith('source_')or k in('frame_ids','time_s','dof_names','foot_support_geom_name','wrist_reference_joint_names','wrist_reference_joint_targets_rad')}
    out.update({'parent_reference_'+k:v for k,v in z.items()if not k.startswith('source_')})
    target=np.asarray(z['foot_target_ankle_world']).copy();target[347:,1]=anchor
    out.update(qpos=q,foot_support_expected=on,foot_support_confidence=confidence,foot_target_ankle_world=target,foot_actual_ankle_world=actual,reference_com_world=com,root_target_world=q[:,:3].copy(),static_penetration_m=pen,static_joint_limit_violation_rad=joint,static_geometry_passed=np.asarray(pen.max()<=.02 and joint.max()<=.05),physical_joint_limit_model_sha256=np.asarray(sha(a.model)),final_contact_repair_version=np.asarray('v19_final_right_ground_hold'))
    unchanged=np.ones(59,bool);unchanged[qa[leg]]=False
    assert np.array_equal(q[:347],q0[:347])and np.array_equal(q[:,unchanged],q0[:,unchanged])
    edges=[]
    for side in range(2):
        change=np.flatnonzero(np.diff(on[:,side].astype(int)))+1
        for f in change:
            move=actual[f,side]-actual[f-1,side]
            if np.linalg.norm(move)>.10:edges.append({'frame':int(f),'side':('left','right')[side],'source_time_s':float(z['source_time_s'][f]),'from_stance':bool(on[f-1,side]),'to_stance':bool(on[f,side]),'ankle_jump_m':float(np.linalg.norm(move)),'ankle_delta_xyz_m':move.tolist()})
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,**out)
    report={'version':'v19_final_right_ground_hold','physics_passed':False,'model_parameters_modified':False,'source_frames_preserved':350,'source_fields_preserved':True,'only_modified_qpos_frames':[347,348,349],'only_modified_joints':[names[i]for i in leg],'root_upper_body_hands_unchanged':True,'contact_label_assumption':'Extend the existing346rightground support through349final display; source human foot measurements are retained unchanged; this is an explicit robotic terminal-condition adaptation','right_ground_anchor_world_m':anchor.tolist(),'parent_final_right_lift_peak_m':float((parent_actual[347:,1,2]-parent_actual[346,1,2]).max()),'new_final_right_anchor_error_max_m':max(r['right_anchor_error_m']for r in rows),'penetration_max_m':float(pen.max()),'joint_limit_violation_max_rad':float(joint.max()),'static_geometry_passed':bool(pen.max()<=.02 and joint.max()<=.05),'unresolved_contact_boundary_jumps_over_10cm':edges,'all_boundary_jumps_solved':not bool(edges),'input_reference':str(a.reference),'input_reference_sha256':sha(a.reference),'model_sha256':sha(a.model),'generator_sha256':sha(__file__),'rows':rows}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report),flush=True)


if __name__=='__main__':main()
