"""Server-side kinematic contract/gradient checks, not a physics success test."""
from pathlib import Path
import argparse
import json
import os

import numpy as np
import mujoco as mj
from fourth_contact_feedback import ReferenceContactFeedback


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if os.name!='posix':raise ValueError('Kinematic checks run only on the server')
    model=mj.MjModel.from_xml_path(str(a.model));ref=dict(np.load(a.reference,allow_pickle=False))
    feedback=ReferenceContactFeedback(model,ref)
    names=list(map(str,ref['dof_names']))
    qa=np.asarray([model.jnt_qposadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n)]for n in names])
    aid=np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in names])
    object_ids=np.asarray(ref['object_ids'],int)
    oqa=np.asarray([model.jnt_qposadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,f'target_object_{i:02d}_free')]for i in object_ids])
    cand=np.asarray(ref['contact_candidate_observed'],bool)
    candidates=np.flatnonzero(cand.any(axis=(1,2)))
    frames=candidates[np.linspace(0,len(candidates)-1,min(8,len(candidates))).round().astype(int)]if len(candidates)else np.asarray([0])
    geometry_contract=feedback.hand_geometry_contract();checks=[];gradient_checks=[]
    for frame in frames:
        feedback.reset();d=mj.MjData(model);d.qpos[:7]=ref['qpos'][frame,:7];d.qpos[qa]=ref['qpos'][frame,7:]
        for i,q in enumerate(oqa):d.qpos[q:q+7]=ref['object_target_pose'][frame,i]
        d.time=float(ref['time_s'][frame]);mj.mj_forward(model,d)
        ctrl=np.zeros(model.nu);ctrl[aid]=ref['qpos'][frame,7:]
        first=None
        for k in range(2):
            d.time=float(ref['time_s'][frame])+k*.02
            before={key:getattr(d,key).copy()for key in ['qpos','qvel','qfrc_applied','xfrc_applied','ctrl']}
            original=ctrl.copy();out=feedback.apply(d,ctrl,float(ref['time_s'][frame])+k*.02)
            immutable=all(np.array_equal(value,getattr(d,key))for key,value in before.items())and np.array_equal(ctrl,original)
            preserved=np.array_equal(out[feedback.preserved_actuator_ids],ctrl[feedback.preserved_actuator_ids])
            finger_range=all(np.all(out[h['aids']]>=h['lower']-1e-9)and np.all(out[h['aids']]<=h['upper']+1e-9)for h in feedback.hands)
            rate=True if first is None else all(np.all(np.abs(out[h['aids']]-first[h['aids']])<=np.minimum(h['speed'],feedback.config['closure_rate_rad_s'])*.02+1e-9)for h in feedback.hands)
            if not(immutable and preserved and finger_range and rate):
                raise ValueError(('Contact hook contract failed',int(frame),immutable,preserved,finger_range,rate))
            checks.append({'frame':int(frame),'call':k,'actual_data_and_input_ctrl_unchanged':immutable,
                           'baseline_body_neck_ctrl_preserved':preserved,'finger_target_in_model_range':finger_range,
                           'target_rate_limit_checked':rate,'phases':[s['phase']for s in feedback.states],
                           'selected_objects':[int(object_ids[s['object_index']])if s['object_index']>=0 else-1 for s in feedback.states]})
            first=out.copy()
        # Numerical distance derivatives independently check signed witness /
        # point-Jacobian direction, including any penetrating source sample.
        selected=np.argwhere(cand[frame])
        if len(selected):
            side,oi=map(int,selected[0]);hand=feedback.hands[side]
            feedback._kin.qpos[:]=d.qpos;mj.mj_forward(model,feedback._kin)
            dist,J=feedback._geometry(hand,oi,True)
            numerical=np.zeros_like(J);eps=1e-5
            for j,qa_hand in enumerate(hand['qa']):
                original=float(feedback._kin.qpos[qa_hand])
                feedback._kin.qpos[qa_hand]=original+eps;mj.mj_forward(model,feedback._kin);plus,_=feedback._geometry(hand,oi)
                feedback._kin.qpos[qa_hand]=original-eps;mj.mj_forward(model,feedback._kin);minus,_=feedback._geometry(hand,oi)
                numerical[:,j]=(plus-minus)/(2*eps)
                feedback._kin.qpos[qa_hand]=original
            mj.mj_forward(model,feedback._kin)
            error=float(np.max(np.abs(numerical-J)))
            gradient_checks.append({'frame':int(frame),'hand':side,'object_id':int(object_ids[oi]),
                                    'signed_tip_distances_m':dist.tolist(),'jacobian_max_abs_error_m_per_rad':error,
                                    'passed':error<.005})
    active=sum(any(p in ['acquire','grasp','release']for p in sample['phases'])for sample in checks)
    report={'schema':'fourth.contact_feedback_kinematic_smoke.v1',
            'status':'passed'if all(g['passed']for g in gradient_checks)and active>0 else'gradient_or_acquisition_check_failed',
            'source_frames_checked':frames.tolist(),'contracts':checks,'gradient_checks':gradient_checks,
            'active_acquisition_samples':active,
            'hand_geometry_contract':geometry_contract,
            'physical_rollout_performed':False,'physical_manipulation_success_claimed':False,
            'scope':'Read-only hook contract, source model limits/rates and numerical signed-distance Jacobians only.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'samples':len(checks),'gradients':gradient_checks,'hand_geometry_contract':geometry_contract}),flush=True)
    if report['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
