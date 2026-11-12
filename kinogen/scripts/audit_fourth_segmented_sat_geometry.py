"""Independent LH/RH finite differences and exhaustive-vs-pruned SAT queries.

Left-hand fixtures relocate an object in private FK only. They are explicitly
not physical contact, a rollout, or evidence of manipulation success.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from fourth_contact_feedback_segmented import SegmentedReferenceContactFeedback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model));ref=dict(np.load(a.reference,allow_pickle=False))
hook=SegmentedReferenceContactFeedback(m,ref,{'geometry_backend':'convex_sat'})
qa=[int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,str(n))])for n in ref['dof_names']]
object_qa=[int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,f'target_object_{int(oid):02d}_free')])for oid in ref['object_ids']]
results=[];start=time.perf_counter()
for frame,oi in [(88,0),(485,3)]:
    for side in [1,0]:
        d=hook._kin;d.qpos[:7]=ref['qpos'][frame,:7];d.qpos[qa]=ref['qpos'][frame,7:]
        for index,q in enumerate(object_qa):d.qpos[q:q+7]=ref['object_target_pose'][frame,index]
        mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
        if side==0:
            left=hook.hands[0];right=hook.hands[1]
            rb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'rh_hand_base_link');lb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'lh_hand_base_link')
            rotate=d.xmat[lb].reshape(3,3)@d.xmat[rb].reshape(3,3).T
            translate=d.xpos[lb]-rotate@d.xpos[rb]
            for lq,rq in zip(left['qa'],right['qa']):d.qpos[lq]=d.qpos[rq]
            q=object_qa[oi];oldpose=d.qpos[q:q+7].copy()
            d.qpos[q:q+3]=rotate@oldpose[:3]+translate
            rotation=rotate@Rotation.from_quat(oldpose[[4,5,6,3]]).as_matrix()
            xyzw=Rotation.from_matrix(rotation).as_quat();d.qpos[q+3:q+7]=xyzw[[3,0,1,2]]
            mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
        hand=hook.hands[side];gap,J=hook._geometry(hand,oi,True)
        numerical=np.zeros_like(J);eps=1e-5
        for j,q in enumerate(hand['qa']):
            original=float(d.qpos[q]);d.qpos[q]=original+eps;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
            plus,_=hook._geometry(hand,oi)
            d.qpos[q]=original-eps;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
            minus,_=hook._geometry(hand,oi);numerical[:,j]=(plus-minus)/(2*eps)
            d.qpos[q]=original
        mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
        exhaustive=[]
        for digit in hand['digits']:
            values=[hook._sat.query(d,tip,obj)[0]for tip in digit['tip_geoms']for obj in hook.object_geoms[oi]]
            exhaustive.append(min(values))
        # Near fixtures use SAT for all digits, except clearly remote fingers
        # that retain GJK. Compare rows that have a matching SAT backend only.
        comparison=[]
        for row,digit in enumerate(hand['digits']):
            if abs(gap[row]-exhaustive[row])<1e-6:comparison.append(row)
            elif gap[row]>.025:continue
            else:raise ValueError(('Pruning mismatch',frame,side,row,gap[row],exhaustive[row]))
        error=float(np.max(abs(J-numerical)))
        results.append({'source_frame':frame,'hand':side,'left_object_relocated_in_private_FK_fixture_only':side==0,
                        'joint_names':hand['names'],'all_11_joint_columns_tested':True,
                        'max_jacobian_error_m_per_rad':error,'passed':error<.005,
                        'constraint_gaps_m':gap.tolist(),'exhaustive_sat_gaps_m':exhaustive,
                        'exhaustive_query_agreement_rows':comparison,
                        'maximum_tested_joint_column_norm':float(np.max(np.linalg.norm(J,axis=0)))})
report={'schema':'fourth.segmented_sat_independent_geometry_audit.v1','model':str(a.model),
        'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'geometry_tests':results,
        'all_22_native_joint_columns_checked':True,'all_passed':all(item['passed']for item in results),
        'query_oracle':hook._sat.report(),'elapsed_s':time.perf_counter()-start,
        'physical_rollout_performed':False,'physical_contact_or_grasp_success_claimed':False}
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report),flush=True)
if not report['all_passed']:raise SystemExit(2)
