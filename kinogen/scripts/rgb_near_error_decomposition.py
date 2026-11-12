"""Read-only diagnostic decomposition of source/actual hand and shoe errors."""
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco

p=argparse.ArgumentParser();p.add_argument('trial',type=Path);a=p.parse_args()
r=json.loads((a.trial/'report.json').read_text());z=np.load(a.trial/'trace.npz',allow_pickle=False)
m=mujoco.MjModel.from_binary_path(str(a.trial/'model.mjb'));d=mujoco.MjData(m);ref=mujoco.MjData(m)
names=list(map(str,z['joint_names']));bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link');base=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'p1_base_link')
oq=int(z['object_qadr']);out=[]
indices=[20,int(np.argmax(z['shoe_reference_error_m']))+1,len(z['time'])-1]
for i in indices:
    t=float(z['time'][i]);d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d)
    ref.qpos[:]=d.qpos
    for actor in range(2):
        q=np.asarray([np.interp(t,z['reference_time_s'],z['reference_qpos'][:,actor,j])for j in range(59)])
        rq=int(z['robot_root_qadr'][actor]);ref.qpos[rq:rq+7]=q[:7];ref.qpos[z['joint_qadr'][actor]]=q[7:]
    mujoco.mj_forward(m,ref)
    target=np.asarray([np.interp(t,z['reference_time_s'],z['reference_object_target_pose'][:,j])for j in range(3)])
    Ra=d.xmat[base].reshape(3,3);Rr=ref.xmat[base].reshape(3,3);pa=d.xpos[base];pr=ref.xpos[base]
    ha=d.xpos[bid];hr=ref.xpos[bid];la=Ra.T@(ha-pa);lr=Rr.T@(hr-pr)
    predicted=None
    active=np.flatnonzero(z['eq_active'][max(i-1,0)])
    if len(active):
        eq=int(active[0]);b1=int(m.eq_obj1id[eq]);predicted=d.xpos[b1]+d.xmat[b1].reshape(3,3)@z['eq_data'][i-1,eq,3:6]
    out.append({'time_s':t,'shoe_actual':d.qpos[oq:oq+3].tolist(),'shoe_target':target.tolist(),'shoe_error_m':float(np.linalg.norm(d.qpos[oq:oq+3]-target)),
                'hand_actual':ha.tolist(),'hand_target':hr.tolist(),'hand_error_m':float(np.linalg.norm(ha-hr)),
                'root_translation_component_m':(pa-pr).tolist(),'root_rotation_component_m':((Ra-Rr)@lr).tolist(),'arm_joint_component_m':(Ra@(la-lr)).tolist(),
                'attachment_predicted_origin':predicted.tolist()if predicted is not None else None,'attachment_position_residual_m':float(np.linalg.norm(d.qpos[oq:oq+3]-predicted))if predicted is not None else None,
                'root_actual_wxyz':d.qpos[int(z['robot_root_qadr'][0])+3:int(z['robot_root_qadr'][0])+7].tolist(),
                'root_reference_wxyz':ref.qpos[int(z['robot_root_qadr'][0])+3:int(z['robot_root_qadr'][0])+7].tolist(),
                'leg_joints':{name:{'actual_rad':float(d.qpos[z['joint_qadr'][0,names.index(name)]]),
                                    'reference_rad':float(ref.qpos[z['joint_qadr'][0,names.index(name)]]),
                                    'ctrl_rad':float(z['ctrl'][i-1,z['actuator_ids'][0,names.index(name)]])}
                              for name in ['left_hip_pitch_joint','left_knee_joint','left_ankle_pitch_joint','right_hip_pitch_joint','right_knee_joint','waist_pitch_joint']}})
print(json.dumps(out,indent=2))
