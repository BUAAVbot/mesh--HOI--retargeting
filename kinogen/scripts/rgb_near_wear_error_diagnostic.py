"""Read-only decomposition of compliant worn-shoe tracking error."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj

p=argparse.ArgumentParser();p.add_argument('trial',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_binary_path(str(a.trial/'model.mjb'));d=mj.MjData(m);r=mj.MjData(m)
z=np.load(a.trial/'trace.npz');b=np.load(a.trial/'root_acceleration_feedback.npz');names=list(map(str,z['joint_names']))
eq=mj.mj_name2id(m,mj.mjtObj.mjOBJ_EQUALITY,'shoe_wear_left');foot=int(m.eq_obj1id[eq]);base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_base_link')
rq=int(z['robot_root_qadr'][0]);oq=int(z['object_qadr']);rows=[]
for sample in sorted(set([0,100,200,300,int(np.argmax(z['shoe_reference_error_m']))+1,len(z['time'])-1])):
    if sample>=len(z['time']):continue
    t=float(z['time'][sample]);step=max(0,sample-1);d.qpos[:]=z['qpos'][sample];r.qpos[:]=d.qpos
    for actor in range(2):
        q=np.asarray([np.interp(t,z['reference_time_s'],z['reference_qpos'][:,actor,j])for j in range(59)])
        adr=int(z['robot_root_qadr'][actor]);r.qpos[adr:adr+7]=q[:7];r.qpos[z['joint_qadr'][actor]]=q[7:]
    mj.mj_kinematics(m,d);mj.mj_kinematics(m,r)
    goal=np.asarray([np.interp(t,z['reference_time_s'],z['reference_object_target_pose'][:,j])for j in range(3)])
    Ra=d.xmat[base].reshape(3,3);Rr=r.xmat[base].reshape(3,3);Rf=d.xmat[foot].reshape(3,3);Rfr=r.xmat[foot].reshape(3,3)
    relref=Rfr.T@(goal-r.xpos[foot]);relcaptured=z['eq_data'][step,eq,3:6];predicted=d.xpos[foot]+Rf@relcaptured
    roottrans=d.xpos[base]-r.xpos[base];rootrotation=(Ra-Rr)@(Rr.T@(goal-r.xpos[base]))
    leg=d.xpos[foot]+Rf@relref-(d.xpos[base]+Ra@(Rr.T@(goal-r.xpos[base])))
    capture=Rf@(relcaptured-relref);residual=d.qpos[oq:oq+3]-predicted
    ci=max(0,np.searchsorted(b['time_s'],t)-1);ids=b['actuator_ids'][ci];offset=b['target_offset_rad'][ci]
    rows.append({'time_s':t,'actual_shoe_xyz_m':d.qpos[oq:oq+3].tolist(),'reference_shoe_xyz_m':goal.tolist(),
        'total_error_xyz_m':(d.qpos[oq:oq+3]-goal).tolist(),'error_norm_m':float(np.linalg.norm(d.qpos[oq:oq+3]-goal)),
        'root_translation_component_xyz_m':roottrans.tolist(),'root_rotation_component_xyz_m':rootrotation.tolist(),
        'leg_configuration_component_xyz_m':leg.tolist(),'captured_transform_component_xyz_m':capture.tolist(),
        'constraint_solver_residual_xyz_m':residual.tolist(),'decomposition_closure_m':float(np.linalg.norm(roottrans+rootrotation+leg+capture+residual-(d.qpos[oq:oq+3]-goal))),
        'wear_captured_relpos_m':relcaptured.tolist(),'wear_reference_relpos_m':relref.tolist(),
        'actual_foot_xyz_m':d.xpos[foot].tolist(),'reference_foot_xyz_m':r.xpos[foot].tolist(),
        'root_error_rad':float(z['root_orientation_error_rad'][step,0])if sample else 0.,
        'leg_joints':{name:{'actual_rad':float(d.qpos[z['joint_qadr'][0,names.index(name)]]),'reference_rad':float(r.qpos[z['joint_qadr'][0,names.index(name)]]),
                           'applied_target_rad':float(z['ctrl'][step,z['actuator_ids'][0,names.index(name)]]),
                           'root_feedback_offset_rad':float(offset[list(ids).index(int(z['actuator_ids'][0,names.index(name)]))])if int(z['actuator_ids'][0,names.index(name)])in ids else 0.}
                      for name in ['left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint','left_ankle_pitch_joint','left_ankle_roll_joint','waist_pitch_joint']}})
out={'trial':str(a.trial),'wear_active_all_steps':bool(np.all(z['eq_active'][:,eq])),
     'actual_phase_names':list(map(str,np.unique(z['shoe_attachment_actual_name']))),'wear_events':json.loads((a.trial/'attachment_events.json').read_text())if(a.trial/'attachment_events.json').exists()else None,
     'samples':rows,'conclusion':'Exact additive vector decomposition separates root translation/rotation, leg configuration, captured attachment transform, and compliant solver residual.'}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
