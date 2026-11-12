"""Read-only diagnosis of an interrupted or fallen full-sequence rollout."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj

p=argparse.ArgumentParser();p.add_argument('--trial-dir',type=Path,required=True);a=p.parse_args()
r=json.loads((a.trial_dir/'report.json').read_text());z=np.load(a.trial_dir/'physical_trace.npz',allow_pickle=False)
mp=Path(r['model']['path']);m=mj.MjModel.from_binary_path(str(mp))if mp.suffix=='.mjb'else mj.MjModel.from_xml_path(str(mp))
t=z['time'];bad=np.flatnonzero(np.diff(t)<=0)+1;n=int(bad[0])if len(bad)else len(t)
t=t[:n];q=z['qpos'][:n];v=z['qvel'][:n];ref=np.load(r['reference']['path'],allow_pickle=False);rt=ref['time_s']-ref['time_s'][0]
up=1-2*(q[:,4]**2+q[:,5]**2);fall=np.flatnonzero(up<.5)
rootref=np.array([np.interp(t,rt,ref['qpos'][:,j])for j in range(3)]).T
names=[]
for i in range(m.nv):
    j=int(m.dof_jntid[i]);names.append((mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)or str(j))+':'+str(i-int(m.jnt_dofadr[j])))
peaks=np.max(np.abs(v),axis=0);order=np.argsort(peaks)[-12:][::-1]
result={'valid_prefix_states':n,'valid_prefix_duration_s':float(t[-1]),'nonmonotonic_times':bad.tolist(),
    'first_root_tilt_fall_time_s':float(t[fall[0]])if len(fall)else None,'root_up_min':float(up.min()),
    'root_xy_error_max_m':float(np.linalg.norm(q[:,:2]-rootref[:,:2],axis=1).max()),
    'root_actual_displacement_xyz_m':(q[-1,:3]-q[0,:3]).tolist(),
    'root_reference_displacement_xyz_m':(rootref[-1]-rootref[0]).tolist(),
    'root_actual_xy_path_length_m':float(np.linalg.norm(np.diff(q[:,:2],axis=0),axis=1).sum()),
    'root_vertical_error_min_m':float((q[:,2]-rootref[:,2]).min()),
    'largest_velocity_dofs':[{'name':names[i],'velocity_peak':float(peaks[i]),'peak_time_s':float(t[np.argmax(np.abs(v[:,i]))])}for i in order],
    'object_hand_force_peak_N':np.max(z['object_hand_normal_force_N'][:max(1,n-1)],axis=0).tolist()}
out=a.trial_dir/'failure_diagnosis.json';out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
