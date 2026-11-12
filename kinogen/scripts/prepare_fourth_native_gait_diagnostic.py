"""Adapt a native Vita clip by one rigid translation for a gait ABI test."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj

R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--motion',type=Path,default=R/'data/vita_demo_20261026/motions/030_0806__0806pm__scene6_7__000_action_000.npz')
p.add_argument('--first',type=int,default=180);p.add_argument('--last',type=int,default=481)
p.add_argument('--model',type=Path,default=R/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_auxrotor_v1.xml')
p.add_argument('--output',type=Path,default=R/'outputs/fourth_full_sequence/reference_native030_gait_diagnostic.npz');a=p.parse_args()
src={k:v for k,v in np.load(a.motion,allow_pickle=False).items()};index=np.arange(a.first,min(a.last,len(src['root_pos'])))
names=src['dof_names'];q=np.c_[src['root_pos'][index],src['root_rot_xyzw'][index][:,[3,0,1,2]],src['dof_pos'][index]].astype(float)
offset=np.array([10.-q[0,0],10.-q[0,1],0.]);q[:,:3]+=offset
m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m)
qa=np.array([m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,str(n))]for n in names])
d.qpos[:7]=q[0,:7];d.qpos[qa]=q[0,7:];mj.mj_forward(m,d)
floor=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'assumed_floor')
feet=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in ['left','right']]
geoms=[g for g in range(m.ngeom)if m.geom_contype[g]and m.geom_bodyid[g]in feet]
clearance=min(float(mj.mj_geomDistance(m,d,floor,g,10.,np.zeros(6)))for g in geoms)
offset[2]=.001-clearance;q[:,2]+=offset[2]
full=np.load(R/'outputs/fourth_full_sequence/reference_v8.npz',allow_pickle=False)
poses=np.tile(full['object_target_pose'][0],(len(index),1,1));t=np.arange(len(index))/float(src['fps'])
np.savez_compressed(a.output,qpos=q,dof_names=names,time_s=t,source_frame_ids=index,object_ids=np.arange(1,7),object_target_pose=poses,
    object_observed_mask=np.ones((len(index),6),bool),human_observed_mask=np.ones(len(index),bool),
    native_motion_source=np.asarray(str(a.motion)),native_motion_source_sha256=np.asarray(hashlib.sha256(a.motion.read_bytes()).hexdigest()),
    root_constant_translation_m=offset,diagnostic_only=np.asarray(True),raw_root_position_m=src['root_pos'][index])
print(json.dumps({'output':str(a.output),'frames':len(index),'duration_s':float(t[-1]),'root_constant_translation_m':offset.tolist(),
    'initial_sole_clearance_after_translation_m':.001,'native_root_delta_m':(q[-1,:3]-q[0,:3]).tolist()}),flush=True)
