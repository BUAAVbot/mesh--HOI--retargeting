from pathlib import Path
import json
import mujoco as mj
import numpy as np

R=Path('/data/KinoGen')
m=mj.MjModel.from_xml_path(str(R/'outputs/fourth_vita_demo/assets/vita_object_05_dynamic_ankle100.xml'))
d=mj.MjData(m)
z=np.load(R/'outputs/fourth_vita_demo/retarget_reference_palm_v6.npz',allow_pickle=False)
o=np.load(R/'data/fourth_sim_handoff/objects/poses_sim.npz',allow_pickle=False)
print('GEOMS',[(mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),m.geom_pos[g].tolist(),m.geom_size[g].tolist()) for g in range(m.ngeom) if 'table' in (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'') or 'target_object_05' in (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or'')])
print('ROOT',z['qpos'][0,:7].tolist())
print('OBJ',[(f,o['position_m'][f,4].tolist(),int(o['status'][f,4]),o['quaternion_wxyz'][f,4].tolist()) for f in [567,581,590,600,620]])
d.qpos[:59]=z['qpos'][0];d.qpos[59:66]=z['object_target_pose'][0];mj.mj_forward(m,d)
print('HANDS',[(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b),d.xpos[b].tolist())for b in range(m.nbody)if any(s in(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b)or'')for s in ['hand_base','wrist_roll','shoulder_pitch'])])
print('JOINTS',[(mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j),float(d.qpos[m.jnt_qposadr[j]]),m.jnt_range[j].tolist())for j in range(m.njnt)if 'left_' in(mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)or'')])
