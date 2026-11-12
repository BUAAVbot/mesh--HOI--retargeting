"""CPU audit of which joints and geom pairs violate the fixed physics gate."""
import json
from pathlib import Path
import mujoco
import numpy as np

base = Path('/data/yihang.bai/training/hdmi_vita_mjlab')
out = base/'outputs/gallery08_p01_trajopt_20261030_2205'
rows = {}
for name in ('baseline_instrumented', 'candidate_s42_instrumented'):
    folder = out/name
    with np.load(folder/'physical_record.npz', allow_pickle=False) as z:
        record = {k:z[k].copy() for k in z.files}
    m = mujoco.MjModel.from_binary_path(str(folder/'compiled_model.mjb'))
    d = mujoco.MjData(m)
    joint_rows = []
    source = np.load(base/'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3/mjlab_motion.npz')
    for j, joint in enumerate(map(str, record['joint_names'])):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'robot/'+joint)
        q = record['joint_pos'][:,j]
        excess = np.maximum(np.maximum(m.jnt_range[jid,0]-q, q-m.jnt_range[jid,1]),0)
        error = q-source['joint_pos'][:,j]
        joint_rows.append(dict(joint=joint,max_excess=float(excess.max()),excess_frame=int(excess.argmax()),
                               rmse=float(np.sqrt(np.mean(error**2))),max_error=float(abs(error).max())))
    contacts = []
    for frame, qpos in enumerate(record['qpos']):
        d.qpos[:] = qpos
        d.qvel[:] = record['qvel'][frame]
        mujoco.mj_forward(m,d)
        for contact in d.contact[:d.ncon]:
            if contact.dist < -.010:
                contacts.append(dict(frame=frame,depth_m=float(-contact.dist),
                    geom1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(contact.geom1)),
                    geom2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(contact.geom2))))
    rows[name] = dict(worst_joints=sorted(joint_rows,key=lambda x:x['max_excess'],reverse=True)[:12],
                      largest_rmse_joints=sorted(joint_rows,key=lambda x:x['rmse'],reverse=True)[:12],
                      contacts=sorted(contacts,key=lambda x:x['depth_m'],reverse=True)[:20],
                      early_contacts=[c for c in contacts if c['frame']<80][:20],
                      support_force_summary=dict(min=float(record['physics_support_force_N'].min()),
                          fraction_above_10pct_weight=float(np.mean(record['physics_support_force_N']>=.1*record['robot_weight_N']))))
(out/'joint_and_contact_diagnosis.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
