"""Save the actual 2s unforced settle state of the six-object table-only probe."""
from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco as mj

R=Path(__file__).resolve().parents[1];A=R/'outputs/fourth_full_sequence/assets'
path=A/'vita_six_objects_newft_source_table_v1_objects_support_probe.xml'
out=A/'vita_six_objects_newft_source_table_v1_settled_2s.npz'
if out.exists():raise ValueError('Do not overwrite frozen settled states')
m=mj.MjModel.from_xml_path(str(path));d=mj.MjData(m);mj.mj_forward(m,d)
q=[d.qpos.copy()];v=[d.qvel.copy()];t=[float(d.time)]
for _ in range(400):
    mj.mj_step(m,d);q.append(d.qpos.copy());v.append(d.qvel.copy());t.append(float(d.time))
addresses=[int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,f'target_object_{i:02d}_free')])for i in range(1,7)]
poses=np.stack([d.qpos[a:a+7]for a in addresses])
np.savez_compressed(out,object_ids=np.arange(1,7),object_pose_xyz_wxyz=poses,
                    final_object_pose_world_xyz_wxyz=poses,
                    object_pose=np.stack([np.asarray(q)[:,a:a+7]for a in addresses],axis=1),
                    qpos=np.asarray(q),qvel=np.asarray(v),time_s=np.asarray(t),
                    model_sha256=np.asarray(hashlib.sha256(path.read_bytes()).hexdigest()),
                    scenario=np.asarray('Six source-initial free objects settling on source-visible convex support hypothesis; no robot, external forces or state writes after reset.'))
out.with_suffix('.json').write_text(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
    'model':str(path),'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'steps':400,
    'duration_s':float(d.time),'object_ids':list(range(1,7)),'object_pose_xyz_wxyz':poses.tolist(),
    'kinematic_or_physical':'real unforced mj_step integration, isolated object/table probe; not a robot task success test'},indent=2)+'\n')
print(json.dumps({'path':str(out),'poses':poses.tolist()}),flush=True)
