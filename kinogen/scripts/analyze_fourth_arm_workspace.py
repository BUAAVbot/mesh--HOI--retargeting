"""Read-only candidate-base FK, never a physical-success claim."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import mujoco as mj
from fourth_arm_object_feedback import ArmObjectFeedback
p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--time',type=float,default=4.5)
p.add_argument('--model',type=Path);p.add_argument('--settled-objects',type=Path);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path)
p.add_argument('--candidate-bases',default='[[0.08,1.8],[-0.08,1.9],[-0.046,1.9],[-0.08,2.0],[-0.12,2.0]]');a=p.parse_args()
r=json.loads((a.trial/'report.json').read_text());mp=a.model or Path(r['model']['path'])
m=mj.MjModel.from_binary_path(str(mp))if mp.suffix=='.mjb'else mj.MjModel.from_xml_path(str(mp.resolve()))
z=np.load(a.trial/'physical_trace.npz');ref=dict(np.load(a.reference,allow_pickle=False));i=int(np.argmin(abs(z['time']-a.time)));d=mj.MjData(m);rows=[]
config=json.loads(a.config.read_text()).get('arm_feedback',{})if a.config else{};config['ik_iterations']=15
settled=None
if a.settled_objects:
    poses=np.load(a.settled_objects,allow_pickle=False);settled=poses['final_object_pose_world_xyz_wxyz']if'final_object_pose_world_xyz_wxyz'in poses else poses['object_pose'][-1]
    if settled.shape!=(6,7):raise ValueError('Expected six settled world xyz+wxyz poses')
for xy in json.loads(a.candidate_bases):
    d.qpos[:]=z['qpos'][i];d.qpos[:2]=xy;d.qpos[3:7]=ref['qpos'][0,3:7]
    if settled is not None:d.qpos[59:]=settled.reshape(-1)
    mj.mj_forward(m,d)
    helper=ArmObjectFeedback(m,ref,config);goal,rotation=helper.acquire_goal(d,1,0);ctrl=z['ctrl'][min(i,len(z['ctrl'])-1)]
    for k in range(80):
        target,error=helper.solve(d,1,goal,rotation,ctrl);helper.last_targets[1]=target
    ik=helper.data;chain=helper.chains[1];ik.qpos[:]=d.qpos;ik.qpos[chain['qa']]=target;mj.mj_forward(m,ik)
    robot=[g for g in range(m.ngeom)if m.geom_contype[g]and m.geom_bodyid[g]>0 and m.geom_bodyid[g]not in helper.object_bids]
    distances=[(float(mj.mj_geomDistance(m,ik,g,tg,2.,np.zeros(6))),mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,tg))for g in robot for tg in helper.table_geoms];closest=min(distances)
    rows.append({'root_position':d.qpos[:3].tolist(),'actual_object':helper.actual_pose(d,0).tolist(),'palm_goal':goal.tolist(),'ik_error_m':error,'candidate_palm':helper.palm(ik,1).tolist(),'min_robot_table_signed_distance_m_robotgeom_tablegeom':closest,'static_table_clearance_pass':bool(closest[0]>=.001),'palm_ik_1cm_pass':bool(error<=.01)})
report={'scope':'Read-only candidate-base FK, not physical success','simulation_steps':0,'model':str(mp),'model_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'actual_robot_trace_time_s':float(z['time'][i]),'object_pose_source':str(a.settled_objects)if a.settled_objects else'previous dynamic trace on its recorded table model','candidates':rows}
if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
