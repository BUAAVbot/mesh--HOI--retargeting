"""Read-only hand-collider/driver diagnosis from a frozen physical trial."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import mujoco as mj

p=argparse.ArgumentParser();p.add_argument('trial',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
model_path=a.trial/'model.mjb';model=mj.MjModel.from_binary_path(str(model_path))
z=np.load(a.trial/'trace.npz',allow_pickle=False);c=np.load(a.trial/'contacts.npz',allow_pickle=False)
names=list(map(str,c['geom_names']));body_names=list(map(str,c['body_names']))
hands=np.asarray([any(token in names[int(g)]for g in pair for token in ('_lh_','_rh_'))for pair in c['geom_ids']])
choices=np.flatnonzero(hands)
if not len(choices):raise ValueError('Trial has no hand contact')
worst=int(choices[np.argmin(c['signed_distance_m'][choices])]);step=int(c['interval_index'][worst]);gids=c['geom_ids'][worst].astype(int)
handg=next(int(g)for g in gids if any(t in names[int(g)]for t in ('_lh_','_rh_')))
body=int(model.geom_bodyid[handg]);anatomical=body_names[body].split('_')[:3];branch='_'.join(anatomical)
branches=[b for b in range(model.nbody)if body_names[b].startswith(branch)]
if not branches:branches=[body]
def geom(g):
    return {'id':g,'name':names[g],'body':body_names[int(model.geom_bodyid[g])],
            'type':mj.mjtGeom(int(model.geom_type[g])).name,'size':model.geom_size[g].tolist(),
            'contype':int(model.geom_contype[g]),'conaffinity':int(model.geom_conaffinity[g]),
            'group':int(model.geom_group[g]),'margin':float(model.geom_margin[g]),'gap':float(model.geom_gap[g]),
            'solref':model.geom_solref[g].tolist(),'solimp':model.geom_solimp[g].tolist(),
            'friction':model.geom_friction[g].tolist(),'condim':int(model.geom_condim[g]),
            'collision_enabled':bool(model.geom_contype[g]or model.geom_conaffinity[g]),
            'mesh_convex_hull_contact':bool(model.geom_type[g]==mj.mjtGeom.mjGEOM_MESH)}
bodies=[];drivers=[]
for b in branches:
    joints=np.flatnonzero(model.jnt_bodyid==b)
    bodies.append({'name':body_names[b],'mass_kg':float(model.body_mass[b]),'principal_inertia_kg_m2':model.body_inertia[b].tolist(),
                   'inertial_position_m':model.body_ipos[b].tolist(),'joints':[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,int(j))for j in joints],
                   'active_collision_geoms':[geom(g)for g in range(model.ngeom)if model.geom_bodyid[g]==b and(model.geom_contype[g]or model.geom_conaffinity[g])]})
    for j in joints:
        for aid in np.flatnonzero(model.actuator_trnid[:,0]==j):
            qa=int(model.jnt_qposadr[j]);va=int(model.jnt_dofadr[j])
            drivers.append({'joint':mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,int(j)),'actuator':mj.mj_id2name(model,mj.mjtObj.mjOBJ_ACTUATOR,int(aid)),
                            'reflected_rotor_kg_m2':float(model.dof_armature[va]),'dof_damping':float(model.dof_damping[va]),'frictionloss':float(model.dof_frictionloss[va]),
                            'kp':float(model.actuator_gainprm[aid,0]),'kv':float(-model.actuator_biasprm[aid,2]),'effort_limited':bool(model.actuator_forcelimited[aid]),
                            'effort_range_Nm':model.actuator_forcerange[aid].tolist(),'target_rad':float(z['ctrl'][step,aid]),
                            'actual_joint_rad_at_start':float(z['qpos'][step,qa]),'actual_joint_vel_rad_s_at_start':float(z['qvel'][step,va]),
                            'actual_drive_torque_Nm':float(z['actuator_force'][step,aid])})
d=mj.MjData(model);d.qpos[:]=z['qpos'][step];d.qvel[:]=z['qvel'][step];d.ctrl[:]=z['ctrl'][step]
if model.neq:model.eq_data[:]=z['eq_data'][step];d.eq_active[:]=z['eq_active'][step]
mj.mj_forward(model,d)
effective=[]
for i in range(d.ncon):
    q=d.contact[i]
    if set([int(q.geom1),int(q.geom2)])==set(gids.tolist()):
        effective.append({'distance_m':float(q.dist),'solref':np.asarray(q.solref).tolist(),'solimp':np.asarray(q.solimp).tolist(),'dim':int(q.dim)})
report={'schema':'rgb_near.frozen_hand_collider_diagnostic.v1','trial':str(a.trial.resolve()),'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'worst_hand_contact':{'interval_index':step,'time_s':float(c['solver_time_s'][worst]),'geom_pair':[geom(int(g))for g in gids],
                              'signed_distance_m':float(c['signed_distance_m'][worst]),'normal_force_N':float(c['normal_force_N'][worst]),
                              'world_force_on_geom2_N':c['force_world_on_geom2_N'][worst].tolist(),'effective_contact_params':effective},
        'anatomical_branch':branch,'finger_body_properties':bodies,'drivers_at_worst_contact':drivers,
        'active_collision_geom_count_in_branch':sum(len(b['active_collision_geoms'])for b in bodies),
        'duplicate_contact_interpretation':'Visual geoms with contype=conaffinity=0 are excluded. Multiple contact points from one capsule/box pair are solver manifold points, not duplicate geoms.',
        'read_only':True}
if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
