"""Export states, motor torques and measured contact wrenches from accepted controls."""
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np

from audit_dynamics_feasibility import required_force
from common import classify_robot_static_bodies, configure_position_pd, load_canonical, sha256, write_json


def main():
    root=Path(sys.argv[1]).resolve()
    variant=sys.argv[2] if len(sys.argv)>2 else 'sbto_refine'
    prefix='accepted_dynamic_window' if variant=='sbto_refine' else 'dynamic_window_self_guard'
    spec=json.loads((root/'window_0_80/experiment.json').read_text())
    report=json.loads((root/(variant+'_replay_1')/'report.json').read_text())
    if not report['passed']:raise ValueError('Export requires a passing strict replay')
    model=mj.MjModel.from_binary_path(spec['model'])
    actuation=configure_position_pd(model,mj)['actuators']
    model.opt.timestep=.005
    ref=load_canonical(root/'window_0_80/reference.npz')
    with np.load(root/variant/'controls.npz',allow_pickle=False) as archive:controls=archive['pd_target']
    robot,static,_=classify_robot_static_bodies(model,mj)
    weight=float(model.body_mass[list(robot)].sum()*np.linalg.norm(model.opt.gravity))
    feet=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'robot/'+side+'_ankle_roll_link') for side in ('left','right')]
    data=mj.MjData(model);data.qpos[:]=ref['qpos_ref'][0];data.qvel[:]=ref['qvel_ref'][0]
    mj.mj_forward(model,data)
    q=[data.qpos.copy()];v=[data.qvel.copy()]
    acceleration=[];torques=[];wrenches=[];foot_positions=[];residual=[];self_depth=[]
    contact_rows=[];offsets=[0];foot_supported=[]
    local=np.zeros(6)
    for step,u in enumerate(controls):
        data.ctrl[:]=u;mj.mj_step(model,data)
        acceleration.append(data.qacc.copy());torques.append(data.actuator_force.copy())
        q.append(data.qpos.copy());v.append(data.qvel.copy())
        foot_positions.append(data.xpos[feet].copy())
        wrench=np.zeros((2,6));total_force=np.zeros(3);penetration=0.
        for cid,c in enumerate(data.contact):
            a,b=model.geom_bodyid[[c.geom1,c.geom2]]
            if a in robot and b in robot:penetration=max(penetration,-float(c.dist))
            if not ((a in robot and b in static) or (b in robot and a in static)):continue
            mj.mj_contactForce(model,data,cid,local)
            sign=1. if b in robot else -1.
            rotation=c.frame.reshape(3,3)
            force=sign*rotation.T@local[:3];moment=sign*rotation.T@local[3:]
            total_force+=force
            body=int(b if b in robot else a)
            if body in feet:
                side=feet.index(body)
                wrench[side,:3]+=force
                wrench[side,3:]+=moment+np.cross(c.pos-data.xpos[body],force)
            contact_rows.append(np.r_[step,c.geom1,c.geom2,c.dist,c.pos,c.frame,local.copy()])
        # Free-base translation is world-frame: self/limit reactions cannot add net force.
        np.testing.assert_allclose(total_force,data.qfrc_constraint[:3],atol=1e-7,rtol=1e-9)
        wrenches.append(wrench);offsets.append(len(contact_rows));self_depth.append(penetration)
        foot_supported.append(wrench[:,2].sum()>.1*weight)
        residual.append(required_force(model,data,data.qacc)-data.qfrc_actuator-data.qfrc_constraint)
    q,v=np.array(q),np.array(v)
    with np.load(root/(variant+'_replay_1')/'physical_record.npz',allow_pickle=False) as expected:
        np.testing.assert_array_equal(q[::4],expected['qpos'])
        np.testing.assert_array_equal(v[::4],expected['qvel'])
    output=root/(prefix+'.npz')
    np.savez_compressed(output,time_s=np.arange(len(q))*.005,qpos=q,qvel=v,
        qacc=np.array(acceleration),actuator_force_Nm=np.array(torques),pd_target=controls,
        contact_wrench_world_at_foot_origin=np.array(wrenches),foot_origin_world=np.array(foot_positions),
        contact_rows=np.array(contact_rows),contact_step_offsets=np.array(offsets),
        dynamics_residual=np.array(residual),self_penetration_m=np.array(self_depth),
        joint_names=ref['joint_names'],actuator_joint_names=np.array([a['joint_name'] for a in actuation]),
        source_frame_range=np.array([0,80]),source_dt=.02,physics_dt=.005)
    write_json(root/(prefix+'.json'),dict(output_sha256=sha256(output),
        reference_sha256=spec['reference_sha256'],model_sha256=spec['model_sha256'],
        controls_sha256=report['controls_sha256'],code_sha256=sha256(__file__),
        source_frames=81,duration_s=1.6,physics_steps=len(controls),
        state_clock='q/v at boundaries; a/tau/wrench at the beginning of each step',
        torque_order='actuator_joint_names; differs from qpos joint order',
        contact_columns='step, geom1, geom2, distance, position_world[3], frame[9], local_force_torque[6]',
        contact_wrench_order='left/right, [Fx,Fy,Fz,Mx,My,Mz], world axes at foot body origin',
        fresh_replay_states_equal=True,foot_vertical_support_fraction=float(np.mean(foot_supported)),
        self_penetration_max_m=float(max(self_depth)),
        self_penetration_5mm_passed=bool(max(self_depth)<=.005),
        dynamics_residual_max_abs=float(np.max(np.abs(residual))),
        simulation_only=True,full_clip_validated=False))


if __name__=='__main__':main()
