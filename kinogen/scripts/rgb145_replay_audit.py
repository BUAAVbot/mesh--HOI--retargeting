"""Independently replay saved controls in the frozen native MuJoCo plant."""
from pathlib import Path
import argparse,hashlib,json,time


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);a=p.parse_args()
    import numpy as np,mujoco as mj
    r=json.loads((a.trial/'report.json').read_text());mp=a.trial/'model.mjb';tp=a.trial/'trace.npz'
    if sha(mp)!=r['model']['sha256']or sha(tp)!=r['trace']['sha256']:raise ValueError('Model/trace SHA mismatch')
    if mj.__version__!=r['mujoco_version']:raise ValueError('Replay must use the original MuJoCo version')
    z=dict(np.load(tp,allow_pickle=False));co=dict(np.load(a.trial/'contacts.npz',allow_pickle=False))
    m=mj.MjModel.from_binary_path(str(mp));d=mj.MjData(m);spec=int(z['state_spec'])
    mj.mj_setState(m,d,z['initial_state'],spec);mj.mj_forward(m,d)
    maximum={'qpos':float(np.max(abs(d.qpos-z['qpos'][0]))),'qvel':float(np.max(abs(d.qvel-z['qvel'][0]))),'actuator_force':0.,'qfrc_constraint':0.,'eq_generalized_force':0.,'contact_position':0.,'contact_force':0.,'contact_torque':0.,'contact_distance':0.}
    checks={'zero_external_forces':bool(not np.any(z['qfrc_applied'])and not np.any(z['xfrc_applied'])),'initial_time_matches':abs(d.time-z['time'][0])<1e-12,'contact_ids_and_counts':True,'contact_time_is_interval_start':bool(np.allclose(co['solver_time_s'],z['time'][co['interval_index']],atol=1e-8))}
    steps=len(z['ctrl']);wall=time.monotonic();offsets=np.searchsorted(co['interval_index'],np.arange(steps+1))
    def update(name,x,y):maximum[name]=max(maximum[name],float(np.max(abs(np.asarray(x)-np.asarray(y))))if np.size(x)else 0.)
    for i in range(steps):
        m.eq_data[:]=z['eq_data'][i];d.eq_active[:]=z['eq_active'][i];d.ctrl[:]=z['ctrl'][i]
        d.qfrc_applied[:]=z['qfrc_applied'][i];d.xfrc_applied[:]=z['xfrc_applied'][i]
        mj.mj_step(m,d)
        update('qpos',d.qpos,z['qpos'][i+1]);update('qvel',d.qvel,z['qvel'][i+1]);update('actuator_force',d.actuator_force,z['actuator_force'][i]);update('qfrc_constraint',d.qfrc_constraint,z['qfrc_constraint'][i])
        eqforce=np.zeros((m.neq,m.nv));weights=np.zeros(d.nefc)
        for eq in range(m.neq):
            weights[:]=0.;mask=(d.efc_type[:d.nefc]==int(mj.mjtConstraint.mjCNSTR_EQUALITY))&(d.efc_id[:d.nefc]==eq)
            weights[mask]=d.efc_force[mask];mj.mj_mulJacTVec(m,d,eqforce[eq],weights)
        update('eq_generalized_force',eqforce,z['eq_generalized_force'][i])
        left,right=offsets[i:i+2]
        if d.ncon!=right-left:checks['contact_ids_and_counts']=False
        else:
            for k in range(d.ncon):
                c=d.contact[k];row=left+k;force=np.zeros(6);mj.mj_contactForce(m,d,k,force);R=np.asarray(c.frame).reshape(3,3)
                if not np.array_equal([c.geom1,c.geom2],co['geom_ids'][row]):checks['contact_ids_and_counts']=False
                update('contact_position',c.pos,co['position_world_m'][row]);update('contact_force',R.T@force[:3],co['force_world_on_geom2_N'][row]);update('contact_torque',R.T@force[3:],co['torque_world_on_geom2_Nm'][row]);update('contact_distance',c.dist,co['signed_distance_m'][row])
    tolerance={k:(1e-8 if k in ('qpos','qvel','contact_position','contact_distance')else 1e-5)for k in maximum}
    checks.update({k+'_reproduced':v<=tolerance[k]for k,v in maximum.items()})
    checks={k:bool(v)for k,v in checks.items()}
    out={'schema':'rgb145.independent_control_replay.v1','replay_passed':all(checks.values()),'physics_task_passed':r['simulation_passed'],'checks':checks,'maximum_absolute_error':maximum,'tolerance':tolerance,'steps':steps,'duration_s':float(d.time),'model_sha256':sha(mp),'trace_sha256':sha(tp),'mujoco_version':mj.__version__,'elapsed_wall_s':time.monotonic()-wall,'meaning':'Replay proves saved trajectory/forces follow the frozen dynamics and actual actuator commands. Task success is a separate gate.'}
    path=a.trial/'independent_replay.json';path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
    return 0 if out['replay_passed']else 3


if __name__=='__main__':raise SystemExit(main())
