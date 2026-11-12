"""CEM diagnostic: refine late leg-motor commands in real physics, without root forcing.

Historical successful controls provide the walking prefix. This is an explicit
motor-plan search auxiliary to PPO, not SUGAR's official RL implementation.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    p.add_argument('--generations',type=int,default=12);p.add_argument('--population',type=int,default=64)
    p.add_argument('--threads',type=int,default=12)
    p.add_argument('--output',type=Path)
    p.add_argument('--start-report',type=Path)
    p.add_argument('--freeze-p1',action='store_true')
    p.add_argument('--max-joint-offset',type=float,default=.22)
    a=p.parse_args()
    import numpy as np,mujoco as mj
    from two_sitting_physics import contacts
    root=a.root;out=a.output or root/'terminal_motor_search';out.mkdir(exist_ok=True)
    modelpath=root/'calibration/two_vita_expert.mjb';model=mj.MjModel.from_binary_path(str(modelpath))
    tracepath=Path('/data/KinoGen/outputs/two_sitting_pipeline/physics_selected450_candidate156/trace.npz')
    old=np.load(tracepath,allow_pickle=False);base=old['ctrl'].copy();initial=old['initial_state'].copy()
    spec=mj.mjtState(int(old['state_spec']));reference=root/'references/phc_transfer_time1p5.npz'
    if not reference.exists():
        z=dict(np.load(root/'references/phc.npz'));z['time_s']*=1.5;np.savez_compressed(reference,**z)
    target=np.load(reference)['qpos'][-1,:,:2]
    rq=np.asarray(old['robot_root_qadr']);rv=np.asarray(old['robot_root_vadr'])
    bn=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,i)or''for i in range(model.nbody)]
    gn=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,i)or''for i in range(model.ngeom)]
    legnames=[side+'_'+name+'_joint'for side in ('left','right')for name in
              ('hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll')]
    jids=np.asarray([[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,prefix+n)for n in legnames]
                     for prefix in ('p1_','p2_')])
    aids=np.asarray([[mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,prefix+'motor_'+n)for n in legnames]
                     for prefix in ('p1_','p2_')])
    if min(jids.flat)<0 or min(aids.flat)<0:raise ValueError('Missing legs')
    feet=[[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+side+'_ankle_roll_link')for side in ('left','right')]
           for prefix in ('p1_','p2_')]
    # Numerical kinematic sensitivity only; this scratch root is never the live plant.
    basis=np.zeros((len(base),2,12,2));scratch=mj.MjData(model)
    for k in range(0,len(base),4):
        scratch.qpos[:]=old['qpos'][k];mj.mj_kinematics(model,scratch);mj.mj_comPos(model,scratch)
        for actor in range(2):
            rows=[]
            for foot in feet[actor]:
                pos=np.zeros((3,model.nv));rot=np.zeros_like(pos);mj.mj_jacBody(model,scratch,pos,rot,foot)
                rows.append(np.concatenate([pos[:,model.jnt_dofadr[jids[actor]]],.15*rot[:,model.jnt_dofadr[jids[actor]]]],axis=0))
            jac=np.concatenate(rows,axis=0);goal=np.concatenate([np.r_[-np.eye(3)[:,:2],np.zeros((3,2))]]*2,axis=0)
            b=np.linalg.solve(jac.T@jac+.003*np.eye(12),jac.T@goal)
            basis[k:k+4,actor]=b
    cut=1260
    data=mj.MjData(model);mj.mj_setState(model,data,initial,spec);mj.mj_forward(model,data)
    for control in base[:cut]:data.ctrl[:]=control;mj.mj_step(model,data)
    seed=np.empty(mj.mj_stateSize(model,spec));mj.mj_getState(model,data,seed,spec)
    prefix_error=float(np.max(np.abs(data.qpos-old['qpos'][cut])))
    if prefix_error>1e-9:raise RuntimeError('Historical prefix differs')
    weights=np.asarray([model.body_subtreemass[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,p+'base_link')]*9.81 for p in ('p1_','p2_')])
    gate_t=np.arange(len(base))*.005
    gate=np.clip((gate_t-7.15)/1.25,0,1);gate=gate*gate*(3-2*gate)
    history=[];rng=np.random.default_rng(174200)
    lower=model.jnt_range[jids,0];upper=model.jnt_range[jids,1]
    def plan(parameters):
        result=base.copy();offset=np.einsum('taij,aj->tai',basis,parameters.reshape(2,2))
        offset=np.clip(offset,-a.max_joint_offset,a.max_joint_offset)*gate[:,None,None]
        delayed=np.zeros_like(offset);delayed[3:]=offset[:-3]
        active=np.flatnonzero(np.any(delayed!=0,axis=(1,2)))
        if len(active):
            selection=np.ix_(active,aids.reshape(-1))
            result[selection]=np.clip(result[selection]+delayed[active].reshape(len(active),-1),
                                      lower.reshape(1,-1),upper.reshape(1,-1))
        if not np.array_equal(result[:cut],base[:cut]):
            raise RuntimeError('Motor plan modified the protected historical prefix')
        return result
    def evaluate(parameters):
        controls=plan(parameters);d=mj.MjData(model);mj.mj_setState(model,d,seed,spec);mj.mj_forward(model,d)
        forces=[];seats=[];linear=[];angular=[];pen=0.;tilt=0.;height=10.;bad=False
        for step in range(cut,len(controls)):
            d.ctrl[:]=controls[step];mj.mj_step(model,d)
            if np.any([w.number for w in d.warning]):bad=True;break
            if d.ncon:pen=max(pen,max(0.,float(-np.min(np.asarray(d.contact.dist)[:d.ncon]))))
            if step%4==0:
                tilt=max(tilt,float(np.arccos(np.clip(1-2*(d.qpos[rq+4]**2+d.qpos[rq+5]**2),-1,1)).max()))
                height=min(height,float(d.qpos[rq+2].min()))
            if step>=len(controls)-40:
                _,seat,_,force,_,_=contacts(model,d,step,gn,bn);forces.append(force);seats.append(seat)
                linear.append([np.linalg.norm(d.qvel[v:v+3])for v in rv]);angular.append([np.linalg.norm(d.qvel[v+3:v+6])for v in rv])
        if bad or not forces:return {'score':-1e6,'parameters':parameters.tolist(),'numerical_fault':True}
        xy=np.linalg.norm(np.asarray([d.qpos[r:r+2]for r in rq])-target,axis=-1)
        force=np.mean(forces,axis=0);fraction=np.mean(seats,axis=0)
        speed=np.max(linear,axis=0);spin=np.max(angular,axis=0);load=force/weights
        valid=bool(height>.45 and tilt<.8 and pen<.025 and np.all(fraction>=.5)and np.all(load>=.1)
                   and np.all(speed<.15)and np.all(spin<.3))
        score=(-30*float(np.sum(xy**2))-15*float(np.sum(speed**2))-2*float(np.sum(spin**2))+
               2*float(np.minimum(load,1.).sum())-200*max(pen-.025,0)-
               10*float(np.maximum(.1-load,0).sum())-8*float(np.maximum(.5-fraction,0).sum()))
        if height<=.45 or tilt>=.8:score-=100
        score+=500*int(valid)+5000*int(valid and xy.max()<.3)-100*float(np.maximum(xy-.285,0).sum())
        return {'score':score,'parameters':parameters.tolist(),'final_xy_m':xy.tolist(),
            'force_N':force.tolist(),'contact_fraction':fraction.tolist(),'final_speed_m_s':speed.tolist(),
            'final_spin_rad_s':spin.tolist(),'max_penetration_m':pen,'max_tilt':tilt,'min_height_m':height,
            'late_physical_gate':valid,'endpoint_candidate_pass':bool(valid and xy.max()<.3)}
    center=np.zeros(4);std=np.full(4,.055);baseline=evaluate(center)
    if a.start_report:
        center=np.asarray(json.loads(a.start_report.read_text())['best']['parameters'])
        std[:]=.015
    best=evaluate(center);best_parameters=center.copy();frozen_p1=center[:2].copy();began=time.monotonic()
    with ThreadPoolExecutor(max_workers=a.threads)as executor:
        for generation in range(a.generations):
            candidates=np.clip(rng.normal(center,std,(a.population,4)),-.16,.16)
            if a.freeze_p1:candidates[:,:2]=frozen_p1
            candidates[0]=best_parameters
            if a.start_report and generation==0:
                for index,dy in enumerate((-.025,-.020,-.015,-.010,-.005,.005,.010,.015),start=1):
                    candidates[index]=center;candidates[index,3]=np.clip(center[3]+dy,-.16,.16)
            if generation==0 and not a.start_report:
                delta=np.clip(target-np.asarray([old['qpos'][-1,r:r+2]for r in rq]),-.13,.13).reshape(-1)
                for i,scale in enumerate((.3,.5,.8,1.),start=1):candidates[i]=scale*delta
            results=list(executor.map(evaluate,candidates));results.sort(key=lambda x:x['score'],reverse=True)
            if results[0]['score']>best['score']:best=results[0];best_parameters=np.asarray(best['parameters'])
            elite=np.asarray([x['parameters']for x in results[:max(6,a.population//5)]])
            center=.25*center+.75*elite.mean(0);std=np.maximum(.007,.25*std+.75*elite.std(0))
            row={'generation':generation+1,'best':best,'mean':center.tolist(),'std':std.tolist(),'wall_s':time.monotonic()-began}
            history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(row),flush=True)
            controls=plan(best_parameters)
            metadata={'method':'CEM late leg-motor plan search diagnostic; not PPO expert',
                'model_sha256':sha(modelpath),'reference_sha256':sha(reference),'old_trace_sha256':sha(tracepath),
                'old_nominal_physical_success_inherited':True,'search_parameters':best_parameters.tolist(),
                'live_root_writes':0,'new_offset_delay_physics_steps':3,'nominal_historical_delay_not_reapplied':True}
            np.savez_compressed(out/'candidate.npz',ctrl=controls,initial_state=initial,state_spec=int(spec),metadata_json=np.asarray(json.dumps(metadata)))
            if best.get('endpoint_candidate_pass'):
                break
    report={'status':'motor_plan_search_finished_pending_full_rollout','baseline':baseline,'best':best,
        'model_sha256':sha(modelpath),'old_trace_sha256':sha(tracepath),'prefix_qpos_error':prefix_error,
        'frozen_prefix_steps':cut,'reference':str(reference),'generations':len(history),
        'source_sha256':sha(__file__),'wall_seconds':time.monotonic()-began,
        'score_version':'physical and endpoint constraint first; root error beyond.285m explicitly penalized',
        'maximum_added_joint_target_offset_rad':a.max_joint_offset,
        'scope':'Real native physics from validated old prefix state; candidate must still run fromtime0 with full unchanged acceptance and independent replay.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
