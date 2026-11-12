"""CPU-only paired held-out feedforward baseline with unmodified200Hz controls.

Quality gates are executed from the existing replay evaluator's AST, preserving
its thresholds and definitions rather than introducing a second acceptance rule.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,os,shutil,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def load_original_gate(source):
    tree=ast.parse(Path(source).read_text());main=next(node for node in tree.body if isinstance(node,ast.FunctionDef)and node.name=='main')
    named=lambda node,name:isinstance(node,ast.Assign)and any(isinstance(target,ast.Name)and target.id==name for target in node.targets)
    start=next(i for i,node in enumerate(main.body)if named(node,'desired'))
    end=next(i for i,node in enumerate(main.body[start:],start)if isinstance(node,ast.Expr)and isinstance(node.value,ast.Call)
             and isinstance(node.value.func,ast.Attribute)and node.value.func.attr=='update'
             and isinstance(node.value.func.value,ast.Name)and node.value.func.value.id=='functional_checks')
    nodes=[node for node in main.body[start:end+1]if not named(node,'replay')]
    block=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    return compile(block,str(source),'exec'),ast.unparse(block)


def evaluate_reference(model,core,raw,replay,compiled_gate,np,interp,mapping):
    names=list(map(str,raw['dof_names']));maps=[mapping(model,names,prefix)for prefix in ('p1_','p2_')]
    ref=SimpleNamespace(names=names,maps=maps,root_qadr=np.asarray([mp['rq']for mp in maps]),
        root_vadr=np.asarray([mp['rv']for mp in maps]),end=float(raw['time_s'][-1]),
        body_cols=np.asarray([i for i,name in enumerate(names)if not name.startswith(('lh_','rh_','neck_'))]))
    ref.horizon=ref.end+1.
    scope={'np':np,'interp':interp,'m':model,'raw':raw,'ref':ref,'replay':replay,
        'qs':core['qpos'],'vs':core['qvel'],'ts':core['time'],'controls':core['ctrl'],
        'seats':core['seat_contact'],'sf':core['seat_support_force_N'],'pens':list(core['max_penetration_m']),
        'reason':core['termination']}
    exec(compiled_gate,scope,scope)
    result={'success':bool(all(scope['checks'].values())),
        'functional_task_success':bool(all(scope['functional_checks'].values())),
        'checks':scope['checks'],'functional_task_checks':scope['functional_checks'],
        'simulated_seconds':float(core['time'][-1]),'physics_steps':len(core['ctrl']),
        'source_duration_seconds':float(raw['source_time_s'][-1]),'execution_reference_duration_seconds':ref.end,
        'root_xy_rms_m':float(np.sqrt(np.mean(scope['xy']**2))),
        'max_root_xy_error_m':scope['xy'].max(0).tolist(),'max_root_z_error_m':scope['ze'].max(0).tolist(),
        'max_root_orientation_error_rad':scope['oe'].max(0).tolist(),
        'body_joint_rms_median_rad':np.median(scope['je'],axis=0).tolist(),
        'max_penetration_m':float(max(core['max_penetration_m'],default=0.)),
        'final_root_xy_error_m':scope['xy'][-1].tolist(),
        'final_0p2s_seat_contact_fraction':scope['fraction'].tolist(),
        'final_seat_vertical_force_mean_N':scope['final_seat_force'].tolist(),
        'final_root_linear_speed_max_m_s':scope['final_linear_speed'].tolist(),
        'final_root_angular_speed_max_rad_s':scope['final_angular_speed'].tolist(),
        'root_height_min_m':scope['root_height'].min(0).tolist(),
        'root_global_tilt_max_rad':scope['root_tilt'].max(0).tolist(),
        'exchanged_distance_m':scope['exchanged_distance'].tolist(),'robot_weight_N':scope['weight'].tolist(),
        'independent_replay':replay,'termination':core['termination']}
    return result,scope


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--control-trial',type=Path)
    p.add_argument('--tag',default='feedforward_baseline')
    a=p.parse_args()
    import numpy as np,mujoco as mj
    from two_sitting_physics import contacts,interp
    from two_sitting_ppo_env import mapping
    root=a.root/'heldout_seed990013';output=root/a.tag
    if(output/'summary.json').exists():raise FileExistsError('Frozen held-out baseline already exists')
    output.mkdir(exist_ok=True)
    model_path=a.root/'calibration/two_vita_expert.mjb';model=mj.MjModel.from_binary_path(str(model_path))
    replay_model=mj.MjModel.from_binary_path(str(model_path))
    heldout_path=root/'initial_states.npz';heldout=dict(np.load(heldout_path,allow_pickle=False))
    if int(heldout['seed'])!=990013 or not bool(heldout['evaluation_only']):raise ValueError('Wrong held-out contract')
    if str(heldout['source_model_sha256'])!=sha(model_path):raise ValueError('Frozen held-out model SHA differs')
    spec=mj.mjtState(int(heldout['state_spec']));states=heldout['initial_states']
    actual_path=(a.control_trial/'trace.npz' if a.control_trial else
                 a.root/'prior_physical_state_pool/candidate156_actual_states_controls.npz')
    actual=dict(np.load(actual_path,allow_pickle=False));control=actual['ctrl']
    if not np.array_equal(states[0],actual['initial_state']):raise ValueError('Nominal source initial state differs from frozen pairedcase0')
    if control.shape!=(1912,104):raise ValueError('Must use unchanged old1912x104 control profile')
    ref_paths={'phc_time1p5':a.root/'references/phc_time_scale1.5.npz',
        'old_original_reference':Path('/data/KinoGen/outputs/two_sitting_pipeline/reference_v3_deploy_slow.npz')}
    refs={key:dict(np.load(path,allow_pickle=False))for key,path in ref_paths.items()}
    source=Path('/data/KinoGen/scripts/two_sitting_replay_candidate.py')
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True);shutil.copy2(source,snapshot/source.name)
    shutil.copy2(__file__,snapshot/Path(__file__).name)
    compiled,gate_text=load_original_gate(snapshot/source.name)
    (snapshot/'extracted_unmodified_metric_and_gate_block.py').write_text(gate_text+'\n')
    names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or str(g)for g in range(model.ngeom)]
    bodies=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,b)or str(b)for b in range(model.nbody)]
    roots=np.asarray(actual['robot_root_qadr'],int)
    def run_case(case):
        data=mj.MjData(model);mj.mj_setState(model,data,states[case],spec);mj.mj_forward(model,data)
        if np.any(data.qfrc_applied)or np.any(data.xfrc_applied):raise ValueError('External initial wrench')
        times=[float(data.time)];qs=[data.qpos.copy()];vs=[data.qvel.copy()]
        seat=[];foot=[];force=[];penetration=[];interperson=[];allrows=[];drive=[];torque=[]
        reason='completed';warnings=np.asarray([w.number for w in data.warning]);start=time.monotonic()
        for step,packet in enumerate(control):
            data.ctrl[:]=packet;mj.mj_step(model,data)
            rows,sc,fc,sf,pen,ip=contacts(model,data,step,names,bodies)
            times.append(float(data.time));qs.append(data.qpos.copy());vs.append(data.qvel.copy())
            seat.append(sc);foot.append(fc);force.append(sf);penetration.append(pen);interperson.append(ip)
            if case==0:allrows.extend(rows);drive.append(data.actuator_force.copy());torque.append(data.qfrc_actuator.copy())
            if np.any(data.qfrc_applied)or np.any(data.xfrc_applied):raise ValueError('External force during replay')
            if (not np.isfinite(data.qpos).all()or not np.isfinite(data.qvel).all()or
                    np.any(np.asarray([w.number for w in data.warning])>warnings)):
                reason='numeric_failure';break
            if (step+1)%4==0 and any(data.qpos[rq+2]<.15 for rq in roots):reason='root_below_0p15m';break
        q,v,t=np.asarray(qs),np.asarray(vs),np.asarray(times);count=len(t)-1
        core={'qpos':q,'qvel':v,'time':t,'ctrl':control[:count],
            'seat_contact':np.asarray(seat),'foot_contact':np.asarray(foot),'seat_support_force_N':np.asarray(force),
            'max_penetration_m':np.asarray(penetration),'max_interperson_penetration_m':np.asarray(interperson),
            'termination':reason}
        # Fresh MjData on a separately loaded frozen model, exactly the same ctrl-only replay criterion.
        replay_data=mj.MjData(replay_model);mj.mj_setState(replay_model,replay_data,states[case],spec);mj.mj_forward(replay_model,replay_data)
        qe=float(np.max(abs(replay_data.qpos-q[0])));ve=float(np.max(abs(replay_data.qvel-v[0])))
        for step,packet in enumerate(control[:count]):
            replay_data.ctrl[:]=packet;mj.mj_step(replay_model,replay_data)
            qe=max(qe,float(np.max(abs(replay_data.qpos-q[step+1]))));ve=max(ve,float(np.max(abs(replay_data.qvel-v[step+1]))))
        replay={'passed':bool(qe<=1e-9 and ve<=1e-9),'max_qpos_absolute_error':qe,'max_qvel_absolute_error':ve,
            'steps':count,'method':'Separately loaded frozen MJB, fresh MjData, complete same integration init, recorded ctrl+native mj_step only.'}
        evaluations={};scopes={}
        for name,raw in refs.items():evaluations[name],scopes[name]=evaluate_reference(model,core,raw,replay,compiled,np,interp,mapping)
        item={'case':case,'heldout_seed':990013,'reference_metrics':evaluations,
            'state_override_after_reset_count':0,'external_force_steps':0,'wall_seconds':time.monotonic()-start}
        case_dir=output/f'case_{case:03d}';case_dir.mkdir(exist_ok=True)
        (case_dir/'metrics.json').write_text(json.dumps(item,indent=2)+'\n')
        if case==0:
            raw=refs['phc_time1p5'];scope=scopes['phc_time1p5']
            fields={key:value for key,value in core.items()if key!='termination'}
            fields.update(initial_state=states[case],state_spec=int(spec),joint_names=actual['joint_names'],
                joint_qadr=actual['joint_qadr'],joint_vadr=actual['joint_vadr'],actuator_ids=actual['actuator_ids'],
                robot_root_qadr=roots,robot_root_vadr=actual['robot_root_vadr'],
                source_time_s=np.interp(t,raw['time_s'],raw['source_time_s']),reference_qpos=scope['desired'],
                root_xy_error_m=scope['xy'],root_z_error_m=scope['ze'],root_orientation_error_rad=scope['oe'],
                actuator_force=np.asarray(drive),qfrc_actuator=np.asarray(torque),geom_names=np.asarray(names),body_names=np.asarray(bodies))
            for key,i,shape,dtype in [('interval_index',0,(),int),('time_s',1,(),float),('geom_ids',2,(2,),int),
                ('body_ids',3,(2,),int),('position_world_m',4,(3,),float),('force_world_on_geom2_N',5,(3,),float),
                ('torque_world_on_geom2_Nm',6,(3,),float),('normal_force_N',7,(),float),('signed_distance_m',8,(),float)]:
                fields['contact_'+key]=np.asarray([row[i]for row in allrows],dtype=dtype).reshape((-1,)+shape)
            np.savez_compressed(case_dir/'trace.npz',**fields)
            shutil.copy2(model_path,case_dir/'model.mjb')
            (case_dir/'report.json').write_text(json.dumps(dict(evaluations['phc_time1p5'],
                controller={'method':('CEM-refined physical200Hz feedforward profile; zero new feedback; no PPO inference'if a.control_trial else'Archived physical200Hz feedforward profile; exactly zero new feedback; no policy inference')},
                inputs={'initial_states':str(heldout_path),'initial_states_sha256':sha(heldout_path),
                        'control_source':str(actual_path),'control_source_sha256':sha(actual_path),
                        'reference':str(ref_paths['phc_time1p5'])},
                state_override_after_reset_count=0,external_force_steps=0,mocap_bodies=0,root_actuators=0),indent=2)+'\n')
            if len(q)==len(actual['qpos']):
                item['case0_source_trace_match']={'qpos_max_abs':float(np.max(abs(q-actual['qpos']))),
                    'qvel_max_abs':float(np.max(abs(v-actual['qvel'])))}
        print(json.dumps({'case':case,'physics_steps':count,'termination':reason,
            'phc_functional':evaluations['phc_time1p5']['functional_task_success'],
            'old_reference_functional':evaluations['old_original_reference']['functional_task_success'],
            'replay':replay['passed']}),flush=True)
        return item,core
    began=time.monotonic()
    if a.control_trial:
        first=run_case(0)
        if (not first[0]['reference_metrics']['phc_time1p5']['functional_task_success']or
            max(first[0]['case0_source_trace_match'].values())!=0.):
            raise RuntimeError('CEM case0 failed exact-source reproduction or PHC functional check; stopped before perturbed cases')
        with ThreadPoolExecutor(max_workers=a.threads)as executor:results=[first,*list(executor.map(run_case,range(1,len(states))))]
    else:
        with ThreadPoolExecutor(max_workers=a.threads)as executor:results=list(executor.map(run_case,range(len(states))))
    summary={'schema':'sonic_expert.paired_heldout_zero_feedback.v1','status':'full200hz_heldout_feedforward_baseline_complete',
        'seed':990013,'cases':len(states),'training_use_prohibited':True,'heldout_states':str(heldout_path),
        'heldout_states_sha256':sha(heldout_path),'model':str(model_path),'model_sha256':sha(model_path),
        'actual_control_source':str(actual_path),'actual_control_source_sha256':sha(actual_path),
        'control_source_kind':'functional_CEM_motor_plan_not_PPO'if a.control_trial else'old_archived_functional_control_profile',
        'reference_paths':{key:str(path)for key,path in ref_paths.items()},
        'reference_sha256s':{key:sha(path)for key,path in ref_paths.items()},
        'gate_source':str(source),'gate_source_sha256':sha(source),'unmodified_gate_extraction':'Existing evaluator AST block from desired-state interpolation through functional_checks.update; replay result independently supplied; no threshold edits.',
        'no_extra_nominal_delay':True,'policy_inference':False,'learned_feedback':False,
        'state_override_after_reset_count':0,'external_force_steps':0,'physical_dt_s':float(model.opt.timestep),
        'requested_control_steps':len(control),'requested_duration_s':len(control)*float(model.opt.timestep),
        'benchmark_source_sha256':sha(__file__),
        'wall_seconds':time.monotonic()-began,'references':{},'results':[item for item,_ in results]}
    for key in refs:
        reports=[item['reference_metrics'][key]for item,_ in results]
        summary['references'][key]={'functional_passes':sum(r['functional_task_success']for r in reports),
            'strict_passes':sum(r['success']for r in reports),'no_fall_cases':sum(r['functional_task_checks']['no_fall']for r in reports),
            'root_xy_rms_mean_m':float(np.mean([r['root_xy_rms_m']for r in reports])),
            'case0':reports[0]}
    shape=(len(states),len(control)+1)
    qpos=np.full((*shape,model.nq),np.nan);qvel=np.full((*shape,model.nv),np.nan)
    state_valid=np.zeros(shape,bool);force=np.full((len(states),len(control),2),np.nan)
    seat=np.zeros((len(states),len(control),2),bool);penetration=np.full((len(states),len(control)),np.nan)
    for case,(_,core)in enumerate(results):
        n=len(core['time']);qpos[case,:n]=core['qpos'];qvel[case,:n]=core['qvel'];state_valid[case,:n]=True
        force[case,:n-1]=core['seat_support_force_N'];seat[case,:n-1]=core['seat_contact'];penetration[case,:n-1]=core['max_penetration_m']
    core_path=output/'all_cases200hz_core.npz'
    np.savez_compressed(core_path,qpos=qpos,qvel=qvel,state_valid=state_valid,
        time_s=actual['time'],ctrl_profile=control,seat_support_force_N=force,seat_contact=seat,
        max_penetration_m=penetration,heldout_case_ids=np.arange(len(states)))
    summary['all_cases_core_arrays']={'path':str(core_path),'sha256':sha(core_path),'bytes':core_path.stat().st_size,
        'note':'Full200Hz q/v and seat-force/contact/penetration for all cases; state_valid marks early termination and NaN padding is not data.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'summary':str(output/'summary.json'),'references':{key:{k:v for k,v in data.items()if k!='case0'}for key,data in summary['references'].items()},'wall_seconds':summary['wall_seconds']}),flush=True)


if __name__=='__main__':main()
