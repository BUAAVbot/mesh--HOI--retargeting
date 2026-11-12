"""CPU-only virtual state probes; no dynamics, training, or controller edits."""
import os,sys,hashlib,json,math,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/data/KinoGen/outputs/sonic_expert_20261028')
SOURCE=Path('/data/KinoGen/scripts/direct_feedforward_v4_expert_20261028')
sys.path[:0]=[str(SOURCE),'/data/KinoGen/scripts']
import numpy as np,torch,mujoco as mj
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(2)
    checkpoint_path=ROOT/'expert_direct_feedforward_v4_tiny_actualstd/checkpoint_000050.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    os.environ['PHYSICAL_FEEDFORWARD_PRIOR']=checkpoint['physical_feedforward_prior']
    os.environ['DIRECT_INITIAL_STATE_PATH']=checkpoint['fixed_initial_state_path']
    os.environ['FF_TRAIN_INITIAL_NOISE']='0'
    from physical_feedforward_env import PhysicalFeedforwardEnv
    from two_sitting_ppo_env import ActorCritic
    env=PhysicalFeedforwardEnv(Path(checkpoint['model_path']),Path(checkpoint['reference_path']),n=1,seed=114991,
        action_scale=checkpoint['action_scale'],sonic_base=False,threads=1,device='cpu')
    actor=ActorCritic(checkpoint['obs_dim'],56);actor.load_state_dict(checkpoint['model']);actor.eval()
    prior=env.prior;out=ROOT/'audit/feedback_local_gain_checkpoint50';out.mkdir(parents=True,exist_ok=True)
    snap=out/'source_snapshot';snap.mkdir(exist_ok=True)
    sources=[SOURCE/'physical_feedforward_env.py',SOURCE/'direct_proprio.py',SOURCE/'sugar_expert_env.py',
        Path('/data/KinoGen/scripts/two_sitting_native_ppo_env.py'),Path('/data/KinoGen/scripts/two_sitting_ppo_env.py'),Path(__file__)]
    for p in sources:shutil.copyfile(p,snap/p.name)
    source_hashes={str(p):sha(p)for p in sources}
    rows=[];pairs=[];phases=[.5,3.,8.,9.]
    names=list(map(str,env.ref.names));knee=names.index('left_knee_joint');kneeqa=int(env.ref.maps[0]['qa'][knee]);kneeva=int(env.ref.maps[0]['va'][knee])
    probes=[('p1_root_x','q',int(env.ref.root_qadr[0])),('p1_root_y','q',int(env.ref.root_qadr[0])+1),
        ('p2_root_x','q',int(env.ref.root_qadr[1])),('p1_left_knee_q','q',kneeqa),
        ('p1_root_vx','v',int(env.ref.root_vadr[0])),('p1_left_knee_v','v',kneeva)]
    motor_names=[mj.mj_id2name(env.model,mj.mjtObj.mjOBJ_ACTUATOR,int(a))for a in env.ref.body_aids]
    lower=env.ref.lower.numpy();upper=env.ref.upper.numpy()
    for phase in phases:
        index=int(round(phase/.005));nomq=prior['qpos'][index].copy();nomv=prior['qvel'][index].copy();env.phase[:]=phase
        control_index=min(index+3,len(prior['ctrl'])-1);nominal64=prior['ctrl'][control_index,env.ref.body_aids].copy()
        def evaluate(name,kind,coordinate,epsilon):
            data=env.datas[0];data.qpos[:]=nomq;data.qvel[:]=nomv;data.time=phase
            if kind=='q':data.qpos[coordinate]+=epsilon
            if kind=='v':data.qvel[coordinate]+=epsilon
            env.previous[:]=0.;env.feedback_previous[:]=0.
            env._cache_one(0);env._generation+=1;env._host_generation=env._generation;env.refresh_policy_sensors()
            nominal=env.base_controls();obs=env.observations()
            with torch.inference_mode():raw=actor.actor(actor.normalize(obs));ungated=(env.action_scale*torch.tanh(raw))[0].numpy()
            packet=nominal.clone();packet[:,env.ref.body_aids_t]+=env.action_scale*torch.tanh(raw)
            packet[:,env.ref.body_aids_t]=packet[:,env.ref.body_aids_t].clamp(env.ref.lower,env.ref.upper)
            packet_delta=(packet[0,env.ref.body_aids_t].numpy()-env.nominal_policy_cpu[0,env.ref.body_aids]).copy()
            gain=float(env.feedback_gain[0]);feedback=packet_delta*gain
            actual=np.clip(nominal64+feedback,lower,upper)-nominal64 if np.any(feedback)else np.zeros(56)
            source=float(prior['source_time_s'][index]);temporal=.15+.85*np.clip((source-4.2)/.7,0,1)
            row={'phase_s':phase,'physics_index':index,'source_time_s':source,'probe':name,'kind':kind,'coordinate':coordinate,'epsilon':epsilon,
                'temporal_gate':float(temporal),'state_error_gate':gain/temporal,'feedback_gain':gain,'ungated_policy_offset_rad':ungated.tolist(),
                'packet_offset_before_gain_rad':packet_delta.tolist(),'gated_feedback_rad':feedback.tolist(),'actual_new_motor_delta_rad':actual.tolist(),
                'nominal_outside_joint_limits_count':int(np.count_nonzero((nominal64<lower)|(nominal64>upper))),
                'observation_norm':float(obs.norm()),'virtual_only_no_mj_step':True}
            rows.append(row);return row
        nominalrow=evaluate('nominal','',0,0.)
        for name,kind,coordinate in probes:
            previous=None
            for epsilon in [1e-5,1e-4,1e-3]:
                positive=evaluate(name,kind,coordinate,epsilon);negative=evaluate(name,kind,coordinate,-epsilon)
                pair={'phase_s':phase,'probe':name,'epsilon':epsilon,'positive_gate':positive['feedback_gain'],'negative_gate':negative['feedback_gain']}
                for key in ['ungated_policy_offset_rad','packet_offset_before_gain_rad','gated_feedback_rad','actual_new_motor_delta_rad']:
                    plus=np.asarray(positive[key]);minus=np.asarray(negative[key]);base=np.asarray(nominalrow[key]);odd=(plus-minus)/2;even=(plus+minus)/2-base
                    entry={'odd':odd.tolist(),'even_minus_nominal':even.tolist(),'odd_l2':float(np.linalg.norm(odd)),'even_l2':float(np.linalg.norm(even)),
                           'odd_slope_l2_per_epsilon':float(np.linalg.norm(odd)/epsilon),'even_slope_l2_per_epsilon':float(np.linalg.norm(even)/epsilon)}
                    if previous and previous[key]['odd_l2']>0 and entry['odd_l2']>0:entry['odd_observed_log_slope']=math.log10(entry['odd_l2']/previous[key]['odd_l2'])
                    pair[key]=entry
                pairs.append(pair);previous=pair
    payload={'schema':'sonic_expert.local_feedback_virtual_gain.v1','checkpoint':str(checkpoint_path),'checkpoint_sha256':sha(checkpoint_path),
        'source_hashes':source_hashes,'prior':checkpoint['physical_feedforward_prior'],'prior_sha256':sha(checkpoint['physical_feedforward_prior']),
        'policy_device':'cpu','cuda_visible_devices':'','physics_steps_executed':0,'training_steps_executed':0,'nominal_phases_s':phases,
        'perturbation_epsilon':[1e-5,1e-4,1e-3],'motor_names':motor_names,'observation_dim':checkpoint['obs_dim'],'action_scale_rad':env.action_scale,
        'history_contract':'All new policy/feedback histories explicitly zero for each virtual probe; no dynamics. q and v are actual CEM nominal values except specified one-coordinate perturbation.',
        'formula':'timegate * min(1,max(root position/quaternion abs error/.01,body joint abs error/.02,body joint velocity abs error*.1,root6velocity abs error*.5)) * (joint-clipped nominal+action_scale*tanh(actor(obs)) - nominal_float32).',
        'mathematical_scope':'For locally smooth policy h(e)=h0+He+O(e²), even norm gate g(e)=c|e| gives leading even bias c|e|h0 and odd term c|e|He=O(e²). Nonzero nominal error, active clip boundaries, float32 resolution, history dynamics, time dependence and nonsmooth actor/observations change this local expansion. This diagnosis does not identify a unique cause of task failures.',
        'rows':rows,'plus_minus_decomposition':pairs}
    (out/'report.json').write_text(json.dumps(payload,indent=2)+'\n');env.close()
    compact=[]
    for phase in phases:
        for name,_,_ in probes:
            selected=[p for p in pairs if p['phase_s']==phase and p['probe']==name]
            compact.append({'phase_s':phase,'probe':name,'odd_feedback_slopes':[p['gated_feedback_rad'].get('odd_observed_log_slope')for p in selected],
                'even_over_odd_at_1e3':selected[-1]['gated_feedback_rad']['even_l2']/max(selected[-1]['gated_feedback_rad']['odd_l2'],1e-30),
                'actual_motor_odd_slopes':[p['actual_new_motor_delta_rad'].get('odd_observed_log_slope')for p in selected]})
    (out/'summary.json').write_text(json.dumps({'checkpoint_sha256':sha(checkpoint_path),'probes':compact,'no_dynamics_or_training':True},indent=2)+'\n')
    print(json.dumps({'output':str(out),'rows':len(rows),'pairs':len(pairs),'summary':compact},indent=2))
if __name__=='__main__':main()
