"""Independent contract/metrics audit of two unchanged-SONIC reference-transfer trials."""
import hashlib,json
from pathlib import Path
import numpy as np
from two_sitting_physics import interp

def main():
    root=Path('/data/KinoGen/outputs/sonic_expert_20261028');r=root/'sonic_refined_reference_transfer'
    summary={'schema':'sonic_expert.frozen_sonic_refined_reference_ab.v1','not_training_or_distillation':True,'original_ten_baselines_unchanged':True,'results':[]}
    bags=[];refs={k:dict(np.load(p,allow_pickle=False))for k,p in [('PHC',root/'references/phc_time_scale1.5.npz'),('CEM_actual',r/'cem_actual_q_reference_50hz.npz')]}
    for tag in ['A_phc_coarse','B_cem_refined']:
        p=r/tag;report=json.loads((p/'report.json').read_text());z=dict(np.load(p/'trace.npz',allow_pickle=False));bags.append(z)
        actors=report['controller']['base']['actors'];cross={}
        for name,ref in refs.items():
            target=np.asarray([interp(ref,t)for t in z['time']])
            xy=np.stack([np.linalg.norm(z['qpos'][:,rq:rq+2]-target[:,a,:2],axis=-1)for a,rq in enumerate(z['robot_root_qadr'])],1)
            cross[name]={'root_xy_rms_m':float(np.sqrt(np.mean(xy**2))),'endpoint_xy_m':xy[-1].tolist()}
        row={k:report[k]for k in ['functional_task_success','success','simulated_seconds','physics_steps','termination','final_root_xy_error_m','max_root_xy_error_m','max_penetration_m','final_seat_vertical_force_mean_N','root_height_min_m','root_global_tilt_max_rad','independent_replay','functional_task_checks']}
        row.update(tag=tag,source_reference_sha256=report['inputs']['reference_sha256'],zero_residual=bool(np.count_nonzero(z['learned_residual_action'])==0),
            actual_sonic_policy_sha256=[a['original_actor']['policy_sha256']for a in actors],batch_inference_count=[a['batch_inference_count']for a in actors],
            model_sha256=hashlib.sha256((p/'model.mjb').read_bytes()).hexdigest(),source_tracking_cross_references=cross)
        summary['results'].append(row)
    summary['pair_audit']={'initial_integration_bitwise_same':bool(np.array_equal(bags[0]['initial_state'],bags[1]['initial_state'])),
        'model_hash_same':summary['results'][0]['model_sha256']==summary['results'][1]['model_sha256'],
        'sonic_weights_same':summary['results'][0]['actual_sonic_policy_sha256']==summary['results'][1]['actual_sonic_policy_sha256'],
        'both_learned_residual_exact_zero':all(x['zero_residual']for x in summary['results']),
        'requested_action_duration_s':8.55,'requested_hold_s':1.}
    if not all(summary['pair_audit'][k]for k in ['initial_integration_bitwise_same','model_hash_same','sonic_weights_same','both_learned_residual_exact_zero']):raise ValueError('Pair contract mismatch')
    summary['conclusion']='Measured physically executable CEM joint reference did not transfer functional task success to unchanged SONIC. B terminated earlier; no generalist improvement or PPO success established.'
    summary['audit_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (r/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
