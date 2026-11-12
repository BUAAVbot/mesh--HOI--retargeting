"""Independently verify zero residual, unchanged references and frozen SONIC provenance."""
from __future__ import annotations
import argparse, ast, hashlib, json
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    a=p.parse_args()
    import numpy as np
    protocol=json.loads((a.root/'benchmark_protocol.json').read_text())
    comparison=Path('/data/KinoGen/outputs/retarget_compare_20261028')
    actual_policy=Path('/data/KinoGen/data/vita_demo_20261026/checkpoints/new_ft15225/model_step_015225.pt')
    policy_sha=sha(actual_policy)
    rows=[]
    for item in protocol['references']:
        method=item['method'];trial=a.root/'sonic_baseline'/method
        report=json.loads((trial/'report.json').read_text())
        with np.load(trial/'trace.npz',allow_pickle=False)as z:
            residual=z['learned_residual_action']
            residual_max=float(np.max(np.abs(residual)))
            trace_steps=len(z['ctrl'])
            consistency=bool(len(z['qpos'])==trace_steps+1 and len(z['qvel'])==trace_steps+1)
            finite=bool(np.isfinite(z['qpos']).all()and np.isfinite(z['qvel']).all()and np.isfinite(z['ctrl']).all())
        reference=Path(item['reference'])
        with np.load(reference,allow_pickle=False)as z,np.load(comparison/method/'evaluation_arrays.npz',allow_pickle=False)as src:
            qpos_identical=bool(np.array_equal(z['qpos'],src['qpos']))
            clock_identical=bool(np.array_equal(z['time_s'],src['time_s']))
            qpos_max=float(np.max(np.abs(z['qpos']-src['qpos'])))
        source=trial/'source_snapshot/two_sitting_eval_ppo.py'
        tree=ast.parse(source.read_text())
        zero_branch=any(isinstance(node,ast.IfExp)and ast.unparse(node.test)=='a.zero_residual'
                        and ast.unparse(node.body).startswith('torch.zeros(')
                        and 'actor.actor('in ast.unparse(node.orelse)for node in ast.walk(tree))
        controller=report['controller'];actors=controller['base']['actors']
        actor_hashes=[actor['original_actor']['policy_sha256']for actor in actors]
        counts=[actor['batch_inference_count']for actor in actors]
        checks={'recorded_residual_exactly_zero':residual_max==0.,
            'frozen_source_has_hard_zero_conditional':zero_branch,
            'zero_residual_override_recorded':controller['zero_residual_override']is True,
            'current_reference_matches_protocol_sha':sha(reference)==item['reference_sha256'],
            'canonical_qpos_bitwise_identical':qpos_identical,'canonical_clock_bitwise_identical':clock_identical,
            'both_original_sonic_hashes_match_current_file':actor_hashes==[policy_sha,policy_sha],
            'actual_batch_inferences_recorded':min(counts)>0,
            'state_and_control_counts_consistent':consistency,'trace_finite':finite,
            'application_state_override_count_zero':report['state_override_after_reset_count']==0,
            'external_force_steps_zero':report['external_force_steps']==0,
            'root_actuators_zero':report['root_actuators']==0,'mocap_zero':report['mocap_bodies']==0,
            'independent_control_replay_passed':report['independent_replay']['passed']is True,
            'no_terminal_seating_feedback':controller['terminal_seating_feedback']is None}
        rows.append({'method':method,'passed':all(checks.values()),'checks':checks,
            'residual_max_abs':residual_max,'canonical_qpos_max_abs_difference':qpos_max,
            'sonic_batch_inferences_per_actor':counts,'original_sonic_sha256':actor_hashes,
            'frozen_model_sha256':sha(trial/'model.mjb'),'trace_sha256':sha(trial/'trace.npz'),
            'frozen_evaluator_sha256':sha(source),'physics_steps':trace_steps})
    calibration=json.loads((a.root/'calibration/report.json').read_text())
    hashes=sorted(set(row['frozen_model_sha256']for row in rows))
    result={'schema':'sonic_expert.zero_residual_independent_audit.v1',
        'passed':all(row['passed']for row in rows)and len(hashes)==1 and hashes[0]==calibration['expert_sha256'],
        'methods_checked':len(rows),'current_original_sonic_checkpoint_sha256':policy_sha,
        'all_baselines_and_expert_use_bitwise_identical_mjb':len(hashes)==1 and hashes[0]==calibration['expert_sha256'],
        'baseline_frozen_mjb_sha256s':hashes,'expert_frozen_mjb_sha256':calibration['expert_sha256'],
        'explanation':'The previous task checkpoint is loaded only for dimensions/metadata. The frozen evaluator chooses an exact zero tensor instead of calling its actor; recorded residuals are all exactly0. Actual frozen SONIC inference remains active.',
        'scope':'Provenance and zero-residual execution audit; this does not turn failed task outcomes into successes.',
        'results':rows,'audit_source_sha256':sha(__file__)}
    out=a.root/'audit';out.mkdir(exist_ok=True)
    (out/'baseline_zero_residual.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key]for key in ('passed','methods_checked','all_baselines_and_expert_use_bitwise_identical_mjb')}))


if __name__=='__main__':main()
