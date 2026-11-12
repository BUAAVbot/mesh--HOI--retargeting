"""Prepare clearly labeled task-only physical seeds after the full prior-model audit."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    a=p.parse_args()
    import numpy as np
    out=a.root/'prior_physical_state_pool';audit=json.loads((out/'report.json').read_text())
    if not audit['model_parameters_identical']or not audit['old_controls_replayed_in_current_model']['passed']:
        raise RuntimeError('Must pass full model/actual-control replay audit first')
    quality=json.loads((out/'per_candidate_quality.json').read_text())
    z=dict(np.load(audit['old_trace'],allow_pickle=False));indices=np.asarray([r['state_index']for r in quality])
    required_physical=np.asarray([r['future_stable']and r['future_seat_match']and r['future_seated_motion_quiet']for r in quality])
    loose_path=np.asarray([r['root_xy_max_m']<=.65 and r['root_z_max_m']<=.30 and
                           r['root_orientation_max_rad']<=.8 and r['body_joint_rms_max_rad']<=.8 for r in quality])
    columns=np.asarray([i for i,n in enumerate(z['joint_names'])if not str(n).startswith(('lh_','rh_','neck_'))])
    aids=np.asarray(z['actuator_ids'])[:,columns].reshape(-1)
    outputs=[]
    for kind,mask in [('physical_contact_valid',required_physical),('task_only',required_physical&loose_path)]:
        ids=indices[mask];chosen=[q for q,accepted in zip(quality,mask)if accepted]
        fields={'q':z['qpos'][ids].copy(),'v':z['qvel'][ids].copy(),
            'ctrl':z['ctrl'][np.minimum(ids,len(z['ctrl'])-1)].copy(),'phase':z['source_time_s'][ids].copy(),
            'previous':np.zeros((len(ids),56)),
            'body_target':z['ctrl'][np.maximum(ids-1,0)][:,aids].copy(),
            'captured_iteration':np.full(len(ids),-1,dtype=np.int64),
            'root_xy_m':np.asarray([q['root_xy_max_m']for q in chosen]),
            'penetration_m':np.asarray([q['future_penetration_max_m']for q in chosen]),
            'old_execution_time_s':z['time'][ids].copy(),'source_phase_s':z['source_time_s'][ids].copy(),
            'old_state_index':ids.copy(),'is_old_physical_prior_seed':np.ones(len(ids),bool),
            'is_new_phc_expert_success':np.zeros(len(ids),bool),
            'strict_phc_path_gate_pass':np.asarray([q['phc_path_gate']for q in chosen],bool),
            'source_expected_seat':np.asarray([q['source_expected_seat']for q in chosen],bool),
            'future_seat_force_mean_N':np.asarray([q['future_seat_force_mean_N']for q in chosen]),
            'future_seat_contact_fraction':np.asarray([q['future_seat_contact_fraction']for q in chosen]),
            'phc_body_joint_rms_max_rad':np.asarray([q['body_joint_rms_max_rad']for q in chosen])}
        for scale,label in ((1.,'source1x'),(1.5,'source1p5x'),(2.,'source2x')):
            fields['phase']=z['source_time_s'][ids]*scale
            path=out/f'candidate156_{label}_{kind}_state_pool.npz'
            np.savez_compressed(path,**fields)
            outputs.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'states':len(ids),
                'mode':kind,'phase_time_scale':scale,'source_phase_coverage_s':[float(z['source_time_s'][ids].min()),float(z['source_time_s'][ids].max())],
                'old_execution_time_coverage_s':[float(z['time'][ids].min()),float(z['time'][ids].max())],
                'terminal_source_phase_states':int(np.sum(z['source_time_s'][ids]>=5.7-1e-8))})
    diagnostic=out/'candidate156_all_candidates_with_masks.npz'
    np.savez_compressed(diagnostic,q=z['qpos'][indices],v=z['qvel'][indices],
        ctrl=z['ctrl'][np.minimum(indices,len(z['ctrl'])-1)],old_state_index=indices,
        old_execution_time_s=z['time'][indices],source_phase_s=z['source_time_s'][indices],
        future_stability_mask=np.asarray([q['future_stable']for q in quality],bool),
        future_seat_match_mask=np.asarray([q['future_seat_match']for q in quality],bool),
        future_seated_quiet_mask=np.asarray([q['future_seated_motion_quiet']for q in quality],bool),
        physical_contact_valid_mask=required_physical,task_only_valid_mask=required_physical&loose_path,
        strict_phc_valid_mask=np.asarray([q['eligible']for q in quality],bool),
        phc_root_xy_max_m=np.asarray([q['root_xy_max_m']for q in quality]),
        phc_root_z_max_m=np.asarray([q['root_z_max_m']for q in quality]),
        phc_root_orientation_max_rad=np.asarray([q['root_orientation_max_rad']for q in quality]),
        phc_body_joint_rms_max_rad=np.asarray([q['body_joint_rms_max_rad']for q in quality]),
        future_seat_force_mean_N=np.asarray([q['future_seat_force_mean_N']for q in quality]),
        source_expected_seat=np.asarray([q['source_expected_seat']for q in quality],bool),
        future_penetration_max_m=np.asarray([q['future_penetration_max_m']for q in quality]))
    receipt={'schema':'sonic_expert.old_physical_task_seed_export.v1','status':'old_prior_task_seed_export_complete',
        'source_model_audit':str(out/'report.json'),'source_model_audit_sha256':sha(out/'report.json'),
        'old_trace':audit['old_trace'],'old_trace_sha256':audit['old_trace_sha256'],
        'current_phc_reference':audit['current_reference'],'current_reference_sha256':audit['current_reference_sha256'],
        'provenance':'Old selected candidate156 customIK+SONIC/PPO functional demonstration; not a newly trained PHC expert result.',
        'new_phc_expert_success_claim':False,'strict_source_success_claim':False,
        'physical_contact_valid_policy':'Keeps future0.2s actual no-fall, penetration<25mm, actual seat load/contact consistency, quiet seated motion; removes the PHC pose/path gate entirely.',
        'task_only_policy':'Same physical gates, with relaxed PHC rootXY<=0.65m/rootZ<=0.30m/rootOrientation<=0.8rad/bodyJointRMS<=0.8rad. This is not strict source tracking.',
        'q_v_ctrl_unchanged':True,'velocities_scaled':False,
        'phase_contract':'source1p5 phase=1.5*old source_time_s, preserves old reference time rate before terminal hold; source2x phase=2*old source_time_s. Old terminal hold states map to reference-end phase. No actual q/v is time-rescaled.',
        'history_contract':'previous new-expert action explicitly zero; body_target is the preceding real motor packet. Consumer must reset its policy/SONIC history; these are physical, not recurrent-controller, state seeds.',
        'candidate156_initial_state':str(out/'candidate156_initial_state.npz'),
        'candidate156_initial_state_sha256':sha(out/'candidate156_initial_state.npz'),
        'files':outputs,'all_candidates_diagnostic':{'path':str(diagnostic),'sha256':sha(diagnostic),
            'candidates':len(indices),'physical_contact_valid_count':int(required_physical.sum()),
            'note':'All469 candidates are retained with masks. Failed seat/contact/quiet candidates are not labeled valid.'},
        'export_source_sha256':sha(__file__)}
    (out/'task_seed_export_report.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps([{'mode':r['mode'],'scale':r['phase_time_scale'],'states':r['states'],
        'phase':r['source_phase_coverage_s'],'terminal':r['terminal_source_phase_states']}for r in outputs]))


if __name__=='__main__':main()
