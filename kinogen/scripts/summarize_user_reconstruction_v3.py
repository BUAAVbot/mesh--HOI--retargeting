"""Collect SHA-bound final V3 audit and preview results, using the standard library."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def record(path):
    p=ROOT/path
    return {'path':path,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}


def main():
    audit_path='reports/user-v3-independent-audit.json'
    scene_path='reports/user-full-scene-v3.json'
    provenance_path='reports/user-v3-canonical-provenance.json'
    audit=json.loads((ROOT/audit_path).read_text());scene=json.loads((ROOT/scene_path).read_text())
    body=json.loads((ROOT/provenance_path).read_text())
    if audit['strict_metric_reconstruction_valid']or scene['strict_metric_reconstruction_valid']:
        raise RuntimeError('Unexpected strict reconstruction claim')
    candidate=audit['candidate'];baseline=audit['baseline_v2_contact']
    for item in scene['outputs']:
        if record(item['path'])['sha256']!=item['sha256']:
            raise RuntimeError('Final scene output no longer matches audit')
    f54=audit['frame54_fixed_rgb_hand_image_comparison']
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'status':'conditional_3d_candidate_exported_image_quality_incomplete',
            'frames':121,'fps':24,'body_independent_controls':29,'hand_independent_controls':12,
            'expanded_mixed_unit_states':79,'robot_links':89,'fingers_per_hand':5,
            'gpu_training_launched':False,'local_ml_runtime_used':False,
            'strict_metric_reconstruction_valid':False,'physical_hardware_validated':False,
            'necessary_physical_sample_checks_passed':audit['audited_necessary_physical_checks_passed'],
            'audited_image_gates_passed':audit['all_audited_image_gates_passed'],
            'candidate_acceptance_checks_passed':audit['candidate_acceptance_checks_passed'],
            'before_after':{k:{'frozen_v2':baseline[k],'v3':candidate[k]}for k in
                 ('ground_max_penetration_m','ground_failed_frames_gt5mm','hand_max_sample_penetration_m',
                  'hand_failed_frames_gt5mm','all_positive_original_body_velocity_limits_passed',
                  'hand_image_fit_valid_sides','hand_strict_valid_sides','hand_observed_joint_entries','hand_uncertain_joint_entries')},
            'frame54_actual_fixed_rgb_hand_image_audit':f54,
            'fixed_material_foot_projection_checks':audit['independent_fixed_foot_projection_checks'],
            'rendered_visible_hand_comparison':scene.get('visible_hand_comparison_summary'),
            'frame54_complete_mesh_visibility':next(r['visible_hand_geometry_comparison']for r in scene['per_frame_mesh_audit']if r['frame']==54),
            'rendered_full_mesh_diagnostics':scene['independent_full_mesh_audit'],
            'independent_xml_fk_max_abs_error':candidate['full_xml_fk_max_abs_error'],
            'canonical_body_candidate_valid_frames':body['body_candidate_valid_frames'],
            'box_strict_valid_frames':audit['box_strict_valid_frames'],
            'box_strict_invalid_frames':audit['box_strict_invalid_frame_count'],
            'source_video_sha256':scene['source_video_sha256'],
            'sources':[record(p)for p in (audit_path,scene_path,provenance_path)],
            'local_preview_files':[x for x in scene['outputs']if x['path'].endswith('.mp4')],
            'server_scene_files':[x for x in scene['outputs']if x['path'].endswith('.glb')],
            'limitations':['Cameras, floor, known box and ROHand mounting remain conditional authored geometry.',
                          'F54 right full-scene visible-hand contour is missing despite a clear source hand: nonpenetration plus a projected simplified proxy is insufficient and this V3 result is not a successful image reconstruction.',
                          'Box strict validity and unobserved joints are not promoted by finite fitted values.',
                          'Pixel-domain diagnostics reuse approximate manual/LK and model-selection holdouts, not a new confirmatory measurement.',
                          'Zero sampled penetration does not prove continuous collision freedom, bilateral grasp or dynamics.',
                          'No physical simulation, policy tracking, hardware test or official full KinoGen pipeline validation.']}
    output=ROOT/'reports/user-reconstruction-v3-summary.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':record(output.relative_to(ROOT).as_posix()),'before_after':report['before_after'],
                      'necessary_checks':report['necessary_physical_sample_checks_passed'],
                      'image_checks':report['audited_image_gates_passed']},indent=2))


if __name__=='__main__':
    main()
