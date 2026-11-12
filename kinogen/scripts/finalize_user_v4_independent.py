#!/usr/bin/env python3
"""Combine immutable core and limited-stage audits without claiming unmeasured quality."""
import argparse
import copy
import json
from pathlib import Path
from verify_user_reconstruction_v3 import ROOT,record,require,check_provenance,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--core',type=Path,default=Path('reports/user-v4-core-independent-audit.json'))
    ap.add_argument('--combined',type=Path,default=Path('reports/user-v4-combined-independent.json'))
    ap.add_argument('--scene',type=Path,default=Path('reports/user-full-scene-v4.json'))
    ap.add_argument('--nonhand-diagnostic',type=Path,required=True)
    ap.add_argument('--body-gate-report',type=Path,action='append',default=[])
    ap.add_argument('--output',type=Path,default=Path('reports/user-v4-independent-audit.json'))
    args=ap.parse_args()
    def path(p):
        p=(ROOT/p).resolve();require(p.is_relative_to(ROOT),'Path outside project');return p
    core,combined,scene,nonhand=[json.loads(path(p).read_text())for p in (args.core,args.combined,args.scene,args.nonhand_diagnostic)]
    require(core['verification_script']['sha256']=='304d7a68277a0c78d23200ccebdde377d95b97c7f3e06560ad86edf4adbf8a19','Unexpected frozen core implementation')
    require(sha(ROOT/core['verification_script']['path'])==core['verification_script']['sha256'],'Core source changed')
    for audit in (combined,nonhand):require(sha(ROOT/audit['verification_script']['path'])==audit['verification_script']['sha256'],'Extended verifier source changed after audit')
    require(all(sha(ROOT/x['path'])==x['sha256']for x in core['inputs']+combined['inputs']),'Core/combined source files changed')
    core_inputs={x['path']:x['sha256']for x in core['inputs']}
    for role in ('body_trajectory','hand_trajectory'):
        binding=scene['inputs'][role];require(core_inputs.get(binding['path'])==binding['sha256'],'Scene is not audited core pair')
        require(any(x['path']==binding['path']and x['sha256']==binding['sha256']for x in combined['inputs']),'Combined audit used another pair')
        nr=nonhand['stage_candidate']['source_body'if role=='body_trajectory'else'source_hand'];require(nr=={**nr,'path':binding['path'],'sha256':binding['sha256']},'Non-hand audit used another pair')
    for binding in scene['outputs']:require(sha(ROOT/binding['path'])==binding['sha256'],'Scene output changed')
    observed={'left':0,'right':0};failures=[];visible_by_frame=[]
    for row in scene['per_frame_mesh_audit']:
        for value in row['visible_hand_geometry_comparison']:
            if value is None:continue
            side=value['side'];observed[side]+=1;c=value['candidate'];failed=c['missing_visible_model_contour']or c['visible_hand_pixels_in_domain']==0
            item={'frame':row['frame'],'side':side,'observed_domain':{'source_pixels':value['source_pixels'],'heldout_external_edge_points':value['heldout_external_edge_points'],'observation_available':value['observation_available']},'candidate':c,'baseline':value.get('frozen_v3',value.get('frozen_v2'))}
            # Exporter versions name the frozen comparison differently; preserve exact source.
            item['source_comparison']=value;visible_by_frame.append(item)
            if failed:failures.append({'frame':row['frame'],'side':side,'visible_hand_pixels_in_domain':c['visible_hand_pixels_in_domain'],'missing_visible_model_contour':c['missing_visible_model_contour']})
    require(len(scene['per_frame_mesh_audit'])==121,'Incomplete visible-scene frame audit')
    require(sum(observed.values())>0,'No real source hand domain was audited')
    body_reports=[];body_gates=[]
    for p in args.body_gate_report:
        pp=path(p);d=json.loads(pp.read_text());entry={'source':record(pp),'provenance':check_provenance(pp)}
        if 'body_contour_gates_passed'in d:
            passed=bool(d['body_contour_gates_passed']);entry.update(passed=passed,frame=d.get('frame'),metrics=d.get('metrics'),rejections=d.get('gate_rejections'));body_gates.append(passed)
        elif 'final'in d and 'selected'in d:
            passed=bool(d['selected']and not d['final']['rejections']);entry.update(passed=passed,producer_status=d.get('status'),actual_final_gates=d['final'],hand_observation_available=d.get('hand_observation_available'));body_gates.append(passed)
        elif any('selected_for_conditional_comparison'in r for r in d.get('records',[])):
            selected=[r for r in d['records']if r['selected_for_conditional_comparison']]
            passed=bool(selected)and all(not r['comparison_rejections']and not [why for why in r['original_zero_target_rejections']if why!='raw_mesh_penetration']for r in selected)
            entry.update(passed=passed,conditional_body_image_records=selected,zero_target_passed=False,source_non_deterioration_passed=False,scope='Original body/pixel gates retained. Sub-mm conditional penetration is not reclassified as a zero-target pass.');body_gates.append(passed)
        else:
            adopted=[]
            for r in d.get('records',[]):
                if not r.get('accepted',r.get('selected',False)):continue
                eligible=[t for t in r.get('trials',[])if not t['rejections']]
                if r.get('selected_phase')is not None:eligible=[t for t in eligible if t['phase']==r['selected_phase']]
                require(eligible,'Adopted stage has no passing original gates')
                adopted.append({'frame':r['frame'],'selected_phase':r.get('selected_phase'),'eligible_passing_trials':eligible})
            entry.update(passed=bool(adopted),adopted_stage_records=adopted,accepted_frames=d.get('accepted_frames',d.get('selected_frames')))
            body_gates.append(bool(adopted))
        body_reports.append(entry)
    available_proxies=[x for x in combined['fixed_rgb_proxy_by_frame'].values()if x['observation_available']]
    unavailable_proxy_frames=[int(f)for f,x in combined['fixed_rgb_proxy_by_frame'].items()if not x['observation_available']]
    proxy_pass=bool(available_proxies)and all(x['candidate']['hand_image_necessary_gates_passed']for x in available_proxies)
    landmarks_pass=all(c['all_available_necessary_gates_passed']for x in combined['fixed_body_landmarks']['by_frame'].values()for c in x['checks'])
    body_pass=bool(body_gates)and all(body_gates)
    out=copy.deepcopy(core)
    out['core_subset_status']=core['status'];out['core_subset_candidate_acceptance_checks_passed']=core['candidate_acceptance_checks_passed'];out['core_subset_all_audited_image_gates_passed']=core['all_audited_image_gates_passed']
    out['independent_v4_extended_sources']=[record(path(p))for p in (args.core,args.combined,args.scene,args.nonhand_diagnostic)]+[record(path(p))for p in args.body_gate_report]
    out['aggregation_script']=record(Path(__file__));out['limited_stage_comparison']=combined
    out['visible_hand_presence_failures']=failures;out['visible_hand_presence_failed_frames']=sorted(set(x['frame']for x in failures));out['visible_hand_source_coverage']={'observed_frame_sides':observed,'unavailable_frame_sides':{s:121-n for s,n in observed.items()},'unavailable_policy':'No observed ROI is unknown, not an image pass.'};out['complete_scene_visible_hand_records']=visible_by_frame
    out['all_available_observed_domain_hand_contours_present']=not failures
    out['owned_frame_hand_proxy_not_applicable_missing_source']=unavailable_proxy_frames
    out['original_adopted_stage_body_gates']={'passed':body_pass,'reports':body_reports,'scope':'Original SHA-bound producer body SAM contour/IoU holdout gates, plus independent both-view fixed material landmarks in combined audit. No new ground-truth measurement or new threshold.'}
    out['all_audited_image_gates_passed']=bool(core['all_audited_image_gates_passed']and proxy_pass and landmarks_pass and body_pass and not failures)
    out['nonhand_original_visual_box_diagnostic']={'source':record(path(args.nonhand_diagnostic)),'method':nonhand['method'],'nonhand_visual_count':nonhand['nonhand_visual_count'],'baseline_frozen_v3':{k:v for k,v in nonhand['baseline_frozen_v3'].items()if k!='per_frame'},'candidate':{k:v for k,v in nonhand['stage_candidate'].items()if k!='per_frame'},'nonhand_no_worse_per_frame':nonhand['nonhand_no_worse_per_frame']}
    out['zero_target_hand_sample_penetration_failed_frames']=[r['frame']for r in core['candidate']['per_frame']if max(r['hand_vertex_depth_m'],r['hand_trianglecenter_depth_m'])>1e-6]
    out['zero_target_nonhand_sample_penetration_failed_frames']=[r['frame']for r in nonhand['stage_candidate']['per_frame']if r['max_nonhand_box_sample_penetration_m']>1e-6]
    out['zero_target_policy']='Separate >1 micrometre sampled-penetration diagnostic. Does not redefine the original >5mm necessary threshold, or promote conditional sub-mm candidates to zero-penetration.'
    out['extended_necessary_physical_sample_checks_passed']=bool(core['audited_necessary_physical_checks_passed']and not nonhand['stage_candidate']['frames_deeper_than_5mm'])
    out['candidate_acceptance_checks_passed']=bool(out['extended_necessary_physical_sample_checks_passed']and out['all_audited_image_gates_passed'])
    out['status']='conditional_3d_candidate_quality_incomplete';out['overall_reconstruction_success']=False;out['strict_metric_reconstruction_valid']=False;out['physical_hardware_validated']=False
    out['quality_scope_note']='Frozen core audits only a subset of image gates and hand52 collision. This aggregate additionally exposes every available-source missing visible hand, adopted stage body gates, and non-hand visual-box sample collisions. No physical calibration, exact continuous/self collision, all-finger observability or bilateral grasp is proved.'
    target=path(args.output);target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':record(target),'core_subset_pass':out['core_subset_candidate_acceptance_checks_passed'],'visible_hand_presence_failures':failures,'nonhand_failed_frames':nonhand['stage_candidate']['frames_deeper_than_5mm'],'all_audited_image_gates_passed':out['all_audited_image_gates_passed'],'extended_physical_sample_checks_passed':out['extended_necessary_physical_sample_checks_passed'],'overall_reconstruction_success':False},indent=2))


if __name__=='__main__':main()
