"""Audit six grasp-schedule proposals against immutable raw source observations.

No rollout, reference, model, input candidate or controller configuration is
modified. Candidate ownership windows are proposals, never ground-truth contact.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--geometry-model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if os.name!='posix':p.error('Server-only read-only source audit')
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    from scipy.ndimage import median_filter
    source=dict(np.load(a.reference,allow_pickle=False));m=mj.MjModel.from_xml_path(str(a.geometry_model.resolve()))
    times=source['time_s'];n=len(times);position=source['raw_source_object_position_m'];quaternion=source['raw_source_object_quaternion_wxyz']
    valid=source['raw_source_object_valid'].astype(bool)&np.isfinite(position).all(2)&np.isfinite(quaternion).all(2)
    status=source['object_status'];candidate=source['contact_candidate_observed'].astype(bool)
    numeric=mj.mj_name2id(m,mj.mjtObj.mjOBJ_NUMERIC,'source_table_plane_abcd');adr=int(m.numeric_adr[numeric]);plane=m.numeric_data[adr:adr+4].copy();plane/=np.linalg.norm(plane[:3])
    def ranges(mask):
        edge=np.diff(np.r_[False,mask,False].astype(int));return list(zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)))
    def intervals(mask):
        return [{'first_frame':int(start),'last_frame':int(stop-1),'start_s':float(times[start]),'end_s':float(times[stop-1]),'observed_frames':int(stop-start)}for start,stop in ranges(mask)]
    def observed_chunks(mask):
        ids=np.flatnonzero(mask)
        if not len(ids):return []
        splits=np.flatnonzero(np.diff(times[ids])>.55)+1
        return [{'first_frame':int(group[0]),'last_frame':int(group[-1]),'start_s':float(times[group[0]]),'end_s':float(times[group[-1]]),'candidate_frames':len(group)}for group in np.split(ids,splits)]
    height=np.full((n,6),np.nan);clearance=np.full_like(height,np.nan);relative_clearance=np.full_like(height,np.nan);speed=np.full_like(height,np.nan)
    origin_gap=np.einsum('noi,i->no',position,plane[:3])+plane[3];baseline=[]
    for oi in range(6):
        ids=np.flatnonzero(valid[:,oi]);rots=Rotation.from_quat(quaternion[ids,oi][:,[1,2,3,0]]).as_matrix();pos=position[ids,oi]
        bid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,f'target_object_{oi+1:02d}')
        minimum=np.full(len(ids),np.inf)
        for gid in np.flatnonzero((m.geom_bodyid==bid)&(m.geom_contype>0)):
            local_R=Rotation.from_quat(m.geom_quat[gid,[1,2,3,0]]).as_matrix();typ=m.geom_type[gid];size=m.geom_size[gid]
            centre=pos+np.einsum('nij,j->ni',rots,m.geom_pos[gid]);centre_gap=centre@plane[:3]+plane[3]
            if typ==mj.mjtGeom.mjGEOM_MESH:
                mid=int(m.geom_dataid[gid]);start=int(m.mesh_vertadr[mid]);number=int(m.mesh_vertnum[mid]);points=m.mesh_vert[start:start+number]@local_R.T+m.geom_pos[gid]
                local_plane_normal=np.einsum('nji,j->ni',rots,plane[:3]);support=np.min(local_plane_normal@points.T,axis=1)
                values=pos@plane[:3]+plane[3]+support
            elif typ==mj.mjtGeom.mjGEOM_CYLINDER:
                axis=np.einsum('nij,j->ni',rots,local_R[:,2]);projection=axis@plane[:3]
                values=centre_gap-size[1]*abs(projection)-size[0]*np.sqrt(np.maximum(0.,1-projection**2))
            else:raise ValueError('Unexpected object physical primitive')
            minimum=np.minimum(minimum,values)
        clearance[ids,oi]=minimum
        initial=ids[ids<30];initial=initial if len(initial)else ids[:min(10,len(ids))]
        base_origin=float(np.median(origin_gap[initial,oi]));base_proxy=float(np.median(clearance[initial,oi]));baseline.append({'object_id':oi+1,'origin_plane_gap_m':base_origin,'proxy_bottom_plane_gap_m':base_proxy,'frames':initial.tolist()})
        height[ids,oi]=origin_gap[ids,oi]-base_origin;relative_clearance[ids,oi]=clearance[ids,oi]-base_proxy
        for start,stop in ranges(valid[:,oi]):
            if stop-start<3:continue
            points=median_filter(position[start:stop,oi],size=(5,1),mode='nearest')
            velocity=np.gradient(points[:,:2],times[start:stop],axis=0)
            speed[start:stop,oi]=np.linalg.norm(velocity,axis=1)
    drafts=[(1,1,88,193),(2,0,218,304),(3,1,313,396),(4,1,417,501),(5,0,527,571),(6,1,631,710)]
    all_events=[]
    for side in range(2):
        for oi in range(6):
            for row in observed_chunks(candidate[:,side,oi]):all_events.append(dict(row,object_id=oi+1,hand=['left','right'][side]))
    def stats(oi,start,end):
        window=(times>=start)&(times<=end);observed=window&valid[:,oi];ids=np.flatnonzero(observed);good=observed&np.isin(status[:,oi],[1,5])
        if not len(ids):return {'observed_frames':0,'raw_valid_fraction':0.}
        first=ids[:min(5,len(ids))];last=ids[-min(5,len(ids)):]
        net=np.median(position[last,oi],axis=0)-np.median(position[first,oi],axis=0)
        pair=observed[1:]&observed[:-1];length=float(np.linalg.norm(np.diff(position[:,oi,:2],axis=0)[pair],axis=1).sum())
        finite_speed=speed[observed,oi];finite_speed=finite_speed[np.isfinite(finite_speed)]
        return {'observed_frames':len(ids),'raw_valid_fraction':float(observed.sum()/max(window.sum(),1)),
            'status1_or5_fraction_of_observed':float(good.sum()/len(ids)),
            'status_counts':{str(int(k)):int(v)for k,v in zip(*np.unique(status[observed,oi],return_counts=True))},
            'net_xyz_m':net.tolist(),'net_xy_distance_m':float(np.linalg.norm(net[:2])),
            'observed_consecutive_xy_path_m':length,
            'maximum_origin_rise_m':float(np.nanmax(height[observed,oi])),
            'maximum_proxy_bottom_rise_m':float(np.nanmax(relative_clearance[observed,oi])),
            'median_proxy_bottom_rise_m':float(np.nanmedian(relative_clearance[observed,oi])),
            'proxy_bottom_rise_range_m':float(np.nanmax(relative_clearance[observed,oi])-np.nanmin(relative_clearance[observed,oi])),
            'first_proxy_bottom_rise_m':float(relative_clearance[ids[0],oi]),'last_proxy_bottom_rise_m':float(relative_clearance[ids[-1],oi]),
            'minimum_proxy_table_gap_m':float(np.nanmin(clearance[observed,oi])),
            'fraction_proxy_penetration_gt15mm':float(np.mean(clearance[observed,oi]<-.015)),
            'smoothed_xy_speed_p95_m_s':float(np.quantile(finite_speed,.95))if len(finite_speed)else None,
            'observed_missing_spans':intervals(window&~valid[:,oi])}
    schedule=[]
    for obj,side,first,last in drafts:
        oi=obj-1;start=float(times[first]);end=float(times[last]);window=np.zeros(n,bool);window[first:last+1]=True
        inside=window&candidate[:,side,oi];coverage=float(inside.sum()/window.sum());during=stats(oi,start,end)
        before=stats(oi,max(0.,start-.75),max(0.,start-1e-6));after=stats(oi,end+1e-6,min(times[-1],end+1.))
        outside=candidate[:,side,oi]&~window;outside_rows=[]
        for row in observed_chunks(outside):
            evidence=stats(oi,row['start_s'],row['end_s']);conflicts=[]
            for other,other_side,other_first,other_last in drafts:
                if other==obj or other_side!=side:continue
                if row['start_s']<=times[other_last]and row['end_s']>=times[other_first]:conflicts.append(other)
            if conflicts:classification='competing_same_hand_candidate_not_second_grasp'
            elif evidence.get('maximum_proxy_bottom_rise_m',0.)>.02 or evidence.get('net_xy_distance_m',0.)>.04:classification='early_motion_or_preparation_uncertain_not_safe_to_label_brush'
            else:classification='possible_preparation_or_brush_not_main_interval'
            outside_rows.append(dict(row,classification=classification,conflicting_main_objects=conflicts,raw_motion=evidence))
        valid_ids=np.flatnonzero(valid[:,oi]);final_position=position[valid_ids[-1],oi]
        near=valid[:,oi]&(np.linalg.norm(position[:,oi,:2]-final_position[:2],axis=1)<.04)&(abs(position[:,oi,2]-final_position[2])<.03)
        group=np.flatnonzero(np.convolve(near.astype(int),np.ones(5,dtype=int),'valid')==5)
        arrival=int(group[0])if len(group)else None
        near_xy=valid[:,oi]&(np.linalg.norm(position[:,oi,:2]-final_position[:2],axis=1)<.04)
        group_xy=np.flatnonzero(np.convolve(near_xy.astype(int),np.ones(5,dtype=int),'valid')==5)
        arrival_xy=int(group_xy[0])if len(group_xy)else None
        late_proxy_level=float(np.nanmedian(relative_clearance[valid_ids[-min(15,len(valid_ids)):],oi]))
        lift=valid[:,oi]&(relative_clearance[:,oi]>.02);moving=valid[:,oi]&(speed[:,oi]>.05)
        lift_spans=intervals(lift);motion_spans=[row for row in intervals(moving)if row['observed_frames']>=3]
        quality=during.get('status1_or5_fraction_of_observed',0.);raw_coverage=during.get('raw_valid_fraction',0.)
        motion_positive=during.get('net_xy_distance_m',0.)>.04;lift_positive=during.get('maximum_proxy_bottom_rise_m',0.)>.02
        score=.25+.2*coverage+.15*raw_coverage+.1*quality+.15*motion_positive+.1*lift_positive
        late=arrival is not None and times[arrival]>end+.5
        if late:score-=.12
        if during.get('fraction_proxy_penetration_gt15mm',0.)>.1:score-=.1
        score=float(np.clip(score,.05,.85))
        reasons=[f'Main-source candidate coverage {int(inside.sum())}/{int(window.sum())} frames; fragments inside this ownership window are merged without editing the source mask.',
            f'Raw pose availability {raw_coverage:.1%}; consecutive-observation XY path {during.get("observed_consecutive_xy_path_m",0.):.3f}m, net {during.get("net_xy_distance_m",0.):.3f}m.',
            f'Estimated proxy-bottom rise max {during.get("maximum_proxy_bottom_rise_m",0.):.3f}m; reconstructed pose and proxy geometry, not measured support/contact.']
        cautions=[]
        if during.get('observed_missing_spans'):cautions.append('Raw object pose has missing intervals inside the proposed grasp. No raw velocity/height is fabricated across them.')
        if late:cautions.append(f'First observed final XYZ basin is later ({times[arrival]:.3f}s) than core contact end ({end:.3f}s). Stacked support changes, settling, continuing unseen hold, pose drift or identity errors remain unresolved; do not equate this with a measured release.')
        if arrival is None:cautions.append('No five-consecutive-observation stable final XYZ basin is confirmed. This does not invalidate the main transport interval, but the final pose is not a measured release cue.')
        if outside_rows:cautions.append('Extra source candidates exist outside the main interval; classifications remain hypotheses and all original candidates are retained.')
        quiet=after.get('observed_frames',0)>=6 and after.get('net_xy_distance_m',1.)<.015 and after.get('proxy_bottom_rise_range_m',1.)<.015
        remaining_descent=max(0.,during.get('last_proxy_bottom_rise_m',0.)-late_proxy_level-.015)
        if after.get('median_proxy_bottom_rise_m',0.)>.02:cautions.append('Post-core proxy height above the original table can reflect resting on the plate stack; height alone is not evidence of continued hand grip.')
        if remaining_descent>.03:cautions.append(f'At the last source candidate the proxy is still {remaining_descent:.3f}m above the late-pose support band. Keep descent/support gating in the controller; do not blindly open at the candidate end.')
        release_window=[end,float(max(end,times[arrival]))if late else end]
        schedule.append({'object_id':obj,'active_hand':['left','right'][side],'active_hand_index':side,
            'proposed_main_grab_release_interval_s':[start,end],'source_frame_interval_inclusive':[first,last],
            'candidate_fragments_merged':observed_chunks(inside),'exact_candidate_runs':intervals(inside),
            'candidate_coverage_fraction':coverage,'raw_before':before,'raw_during':during,'raw_after':after,
            'raw_proxy_lift_spans_gt20mm':lift_spans,'raw_smoothed_motion_spans_gt50mm_s':motion_spans,
            'first_final_position_basin_frame':arrival,'first_final_position_basin_time_s':float(times[arrival])if arrival is not None else None,
            'first_final_xy_basin_frame':arrival_xy,'first_final_xy_basin_time_s':float(times[arrival_xy])if arrival_xy is not None else None,
            'late_observed_proxy_level_above_initial_m':late_proxy_level,
            'remaining_descent_above_late_proxy_band_at_core_end_m':remaining_descent,
            'post_core_pose_locally_quiet':bool(quiet),'release_boundary_confidence_label':'moderate'if quiet and remaining_descent<=.03 else'low',
            'release_uncertainty_review_window_s':release_window,'outside_main_candidates':outside_rows,
            'confidence_heuristic_not_probability':score,'confidence_label':'moderate'if score>=.55 else'low',
            'reasons':reasons,'cautions':cautions,'is_ground_truth':False})
    for side in range(2):
        rows=sorted([row for row in schedule if row['active_hand_index']==side],key=lambda r:r['proposed_main_grab_release_interval_s'][0]);previous=0.
        for row in rows:
            start,end=row['proposed_main_grab_release_interval_s']
            row['predictive_acquire_planning_hint']={'earliest_after_previous_same_hand_core_s':previous,
                'maximum_nonoverlapping_lead_s':max(0.,start-previous),
                'note':'Evidence only: controller lookahead must be clamped by actual same-hand ownership/release and physical safety. Original candidate mask must not be changed to emulate predictive acquire.'}
            previous=end+.1
    payload={key:source[key]for key in source if key.startswith(('raw_source_','contact_candidate','human_'))or key in ['time_s','object_status','object_observed_mask','object_imputed_mask','object_confidence','source_frame_ids','frame_ids']}
    payload.update(analysis_origin_rise_m=height,analysis_proxy_bottom_clearance_m=clearance,analysis_proxy_bottom_rise_m=relative_clearance,
        analysis_smoothed_xy_speed_m_s=speed,analysis_raw_valid_mask=valid,analysis_support_plane_abcd=plane,
        proposed_main_frame_intervals=np.asarray([[row['source_frame_interval_inclusive'][0],row['source_frame_interval_inclusive'][1]]for row in schedule]),
        proposed_active_hand_index=np.asarray([row['active_hand_index']for row in schedule]),schedule_is_ground_truth=np.asarray(False))
    a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output.with_suffix('.npz'),**payload)
    report={'schema':'fourth_six_object_contact_schedule_evidence.v2','scope':'Independent read-only evidence audit. Main intervals are candidate proposals, not measured contact/grasp/release ground truth. No physics or source/model/reference/config mutation.',
        'reference':str(a.reference),'reference_sha256':sha(a.reference),'geometry_model':str(a.geometry_model),'geometry_model_sha256':sha(a.geometry_model),'source_script_sha256':sha(__file__),
        'frames':n,'duration_s':float(times[-1]-times[0]),'raw_object_valid_counts':valid.sum(0).tolist(),'raw_human_observed_count':int(source['human_observed_mask'].sum()),
        'original_arm_event_merge_gap_s':.55,'original_arm_event_count':len(all_events),'original_arm_events':all_events,
        'source_plane_abcd':plane.tolist(),'initial_height_baselines':baseline,'schedule':schedule,
        'heuristic_confidence_rule':'0.25+0.20*candidateCoverage+0.15*rawPoseCoverage+0.10*status1or5Fraction+0.15*netXY>4cm+0.10*proxyLift>2cm; -0.12 for delayed final basin; -0.10 for >10% deep proxy-table overlap; capped0.85. Not a probability.',
        'source_candidates_preserved_in_npz':True,'analysis_gap_policy':'No interpolation across invalid raw object observations. XY speed uses five-frame coordinate median only inside continuous valid blocks.',
        'source_status_meanings':{'0':'missing_or_failed','1':'silhouette_constrained_5dof','2':'occluded_outer_boundary','3':'unreliable_silhouette_fit','4':'pose_jump_or_constraint_boundary','5':'partial_visibility_prior_assisted_estimate'},
        'planning_cautions':['A source proximity candidate can be brush/preparation or a nearby object; it does not measure grip force.','Final observed object pose may be after later plates were stacked; last-valid timestamps are not release timestamps.','A nonzero height over the original table is consistent with resting on the growing plate stack.','Raw-pose continued motion after a candidate ends cannot by itself distinguish retained grip, supported slide, settling or tracking error.','Predictive acquire belongs in a separate plan; changing raw candidate arrays would erase evidence.']}
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    compact={'schema':'fourth_six_object_contact_schedule_candidate.v1','state':'candidate_only_not_ground_truth_or_physics_validated',
        'evidence_json':str(a.output),'evidence_sha256':sha(a.output),'preserved_source_arrays_npz':str(a.output.with_suffix('.npz')),
        'time_origin':'source time_s; interval bounds are inclusive source observation times',
        'events':[{'object_id':row['object_id'],'hand':row['active_hand'],'hand_index':row['active_hand_index'],
            'main_candidate_ownership_interval_s':row['proposed_main_grab_release_interval_s'],
            'source_frame_interval_inclusive':row['source_frame_interval_inclusive'],
            'main_interval_confidence_heuristic_not_probability':row['confidence_heuristic_not_probability'],
            'release_boundary_confidence':row['release_boundary_confidence_label'],
            'predictive_acquire':row['predictive_acquire_planning_hint'],
            'release_requires_actual_support_or_validated_handoff':True,'cautions':row['cautions']}for row in schedule],
        'controller_integration_required':['Keep raw candidate arrays unchanged; use this separate ownership plan for predictive acquire.','Do not start the next object on one hand until the previous actual grasp/release state permits it.','Do not equate candidate-end time with finger opening: descent and support/contact observations must gate release.'],
        'source_reference_sha256':sha(a.reference)}
    a.output.with_suffix('.candidate.json').write_text(json.dumps(compact,indent=2)+'\n')
    try:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(6,2,figsize=(15,14),sharex=True)
        for oi,row in enumerate(schedule):
            left,right=axes[oi];left.plot(times,height[:,oi]*100,label='origin rise',alpha=.6);left.plot(times,relative_clearance[:,oi]*100,label='proxy bottom rise')
            left.axhline(row['late_observed_proxy_level_above_initial_m']*100,color='green',ls='--',lw=.8,label='late observed proxy level')
            right.plot(times,speed[:,oi],label='raw-block median XY speed',lw=1.)
            for axis in [left,right]:
                for start,stop in ranges(~valid[:,oi]):axis.axvspan(times[start],times[min(stop,n-1)],color='.7',alpha=.3)
                for side,color in [(0,'teal'),(1,'purple')]:
                    for start,stop in ranges(candidate[:,side,oi]):axis.axvspan(times[start],times[min(stop,n-1)],color=color,alpha=.2)
                start,end=row['proposed_main_grab_release_interval_s'];axis.axvspan(start,end,color='gold',alpha=.18);axis.grid(alpha=.2)
            left.set_ylabel(f'object{oi+1} / cm');right.set_ylabel('m/s');left.axhline(2.,color='k',ls=':',lw=.7)
        axes[0,0].legend(fontsize=8);axes[0,1].legend(fontsize=8);axes[-1,0].set_xlabel('source time / s');axes[-1,1].set_xlabel('source time / s')
        fig.suptitle('Raw source evidence only: grey=missing pose; teal/purple=left/right candidates; gold=main proposal',fontsize=12);fig.tight_layout();fig.savefig(a.output.with_suffix('.png'),dpi=150);plt.close(fig)
    except ImportError:pass
    print(json.dumps({'output':str(a.output),'arm_events':len(all_events),'schedule':[{'object_id':row['object_id'],'hand':row['active_hand'],'core_s':row['proposed_main_grab_release_interval_s'],'candidate_coverage':row['candidate_coverage_fraction'],'confidence':row['confidence_heuristic_not_probability'],'lift_max_m':row['raw_during'].get('maximum_proxy_bottom_rise_m'),'final_basin_s':row['first_final_position_basin_time_s'],'cautions':row['cautions']}for row in schedule]}),flush=True)


if __name__=='__main__':main()
