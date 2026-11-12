"""Read recorded intervals to distinguish survival from passing state bounds.

No dynamics, state repairs, model edits, training or task acceptance decisions.
The input trial remains unchanged; JSON goes to stdout or a new --output file.
"""
from pathlib import Path
import argparse,hashlib,json
import numpy as np


STATE_GATES={
    'root_horizontal_tracking':('root_horizontal_error_m','max_root_xy_error_m'),
    'root_height_tracking':('root_height_error_m','max_root_z_error_m'),
    'root_orientation_tracking':('root_orientation_error_rad','max_root_orientation_rad'),
    'joint_limits':('joint_limit_violation_rad','max_joint_limit_violation_rad'),
}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def first_event(values,bad,times,source,time_basis,threshold=None):
    values=np.asarray(values);bad=np.asarray(bad,dtype=bool)
    indices=np.flatnonzero(bad)
    if not len(indices):return None
    i=int(indices[0]);value=float(values[i])
    return {'interval_index':i,'execution_time_s':float(times[i]),'source_time_s':float(source[i]),
            'value':value if np.isfinite(value)else None,'threshold':threshold,'time_basis':time_basis}


def audit(trial,model_path=None):
    trial=Path(trial).resolve();report_path=trial/'report.json';trace_path=trial/'trace.npz';contact_path=trial/'contacts.npz'
    report=json.loads(report_path.read_text());thresholds=report['quality_thresholds']
    trace=dict(np.load(trace_path,allow_pickle=False));contact=dict(np.load(contact_path,allow_pickle=False))
    time=np.asarray(trace['time'],float);n=len(time)-1;source=np.asarray(trace['source_time_s'],float)
    if n<1 or source.shape!=time.shape or not np.isfinite(time).all() or not np.isfinite(source).all() or np.any(np.diff(time)<=0):raise ValueError('Invalid state/source timeline')
    reference_execution=np.asarray(trace['reference_execution_time_s'],float);reference_source=np.asarray(trace['reference_source_time_s'],float)
    if reference_execution.shape!=reference_source.shape or np.any(np.diff(reference_execution)<=0) or np.any(np.diff(reference_source)<=0):raise ValueError('Invalid reference clock map')
    if not np.allclose(source,np.interp(time,reference_execution,reference_source),atol=1e-8,rtol=0):raise ValueError('Recorded source clock differs from reference interpolation')
    input_hashes={name:sha(path)for name,path in [('report',report_path),('trace',trace_path),('contacts',contact_path)]}
    if report.get('trace',{}).get('sha256')not in (None,input_hashes['trace']):raise ValueError('Recorded trace hash differs from trial report')
    end_time=time[1:];end_source=source[1:];start_time=time[:-1];start_source=source[:-1]
    gates={};bad_by_gate={}
    for gate,(key,threshold_key)in STATE_GATES.items():
        values=np.asarray(trace[key],float);threshold=float(thresholds[threshold_key])
        if values.shape!=(n,) or not np.isfinite(threshold) or threshold<0:raise ValueError('Invalid state gate '+gate)
        bad=(values>threshold)|~np.isfinite(values);bad_by_gate[gate]=bad
        gates[gate]={'threshold':threshold,'comparison':'value > threshold, or nonfinite','passed_all_recorded_intervals':bool(not bad.any()),
                     'first_exceedance':first_event(values,bad,end_time,end_source,'post-integration state: trace.time[interval_index+1]',threshold)}
    interval=np.asarray(contact['interval_index'],int);solver=np.asarray(contact['solver_time_s'],float);distance=np.asarray(contact['signed_distance_m'],float)
    if interval.ndim!=1 or solver.shape!=interval.shape or distance.shape!=interval.shape or np.any(interval<0) or np.any(interval>=n):raise ValueError('Invalid contact interval records')
    if not np.isfinite(solver).all() or not np.allclose(solver,start_time[interval],atol=1e-8,rtol=0):raise ValueError('Contact solver time is not the corresponding interval start')
    penetration=np.zeros(n);np.maximum.at(penetration,interval,np.maximum(-distance,0.))
    nonfinite_contact=np.zeros(n,dtype=bool);np.logical_or.at(nonfinite_contact,interval,~np.isfinite(distance))
    threshold=float(thresholds['max_penetration_m']);bad=(penetration>threshold)|nonfinite_contact
    bad_by_gate['penetration']=bad
    gates['penetration']={'threshold':threshold,'comparison':'maximum contact penetration > threshold, or nonfinite contact distance',
                          'aggregation':'np.maximum.at over contacts.interval_index; no-contact intervals have zero recorded penetration',
                          'passed_all_recorded_intervals':bool(not bad.any()),
                          'first_exceedance':first_event(penetration,bad,start_time,start_source,'solver contact geometry: trace.time[interval_index] / contacts.solver_time_s',threshold)}
    union=np.logical_or.reduce(list(bad_by_gate.values()));failed_intervals=np.flatnonzero(union)
    first_interval=int(failed_intervals[0])if len(failed_intervals)else None
    events=[dict(gate=name,**row['first_exceedance'])for name,row in gates.items()if row['first_exceedance']is not None]
    events.sort(key=lambda row:(row['execution_time_s'],row['interval_index'],row['gate']))
    first=events[0]if events else None
    prefix_end=float(time[first_interval])if first_interval is not None else float(time[-1])
    prefix_source=float(source[first_interval])if first_interval is not None else float(source[-1])
    additional={}
    for key in ('qpos','qvel'):
        values=np.asarray(trace[key]);finite=np.isfinite(values).reshape(len(values),-1).all(axis=1)
        if len(finite)!=n+1:raise ValueError('Invalid '+key+' state count')
        indices=np.flatnonzero(~finite)
        additional[key+'_finite']={'passed_all_recorded_states':bool(finite.all()),'first_nonfinite_state':None if not len(indices)else{'state_index':int(indices[0]),'execution_time_s':float(time[indices[0]]),'source_time_s':float(source[indices[0]])}}
    for key in ('qfrc_applied','xfrc_applied'):
        if key not in trace:additional[key+'_zero']={'available':False};continue
        values=np.asarray(trace[key]);peak=np.max(np.abs(values).reshape(n,-1),axis=1);bad=(peak>0)|~np.isfinite(peak)
        additional[key+'_zero']={'available':True,'passed_all_recorded_intervals':bool(not bad.any()),'first_exceedance':first_event(peak,bad,start_time,start_source,'applied force input: trace.time[interval_index]',0.),'evidence':'recorded applied-force arrays; not an independent state-write instrument'}
    force=np.asarray(trace['actuator_force']);passive=np.asarray(trace['passive_actuator_ids'],int)
    if force.ndim!=2 or len(force)!=n:raise ValueError('Invalid actuator force records')
    peak=np.max(np.abs(force[:,passive]),axis=1)if len(passive)else np.zeros(n);bad=(peak>=1e-10)|~np.isfinite(peak)
    additional['passive_compatibility_zero_force']={'passed_all_recorded_intervals':bool(not bad.any()),'comparison':'absolute force < 1e-10','first_exceedance':first_event(peak,bad,start_time,start_source,'solver actuator force: trace.time[interval_index]',1e-10)}
    model_path=Path(model_path)if model_path else trial/'model.mjb'
    if model_path.exists():
        try:import mujoco as mj
        except ImportError:additional['actuator_effort_limits']={'interval_audit_available':False,'reported_aggregate_passed':report.get('checks',{}).get('actuator_effort_limits'),'reason':'MuJoCo unavailable; model not loaded'}
        else:
            model_sha=sha(model_path)
            if report.get('model',{}).get('sha256')not in(None,model_sha):raise ValueError('Model hash differs from trial report')
            model=mj.MjModel.from_binary_path(str(model_path))if model_path.suffix=='.mjb'else mj.MjModel.from_xml_path(str(model_path))
            powered=np.asarray(trace['powered_actuator_ids'],int);ids=powered[model.actuator_forcelimited[powered].astype(bool)]
            excess=np.maximum(model.actuator_forcerange[ids,0]-force[:,ids],force[:,ids]-model.actuator_forcerange[ids,1])
            peak=np.max(excess,axis=1)if len(ids)else np.full(n,-np.inf);bad=(peak>1e-7)|np.isnan(peak)
            additional['actuator_effort_limits']={'interval_audit_available':True,'model_sha256':model_sha,'actual_force_limited_powered_channels':ids.tolist(),'passed_all_recorded_intervals':bool(not bad.any()),'first_exceedance':first_event(peak,bad,start_time,start_source,'solver actuator force: trace.time[interval_index]',1e-7),'value_meaning':'force excess beyond unchanged actuator_forcerange, with1e-7 tolerance'}
    else:additional['actuator_effort_limits']={'interval_audit_available':False,'reported_aggregate_passed':report.get('checks',{}).get('actuator_effort_limits'),'reason':'Frozen model unavailable'}
    additional['solver_warnings']={'reported_finite_no_solver_warning':report.get('checks',{}).get('finite_no_solver_warning'),'first_warning_time_available':False,'reason':'Trace has no per-interval warning counters; do not invent a warning timestamp'}
    return {'schema':'rgb145.recorded_progress_audit.v1','trial':str(trial),'input_sha256':input_hashes,'script_sha256':sha(__file__),
            'timing_contract':{'state_errors':'interval i is observed after integration at trace.time[i+1] and source_time_s[i+1]',
                               'contact_geometry_and_forces':'interval i solver quantities describe trace.time[i], as recorded in contacts.solver_time_s; not the next state timestamp',
                               'execution_time_origin':'trace clock; warm-up, if any, is separately recorded by the original trial',
                               'valid_prefix_end':'start of the first interval with any selected bound failure; all earlier intervals pass these selected bounds'},
            'first_state_bound_exceedance':first,'per_gate':gates,
            'selected_state_bounds_prefix':{'gates':list(gates),'passed_all_recorded_intervals':first_interval is None,'first_failing_interval_index':first_interval,
                                            'complete_passing_intervals':first_interval if first_interval is not None else n,'execution_duration_s':prefix_end-float(time[0]),'end_execution_time_s':prefix_end,'source_time_reached_s':prefix_source,
                                            'claim':'Only these recorded instantaneous state/contact bounds; not a complete task-quality prefix acceptance'},
            'recorded_survival':{'physics_steps':n,'execution_duration_s':float(time[-1]-time[0]),'source_time_reached_s':float(source[-1]),'is_success_metric':False},
            'additional_recorded_checks':additional,
            'full_task_result':{'status':report.get('status'),'simulation_passed':report.get('simulation_passed'),'end_reason':report.get('end_reason'),'checks':report.get('checks')},
            'limitations':['Aggregate body joint RMS median/p95, stance coverage, final support and final seated load remain the original task checks; this audit does not replace them.',
                           'A passing instantaneous-bound prefix does not prove adequate support force, successful task progress, calibrated physical parameters or independent replay acceptance.',
                           'Initial root-error bounds are not independently reconstructed; the selected state-error series begins after the first integration step.',
                           'No physics integration, controller evaluation, state repair, parameter mutation or success-status modification is performed.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trial',type=Path,required=True);p.add_argument('--model',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.output and a.output.exists():raise FileExistsError(a.output)
    result=audit(a.trial,a.model)
    text=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
