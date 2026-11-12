"""Collect completed solver experiments without treating origin rise as success."""
from pathlib import Path
import hashlib,json
import numpy as np
root=Path(__file__).resolve().parents[1]
cases=[
 ('physics_o6_stiff_eq_C7_v1','raw continuous source scene','thin','C7'),
 ('physics_o6_stiff_eq_C8_locked_v1','raw continuous source scene','thin','C8 locked palm'),
 ('physics_o6_stiff_eq_C11_segments_v1','raw continuous source scene','thin','C11 actual contact segments'),
 ('physics_o6_stiff_C12_loadmargin_v1','raw continuous source scene','thin','C11 load1.08'),
 ('physics_o6_stiff_C13_measured100_v1','raw continuous source scene','thin','C11 measured100'),
 ('physics_single_plate_microstep_benchmark_v1','explicit reset PD free-root benchmark','thin','micro C11 old required5'),
 ('physics_single_plate_microstep_v2_cached','explicit reset PD free-root benchmark','thin','micro v5b cached'),
 ('physics_single_plate_microstep_v2_segments','explicit reset PD free-root benchmark','thin','micro v5b actual segments'),
 ('physics_single_plate_microstep_v7_reaction','explicit reset PD free-root benchmark','thin','measured reaction FF'),
 ('physics_single_plate_microstep_v8_probe','explicit reset PD free-root benchmark','thin','reaction FF plus4mm probe'),
 ('physics_single_plate_microstep_v9_pressure','explicit reset PD free-root benchmark','thin','reaction FF plus pressure memory and pause'),
 ('physics_single_plate_microstep_lip_v1','explicit reset PD free-root benchmark','lip invalid unclipped template','cached'),
 ('physics_single_plate_microstep_v7_lip_reaction','explicit reset PD free-root benchmark','lip invalid unclipped template','reaction FF'),
 ('physics_single_plate_microstep_v7_lip_runtime_v2','explicit reset PD free-root benchmark','lip legal yaw5mrad margin','reaction FF'),
 ('physics_single_plate_microstep_v7_lip_runtime_v3','explicit reset PD free-root benchmark','lip legal yaw17mrad margin','reaction FF'),
 ('physics_single_plate_microstep_v8_lip_runtime_v2','explicit reset PD free-root benchmark','lip legal yaw5mrad margin','reaction FF plus4mm probe'),
]
rows=[]
keys=['success','duration_s','termination','stable_3cm_support_max_duration_s',
      'max_tactile_rise_m','max_mimic_error_rad','hand_max_abs_actuator_force_Nm',
      'post_reset_object_state_writes','external_forces_zero',
      'measured_normal_budget_feasible_steps','body_controller','learned_leg_policy_used']
for name,initial,geometry,control in cases:
    trial=root/'outputs/fourth_full_sequence'/name
    row={'trial':str(trial),'initial_condition_class':initial,'object_geometry':geometry,'control':control,
         'clean_geometry_comparison': 'invalid' not in geometry,'report_available':(trial/'report.json').exists()}
    if row['report_available']:
        r=json.loads((trial/'report.json').read_text())
        row['metrics']={k:r.get(k) for k in keys}
        hp=trial/'grasp_history.npz'
        if hp.exists():
            h=np.load(hp);cols=list(map(str,h['columns']));values=h['values']
            key='object_min_clearance_from_table_plane_m'
            if key in cols:
                height=values[:,cols.index(key)];table=values[:,cols.index('table_normal_N')]
                row['sampled_lowest_collider_peak_m']=float(height.max())
                good=(height>=.03)&(table<.15)
                longest=0.;begin=None
                for t,yes in zip(values[:,0],good):
                    if yes:
                        if begin is None:begin=float(t)
                        longest=max(longest,float(t)-begin)
                    else:begin=None
                row['sampled_height_unloaded_continuous_s']=longest
                row['sampled_metrics_scope']='Runner5ms packet-end diagnostic. Full1ms native independent audit is separate; no task success inferred.'
        row['report_sha256']=hashlib.sha256((trial/'report.json').read_bytes()).hexdigest()
        row['model_sha256']=hashlib.sha256((trial/'effective_model.mjb').read_bytes()).hexdigest()
    rows.append(row)
out=root/'reports/fourth_solver_experiment_ledger_v2.json'
out.write_text(json.dumps({'scope':'Machine-readable experiment inventory. Original thin-hand collision geometry and0.4Nm native hard cap retained throughout. Explicit reset PD benchmarks are not claimed as SONIC policy or full raw-scene reproduction.',
  'quality_rule':'Object lowest collision point above tabletop30mm for2s with actual physical support. Origin rise is diagnostic only.',
  'invalid_lip_template_note':'Original lip v1 static XML template yaw1.30598 exceeded executed effective model upper1.2431454. Those completed actuator-clipped diagnostics are excluded from clean geometry comparisons; runtime v2/v3 use explicitly legal reset and closed joints.',
  'rows':rows},indent=2)+'\n')
print(json.dumps({'output':str(out),'completed_reports':sum(r['report_available']for r in rows),'declared_successes':[r['trial']for r in rows if r.get('metrics',{}).get('success')],'clean_comparison_successes':[r['trial']for r in rows if r['clean_geometry_comparison']and r.get('metrics',{}).get('success')],'latest_geometry_trials':[{k:r.get(k)for k in ['trial','sampled_lowest_collider_peak_m','sampled_height_unloaded_continuous_s','metrics']}for r in rows[-3:]]}),flush=True)
