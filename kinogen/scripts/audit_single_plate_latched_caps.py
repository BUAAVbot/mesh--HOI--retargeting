"""Cold-geometry audit: is an old normal allocation still sufficient now?"""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from fourth_o6_load_hysteresis_cached_c10 import O6CoupledReferenceContactFeedback
from fourth_contact_wrench_balanced_c9 import contact_columns,wrench_lp
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
out=a.output;m=mj.MjModel.from_binary_path(str(out/'effective_model.mjb'))
raw=np.load(out/'physical_trace.npz');ref=dict(np.load(out/'source_reference.npz',allow_pickle=False))
cfg=json.loads((out/'finger_config.json').read_text());h=O6CoupledReferenceContactFeedback(m,ref,cfg)
P=np.zeros((m.nv,6));P[h.hands[1]['va']]=h.hands[1]['coupling_matrix']
report=json.loads((out/'report.json').read_text());obj=m.body('target_object_01').id
d=mj.MjData(m);rows=[]
for time in [12.,14.,16.,18.,19.,20.,21.,22.,24.,25.]:
    allocations=[x for x in report['load_allocations']if x['time_s']<=time]
    if not allocations:continue
    old=allocations[-1]['normal_targets_N'];caps=np.array([old[x]for x in ['thumb','index','middle','ring','pinky']])
    i=int(np.argmin(abs(raw['time_s']-time)));d.qpos[:]=raw['qpos'][i];d.qvel[:]=0.;mj.mj_forward(m,d)
    c=contact_columns(m,d,obj,tuple('rh_'+x for x in ['thumb','index','middle','ring','pinky']),P,max_gap_m=0.)
    args=dict(columns=c['columns'],normal_rows=c['normal_rows'],gravity_wrench=np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],motor_torque_matrix=c['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@d.qfrc_bias)
    any_lp=wrench_lp(**args,normal_cap_N=4.6,motor_cap_Nm=.35)
    old_lp=wrench_lp(**args,normal_cap_N=caps,motor_cap_Nm=.4)
    rows.append({'time_s':float(raw['time_s'][i]),'old_normal_targets_N':caps.tolist(),
        'global_motor35_normal4p6_feasible':any_lp['feasible'],'old_normal_caps_motor40_feasible':old_lp['feasible'],
        'current_global_minimax_normal_N':any_lp.get('normal_per_digit_N'),
        'old_caps_residual_force_equivalent_L1':old_lp.get('residual_force_equivalent_L1'),
        'actual_geometric_patch_count':len(c['patches'])})
result={'scope':'Cold geometry from exact recorded1ms qpos, not warmstart replay and not a success certificate.',
        'old_target_infeasible_while_global_feasible_samples':sum(x['global_motor35_normal4p6_feasible']and not x['old_normal_caps_motor40_feasible']for x in rows),'samples':rows}
(out/'independent_latched_caps.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
