"""Analytical force/wrench/motor-budget controls for the standalone LP."""
from pathlib import Path
import json
import numpy as np
from fourth_contact_wrench import pyramid_rays,wrench_lp,two_point_gravity_axis_certificate

points=np.asarray([[.05,0,0],[-.025,np.sqrt(3)*.025,0],[-.025,-np.sqrt(3)*.025,0]])
centre=np.asarray([0,0,.05]);mass=.24;gravity=np.asarray([0.,0.,-9.81]);external=np.r_[mass*gravity,np.zeros(3)]
columns=[];normal=[]
# Native frame columns are world +Z normal, +X/+Y tangents.
frame=np.asarray([[0.,1,0],[0,0,1],[1,0,0]])
for i,point in enumerate(points):
    for ray in pyramid_rays(3,[.5,.5]):
        force=frame@ray[:3];torque=np.cross(point-centre,force);columns.append(np.r_[force,torque])
        row=np.zeros(3);row[i]=ray[0];normal.append(row)
N=np.asarray(normal).T;M=-.1*N
good=wrench_lp(columns,normal,external,normal_cap_N=.8,motor_torque_matrix=M,motor_gravity_bias_Nm=np.zeros(3),motor_cap_Nm=.1)
badN=wrench_lp(columns,normal,external,normal_cap_N=.7)
badMotor=wrench_lp(columns,normal,external,normal_cap_N=None,motor_torque_matrix=M,motor_gravity_bias_Nm=np.zeros(3),motor_cap_Nm=.05)
expected=mass*9.81/3
checks={'three_point_analytic_support_feasible':good['feasible'],
        'minimum_force_matches_mass_over_three':abs(good['minimum_maximum_digit_normal_N']-expected)<1e-7,
        'per_digit_budget_too_small_rejected':not badN['feasible'],
        'projected_motor_budget_too_small_rejected':not badMotor['feasible'],
        'required_motor_torque_matches_lever_times_force':max(abs(np.asarray(good['required_motor_torque_Nm'])-.1*expected))<1e-7}
certificate=two_point_gravity_axis_certificate([[0,0,0],[0,.02,0]],[.1,0,.05],mass,gravity)
checks['two_point_free_axis_moment_matches_analytic']=abs(certificate['absolute_unbalanced_axis_torque_Nm']-mass*9.81*.1)<1e-10
checks={key:bool(value)for key,value in checks.items()}
report={'schema':'fourth.contact_wrench_analytic_controls.v1','checks':checks,'expected_support_per_digit_N':expected,
        'feasible_tripod_solution':good,'too_small_normal_budget':badN,'too_small_motor_budget':badMotor,
        'two_point_axis_certificate':certificate,'physical_rollout_performed':False,'status':'passed'if all(checks.values())else'failed'}
out=Path(__file__).resolve().parents[1]/'reports/fourth_contact_wrench_analytic_controls.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
if report['status']!='passed':raise SystemExit(2)
