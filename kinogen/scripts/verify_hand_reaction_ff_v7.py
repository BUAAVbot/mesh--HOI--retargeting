"""Numerical fixture only: two geometry orders and real settled contact forces."""
from pathlib import Path
import json
import numpy as np
import mujoco as mj
from fourth_hand_reaction_ff_v7 import HandReactionFeedforward

records=[]
for object_first in (False,True):
    robot='''<body name="robot" pos="0 0 .1"><freejoint name="root"/>
      <geom name="sole" type="box" size=".1 .15 .1" mass="50"/>
      <body name="rh_fixture" pos="0 0 .2"><joint name="arm" axis="0 1 0" damping="1"/>
        <geom name="shelf" type="box" pos=".2 0 0" size=".12 .08 .02" mass=".5"/></body></body>'''
    obj='''<body name="target_object_01" pos=".2 0 .36"><freejoint name="obj"/>
      <geom name="load" type="box" size=".04 .04 .02" mass="1"/></body>'''
    xml='''<mujoco><option timestep=".001" gravity="0 0 -9.81" integrator="implicitfast"/>
      <default><geom friction="1 .005 .0001" solref=".005 1"/></default>
      <worldbody><geom type="plane" size="2 2 .1"/>'''+((obj+robot)if object_first else(robot+obj))+'''</worldbody>
      <actuator><position name="arm_motor" joint="arm" kp="4000" kv="40"/></actuator></mujoco>'''
    m=mj.MjModel.from_xml_string(xml);d=mj.MjData(m)
    for _ in range(3000):mj.mj_step(m,d)
    mj.mj_forward(m,d)
    obj_id=m.body('target_object_01').id;hand_id=m.body('rh_fixture').id
    ff=HandReactionFeedforward(m,[obj_id],filter_tau_s=0.)
    before={name:getattr(d,name).copy()for name in ['qpos','qvel','ctrl','qfrc_applied','xfrc_applied']}
    geom_before=m.geom_size.copy();raw=ff.update(d).copy()
    independent=np.zeros(m.nv);zforce=0.;orders=[]
    for patch in ff.records[-1]['patches']:
        force=np.asarray(patch['force_on_hand_world_N']);zforce+=force[2];orders.append(patch['hand_geom_order'])
        mj.mj_applyFT(m,ff.kin,force,np.asarray(patch['torque_on_hand_world_Nm']),np.asarray(patch['contact_position_world_m']),hand_id,independent)
    err=float(np.max(abs(raw-independent)))
    root_dof=int(m.jnt_dofadr[m.joint('root').id]);arm_dof=int(m.jnt_dofadr[m.joint('arm').id])
    assert err<1e-10
    assert abs(zforce+9.81)<.03,(object_first,zforce)
    assert abs(raw[root_dof+2]+9.81)<.03
    assert raw[arm_dof]>1.8,raw[arm_dof]
    for name,value in before.items():assert np.array_equal(getattr(d,name),value),name
    assert np.array_equal(m.geom_size,geom_before)
    filtered=HandReactionFeedforward(m,[obj_id],filter_tau_s=.1);filtered.update(d)
    max_filter_err=0.
    for _ in range(20):
        previous=filtered.filtered.copy();previous_time=float(d.time);mj.mj_step(m,d)
        value=filtered.update(d).copy();expected=previous+(-np.expm1(-(d.time-previous_time)/.1))*(filtered.raw-previous)
        max_filter_err=max(max_filter_err,float(np.max(abs(value-expected))))
    assert max_filter_err<1e-12
    records.append({'object_declared_first':object_first,'hand_geometry_orders':sorted(set(orders)),
                    'total_force_on_hand_z_N':float(zforce),'root_z_reaction_N':float(raw[root_dof+2]),
                    'arm_reaction_Nm':float(raw[arm_dof]),'mj_applyFT_max_difference':err,
                    'exact_exponential_filter_max_error':max_filter_err,'input_state_and_geometry_unchanged':True})
assert {r['hand_geometry_orders'][0]for r in records}=={1,2}
report={'success':True,'scope':'Isolated numerical fixtures, not a grasp trial or success claim.',
        'inverse_statics_sign':'tau_motor = bias - reaction; negative hand root-z reaction increases predicted floor load.',
        'records':records}
path=Path(__file__).resolve().parents[1]/'reports/hand_reaction_ff_v7_fixture.json'
path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
