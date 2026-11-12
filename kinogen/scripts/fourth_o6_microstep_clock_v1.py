"""Explicit 5 ms control packets with five 1 ms physical integration steps.

Only controller construction temporarily uses its checkpoint training dt check;
all actual model integration is at 1 ms. Delay packets remain 3 x 5 ms = 15 ms.
"""
import numpy as np
import mujoco as mj
from fourth_sonic_controller import Controller as SourceController

CONTROL_DT = .005
INTEGRATION_DT = .001
SUBSTEPS = 5
_stats = {}


def Controller(model, reference, config=None):
    actual = float(model.opt.timestep)
    if abs(actual-INTEGRATION_DT)>1e-12:
        raise ValueError('Microstep model must explicitly store1ms physical dt')
    model.opt.timestep = CONTROL_DT
    try:
        result = SourceController(model, reference, config)
    finally:
        model.opt.timestep = actual
    return result


def _entry(model, data):
    key = (id(model),id(data))
    if key not in _stats:
        equalities=[i for i in range(model.neq)if model.eq_type[i]==mj.mjtEq.mjEQ_JOINT]
        _stats[key]={'eq':equalities,'packet_peak':0.,'overall_peak':0.,'microsteps':0,'trace':None}
    return _stats[key]


def step_packet(model, data):
    if abs(float(model.opt.timestep)-INTEGRATION_DT)>1e-12:
        raise RuntimeError('Actual integration clock changed')
    stats = _entry(model,data)
    stats['packet_peak'] = 0.
    initial = float(data.time)
    for _ in range(SUBSTEPS):
        applied_ctrl=data.ctrl.copy()
        mj.mj_step(model,data)
        errors=[abs(float(data.qpos[int(model.jnt_qposadr[int(model.eq_obj1id[i])])]
                    -model.eq_data[i,1]*data.qpos[int(model.jnt_qposadr[int(model.eq_obj2id[i])])]))
                for i in stats['eq']]
        peak=max(errors,default=0.)
        stats['packet_peak']=max(stats['packet_peak'],peak)
        stats['overall_peak']=max(stats['overall_peak'],peak)
        stats['microsteps']+=1
        trace=stats['trace']
        if trace is not None:
            trace['time_s'].append(float(data.time));trace['qpos'].append(data.qpos.copy());trace['qvel'].append(data.qvel.copy())
            trace['ctrl'].append(applied_ctrl);trace['actuator_force'].append(data.actuator_force.copy())
            trace['qfrc_applied'].append(data.qfrc_applied.copy());trace['xfrc_applied'].append(data.xfrc_applied.copy())
    if abs(float(data.time)-initial-CONTROL_DT)>1e-9:
        raise RuntimeError('Numerical reset or inconsistent microstep clock')


def packet_mimic_peak(model, data):
    return float(_entry(model,data)['packet_peak'])


def numerical_report(model, data):
    s=_entry(model,data)
    return {'control_packet_dt_s':CONTROL_DT,'physical_integration_dt_s':INTEGRATION_DT,
            'physical_substeps_per_packet':SUBSTEPS,'nominal_actuator_delay_s':.015,
            'policy_dt_s':.02,'finger_feedback_dt_s':.02,
            'integration_steps':s['microsteps'],'microstep_max_mimic_rad':s['overall_peak'],
            'state_recording':'Every1ms physical state and repeated heldctrl are recorded when start_micro_recording is enabled.'}


def start_micro_recording(model,data,state_spec,initial_state):
    trace={k:[]for k in ['time_s','qpos','qvel','ctrl','actuator_force','qfrc_applied','xfrc_applied']}
    trace['time_s']=[float(data.time)];trace['qpos']=[data.qpos.copy()];trace['qvel']=[data.qvel.copy()]
    trace['initial_state']=np.asarray(initial_state).copy();trace['state_spec']=np.asarray(state_spec)
    _entry(model,data)['trace']=trace


def write_micro_trace(path,model,data):
    trace=_entry(model,data)['trace']
    if trace is None:raise RuntimeError('Microstep recorder not initialized')
    np.savez_compressed(path,**{k:np.asarray(v)for k,v in trace.items()})
