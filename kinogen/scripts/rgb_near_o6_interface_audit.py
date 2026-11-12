"""Verify O6 six-command ABI, command lookup provenance and zero passive drives."""
from pathlib import Path
import argparse,json,hashlib

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--hand-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Refusing overwrite '+str(a.output))
    import numpy as np,mujoco as mj
    m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);zeros=np.flatnonzero(np.all(m.actuator_gainprm==0,axis=1)&np.all(m.actuator_biasprm==0,axis=1))
    force=[]
    for i in range(3):
        d.ctrl[:]=np.linspace(-.3,1.3,m.nu)*(i+1);d.qvel[:]=.1*i
        mj.mj_forward(m,d);force.append(float(np.max(abs(d.actuator_force[zeros]))))
    if any(force):raise ValueError('Passive channel emits actuator force')
    lookup=[]
    for side in ['LEFT','RIGHT']:
        path=next((a.hand_root/side).glob('*.json'));cal=json.loads(path.read_text())
        for name,jc in cal['joints'].items():
            if jc.get('passive'):continue
            arr=np.asarray(jc['angle_rad']);inc=np.asarray(jc['increasing_rad']);dec=np.asarray(jc['decreasing_rad'])
            if arr.shape!=(256,)or not np.all(np.isfinite(arr)):raise ValueError('Unexpected command table')
            lookup.append({'side':side,'joint':name,'sdk_channel':jc['sdk_channel'],'motor_index':jc['motor_index'],
                           'q_at_commands':{str(k):float(arr[k])for k in [0,64,128,192,255]},'max_hysteresis_rad':float(np.max(abs(inc-dec))),
                           'status':jc['calibration_status'],'lookup_source_sha256':sha(path),'monotonic_decreasing':bool(np.all(np.diff(arr)<=1e-9))})
    report={'model':str(a.model),'model_sha256':sha(a.model),'nu_abi':m.nu,'zero_force_channel_ids':zeros.tolist(),
            'passive_actuator_max_force_for_three_ctrl_qvel_states':force,'direct_powered_channels':m.nu-len(zeros),
            'joint_equality_count':int(sum(m.eq_type==mj.mjtEq.mjEQ_JOINT)),'command_lookups':lookup,
            'interpretation':['Controller target angles must be looked up from SDK command profiles; command255 is baseline open.',
                              'Reported passive actuator forces are direct actuator channels only. Mimic constraint and contact reactions may be nonzero and must be logged separately.',
                              'This is a forward-model ABI diagnostic, not a physical roll-out or empirical torque calibration.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ['command_lookups','interpretation']}))

if __name__=='__main__':main()
