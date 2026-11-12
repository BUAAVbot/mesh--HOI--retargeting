"""Freeze a MuJoCo ankle-PD gain variant without changing source physics geometry."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'outputs/fourth_vita_demo/assets'
PITCH=('left_ankle_pitch_joint','right_ankle_pitch_joint')
ROLL=('left_ankle_roll_joint','right_ankle_roll_joint')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',choices=['v1','v2'],default='v1')
    args=ap.parse_args()
    name='vita_object_05_dynamic'+('_v2'if args.base=='v2'else '')
    src=ASSETS/(name+'.xml')
    target=ASSETS/(name+'_ankle100.xml')
    root=ET.parse(src).getroot()
    before={}
    after={}
    actuators={x.get('name'):x for x in root.findall('./actuator/position')}
    for joint in PITCH+ROLL:
        key='motor_'+joint
        if key not in actuators:raise ValueError('Missing actuator '+key)
        motor=actuators[key]
        before[joint]={'kp':float(motor.get('kp')),'kv':float(motor.get('kv'))}
        if joint in PITCH:
            motor.set('kp','100');motor.set('kv','5')
        else:
            motor.set('kp',str(2*float(motor.get('kp'))))
            motor.set('kv',str(2*float(motor.get('kv'))))
        after[joint]={'kp':float(motor.get('kp')),'kv':float(motor.get('kv'))}
    original=mujoco.MjModel.from_xml_path(str(src))
    text=ET.tostring(root,encoding='unicode')
    variant=mujoco.MjModel.from_xml_string(text)
    if (original.nq,original.nv,original.nu)!=(variant.nq,variant.nv,variant.nu):
        raise RuntimeError('Gain-only variant changed model dimensions')
    target.write_text(text,encoding='utf-8')
    report={'schema':'fourth_vita_gain_variant.v1','status':'compiled','base':str(src),
            'base_sha256':sha(src),'variant':str(target),'variant_sha256':sha(target),
            'nq':variant.nq,'nv':variant.nv,'nu':variant.nu,
            'before':before,'after':after,
            'what_changed':'Only 4 ankle position actuator kp/kv XML attributes',
            'control_assumption_not_original_training_contract':True,
            'geometry_mass_inertia_friction_unchanged':True}
    path=ASSETS/(name+'_ankle100_report.json')
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'variant':str(target),'report':str(path),'nq':variant.nq,'nv':variant.nv,'nu':variant.nu}))

if __name__=='__main__':main()
