"""Isolate an engineering 0.4 Nm hard cap on the 12 active O6 motors."""
from pathlib import Path
import argparse,hashlib,json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj

ROOT=Path(__file__).resolve().parents[1];A=ROOT/'outputs/fourth_full_sequence/assets'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,default=A/'vita_six_objects_newft_o6_calibrated_mimic_v2.xml')
p.add_argument('--output',type=Path,default=A/'vita_six_objects_newft_o6_calibrated_mimic_v3.xml');a=p.parse_args()
if a.output.exists():raise ValueError('Never overwrite a frozen torque variant')
xml=ET.parse(a.base).getroot();m=mj.MjModel.from_xml_path(str(a.base));changed=[]
for act in xml.find('actuator'):
    name=act.get('name','');aid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,name)
    if name.startswith(('motor_lh_','motor_rh_'))and np.any(m.actuator_gear[aid]!=0):
        act.set('forcelimited','true');act.set('forcerange','-.4 .4');changed.append(aid)
if len(changed)!=12:raise ValueError('Expected exactly 12 active O6 finger motors')
ET.SubElement(xml.find('custom'),'numeric',name='o6_engineering_hard_motor_torque_limit_Nm',data='.4')
a.output.write_text(ET.tostring(xml,encoding='unicode'),encoding='utf-8');n=mj.MjModel.from_xml_path(str(a.output))
preserve=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_range','dof_armature','dof_damping',
          'geom_pos','geom_quat','geom_size','geom_type','geom_contype','geom_conaffinity','eq_data','eq_solref','eq_solimp',
          'actuator_gainprm','actuator_biasprm','actuator_gear']
for k in preserve:
    if not np.array_equal(getattr(m,k),getattr(n,k)):raise ValueError('Unexpected change '+k)
others=[i for i in range(m.nu)if i not in changed]
if not np.array_equal(m.actuator_forcerange[others],n.actuator_forcerange[others]):raise ValueError('Changed nonhand actuator cap')
if not np.all(n.actuator_forcelimited[changed])or not np.allclose(n.actuator_forcerange[changed],[-.4,.4],atol=0,rtol=0):raise ValueError('Hard torque cap not compiled')
report={'schema':'fourth.o6_engineering_hard_motor_torque_variant.v1','model':str(a.output),'model_sha256':sha(a.output),
        'base_model':str(a.base),'base_sha256':sha(a.base),'changed_active_finger_motor_ids':changed,
        'changed_motor_names':[n.actuator(i).name for i in changed],'hard_forcerange_Nm':[-.4,.4],
        'preserved_exactly':preserve,'all_28_body_motor_caps_unchanged':True,'contract_unchanged':'o6_calibrated_linear_joint_contracts_v2.urdf',
        'interpretation':'Engineering simulation hard motor-torque cap; not a measured hardware rating. Contact impact forces can still exceed static feedback goals.',
        'candidate_geometry_unchanged':True,'physical_rollout_performed':False}
a.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report),flush=True)
