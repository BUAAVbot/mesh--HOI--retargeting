"""Version the O6 URDF's linear mimic baseline without changing body plant.

The JSON quadratic runtime curves are retained as provenance, not applied.
Thumb zero rotation is taken once from URDF origin. Ten passive coordinates
are driven through true MuJoCo joint equality constraints, never duplicate PD.
"""
from pathlib import Path
import argparse,copy,hashlib,json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'outputs/fourth_full_sequence/assets'
SOURCE=ROOT/'data/linkerhand_o6_06_20261027/06'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(a):return' '.join(format(float(v),'.17g')for v in np.asarray(a).reshape(-1))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model',type=Path,default=A/'vita_six_objects_newft_source_table_segmented_hand_v1.xml')
    parser.add_argument('--output',type=Path,default=A/'vita_six_objects_newft_o6_calibrated_mimic_v2.xml')
    parser.add_argument('--contract',type=Path,default=A/'o6_calibrated_linear_joint_contracts_v2.urdf')
    args=parser.parse_args()
    if args.output.exists()or args.contract.exists():raise ValueError('Never overwrite frozen O6 assets')
    xml=ET.parse(args.base_model).getroot();old=mj.MjModel.from_xml_path(str(args.base_model))
    contract=ET.Element('robot',name='o6_22_native_joint_contracts_linear_mimic_baseline')
    source_files=[];joint_rows=[];passive=[];independent=[];thumb_changed=[]
    equality=xml.find('equality')
    if equality is None:equality=ET.SubElement(xml,'equality')
    for directory,filename in [('LEFT','linkerhand_o6_left_test.urdf'),('RIGHT','linkerhand_o6_right.urdf')]:
        path=SOURCE/directory/filename;source=ET.parse(path).getroot()
        source_files.append({'path':str(path),'sha256':sha(path)})
        for definition in source.findall('joint'):
            if definition.get('type')=='fixed':continue
            name=definition.get('name');joint=xml.find(f".//joint[@name='{name}']")
            if joint is None:raise ValueError('Source joint missing from robot '+name)
            j=mj.mj_name2id(old,mj.mjtObj.mjOBJ_JOINT,name);bid=int(old.jnt_bodyid[j]);bn=old.body(bid).name
            body=xml.find(f".//body[@name='{bn}']");origin=definition.find('origin')
            pos=np.asarray(list(map(float,origin.get('xyz','0 0 0').split())))
            rpy=np.asarray(list(map(float,origin.get('rpy','0 0 0').split())))
            quat=Rotation.from_euler('xyz',rpy).as_quat()[[3,0,1,2]]
            if not np.allclose(old.body_pos[bid],pos,atol=1e-10):raise ValueError('Unexpected new O6 link offset '+name)
            axis=np.asarray(list(map(float,definition.find('axis').get('xyz').split())))
            if not np.allclose(old.jnt_axis[j],axis,atol=1e-10):raise ValueError('Unexpected O6 native axis '+name)
            angle_match=abs(float(old.body_quat[bid]@quat))>1-1e-12
            if 'thumb_cmc_yaw'in name:
                body.set('quat',fmt(quat));thumb_changed.append({'joint':name,'body':bn,'origin_rpy_rad':rpy.tolist(),'new_body_quaternion_wxyz':quat.tolist()})
            elif not angle_match:raise ValueError('Unexpected nonthumb O6 joint-frame change '+name)
            limits=definition.find('limit');joint.set('range',limits.get('lower')+' '+limits.get('upper'));joint.set('limited','true')
            contract.append(copy.deepcopy(definition))
            mimic=definition.find('mimic');row={'joint':name,'body':bn,'limits':dict(limits.attrib),'origin_rpy_rad':rpy.tolist()}
            if mimic is not None:
                parent=mimic.get('joint');factor=float(mimic.get('multiplier','1'));offset=float(mimic.get('offset','0'))
                ET.SubElement(equality,'joint',name='o6_mimic_'+name,joint1=name,joint2=parent,
                              polycoef=fmt([offset,factor,0,0,0]),solref='.01 1',solimp='.95 .99 .001 .5 2')
                actuator=xml.find(f"./actuator/*[@name='motor_{name}']")
                if actuator is None:raise ValueError('Native compatibility actuator absent '+name)
                actuator.tag='general'
                for shortcut in ['kp','kv','dampratio','inheritrange','timeconst']:
                    actuator.attrib.pop(shortcut,None)
                actuator.set('gainprm',fmt(np.zeros(10)));actuator.set('biasprm',fmt(np.zeros(10)));actuator.set('gear',fmt(np.zeros(6)))
                passive.append(name);row['linear_mimic']={'parent':parent,'multiplier':factor,'offset':offset}
            else:independent.append(name)
            joint_rows.append(row)
    args.contract.write_text(ET.tostring(contract,encoding='unicode'),encoding='utf-8')
    custom=xml.find('custom')
    if custom is None:custom=ET.SubElement(xml,'custom')
    mask=np.zeros(old.nu)
    for name in passive:mask[mj.mj_name2id(old,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+name)]=1
    ET.SubElement(custom,'numeric',name='o6_passive_finger_motor_mask',data=fmt(mask))
    ET.SubElement(custom,'text',name='o6_joint_contract_path',data=str(args.contract))
    ET.SubElement(custom,'text',name='o6_joint_contract_sha256',data=sha(args.contract))
    args.output.write_text(ET.tostring(xml,encoding='unicode'),encoding='utf-8')
    new=mj.MjModel.from_xml_path(str(args.output))
    if(new.nq,new.nv,new.nu,new.nbody)!=(old.nq,old.nv,old.nu,old.nbody):raise ValueError('Native state/actuator/body topology changed')
    unchanged=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','dof_armature','dof_damping','dof_frictionloss','geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity']
    for key in unchanged:
        if not np.array_equal(getattr(old,key),getattr(new,key)):raise ValueError('Unexpected physics/shape edit '+key)
    passive_ids=[mj.mj_name2id(new,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in passive]
    active_ids=[i for i in range(new.nu)if i not in passive_ids]
    for key in ['actuator_gainprm','actuator_biasprm','actuator_gear']:
        if not np.array_equal(getattr(old,key)[active_ids],getattr(new,key)[active_ids]):raise ValueError('Changed active driver '+key)
        if np.any(getattr(new,key)[passive_ids]!=0):raise ValueError('Passive actuator still contributes '+key)
    nonhand=[j for j in range(old.njnt)if not(old.joint(j).name or'').startswith(('lh_','rh_'))]
    if not np.array_equal(old.jnt_range[nonhand],new.jnt_range[nonhand]):raise ValueError('Changed body joint limits')
    changed_body_indices=np.flatnonzero(np.any(old.body_quat!=new.body_quat,axis=1)).tolist()
    if set(changed_body_indices)!={int(new.jnt_bodyid[mj.mj_name2id(new,mj.mjtObj.mjOBJ_JOINT,n)])for n in ['lh_thumb_cmc_yaw','rh_thumb_cmc_yaw']}:
        raise ValueError('Unexpected body frame calibration change')
    if new.neq-old.neq!=10 or len(independent)!=12:raise ValueError('Expected 10 passive equalities and 12 independent hand commands')
    report={'schema':'fourth.o6_calibrated_linear_mimic_plant.v1','model':str(args.output),'model_sha256':sha(args.output),
            'base_model':str(args.base_model),'base_model_sha256':sha(args.base_model),
            'joint_contract':str(args.contract),'joint_contract_sha256':sha(args.contract),'source_urdfs':source_files,
            'native_qpos_count':new.nq,'native_qvel_count':new.nv,'compatibility_actuator_count':new.nu,
            'effective_active_actuator_count':len(active_ids),'passive_motor_ids':passive_ids,'passive_motor_joint_names':passive,
            'independent_finger_joint_names':independent,'thumb_zero_calibration':thumb_changed,'joint_contracts':joint_rows,
            'preserved_exactly':unchanged,'all_28_learned_body_parameters_unchanged':True,
            'only_two_thumb_body_zero_rotations_changed':True,'new_joint_equalities':10,
            'mimic_model':'URDF endpoint_linear_fallback; JSON quadratic_runtime curves not applied.',
            'mimic_equality_solver_parameters':{'solref':[.01,1],'solimp':[.95,.99,.001,.5,2]},
            'measured_calibration_scope':'Right partial_passed: thumb3 and pinky2 measured; right index/middle/ring transferred; left mirrored and not measured.',
            'geometry_source':'All 24 package visual STL files are byte-identical to existing complete robot visual meshes (independently checked by root). Existing 24 validated simplified per-link collision meshes retained.',
            'source_effort_100_Nm_is_not_a_measured_motor_torque':True,'feedback_motor_cap_Nm':.4,
            'zero_offset_policy':'URDF origin only, applied once. No JSON zero offsets added.',
            'reset_requirement':'Project all 22 finger qpos/qvel onto the 12-command linear mimic manifold before initial reset; no state projection during integration.',
            'physical_rollout_performed':False}
    args.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k]for k in ['model','model_sha256','joint_contract','joint_contract_sha256','effective_active_actuator_count','passive_motor_joint_names','new_joint_equalities','thumb_zero_calibration']}),flush=True)

if __name__=='__main__':main()
