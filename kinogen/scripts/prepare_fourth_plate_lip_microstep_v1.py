"""Explicit dish-only lip geometry with audited original hands and micro solver."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco as mj
root = Path(__file__).resolve().parents[1]
source = root/'outputs/fourth_full_sequence/assets/vita_six_objects_o6_plate1_deep_lip_in10_up4_v1.xml'
baseline_path = root/'outputs/fourth_full_sequence/assets/vita_six_objects_o6_microstep_v1.mjb'
output = root/'outputs/fourth_full_sequence/assets/vita_six_objects_o6_plate1_deep_lip_microstep_v1.mjb'
if output.exists():
    raise FileExistsError(output)
m = mj.MjModel.from_xml_path(str(source))
b = mj.MjModel.from_binary_path(str(baseline_path))
eq = np.flatnonzero(m.eq_type == mj.mjtEq.mjEQ_JOINT)
assert len(eq)==10
m.opt.timestep=.001
m.eq_solimp[eq] = [.999,.9999,.001,.5,2]
m.eq_solref[eq] = [.002,1]
fields=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
        'jnt_type','jnt_range','jnt_axis','dof_damping','dof_armature',
        'actuator_gainprm','actuator_biasprm','actuator_gear','actuator_forcerange',
        'actuator_forcelimited','eq_data','eq_solimp','eq_solref']
# The original frozen rollout has explicit runtime motor/joint configuration.
# Recompiling the morphology XML does not preserve those runtime edits.
# Restore the exact executed baseline parameters before making the comparison.
runtime_fields=['jnt_range','jnt_margin','dof_damping','dof_armature','dof_frictionloss',
                'actuator_gainprm','actuator_biasprm','actuator_gear',
                'actuator_forcerange','actuator_forcelimited',
                'actuator_ctrlrange','actuator_ctrllimited']
restored={}
for key in runtime_fields:
    before=getattr(m,key).copy()
    reference=getattr(b,key)
    assert before.shape==reference.shape
    restored[key]={'different_entries_before_restoration':int(np.count_nonzero(before!=reference)),
                   'maximum_absolute_difference_before_restoration':float(np.max(abs(before.astype(float)-reference.astype(float))))}
    getattr(m,key)[:]=reference
preserved={k:bool(np.array_equal(getattr(m,k),getattr(b,k))) for k in fields}
geom_fields=['geom_type','geom_pos','geom_quat','geom_size','geom_friction',
             'geom_contype','geom_conaffinity','geom_condim','geom_solimp','geom_solref']
rows=[]
for g in range(b.ngeom):
    body=b.body(int(b.geom_bodyid[g])).name or ''
    if body=='target_object_01':
        continue
    name=b.geom(g).name
    if not name:
        oldbody=int(b.geom_bodyid[g])
        newbody=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,body)
        oldids=np.flatnonzero(b.geom_bodyid==oldbody)
        newids=np.flatnonzero(m.geom_bodyid==newbody)
        assert len(oldids)==len(newids)
        ordinal=int(np.flatnonzero(oldids==g)[0])
        new=int(newids[ordinal])
        name=f'{body}:anonymous_geom_ordinal_{ordinal}'
    else:
        new=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,name)
    checks={k:bool(new>=0 and np.array_equal(getattr(b,k)[g],getattr(m,k)[new])) for k in geom_fields}
    mid=int(b.geom_dataid[g]);nid=int(m.geom_dataid[new])
    if b.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
        for key,adr,count in [('mesh_vert','mesh_vertadr','mesh_vertnum'),('mesh_face','mesh_faceadr','mesh_facenum')]:
            oldarray=getattr(b,key)[int(getattr(b,adr)[mid]):int(getattr(b,adr)[mid]+getattr(b,count)[mid])]
            newarray=getattr(m,key)[int(getattr(m,adr)[nid]):int(getattr(m,adr)[nid]+getattr(m,count)[nid])]
            checks[key]=bool(np.array_equal(oldarray,newarray))
    rows.append({'name':name,'body':body,'unchanged':all(checks.values()),'checks':checks})
assert all(preserved.values()),preserved
assert all(r['unchanged'] for r in rows),[r for r in rows if not r['unchanged']]
assert m.nq==b.nq and m.nv==b.nv and m.nu==b.nu
mj.mj_saveModel(m,str(output),None)
report={'scope':'Independent plate1-only deep hollow lip morphology and declared numerical microstep variant; original hands, original actuators, original six object masses and inertias unchanged.',
        'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'baseline':str(baseline_path),'output':str(output),'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'integration_dt_s':float(m.opt.timestep),'joint_eq_solref':m.eq_solref[eq].tolist(),
        'runtime_core_parameters_restored_from_executed_baseline':restored,
        'exact_body_joint_actuator_equalities':preserved,'all_original_non_plate1_geometry_preserved':True,
        'geometry_audit':rows,'pass':True}
output.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['output','output_sha256','all_original_non_plate1_geometry_preserved','pass']}),flush=True)
