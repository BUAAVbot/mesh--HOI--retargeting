"""Freeze an explicit O6 transmission-impedance variant; no geometry changes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import mujoco as mj

R = Path(__file__).resolve().parents[1]
source_plan = R/'outputs/fourth_full_sequence/physics_o6_contact_preslide_v4_extended40/source_plan.json'
plan = json.loads(source_plan.read_text())
base = Path(plan['model'])
if not base.is_absolute(): base = R/base
out = R/'outputs/fourth_full_sequence/assets/vita_six_objects_o6_stiff_eq_v1.mjb'
if out.exists(): raise FileExistsError(out)
model = mj.MjModel.from_binary_path(str(base))
preserved = ['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
             'jnt_range','jnt_pos','jnt_axis','dof_armature','dof_damping',
             'geom_pos','geom_quat','geom_size','geom_type','geom_contype','geom_conaffinity',
             'geom_friction','geom_solref','geom_solimp','mesh_vert','mesh_face',
             'actuator_gainprm','actuator_biasprm','actuator_gear','actuator_forcerange',
             'actuator_forcelimited','eq_data','eq_solref']
before = {key:getattr(model,key).copy() for key in preserved}
old = model.eq_solimp.copy()
changed = [i for i in range(model.neq) if model.eq_type[i] == mj.mjtEq.mjEQ_JOINT]
assert len(changed) == 10
model.eq_solimp[changed] = [.999,.9999,.001,.5,2.]
mj.mj_saveModel(model,str(out),None)
after = mj.MjModel.from_binary_path(str(out))
assert all(np.array_equal(before[key],getattr(after,key)) for key in preserved)
assert float(after.opt.timestep) == .005
report = {'scope':'Explicit numerical transmission impedance variant from the original effective model. '
                  'No physical hand collider, inertia, motor capability, object geometry or state is changed.',
          'base_model':str(base),'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),
          'variant':str(out),'variant_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
          'changed_joint_equality_ids':changed,'old_eq_solimp':old[changed].tolist(),
          'new_eq_solimp':after.eq_solimp[changed].tolist(),'preserved_exactly':preserved,
          'physical_dt_s':float(after.opt.timestep),'objects_moved_or_welded':False,
          'interpretation':'Numerical experiment, not measured hardware stiffness; requires a fresh raw-scene rollout.'}
out.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n')
plan['model'] = str(out)
plan['numerical_variant_provenance'] = report
newplan = R/'reports/fourth_o6_stiff_eq_push_plan_v1.json'
newplan.write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'model':str(out),'plan':str(newplan),'preserved_all_checked_arrays':True}))
