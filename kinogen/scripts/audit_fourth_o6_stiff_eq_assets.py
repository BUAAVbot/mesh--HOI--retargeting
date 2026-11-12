"""Compare effective models by arrays, independently of the trial controller."""
from pathlib import Path
import hashlib,json
import numpy as np
import mujoco as mj
R=Path(__file__).resolve().parents[1]
manifest=R/'outputs/fourth_full_sequence/assets/vita_six_objects_o6_stiff_eq_v1.manifest.json'
src=json.loads(manifest.read_text());base=Path(src['base_model']);b=mj.MjModel.from_binary_path(str(base))
rows=[]
for name in ['physics_o6_stiff_eq_raw_push_v1','physics_o6_stiff_eq_reach_v1',
             'physics_o6_stiff_eq_C7_v1','physics_o6_stiff_eq_C8_locked_v1']:
    path=R/'outputs/fourth_full_sequence'/name/'effective_model.mjb'
    if not path.exists():continue
    m=mj.MjModel.from_binary_path(str(path));checks={k:bool(np.array_equal(getattr(b,k),getattr(m,k)))for k in src['preserved_exactly']}
    equalities=np.flatnonzero(m.eq_type==mj.mjtEq.mjEQ_JOINT)
    rows.append({'trial':name,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'unchanged_arrays':checks,'all_preserved':all(checks.values()),
                 'only_declared_eq_solimp':bool(np.array_equal(m.eq_solimp[equalities],np.tile([.999,.9999,.001,.5,2.],(10,1)))),
                 'physical_dt_unchanged':bool(m.opt.timestep==b.opt.timestep)})
out={'scope':'Effective model audit only; not a manipulation success claim.',
     'source_manifest':str(manifest),'base_model':str(base),'rows':rows,
     'all_hands_geometry_inertia_torque_limits_preserved':all(r['all_preserved']for r in rows)}
p=R/'reports/fourth_o6_stiff_eq_effective_asset_audit.json';p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
