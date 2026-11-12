"""Check that the morphology-only dish variant preserved contact material."""
from pathlib import Path
import json,numpy as np,mujoco as mj
r=Path(__file__).resolve().parents[1]
b=mj.MjModel.from_binary_path(str(r/'outputs/fourth_full_sequence/assets/vita_six_objects_o6_microstep_v1.mjb'))
m=mj.MjModel.from_binary_path(str(r/'outputs/fourth_full_sequence/assets/vita_six_objects_o6_plate1_deep_lip_microstep_v1.mjb'))
keys=['geom_friction','geom_condim','geom_priority','geom_solimp','geom_solref','geom_margin','geom_gap','geom_contype','geom_conaffinity']
old=[g for g in range(b.ngeom)if b.body(int(b.geom_bodyid[g])).name=='target_object_01'and(b.geom_contype[g]or b.geom_conaffinity[g])]
new=[g for g in range(m.ngeom)if m.body(int(m.geom_bodyid[g])).name=='target_object_01'and(m.geom_contype[g]or m.geom_conaffinity[g])]
rows=[]
for g in new:
    matches=[o for o in old if all(np.array_equal(getattr(m,k)[g],getattr(b,k)[o])for k in keys)]
    diffs=[[k for k in keys if not np.array_equal(getattr(m,k)[g],getattr(b,k)[o])]for o in old]
    rows.append({'name':m.geom(g).name,'material_and_mask_matches_original':bool(matches),'closest_reference_difference_keys':min(diffs,key=len),'parameters':{k:np.asarray(getattr(m,k)[g]).tolist()for k in keys}})
oldparams=[{k:np.asarray(getattr(b,k)[g]).tolist()for k in keys}for g in old]
oldparams=list({json.dumps(p,sort_keys=True):p for p in oldparams}.values())
report={'scope':'Read-only material, contact solver and collision mask comparison; mesh morphology deliberately differs.','original_collision_geoms':len(old),'new_collision_geoms':len(new),'original_contact_parameter_sets':oldparams,'all_new_contacts_use_original_material_solver_mask':all(x['material_and_mask_matches_original']for x in rows),'rows':rows}
bitdiff=[]
proto=old[0]
for g in new:
    for other in range(b.ngeom):
        if b.body(int(b.geom_bodyid[other])).name=='target_object_01':continue
        expected=bool((int(b.geom_contype[proto])&int(b.geom_conaffinity[other]))or(int(b.geom_contype[other])&int(b.geom_conaffinity[proto])))
        actual=bool((int(m.geom_contype[g])&int(b.geom_conaffinity[other]))or(int(b.geom_contype[other])&int(m.geom_conaffinity[g])))
        if expected!=actual:bitdiff.append({'new_geom':m.geom(g).name,'other_geom':b.geom(other).name,'other_body':b.body(int(b.geom_bodyid[other])).name,'expected':expected,'actual':actual})
report['pairwise_bitmask_enable_differences']=bitdiff
report['all_contact_enable_pairs_semantically_equal']=not bitdiff
(r/'reports/single_plate_lip_contact_parameters_v1.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({**{k:v for k,v in report.items()if k not in ['rows','pairwise_bitmask_enable_differences']},'pairwise_diff_count':len(bitdiff),'pairwise_first5':bitdiff[:5],'difference_key_sets':list({tuple(x['closest_reference_difference_keys'])for x in rows})}),flush=True)
