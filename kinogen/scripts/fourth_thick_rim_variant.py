"""Add an explicitly synthetic, graspable hollow rim to six source tableware meshes.

Robot meshes, collisions, joints and motor limits stay byte-for-byte unchanged
in XML. The original source assets remain untouched. This is a geometry variant
with retained engineering mass/inertia assumptions, not a reconstruction result.
"""
from pathlib import Path
import argparse, hashlib, json, math
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(a): return ' '.join('%.12g'%float(x) for x in np.asarray(a).ravel())

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--base',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--radial-out-m',type=float,default=.004)
p.add_argument('--radial-in-m',type=float,default=.006)
p.add_argument('--top-rise-m',type=float,default=.008)
p.add_argument('--bottom-drop-m',type=float,default=.004)
a=p.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
xml=ET.parse(a.base).getroot(); original=mj.MjModel.from_xml_path(str(a.base.resolve()))
asset=xml.find('asset'); rows=[]; robot=[ET.tostring(b) for b in xml.find('worldbody').findall('body') if not b.get('name','').startswith('target_object_')]
for i in range(1,7):
    name=f'target_object_{i:02d}'; body=xml.find('worldbody').find(f"body[@name='{name}']")
    wall=body.find(f"geom[@name='{name}_wall_00']"); mesh=asset.find(f"mesh[@name='{wall.get('mesh')}']")
    vertices=np.asarray(list(map(float,mesh.get('vertex').split()))).reshape(-1,3)
    radius=float(np.linalg.norm(vertices[:,:2],axis=1).max()); top=float(vertices[:,2].max())
    outer=radius+a.radial_out_m; inner=radius-a.radial_in_m; low=top-a.bottom_drop_m; high=top+a.top_rise_m
    color=body.find(f"geom[@name='{name}_visual_geom']").get('rgba')
    for k in range(20):
        theta=2*math.pi*k/20; vv=[]
        for z in (low,high):
            for r in (inner,outer):
                for t in (theta-math.pi/20,theta+math.pi/20): vv.append([r*math.cos(t),r*math.sin(t),z])
        mn=f'{name}_synthetic_thick_rim_mesh_{k:02d}'
        ET.SubElement(asset,'mesh',name=mn,vertex=num(vv))
        ET.SubElement(body,'geom',name=f'{name}_synthetic_thick_rim_{k:02d}',type='mesh',mesh=mn,
                      rgba=color,group='1',density='0',contype='1',conaffinity='1',friction=wall.get('friction'))
    bid=original.body(name).id
    rows.append({'object_id':i,'original_top_radius_m':radius,'original_height_m':top,
                 'synthetic_rim_outer_radius_m':outer,'synthetic_rim_inner_radius_m':inner,
                 'synthetic_rim_bottom_z_m':low,'synthetic_rim_top_z_m':high,
                 'synthetic_rim_vertical_thickness_m':high-low,'synthetic_rim_radial_width_m':outer-inner,
                 'top_opening_minimum_inscribed_radius_m':inner*math.cos(math.pi/20),
                 'retained_mass_kg':float(original.body_mass[bid]),
                 'retained_inertia_kg_m2':original.body_inertia[bid].tolist(),
                 'mass_and_inertia_are_engineering_assumptions':True,
                 'visual_and_collision_use_identical_added_mesh':True,'top_opening_not_capped':True})
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(ET.tostring(xml,encoding='unicode'),encoding='utf-8')
new=mj.MjModel.from_xml_path(str(a.output.resolve()))
after=[ET.tostring(b) for b in xml.find('worldbody').findall('body') if not b.get('name','').startswith('target_object_')]
assert robot==after
preserve=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_range','dof_armature','dof_damping','eq_data','eq_solref','eq_solimp','actuator_gainprm','actuator_biasprm','actuator_gear','actuator_forcerange']
for k in preserve: assert np.array_equal(getattr(original,k),getattr(new,k)),k
assert (new.nq,new.nv,new.nu)==(101,94,52)
for g in range(original.ngeom):
    name=original.geom(g).name
    if not name: continue
    ng=new.geom(name).id
    for k in ['geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity']:
        assert np.array_equal(getattr(original,k)[g],getattr(new,k)[ng]),(name,k)
report={'schema':'fourth.synthetic_thick_tableware_rims.v1','model':str(a.output),'model_sha256':sha(a.output),
        'source_model':str(a.base),'source_model_sha256':sha(a.base),'objects':rows,
        'robot_hand_visual_and_collision_unchanged':True,'all_existing_geoms_unchanged':True,
        'original_body_initial_positions_unchanged':True,'original_bottom_support_unchanged':True,
        'preserved_exactly':preserve,'added_geoms':new.ngeom-original.ngeom,
        'interpretation':'Synthetic grasp-assistance geometry, not measured original tableware. Existing estimated masses and inertias retained to isolate geometry change. No physical rollout or successful grasp claimed.'}
a.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report),flush=True)
