"""Snapshot recovered training meshes into KinoGen and emit isolated model paths."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

R=Path(__file__).resolve().parents[1]
F=R/'dependencies/fourth_sonic/newft_spawn_asset'
A=R/'outputs/fourth_full_sequence/assets'
SOURCE=Path('/data/yunhao.xia/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/vita02a/urdf/vita02a_shoes.urdf')
CFG=Path('/data/yunhao.xia/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots/vita02a.py')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    F.mkdir(parents=True,exist_ok=True);manifest={};mapping={}
    def freeze(path,relative):
        dest=F/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists()and sha(dest)!=sha(path):raise ValueError('Frozen snapshot already differs: '+str(dest))
        if not dest.exists():shutil.copy2(path,dest)
        manifest[str(relative)]={'source':str(path),'sha256':sha(dest),'bytes':dest.stat().st_size}
        return dest
    freeze(SOURCE,Path('urdf/vita02a_shoes.urdf'))
    freeze(CFG,Path('provenance/vita02a_training_cfg.py'))
    urdf=ET.parse(SOURCE).getroot()
    for mesh in urdf.findall('.//mesh'):
        source=(SOURCE.parent/mesh.get('filename')).resolve()
        relative=source.relative_to(SOURCE.parent.parent)
        mapping[str(source)]=freeze(source,relative)
    models=[]
    for name in ['vita_six_objects_newft_isaac52','vita_six_objects_newft_isaac52_selfcontacts']:
        source=A/(name+'.xml');xml=ET.parse(source).getroot()
        replaced=0
        for mesh in xml.findall('./asset/mesh'):
            path=str(Path(mesh.get('file','')).resolve())
            if path in mapping:mesh.set('file',str(mapping[path]));replaced+=1
        out=A/(name+'_isolated.xml')
        if out.exists():raise ValueError('Isolated model already exists; do not overwrite '+str(out))
        out.write_text(ET.tostring(xml,encoding='unicode'),encoding='utf-8')
        old=mj.MjModel.from_xml_path(str(source));new=mj.MjModel.from_xml_path(str(out))
        arrays=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
                'geom_type','geom_bodyid','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity',
                'jnt_type','jnt_axis','jnt_pos','jnt_range','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm',
                'mesh_vert','mesh_face']
        if any(not np.array_equal(getattr(old,k),getattr(new,k))for k in arrays):
            raise ValueError('Isolating model changed physical array')
        models.append({'original_model':str(source),'original_sha256':sha(source),
                       'isolated_model':str(out),'isolated_sha256':sha(out),
                       'mesh_paths_replaced':replaced,'physical_arrays_exactly_equal':arrays})
    report={'schema':'fourth.newft_training_asset_snapshot.v1','snapshot_root':str(F),
            'files':manifest,'file_count':len(manifest),'models':models,
            'source_project_modified':False,'original_runtime_models_overwritten':False}
    (A/'newft_isolated_asset_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps({'frozen_files':len(manifest),'models':models},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
