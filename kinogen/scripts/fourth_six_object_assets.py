"""Build all six source tableware objects as free rigid bodies with hollow proxies.

The original Vita52 inertias/actuators are copied from the frozen successful
single-object model. Object masses, shell inertias, friction and the finite
support footprint are explicit estimates; the source meshes remain visual only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco as mj

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/fourth_sim_handoff'
BASE = ROOT/'outputs/fourth_vita_demo/assets/vita_object_05_dynamic_ankle100.xml'
OUTPUT = ROOT/'outputs/fourth_full_sequence/assets'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def num(x):
    return ' '.join(f'{float(v):.12g}' for v in x)


def json_write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=OUTPUT)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    xml = ET.parse(BASE).getroot()
    world, asset = xml.find('worldbody'), xml.find('asset')
    for body in list(world.findall('body')):
        if body.get('name', '').startswith('target_object_'):
            world.remove(body)
    for mesh in list(asset.findall('mesh')):
        if mesh.get('name', '').startswith('target_object_'):
            asset.remove(mesh)
    poses = np.load(DATA/'objects/poses_sim.npz', allow_pickle=False)
    valid = poses['valid']
    rawpos = poses['position_m']
    infos = [json.loads((DATA/f'objects/object_{i:02d}/model_info.json').read_text()) for i in range(1,7)]
    qa = json.loads((DATA/'provenance/object_pose_qa.json').read_text())
    plane = qa['table_plane']
    n0, o0 = np.asarray(plane['normal_world']), np.asarray(plane['origin_world'])
    normal = np.asarray([n0[0], -n0[2], n0[1]])
    origin = np.asarray([o0[0], -o0[2], o0[1]])
    normal /= np.linalg.norm(normal)
    xy = rawpos[:,:,:2][valid]
    rmax = max(float(x['dimensions_xyz_m'][0])/2 for x in infos)
    low, high = xy.min(0)-rmax-.02, xy.max(0)+rmax+.02
    centre_xy = (low+high)/2
    half_xy = (high-low)/2
    table_top = float((normal@origin-normal[:2]@centre_xy)/normal[2])
    quat = np.r_[1+normal[2], np.cross([0.,0.,1.], normal)]
    quat /= np.linalg.norm(quat)
    thickness = .025
    table = world.find("geom[@name='table_support']")
    table.set('pos', num(np.r_[centre_xy,table_top]-thickness*normal))
    table.set('quat', num(quat))
    table.set('size', num([*half_xy,thickness]))
    floor = world.find("geom[@name='assumed_floor']")
    floor_z = float(floor.get('pos').split()[2])
    colors = ['.84 .46 .60 1','.90 .86 .75 1','.38 .59 .78 1',
              '.43 .68 .46 1','.58 .73 .81 1','.92 .87 .76 1']
    masses = [.24,.24,.22,.20,.25,.18]
    object_reports = []
    for i, info, mass, color in zip(range(1,7),infos,masses,colors):
        name = f'target_object_{i:02d}'
        path = DATA/f'objects/object_{i:02d}/mesh.obj'
        z = np.load(DATA/f'objects/object_{i:02d}/mesh.npz',allow_pickle=False)
        verts = np.asarray(z['vertices'])
        dims = np.asarray(info['dimensions_xyz_m'])
        radius, height = float(max(dims[:2])/2), float(dims[2])
        radial = np.linalg.norm(verts[:,:2],axis=1)
        bottom_ring = verts[:,2] <= verts[:,2].min()+height*.05
        base_radius = min(radius, max(radius*.6,float(radial[bottom_ring].max())))
        wall = min(.0045, radius*.07)
        bottom_h = min(.002, height*.12)
        ix = mass*(3*radius*radius+height*height)/12
        iz = mass*radius*radius/2
        ET.SubElement(asset,'mesh',name=name+'_visual',file=str(path))
        body = ET.SubElement(world,'body',name=name,pos=num(rawpos[0,i-1]),
                             quat=num(poses['quaternion_wxyz'][0,i-1]))
        ET.SubElement(body,'freejoint',name=name+'_free')
        ET.SubElement(body,'inertial',pos=num([0,0,height*.35]),mass=str(mass),diaginertia=num([ix,ix,iz]))
        ET.SubElement(body,'geom',name=name+'_visual_geom',type='mesh',mesh=name+'_visual',
                      rgba=color,contype='0',conaffinity='0',density='0',group='1')
        ET.SubElement(body,'geom',name=name+'_bottom_collision',type='cylinder',
                      pos=num([0,0,bottom_h]),size=num([base_radius-wall*.3,bottom_h]),
                      density='0',contype='1',conaffinity='1',friction='.65 .01 .0005',group='0')
        # Each convex sector is a small sloping shell wall. Their union leaves
        # the central cavity empty and avoids a convex hull across the opening.
        sectors = 20
        for k in range(sectors):
            angle = 2*math.pi*k/sectors
            points=[]
            for zz, outer in [(bottom_h,base_radius),(height,radius)]:
                for rr in [outer-wall,outer]:
                    for theta in [angle-math.pi/sectors,angle+math.pi/sectors]:
                        points.append([rr*math.cos(theta),rr*math.sin(theta),zz])
            mn = f'{name}_wall_mesh_{k:02d}'
            ET.SubElement(asset,'mesh',name=mn,vertex=num(np.asarray(points).ravel()))
            ET.SubElement(body,'geom',name=f'{name}_wall_{k:02d}',type='mesh',mesh=mn,
                          density='0',contype='1',conaffinity='1',friction='.65 .01 .0005',group='0')
        object_reports.append({'id':i,'name':info['name'],'body':name,'joint':name+'_free',
            'initial_pose_raw_source':np.r_[rawpos[0,i-1],poses['quaternion_wxyz'][0,i-1]].tolist(),
            'source_model_info':info,'source_mesh_sha256':sha(path),
            'collision':{'kind':'hollow_tapered_shell','bottom_disc':1,'wall_convex_sectors':sectors,
                         'base_radius_m':base_radius,'top_radius_m':radius,'height_m':height,
                         'wall_thickness_m':wall,'bottom_half_thickness_m':bottom_h,'opening_not_capped':True},
            'assumed_physics':{'mass_kg':mass,'inertial_com_local_m':[0,0,height*.35],
                              'diagonal_inertia_kg_m2':[ix,ix,iz],'friction':[.65,.01,.0005],
                              'measured':False},'source_valid_frames':int(valid[:,i-1].sum())})
    path = a.output_dir/'vita_six_objects_dynamic.xml'
    path.write_text(ET.tostring(xml,encoding='unicode'),encoding='utf-8')
    model = mj.MjModel.from_xml_path(str(path))
    if (model.nq,model.nv,model.nu)!=(101,94,52):
        raise ValueError((model.nq,model.nv,model.nu))
    def mapping(name):
        j = mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,name)
        return {'qpos':int(model.jnt_qposadr[j]),'qvel':int(model.jnt_dofadr[j])}
    for item in object_reports:
        item['address']=mapping(item['joint'])
        item['body_id']=int(mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,item['body']))
    data = mj.MjData(model)
    data.qpos[:3]=[20,20,2]
    mj.mj_forward(model,data)
    initial_cross=[]
    for c in data.contact:
        b1,b2=int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])
        n1,n2=(mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,b) or 'world' for b in [b1,b2])
        if n1.startswith('target_object_') and n2.startswith('target_object_'):
            initial_cross.append({'bodies':[n1,n2],'distance_m':float(c.dist)})
    # Each object is independently dropped upright onto the support plane.
    # Other bodies are moved far away only in this isolated validation reset.
    support_probes=[]
    for item in object_reports:
        d = mj.MjData(model)
        d.qpos[:3]=[20,20,2]
        for other in object_reports:
            q=other['address']['qpos']
            d.qpos[q:q+3]=[30+other['id'],30,2]
            d.qpos[q+3:q+7]=[1,0,0,0]
        q=item['address']['qpos'];v=item['address']['qvel']
        d.qpos[q:q+3]=[*centre_xy,table_top+.005]
        d.qpos[q+3:q+7]=quat
        mj.mj_forward(model,d)
        maxpen=0.
        for _ in range(200):
            mj.mj_step(model,d)
            for c in d.contact:
                if item['body_id'] in [int(model.geom_bodyid[c.geom1]),int(model.geom_bodyid[c.geom2])]:
                    maxpen=max(maxpen,-float(c.dist))
        top_here=(normal@origin-normal[:2]@d.qpos[q:q+2])/normal[2]
        stable = bool(np.isfinite(d.qpos).all() and abs(d.qpos[q+2]-top_here)<.025 and np.linalg.norm(d.qvel[v:v+3])<.05)
        support_probes.append({'object_id':item['id'],'steps':200,'duration_s':1.,
                              'final_bottom_world_m':d.qpos[q:q+3].tolist(),'final_speed_m_s':float(np.linalg.norm(d.qvel[v:v+3])),
                              'max_penetration_m':maxpen,'passed':stable})
    report={'schema':'fourth_six_object_assets.v1','model':str(path),'model_sha256':sha(path),
            'base_frozen_model':str(BASE),'base_sha256':sha(BASE),
            'nq':model.nq,'nv':model.nv,'nu':model.nu,'robot_freejoint':'vita_demo_free_root',
            'robot_address':mapping('vita_demo_free_root'),'actuator_names':[mj.mj_id2name(model,mj.mjtObj.mjOBJ_ACTUATOR,i) for i in range(model.nu)],
            'robot_inertias_and_actuators_preserved_from_base':True,
            'table_support':{'name':'table_support','normal':normal.tolist(),'plane_origin':origin.tolist(),
                             'centre_xy':centre_xy.tolist(),'top_z_at_centre_m':table_top,'half_size_xy_m':half_xy.tolist(),
                             'world_xy_bounds_approx_m':[low.tolist(),high.tolist()],
                             'footprint_source':'all valid object source centres plus maximum tableware radius and 2cm margin',
                             'assumption':True},
            'floor_support':{'name':'assumed_floor','z_m':floor_z,'source':'unchanged from frozen single-object model'},
            'objects':object_reports,'raw_source_initial_object_object_contacts':initial_cross,
            'isolated_static_support_probes':support_probes,
            'all_support_probes_passed':all(x['passed'] for x in support_probes),
            'limitations':['Object masses, shell inertia and friction are engineering assumptions, not measured.',
                           'Canonical visual meshes are source estimates; shell proxies approximate local radial profiles.',
                           'Raw source initial cross-object contacts are reported, not removed by collision filtering.',
                           'Static probes prove table support only, not full humanoid interaction success.']}
    json_write(a.output_dir/'assets_report.json',report)
    print(json.dumps({'model':str(path),'sha256':sha(path),'nq':model.nq,'nv':model.nv,'nu':model.nu,
                      'support_passed':report['all_support_probes_passed'],'raw_initial_cross_contacts':len(initial_cross),
                      'table':report['table_support'],'floor':report['floor_support']},ensure_ascii=False),flush=True)
    if not report['all_support_probes_passed']:
        raise SystemExit(3)


if __name__=='__main__':
    main()
