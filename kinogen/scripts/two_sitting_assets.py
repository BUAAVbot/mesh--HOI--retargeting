"""Build a separate dual-Vita scene for the 172-frame sit/swap/sit source.

Source meshes remain visual observations. Ground and two stationary seat
colliders are fitted, explicitly recorded engineering approximations.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def num(values):
    return ' '.join(format(float(v), '.17g') for v in values)


def save_xml(tree, path):
    ET.indent(tree)
    path.write_text(ET.tostring(tree, encoding='unicode')+'\n', encoding='utf-8')


def base_document():
    root = ET.Element('mujoco', model='two_sitting_people_vita')
    ET.SubElement(root, 'compiler', angle='radian')
    ET.SubElement(root, 'option', timestep='.005', gravity='0 0 -9.81',
                  integrator='implicitfast', iterations='100', cone='elliptic')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='800')
    ET.SubElement(visual, 'headlight', ambient='.5 .5 .5', diffuse='.65 .65 .65')
    ET.SubElement(root, 'statistic', center='0 1 1', extent='4')
    return root


def load_robot(path):
    source = ET.parse(path).getroot()
    body = source.find("worldbody/body[@name='base_link']")
    if body is None:
        raise ValueError('Expected unprefixed near-source Vita base_link')
    used = {g.get('mesh') for g in body.findall('.//geom') if g.get('mesh')}
    assets = ET.Element('asset')
    for item in source.find('asset'):
        if item.get('name') in used:
            item = copy.deepcopy(item)
            if item.get('file') and not Path(item.get('file')).is_absolute():
                compiler = source.find('compiler')
                meshdir = compiler.get('meshdir', '') if compiler is not None else ''
                item.set('file', str((path.parent/meshdir/item.get('file')).resolve()))
            assets.append(item)
    if len(assets) != len(used):
        raise ValueError('Robot mesh dependencies missing')
    motors = source.find('actuator')
    if motors is None or len(motors) != 52:
        raise ValueError('Expected complete 52-channel Vita source')
    equality = source.find('equality')
    # Only robot-internal equalities are retained; scene/object constraints are excluded.
    joints = {j.get('name') for j in body.iter() if j.tag in ('joint', 'freejoint')}
    eq = ET.Element('equality')
    if equality is not None:
        for item in equality:
            if item.tag == 'joint' and item.get('joint1') in joints and (item.get('joint2') is None or item.get('joint2') in joints):
                eq.append(copy.deepcopy(item))
    return copy.deepcopy(body), assets, copy.deepcopy(motors), eq


def prefixed(body, motors, equalities, prefix, contype, affinity):
    body = copy.deepcopy(body)
    names = {x.get('name') for x in body.iter() if x.get('name')}
    for x in body.iter():
        if x.get('name'):
            x.set('name', prefix+x.get('name'))
        if x.tag == 'geom' and int(x.get('contype', '0')):
            x.set('contype', str(contype)); x.set('conaffinity', str(affinity))
    result_motors = copy.deepcopy(motors)
    for x in result_motors:
        x.set('name', prefix+x.get('name'))
        x.set('joint', prefix+x.get('joint'))
    result_eq = copy.deepcopy(equalities)
    for x in result_eq:
        if x.get('name'): x.set('name', prefix+x.get('name'))
        for key in ('joint1', 'joint2', 'body1', 'body2'):
            if x.get(key) in names: x.set(key, prefix+x.get(key))
    return body, result_motors, result_eq


def support_fit(data, np):
    U = np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    source = np.load(data/'scene/pi3_mesh.npz', allow_pickle=False)
    vertices = np.asarray(source['vertices'], float)@U.T
    faces = np.asarray(source['faces'], int)
    tri = vertices[faces]; centers = tri.mean(axis=1)
    cross = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    area = np.linalg.norm(cross, axis=1)/2
    normals = cross/np.maximum(2*area[:,None], 1e-20)
    horizontal = (abs(normals[:,2])>.88)&(area>1e-10)
    floor = horizontal&(centers[:,2]>-.20)&(centers[:,2]<.18)
    if np.sum(floor)<20: raise ValueError('Insufficient near-horizontal ground observations')
    bins = np.floor(centers[floor,2]/.01).astype(int)
    hist = np.bincount(bins-bins.min(), weights=area[floor])
    mode = (int(np.argmax(hist))+int(bins.min())+.5)*.01
    selected = floor&(abs(centers[:,2]-mode)<.035)
    X = np.c_[centers[selected,:2], np.ones(np.sum(selected))]
    weights = np.sqrt(area[selected]); plane = np.linalg.lstsq(X*weights[:,None], centers[selected,2]*weights, rcond=None)[0]
    for _ in range(4):
        residual = abs(X@plane-centers[selected,2]); w = weights*np.minimum(1., .015/np.maximum(residual,1e-10))
        plane = np.linalg.lstsq(X*w[:,None],centers[selected,2]*w,rcond=None)[0]
    if np.linalg.norm(plane[:2])>.15: raise ValueError('Unexpected ground slope: '+str(plane))
    seats=[]
    for person in (1,2):
        h=np.load(data/f'human/person_{person:02d}/smplx_meshes_world.npz',allow_pickle=False)
        j=np.asarray(h['joints_world'],float)@U.T
        center=np.median(j[:min(8,len(j)),0,:2],axis=0)
        dist=np.linalg.norm(centers[:,:2]-center,axis=1)
        candidate=horizontal&(dist<.55)&(centers[:,2]>.22)&(centers[:,2]<.65)
        if np.sum(candidate)<8: raise ValueError('No observed seat surface near source person '+str(person))
        b=np.floor(centers[candidate,2]/.01).astype(int)
        histogram=np.bincount(b-b.min(),weights=area[candidate])
        peak=(int(np.argmax(histogram))+int(b.min())+.5)*.01
        pick=candidate&(abs(centers[:,2]-peak)<.025)
        p=centers[pick]; w=area[pick]; top=float(np.average(p[:,2],weights=w))
        cov=np.cov(p[:,:2].T,aweights=w); _,eig=np.linalg.eigh(cov); axis=eig[:,-1]
        if axis[0]<0: axis=-axis
        axes=np.column_stack([axis,[-axis[1],axis[0]]])
        xy=p[:,:2]@axes
        low=np.quantile(xy,.02,axis=0)-.01; high=np.quantile(xy,.98,axis=0)+.01
        # Do not invent an enlarged seat to cover a moving body. Dimensions come
        # from this observed planar patch and are checked against furniture scale.
        size=high-low
        if np.any(size<.18) or np.any(size>.95): raise ValueError('Implausible source seat patch dimensions '+str(size))
        mid=axes@((low+high)/2); yaw=float(np.arctan2(axis[1],axis[0]))
        corners=np.array([[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]])@axes.T
        seats.append({'name':f'seat_{person}_proxy','kind':'seat','center_xy_m':mid.tolist(),
                      'top_z_m':top,'half_size_xy_m':(size/2).tolist(),'yaw_rad':yaw,
                      'xy_polygon_m':corners.tolist(),'source_initial_pelvis_xy_m':center.tolist(),
                      'source_triangle_count':int(np.sum(pick)),'source_surface_area_m2':float(np.sum(w)),
                      'fit_method':'area-weighted near-horizontal source triangle height mode and planar PCA patch',
                      'thickness_m':.055,'friction':[.8,.01,.0005]})
    return vertices,faces,{'schema':'two_sitting_support.v1','source_to_sim_U':U.tolist(),
              'ground':{'name':'ground_proxy','z_eq_ax_by_c':plane.tolist(),'triangle_count':int(np.sum(selected)),
                        'fit_method':'area-weighted robust fit of modal foreground ground triangles',
                        'friction':[1.,.01,.0005]},'seats':seats,
              'limitations':['Ground and seats are fixed simplified colliders fitted to RGB-only estimated geometry.',
                             'Furniture rear/occluded geometry and material parameters are not measured.',
                             'No seat enlargement based on later body trajectories.']}


def scene_colliders(world,spec,np,mj):
    ground=spec['ground']; a,b,c=ground['z_eq_ax_by_c']
    normal=np.array([-a,-b,1.]);normal/=np.linalg.norm(normal);quat=np.empty(4)
    mj.mju_quatZ2Vec(quat,normal)
    ET.SubElement(world,'geom',name=ground['name'],type='plane',pos=num([0,0,c]),quat=num(quat),
                  size='0 0 .05',contype='2',conaffinity='5',friction=num(ground['friction']),rgba='.35 .4 .35 1',group='2')
    for seat in spec['seats']:
        x,y=seat['center_xy_m']; top=seat['top_z_m']; hx,hy=seat['half_size_xy_m']; h=seat['thickness_m']
        ET.SubElement(world,'geom',name=seat['name'],type='box',pos=num([x,y,top-h/2]),
                      euler=num([0,0,seat['yaw_rad']]),size=num([hx,hy,h/2]),
                      contype='2',conaffinity='5',friction=num(seat['friction']),rgba='.65 .4 .18 .7',group='0')
        groundz=float(x*a+y*b+c);height=(top-h-groundz)/2
        ET.SubElement(world,'geom',name=seat['name']+'_pedestal',type='box',pos=num([x,y,groundz+height]),
                      euler=num([0,0,seat['yaw_rad']]),size=num([min(hx,.07),min(hy,.07),max(height,.02)]),
                      contype='2',conaffinity='5',friction=num(seat['friction']),rgba='.4 .3 .2 .6',group='0')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/two_sitting_people/rgb_174114_sim_handoff')
    p.add_argument('--source',type=Path,default=ROOT/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_isaac52_isolated.xml')
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/two_sitting_pipeline/assets')
    p.add_argument('--ankle-pitch-lower',type=float)
    p.add_argument('--ankle-pitch-upper',type=float)
    a=p.parse_args()
    import numpy as np,mujoco as mj
    a.output_dir.mkdir(parents=True,exist_ok=True)
    robot,assets,motors,equality=load_robot(a.source)
    if (a.ankle_pitch_lower is None)!=(a.ankle_pitch_upper is None):
        raise ValueError('Provide both ankle pitch bounds')
    if a.ankle_pitch_lower is not None:
        if a.ankle_pitch_lower>=a.ankle_pitch_upper:raise ValueError('Invalid ankle pitch bounds')
        for joint in robot.iter('joint'):
            if (joint.get('name')or'').endswith('ankle_pitch_joint'):
                joint.set('range',num([a.ankle_pitch_lower,a.ankle_pitch_upper]));joint.set('limited','true')
    vertices,faces,spec=support_fit(a.data,np)
    visual=a.output_dir/'scene_visual.obj'
    with visual.open('w',encoding='ascii')as f:
        for v in vertices:f.write('v '+num(v)+'\n')
        for tri in faces:f.write('f '+' '.join(str(int(i)+1)for i in tri)+'\n')
    outputs={}
    for single in (True,False):
        doc=base_document();ass=copy.deepcopy(assets);doc.append(ass);world=ET.SubElement(doc,'worldbody')
        ET.SubElement(world,'light',name='key',pos='0 -3 5',dir='0 1 -1')
        scene_colliders(world,spec,np,mj)
        act=ET.SubElement(doc,'actuator');eq=ET.SubElement(doc,'equality')
        if single:
            world.append(copy.deepcopy(robot));act.extend(copy.deepcopy(motors));eq.extend(copy.deepcopy(equality))
            path=a.output_dir/'vita_single_proxy.xml'
        else:
            ET.SubElement(ass,'mesh',name='source_scene_mesh',file=str(visual.resolve()))
            ET.SubElement(world,'geom',name='source_scene_visual',type='mesh',mesh='source_scene_mesh',
                          contype='0',conaffinity='0',rgba='.57 .6 .55 1',group='1')
            for person,ctype,affinity in((1,1,6),(2,4,3)):
                body,actuators,eqs=prefixed(robot,motors,equality,f'p{person}_',ctype,affinity)
                world.append(body);act.extend(actuators);eq.extend(eqs)
            path=a.output_dir/'two_vita.xml'
        if len(eq)==0:doc.remove(eq)
        save_xml(doc,path);model=mj.MjModel.from_xml_path(str(path))
        expected=(59,58,52)if single else(118,116,104)
        if(model.nq,model.nv,model.nu)!=expected:raise ValueError('Unexpected full Vita dimensions')
        mj.mj_saveModel(model,str(path.with_suffix('.mjb')),None)
        outputs['single_proxy'if single else'dual_scene']={'path':str(path),'sha256':sha(path),'nq':model.nq,'nv':model.nv,'nu':model.nu,'nmocap':model.nmocap}
    (a.output_dir/'support_geometry.json').write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
    report={'schema':'two_sitting_assets.v1','source_robot':str(a.source),'source_robot_sha256':sha(a.source),
            'source_scene_sha256':sha(a.data/'scene/pi3_mesh.npz'),'source_to_sim_U':spec['source_to_sim_U'],
            'outputs':outputs,'support_geometry':spec,'robot_mass_inertia':'preserved from supplied Vita near-source model',
            'dynamic_free_roots':2,'dynamic_objects':0,'furniture_fixed':True,
            'same_robot_self_collision':False,'inter_robot_collision':True,'physical_sequence_passed':False}
    report['explicit_ankle_pitch_bounds_rad']=[a.ankle_pitch_lower,a.ankle_pitch_upper]
    (a.output_dir/'assets_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
