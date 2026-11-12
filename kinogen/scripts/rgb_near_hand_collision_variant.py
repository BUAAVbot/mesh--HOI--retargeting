"""Replace whole-finger proxies with the user's per-link left/right hand assets."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(x):return ' '.join(f'{float(v):.17g}'for v in x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--urdf',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hand-self-collision',action='store_true',help='Enable collisions within each individual hand only; preserve other robot self masks')
    a=p.parse_args()
    for path in [a.output,a.output.with_suffix('.mjb'),a.output.with_suffix('.json')]:
        if path.exists():raise FileExistsError('Refusing to overwrite '+str(path))
    import numpy as np
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    old=mj.MjModel.from_xml_path(str(a.source))
    xml=ET.parse(a.source).getroot();urdf=ET.parse(a.urdf).getroot();asset=xml.find('asset')
    links={x.get('name'):x for x in urdf.findall('link')}
    incoming={j.find('child').get('link'):j for j in urdf.findall('joint')}
    audits=[];added_assets={};right_old_sources=[];collision_count=0
    def origin(elem):
        o=elem.find('origin')
        pos=np.zeros(3)if o is None else np.asarray([float(x)for x in o.get('xyz','0 0 0').split()])
        rpy=np.zeros(3)if o is None else np.asarray([float(x)for x in o.get('rpy','0 0 0').split()])
        R=Rotation.from_euler('xyz',rpy).as_matrix();quat=np.empty(4);mj.mju_mat2Quat(quat,R.ravel())
        return pos,R,quat
    for b in xml.findall('.//body'):
        bname=b.get('name','')
        if not bname.startswith(('p1_lh_','p1_rh_','p2_lh_','p2_rh_')):continue
        prefix=bname[:3];linkname=bname[3:];link=links[linkname];jid=incoming[linkname]
        bid=mj.mj_name2id(old,mj.mjtObj.mjOBJ_BODY,bname)
        upos,uR,_=origin(jid);bodyR=np.empty(9);mj.mju_quat2Mat(bodyR,old.body_quat[bid]);bodyR=bodyR.reshape(3,3)
        translation_error=float(np.max(abs(upos-old.body_pos[bid])))
        rotation_error=float(np.linalg.norm(Rotation.from_matrix(uR@bodyR.T).as_rotvec()))
        if translation_error>2e-5 or rotation_error>2e-4:
            raise ValueError(f'Hand link frame differs from user URDF: {bname}: {translation_error}, {rotation_error}')
        old_geoms=[g.attrib.copy()for g in b.findall('geom')]
        if linkname.startswith('rh_'):
            right_old_sources += [g.get('mesh') for g in b.findall('geom')if g.get('mesh')]
        old_collision=[g.attrib.copy()for g in b.findall('geom')if int(g.get('contype','0'))]
        for g in list(b.findall('geom')):b.remove(g)
        entries=[]
        for kind in ['visual','collision']:
            for i,part in enumerate(link.findall(kind)):
                mesh=part.find('geometry/mesh')
                if mesh is None:raise ValueError('Expected per-link STL hand geometry in user URDF')
                filename=mesh.get('filename')
                if filename.startswith('package://'):raise ValueError('Use the audited relative URDF')
                path=(a.urdf.parent/filename).resolve()
                if not path.is_file():raise FileNotFoundError(path)
                scale=mesh.get('scale','1 1 1')
                key=(kind,linkname,i)
                name='user_'+kind+'_'+linkname+'_'+str(i)
                if key not in added_assets:
                    ET.SubElement(asset,'mesh',name=name,file=str(path),scale=scale)
                    added_assets[key]={'name':name,'file':str(path),'sha256':sha(path),'scale':scale}
                pos,_,quat=origin(part)
                attributes={'name':prefix+kind+'_'+linkname+'_'+str(i),'type':'mesh','mesh':name,
                            'pos':num(pos),'quat':num(quat),'density':'0'}
                if kind=='collision':
                    actor_type=1 if prefix=='p1_'else 4
                    actor_affinity=6 if prefix=='p1_'else 3
                    hand_bit={('p1_','lh_'):16,('p1_','rh_'):32,('p2_','lh_'):64,('p2_','rh_'):128}[(prefix,linkname[:3])]
                    attributes.update(contype=str(actor_type|(hand_bit if a.hand_self_collision else 0)),
                                      conaffinity=str(actor_affinity|(hand_bit if a.hand_self_collision else 0)),group='0',rgba='.65 .65 .65 1')
                    collision_count+=1
                else:attributes.update(contype='0',conaffinity='0',group='1',rgba='.62 .63 .62 1')
                ET.SubElement(b,'geom',**attributes)
                entries.append({'kind':kind,'mesh_asset':name,'path':str(path),'sha256':sha(path),'urdf_origin_xyz':pos.tolist(),'urdf_origin_quat_wxyz':quat.tolist(),'scale':scale})
        if len(link.findall('collision'))!=1:raise ValueError('Expected exactly one separated collision hull per hand link')
        audits.append({'body':bname,'urdf_link':linkname,'translation_frame_error_m':translation_error,
                       'rotation_frame_error_rad':rotation_error,'old_collision':old_collision,'new_geometry':entries})
    used={g.get('mesh')for g in xml.findall('.//geom')if g.get('mesh')}
    for elem in list(asset):
        if elem.tag=='mesh'and elem.get('name')not in used:asset.remove(elem)
    ET.indent(xml);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(ET.tostring(xml,encoding='unicode')+'\n',encoding='utf-8')
    new=mj.MjModel.from_xml_path(str(a.output))
    if (old.nq,old.nv,old.nu)!=(new.nq,new.nv,new.nu):raise ValueError('Control/state dimensions changed')
    for key in ['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat',
                'jnt_axis','jnt_pos','jnt_range','dof_armature','actuator_gainprm','actuator_biasprm','actuator_forcerange']:
        if not np.array_equal(getattr(old,key),getattr(new,key)):raise ValueError('Model contract changed: '+key)
    mj.mj_saveModel(new,str(a.output.with_suffix('.mjb')))
    source_meshes={x.get('name'):x.get('file')for x in ET.parse(a.source).getroot().find('asset')if x.tag=='mesh'}
    reused_right_old_files=sorted({source_meshes[name]for name in right_old_sources if name in source_meshes})
    report={'schema':'rgb_near.user_perlink_hand_collision.v3','model':str(a.output.resolve()),'model_sha256':sha(a.output),
            'model_mjb_sha256':sha(a.output.with_suffix('.mjb')),'source':str(a.source.resolve()),'source_sha256':sha(a.source),
            'user_urdf':str(a.urdf.resolve()),'user_urdf_sha256':sha(a.urdf),
            'hand_links':len(audits),'active_hand_collision_geoms':collision_count,'active_geoms_per_hand':collision_count//4,
            'old_active_geoms_per_hand':7,'whole_proximal_finger_capsules_removed':True,'distal_hulls_follow_dip':True,
            'left_right_assets_uniquely_named':True,'right_old_mesh_sources':reused_right_old_files,
            'within_hand_self_collision_enabled':a.hand_self_collision,
            'other_robot_self_collision_unchanged_disabled':True,
            'within_hand_bits':{'p1_lh_':16,'p1_rh_':32,'p2_lh_':64,'p2_rh_':128}if a.hand_self_collision else None,
            'contract_arrays_unchanged':['body mass/inertia/COM/orientation','body/joint transforms and ranges','actuator gains/bias/force limits','armature'],
            'hand_frame_audits':audits,'mesh_assets':list(added_assets.values()),
            'limitations':['Each link mesh is used as its MuJoCo convex hull; holes within a single link may be filled, but separate fingers are not merged into one hull.',
                           'Other-body robot self collisions remain disabled. Individual-hand self collisions are an explicit optional change; cross-actor, shoe and environment contacts stay enabled.',
                           'This asset edit does not validate inferred grasp trajectories or prove force ground truth. Existing finger motor rotor assumptions belong to the separate physics freeze.']}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items()if k not in('hand_frame_audits','mesh_assets')}),flush=True)


if __name__=='__main__':main()
