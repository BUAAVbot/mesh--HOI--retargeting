"""Derive a calibrated six-command O6 hand scene without altering frozen assets."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(x):return ' '.join(format(float(v),'.17g')for v in x)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--hand-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--hand-self-collision',action='store_true')
    a=p.parse_args()
    for q in [a.output,a.output.with_suffix('.mjb'),a.output.with_suffix('.json')]:
        if q.exists():raise FileExistsError('Refusing overwrite '+str(q))
    import numpy as np
    import mujoco as mj
    from scipy.spatial.transform import Rotation
    old=mj.MjModel.from_xml_path(str(a.source));xml=ET.parse(a.source).getroot();asset=xml.find('asset')
    eq=xml.find('equality')
    if eq is None:eq=ET.SubElement(xml,'equality')
    profiles={};side_data={};mesh_records={};frame_changes=[];couplings=[];drivers=[]
    def origin(element):
        o=element.find('origin');pos=np.zeros(3)if o is None else np.fromstring(o.get('xyz','0 0 0'),sep=' ')
        rpy=np.zeros(3)if o is None else np.fromstring(o.get('rpy','0 0 0'),sep=' ')
        quat=Rotation.from_euler('xyz',rpy).as_quat(scalar_first=True)
        return pos,quat
    for side,short in [('LEFT','lh_'),('RIGHT','rh_')]:
        folder=a.hand_root/side;up=next(folder.glob('*.urdf'));jp=next(folder.glob('*.json'))
        ur=ET.parse(up).getroot();cal=json.loads(jp.read_text())
        links={b.get('name'):b for b in ur.findall('link')};joints={j.get('name'):j for j in ur.findall('joint')}
        incoming={j.find('child').get('link'):j for j in ur.findall('joint')}
        side_data[short]=(up,links,joints,incoming,cal)
        profiles[side]={'urdf':str(up.resolve()),'urdf_sha256':sha(up),'calibration':str(jp.resolve()),'calibration_sha256':sha(jp),
                        'profile_id':cal.get('profile_id'),'calibration_scope':cal.get('calibration_scope'),
                        'command_names':cal['command_names'],'baseline_command_u8':cal.get('baseline_command_u8'),
                        'joint_status':{k:v.get('calibration_status')for k,v in cal['joints'].items()}}
    for body in xml.findall('.//body'):
        name=body.get('name','')
        if not name.startswith(('p1_lh_','p1_rh_','p2_lh_','p2_rh_')):continue
        actor=name[:3];ln=name[3:];short=ln[:3];up,links,joints,incoming,cal=side_data[short]
        if ln in incoming:
            uj=incoming[ln];pos,quat=origin(uj);bid=mj.mj_name2id(old,mj.mjtObj.mjOBJ_BODY,name)
            frame_changes.append({'body':name,'old_pos':old.body_pos[bid].tolist(),'new_pos':pos.tolist(),'old_quat':old.body_quat[bid].tolist(),'new_quat':quat.tolist()})
            body.set('pos',fmt(pos));body.set('quat',fmt(quat))
            j=body.find('joint');lim=uj.find('limit');j.set('axis',uj.find('axis').get('xyz'));j.set('range',lim.get('lower')+' '+lim.get('upper'))
            jname=uj.get('name');jc=cal['joints'][jname]
            if jc.get('passive'):
                coeff=jc['coupling_coefficients']
                if len(coeff)>5 and any(coeff[5:]):raise ValueError('Unsupported polynomial degree')
                coeff=(coeff+[0.]*5)[:5]
                ename='o6_mimic_'+actor+jname
                ET.SubElement(eq,'joint',name=ename,joint1=actor+jname,joint2=actor+jc['source_joint'],polycoef=fmt(coeff),solref='.006 1',solimp='.95 .99 .001')
                motor=xml.find(f"actuator/position[@joint='{actor+jname}']")
                if motor is None:raise ValueError('Missing ABI motor '+actor+jname)
                motor.set('kp','0');motor.set('kv','0')
                # Force range is an ABI bound only; gain and bias are zero.
                couplings.append({'equality':ename,'dependent_joint':actor+jname,'driver_joint':actor+jc['source_joint'],'polycoef':coeff,'actuator':motor.get('name'),
                                  'urdf_endpoint_linear_fallback':uj.find('mimic').attrib,'calibration_status':jc.get('calibration_status')})
            else:
                drivers.append({'joint':actor+jname,'sdk_channel':jc['sdk_channel'],'motor_index':jc['motor_index'],'range_rad':[float(lim.get('lower')),float(lim.get('upper'))],
                                'zero_command_u8':jc.get('zero_command_u8'),'calibration_status':jc.get('calibration_status')})
        for g in list(body.findall('geom')):body.remove(g)
        for kind in ['visual','collision']:
            parts=links[ln].findall(kind)
            if len(parts)!=1:raise ValueError('Expected one per-link mesh '+ln)
            part=parts[0];mesh=part.find('geometry/mesh');path=(up.parent/mesh.get('filename')).resolve();key=kind+'_'+ln
            if key not in mesh_records:
                meshname='o6_v4_'+key;scale=mesh.get('scale','1 1 1')
                ET.SubElement(asset,'mesh',name=meshname,file=str(path),scale=scale)
                mesh_records[key]={'name':meshname,'file':str(path),'sha256':sha(path),'scale':scale}
            pos,quat=origin(part);at={'name':actor+kind+'_'+ln+'_0','type':'mesh','mesh':mesh_records[key]['name'],'pos':fmt(pos),'quat':fmt(quat),'density':'0'}
            if kind=='collision':
                hb={('p1_','lh_'):16,('p1_','rh_'):32,('p2_','lh_'):64,('p2_','rh_'):128}[(actor,short)]if a.hand_self_collision else 0
                at.update(contype=str((1 if actor=='p1_'else 4)|hb),conaffinity=str((6 if actor=='p1_'else 3)|hb),group='0',rgba='.65 .65 .65 1')
            else:at.update(contype='0',conaffinity='0',group='1',rgba='.62 .63 .62 1')
            ET.SubElement(body,'geom',**at)
    used={g.get('mesh')for g in xml.findall('.//geom')if g.get('mesh')}
    for el in list(asset):
        if el.tag=='mesh'and el.get('name')not in used:asset.remove(el)
    ET.indent(xml);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(ET.tostring(xml,encoding='unicode')+'\n',encoding='utf-8')
    new=mj.MjModel.from_xml_path(str(a.output))
    if (new.nq,new.nv,new.nu,new.neq)!=(125,122,104,24):raise ValueError('Unexpected scene/ABI dimensions')
    for key in ['body_mass','body_inertia','body_ipos','body_iquat','dof_armature']:
        if not np.array_equal(getattr(old,key),getattr(new,key)):raise ValueError('Preserved body inertia changed '+key)
    body_unchanged=[]
    for j in range(old.njnt):
        n=mj.mj_id2name(old,mj.mjtObj.mjOBJ_JOINT,j)
        if n.startswith(('p1_lh_','p1_rh_','p2_lh_','p2_rh_')):continue
        for key in ['jnt_range','jnt_axis','jnt_pos']:
            if not np.array_equal(getattr(old,key)[j],getattr(new,key)[j]):raise ValueError('Non-hand joint modified '+n)
        body_unchanged.append(n)
    zero=np.flatnonzero(np.all(new.actuator_gainprm==0,axis=1)&np.all(new.actuator_biasprm==0,axis=1))
    if len(zero)!=20 or len(couplings)!=20 or len(drivers)!=24:raise ValueError('O6 drive counts invalid')
    audit=[]
    for c in couplings:
        dj=mj.mj_name2id(new,mj.mjtObj.mjOBJ_JOINT,c['driver_joint']);dep=mj.mj_name2id(new,mj.mjtObj.mjOBJ_JOINT,c['dependent_joint'])
        xs=np.linspace(*new.jnt_range[dj],101);ys=np.polynomial.polynomial.polyval(xs,c['polycoef'])
        audit.append({'dependent_joint':c['dependent_joint'],'dependent_polynomial_range_rad':[float(ys.min()),float(ys.max())],
                      'urdf_range_rad':new.jnt_range[dep].tolist(),'upper_margin_rad':float(new.jnt_range[dep,1]-ys.max())})
        if ys.min()<new.jnt_range[dep,0]-1e-7 or ys.max()>new.jnt_range[dep,1]+1e-7:raise ValueError('Coupling violates calibrated limits')
    mj.mj_saveModel(new,str(a.output.with_suffix('.mjb')))
    report={'schema':'rgb_near.O6_calibrated.v4','model':str(a.output.resolve()),'model_sha256':sha(a.output),'model_mjb_sha256':sha(a.output.with_suffix('.mjb')),
            'source':str(a.source.resolve()),'source_sha256':sha(a.source),'source_zip_sha256':'9c5ec039afc3e510ec101370b269f8e6ebac61e8402f0e326c032baa5fb911c1',
            'nq':new.nq,'nv':new.nv,'nu_abi':new.nu,'powered_channels':new.nu-len(zero),'o6_independent_drivers_per_hand':6,'o6_passive_mimics_per_hand':5,
            'hand_links_per_hand':12,'hand_active_collision_geoms':48,'within_hand_self_collision_enabled':a.hand_self_collision,'other_body_self_collision_enabled':False,
            'source_profiles':profiles,'drivers':drivers,'couplings':couplings,'coupling_range_audit':audit,'zero_force_actuator_ids':zero.tolist(),
            'hand_frame_changes':frame_changes,'mesh_assets':list(mesh_records.values()),'nonhand_joints_unchanged':body_unchanged,
            'limitations':['O6 per-link meshes collide as convex hulls; single-link cavities are filled.','Self collision remains disabled by default within a robot; optional individual-hand masks need contact audit.',
                          'Runtime quadratic calibration is used instead of URDF endpoint-linear fallback. Command255 is open; command mapping must use calibration lookups, not linear range scaling.',
                          'Right thumb and pinky have measured calibration; index/middle/ring are transferred. Left calibration is mirrored/transferred, not independently measured.',
                          'Original robot mass/inertia/COM are preserved. URDF effort100 is not a measured motor torque; powered hand position gains8/kv.2 and ±3Nm are conservative inherited assumptions.',
                          'Twenty dependent ABI actuator channels have zero gain and bias and therefore produce zero direct actuator force. Mimic constraint reactions transmit force; they are not independent motors.',
                          'Equality compliance solref .006s/1 and solimp .95/.99/.001 is a declared simulation approximation; no measured tendon elasticity is provided.']}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items()if k not in ['source_profiles','drivers','couplings','coupling_range_audit','hand_frame_changes','mesh_assets','nonhand_joints_unchanged','limitations']}))

if __name__=='__main__':main()
