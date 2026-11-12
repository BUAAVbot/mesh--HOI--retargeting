"""Extract an unprefixed single Vita proxy from the two-actor scene XML."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prefix',default='p1_');a=p.parse_args()
    for path in [a.output,a.output.with_suffix('.mjb'),a.output.with_suffix('.json')]:
        if path.exists():raise FileExistsError('Refusing overwrite '+str(path))
    import mujoco as mj
    import numpy as np
    source=ET.parse(a.source).getroot();old=mj.MjModel.from_xml_path(str(a.source))
    r=ET.Element('mujoco',model='rgb_near_single_hand_v3_proxy')
    for tag in ['compiler','option','visual','statistic','default']:
        elem=source.find(tag)
        if elem is not None:r.append(copy.deepcopy(elem))
    b=copy.deepcopy(source.find(f"worldbody/body[@name='{a.prefix}base_link']"))
    if b is None:raise ValueError('Actor root missing')
    for elem in b.iter():
        if elem.get('name','').startswith(a.prefix):elem.set('name',elem.get('name')[len(a.prefix):])
        if elem.tag=='geom'and int(elem.get('contype','0')):
            # Strip the other actor bit while preserving optional hand bits.
            elem.set('conaffinity',str(int(elem.get('conaffinity','0')) & ~4))
    asset=ET.SubElement(r,'asset');used={g.get('mesh')for g in b.findall('.//geom')if g.get('mesh')}
    for x in source.find('asset'):
        if x.tag=='mesh'and x.get('name')in used:asset.append(copy.deepcopy(x))
    world=ET.SubElement(r,'worldbody');world.append(b)
    ET.SubElement(world,'geom',name='ground_proxy',type='plane',pos='0 0 0',size='0 0 .05',
                  contype='2',conaffinity='3',friction='1 .01 .0005')
    act=ET.SubElement(r,'actuator')
    for original in source.find('actuator'):
        if not original.get('name','').startswith(a.prefix):continue
        x=copy.deepcopy(original)
        for key in ['name','joint']:
            if x.get(key,'').startswith(a.prefix):x.set(key,x.get(key)[len(a.prefix):])
        act.append(x)
    # Preserve within-actor calibrated tendon/mimic constraints. Shoe welds
    # belong only to the actual shared scene and are intentionally omitted.
    equality=source.find('equality')
    if equality is not None:
        kept=[]
        for original in equality:
            if original.tag!='joint':continue
            if not all(original.get(k,'').startswith(a.prefix)for k in ['joint1','joint2']):continue
            x=copy.deepcopy(original)
            for key in ['joint1','joint2']:
                x.set(key,x.get(key)[len(a.prefix):])
            x.set('name',x.get('name','').replace(a.prefix,''))
            kept.append(x)
        if kept:
            eq=ET.SubElement(r,'equality')
            for x in kept:eq.append(x)
    ET.indent(r);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(ET.tostring(r,encoding='unicode')+'\n',encoding='utf-8')
    new=mj.MjModel.from_xml_path(str(a.output))
    if (new.nq,new.nv,new.nu)!=(59,58,52):raise ValueError('Expected Vita52+free root proxy')
    errors={k:0. for k in ['body_mass','body_inertia','body_ipos','body_iquat']}
    for bid in range(1,new.nbody):
        name=mj.mj_id2name(new,mj.mjtObj.mjOBJ_BODY,bid)
        ob=mj.mj_name2id(old,mj.mjtObj.mjOBJ_BODY,a.prefix+name)
        for key in errors:errors[key]=max(errors[key],float(np.max(abs(getattr(new,key)[bid]-getattr(old,key)[ob]))))
    if any(errors.values()):raise ValueError('Proxy body properties changed')
    mj.mj_saveModel(new,str(a.output.with_suffix('.mjb')))
    report={'source':str(a.source),'source_sha256':sha(a.source),'proxy':str(a.output),'proxy_sha256':sha(a.output),
            'nq':new.nq,'nv':new.nv,'nu':new.nu,'neq':new.neq,'inertia_max_abs_errors':errors,
            'semantics':'Inference/IK mirror only; actual scene integrations use shared two-actor model.'}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))


if __name__=='__main__':main()
