"""Derive a wider inferred shoe cavity without changing frozen source assets."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def num(x):return ' '.join(f'{float(v):.17g}' for v in x)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=float,default=.14)
    p.add_argument('--reference',type=Path,help='Optional reference for read-only fitted foot/shoe clearance audit')
    a=p.parse_args()
    if a.source.resolve()==a.output.resolve():raise ValueError('Never overwrite frozen source model')
    validation_dir=a.output.parent/(a.output.stem+'_validation')
    geometry_path=a.output.with_name(a.output.stem+'_wear_geometry.json')
    for path in [a.output,a.output.with_suffix('.mjb'),a.output.with_suffix('.json'),validation_dir,geometry_path]:
        if path.exists():raise FileExistsError('Choose a fresh output path; refusing to overwrite '+str(path))
    import numpy as np
    import mujoco as mj
    from rgb_near_sim_assets import validate_shoe_support
    r=ET.parse(a.source).getroot();shoe=r.find("worldbody/body[@name='shoe']")
    old=mj.MjModel.from_xml_path(str(a.source))
    gid=mj.mj_name2id(old,mj.mjtObj.mjOBJ_GEOM,'shoe_sole_proxy')
    Q=np.empty(9);mj.mju_quat2Mat(Q,old.geom_quat[gid]);Q=Q.reshape(3,3)
    sole=shoe.find("geom[@name='shoe_sole_proxy']")
    old_width=float(sole.get('size').split()[1])*2
    size=np.asarray([float(x)for x in sole.get('size').split()]);size[1]=a.width/2;sole.set('size',num(size))
    toe=shoe.find("geom[@name='shoe_toe_wall_proxy']")
    s=np.asarray([float(x)for x in toe.get('size').split()]);s[1]=a.width/2;toe.set('size',num(s))
    height=None
    for tag,sign in [('left',-1),('right',1)]:
        wall=shoe.find(f"geom[@name='shoe_{tag}_wall_proxy']")
        s=np.asarray([float(x)for x in wall.get('size').split()])
        local=Q.T@np.asarray([float(x)for x in wall.get('pos').split()])
        local[1]=sign*(a.width/2-s[1]);wall.set('pos',num(Q@local));height=2*s[2]
    inertial=shoe.find('inertial');mass=float(inertial.get('mass'))
    length=2*size[0]
    inertia=mass/12*np.asarray([a.width*a.width+height*height,length*length+height*height,length*length+a.width*a.width])
    inertial.set('diaginertia',num(inertia))
    ET.indent(r);a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(ET.tostring(r,encoding='unicode')+'\n',encoding='utf-8')
    model=mj.MjModel.from_xml_path(str(a.output))
    robot_ids=[b for b in range(model.nbody)if(mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,b)or'').startswith(('p1_','p2_'))]
    for key in ['body_mass','body_inertia','body_ipos','body_iquat']:
        if not np.array_equal(getattr(model,key)[robot_ids],getattr(old,key)[robot_ids]):raise ValueError('Robot properties changed')
    mj.mj_saveModel(model,str(a.output.with_suffix('.mjb')))
    validation_dir.mkdir()
    support=validate_shoe_support(r.find('asset'),shoe,validation_dir,mj,np)
    report={'source_model':str(a.source.resolve()),'source_sha256':sha(a.source),
            'model':str(a.output.resolve()),'model_sha256':sha(a.output),
            'old_assumed_width_m':old_width,'new_assumed_width_m':a.width,
            'new_inner_cavity_width_m':a.width-.008,'source_geometry_modified':False,
            'shoe_mass_kg':mass,'shoe_diaginertia':inertia.tolist(),
            'robot_inertias_unchanged':True,'shoe_static_support_test':support,
            'reason':'Original partial shoe surface does not observe cross-shoe width; initial 0.11 m cavity intersected 0.12564 m wide Vita foot. Wider explicit proxy is needed for feasible inferred wear.',
            'inferred_geometry_not_measurement':True}
    if a.reference:
        subprocess.run([sys.executable,str(Path(__file__).with_name('rgb_near_wear_geometry.py')),
                        '--model',str(a.output),'--reference',str(a.reference),
                        '--output',str(geometry_path)],check=True,capture_output=True,text=True)
        geometry=json.loads(geometry_path.read_text())
        report['fitted_wear_geometry_audit']={'path':str(geometry_path.resolve()),'sha256':sha(geometry_path),
            'reference':str(a.reference.resolve()),'reference_sha256':sha(a.reference),
            'all_robot_shoe_min_distance_m':geometry['all_robot_shoe_min_distance_m'],
            'required_foot_and_shoe_floor_lift_m':geometry['required_foot_and_shoe_floor_lift_m'],
            'recommended_foot_min_z_m':geometry['recommended_foot_min_z_m'],
            'note':'Audits the fitted wear transform, not a dynamic trial. Foot/floor correction must still be applied in the reference.'}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
