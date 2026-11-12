"""New geometry branch: restore only source pelvis/hip cylinders, preserving O6."""
from pathlib import Path
import argparse,json,hashlib,copy,xml.etree.ElementTree as ET
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v):return' '.join(format(float(x),'.17g')for x in v)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--source-urdf',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    for path in [a.output,a.output.with_suffix('.mjb'),a.output.with_suffix('.json')]:
        if path.exists():raise FileExistsError('Refusing overwrite '+str(path))
    import numpy as np,mujoco as mj
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    old=mj.MjModel.from_xml_path(str(a.source));xml=ET.parse(a.source).getroot();ur=ET.parse(a.source_urdf)
    selected=['collision_pelvis_cylinder','collision_l_hip_roll_cylinder','collision_l_hip_front_roll_cylinder','collision_r_hip_roll_cylinder','collision_r_hip_front_roll_cylinder']
    changes=[];inventory=[]
    for link in ur.findall('link'):
        for c in link.findall('collision'):
            cyl=c.find('geometry/cylinder')
            if cyl is None:continue
            for actor in ['p1_','p2_']:
                name=actor+c.get('name');g=xml.find(f".//geom[@name='{name}']")
                inventory.append({'source_collision':c.get('name'),'body':link.get('name'),'actor':actor,'source_geometry_xml':ET.tostring(c,encoding='unicode'),
                                  'compiled_geom_exists':g is not None,'compiled_geometry_before':g.attrib.copy()if g is not None else None,'selected_for_restore':c.get('name')in selected})
                if c.get('name')not in selected:continue
                if g is None or g.get('type')!='capsule':raise ValueError('Selected capsule missing '+name)
                expected=np.asarray([float(cyl.get('radius')),float(cyl.get('length'))/2])
                if np.max(abs(np.fromstring(g.get('size'),sep=' ')[:2]-expected))>1e-8:raise ValueError('Source size mismatch '+name)
                before=g.attrib.copy();g.set('type','cylinder');changes.append({'geom':name,'before':before,'after':g.attrib.copy(),'source_collision_xml':ET.tostring(c,encoding='unicode')})
    if len(changes)!=10:raise ValueError('Need five pelvis/hip primitives per actor')
    ET.indent(xml);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(ET.tostring(xml,encoding='unicode')+'\n',encoding='utf-8')
    m=mj.MjModel.from_xml_path(str(a.output))
    for key in ['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_range','jnt_axis','jnt_pos','dof_armature','actuator_gainprm','actuator_biasprm','actuator_forcerange','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','eq_data','eq_active0']:
        if not np.array_equal(getattr(old,key),getattr(m,key)):raise ValueError('Preserved contract differs '+key)
    mj.mj_saveModel(m,str(a.output.with_suffix('.mjb')))
    ref=np.load(a.reference,allow_pickle=False);q=np.empty(m.nq);q[:59]=ref['qpos'][0,0];q[59:118]=ref['qpos'][0,1];q[118:]=ref['object_target_pose'][0]
    data=mj.MjData(m);data.qpos[:]=q;mj.mj_fwdPosition(m,data)
    seat=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'bench_seat_proxy');seat_top=float(m.geom_pos[seat,2]+m.geom_size[seat,2]);initial=[]
    for c in changes:
        g=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,c['geom']);az=abs(data.geom_xmat[g].reshape(3,3)[2,2]);r,h=m.geom_size[g,:2]
        lower=float(data.geom_xpos[g,2]-h*az-r*np.sqrt(max(0,1-az*az)));initial.append({'geom':c['geom'],'source_cylinder_bottom_world_z_m':lower,'initial_gap_above_seat_m':lower-seat_top})
    feasibility=[]
    for actor in ['p1_','p2_']:
        root=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,actor+'vita_demo_free_root');ra=int(m.jnt_qposadr[root])
        for shift in [.035,.040,.042,.045]:
            for side in ['left','right']:
                body=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,actor+side+'_ankle_roll_link');data.qpos[:]=q;mj.mj_fwdPosition(m,data)
                P=data.xpos[body].copy();R=data.xmat[body].reshape(3,3).copy()
                suffixes=['hip_pitch_joint','hip_roll_joint','hip_yaw_joint','knee_joint','ankle_pitch_joint','ankle_roll_joint']
                js=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,actor+side+'_'+s)for s in suffixes]
                qa=m.jnt_qposadr[js];lo=m.jnt_range[js,0]+1e-6;hi=m.jnt_range[js,1]-1e-6
                start=q[qa].copy();base=q.copy();base[ra+2]-=shift
                def fun(x):
                    data.qpos[:]=base;data.qpos[qa]=x;mj.mj_fwdPosition(m,data)
                    return np.r_[data.xpos[body]-P,.1*Rotation.from_matrix(data.xmat[body].reshape(3,3)@R.T).as_rotvec()]
                sol=least_squares(fun,np.clip(start,lo,hi),bounds=(lo,hi),max_nfev=120,ftol=1e-10,xtol=1e-10,gtol=1e-10);res=fun(sol.x)
                feasibility.append({'actor':actor,'side':side,'root_lowering_m':shift,'ankle_position_error_m':float(np.linalg.norm(res[:3])),
                                    'ankle_rotation_error_rad':float(np.linalg.norm(res[3:])/.1),'joint_names':[actor+side+'_'+s for s in suffixes],
                                    'q_original_rad':start.tolist(),'q_solved_rad':sol.x.tolist(),'delta_q_rad':(sol.x-start).tolist(),'smallest_limit_margin_rad':float(np.min(np.minimum(sol.x-lo,hi-sol.x))),
                                    'scope':'Single-frame isolated leg FK reachability only; no full-body collision or seat equilibrium certification.'})
    # An isolated native cylinder drop checks its flat-end support behavior.
    # Test-body unit mass/inertia are diagnostics, not changes to robot inertia.
    support=[]
    for side in ['l','r']:
        name='p1_collision_'+side+'_hip_roll_cylinder';gid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,name);data.qpos[:]=q;mj.mj_fwdPosition(m,data)
        quat=np.empty(4);mj.mju_mat2Quat(quat,data.geom_xmat[gid]);r,h=m.geom_size[gid,:2]
        test=ET.Element('mujoco',model='native_source_cylinder_support_probe');ET.SubElement(test,'option',timestep='.001',integrator='implicitfast')
        w=ET.SubElement(test,'worldbody');ET.SubElement(w,'geom',type='plane',size='1 1 .02',friction='.8 .01 .001')
        b=ET.SubElement(w,'body',pos='0 0 .12',quat=fmt(quat));ET.SubElement(b,'freejoint');ET.SubElement(b,'geom',type='cylinder',size=fmt([r,h]),mass='1',friction='.8 .01 .001')
        tm=mj.MjModel.from_xml_string(ET.tostring(test,encoding='unicode'));td=mj.MjData(tm);maxpen=0.
        for _ in range(1500):
            mj.mj_step(tm,td)
            maxpen=max(maxpen,max([-float(td.contact[i].dist)for i in range(td.ncon)]+[0.]))
        support.append({'geom':name,'duration_s':1.5,'test_body_mass_kg':1.,'max_contact_penetration_m':maxpen,'final_velocity_norm':float(np.linalg.norm(td.qvel)),
                        'final_root_z_m':float(td.qpos[2]),'scope':'Isolated primitive settles under gravity; not evidence of articulated Vita balance.'})
    report={'schema':'rgb_near.source_pelvis_hip_cylinder.v1','model':str(a.output.resolve()),'model_sha256':sha(a.output),'model_mjb_sha256':sha(a.output.with_suffix('.mjb')),
            'source':str(a.source.resolve()),'source_sha256':sha(a.source),'source_urdf':str(a.source_urdf.resolve()),'source_urdf_sha256':sha(a.source_urdf),
            'conversion_code':'scripts/fourth_newft_isaac_plant.py:79','conversion_reason':'Explicitly mirrors recovered Isaac training spawn replace_cylinders_with_capsules=True; not an unintentional mesh importer error.',
            'variant_semantics':'Geometry correction branch restores only source pelvis+four hip primitives per actor; it no longer exactly matches the training capsule plant.',
            'source_cylinder_inventory':inventory,'modified_geoms':changes,'modified_geom_count':10,'all_other_compiled_contract_arrays_unchanged':True,
            'nq':m.nq,'nv':m.nv,'nu':m.nu,'neq':m.neq,'seat_top_m':seat_top,'initial_cylinder_seat_gaps':initial,'single_frame_leg_reachability':feasibility,'isolated_cylinder_support_tests':support,
            'limitations':['No frozen original XML/MJB or trial is overwritten.','The initial reference still relies on old capsule extent. New geometry needs seated root/leg/hand/shoe contact initialization before articulated dynamics.',
                          'Leg feasibility preserves initial ankle pose for each side independently within joint limits; simultaneous seat/feet/shoe/hand/body collision and full-sequence constraints are not solved here.']}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'model':str(a.output),'sha256':sha(a.output),'changed':10,'source_cylinders_per_actor':len(inventory)//2,'max_initial_leg_position_error_m':max(x['ankle_position_error_m']for x in feasibility),'support':support}))
if __name__=='__main__':main()
