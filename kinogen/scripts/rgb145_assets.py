"""Build a single-Vita O6 stair scene from observed separate terrain patches."""
from pathlib import Path
import argparse,copy,json,hashlib,xml.etree.ElementTree as ET
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fmt(v):return' '.join(format(float(x),'.17g')for x in v)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--support',type=Path,required=True);p.add_argument('--robot-source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    import numpy as np,mujoco as mj
    from PIL import Image
    from scipy.spatial import ConvexHull
    from rgb145_support import height_at_xy
    a.output_dir.mkdir(parents=True,exist_ok=True);scene_path=a.output_dir/'vita_stairs_O6_training.xml';proxy_path=a.output_dir/'vita_inference_O6_training.xml'
    for target in [scene_path,proxy_path,scene_path.with_suffix('.mjb'),proxy_path.with_suffix('.mjb')]:
        if target.exists():raise FileExistsError('Refusing overwrite '+str(target))
    spec=json.loads(a.support.read_text());raw=np.load(a.source/'scene/pi3_mesh.npz');U=np.asarray(spec['U_yup_to_zup']);v=raw['vertices']@U.T;faces=raw['faces'];colors=raw['colors']
    raw_obj=a.output_dir/'raw_scene_zup_facecolor.obj';atlas_path=a.output_dir/'raw_scene_facecolor.png'
    # One constant RGB tile per source triangle avoids UV interpolation across
    # unrelated palette entries. Original geometry/topology are preserved.
    if not raw_obj.exists()or not atlas_path.exists():
        n=int(np.ceil(np.sqrt(len(faces))));size=2*n;rgb=np.rint(colors[faces].mean(1)).astype(np.uint8);pixels=np.zeros((size,size,3),np.uint8)
        ids=np.arange(len(faces));rr=ids//n;cc=ids%n
        for dr in [0,1]:
            for dc in [0,1]:pixels[2*rr+dr,2*cc+dc]=rgb
        Image.fromarray(pixels).save(atlas_path)
        with raw_obj.open('w',encoding='utf-8',newline='\n')as stream:
            for pos in v:stream.write('v '+' '.join(format(float(z),'.9g')for z in pos)+'\n')
            for r0,c0 in zip(rr,cc):stream.write(f'vt {(2*c0+1)/size:.9g} {1-(2*r0+1)/size:.9g}\n')
            for i,face in enumerate(faces):stream.write('f '+' '.join(f'{int(z)+1}/{i+1}'for z in face)+'\n')
    old=mj.MjModel.from_xml_path(str(a.robot_source));root=ET.parse(a.robot_source).getroot();root.set('model','rgb145_single_vita_O6_training_stairs');world=root.find('worldbody');assets=root.find('asset')
    for g in list(world.findall('geom')):world.remove(g)
    changed_passive=[]
    for motor in root.find('actuator'):
        jname=motor.get('joint');jid=mj.mj_name2id(old,mj.mjtObj.mjOBJ_JOINT,jname);aid=mj.mj_name2id(old,mj.mjtObj.mjOBJ_ACTUATOR,motor.get('name'))
        if np.all(old.actuator_gainprm[aid]==0)and np.all(old.actuator_biasprm[aid]==0):
            motor.set('gear','0');motor.set('kp','0');motor.set('kv','0');changed_passive.append(motor.get('name'))
    if len(changed_passive)!=10:raise ValueError('Expected ten unpowered O6 follower ABI channels')
    def plane_quat(b):
        nn=np.asarray([-b[0],-b[1],1.]);nn/=np.linalg.norm(nn);qq=np.r_[1+nn[2],-nn[1],nn[0],0.];return qq/np.linalg.norm(qq)
    ground=spec['ground'];bg=np.asarray(ground['z_eq_ax_by_c']);ET.SubElement(world,'geom',name=ground['name'],type='plane',pos=fmt([0,0,bg[2]]),quat=fmt(plane_quat(bg)),size='0 0 .05',contype='2',conaffinity='1',friction=fmt(ground['friction']),group='0',rgba='.6 .6 .6 1')
    terrain=[]
    for item in spec['supports']:
        xy=np.asarray(item['xy_polygon_m']);beta=np.asarray(item['top_z_eq_ax_by_c']);top=np.c_[xy,xy@beta[:2]+beta[2]];bottom=np.c_[xy,np.full(len(xy),item['bottom_z_m'])];verts=np.r_[top,bottom];hull=ConvexHull(verts);fs=hull.simplices.copy();normal=np.cross(verts[fs[:,1]]-verts[fs[:,0]],verts[fs[:,2]]-verts[fs[:,0]]);flip=np.einsum('ij,ij->i',normal,hull.equations[:,:3])<0;fs[flip]=fs[flip][:,[0,2,1]]
        name=item['name']+'_mesh';ET.SubElement(assets,'mesh',name=name,vertex=fmt(verts.ravel()),face=' '.join(str(int(z))for z in fs.ravel()))
        ET.SubElement(world,'geom',name=item['name'],type='mesh',mesh=name,density='0',contype='2',conaffinity='1',friction=fmt(item['friction']),group='0',rgba='.65 .4 .2 1'if item['kind']=='seat'else'.65 .65 .67 1');terrain.append(item['name'])
    ET.SubElement(assets,'mesh',name='rgb145_raw_scene_visual',file=str(raw_obj.resolve()),maxhullvert='64')
    ET.SubElement(assets,'texture',name='rgb145_raw_face_colors',type='2d',file=str(atlas_path.resolve()))
    ET.SubElement(assets,'material',name='rgb145_raw_material',texture='rgb145_raw_face_colors',specular='0',shininess='0',rgba='1 1 1 1')
    ET.SubElement(world,'geom',name='raw_scene_visual',type='mesh',mesh='rgb145_raw_scene_visual',material='rgb145_raw_material',density='0',contype='0',conaffinity='0',group='1')
    human=np.load(a.source/'human/person_01/root_poses_world.npz');T=human['world_from_pelvis'][0];xy=(U@T[:3,3])[:2];body=world.find("body[@name='base_link']");body.set('pos',fmt([xy[0],xy[1],.9]));body.set('quat',fmt([np.cos(np.pi/4),0,0,np.sin(np.pi/4)]))
    statistic=root.find('statistic')
    if statistic is not None:statistic.attrib.update(center='0 1 .9',extent='3')
    cameras=np.load(a.source/'cameras_world.npz');camout={k:cameras[k].copy()for k in cameras.files}
    if'world_from_camera'in camout:
        camout['world_from_camera'][:,:3,:]=np.einsum('ij,tjk->tik',U,camout['world_from_camera'][:,:3,:]);np.savez_compressed(a.output_dir/'cameras_zup.npz',**camout)
        ct=camout['world_from_camera'][0];rr=ct[:3,:3]@np.diag([1,-1,-1]);quat=np.empty(4);mj.mju_mat2Quat(quat,rr.ravel());K=camout['K'][0];fovy=float(np.degrees(2*np.arctan(720/(2*K[1,1]))));ET.SubElement(world,'camera',name='source_first_frame',pos=fmt(ct[:3,3]),quat=fmt(quat),fovy=str(fovy))
    # Correct only the neutral default seed height using actual training foot
    # collision geometry, not the source person's incompatible root height.
    interim=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));state=mj.MjData(interim);mj.mj_fwdPosition(interim,state);gaps=[]
    for g in range(interim.ngeom):
        bname=mj.mj_id2name(interim,mj.mjtObj.mjOBJ_BODY,int(interim.geom_bodyid[g]))or''
        if not bname.endswith('ankle_roll_link')or not interim.geom_contype[g]:continue
        nn=np.r_[-bg[:2],1.];centre=state.geom_xpos[g];rot=state.geom_xmat[g].reshape(3,3)
        if interim.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
            mid=int(interim.geom_dataid[g]);vi=int(interim.mesh_vertadr[mid]);vn=int(interim.mesh_vertnum[mid]);vv=interim.mesh_vert[vi:vi+vn]@rot.T+centre;gap=float((vv@nn-bg[2]).min())
        elif interim.geom_type[g]==mj.mjtGeom.mjGEOM_CAPSULE:gap=float(centre@nn-bg[2]-interim.geom_size[g,0]*np.linalg.norm(nn)-interim.geom_size[g,1]*abs(rot[:,2]@nn))
        else:raise ValueError('Unexpected training foot collider '+str(interim.geom_type[g]))
        gaps.append(gap)
    z=.9+.002-min(gaps);body.set('pos',fmt([xy[0],xy[1],z]));ET.indent(root);m=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    if(m.nq,m.nv,m.nu,m.neq)!=(59,58,52,10):raise ValueError('State/action/equality ABI changed')
    for key in ['body_mass','body_inertia','body_ipos','body_iquat','jnt_axis','jnt_pos','jnt_range','dof_armature','actuator_gainprm','actuator_biasprm','actuator_forcerange']:
        if not np.array_equal(getattr(old,key),getattr(m,key)):raise ValueError('Canonical robot contract changed: '+key)
    for obj,count in [(mj.mjtObj.mjOBJ_JOINT,m.njnt),(mj.mjtObj.mjOBJ_ACTUATOR,m.nu)]:
        if[mj.mj_id2name(old,obj,i)for i in range(count)]!=[mj.mj_id2name(m,obj,i)for i in range(count)]:raise ValueError('Canonical names/order changed')
    scene_path.write_text(ET.tostring(root,encoding='unicode')+'\n',encoding='utf-8');mj.mj_saveModel(m,str(scene_path.with_suffix('.mjb')))
    inference=copy.deepcopy(root);inference.find('worldbody').remove(inference.find("worldbody/geom[@name='raw_scene_visual']"))
    for item in list(inference.find('asset')):
        if item.get('name','').startswith('rgb145_raw_'):inference.find('asset').remove(item)
    proxy_path.write_text(ET.tostring(inference,encoding='unicode')+'\n',encoding='utf-8');pm=mj.MjModel.from_xml_path(str(proxy_path));mj.mj_saveModel(pm,str(proxy_path.with_suffix('.mjb')))
    for key in ['qpos0','body_mass','body_inertia','actuator_gainprm','actuator_biasprm','actuator_gear','eq_data']:
        if not np.array_equal(getattr(m,key),getattr(pm,key)):raise ValueError('Inference model differs '+key)
    deps=[]
    for item in assets:
        if item.get('file'):
            path=Path(item.get('file'));deps.append({'file':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
    report={'schema':'rgb145.single_vita_O6_training_stair_assets.v1','model':str(scene_path.resolve()),'model_sha256':sha(scene_path),'model_mjb_sha256':sha(scene_path.with_suffix('.mjb')),'inference_model':str(proxy_path.resolve()),'inference_model_sha256':sha(proxy_path),
            'source_robot':str(a.robot_source.resolve()),'source_robot_sha256':sha(a.robot_source),'source_scene_npz_sha256':sha(a.source/'scene/pi3_mesh.npz'),'support_geometry':str(a.support.resolve()),'support_geometry_sha256':sha(a.support),
            'U_yup_to_zup':U.tolist(),'nq':m.nq,'nv':m.nv,'nu':m.nu,'neq':m.neq,'powered_channels':42,'passive_zero_gear_gain_bias_channels':changed_passive,'robot_mass_kg':float(m.body_mass.sum()),
            'training_capsules_and_body_inertia_preserved':True,'within_individual_hand_self_collision_enabled':True,'other_robot_self_collision_enabled':False,'extra_finger_armature_added_here':False,'default_free_root_qpos0':m.qpos0[:7].tolist(),
            'ground':spec['ground'],'terrain_collision_geoms':terrain,'raw_visual_vertices':len(v),'raw_visual_faces':len(faces),'raw_vertex_colors_representation':'Per-triangle mean original RGB, constant2x2 atlas tile; original vertices/faces rotated by U only.','raw_scene_collision_enabled':False,
            'physics_dt_s':float(m.opt.timestep),'dependencies':deps,'limitations':spec['limitations']+['O6 right calibration partial and left mirrored/transferred; no measured finger torque calibration.','Terrain friction is an assumption. Native MuJoCo contact stiffness/defaults are unchanged.','Estimated5e-5 finger rotor should be added once by the physics runtime to12 powered finger channels only; not the10 passive followers.','Neutral default is only a standing seed; the350frame contact-corrected reference and dynamic demo remain separate work.']}
    (a.output_dir/'assets_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in['dependencies','limitations','ground']}),flush=True)
if __name__=='__main__':main()
