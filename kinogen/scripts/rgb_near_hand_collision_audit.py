"""Audit effective hand collision geometry and render open/DIP-only overlays."""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-model',type=Path,required=True);p.add_argument('--old-xml',type=Path,required=True)
    p.add_argument('--new-model',type=Path,required=True);p.add_argument('--self-model',type=Path)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--coupled-motion',action='store_true',help='Use driver MCP curl and project joint equality polynomials; diagnostic O6 six-command motion')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.spatial import ConvexHull
    from PIL import Image,ImageDraw,ImageFont
    def load(path):return mj.MjModel.from_binary_path(str(path))if path.suffix=='.mjb'else mj.MjModel.from_xml_path(str(path))
    old=load(a.old_model);new=load(a.new_model);ref=np.load(a.reference,allow_pickle=False)
    records={};drawings={};prefixes=['p1_lh_','p1_rh_','p2_lh_','p2_rh_']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    def init(m):
        d=mj.MjData(m);d.qpos[:59]=ref['qpos'][0,0];d.qpos[59:118]=ref['qpos'][0,1];d.qpos[118:125]=ref['object_target_pose'][0]
        for j in range(m.njnt):
            n=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)or''
            if any(n.startswith(x)for x in prefixes):d.qpos[m.jnt_qposadr[j]]=m.jnt_range[j,0]
        mj.mj_fwdPosition(m,d);return d
    def project(m,q):
        for e in range(m.neq):
            if m.eq_type[e]!=mj.mjtEq.mjEQ_JOINT:continue
            j1,j2=int(m.eq_obj1id[e]),int(m.eq_obj2id[e])
            q[m.jnt_qposadr[j1]]=np.polynomial.polynomial.polyval(q[m.jnt_qposadr[j2]],m.eq_data[e,:5])
    def snapshot_model(m,d,ids,active):
        # Render explicit convex hull triangles for active collision meshes.
        # A global mjVIS_CONVEXHULL flag would also hull the visual meshes and
        # conceal the mismatch, so visual and collision geometry are separate.
        root=ET.Element('mujoco',model='hand_collision_snapshot')
        visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='640',offheight='480')
        ET.SubElement(visual,'headlight',ambient='.6 .6 .6',diffuse='.7 .7 .7')
        ET.SubElement(root,'statistic',extent='.4')
        assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody')
        def fmt(arr):return' '.join(format(float(x),'.12g')for x in np.asarray(arr).ravel())
        for g in ids:
            collision=g in active;kind=int(m.geom_type[g]);quat=np.empty(4);mj.mju_mat2Quat(quat,d.geom_xmat[g])
            at={'name':'snapshot_'+str(g),'pos':fmt(d.geom_xpos[g]),'quat':fmt(quat),
                'contype':'0','conaffinity':'0','group':'0'if collision else'1',
                'rgba':'.1 .75 1 1'if collision else'.7 .72 .74 1'}
            if kind==mj.mjtGeom.mjGEOM_MESH:
                mid=int(m.geom_dataid[g]);vs=int(m.mesh_vertadr[mid]);vn=int(m.mesh_vertnum[mid]);v=m.mesh_vert[vs:vs+vn].copy()
                if collision:
                    hull=ConvexHull(v);faces=hull.simplices.copy()
                    normals=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]])
                    flip=np.einsum('ij,ij->i',normals,hull.equations[:,:3])<0
                    faces[flip]=faces[flip][:,[0,2,1]]
                else:
                    fs=int(m.mesh_faceadr[mid]);fn=int(m.mesh_facenum[mid]);faces=m.mesh_face[fs:fs+fn].copy()
                name='snapshot_mesh_'+str(g)
                ET.SubElement(assets,'mesh',name=name,vertex=fmt(v),face=' '.join(str(int(x))for x in faces.ravel()))
                at.update(type='mesh',mesh=name)
            elif kind==mj.mjtGeom.mjGEOM_CAPSULE:at.update(type='capsule',size=fmt(m.geom_size[g,:2]))
            elif kind==mj.mjtGeom.mjGEOM_SPHERE:at.update(type='sphere',size=fmt(m.geom_size[g,:1]))
            else:raise ValueError('Unexpected hand primitive '+str(kind))
            ET.SubElement(world,'geom',**at)
        result=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));state=mj.MjData(result);mj.mj_forward(result,state)
        return result,state
    def record(m,xmlpath):
        x=ET.parse(xmlpath).getroot();assets={z.get('name'):z.attrib for z in x.find('asset')if z.tag=='mesh'}
        srcgeoms={g.get('name'):g for g in x.findall('.//geom')if g.get('name')}
        rows=[]
        for g in range(m.ngeom):
            b=int(m.geom_bodyid[g]);name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b)or''
            if not(any(name.startswith(x)for x in prefixes)or('wrist_'in name)):continue
            gn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or''
            mid=int(m.geom_dataid[g])if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH else-1
            mn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_MESH,mid)if mid>=0 else None
            jrows=[]
            for j in range(m.njnt):
                if m.jnt_bodyid[j]!=b:continue
                v=int(m.jnt_dofadr[j]);aids=np.flatnonzero(m.actuator_trnid[:,0]==j)
                jrows.append({'joint':mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j),'armature_kg_m2':float(m.dof_armature[v]),
                              'damping':float(m.dof_damping[v]),'actuators':[{'name':mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,int(i)),
                              'kp':float(m.actuator_gainprm[i,0]),'kv':float(-m.actuator_biasprm[i,2]),'force_range':m.actuator_forcerange[i].tolist()}for i in aids]})
            rows.append({'geom':gn,'body':name,'active':bool(m.geom_contype[g]or m.geom_conaffinity[g]),
                         'type':int(m.geom_type[g]),'size':m.geom_size[g].tolist(),'local_pos_compiled':m.geom_pos[g].tolist(),
                         'local_quat_compiled':m.geom_quat[g].tolist(),'mesh':mn,'source_mesh_asset':assets.get(mn),
                         'contype':int(m.geom_contype[g]),'conaffinity':int(m.geom_conaffinity[g]),
                         'friction':m.geom_friction[g].tolist(),'solref':m.geom_solref[g].tolist(),'solimp':m.geom_solimp[g].tolist(),
                         'margin':float(m.geom_margin[g]),'condim':int(m.geom_condim[g]),'body_mass_kg':float(m.body_mass[b]),
                         'body_principal_inertia':m.body_inertia[b].tolist(),'joint_drive':jrows})
        return rows
    for label,m,path in [('old_effective',old,a.old_xml),('new_perlink',new,a.new_model)]:
        d=init(m);bn=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,i)or'world'for i in range(m.nbody)]
        hand=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link')
        ids=[g for g in range(m.ngeom)if bn[int(m.geom_bodyid[g])].startswith('p1_rh_')]
        active=[g for g in ids if m.geom_contype[g]or m.geom_conaffinity[g]]
        visual=[g for g in ids if not(m.geom_contype[g]or m.geom_conaffinity[g])]
        q0=d.qpos.copy();beforeP=d.geom_xpos.copy();beforeR=d.geom_xmat.copy()
        q1=q0.copy();dip=[]
        for digit in ['index','middle','ring','pinky']:
            j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_rh_'+digit+'_dip');qa=int(m.jnt_qposadr[j]);q1[qa]=min(float(m.jnt_range[j,1]),q0[qa]+.8);dip.append(int(j))
        if a.coupled_motion:
            for digit in ['index','middle','ring','pinky']:
                j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_rh_'+digit+'_mcp_pitch');q1[m.jnt_qposadr[j]]=.65
            for suffix,val in [('thumb_cmc_pitch',.25),('thumb_cmc_yaw',.35)]:
                j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_rh_'+suffix);q1[m.jnt_qposadr[j]]=val
            project(m,q1)
        d.qpos[:]=q1;mj.mj_fwdPosition(m,d)
        posdelta=float(np.max(abs(d.geom_xpos[active]-beforeP[active])));rotdelta=float(np.max(abs(d.geom_xmat[active]-beforeR[active])))
        vdelta=float(np.max(abs(d.geom_xpos[visual]-beforeP[visual])))
        info={'hand_geoms':record(m,path),'active_geoms_per_hand':{x:sum(bool(m.geom_contype[g]or m.geom_conaffinity[g])for g in range(m.ngeom)if bn[int(m.geom_bodyid[g])].startswith(x))for x in prefixes},
              'dip_perturbation_rad':None if a.coupled_motion else .8,'motion_description':'MCP .65rad, thumb pitch .25/yaw .35, passive equality projection'if a.coupled_motion else'DIP only +.8rad',
              'active_collision_max_position_change_m':posdelta,'active_collision_max_rotation_change':rotdelta,
              'visual_max_position_change_m':vdelta,'distal_bodies_without_active_collision':[],
              'interfinger_clearance_samples':[]}
        for x in prefixes:
            for digit in ['index','middle','ring','pinky']:
                b=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,x+digit+'_distal')
                if not any(m.geom_bodyid[g]==b and(m.geom_contype[g]or m.geom_conaffinity[g])for g in range(m.ngeom)):
                    info['distal_bodies_without_active_collision'].append(x+digit+'_distal')
        for pose,qp in [('open',q0),('coupled_curl'if a.coupled_motion else'dip_only_0p8',q1)]:
            d.qpos[:]=qp;mj.mj_fwdPosition(m,d)
            for d1,d2 in [('index','middle'),('middle','ring'),('ring','pinky')]:
                g1=[g for g in active if bn[int(m.geom_bodyid[g])].startswith('p1_rh_'+d1)]
                g2=[g for g in active if bn[int(m.geom_bodyid[g])].startswith('p1_rh_'+d2)]
                gap=min(float(mj.mj_geomDistance(m,d,g,h,.2,np.zeros(6)))for g in g1 for h in g2)
                distal1=[g for g in g1 if bn[int(m.geom_bodyid[g])].endswith('_distal')]
                distal2=[g for g in g2 if bn[int(m.geom_bodyid[g])].endswith('_distal')]
                dg=min(float(mj.mj_geomDistance(m,d,g,h,.2,np.zeros(6)))for g in distal1 for h in distal2)if distal1 and distal2 else None
                info['interfinger_clearance_samples'].append({'pose':pose,'pair':d1+'_'+d2,'min_all_link_distance_m':gap,'distal_link_distance_m':dg})
        records[label]=info
        # Renderer-only snapshot models preserve the actual source triangles
        # while rendering explicit collision hulls from compiled vertices.
        d.qpos[:]=q0;mj.mj_fwdPosition(m,d);R=d.xmat[hand].reshape(3,3).copy()
        centre=d.xpos[hand]+R@np.asarray([0,0,.095]);eye=centre+R@np.asarray([.29,.20,0.])
        forward=centre-eye;forward/=np.linalg.norm(forward)
        camera=mj.MjvCamera();camera.lookat[:]=centre;camera.distance=np.linalg.norm(centre-eye)
        camera.azimuth=np.degrees(np.arctan2(forward[1],forward[0]));camera.elevation=np.degrees(np.arcsin(forward[2]))
        canvas=Image.new('RGB',(1920,1080),'#141b23');draw=ImageDraw.Draw(canvas);overlay=[]
        option=mj.MjvOption()
        for row,qp in enumerate([q0,q1]):
            d.qpos[:]=qp;mj.mj_fwdPosition(m,d)
            display,display_data=snapshot_model(m,d,ids,active)
            with mj.Renderer(display,height=480,width=640)as renderer:
                for col,text in enumerate(['Visual mesh','Active collision','Overlay: collision cyan']):
                    option.geomgroup[:]=[col!=0,col!=1,0,0,0,0]
                    for g in range(display.ngeom):
                        collision=display.geom_group[g]==0
                        display.geom_rgba[g]=[.1,.75,1.,.65 if col==2 else 1.]if collision else[.7,.72,.74,.22 if col==2 else 1.]
                    renderer.update_scene(display_data,camera=camera,scene_option=option)
                    for view in renderer.scene.camera:view.up[:]=R[:,2]
                    pic=Image.fromarray(renderer.render());canvas.paste(pic,(640*col,60+540*row))
                    draw.text((640*col+12,540*row+7),label+' | '+text,font=font,fill='white')
                    draw.text((640*col+12,540*row+32),'Open fingers'if row==0 else('O6 driver curl + passive polynomial'if a.coupled_motion else'DIP only +0.8 rad; MCP unchanged'),font=font,fill='#ffcb60')
                    if col==2:overlay.append(pic.copy())
        out=a.output_dir/(label+'_open_dip_overlay.jpg');canvas.save(out,quality=94);drawings[label]=overlay
    compare=Image.new('RGB',(1280,1080),'#141b23');draw=ImageDraw.Draw(compare)
    for row,label in enumerate(['old_effective','new_perlink']):
        for col,pic in enumerate(drawings[label]):
            compare.paste(pic,(640*col,60+540*row));draw.text((640*col+12,540*row+8),label+(' | open'if col==0 else(' | O6 coupled curl'if a.coupled_motion else' | DIP bent')),font=font,fill='white')
            draw.text((640*col+12,540*row+34),'Collision cyan / visual transparent',font=font,fill='#ffcb60')
    compare_path=a.output_dir/'hand_collision_before_after.jpg';compare.save(compare_path,quality=94)
    self_audit=None
    if a.self_model:
        m=load(a.self_model);d=init(m);q0=d.qpos.copy();self_audit={'model':str(a.self_model),'model_sha256':sha(a.self_model),'poses':[]}
        for label,mcp,dip in [('open',0.,0.),('dip_only',0.,.8),('moderate_curl',.45,.8)]:
            d.qpos[:]=q0
            for j in range(m.njnt):
                n=mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,j)or''
                if not any(n.startswith(x)for x in prefixes):continue
                if n.endswith('_dip'):d.qpos[m.jnt_qposadr[j]]=dip
                if n.endswith('_mcp_pitch'):d.qpos[m.jnt_qposadr[j]]=mcp
            if a.coupled_motion:project(m,d.qpos)
            mj.mj_fwdPosition(m,d);rows=[]
            for ci in range(d.ncon):
                c=d.contact[ci];b=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'world'for g in[c.geom1,c.geom2]]
                if any(all(x.startswith(pre)for x in b)for pre in prefixes):
                    rows.append({'bodies':b,'geoms':[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,int(g))for g in[c.geom1,c.geom2]],'distance_m':float(c.dist)})
            self_audit['poses'].append({'pose':label,'within_hand_contact_points':len(rows),'max_penetration_m':max([-v['distance_m']for v in rows]+[0.]),'contacts':rows})
    report={'schema':'rgb_near.hand_collision_audit.v1','old_effective_model':str(a.old_model),'old_sha256':sha(a.old_model),
            'new_model':str(a.new_model),'new_sha256':sha(a.new_model),'diagnostic_only':True,'coupled_motion':a.coupled_motion,'records':records,
            'collision_rendering':'Explicit SciPy convex hull of each compiled MuJoCo collision mesh vertex set; visual meshes remain original source triangles. Primitive capsules rendered directly.',
            'within_hand_self_collision_variant':self_audit,'comparison_image':str(compare_path),'comparison_sha256':sha(compare_path),
            'interpretation':['Comparison models are identified by their exact source paths and hashes; labels old/new do not imply either model has a specific collider type.',
                              'New convex collision assets are per user URDF link with unique left/right names; distal colliders follow DIP.',
                              'Coupled-motion diagnostics project dependent joints with model equality polynomials. A bare zero mj_geomDistance may be an unresolved query and is not evidence of physical contact; consult self-contact rows.',
                              'Default training-compatible model still disables same-robot self contacts, including finger/palm contacts; optional hand-only self model must be evaluated separately.']}
    (a.output_dir/'hand_collision_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'old':{k:v for k,v in records['old_effective'].items()if k!='hand_geoms'},
                      'new':{k:v for k,v in records['new_perlink'].items()if k!='hand_geoms'},
                      'self_audit':None if self_audit is None else[{k:v for k,v in row.items()if k!='contacts'}for row in self_audit['poses']],
                      'image':str(compare_path)}),flush=True)


if __name__=='__main__':main()
