"""Render local candidate hand+shoe snapshots, explicitly without dynamics/arm IK."""
from pathlib import Path
import argparse,json,xml.etree.ElementTree as ET
def fmt(v):return' '.join(format(float(x),'.12g')for x in v)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',type=Path,required=True);a=p.parse_args()
    import numpy as np,mujoco as mj
    from scipy.spatial import ConvexHull
    from PIL import Image,ImageDraw,ImageFont
    r=json.loads(a.candidate.read_text());b=r['best'];c=json.loads((a.candidate.parent/'keyframe_contract.json').read_text());m=mj.MjModel.from_xml_path(r['model']);d=mj.MjData(m);ref=np.load(r['reference'],allow_pickle=False);f=r['frame']
    q0=np.r_[ref['qpos'][f,0],ref['qpos'][f,1],ref['object_target_pose'][f]];d.qpos[:]=q0;mj.mj_forward(m,d)
    hb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'p1_rh_hand_base_link');sb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'shoe');P0=d.xpos[hb].copy();R0=d.xmat[hb].reshape(3,3).copy();P=np.asarray(b['hand_position_world_m']);R=np.asarray(b['hand_rotation_world']);up=np.asarray(c['shoe_up_world'])
    hand=[g for g in range(m.ngeom)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith('p1_rh_')];shoe=[g for g in range(m.ngeom)if m.geom_bodyid[g]==sb];ids=hand+shoe
    qa=[int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)])for n in r['driver_names']];target=np.asarray(b['driver_angles_rad'])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);canvas=Image.new('RGB',(1800,1040),'#171e27');draw=ImageDraw.Draw(canvas)
    for row,(theta,offset,label)in enumerate([(np.zeros(6),up*.08,'Open pregrasp: shoeUp +80mm'),(target,np.zeros(3),'Closed candidate: gaps0.73/2.08mm')]):
        d.qpos[:]=q0;d.qpos[qa]=theta
        for e in range(m.neq):
            if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:
                j1,j2=int(m.eq_obj1id[e]),int(m.eq_obj2id[e]);d.qpos[m.jnt_qposadr[j1]]=np.polynomial.polynomial.polyval(d.qpos[m.jnt_qposadr[j2]],m.eq_data[e,:5])
        mj.mj_kinematics(m,d);delta=R@R0.T
        root=ET.Element('mujoco',model='static_grasp_geometry_review');visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='600',offheight='460');ET.SubElement(visual,'headlight',ambient='.65 .65 .65',diffuse='.7 .7 .7');ET.SubElement(root,'statistic',extent='.4')
        assets=ET.SubElement(root,'asset');world=ET.SubElement(root,'worldbody');centres=[]
        for g in ids:
            active=bool(m.geom_contype[g]);is_hand=g in hand;pos=d.geom_xpos[g].copy();rot=d.geom_xmat[g].reshape(3,3).copy()
            if is_hand:pos=delta@(pos-P0)+P+offset;rot=delta@rot
            quat=np.empty(4);mj.mju_mat2Quat(quat,rot.ravel());name=m.geom(g).name or'source_visual_'+str(g)
            at={'name':name,'pos':fmt(pos),'quat':fmt(quat),'contype':'0','conaffinity':'0','group':str(0 if is_hand and active else 1 if is_hand else 2 if active else 3),
                'rgba':'.1 .75 1 .6'if is_hand and active else'.78 .8 .82 1'if is_hand else'1 .48 .08 .38'if active else'.16 .17 .18 1'}
            if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
                mid=int(m.geom_dataid[g]);vi=int(m.mesh_vertadr[mid]);vn=int(m.mesh_vertnum[mid]);fi=int(m.mesh_faceadr[mid]);fn=int(m.mesh_facenum[mid]);v=m.mesh_vert[vi:vi+vn];faces=m.mesh_face[fi:fi+fn].copy()
                if active:
                    hull=ConvexHull(v);faces=hull.simplices.copy();normal=np.cross(v[faces[:,1]]-v[faces[:,0]],v[faces[:,2]]-v[faces[:,0]]);flip=np.einsum('ij,ij->i',normal,hull.equations[:,:3])<0;faces[flip]=faces[flip][:,[0,2,1]]
                mn='mesh_'+str(g);ET.SubElement(assets,'mesh',name=mn,vertex=fmt(v.ravel()),face=' '.join(str(int(i))for i in faces.ravel()));at.update(type='mesh',mesh=mn)
            else:at.update(type='box',size=fmt(m.geom_size[g]))
            ET.SubElement(world,'geom',**at);centres.append(pos)
        if row==1:
            for i,contact in enumerate(b['contact_evidence']):
                witness=np.asarray(contact['qp_witness_world'])
                ET.SubElement(world,'geom',name='contact_'+str(i),type='sphere',size='.0025',pos=fmt(witness[3:]),rgba='1 .08 .08 1',contype='0',conaffinity='0',group='4')
        display=mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'));state=mj.MjData(display);mj.mj_forward(display,state);centre=(P+offset+np.mean(np.asarray([d.geom_xpos[g]for g in shoe]),axis=0))/2
        with mj.Renderer(display,height=460,width=600)as renderer:
            for col,angle in enumerate([0.,1.5,3.1]):
                eye=centre+.55*(np.cos(angle)*R[:,0]+np.sin(angle)*R[:,1])+.07*R[:,2];forward=centre-eye;forward/=np.linalg.norm(forward)
                cam=mj.MjvCamera();cam.lookat[:]=centre;cam.distance=np.linalg.norm(centre-eye);cam.azimuth=np.degrees(np.arctan2(forward[1],forward[0]));cam.elevation=np.degrees(np.arcsin(forward[2]))
                opt=mj.MjvOption();opt.geomgroup[:]=[col==2,col!=2,1,1,1,0];renderer.update_scene(state,camera=cam,scene_option=opt)
                for sc in renderer.scene.camera:sc.up[:]=R[:,2]
                pic=Image.fromarray(renderer.render());canvas.paste(pic,(col*600,60+row*520));draw.text((col*600+10,row*520+5),label,font=font,fill='white');draw.text((col*600+10,row*520+31),'STATIC GEOMETRY | orange=inferred shoe proxy',font=font,fill='#ffb96b')
    path=a.candidate.parent/'grasp_candidate_local_review.jpg';canvas.save(path,quality=94);print(path)
if __name__=='__main__':main()
