"""Render clearly labelled kinematic reference keyframes, never physics."""
from pathlib import Path
import argparse
import hashlib
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--reference',default='outputs/rgb_near_pipeline/reference_v16_O6_grasp.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v4_self.xml')
    p.add_argument('--output-prefix',default='previews/rgb_near_v16_reference')
    p.add_argument('--frames',type=int,nargs='+',default=[260,330])
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from PIL import Image,ImageDraw,ImageFont
    z=np.load(a.root/a.reference,allow_pickle=False);m=mj.MjModel.from_xml_path(str(a.root/a.scene));d=mj.MjData(m)
    maps=[]
    for pre in('p1_','p2_'):
        rootj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root')
        js=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+str(n))for n in z['dof_names']]
        if min([rootj,*js])<0:raise ValueError('Joint ABI mismatch')
        maps.append((int(m.jnt_qposadr[rootj]),m.jnt_qposadr[js]))
    sj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free');sq=int(m.jnt_qposadr[sj])
    m.vis.global_.offwidth=max(640,m.vis.global_.offwidth);m.vis.global_.offheight=max(704,m.vis.global_.offheight)
    m.vis.headlight.ambient=[.45,.45,.45]
    option=mj.MjvOption();option.geomgroup[:]=[0,1,1,1,1,1]
    cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    out=a.root/a.output_prefix;out.parent.mkdir(parents=True,exist_ok=True);files=[]
    with mj.Renderer(m,height=704,width=640)as renderer:
        for f in a.frames:
            for actor,(rq,qa)in enumerate(maps):d.qpos[rq:rq+7]=z['qpos'][f,actor,:7];d.qpos[qa]=z['qpos'][f,actor,7:]
            d.qpos[sq:sq+7]=z['object_target_pose'][f];mj.mj_forward(m,d)
            canvas=Image.new('RGB',(1280,800),'#15191f')
            for panel,lookat,dist,az,el in[(0,[-.45,-.05,.67],3.,90.,-12.),(1,z['object_target_pose'][f,:3],.53,125.,-20.)]:
                cam.lookat[:]=lookat;cam.distance=dist;cam.azimuth=az;cam.elevation=el
                renderer.update_scene(d,camera=cam,scene_option=option);canvas.paste(Image.fromarray(renderer.render()),(panel*640,56))
            draw=ImageDraw.Draw(canvas);draw.text((12,5),'KINEMATIC REFERENCE ONLY | NOT A PHYSICS ROLLOUT',font=font,fill='#ffcd64')
            draw.text((12,30),'O6 six drivers + five calibrated polynomial mimics | inferred soft-assisted grip',font=font,fill='white')
            text=f"frame {f}/397 | source t={z['time_s'][f]:.3f}s"
            if 'O6_driver_command_u8'in z.files:text+=' | right O6 commands '+str(z['O6_driver_command_u8'][f].tolist())
            draw.text((12,778),text,font=font,fill='#b9c8db')
            path=out.with_name(out.name+f'_frame{f:03d}.png');canvas.save(path);files.append(str(path))
    report={'mode':'kinematic_reference_not_physics','frames':a.frames,'files':files,
            'reference_sha256':hashlib.sha256((a.root/a.reference).read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256((a.root/a.scene).read_bytes()).hexdigest()}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
