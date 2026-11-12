"""Clearly labelled kinematic snapshots; no physics steps or force claims."""
import argparse,json,hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import mujoco as mj,numpy as np
    from PIL import Image,ImageDraw,ImageFont
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);z=np.load(a.reference,allow_pickle=False)
    m.vis.global_.offwidth=800;m.vis.global_.offheight=480;m.vis.headlight.ambient[:]=[.5,.5,.5]
    cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE;opt=mj.MjvOption();opt.geomgroup[:]=[0,1,1,1,1,1]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19);canvas=Image.new('RGB',(1600,1560),'#15191f');draw=ImageDraw.Draw(canvas)
    frames=[0,110,150,200,290,len(z['time_s'])-1]
    with mj.Renderer(m,height=480,width=800)as renderer:
        for k,f in enumerate(frames):
            d.qpos[:]=z['qpos'][f];mj.mj_forward(m,d)
            cam.lookat[:]=d.qpos[:3]+[0,0,.1];cam.distance=3.1;cam.azimuth=70.;cam.elevation=-17.
            renderer.update_scene(d,camera=cam,scene_option=opt);x=(k%2)*800;y=(k//2)*520;canvas.paste(Image.fromarray(renderer.render()),(x,y+40))
            draw.text((x+8,y+3),f'KINEMATIC / NOT PHYSICS | f{f} t={z["time_s"][f]:.2f}s',font=font,fill='#ffce69')
    a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output,quality=94)
    print(json.dumps({'mode':'kinematic_only','frames':frames,'output':str(a.output),'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest()}))


if __name__=='__main__':main()
