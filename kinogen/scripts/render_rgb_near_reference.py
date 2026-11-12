"""Render a retargeted reference for visual inspection, without advancing physics."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shoe-proxy-overlay',action='store_true')
    a = p.parse_args()
    import numpy as np
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    model=mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    data=mujoco.MjData(model)
    ref=np.load(a.reference,allow_pickle=False)
    model.vis.global_.offwidth=800;model.vis.global_.offheight=720
    model.vis.headlight.ambient[:]=[.5,.5,.5]
    names=list(map(str,ref['dof_names']))
    maps=[]
    for actor in (1,2):
        prefix=f'p{actor}_'
        root=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,prefix+'vita_demo_free_root')
        joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,prefix+n)for n in names]
        if root<0 or min(joints)<0:raise ValueError('Missing actor joints')
        maps.append((int(model.jnt_qposadr[root]),model.jnt_qposadr[joints]))
    shoe=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'shoe_free')
    if shoe<0:raise ValueError('Missing shoe')
    oq=int(model.jnt_qposadr[shoe]);sb=int(model.jnt_bodyid[shoe])
    if a.shoe_proxy_overlay:
        for g in range(model.ngeom):
            if model.geom_bodyid[g]==sb and (model.geom_contype[g] or model.geom_conaffinity[g]):
                model.geom_group[g]=2;model.geom_matid[g]=-1;model.geom_rgba[g]=[1.,.46,.08,.4]
    times=ref['time_s']-ref['time_s'][0]
    fps=(len(times)-1)/float(times[-1])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    option=mujoco.MjvOption();option.geomgroup[:]=[0,1,1,1,1,1]
    cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
    a.output.parent.mkdir(parents=True,exist_ok=True)
    partial=a.output.with_suffix('.partial.mp4')
    enc=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s:v','1600x800','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    snapshots=[];keys=set(np.linspace(0,len(times)-1,9).round().astype(int))
    try:
        with mujoco.Renderer(model,height=720,width=800)as renderer:
            for i,t in enumerate(times):
                for actor,(root,qa)in enumerate(maps):
                    q=ref[f'qpos_p{actor+1}'][i]
                    data.qpos[root:root+7]=q[:7];data.qpos[qa]=q[7:]
                data.qpos[oq:oq+7]=ref['object_target_pose'][i]
                mujoco.mj_forward(model,data)
                canvas=Image.new('RGB',(1600,800),'#15191f')
                for panel in range(2):
                    if panel==0:
                        cam.lookat[:]=[-.45,0.,.65];cam.distance=3.0;cam.azimuth=90.;cam.elevation=-18.
                    else:
                        cam.lookat[:]=data.xpos[sb];cam.distance=.72;cam.azimuth=125.;cam.elevation=-25.
                    renderer.update_scene(data,camera=cam,scene_option=option)
                    canvas.paste(Image.fromarray(renderer.render()),(800*panel,44))
                draw=ImageDraw.Draw(canvas)
                draw.text((12,8),'KINEMATIC REFERENCE | NOT A PHYSICS ROLLOUT | O6 segmented hands',font=font,fill='#ffce69')
                if a.shoe_proxy_overlay:
                    draw.text((812,52),'Orange: inferred shoe completion / collision proxy',font=font,fill='#ffce69')
                draw.text((12,770),f't={t:.3f}s | source frame {int(ref["frame_ids"][i])} | inferred fingers / contact-optimized shoe',font=font,fill='white')
                enc.stdin.write(canvas.tobytes())
                if i in keys:snapshots.append(canvas.resize((640,320)))
        enc.stdin.close()
        error=enc.stderr.read().decode(errors='replace')
        if enc.wait()!=0:raise RuntimeError(error)
    except BaseException:
        enc.kill();enc.wait();raise
    partial.replace(a.output)
    sheet=Image.new('RGB',(1920,960))
    for i,snap in enumerate(snapshots):sheet.paste(snap,((i%3)*640,(i//3)*320))
    contact=a.output.with_name(a.output.stem+'_contactsheet.jpg');sheet.save(contact,quality=92)
    out={'mode':'kinematic_reference_only','simulation_steps':0,'source_frames':len(times),
         'duration_s':float(times[-1]),'fps':fps,'model':str(a.model),'model_sha256':sha(a.model),
         'reference':str(a.reference),'reference_sha256':sha(a.reference),
         'video_sha256':sha(a.output),'contactsheet':str(contact),'contactsheet_sha256':sha(contact),
         'shoe_proxy_overlay':{'enabled':a.shoe_proxy_overlay,'meaning':'Inferred collision geometry; raw mesh retained; rendering-only change'}}
    a.output.with_suffix('.render.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
