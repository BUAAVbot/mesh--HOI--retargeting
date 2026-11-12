"""Render recorded robot physics alongside the matching original RGB frames."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-video',type=Path,default=Path('data/rgb_145317_sim_handoff/input.mp4'))
    p.add_argument('--fps',type=float,default=30.)
    p.add_argument('--right-view',choices=['source','feet','ego'],default='source')
    p.add_argument('--seconds',type=float)
    p.add_argument('--visual-model',type=Path)
    p.add_argument('--title',default='RGB145',help='Visible sequence label; does not change the acceptance badge')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    r=json.loads((a.trial/'report.json').read_text())
    mp=a.trial/'model.mjb';tp=a.trial/'trace.npz'
    if sha(mp)!=r['model']['sha256']or sha(tp)!=r['trace']['sha256']:raise ValueError('Frozen model/trace hash mismatch')
    z=np.load(tp,allow_pickle=False);m=mj.MjModel.from_binary_path(str(mp));d=mj.MjData(m)
    if a.visual_model:
        visual=mj.MjModel.from_xml_path(str(a.visual_model.resolve()))
        for key in ('body_pos','body_quat','body_mass','jnt_pos','jnt_axis','jnt_qposadr','jnt_dofadr'):
            if key=='body_quat':
                if np.min(abs(np.sum(m.body_quat*visual.body_quat,axis=1)))<1.-1e-11:raise ValueError('Visual model body rotation mismatch')
                continue
            if not np.allclose(getattr(m,key),getattr(visual,key),atol=2e-7,rtol=1e-7):raise ValueError('Visual model articulation mismatch: '+key)
        m=visual;d=mj.MjData(m)
    m.vis.global_.offwidth=800;m.vis.global_.offheight=720;m.vis.headlight.ambient[:]=[.5,.5,.5]
    opt=mj.MjvOption();opt.geomgroup[:]=[0,1,1,1,1,1]
    cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE
    rq=int(z['robot_root_qadr'][0]);times=z['time'];source_times=z['source_time_s']
    render_end=times[-1]if a.seconds is None else min(times[-1],a.seconds)
    sample=np.arange(times[0],render_end+1e-8,1/a.fps)
    ids=np.clip(np.searchsorted(times,sample,side='right')-1,0,len(times)-1)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    a.output.parent.mkdir(parents=True,exist_ok=True);partial=a.output.with_suffix('.partial.mp4')
    enc=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s:v','1600x800','-r',str(a.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    decoder=None;current=-1;original=None
    if a.right_view=='source':
        info=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate','-of','json',str(a.source_video)],capture_output=True,text=True,check=True).stdout)['streams'][0]
        width,height=int(info['width']),int(info['height']);num,den=map(float,info['r_frame_rate'].split('/'));source_fps=num/den
        decoder=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(a.source_video),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    ankle=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n+'_ankle_roll_link')for n in ('left','right')]
    head=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'neck_yaw_link');ego_poses=[]
    snaps=[];keys=set(np.linspace(0,len(ids)-1,min(9,len(ids))).round().astype(int))
    try:
        with mj.Renderer(m,height=720,width=800)as renderer:
            for f,i in enumerate(ids):
                d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];k=max(0,i-1)
                if 'eq_data'in z:m.eq_data[:]=z['eq_data'][k]
                if 'eq_active'in z:d.eq_active[:]=z['eq_active'][k]
                mj.mj_forward(m,d)
                canvas=Image.new('RGB',(1600,800),'#15191f')
                cam.lookat[:]=d.qpos[rq:rq+3]+[0,0,.15];cam.distance=3.2;cam.azimuth=70.;cam.elevation=-15.
                renderer.update_scene(d,camera=cam,scene_option=opt);canvas.paste(Image.fromarray(renderer.render()),(0,56))
                if a.right_view=='source':
                    target=int(np.clip(np.rint(source_times[i]*source_fps),0,int(z['source_frame_ids'][-1])))
                    while current<target:
                        raw=decoder.stdout.read(width*height*3)
                        if len(raw)!=width*height*3:raise ValueError('Source video ended before requested frame')
                        original=Image.fromarray(np.frombuffer(raw,dtype=np.uint8).reshape(height,width,3));current+=1
                    canvas.paste(ImageOps.pad(original,(800,720),color='#15191f'),(800,56))
                elif a.right_view=='feet':
                    if min(ankle)<0:raise ValueError('Missing foot bodies')
                    cam.lookat[:]=np.mean(d.xpos[ankle],axis=0);cam.distance=1.15;cam.azimuth=80.;cam.elevation=-20.
                    renderer.update_scene(d,camera=cam,scene_option=opt);canvas.paste(Image.fromarray(renderer.render()),(800,56))
                else:
                    if head<0:raise ValueError('Missing head mount body')
                    R=d.xmat[head].reshape(3,3);origin=d.xpos[head]+R@np.array([.16,0,.12])
                    renderer.update_scene(d,camera=cam,scene_option=opt)
                    for gl in renderer.scene.camera:
                        gl.pos[:]=origin;gl.forward[:]=R[:,0];gl.up[:]=R[:,2]
                    canvas.paste(Image.fromarray(renderer.render()),(800,56))
                    T=np.eye(4);T[:3,:3]=np.c_[-R[:,1],-R[:,2],R[:,0]];T[:3,3]=origin;ego_poses.append(T)
                draw=ImageDraw.Draw(canvas);badge='ACCEPTED'if r['simulation_passed']else'DIAGNOSTIC / NOT ACCEPTED'
                draw.text((12,5),a.title+' | VITA + O6 | RECORDED PHYSICS | '+badge,font=font,fill='white')
                draw.text((12,29),'Estimated scene geometry / physical parameters | '+r['status'],font=font,fill='#ffce69')
                label={'source':'ORIGINAL RGB (source clock)','feet':'ACTUAL FEET / TERRAIN CONTACT','ego':'SYNTHETIC HEAD RGB | ASSUMED CAMERA MOUNT'}[a.right_view]
                draw.text((812,60),label,font=font,fill='#ffce69')
                footer=f'execution {times[i]:.3f}s | source {source_times[i]:.3f}s'
                if i>0:footer+=f' | root XY {z["root_horizontal_error_m"][k]*100:.1f}cm | tilt error {z["root_orientation_error_rad"][k]:.2f}rad'
                draw.text((12,778),footer,font=font,fill='#ffce69');enc.stdin.write(canvas.tobytes())
                if f in keys:snaps.append(canvas.resize((640,320)))
        enc.stdin.close();error=enc.stderr.read().decode(errors='replace')
        if enc.wait()!=0:raise RuntimeError(error)
        partial.replace(a.output)
    finally:
        if enc.poll()is None:enc.kill();enc.wait()
        if decoder is not None:decoder.stdout.close();decoder.terminate();decoder.wait()
    sheet=Image.new('RGB',(1920,((len(snaps)+2)//3)*320))
    for j,im in enumerate(snaps):sheet.paste(im,((j%3)*640,(j//3)*320))
    contact=a.output.with_name(a.output.stem+'_contactsheet.jpg');sheet.save(contact,quality=92)
    out={'mode':'actual_recorded_physics','trial_status':r['status'],'model_sha256':sha(mp),'trace_sha256':sha(tp),'frames':len(ids),'fps':a.fps,'physics_duration_s':float(times[-1]),'source_time_covered_s':float(source_times[-1]),'source_frames':int(r['source_frames']),'video_sha256':sha(a.output),'contactsheet_sha256':sha(contact),'right_view':a.right_view,'source_video_sha256':sha(a.source_video)if a.right_view=='source'else None}
    out['rendered_execution_range_s']=[float(sample[0]),float(sample[-1])]
    out['visual_model_sha256']=sha(a.visual_model)if a.visual_model else None
    if ego_poses:
        camera_path=a.output.with_suffix('.cameras.npz');fy=360./np.tan(np.radians(m.vis.global_.fovy)/2)
        np.savez_compressed(camera_path,execution_time_s=times[ids],source_time_s=source_times[ids],world_from_camera_cv=np.asarray(ego_poses),intrinsics=np.array([[fy,0,400.],[0,fy,360.],[0,0,1.]]),resolution_wh=np.array([800,720]),mount_body=np.asarray('neck_yaw_link'),mount_offset_body_m=np.array([.16,0,.12]),assumed_synthetic_camera=np.asarray(True))
        out['synthetic_camera_sha256']=sha(camera_path)
    a.output.with_suffix('.render.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)


if __name__=='__main__':main()
