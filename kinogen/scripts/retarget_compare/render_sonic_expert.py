"""Render recorded native physics, keeping failed episodes visible to the common end time."""
from __future__ import annotations
import argparse, hashlib, json, math, os, subprocess
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','egl')
os.environ.setdefault('PYOPENGL_PLATFORM','egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID','3')


def encoder(path,w,h):
    path.parent.mkdir(parents=True,exist_ok=True)
    return subprocess.Popen(['ffmpeg','-y','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{w}x{h}','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)


def close(p):
    p.stdin.close();error=p.stderr.read()
    if p.wait():raise RuntimeError(error.decode(errors='replace'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen/outputs/sonic_expert_20261028'))
    p.add_argument('--trial',type=Path);p.add_argument('--tag',default='sonic_all_methods')
    p.add_argument('--title');p.add_argument('--subtitle')
    a=p.parse_args()
    import numpy as np,mujoco as mj
    from PIL import Image,ImageDraw,ImageFont
    if a.trial:trials=[(a.trial.name,a.trial)]
    else:
        results=json.loads((a.root/'sonic_comparison.json').read_text())['results']
        trials=[(row['method'],a.root/'sonic_baseline'/row['method'])for row in results]
    out=a.root/'media';out.mkdir(exist_ok=True)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,23);small=ImageFont.truetype(fontpath,17)
    total_end=6.7
    if a.trial:
        total_end=max(total_end,json.loads((a.trial/'report.json').read_text())['simulated_seconds'])
    sample=np.arange(0,total_end+1e-8,1/30)
    videos=[]
    for label,trial in trials:
        trace=trial/'trace.npz';report=json.loads((trial/'report.json').read_text())
        trace_sha=hashlib.sha256(trace.read_bytes()).hexdigest()
        video=out/(label+'.mp4');metadata=out/(label+'.render.json')
        if (video.exists()and metadata.exists()and
            json.loads(metadata.read_text()).get('trace_sha256')==trace_sha and
            json.loads(metadata.read_text()).get('custom_title')==a.title and
            json.loads(metadata.read_text()).get('custom_subtitle')==a.subtitle):
            videos.append((label,video));continue
        z=np.load(trace,allow_pickle=False);times=z['time'];q=z['qpos']
        model=mj.MjModel.from_binary_path(str(trial/'model.mjb'));data=mj.MjData(model)
        model.vis.global_.offwidth,model.vis.global_.offheight=960,600
        model.vis.headlight.ambient[:]=.65;model.vis.headlight.diffuse[:]=.7
        for g in range(model.ngeom):
            owner=mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g]))or''
            if owner.startswith(('p1_','p2_')) and model.geom_contype[g]==0:
                model.geom_group[g]=1;model.geom_matid[g]=-1
                model.geom_rgba[g]=[.22,.55,.93,1.]if owner.startswith('p1_')else[.96,.48,.18,1.]
        option=mj.MjvOption();option.geomgroup[:]=[0,1,1,0,0,0]
        camera=mj.MjvCamera();camera.type=mj.mjtCamera.mjCAMERA_FREE
        camera.lookat[:]=[-.2,1.3,.75];camera.distance=4.2;camera.azimuth=100;camera.elevation=-21
        process=encoder(video,960,704);keys=set(np.linspace(0,len(sample)-1,6).round().astype(int))
        snapshots=[]
        with mj.Renderer(model,height=600,width=960)as renderer:
            for frame,t in enumerate(sample):
                i=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-1))
                data.qpos[:]=q[i];mj.mj_kinematics(model,data);mj.mj_comPos(model,data)
                renderer.update_scene(data,camera=camera,scene_option=option)
                canvas=Image.new('RGB',(960,704),'#101b2a');canvas.paste(Image.fromarray(renderer.render()),(0,64))
                draw=ImageDraw.Draw(canvas);passed=report.get('functional_task_success',False)
                draw.text((14,7),a.title or label+' | ACTUAL PHYSICS | '+('FUNCTIONAL PASS'if passed else'FAILED ACCEPTANCE'),font=font,fill='#8fdeb5'if passed else'#ffbd79')
                draw.text((14,38),a.subtitle or 'Blue=P1 / orange=P2 | free roots | joint motors | recorded integrated states',font=small,fill='white')
                note=f'execution {t:.3f}s | recorded {times[i]:.3f}s | end={times[-1]:.3f}s'
                if t>times[-1]+.01:note+=' | TERMINATED: LAST STATE HELD'
                draw.text((14,674),note,font=small,fill='#ffbd79')
                process.stdin.write(canvas.tobytes())
                if frame in keys:snapshots.append(canvas.resize((480,352)))
        close(process)
        sheet=Image.new('RGB',(3*480,2*352),'#101b2a')
        for i,im in enumerate(snapshots):sheet.paste(im,((i%3)*480,(i//3)*352))
        sheet.save(out/(label+'_contactsheet.jpg'),quality=94)
        metadata.write_text(json.dumps({'trace_sha256':trace_sha,'frames':len(sample),'fps':30,
            'physics_trace_frames':len(times),'recorded_end_s':float(times[-1]),
            'camera':'common overview100deg','only_visual_state_writes':True,
            'simulation_state_overrides_after_reset':0,
            'custom_title':a.title,'custom_subtitle':a.subtitle},indent=2)+'\n')
        videos.append((label,video));print('rendered',label,flush=True)
    if len(videos)>1:
        cols=min(4,len(videos));rows=math.ceil(len(videos)/cols);w,h=480*cols,352*rows+64
        reads=[subprocess.Popen(['ffmpeg','-nostdin','-v','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE)for _,path in videos]
        output=out/(a.tag+'.mp4');enc=encoder(output,w,h)
        for frame,t in enumerate(sample):
            canvas=Image.new('RGB',(w,h),'#101b2a');draw=ImageDraw.Draw(canvas)
            draw.text((12,7),'Frozen SONIC | ALL10 RETARGET CONFIGURATIONS | recorded physical execution',font=font,fill='white')
            draw.text((12,37),'Original time; same physical plant and controller; failure states retained',font=small,fill='#ffbd79')
            for j,reader in enumerate(reads):
                raw=reader.stdout.read(960*704*3)
                if len(raw)!=960*704*3:raise RuntimeError('Incomplete video')
                im=Image.frombytes('RGB',(960,704),raw).resize((480,352),Image.Resampling.LANCZOS)
                canvas.paste(im,((j%cols)*480,64+(j//cols)*352))
            enc.stdin.write(canvas.tobytes())
            if frame==len(sample)-1:canvas.save(out/(a.tag+'_final.jpg'),quality=95)
        close(enc)
        for reader in reads:
            reader.stdout.close()
            if reader.wait():raise RuntimeError('Decoder failed')
    (out/(a.tag+'.manifest.json')).write_text(json.dumps({'videos':[str(p)for _,p in videos],
        'frames':len(sample),'fps':30,'native_physics':True},indent=2)+'\n')


if __name__=='__main__':main()
