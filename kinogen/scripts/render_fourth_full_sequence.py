"""Render every source frame as a labelled reference/physical comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--trial-dir',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fps',type=float,default=30.)
    p.add_argument('--view',choices=['whole','table'],default='whole')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from PIL import Image,ImageDraw,ImageFont
    ref=np.load(a.reference,allow_pickle=False);times=ref['time_s']-ref['time_s'][0]
    poses=ref['object_target_pose'];ids=np.asarray(ref['object_ids']).reshape(-1)
    model=mj.MjModel.from_binary_path(str(a.model.resolve()))if a.model.suffix=='.mjb'else mj.MjModel.from_xml_path(str(a.model.resolve()));data=mj.MjData(model)
    qa=np.asarray([model.jnt_qposadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,str(n))]for n in ref['dof_names']])
    oqa=np.asarray([model.jnt_qposadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,f'target_object_{i:02d}_free')]for i in ids])
    trial=json.loads((a.trial_dir/'report.json').read_text())if a.trial_dir else {}
    controller_label=('SONIC legs/waist + object-arm IK + finger PD'if trial.get('sonic',{}).get('arm_object_feedback')
                      else 'SONIC body + reference hand/neck PD'if trial.get('controller')=='sonic'else trial.get('controller','none'))
    trace=np.load(a.trial_dir/'physical_trace.npz',allow_pickle=False)if a.trial_dir else None
    if trace is not None and (trial['model']['sha256']!=sha(a.model)or trial['reference']['sha256']!=sha(a.reference)):
        raise ValueError('Trace inputs differ from render inputs')
    if trace is not None and (float(trace['time'][0]) > float(times[0])+1e-8
                             or float(trace['time'][-1]) < float(times[-1])-1e-8):
        raise ValueError('Physical trace does not cover every reference frame; refusing to extend a partial run by holding its last state')
    indices=np.minimum(np.searchsorted(trace['time'],times),len(trace['time'])-1)if trace is not None else None
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,640);model.vis.global_.offheight=max(model.vis.global_.offheight,704)
    model.vis.headlight.ambient=[.45,.45,.45]
    renderer=mj.Renderer(model,height=704,width=640);opt=mj.MjvOption();opt.geomgroup[:]=[0,1,1,1,1,1]
    camera=mj.MjvCamera();camera.type=mj.mjtCamera.mjCAMERA_FREE
    object_xyz=poses[:,:,:3].reshape(-1,3)
    centre=np.median(object_xyz,axis=0)
    if a.view=='table':
        camera.lookat[:]=centre;camera.distance=1.4;camera.azimuth=135;camera.elevation=-48
    else:
        points=np.r_[ref['qpos'][:,:3],object_xyz]
        lo,hi=points.min(axis=0),points.max(axis=0);camera.lookat[:]=(lo+hi)/2;camera.lookat[2]-=.25
        camera.distance=max(3.,np.linalg.norm(hi-lo)*1.45);camera.azimuth=135;camera.elevation=-18
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,16);small=ImageFont.truetype(fontpath,14)
    observed=np.asarray(ref.get('object_observed_mask',np.ones((len(times),6),bool)))
    human=np.asarray(ref.get('human_observed_mask',np.ones(len(times),bool)))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    partial=a.output.with_suffix('.partial.mp4')
    encoder=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s:v','1280x800','-r',str(a.fps),'-i','pipe:0','-an','-c:v','libx264','-crf','20','-preset','veryfast','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    snapshots=[];keys=set(np.linspace(0,len(times)-1,9).round().astype(int));maxgap=0.
    try:
        for i,t in enumerate(times):
            canvas=Image.new('RGB',(1280,800),'#121820');draw=ImageDraw.Draw(canvas)
            data.qpos[:7]=ref['qpos'][i,:7];data.qpos[qa]=ref['qpos'][i,7:]
            for j,k in enumerate(oqa):data.qpos[k:k+7]=poses[i,j]
            mj.mj_forward(model,data);renderer.update_scene(data,camera=camera,scene_option=opt);canvas.paste(Image.fromarray(renderer.render()),(0,56))
            if trace is not None:
                ii=indices[i];data.qpos[:]=trace['qpos'][ii];data.qvel[:]=trace['qvel'][ii];mj.mj_forward(model,data)
                maxgap=max(maxgap,abs(float(trace['time'][ii])-float(t)))
                renderer.update_scene(data,camera=camera,scene_option=opt);canvas.paste(Image.fromarray(renderer.render()),(640,56))
            draw=ImageDraw.Draw(canvas)
            draw.text((12,6),'Retargeted Vita reference | adaptations labelled',font=font,fill='white')
            label=trial.get('status','REFERENCE ONLY - physical trial pending')
            draw.text((652,6),'Actual physics | '+label,font=font,fill='white')
            draw.text((12,31),f'Source {i:03d}/922 | t={t:.3f}s | human '+('observed'if human[i]else'IMPUTED'),font=font,fill='#ffcd64')
            draw.text((652,31),controller_label+' | assumed physics',font=small,fill='#b9c8db')
            flags='  '.join(f"{oid}:{'obs'if observed[i,j]else'FILL'}"for j,oid in enumerate(ids))
            draw.text((12,765),'Object source labels  '+flags,font=small,fill='#ffcd64')
            if trace is not None:
                ii=max(1,indices[i]);pen=float(trace['max_penetration_m'][min(ii-1,len(trace['max_penetration_m'])-1)])
                draw.text((652,765),f"physics t={trace['time'][indices[i]]:.3f}s | max penetration={pen*1000:.1f} mm",font=small,fill='#ffcd64')
            encoder.stdin.write(np.asarray(canvas).tobytes())
            if i in keys:snapshots.append(canvas.resize((640,400)))
        encoder.stdin.close();err=encoder.stderr.read().decode(errors='replace')
        if encoder.wait()!=0:raise RuntimeError(err)
        partial.replace(a.output)
    finally:
        renderer.close()
        if encoder.poll()is None:encoder.kill();encoder.wait()
    sheet=Image.new('RGB',(1920,1200))
    for i,im in enumerate(snapshots):sheet.paste(im,((i%3)*640,(i//3)*400))
    sheetpath=a.output.with_name(a.output.stem+'_contactsheet.jpg');sheet.save(sheetpath,quality=92)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,width,height,r_frame_rate','-of','json',str(a.output)]))['streams'][0]
    if int(probe['nb_read_frames'])!=len(times):raise ValueError('Video frame count mismatch')
    result={'source_frames':len(times),'rendered_frames':int(probe['nb_read_frames']),'reference_sha256':sha(a.reference),'model_sha256':sha(a.model),
        'trace_sha256':sha(a.trial_dir/'physical_trace.npz')if a.trial_dir else None,'max_source_to_physics_sampling_gap_s':maxgap,
        'trial_status':trial.get('status'),'source_duration_s':float(times[-1]),'fps':a.fps,'view':a.view,'video':str(a.output),'video_sha256':sha(a.output),'probe':probe,
        'human_filled_frames':int((~human).sum()),'object_filled_frames':(~observed).sum(axis=0).tolist()}
    a.output.with_suffix('.render.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
