"""Export every retargeted source frame, explicitly labelled as kinematic."""
from pathlib import Path
import argparse,hashlib,json,subprocess


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--source-video',type=Path,default=Path('data/rgb_145317_sim_handoff/input.mp4'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np,mujoco as mj
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    if a.output.exists():raise FileExistsError(a.output)
    z=np.load(a.reference,allow_pickle=False);source=z['source_time_s'];ids=z['frame_ids']
    info=json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames','-of','json',str(a.source_video)],capture_output=True,text=True,check=True).stdout)['streams'][0]
    w,h=info['width'],info['height'];num,den=map(float,info['r_frame_rate'].split('/'));fps=num/den
    if not np.array_equal(ids,np.arange(len(ids)))or not np.allclose(source,np.arange(len(ids))/fps,atol=2e-5):raise ValueError('Expected complete contiguous source clock')
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m)
    m.vis.global_.offwidth=640;m.vis.global_.offheight=480;m.vis.headlight.ambient[:]=[.5,.5,.5]
    cam=mj.MjvCamera();cam.type=mj.mjtCamera.mjCAMERA_FREE
    opt=mj.MjvOption();opt.geomgroup[:]=[0,1,1,1,1,1]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    a.output.parent.mkdir(parents=True,exist_ok=True);partial=a.output.with_suffix('.partial.mp4')
    dec=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-i',str(a.source_video),'-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
    enc=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s:v','1280x560','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(partial)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    snaps=[];keys=set(np.linspace(0,len(ids)-1,6).round().astype(int))
    try:
        with mj.Renderer(m,height=480,width=640)as renderer:
            for f in range(len(ids)):
                raw=dec.stdout.read(w*h*3)
                if len(raw)!=w*h*3:raise ValueError('Incomplete original RGB frame '+str(f))
                original=Image.fromarray(np.frombuffer(raw,dtype=np.uint8).reshape(h,w,3))
                d.qpos[:]=z['qpos'][f];mj.mj_forward(m,d)
                cam.lookat[:]=d.qpos[:3]+[0,0,.1];cam.distance=3.1;cam.azimuth=70.;cam.elevation=-17.
                renderer.update_scene(d,camera=cam,scene_option=opt)
                im=Image.new('RGB',(1280,560),'#15191f');im.paste(Image.fromarray(renderer.render()),(0,56));im.paste(ImageOps.pad(original,(640,480),color='#15191f'),(640,56))
                draw=ImageDraw.Draw(im);draw.text((12,4),'STAGE RESULT | VITA + O6 | KINEMATIC RETARGETING',font=font,fill='white')
                draw.text((12,29),'NOT PHYSICS VALIDATED | 350/350 source frames | Estimated scene / inferred contacts',font=small,fill='#ffce69')
                draw.text((652,61),'ORIGINAL RGB',font=font,fill='#ffce69')
                stage='approach / turn'if source[f]<3.3 else'ascend'if source[f]<5.2 else'turn / descend'if source[f]<8.7 else'sit down'
                draw.text((12,538),f'Source {source[f]:.3f}s | frame {int(ids[f]):03d}/349 | {stage} | No simulated force claims',font=small,fill='#ffce69')
                enc.stdin.write(im.tobytes())
                if f in keys:snaps.append(im.resize((640,280)))
                if f%100==0:print(json.dumps({'rendered_frames':f+1,'total':len(ids)}),flush=True)
        enc.stdin.close();error=enc.stderr.read().decode(errors='replace')
        if enc.wait()!=0:raise RuntimeError(error)
        partial.replace(a.output)
    finally:
        if enc.poll()is None:enc.kill();enc.wait()
        dec.stdout.close();dec.terminate();dec.wait()
    sheet=Image.new('RGB',(1280,840))
    for i,im in enumerate(snaps):sheet.paste(im,((i%2)*640,(i//2)*280))
    sheet_path=a.output.with_name(a.output.stem+'_contactsheet.jpg');sheet.save(sheet_path,quality=93)
    receipt={'mode':'full_source_clock_kinematic_preview','frames':len(ids),'fps':fps,'duration_s':len(ids)/fps,'physics_validated':False,'reference_sha256':sha(a.reference),'scene_model_sha256':sha(a.model),'source_video_sha256':sha(a.source_video),'video_sha256':sha(a.output),'source_frames_dropped':0,'source_frame_ids_unchanged':True,'contacts':'Inferred constraints, not measured or simulated forces','simulation_status':'Separate physical trials currently fail quality gates'}
    a.output.with_suffix('.render.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
