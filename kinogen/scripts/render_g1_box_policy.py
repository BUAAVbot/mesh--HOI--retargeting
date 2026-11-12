"""Offline video of captured PhysX link states; never recompute or replace motion."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image,ImageDraw


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--episode',type=int,default=0)
    p.add_argument('--azimuth',type=float,default=35.)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh output video')
    trace_path=args.evaluation/'trace.npz'
    trace=np.load(trace_path,allow_pickle=False)
    indices=np.flatnonzero(trace['episode_id']==args.episode)
    if len(indices)<2:raise ValueError('No complete episode to render')
    names=list(map(str,trace['body_names']))
    geometry=args.evaluation/'render-geometry'
    manifest=json.loads((geometry/'manifest.json').read_text())
    if names!=manifest['robot_body_names']:raise ValueError('Live mesh/body order changed')
    if str(trace['quaternion_order'])!='wxyz':raise ValueError('Only explicit PhysX WXYZ supported')
    spec=mujoco.MjSpec()
    spec.worldbody.add_light(pos=[1,-2,4],dir=[0,0,-1])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE,size=[5,5,.1],rgba=[.83,.85,.86,1])
    body_map={name:spec.worldbody.add_body(name=name,mocap=True)for name in names}
    box_body=spec.worldbody.add_body(name='captured_box',mocap=True)
    table_body=spec.worldbody.add_body(name='captured_table',mocap=True)
    for i,row in enumerate(manifest['rows']):
        mesh=f'visual_{i}'
        spec.add_mesh(name=mesh,file=str((geometry/row['file']).resolve()))
        body=body_map[row['body']]if row['group']=='robot'else box_body if row['group']=='object'else table_body
        color=row['rgba']if row['group']=='robot'else[.55,.29,.1,1.]if row['group']=='object'else[.43,.32,.21,1.]
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH,meshname=mesh,rgba=color,contype=0,conaffinity=0)
    model=spec.compile();data=mujoco.MjData(model)
    ids=[int(model.body(n).mocapid[0])for n in names]
    bid=int(model.body('captured_box').mocapid[0]);tid=int(model.body('captured_table').mocapid[0])
    data.mocap_pos[tid]=manifest['table_root'];data.mocap_quat[tid]=manifest['table_quat']
    camera=mujoco.MjvCamera();camera.lookat[:]=[.16,0.,.73]
    camera.distance=2.6;camera.azimuth=args.azimuth;camera.elevation=-15.
    model.vis.headlight.ambient[:]=[.40,.40,.40];model.vis.headlight.diffuse[:]=[.7,.7,.7]
    width,height=1280,720
    model.vis.global_.offwidth=width;model.vis.global_.offheight=height
    dt=float(np.median(np.diff(trace['timestamps'][indices])))
    if dt<=0:raise ValueError('Invalid actual episode timebase')
    fps=1/dt;args.output.parent.mkdir(parents=True,exist_ok=True)
    process=subprocess.Popen(['/usr/bin/ffmpeg','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-',
        '-an','-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p',str(args.output)],stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model,height=height,width=width)as renderer:
            for frame,index in enumerate(indices):
                data.mocap_pos[ids]=trace['body_pos'][index]
                data.mocap_quat[ids]=trace['body_quat'][index]
                data.mocap_pos[bid]=trace['box_pos'][index];data.mocap_quat[bid]=trace['box_quat'][index]
                if not np.isfinite(data.mocap_pos).all():raise ValueError('Cannot render nonfinite physics state')
                mujoco.mj_forward(model,data);renderer.update_scene(data,camera=camera)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.rectangle([0,0,width,57],fill='black')
                draw.text((12,7),'G1 + bilateral five-finger ROHand | Actual GPU PhysX policy rollout',fill='white')
                draw.text((12,25),f'Episode {args.episode} | t={trace["timestamps"][index]:.2f}s | hand/box contacts L/R: {trace["hand_contact_force_left_N"][index]:.2f}/{trace["hand_contact_force_right_N"][index]:.2f} N | table: {trace["table_contact_force_N"][index]:.2f} N',fill='white')
                draw.text((12,41),'Offline display of captured link states; no reference-pose replacement or new dynamics',fill='white')
                process.stdin.write(np.asarray(image).tobytes())
                if frame in (0,len(indices)//2,len(indices)-1):image.save(args.output.with_name(args.output.stem+f'_{frame:04d}.jpg'))
    finally:
        process.stdin.close();returncode=process.wait()
    if returncode:raise RuntimeError('Video encoding failed')
    provenance={'render_source':'actual_physics_trace','trace_sha256':sha(trace_path),
        'checkpoint_sha256':str(trace['source_checkpoint_sha256']),'video_sha256':sha(args.output),
        'source_sample_indices':indices.tolist(),'episode':args.episode,'frames':len(indices),'fps':fps,
        'renderer':'MuJoCo visual renderer only; physics generated in IsaacSim4.5',
        'geometry_manifest_sha256':sha(geometry/'manifest.json'),'reference_poses_substituted':False,
        'render_implementation_sha256':sha(__file__)}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
