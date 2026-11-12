"""Render recorded dual-Vita states and explicitly synthetic ego camera views."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fps', type=float, default=23.)
    p.add_argument('--speed', type=float, default=1.)
    p.add_argument('--right-view', choices=['shoe', 'ego'], default='shoe')
    p.add_argument('--shoe-proxy-overlay',action='store_true')
    a = p.parse_args()
    import numpy as np
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    if a.fps <= 0 or a.speed <= 0:
        raise ValueError('Positive presentation rate/speed required')
    report = json.loads((a.trial/'report.json').read_text())
    model_path, trace_path = a.trial/'model.mjb', a.trial/'trace.npz'
    if sha(model_path) != report['model']['sha256'] or sha(trace_path) != report['trace']['sha256']:
        raise ValueError('Model/trace changed after trial')
    model = mujoco.MjModel.from_binary_path(str(model_path))
    model.vis.global_.offwidth, model.vis.global_.offheight = 800, 720
    model.vis.headlight.ambient[:] = [.5, .5, .5]
    data = mujoco.MjData(model)
    z = np.load(trace_path, allow_pickle=False)
    q, times = z['qpos'], z['time']
    if q.shape != (len(times), model.nq) or np.any(np.diff(times) <= 0):
        raise ValueError('Invalid timed trace')
    shoe_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'shoe_free')
    shoe_body = int(model.jnt_bodyid[shoe_jid])
    if a.shoe_proxy_overlay:
        for g in range(model.ngeom):
            if model.geom_bodyid[g]==shoe_body and (model.geom_contype[g] or model.geom_conaffinity[g]):
                model.geom_group[g]=2
                model.geom_matid[g]=-1
                model.geom_rgba[g]=[1.,.46,.08,.4]
    head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'p1_neck_yaw_link')
    if a.right_view == 'ego' and head < 0:
        raise ValueError('Head attachment body absent')
    sample = np.arange(times[0], times[-1]+1e-8, a.speed/a.fps)
    ids = np.clip(np.searchsorted(times, sample, side='right')-1, 0, len(times)-1)
    option = mujoco.MjvOption()
    option.geomgroup[:] = [0, 1, 1, 1, 1, 1]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    partial = a.output.with_suffix('.partial.mp4')
    encoder = subprocess.Popen(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', '1600x800', '-r', str(a.fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    keys = set(np.linspace(0, len(ids)-1, min(9, len(ids))).round().astype(int))
    snapshots, ego = [], []
    try:
        with mujoco.Renderer(model, height=720, width=800) as renderer:
            for f, i in enumerate(ids):
                data.qpos[:] = q[i]
                data.qvel[:] = z['qvel'][i]
                k = max(0, i-1)
                if 'eq_active' in z:
                    data.eq_active[:] = z['eq_active'][k]
                if 'eq_data' in z:
                    model.eq_data[:] = z['eq_data'][k]
                mujoco.mj_forward(model, data)
                shoe = data.xpos[shoe_body].copy()
                canvas = Image.new('RGB', (1600, 800), '#15191f')
                for panel in range(2):
                    if panel == 0:
                        cam.lookat[:] = [-.45, .0, .65]
                        cam.distance, cam.azimuth, cam.elevation = 3.45, 90., -18.
                    elif a.right_view == 'shoe':
                        cam.lookat[:] = shoe
                        cam.distance, cam.azimuth, cam.elevation = .85, 125., -25.
                    else:
                        rotation = data.xmat[head].reshape(3, 3)
                        eye = data.xpos[head] + rotation @ np.asarray([.08, 0., .08])
                        forward = shoe-eye
                        forward /= np.linalg.norm(forward)
                        right = np.cross(forward, np.asarray([0., 0., 1.]))
                        right /= np.linalg.norm(right)
                        down = np.cross(forward, right)
                        quaternion = np.empty(4)
                        mujoco.mju_mat2Quat(quaternion, np.column_stack([right, down, forward]).ravel())
                        ego.append(np.r_[eye, quaternion])
                        cam.lookat[:] = eye+.8*forward
                        cam.distance = .8
                        cam.azimuth = np.rad2deg(np.arctan2(forward[1], forward[0]))
                        cam.elevation = np.rad2deg(np.arcsin(np.clip(forward[2], -1, 1)))
                    renderer.update_scene(data, camera=cam, scene_option=option)
                    if panel == 1 and a.right_view == 'ego':
                        for view in renderer.scene.camera:
                            view.up[:] = -down
                        views = renderer.scene.camera
                        actual_eye = (np.asarray(views[0].pos)+np.asarray(views[1].pos))/2
                        if np.linalg.norm(actual_eye-eye)>1e-4 or np.linalg.norm(views[0].forward-forward)>1e-4:
                            raise ValueError('Rendered camera differs from exported pose')
                    canvas.paste(Image.fromarray(renderer.render()), (800*panel, 56))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 5), 'RGB 261026 | TWO VITA52 + BENCH + SHOE | '+report['status'], font=font, fill='white')
                draw.text((12, 29), f'Recorded physics | assumed geometry/physical parameters | {report["mode"]} | playback {a.speed:g}x', font=font, fill='#b9c8db')
                if a.right_view == 'ego':
                    draw.text((812, 62), 'SYNTHETIC EGO: head position + virtual look-at gimbal', font=font, fill='#ffcd64')
                if a.shoe_proxy_overlay:
                    draw.text((812, 88), 'Orange: inferred shoe completion / collision proxy',font=font,fill='#ffcd64')
                footer = f't={times[i]:.3f}s | state {i+1}/{len(times)}'
                if i>0:
                    footer += f' | shoe target error {z["shoe_reference_error_m"][k]*100:.1f} cm'
                    footer += f' | max motor torque {np.max(np.abs(z["actuator_force"][k])):.1f} Nm'
                    if 'eq_constraint_wrench_world_on_shoe' in z:
                        load = np.linalg.norm(z['eq_constraint_wrench_world_on_shoe'][k,:,:3], axis=1).sum()
                        footer += f' | attachment load {load:.1f} N'
                draw.text((12, 778), footer, font=font, fill='#ffcd64')
                encoder.stdin.write(np.asarray(canvas).tobytes())
                if f in keys:
                    snapshots.append(canvas.copy())
                if f == 0:
                    canvas.save(a.output.with_name(a.output.stem+'_first.jpg'), quality=94)
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors='replace')
        if encoder.wait() != 0:
            raise RuntimeError(error)
        partial.replace(a.output)
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    sheet = Image.new('RGB', (1920, ((len(snapshots)+2)//3)*320), '#15191f')
    for j, snap in enumerate(snapshots):
        sheet.paste(snap.resize((640, 320)), ((j%3)*640, (j//3)*320))
    sheet_path = a.output.with_name(a.output.stem+'_contactsheet.jpg')
    sheet.save(sheet_path, quality=92)
    probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate', '-of', 'json', str(a.output)],
        capture_output=True, check=True, text=True).stdout)['streams'][0]
    if int(probe['nb_read_frames']) != len(ids):
        raise ValueError('Encoded frame count differs')
    out = {'mode':'recorded_dual_robot_physics_render', 'trial_status':report['status'],
           'model_sha256':sha(model_path), 'trace_sha256':sha(trace_path), 'frames':len(ids),
           'source_frames':report['source_frames'], 'physics_duration_s':float(times[-1]),
           'fps':a.fps, 'playback_speed':a.speed, 'video':str(a.output.resolve()),
           'video_sha256':sha(a.output), 'contactsheet':str(sheet_path.resolve()),
           'contactsheet_sha256':sha(sheet_path), 'probe':probe,
           'shoe_proxy_overlay':{'enabled':a.shoe_proxy_overlay,'meaning':'Orange translucent inferred shoe completion; raw mesh retained; rendering-only change'}}
    if ego:
        camera_path = a.output.with_suffix('.ego_camera.npz')
        np.savez_compressed(camera_path, time_s=times[ids], camera_pose_xyz_wxyz=np.asarray(ego),
            axes=np.asarray('OpenCV X right, Y down, Z forward'), mount_body=np.asarray('p1_neck_yaw_link'),
            orientation=np.asarray('synthetic per-frame look-at shoe gimbal, not measured head orientation'),
            width=np.asarray(800), height=np.asarray(720), vertical_fov_deg=np.asarray(model.vis.global_.fovy))
        out['synthetic_ego_camera'] = {'path':str(camera_path), 'sha256':sha(camera_path), 'calibrated':False}
    a.output.with_suffix('.render.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out), flush=True)


if __name__ == '__main__':
    main()
