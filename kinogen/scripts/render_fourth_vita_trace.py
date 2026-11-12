"""Render recorded MuJoCo physics states; never advances or alters a trial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trial-report', type=Path)
    p.add_argument('--object-body')
    p.add_argument('--fps', type=float, default=30)
    p.add_argument('--playback-speed', type=float, default=1.0)
    p.add_argument('--contacts', type=Path)
    p.add_argument('--caption', default='Assumed mass/friction | native PD control')
    p.add_argument('--start-time-s', type=float)
    p.add_argument('--end-time-s', type=float)
    p.add_argument('--hold-verification', type=Path,
                   help='Independent hold report; checked against frozen trace/model hashes')
    p.add_argument('--right-view', choices=['contact', 'ego'], default='contact')
    p.add_argument('--contact-azimuth', type=float, default=155.)
    p.add_argument('--contact-elevation', type=float, default=-25.)
    p.add_argument('--contact-distance', type=float, default=.75)
    p.add_argument('--ego-body', default='neck_yaw_link')
    p.add_argument('--ego-pitch-deg', type=float, default=10.)
    p.add_argument('--ego-aim-initial-object', action='store_true',
                   help='Choose an assumed fixed mount rotation aimed at the initial object position')
    args = p.parse_args()
    if args.fps <= 0 or args.playback_speed <= 0:
        p.error('fps and playback speed must be positive')
    import numpy as np
    import mujoco
    from PIL import Image, ImageDraw, ImageFont

    z = np.load(args.trace, allow_pickle=False)
    q = z['qpos']
    time_key = next((key for key in ('time', 'time_s', 'times') if key in z), None)
    if time_key is None:
        raise ValueError('Trace has no recorded physics timestamps')
    times = z[time_key]
    if times.ndim != 1 or len(times) != len(q) or len(q) < 2:
        raise ValueError('Expected a complete timed physics trace')
    if np.any(np.diff(times) <= 0) or not np.isfinite(q).all():
        raise ValueError('Nonmonotonic time or nonfinite qpos')
    model = (mujoco.MjModel.from_binary_path(str(args.model.resolve())) if args.model.suffix=='.mjb'
             else mujoco.MjModel.from_xml_path(str(args.model.resolve())))
    if q.shape[1] != model.nq:
        raise ValueError('Trace qpos does not match frozen trial model')
    data = mujoco.MjData(model)
    if args.object_body:
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.object_body)
    else:
        free_bodies = [int(model.jnt_bodyid[i]) for i in range(model.njnt)
                       if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE]
        object_id = free_bodies[-1] if len(free_bodies) > 1 else -1
    if object_id < 0:
        raise ValueError('Pass the dynamic object body name')
    ego_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.ego_body)
    if args.right_view == 'ego' and ego_body < 0:
        raise ValueError('Synthetic ego camera attachment body missing')
    trial = json.loads(args.trial_report.read_text(encoding='utf-8')) if args.trial_report else {}
    contacts = json.loads(args.contacts.read_text(encoding='utf-8')) if args.contacts else None
    if contacts is not None and len(contacts) != len(times)-1:
        raise ValueError('Contact records do not match trace intervals')
    status = str(trial.get('status', 'DIAGNOSTIC'))
    verification = None
    verification_metrics = None
    if args.hold_verification:
        verification = json.loads(args.hold_verification.read_text(encoding='utf-8'))
        if verification['trace_sha256'] != sha(args.trace) or verification['model_sha256'] != sha(args.model):
            raise ValueError('Hold verification does not describe this exact trace/model')
        metrics = np.load(args.hold_verification.with_suffix('.metrics.npz'), allow_pickle=False)
        verification_metrics = dict(zip(map(str, metrics['columns']), metrics['values'].T))
        status = 'VERIFIED STABLE HOLD' if verification['success'] else 'NO VERIFIED STABLE HOLD'
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(704, model.vis.global_.offheight)
    model.vis.headlight.ambient = [.45, .45, .45]
    renderer = mujoco.Renderer(model, height=704, width=640)
    option = mujoco.MjvOption()
    # Asset models put colliders in group0 and visual meshes in group1.
    option.geomgroup[:] = [0, 1, 1, 1, 1, 1]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    start = float(times[0]) if args.start_time_s is None else args.start_time_s
    end = float(times[-1]) if args.end_time_s is None else args.end_time_s
    if not times[0] <= start < end <= times[-1]+1e-8:
        raise ValueError('Requested video window must lie within recorded physics')
    sample_times = np.arange(start, end + 1e-8, args.playback_speed/args.fps)
    indices = np.searchsorted(times, sample_times, side='right')-1
    indices = np.clip(indices, 0, len(times)-1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix('.partial.mp4')
    command = [shutil.which('ffmpeg') or 'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
               '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', '1280x800', '-r', str(args.fps),
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-metadata',
               'comment=Render of actual recorded MuJoCo states; status '+status+'; trace '+sha(args.trace), str(partial)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    snapshots = []
    ego_poses = []
    ego_mount_rotation = None
    keys = set(np.linspace(0, len(indices)-1, min(9, len(indices))).round().astype(int))
    try:
        for frame, i in enumerate(indices):
            data.qpos[:] = q[i]
            if 'qvel' in z:
                data.qvel[:] = z['qvel'][i]
            mujoco.mj_forward(model, data)
            obj = data.xpos[object_id].copy()
            # Robot free root comes first in the generated scene; bodies near
            # it set framing, while the source scene mesh remains visible.
            root = data.xpos[1]
            near = data.xpos[1:][np.linalg.norm(data.xpos[1:]-root, axis=1) < 2.5]
            low, high = near.min(axis=0), near.max(axis=0)
            target = (low+high)/2
            distance = max(2.1, float(np.linalg.norm(high-low))*1.25)
            canvas = Image.new('RGB', (1280,800), '#15191f')
            panels = [(0,target,distance,135,-15),
                      (1,obj,args.contact_distance,args.contact_azimuth,args.contact_elevation)]
            if args.right_view == 'ego':
                rotation = data.xmat[ego_body].reshape(3, 3)
                eye = data.xpos[ego_body] + rotation @ np.asarray([.08, 0, .08])
                if ego_mount_rotation is None:
                    angle = np.deg2rad(args.ego_pitch_deg)
                    f_local = rotation.T @ (obj-eye) if args.ego_aim_initial_object else np.asarray([np.cos(angle), 0, -np.sin(angle)])
                    f_local /= np.linalg.norm(f_local)
                    r_local = np.cross(f_local, np.asarray([0., 0., 1.]))
                    r_local /= np.linalg.norm(r_local)
                    ego_mount_rotation = np.column_stack([r_local, np.cross(f_local, r_local), f_local])
                camera_rotation = rotation @ ego_mount_rotation
                right, down, forward = camera_rotation.T
                quat = np.empty(4)
                mujoco.mju_mat2Quat(quat, camera_rotation.ravel())
                ego_poses.append(np.r_[eye, quat])
                az = np.rad2deg(np.arctan2(forward[1], forward[0]))
                el = np.rad2deg(np.arcsin(np.clip(forward[2], -1, 1)))
                panels[1] = (1, eye + .8*forward, .8, az, el)
            for panel, lookat, dist, az, el in panels:
                cam.lookat[:] = lookat
                cam.distance = dist
                cam.azimuth = az
                cam.elevation = el
                renderer.update_scene(data, camera=cam, scene_option=option)
                if panel == 1 and args.right_view == 'ego':
                    gl = renderer.scene.camera
                    # The free-camera angles do not encode head roll. Apply
                    # the assumed mount's up vector to both GL cameras.
                    for view in gl:
                        view.up[:] = -down
                    rendered_eye = (np.asarray(gl[0].pos)+np.asarray(gl[1].pos))/2
                    if np.linalg.norm(rendered_eye-eye) > 1e-4 or np.linalg.norm(np.asarray(gl[0].forward)-forward) > 1e-4 or np.linalg.norm(np.asarray(gl[0].up)+down) > 1e-4:
                        raise ValueError(f'Rendered ego camera differs: eye {rendered_eye} vs {eye}; forward {gl[0].forward} vs {forward}')
                canvas.paste(Image.fromarray(renderer.render()), (panel*640,56))
            draw = ImageDraw.Draw(canvas)
            draw.text((12,7), 'Vita02a | MUJOCO PHYSICS | '+status[:62], font=font, fill='white')
            draw.text((12,31), args.caption+f' | playback {args.playback_speed:g}x', font=font, fill='#b9c8db')
            if args.right_view == 'ego':
                draw.text((656,62), 'SYNTHETIC HEAD CAMERA (assumed mount)', font=font, fill='#ffcd64')
            torque = z['actuator_force'][i-1] if 'actuator_force' in z and i > 0 else None
            maximum = float(np.max(np.abs(torque))) if torque is not None else 0.0
            footer = f't={times[i]-times[0]:.3f}s | state {i+1}/{len(times)} | peak motor torque={maximum:.2f} Nm'
            if verification_metrics is not None:
                vi = int(np.clip(np.searchsorted(verification_metrics['time_s'], times[i], side='right')-1,
                                 0, len(verification_metrics['time_s'])-1))
                footer = (f't={times[i]:.3f}s | lowest plate clearance '
                          f"{verification_metrics['lowest_table_plane_clearance_m'][vi]*100:.2f} cm | "
                          f"hand support {verification_metrics['hand_up_N'][vi]:.2f} N | "
                          f"verified hold {verification['longest_verified_hold_s']:.2f} s")
            if contacts is not None and i > 0:
                contact = contacts[i-1]
                footer += f" | hand/bowl {contact['object_hand_normal_force_N']:.2f} N | penetration {contact['max_penetration_m']*1000:.1f} mm"
            draw.text((12,768), footer, font=font, fill='#ffcd64')
            encoder.stdin.write(np.asarray(canvas).tobytes())
            if frame in keys:
                snapshots.append(canvas.copy())
            if frame == 0:
                canvas.save(args.output.with_name(args.output.stem+'_first.jpg'), quality=94)
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors='replace')
        if encoder.wait() != 0:
            raise RuntimeError(error)
        partial.replace(args.output)
    finally:
        renderer.close()
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    sheet = Image.new('RGB', (1920, ((len(snapshots)+2)//3)*400), '#15191f')
    for j, snap in enumerate(snapshots):
        sheet.paste(snap.resize((640,400)), ((j%3)*640, (j//3)*400))
    sheet_path = args.output.with_name(args.output.stem+'_contactsheet.jpg')
    sheet.save(sheet_path, quality=92)
    probe = json.loads(subprocess.run([shutil.which('ffprobe') or 'ffprobe', '-v', 'error',
                '-count_frames', '-select_streams', 'v:0', '-show_entries',
                'stream=nb_read_frames,r_frame_rate,width,height', '-of', 'json', str(args.output)],
                capture_output=True, check=True, text=True).stdout)['streams'][0]
    if int(probe['nb_read_frames']) != len(indices):
        raise RuntimeError('Video lost frames')
    report = {'mode': 'render_of_recorded_physics_states', 'trial_status': status,
              'trace_sha256': sha(args.trace), 'model_sha256': sha(args.model),
              'trace_states': len(times), 'render_frames': len(indices), 'fps': args.fps,
              'playback_speed': args.playback_speed, 'caption': args.caption,
              'right_view': args.right_view,
              'contact_camera': {'distance_m': args.contact_distance,
                                 'azimuth_deg': args.contact_azimuth,
                                 'elevation_deg': args.contact_elevation},
              'simulation_time_s': float(times[-1]-times[0]),
              'rendered_time_window_s': [float(start), float(end)],
              'hold_verification': None if verification is None else {
                  'path': str(args.hold_verification), 'sha256': sha(args.hold_verification),
                  'success': verification['success'], 'hold_s': verification['longest_verified_hold_s']},
              'video': str(args.output), 'video_sha256': sha(args.output),
              'contactsheet': str(sheet_path), 'contactsheet_sha256': sha(sheet_path), 'probe': probe}
    if ego_poses:
        ego_path = args.output.with_suffix('.ego_camera.npz')
        np.savez_compressed(ego_path, time_s=times[indices],
                            camera_pose_xyz_wxyz=np.asarray(ego_poses),
                            camera_axes=np.asarray('OpenCV: X right, Y down, Z forward'),
                            mount_body=np.asarray(args.ego_body),
                            mount_translation_m=np.asarray([.08, 0, .08]),
                            downward_mount_pitch_deg=np.asarray(args.ego_pitch_deg),
                            mount_rotation_right_down_forward=ego_mount_rotation,
                            aimed_at_initial_object=np.asarray(args.ego_aim_initial_object),
                            vertical_fov_deg=np.asarray(model.vis.global_.fovy),
                            render_width=np.asarray(640), render_height=np.asarray(704))
        report['synthetic_ego_camera'] = {'path': str(ego_path), 'sha256': sha(ego_path),
                                        'mount_body': args.ego_body, 'calibrated': False,
                                        'mount_orientation': 'fixed initial-object aim' if args.ego_aim_initial_object else 'fixed pitch'}
    args.output.with_suffix('.render.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
