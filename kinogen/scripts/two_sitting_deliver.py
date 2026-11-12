"""Render recorded two-Vita dynamics and export explicitly qualified HOI data.

Rendering copies frozen states into a separate visualization model. It does not
integrate or modify the trial. A failed physical trial remains FAILED in both
the video and metadata. Optional reference video is clearly KINEMATIC.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def receipt(path):
    p = Path(path).resolve()
    return {'path': str(p), 'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                   allow_nan=False)+'\n', encoding='utf-8')


def load_numeric(path, np):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def classify_report(report):
    checks = report.get('checks') or {}
    passed = report.get('simulation_passed', report.get('success')) is True
    if checks and any(value is not True for value in checks.values()):
        passed = False
    source_pass = report.get('source_tracking_pass', passed) is True
    if not source_pass:
        passed = False
    functional_checks = report.get('functional_task_checks') or {}
    functional_pass = report.get('functional_physical_demo_pass', report.get('functional_task_success', passed)) is True
    if functional_checks and any(value is not True for value in functional_checks.values()):
        functional_pass = False
    if functional_pass and not source_pass:
        label = 'FUNCTIONAL TASK PASS | SOURCE TRACKING FAIL'
        state = 'functional_physical_demo_pass_source_tracking_failed'
    elif passed:
        label = 'PHYSICS PASS'
        state = 'accepted_assumed_physics'
    else:
        label = 'PHYSICS FAIL'
        state = 'diagnostic_physics_not_accepted'
    return {'simulation_passed': passed, 'functional_task_pass': functional_pass,
            'source_tracking_pass': source_pass, 'display_status': label, 'status': state,
            'maximum_source_root_xy_error_m': report.get('max_root_xy_error_m'),
            'trial_limitations': report.get('limitations', []),
            'validation_scope': 'Functional task completion and strict source-trajectory reproduction are separate. A selected recorded-control demonstration does not establish policy success rate or generalization.'}


def render(model, z, qposes, times, output, label, passed, fps, speed, np, mj, quality=None):
    from PIL import Image, ImageDraw, ImageFont
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    model.vis.headlight.ambient[:] = [.55, .55, .55]
    model.vis.headlight.diffuse[:] = [.65, .65, .65]
    data = mj.MjData(model)
    roots = np.asarray(z['robot_root_qadr'], dtype=int)
    points = np.concatenate([qposes[:, adr:adr+3] for adr in roots], axis=0)
    lo, hi = points.min(0), points.max(0)
    look = (lo+hi)/2
    look[2] = max(.65, float(look[2])*.7)
    extent = max(2., float(np.linalg.norm((hi-lo)[:2]))+1.4)
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = look
    camera.distance = extent*1.35
    camera.azimuth, camera.elevation = 100., -20.
    opt = mj.MjvOption()
    opt.geomgroup[:] = [0, 1, 1, 1, 1, 1]
    # Keep an unmeshed ground surface visible without displaying collision
    # volumes on top of the robot's visual meshes.
    for gid in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, gid) or ''
        if model.geom_type[gid] == mj.mjtGeom.mjGEOM_PLANE and ('floor' in name or 'ground' in name):
            model.geom_group[gid] = 1
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:
        font = ImageFont.truetype(font_path, 19)
    except OSError:
        font = ImageFont.load_default()
    sample_t = np.arange(times[0], times[-1]+1e-9, speed/fps)
    if len(sample_t) == 0:
        sample_t = times[:1]
    ids = np.minimum(np.searchsorted(times, sample_t, side='left'), len(times)-1)
    # Every image is one recorded state; presentation never smooths a fall away.
    prev = np.maximum(ids-1, 0)
    ids = np.where(abs(times[prev]-sample_t) < abs(times[ids]-sample_t), prev, ids)
    partial = output.with_name(output.stem+'.partial.mp4')
    if partial.exists():
        raise FileExistsError(partial)
    cmd = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', '1280x800',
           '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
           '-crf', '20', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(partial)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = []
    keyframes = set(np.linspace(0, len(ids)-1, min(9, len(ids))).round().astype(int))
    quality = quality or {'simulation_passed': passed, 'functional_task_pass': passed,
                          'source_tracking_pass': passed, 'display_status': 'PHYSICS PASS' if passed else 'PHYSICS FAIL'}
    status = quality['display_status'] if label == 'RECORDED PHYSICS' else 'REFERENCE ONLY'
    try:
        with mj.Renderer(model, height=720, width=1280) as renderer:
            for frame, index in enumerate(ids):
                data.qpos[:] = qposes[index]
                data.qvel[:] = 0
                if label == 'RECORDED PHYSICS' and 'qvel' in z:
                    data.qvel[:] = z['qvel'][index]
                data.time = float(times[index])
                mj.mj_forward(model, data)
                renderer.update_scene(data, camera=camera, scene_option=opt)
                image = Image.new('RGB', (1280, 800), '#15191f')
                image.paste(Image.fromarray(renderer.render()), (0, 56))
                draw = ImageDraw.Draw(image)
                partial_functional = label == 'RECORDED PHYSICS' and quality.get('functional_task_pass') and not quality.get('source_tracking_pass')
                color = '#ffd68a' if partial_functional else '#ffb4a2' if label == 'RECORDED PHYSICS' and not passed else '#b5e9cb'
                title = 'TWO VITA | '+status if partial_functional else f'TWO VITA | STAND - EXCHANGE SEATS - SIT | {label} | {status}'
                subtitle = 'Actual physics: rise, exchange seats, sit stably | assumed geometry/materials' if partial_functional else 'Shared world | approximate source scale | assumed furniture/material parameters'
                draw.text((12, 4), title, fill=color, font=font)
                draw.text((12, 29), subtitle, fill='#b9c8db', font=font)
                footer = f't={times[index]:.3f}s | sample {index+1}/{len(times)} | playback {speed:g}x'
                if label == 'RECORDED PHYSICS' and 'actuator_force' in z and index:
                    footer += f' | peak motor force/torque {np.max(abs(z["actuator_force"][index-1])):.1f}'
                if label != 'RECORDED PHYSICS':
                    footer += ' | NO DYNAMICS / NOT TASK SUCCESS'
                draw.text((12, 777), footer, fill='#ffcd64', font=font)
                proc.stdin.write(np.asarray(image).tobytes())
                if frame in keyframes:
                    frames.append(image.copy())
                if frame == 0:
                    image.save(output.with_name(output.stem+'_first.jpg'), quality=94)
        proc.stdin.close()
        err = proc.stderr.read().decode(errors='replace')
        if proc.wait() != 0:
            raise RuntimeError(err)
        partial.replace(output)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    sheet = Image.new('RGB', (1920, ((len(frames)+2)//3)*400), '#15191f')
    for j, image in enumerate(frames):
        sheet.paste(image.resize((640, 400)), ((j % 3)*640, (j//3)*400))
    sheetpath = output.with_name(output.stem+'_contactsheet.jpg')
    sheet.save(sheetpath, quality=92)
    probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate',
        '-of', 'json', str(output)], check=True, capture_output=True, text=True).stdout)['streams'][0]
    if int(probe['nb_read_frames']) != len(ids):
        raise ValueError('Video frame count mismatch')
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(output),
                    '-f', 'null', '-'], check=True, capture_output=True)
    result = {'kind': label, 'status': status, 'mujoco_version': mj.__version__,
              'quality': quality if label == 'RECORDED PHYSICS' else {'dynamics': False}, 'video': receipt(output),
              'contactsheet': receipt(sheetpath), 'frames': len(ids),
              'duration_s': float(times[-1]-times[0]), 'presentation_fps': fps,
              'presentation_speed': speed, 'recorded_state_indices': ids.tolist(),
              'camera': {'lookat': look.tolist(), 'distance': float(camera.distance),
                         'azimuth': float(camera.azimuth), 'elevation': float(camera.elevation)},
              'decoded_video_verified': True, 'probe': probe}
    write_json(output.with_suffix('.render.json'), result)
    return result


def export_ego(model, z, output_dir, passed, fps, speed, poses_only, np, mj,
               model_input=None, trace_input=None, quality=None):
    """Rigid virtual camera on each actual neck-yaw body, without a look-at gimbal.

    MjvGLCamera fields are the public installed MuJoCo 3.7 header contract:
    pos/forward/up and perspective frustum. Both GL eyes receive the same pose
    and centered frustum, so the rendered monocular image matches exported K/T.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    quality = quality or {'simulation_passed': passed, 'functional_task_pass': passed,
                          'source_tracking_pass': passed, 'display_status': 'PHYSICS PASS' if passed else 'PHYSICS FAIL'}
    poses_path = output_dir/'two_vita_synthetic_head_cameras.npz'
    metadata_path = output_dir/'two_vita_synthetic_head_cameras.metadata.json'
    outputs = [output_dir/f'p{i}_synthetic_ego.mp4' for i in (1, 2)]
    if poses_path.exists() or metadata_path.exists() or any(p.exists() for p in outputs):
        raise FileExistsError('Use a new ego output directory; existing artifacts are preserved.')
    times = np.asarray(z['time'])
    heads = np.asarray([mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'p{i}_neck_yaw_link') for i in (1, 2)], dtype=int)
    if np.any(heads < 0):
        raise ValueError('Both named neck_yaw_link bodies are required for rigid ego cameras')
    # Vita +X is forward, +Y left, +Z up. The virtual mount sits ahead of the
    # head shell; this is a declared design assumption, not a measured extrinsic.
    mount_translation = np.asarray([.13, 0., .15])
    body_from_camera_rotation = np.asarray([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    body_from_camera = np.eye(4)
    body_from_camera[:3, :3] = body_from_camera_rotation
    body_from_camera[:3, 3] = mount_translation
    world_from_camera = np.repeat(np.eye(4)[None, None], len(times)*2, axis=0).reshape(len(times), 2, 4, 4)
    world_from_mount_body = np.empty_like(world_from_camera)
    camera_pose = np.empty((len(times), 2, 7))
    data = mj.MjData(model)
    for index, qpos in enumerate(z['qpos']):
        data.qpos[:] = qpos
        mj.mj_kinematics(model, data)
        for actor, head in enumerate(heads):
            body = np.eye(4)
            body[:3, :3] = data.xmat[head].reshape(3, 3)
            body[:3, 3] = data.xpos[head]
            world_from_mount_body[index, actor] = body
            transform = body@body_from_camera
            world_from_camera[index, actor] = transform
            camera_pose[index, actor, :3] = transform[:3, 3]
            mj.mju_mat2Quat(camera_pose[index, actor, 3:], transform[:3, :3].ravel())
    width, height, vertical_fov_deg = 1280, 720, 75.
    near, far = .02, 100.
    top = near*np.tan(np.deg2rad(vertical_fov_deg/2))
    focal = height/(2*np.tan(np.deg2rad(vertical_fov_deg/2)))
    K = np.asarray([[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]])
    annotated_K = K.copy()
    annotated_K[1, 2] += 56
    sample = np.arange(times[0], times[-1]+1e-9, speed/fps)
    if not len(sample):
        sample = times[:1]
    ids = np.minimum(np.searchsorted(times, sample, side='left'), len(times)-1)
    before = np.maximum(ids-1, 0)
    ids = np.where(abs(times[before]-sample) < abs(times[ids]-sample), before, ids)
    np.savez_compressed(poses_path,
        state_time_s=times,
        source_time_s=z.get('source_time_s', times),
        actor_names=np.asarray(['p1', 'p2']),
        mount_body_names=np.asarray(['p1_neck_yaw_link', 'p2_neck_yaw_link']),
        body_from_camera=body_from_camera,
        world_from_mount_body=world_from_mount_body,
        world_from_camera=world_from_camera,
        camera_pose_xyz_wxyz=camera_pose,
        virtual_intrinsics=K,
        annotated_video_intrinsics=annotated_K,
        distortion_coefficients=np.zeros(5),
        viewport_width_px=np.asarray(width), viewport_height_px=np.asarray(height),
        annotated_video_content_xywh=np.asarray([0, 56, width, height]),
        near_far_m=np.asarray([near, far]), vertical_fov_deg=np.asarray(vertical_fov_deg),
        video_recorded_state_index=ids, video_state_time_s=times[ids],
        simulation_passed=np.asarray(passed),
        functional_task_pass=np.asarray(quality['functional_task_pass']),
        source_tracking_pass=np.asarray(quality['source_tracking_pass']),
        physical_camera_calibrated=np.asarray(False),
        source_ego_video=np.asarray(False))
    meta = {
        'schema': 'two_vita_synthetic_rigid_head_camera.v1',
        'mujoco_version': mj.__version__,
        'kind': 'actual_physics_state_synthetic_ego',
        'simulation_passed': passed, 'functional_task_pass': quality['functional_task_pass'],
        'source_tracking_pass': quality['source_tracking_pass'],
        'physics_status': quality['display_status'], 'trial_quality': quality,
        'camera_poses': receipt(poses_path),
        'input_frozen_model': receipt(model_input) if model_input else None,
        'input_actual_physics_trace': receipt(trace_input) if trace_input else None,
        'mount_body_names': ['p1_neck_yaw_link', 'p2_neck_yaw_link'],
        'rigid_mount_translation_in_body_m': mount_translation.tolist(),
        'body_from_camera_rotation': body_from_camera_rotation.tolist(),
        'camera_axes': 'OpenCV X right, Y down, Z forward; world right-handed +Z up',
        'extrinsics': 'world_from_camera maps camera coordinates to world; inv(T) maps world into camera',
        'orientation': 'Rigid on actual neck-yaw body; no per-frame object/scene look-at, no gimbal',
        'physical_camera_calibrated': False,
        'source_ego_video': False,
        'mount_provenance': 'Assumed virtual mount in front of head shell; not a measured hardware camera transform',
        'intrinsics_provenance': 'Assumed 75-degree vertical FOV, pinhole, zero distortion; not source-camera calibration',
        'viewport': {'width': width, 'height': height, 'K': K.tolist(), 'near_far_m': [near, far]},
        'annotated_video': {'width': width, 'height': 800, 'content_xywh': [0, 56, width, height], 'K_with_header_offset': annotated_K.tolist()},
        'state_count': len(times), 'video_frames_each': len(ids),
        'presentation_fps': fps, 'presentation_speed': speed,
        'rendered': not poses_only, 'videos': [],
        'limitations': ['Scene has source reconstruction holes and estimated scale.',
                       'These images show the actual recorded physical trial; functional completion and source tracking are separate labels. They are not original egocentric captures.',
                       'The mount is rigid on neck_yaw_link, so it follows that link rather than neck_pitch_link.']}
    if not poses_only:
        from PIL import Image, ImageDraw, ImageFont
        model.vis.global_.offwidth, model.vis.global_.offheight = width, height
        model.vis.headlight.ambient[:] = [.55, .55, .55]
        option = mj.MjvOption()
        option.geomgroup[:] = [0, 1, 1, 1, 1, 1]
        free_camera = mj.MjvCamera()
        free_camera.type = mj.mjtCamera.mjCAMERA_FREE
        free_camera.distance = 1.
        try:
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
        except OSError:
            font = ImageFont.load_default()
        for actor, output in enumerate(outputs):
            partial = output.with_name(output.stem+'.partial.mp4')
            if partial.exists():
                raise FileExistsError(partial)
            proc = subprocess.Popen(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{width}x800', '-r', str(fps),
                '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            snapshots = []
            keys = set(np.linspace(0, len(ids)-1, min(9, len(ids))).round().astype(int))
            max_pose_error = 0.
            try:
                with mj.Renderer(model, height=height, width=width) as renderer:
                    for frame, index in enumerate(ids):
                        data.qpos[:] = z['qpos'][index]
                        data.qvel[:] = z['qvel'][index]
                        data.time = float(times[index])
                        mj.mj_forward(model, data)
                        eye = world_from_camera[index, actor, :3, 3]
                        rotation = world_from_camera[index, actor, :3, :3]
                        forward, up = rotation[:, 2], -rotation[:, 1]
                        free_camera.lookat[:] = eye+forward
                        renderer.update_scene(data, camera=free_camera, scene_option=option)
                        # Public mjvScene.camera[] entries are both set to the
                        # same monocular perspective; no stereo eye offset.
                        for gl_camera in renderer.scene.camera:
                            gl_camera.pos[:] = eye
                            gl_camera.forward[:] = forward
                            gl_camera.up[:] = up
                            gl_camera.orthographic = 0
                            gl_camera.frustum_center = 0.
                            gl_camera.frustum_near = near
                            gl_camera.frustum_far = far
                            gl_camera.frustum_top = top
                            gl_camera.frustum_bottom = -top
                            gl_camera.frustum_width = 2*top*width/height
                            error = max(float(np.linalg.norm(gl_camera.pos-eye)),
                                        float(np.linalg.norm(gl_camera.forward-forward)),
                                        float(np.linalg.norm(gl_camera.up-up)))
                            max_pose_error = max(max_pose_error, error)
                            if error > 2e-5:
                                raise ValueError('Ego GL camera differs from exported rigid pose')
                        canvas = Image.new('RGB', (width, 800), '#15191f')
                        canvas.paste(Image.fromarray(renderer.render()), (0, 56))
                        draw = ImageDraw.Draw(canvas)
                        partial_functional = quality['functional_task_pass'] and not quality['source_tracking_pass']
                        draw.text((12, 4), f'P{actor+1} SYNTHETIC HEAD RGB | '+quality['display_status'], font=font, fill='#ffd68a' if partial_functional else '#b5e9cb' if passed else '#ffb4a2')
                        draw.text((12, 29), 'Assumed camera mount / intrinsics | NOT calibrated source ego | actual trial states', font=font, fill='#b9c8db')
                        draw.text((12, 777), f't={times[index]:.3f}s | recorded state {index+1}/{len(times)} | playback {speed:g}x | neck-yaw rigid mount', font=font, fill='#ffcd64')
                        proc.stdin.write(np.asarray(canvas).tobytes())
                        if frame in keys:
                            snapshots.append(canvas.copy())
                        if frame == 0:
                            canvas.save(output.with_name(output.stem+'_first.jpg'), quality=94)
                proc.stdin.close()
                err = proc.stderr.read().decode(errors='replace')
                if proc.wait() != 0:
                    raise RuntimeError(err)
                partial.replace(output)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
            sheet = Image.new('RGB', (1920, ((len(snapshots)+2)//3)*400), '#15191f')
            for i, image in enumerate(snapshots):
                sheet.paste(image.resize((640, 400)), ((i % 3)*640, (i//3)*400))
            sheet_path = output.with_name(output.stem+'_contactsheet.jpg')
            sheet.save(sheet_path, quality=92)
            probe = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-count_frames',
                '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate',
                '-of', 'json', str(output)], check=True, capture_output=True, text=True).stdout)['streams'][0]
            if int(probe['nb_read_frames']) != len(ids):
                raise ValueError('Ego video frame count differs from selected recorded states')
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True, capture_output=True)
            meta['videos'].append({'actor': actor+1, 'video': receipt(output),
                                   'contactsheet': receipt(sheet_path), 'probe': probe,
                                   'rendered_pose_max_error': max_pose_error,
                                   'decoded_video_verified': True})
    write_json(metadata_path, meta)
    return {'camera_poses': receipt(poses_path), 'metadata': receipt(metadata_path),
            'videos': meta['videos'], 'rendered': not poses_only}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--contacts', type=Path)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--reference-video', type=Path)
    parser.add_argument('--ego-video-dir', type=Path,
                        help='Optional directory for two actual-state synthetic rigid head cameras and RGB videos')
    parser.add_argument('--ego-poses-only', action='store_true',
                        help='With --ego-video-dir, export camera poses/intrinsics/metadata without rendering ego RGB')
    parser.add_argument('--fps', type=float, default=30.)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--skip-video', action='store_true')
    a = parser.parse_args()
    if a.fps <= 0 or a.speed <= 0:
        parser.error('Positive fps and speed required')
    if a.ego_poses_only and a.ego_video_dir is None:
        parser.error('--ego-poses-only requires --ego-video-dir')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco as mj
    import numpy as np
    z = load_numeric(a.trace, np)
    ref = load_numeric(a.reference, np)
    model = mj.MjModel.from_binary_path(str(a.model)) if a.model.suffix == '.mjb' else mj.MjModel.from_xml_path(str(a.model))
    report_path = a.trace.parent/'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    # Functional completion does not change strict source-tracking acceptance.
    quality = classify_report(report)
    passed = quality['simulation_passed']
    times = np.asarray(z['time'], dtype=float)
    qposes = z['qpos']
    if qposes.shape != (len(times), model.nq) or len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError('Invalid recorded state timeline')
    if not np.isfinite(qposes).all() or not np.isfinite(z['qvel']).all():
        raise ValueError('Nonfinite physical state')
    qa = np.asarray(z['joint_qadr'], dtype=int)
    va = np.asarray(z.get('joint_vadr', []), dtype=int)
    aids = np.asarray(z.get('actuator_ids', []), dtype=int)
    names = np.asarray(z['joint_names']).astype(str)
    if qa.shape != (2, 52):
        raise ValueError('Both complete named Vita52 actors required')
    if va.shape != (2, 52):
        va = np.asarray([[model.jnt_dofadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f'p{i+1}_'+name)] for name in names] for i in range(2)])
    if aids.shape != (2, 52):
        aids = np.asarray([[mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f'p{i+1}_motor_'+name) for name in names] for i in range(2)])
    roots = np.asarray(z['robot_root_qadr'], dtype=int)
    n = len(times)
    if len(z['ctrl']) != n-1 or len(z['actuator_force']) != n-1:
        raise ValueError('Control and state intervals differ')
    c = load_numeric(a.contacts, np) if a.contacts else {
        key[len('contact_'):]: value for key, value in z.items() if key.startswith('contact_')}
    for key in ('geom_names', 'body_names'):
        if key in z and key not in c:
            c[key] = z[key]
    payload = {
        'state_time_s': times,
        'source_time_s': z.get('source_time_s', np.interp(times, ref['time_s'], ref.get('source_time_s', ref['time_s']))),
        'control_interval_start_time_s': times[:-1],
        'actor_names': np.asarray(['person_01_to_vita', 'person_02_to_vita']),
        'joint_names_unprefixed': names,
        'root_pose_xyz_wxyz': np.stack([qposes[:, adr:adr+7] for adr in roots], axis=1),
        'joint_position_rad': np.stack([qposes[:, adr] for adr in qa], axis=1),
        'joint_velocity_rad_s': np.stack([z['qvel'][:, adr] for adr in va], axis=1),
        'joint_control_packet': np.stack([z['ctrl'][:, adr] for adr in aids], axis=1),
        'joint_drive_torque_Nm': np.stack([z['actuator_force'][:, adr]*model.actuator_gear[adr, 0] for adr in aids], axis=1),
        'joint_reflected_armature_kg_m2': model.dof_armature[va],
        'body_mass_kg': model.body_mass.copy(),
        'body_inertia_principal_kg_m2': model.body_inertia.copy(),
        'body_inertial_offset_local_m': model.body_ipos.copy(),
        'body_inertial_orientation_wxyz': model.body_iquat.copy(),
        'geom_friction': model.geom_friction.copy(),
        'geom_body_ids': model.geom_bodyid.copy(),
        'body_names': np.asarray([mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) or 'world' for b in range(model.nbody)]),
        # This sample has no independent dynamic object; seats are fixed scene.
        'dynamic_object_names': np.asarray([], dtype='U1'),
        'dynamic_object_pose_xyz_wxyz': np.empty((n, 0, 7)),
        'simulation_passed': np.asarray(passed),
        'functional_task_pass': np.asarray(quality['functional_task_pass']),
        'source_tracking_pass': np.asarray(quality['source_tracking_pass']),
        'source_yup_to_physics_zup': np.asarray([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]),
    }
    data = mj.MjData(model)
    bpos, bquat = [], []
    for q in qposes:
        data.qpos[:] = q
        mj.mj_kinematics(model, data)
        bpos.append(data.xpos.copy())
        bquat.append(data.xquat.copy())
    payload['body_position_world_m'] = np.asarray(bpos)
    payload['body_orientation_wxyz'] = np.asarray(bquat)
    for key, value in c.items():
        payload['collision_'+key] = value
    if 'interval_index' in c and 'body_ids' in c:
        payload['contact_graph_interval_index'] = c['interval_index']
        payload['contact_graph_body_ids'] = c['body_ids']
    for key, value in z.items():
        if key.startswith(('root_', 'joint_rms', 'seat_', 'foot_', 'max_penetration', 'qfrc_constraint', 'qfrc_applied', 'xfrc_applied')):
            payload['physics_'+key] = value
    for key, value in ref.items():
        payload['reference_'+key] = value
    source_inputs = {}
    for actor in (1, 2):
        p = a.data/f'human/person_{actor:02}/smplx_params_world.npz'
        if p.exists():
            source_inputs[f'person{actor}_smplx'] = receipt(p)
            for key, value in load_numeric(p, np).items():
                payload[f'source_yup_person{actor}_smplx_'+key] = value
    for stem in ('timeline', 'cameras_world', 'coordinates'):
        p = a.data/(stem+'.npz')
        if p.exists():
            source_inputs[stem] = receipt(p)
            for key, value in load_numeric(p, np).items():
                payload['source_yup_'+stem+'_'+key] = value
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out = a.output_dir/'hoi_motion_physics.npz'
    if out.exists():
        raise FileExistsError(out)
    np.savez_compressed(out, **payload)
    contact_count = int(len(c.get('interval_index', [])))
    meta = {
        'schema': 'two_sitting_dual_vita_hoi.v1',
        'mujoco_version': mj.__version__,
        'status': quality['status'],
        'simulation_passed': passed, 'functional_task_pass': quality['functional_task_pass'],
        'source_tracking_pass': quality['source_tracking_pass'], 'trial_quality': quality,
        'state_frames': n, 'physics_intervals': n-1, 'contact_points': contact_count,
        'physics_duration_s': float(times[-1]-times[0]), 'actors': 2, 'dofs_per_actor': 52,
        'coordinates': 'right-handed +Z up, approximate metres', 'quaternion_order': 'wxyz',
        'source_coordinates': 'arrays prefixed source_yup retain original right-handed +Y up',
        'control_packet_units': 'controller dependent: position rad or torque Nm; see trial controller report',
        'collision_wrench_convention': 'world frame force/torque on geom2 at contact point; negate for geom1',
        'contact_graph': 'one graph edge per contact point and integration interval; ids index names arrays',
        'timing': 'N+1 states for N controls/contact intervals; state i+1 follows control interval i',
        'scene': 'fixed source scene and inferred furniture collision proxies; no dynamic object track',
        'physical_parameters': 'robot inertias from source Vita asset; furniture collision and friction assumed',
        'force_provenance': 'MuJoCo simulated forces and drive torques; not measured ground truth',
        'source_hand_status': 'default SMPL-X hand pose; no observed MANO reconstruction',
        'source_inputs': source_inputs, 'input_model': receipt(a.model), 'input_trace': receipt(a.trace),
        'input_reference': receipt(a.reference), 'trial_report': report,
        'dataset': receipt(out),
    }
    write_json(a.output_dir/'hoi_motion_physics.metadata.json', meta)
    videos = {}
    if not a.skip_video:
        videos['physics'] = render(model, z, qposes, times, a.video, 'RECORDED PHYSICS', passed, a.fps, a.speed, np, mj, quality)
    if a.reference_video:
        rq = np.asarray(ref['qpos'])
        rtimes = np.asarray(ref['time_s'])
        fullq = np.repeat(model.qpos0[None], len(rtimes), axis=0)
        for actor in range(2):
            fullq[:, roots[actor]:roots[actor]+7] = rq[:, actor, :7]
            fullq[:, qa[actor]] = rq[:, actor, 7:]
        videos['kinematic_reference'] = render(model, z, fullq, rtimes, a.reference_video, 'KINEMATIC REFERENCE', False, a.fps, a.speed, np, mj)
    ego = None
    if a.ego_video_dir:
        ego = export_ego(model, z, a.ego_video_dir, passed, a.fps, a.speed,
                         a.ego_poses_only, np, mj, a.model, a.trace, quality)
    result = {'dataset': receipt(out), 'metadata': receipt(a.output_dir/'hoi_motion_physics.metadata.json'),
              'simulation_passed': passed, 'functional_task_pass': quality['functional_task_pass'],
              'source_tracking_pass': quality['source_tracking_pass'],
              'display_status': quality['display_status'], 'videos': videos, 'synthetic_ego': ego}
    write_json(a.output_dir/'delivery_manifest.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
