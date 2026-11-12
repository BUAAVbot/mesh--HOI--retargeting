"""Render the supplied Vita52 clips as kinematic replays, without simulation.

Run on the server: CUDA_VISIBLE_DEVICES=7 MUJOCO_GL=egl python this_file.py.
Only mj_forward is used. All source joints/frames are retained; no policies,
contacts, props, interpolation, joint clipping, or per-frame root corrections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def slug(text):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('_') or 'motion'


def load_robot(urdf, mujoco, np, plane_z):
    """Import URDF visuals in memory; never rewrite a manufacturer asset."""
    xml = ET.parse(urdf).getroot()
    children = {j.find('child').get('link') for j in xml.findall('joint')}
    roots = [l.get('name') for l in xml.findall('link') if l.get('name') not in children]
    if len(roots) != 1:
        raise ValueError(f'Expected one URDF root, found {roots}')
    source_names = [j.get('name') for j in xml.findall('joint') if j.get('type') != 'fixed']
    if len(source_names) != 52:
        raise ValueError(f'Expected Vita52, found {len(source_names)} movable joints')
    meshes = {}
    for mesh in xml.findall('.//mesh'):
        name = mesh.get('filename')
        if name.startswith('package://'):
            package, relative = name[len('package://'):].split('/', 1)
            path = urdf.parent.parent/relative
        else:
            path = (urdf.parent/name).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        meshes[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        mesh.set('filename', str(path))
    extension = xml.find('mujoco')
    if extension is None:
        extension = ET.SubElement(xml, 'mujoco')
    compiler = extension.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(extension, 'compiler')
    compiler.attrib.update(discardvisual='false', fusestatic='false', strippath='false',
                           balanceinertia='true', boundmass='0.000001', boundinertia='0.000000001')
    spec = mujoco.MjSpec.from_string(ET.tostring(xml, encoding='unicode'))
    base = spec.body(roots[0])
    base.add_freejoint(name='vita_replay_free_root')
    # The renderer has no dynamics or physical-contact interpretation.
    for geom in spec.geoms:
        geom.contype = 0
        geom.conaffinity = 0
    spec.worldbody.add_geom(name='declared_reference_plane', type=mujoco.mjtGeom.mjGEOM_PLANE,
                           pos=[0, 0, plane_z], size=[0, 0, .02], rgba=[.20, .23, .27, 1],
                           contype=0, conaffinity=0, group=2)
    spec.worldbody.add_light(name='replay_key', pos=[2, -2, 4], dir=[-1, 1, -2],
                             diffuse=[.85, .85, .85], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.worldbody.add_light(name='replay_fill', pos=[-2, 1, 3], dir=[1, -1, -1],
                             diffuse=[.50, .50, .55], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.visual.global_.offwidth = 640
    spec.visual.global_.offheight = 800
    spec.visual.global_.fovy = 42
    spec.visual.headlight.ambient = [.45, .45, .45]
    spec.visual.headlight.diffuse = [.8, .8, .8]
    spec.option.disableflags = int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    model = spec.compile()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'vita_replay_free_root')
    if model.nq != 59 or model.nv != 58 or jid < 0:
        raise RuntimeError(f'Unexpected imported model: nq={model.nq},nv={model.nv}')
    addresses = {name: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]) for name in source_names}
    if len(set(addresses.values())) != 52:
        raise RuntimeError('Joint address mapping is not one-to-one')
    # URDF importer puts visuals in group1 and collision meshes in group0.
    visuals = np.flatnonzero(model.geom_group == 1)
    if len(visuals) < 50:
        raise RuntimeError(f'Visuals were discarded: only {len(visuals)} group1 geoms')
    hand_bodies = {}
    hand_geom_ids = {}
    for side, prefix in [('left', 'lh_'), ('right', 'rh_')]:
        hand_bodies[side] = [i for i in range(1, model.nbody) if
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '').startswith(prefix)]
        if len(hand_bodies[side]) < 5:
            raise RuntimeError(f'Missing {side} finger bodies')
        hand_geom_ids[side] = np.asarray([g for g in visuals if model.geom_bodyid[g] in hand_bodies[side]], dtype=int)
        model.geom_group[hand_geom_ids[side]] = 3 if side == 'left' else 4
    return model, addresses, int(model.jnt_qposadr[jid]), visuals, hand_geom_ids, {
        'urdf': str(urdf), 'urdf_sha256': digest(urdf), 'mesh_assets': meshes,
        'source_root_link': roots[0], 'source_movable_joint_count': len(source_names),
        'compiled_nq': model.nq, 'compiled_nv': model.nv, 'visual_geom_count': len(visuals),
        'render_only_inertia_regularization': {'boundmass_kg': 1e-6, 'boundinertia_kg_m2': 1e-9, 'balanceinertia': True},
        'physics_simulation': False, 'physics_steps': 0, 'collision_display': False,
        'isolated_hand_visual_geom_ids': {s: ids.tolist() for s, ids in hand_geom_ids.items()},
        'isolated_hand_visual_groups': {'left': 3, 'right': 4}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/vita_demo_20261026')
    parser.add_argument('--urdf', type=Path)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/previews')
    parser.add_argument('--report-dir', type=Path, default=ROOT/'reports/vita_motion_demos')
    parser.add_argument('--select', default='*', help='Match source NPZ filename stem')
    parser.add_argument('--limit-clips', type=int, default=0, help='0 means all selected clips; never limits frames')
    parser.add_argument('--gpu', type=int, choices=(6, 7), default=7)
    parser.add_argument('--plane-z', type=float, default=0., help='Declared visual reference plane only')
    parser.add_argument('--root-offset', type=float, nargs=3, default=(0., 0., 0.), help='One constant translation for every source frame')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--preview-only', action='store_true', help='Render one JPEG per selected clip; never claim a complete video')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, str(args.gpu)):
        raise RuntimeError('CUDA_VISIBLE_DEVICES disagrees with reserved physical GPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['MUJOCO_GL'] = 'egl'
    # EGL enumerates physical devices independently of CUDA_VISIBLE_DEVICES.
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(args.gpu)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import fnmatch
    import numpy as np
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg is required for lossless frame scheduling')
    urdf = args.urdf or args.data_dir/'robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf'
    candidates = sorted(args.data_dir.rglob('*.npz'))
    clips = [(i+1, p) for i, p in enumerate(candidates) if fnmatch.fnmatch(p.stem, args.select)]
    if args.limit_clips:
        clips = clips[:args.limit_clips]
    if not clips:
        raise FileNotFoundError(f'No NPZ clips under {args.data_dir}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    model, addresses, freeadr, robot_geoms, hand_geoms, robot_report = load_robot(urdf, mujoco, np, args.plane_z)
    data = mujoco.MjData(model)
    hand_base_ids = {side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix+'hand_base_link')
                     for side, prefix in [('left', 'lh_'), ('right', 'rh_')]}
    option = mujoco.MjvOption()
    option.geomgroup[:] = 0
    option.geomgroup[1:5] = 1
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    hand_options = {}
    for side, group in [('left', 3), ('right', 4)]:
        hand_options[side] = mujoco.MjvOption()
        hand_options[side].geomgroup[:] = 0
        hand_options[side].geomgroup[group] = 1
    full_renderer = mujoco.Renderer(model, height=704, width=440)
    hand_renderer = mujoco.Renderer(model, height=352, width=400)
    fontpath = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(fontpath, 20) if Path(fontpath).exists() else ImageFont.load_default()
    small = ImageFont.truetype(fontpath, 16) if Path(fontpath).exists() else font
    offset = np.asarray(args.root_offset, dtype=float)
    manifest = {'mode': 'kinematic_replay', 'policy_inference': False, 'simulation_steps': 0,
        'mujoco_version': mujoco.__version__, 'physical_gpu': args.gpu,
        'root_offset_xyz_m': offset.tolist(), 'reference_plane_z_m': args.plane_z,
        'plane_is_a_visual_reference_not_a_contact_result': True, 'robot': robot_report,
        'script_sha256': digest(__file__), 'clips': []}

    def set_frame(clip, frame, qadr):
        data.qpos[freeadr:freeadr+3] = clip['root_pos'][frame]+offset
        xyzw = clip['root_rot_xyzw'][frame]
        data.qpos[freeadr+3:freeadr+7] = xyzw[[3, 0, 1, 2]]
        data.qpos[qadr] = clip['dof_pos'][frame]
        mujoco.mj_forward(model, data)
        if not np.array_equal(data.qpos[qadr], clip['dof_pos'][frame]):
            raise RuntimeError('Forward evaluation altered source joint values')

    def bounds(geoms=robot_geoms):
        centers = data.geom_xpos[geoms]
        rotations = data.geom_xmat[geoms].reshape(-1, 3, 3)
        aabb = model.geom_aabb[geoms]
        centers = centers+np.einsum('nij,nj->ni', rotations, aabb[:, :3])
        extents = np.einsum('nij,nj->ni', np.abs(rotations), aabb[:, 3:])
        return (centers-extents).min(0), (centers+extents).max(0)

    def camera(target, distance, azimuth, elevation):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = target
        cam.distance = distance
        cam.azimuth = azimuth
        cam.elevation = elevation
        return cam

    try:
        for number, source in clips:
            started = time.monotonic()
            # Materialize once: NpzFile indexing otherwise decompresses the full
            # trajectory again on every rendered frame.
            with np.load(source, allow_pickle=False) as bundle:
                clip = {key: bundle[key] for key in bundle.files}
            needed = ('root_pos', 'root_rot_xyzw', 'dof_pos', 'dof_names', 'fps')
            if any(k not in clip for k in needed):
                raise ValueError(f'Missing motion fields in {source}')
            n = len(clip['root_pos'])
            names = [str(x) for x in clip['dof_names']]
            if len(names) != 52 or len(set(names)) != 52 or set(names) != set(addresses):
                raise ValueError('NPZ joint names do not match all 52 URDF joints')
            if clip['root_pos'].shape != (n, 3) or clip['root_rot_xyzw'].shape != (n, 4) or clip['dof_pos'].shape != (n, 52):
                raise ValueError('Motion tensor shapes are inconsistent')
            fps = float(clip['fps'])
            if n < 1 or not math.isclose(fps, 30., abs_tol=1e-6):
                raise ValueError(f'Expected nonempty original 30FPS motion, got T={n},fps={fps}')
            if any(not np.isfinite(clip[k]).all() for k in ('root_pos', 'root_rot_xyzw', 'dof_pos')):
                raise ValueError('Nonfinite input motion')
            norm_error = float(np.max(abs(np.linalg.norm(clip['root_rot_xyzw'], axis=1)-1)))
            if norm_error > .001:
                raise ValueError(f'Invalid source quaternion norms: {norm_error}')
            qadr = np.asarray([addresses[x] for x in names], dtype=int)
            name = f'vita17_{number:02d}_{slug(source.stem)}'
            video = args.output_dir/(name+'.mp4')
            reportpath = args.report_dir/(name+'.json')
            if (video.exists() or reportpath.exists()) and not args.replace and not args.preview_only:
                raise FileExistsError(f'{name} exists; use --replace explicitly')
            # All source frames determine a fixed per-clip view distance; cameras
            # follow the moving body without changing source motion coordinates.
            extent = np.zeros(3)
            for frame in range(n):
                set_frame(clip, frame, qadr)
                low, high = bounds()
                extent = np.maximum(extent, high-low)
            fovy = math.radians(float(model.vis.global_.fovy))
            distance = max(2.1, 1.12*max(extent[2]/(2*math.tan(fovy/2)),
                           np.linalg.norm(extent[:2])/(2*math.tan(fovy/2)*(440/704))))
            source_sha = digest(source)
            partial = video.with_suffix('.partial.mp4')
            command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                '-pix_fmt', 'rgb24', '-s:v', '1280x800', '-r', str(fps), '-i', 'pipe:0', '-an',
                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p',
                '-metadata', 'comment=Kinematic Vita52 replay; no policy or physics simulation; source '+source_sha,
                '-movflags', '+faststart', str(partial)]
            encoder = None if args.preview_only else subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            keyindices = set(np.linspace(0, n-1, min(9, n)).round().astype(int).tolist())
            snapshots = []
            try:
                for frame in range(1 if args.preview_only else n):
                    set_frame(clip, frame, qadr)
                    low, high = bounds()
                    target = (low+high)/2
                    w, x, y, z = data.qpos[freeadr+3:freeadr+7]
                    heading = math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
                    canvas = Image.new('RGB', (1280, 800), (21, 25, 31))
                    draw = ImageDraw.Draw(canvas)
                    for view, az in enumerate((heading+45, heading-45)):
                        full_renderer.update_scene(data, camera=camera(target, distance, az, -12), scene_option=option)
                        canvas.paste(Image.fromarray(full_renderer.render()), (view*440, 56))
                    for side_index, side in enumerate(('left', 'right')):
                        hand_low, hand_high = bounds(hand_geoms[side])
                        hand_target = (hand_low+hand_high)/2
                        hand_distance = max(.18, 1.2*np.linalg.norm(hand_high-hand_low)/(2*math.tan(fovy/2)))
                        # Look perpendicular to the palm, not along a common
                        # body-frame camera that makes one hand edge-on.
                        normal = data.xmat[hand_base_ids[side]].reshape(3, 3)[:, 0]
                        hand_azimuth = math.degrees(math.atan2(normal[1], normal[0]))
                        hand_elevation = -math.degrees(math.asin(float(np.clip(normal[2], -1, 1))))
                        hand_renderer.update_scene(data, camera=camera(hand_target, hand_distance, hand_azimuth, hand_elevation), scene_option=hand_options[side])
                        canvas.paste(Image.fromarray(hand_renderer.render()), (880, 56+352*side_index))
                    draw.text((16, 9), f'Vita52 | {number:02d} {source.stem[:61]}', font=font, fill=(240, 245, 250))
                    draw.text((12, 62), 'FRONT LEFT', font=small, fill='white')
                    draw.text((452, 62), 'FRONT RIGHT', font=small, fill='white')
                    draw.text((892, 62), 'LEFT: ISOLATED HAND VIEW', font=small, fill='white')
                    draw.text((892, 414), 'RIGHT: ISOLATED HAND VIEW', font=small, fill='white')
                    draw.rectangle((0, 760, 1280, 800), fill=(21, 25, 31))
                    draw.text((12, 770), f'KINEMATIC REPLAY | 52 joints + free root | {frame+1}/{n} | {frame/fps:.2f}s | {fps:g} FPS', font=small, fill=(255, 205, 100))
                    if args.preview_only:
                        previewpath = args.output_dir/(name+'_camera_preview.jpg')
                        canvas.save(previewpath, quality=94)
                        print(json.dumps({'single_frame_preview': str(previewpath), 'source_frame_index': 0}), flush=True)
                        continue
                    encoder.stdin.write(np.asarray(canvas).tobytes())
                    if frame == 0:
                        firstpath = args.output_dir/(name+'_first.jpg')
                        canvas.save(firstpath, quality=94)
                        print(json.dumps({'first_jpeg': str(firstpath), 'mode': 'kinematic_replay'}), flush=True)
                    if frame in keyindices:
                        snapshots.append((frame, canvas.copy()))
                    if frame % 150 == 0:
                        print(json.dumps({'clip': source.stem, 'frame': frame+1, 'total_frames': n}), flush=True)
                if args.preview_only:
                    continue
                encoder.stdin.close()
                error = encoder.stderr.read().decode(errors='replace')
                if encoder.wait() != 0:
                    raise RuntimeError(error)
            except BaseException:
                if encoder is not None and encoder.poll() is None:
                    encoder.kill()
                    encoder.wait()
                raise
            partial.replace(video)
            # Preserve all frames, dimensions and cadence while bounding delivery
            # files for the existing <=16MiB verified transfer helper.
            delivery_crf = 20
            for crf in (24, 28, 32):
                if video.stat().st_size <= 16*1024*1024:
                    break
                reduced = video.with_suffix('.delivery.mp4')
                subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(video), '-map', '0:v:0', '-map_metadata', '0', '-an', '-c:v', 'libx264',
                    '-preset', 'veryfast', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(reduced)], check=True)
                reduced.replace(video)
                delivery_crf = crf
            if video.stat().st_size > 16*1024*1024:
                raise RuntimeError('Delivery video exceeds 16MiB after finite encoding retries')
            sheet = Image.new('RGB', (1920, math.ceil(len(snapshots)/3)*400), '#15191f')
            for index, (frame, image) in enumerate(snapshots):
                sheet.paste(image.resize((640, 400)), ((index % 3)*640, (index//3)*400))
            sheetpath = args.output_dir/(name+'_contactsheet.jpg')
            sheet.save(sheetpath, quality=92)
            for tag, (frame, image) in zip(('first', 'middle', 'last'), (snapshots[0], snapshots[len(snapshots)//2], snapshots[-1])):
                image.save(args.output_dir/(name+'_'+tag+'.jpg'), quality=94)
            probe = subprocess.run([shutil.which('ffprobe') or 'ffprobe', '-v', 'error', '-count_frames',
                '-select_streams', 'v:0', '-show_entries', 'stream=nb_read_frames,r_frame_rate,width,height',
                '-of', 'json', str(video)], capture_output=True, text=True, check=True)
            stream = json.loads(probe.stdout)['streams'][0]
            if int(stream['nb_read_frames']) != n:
                raise RuntimeError('Encoded frame count does not match the complete input')
            report = {'mode': 'kinematic_replay', 'source': str(source), 'source_sha256': source_sha,
                'urdf_sha256': robot_report['urdf_sha256'], 'joint_names_in_source_order': names,
                'joint_qpos_addresses_in_source_order': qadr.tolist(), 'source_frames': n, 'rendered_frames': n,
                'source_fps': fps, 'encoded_video_probe': stream, 'all_source_joint_values_preserved': True,
                'root_translation_constant_offset_xyz_m': offset.tolist(), 'root_xyzw_reordered_to_wxyz': True,
                'source_quaternion_norm_max_error': norm_error, 'interpolation': False, 'joint_clipping': False,
                'reference_plane_z_m': args.plane_z, 'no_props_or_contact_markers': True,
                'policy_inference': False, 'physics_steps': 0, 'camera_distance_m': distance,
                'hand_camera': 'Dynamic geometric AABB center, diagonal-fit distance, view perpendicular to hand-base local X, isolated same-side visual group only',
                'fullbody_views_preserve_complete_scene_occlusion': True,
                'hand_views_are_explicitly_isolated': True,
                'video': str(video), 'video_sha256': digest(video),
                'contactsheet': str(sheetpath), 'contactsheet_sha256': digest(sheetpath),
                'contactsheet_source_frame_indices': [int(f) for f, _ in snapshots],
                'video_bytes': video.stat().st_size, 'video_delivery_crf': delivery_crf,
                'video_delivery_max_bytes': 16*1024*1024,
                'elapsed_seconds': time.monotonic()-started}
            report['source_pickle_bindings'] = {key: str(value.reshape(-1)[0].item()) for key, value in clip.items()
                if 'pickle' in key.lower() and 'sha' in key.lower() and value.size == 1}
            atomic_json(reportpath, report)
            manifest['clips'].append(report)
            atomic_json(args.report_dir/'manifest.json', manifest)
            print(json.dumps({'completed': name, 'frames': n, 'seconds': report['elapsed_seconds'], 'video': str(video)}), flush=True)
    finally:
        full_renderer.close()
        hand_renderer.close()


if __name__ == '__main__':
    main()
