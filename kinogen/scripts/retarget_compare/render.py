"""Render full-frame kinematic comparisons with one fixed camera and no rollout."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', '3')


def encode(path, width, height, fps=30):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(['ffmpeg', '-nostdin', '-y', '-hide_banner', '-loglevel', 'error',
                             '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
                             '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                             '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                             str(path)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def close_encoder(process):
    process.stdin.close()
    error = process.stderr.read().decode(errors='replace')
    if process.wait() != 0:
        raise RuntimeError(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/KinoGen/outputs/retarget_compare_20261028'))
    parser.add_argument('--view', choices=['overview', 'side'], default='overview')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--tag', default='comparison')
    args = parser.parse_args()
    import numpy as np
    import mujoco as mj
    from PIL import Image, ImageDraw, ImageFont
    try:
        from retarget_compare_evaluate import BODY_MAP, METHODS, find_reference, load_reference
    except ModuleNotFoundError:
        from evaluate import BODY_MAP, METHODS, find_reference, load_reference

    protocol = json.loads((args.root / 'shared_input/source_manifest.json').read_text())
    model_path = Path(protocol['model']).parent / 'two_vita_solver20.xml'
    model = mj.MjModel.from_xml_path(str(model_path))
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 600
    model.vis.headlight.ambient[:] = [.65, .65, .65]
    model.vis.headlight.diffuse[:] = [.7, .7, .7]
    data = mj.MjData(model)
    with np.load(args.root / 'shared_input/common_observations.npz', allow_pickle=False) as archive:
        source = archive['repaired_source_joints_zup'].copy()
    times = np.arange(172) / 30.
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-.2, 1.3, .75]
    camera.distance = 4.2
    camera.azimuth = 100. if args.view == 'overview' else 15.
    camera.elevation = -21.
    option = mj.MjvOption()
    option.geomgroup[:] = [0, 1, 1, 0, 0, 0]
    colors = [[.22, .55, .93, 1.], [.96, .48, .18, 1.]]
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or '' for g in range(model.ngeom)]
    owners = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or '' for g in range(model.ngeom)]
    original_groups = model.geom_group.copy()
    for g, name in enumerate(names):
        if name == 'ground_proxy':
            model.geom_group[g] = 2
        if owners[g].startswith(('p1_', 'p2_')) and model.geom_contype[g] == 0:
            model.geom_group[g] = 1
            model.geom_matid[g] = -1
            model.geom_rgba[g] = colors[0 if owners[g].startswith('p1_') else 1]
    visible_groups = model.geom_group.copy()
    root_q = []
    for actor in (1, 2):
        body = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, f'p{actor}_base_link')
        joint = next(j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE and model.jnt_bodyid[j] == body)
        root_q.append(int(model.jnt_qposadr[joint]))
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 24)
    small = ImageFont.truetype(font_path, 19)
    output = args.root / 'media' / args.view
    output.mkdir(parents=True, exist_ok=True)
    camera_record = {'lookat': camera.lookat.tolist(), 'distance': camera.distance,
                     'azimuth': camera.azimuth, 'elevation': camera.elevation}
    methods = ['source'] + [method for method in METHODS if find_reference(args.root, method)]
    if args.only:
        methods = [method for method in methods if method in args.only]
    else:
        methods = [method for method in methods if not method.startswith('umr_')]
    skeleton = [(0, 1), (0, 2), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11),
                (0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (9, 13), (9, 14),
                (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]
    keyframes = [0, 50, 100, 171]
    produced = []
    for method in methods:
        reference = find_reference(args.root, method) if method != 'source' else None
        reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest() if reference else hashlib.sha256(source.tobytes()).hexdigest()
        video = output / f'{method}.mp4'
        metadata = output / f'{method}.render.json'
        if video.exists() and metadata.exists():
            old = json.loads(metadata.read_text())
            if old.get('reference_sha256') == reference_hash and old.get('camera') == camera_record and old.get('render_version') == 2:
                produced.append({'method': method, 'video': str(video)})
                continue
        qposes = load_reference(reference, protocol['dof_names'], np)[0] if reference else None
        model.geom_group[:] = visible_groups
        if method == 'source':
            for g, name in enumerate(names):
                if owners[g].startswith(('p1_', 'p2_')):
                    model.geom_group[g] = 4
        title = 'SOURCE: reconstructed SMPL-X joints' if method == 'source' else METHODS[method]
        frames = []
        process = encode(video, 960, 704)
        with mj.Renderer(model, height=600, width=960) as renderer:
            for frame in range(172):
                if qposes is not None:
                    for actor, address in enumerate(root_q):
                        data.qpos[address:address + 59] = qposes[frame, actor]
                mj.mj_kinematics(model, data)
                mj.mj_comPos(model, data)
                renderer.update_scene(data, camera=camera, scene_option=option)
                if method == 'source':
                    for actor in (0, 1):
                        for begin, end in skeleton:
                            geom = renderer.scene.geoms[renderer.scene.ngeom]
                            mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(colors[actor], np.float32))
                            mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_CAPSULE, .025,
                                             source[frame, actor, begin], source[frame, actor, end])
                            renderer.scene.ngeom += 1
                picture = Image.fromarray(renderer.render())
                canvas = Image.new('RGB', (960, 704), '#101b2a')
                canvas.paste(picture, (0, 64))
                draw = ImageDraw.Draw(canvas)
                draw.text((16, 8), title, font=font, fill='white')
                draw.text((16, 38), 'KINEMATIC ONLY | same Vita52 / scene / camera | original30Hz', font=small, fill='#b7cadb')
                footer = f'frame {frame:03d}/171   t={times[frame]:.3f}s   blue=P1 / orange=P2'
                if 63 <= frame <= 69:
                    footer += '   P2: interpolated source'
                draw.text((16, 674), footer, font=small, fill='#ffce80' if 63 <= frame <= 69 else 'white')
                process.stdin.write(np.asarray(canvas).tobytes())
                if frame in keyframes:
                    canvas.save(output / f'{method}_f{frame:03d}.jpg', quality=94)
        close_encoder(process)
        metadata.write_text(json.dumps({'method': method, 'reference': str(reference) if reference else None,
                            'reference_sha256': reference_hash, 'camera': camera_record, 'frames': 172, 'render_version': 2,
                            'fps': 30, 'last_pts_s': 5.7, 'physical_rollout': False,
                            'presentation': 'Full original172frames, no interpolated robot rendering or success cropping.'}, indent=2) + '\n')
        produced.append({'method': method, 'video': str(video)})
        print(json.dumps(produced[-1]), flush=True)
    # Every pane decodes the same172 frames. Source is3D observations, not simulation.
    columns = min(4, max(1, len(produced)))
    rows = math.ceil(len(produced) / columns)
    width, height = columns * 480, rows * 352 + 64
    grid_path = output / f'{args.tag}.mp4'
    decoders = [subprocess.Popen(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', item['video'],
                                 '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE) for item in produced]
    encoder = encode(grid_path, width, height)
    for frame in range(172):
        canvas = Image.new('RGB', (width, height), '#0b1522')
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 8), 'Two-person seat exchange | RETARGETING COMPARISON', font=font, fill='white')
        draw.text((16, 38), 'Kinematic references, not physical-policy success | unchanged scene and original timing', font=small, fill='#b7cadb')
        for index, decoder in enumerate(decoders):
            data_bytes = decoder.stdout.read(960 * 704 * 3)
            if len(data_bytes) != 960 * 704 * 3:
                raise RuntimeError('Incomplete comparison pane')
            image = Image.frombytes('RGB', (960, 704), data_bytes).resize((480, 352), Image.Resampling.LANCZOS)
            canvas.paste(image, ((index % columns) * 480, 64 + (index // columns) * 352))
        encoder.stdin.write(np.asarray(canvas).tobytes())
        if frame == 171:
            canvas.save(output / f'{args.tag}_final_frame.jpg', quality=96)
    close_encoder(encoder)
    for decoder in decoders:
        decoder.stdout.close()
        error = decoder.stderr.read()
        if decoder.wait() != 0:
            raise RuntimeError(error.decode(errors='replace'))
    sheet = Image.new('RGB', (4 * 480, len(produced) * 352 + 50), '#0b1522')
    ImageDraw.Draw(sheet).text((16, 12), 'Same source frames: 0 / 50 / 100 / 171 | kinematic comparison', font=font, fill='white')
    for row, item in enumerate(produced):
        for column, frame in enumerate(keyframes):
            image = Image.open(output / f"{item['method']}_f{frame:03d}.jpg").resize((480, 352), Image.Resampling.LANCZOS)
            sheet.paste(image, (column * 480, 50 + row * 352))
    sheet.save(output / f'{args.tag}_contactsheet.jpg', quality=95)
    (output / f'{args.tag}.manifest.json').write_text(json.dumps({'view': args.view, 'camera': camera_record, 'methods': produced,
                                                     'comparison': str(grid_path), 'frames': 172, 'physics_success_claimed': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
