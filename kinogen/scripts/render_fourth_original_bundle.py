"""Render the reviewed original handoff visual replay, without dynamics or retargeting."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    # These two source files were read and reviewed before this import.
    sys.path.insert(0, str(a.data.resolve()/'scripts'))
    from replay_mujoco import MujocoReplay
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    replay = MujocoReplay(a.data)
    replay.model.vis.global_.offwidth = 1280
    replay.model.vis.global_.offheight = 720
    count = len(replay.source.times)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    partial = a.output.with_suffix('.partial.mp4')
    encoder = subprocess.Popen([
        'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', '1280x800', '-r', '30',
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '20', '-preset', 'veryfast',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(partial)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    snapshots = []
    keys = set(np.linspace(0, count-1, 9).round().astype(int))
    try:
        with mujoco.Renderer(replay.model, height=720, width=1280) as renderer:
            for frame in range(count):
                replay.set_frame(frame)
                mujoco.mjr_uploadMesh(replay.model, renderer._mjr_context, replay.body_mesh)
                renderer.update_scene(replay.data, camera='capture')
                canvas = Image.new('RGB', (1280, 800), '#15191f')
                canvas.paste(Image.fromarray(renderer.render()), (0, 56))
                draw = ImageDraw.Draw(canvas)
                draw.text((12, 5), 'ORIGINAL fourth_sim_handoff | HUMAN + SIX OBJECTS + RECONSTRUCTED SCENE', font=font, fill='white')
                draw.text((12, 30), 'Visual data replay | source capture view | nominal 30 FPS | no physics / no robot retargeting', font=font, fill='#b9c8db')
                valid_human = int(replay.source.body_rows[frame]) >= 0
                object_count = int(np.sum(replay.source.valid[frame]))
                draw.text((12, 778), f'Source frame {frame}/{count-1} | PTS {replay.source.times[frame]:.3f}s | human {"valid" if valid_human else "MISSING"} | visible objects {object_count}/6 | invalid data hidden', font=font, fill='#ffcd64')
                encoder.stdin.write(np.asarray(canvas).tobytes())
                if frame in keys:
                    snapshots.append(canvas.copy())
                if frame == 570:
                    canvas.save(a.output.with_name(a.output.stem+'_frame570.jpg'), quality=94)
        encoder.stdin.close()
        error = encoder.stderr.read().decode(errors='replace')
        if encoder.wait() != 0:
            raise RuntimeError(error)
        partial.replace(a.output)
    finally:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
    sheet = Image.new('RGB', (1920, 1200), '#15191f')
    for i, snap in enumerate(snapshots):
        sheet.paste(snap.resize((640, 400)), ((i % 3)*640, (i // 3)*400))
    sheet_path = a.output.with_name(a.output.stem+'_contactsheet.jpg')
    sheet.save(sheet_path, quality=92)
    probe = json.loads(subprocess.run([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_read_frames,width,height,r_frame_rate',
        '-of', 'json', str(a.output)], check=True, capture_output=True, text=True).stdout)['streams'][0]
    if int(probe['nb_read_frames']) != count:
        raise ValueError('Encoded video frame count differs from source frame count')
    report = {'mode':'original_bundle_visual_replay', 'physics_enabled':False,
              'robot_retargeting':False, 'source_frames':count,
              'human_valid_frames':int(np.sum(replay.source.body_rows >= 0)),
              'human_missing_frames':int(np.sum(replay.source.body_rows < 0)),
              'object_valid_frames':np.sum(replay.source.valid, axis=0).tolist(),
              'source_last_pts_s':float(replay.source.times[-1]),
              'presentation':'one frame per source frame, nominal 30 FPS; original PTS displayed in each frame',
              'replay_camera':'original fixed capture camera from package MJCF',
              'source_model_sha256':sha(a.data/'mujoco/replay.xml'),
              'source_timeline_sha256':sha(a.data/'timeline.npz'),
              'reviewed_replay_script_sha256':sha(a.data/'scripts/replay_mujoco.py'),
              'video':str(a.output.resolve()), 'video_sha256':sha(a.output),
              'contactsheet':str(sheet_path.resolve()), 'contactsheet_sha256':sha(sheet_path),
              'probe':probe}
    a.output.with_suffix('.render.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
