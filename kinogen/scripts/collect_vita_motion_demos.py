"""SHA-verified local delivery of all 17 server-rendered Vita replays."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
from urllib.parse import quote
import re
import subprocess

from pull_raw_artifact import file_sha

ROOT = Path(__file__).resolve().parents[1]


def relative_remote(path):
    value = str(PurePosixPath(path).relative_to('/data/KinoGen'))
    if not re.fullmatch(r'outputs/previews/vita17_[A-Za-z0-9_.-]+\.(mp4|jpg)', value):
        raise ValueError('Expected an explicit Vita preview artifact')
    return value


def transfer(path, expected):
    target = ROOT/'previews'/PurePosixPath(path).name
    if target.is_file() and file_sha(target) == expected:
        return {'source': path, 'local_path': str(target), 'bytes': target.stat().st_size,
                'sha256': expected, 'already_verified': True}
    partial = target.with_name(target.name+'.partial')
    partial.unlink(missing_ok=True)
    try:
        # Legacy scp reaches the project container's /data/KinoGen path; the
        # server's default SFTP endpoint exposes a different filesystem.
        subprocess.run(['scp', '-O', '-q', '-o', 'BatchMode=yes',
                        'yz-ai-4090-01.bce:/data/KinoGen/'+path, str(partial)],
                       check=True, timeout=300)
        actual = file_sha(partial)
        if actual != expected:
            raise RuntimeError(f'SHA mismatch for {path}: {actual} != {expected}')
        size = partial.stat().st_size
        partial.replace(target)
        return {'source': path, 'local_path': str(target), 'bytes': size,
                'sha256': expected, 'already_verified': False}
    finally:
        partial.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'reports/vita_motion_demos/manifest.json')
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    clips = manifest['clips']
    if len(clips) != 17 or sum(x['source_frames'] for x in clips) != 6415:
        raise ValueError('Final local delivery requires all 17 clips and 6415 source frames')
    items = []
    for clip in clips:
        if clip['mode'] != 'kinematic_replay' or clip['physics_steps'] != 0 or clip['policy_inference']:
            raise ValueError('Replay gallery cannot be labeled as a physics-policy result')
        items.extend([(relative_remote(clip['video']), clip['video_sha256']),
                      (relative_remote(clip['contactsheet']), clip['contactsheet_sha256'])])
    receipts = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(transfer, *item): item for item in items}
        for future in as_completed(pending):
            receipts.append(future.result())
            print(json.dumps({'verified_artifacts': len(receipts), 'total_artifacts': len(items),
                              'file': pending[future][0]}), flush=True)
    cards = []
    for i, clip in enumerate(clips, 1):
        video = Path(clip['video']).name
        poster = Path(clip['contactsheet']).name
        title = Path(clip['source']).stem
        cards.append(f'''<article><h2>{i:02d} · {html.escape(title)}</h2>
<p>{clip['source_frames']}帧 · {clip['source_fps']:g} FPS · {clip['source_frames']/clip['source_fps']:.2f}秒</p>
<video controls preload="none" poster="{quote(poster)}" src="{quote(video)}"></video>
<p><a href="{quote(video)}">单独打开视频</a> · <a href="{quote(poster)}">查看关键帧</a></p></article>''')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Vita02a · 17段动作演示</title>
<style>body{margin:0;background:#111722;color:#edf1f7;font:16px/1.6 system-ui,sans-serif}main{max-width:1450px;margin:auto;padding:30px}h1{font-size:30px}header{margin-bottom:24px}header p{max-width:1000px;color:#c3cede}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,550px),1fr));gap:22px}article{background:#1c2635;border:1px solid #35445a;border-radius:12px;padding:18px}h2{font-size:17px;overflow-wrap:anywhere}article p{color:#b9c8db}video{width:100%;border-radius:6px;background:black}a{color:#8cc8ff}</style>
<main><header><h1>Vita02a · 17段动作演示</h1><p>完整6415帧，原始30 FPS。左侧为两个全身视角，右侧为左右五指手部近景。逐帧保留来源的根部位姿与52个关节状态。</p>
<p><strong>本页为运动学参考回放。</strong>没有策略推理、动力学积分或物体交互；地面仅是显示参考。视频不构成接触力、驱动力矩或物理跟踪成功的证明。</p></header><div class="grid">'''
    page += '\n'.join(cards) + '</div></main></html>'
    index = ROOT/'previews/vita17_index.html'
    index.write_text(page, encoding='utf-8')
    report = {'status': 'all17_kinematic_previews_locally_sha_verified', 'clips': 17,
              'source_frames': 6415, 'fps': 30, 'policy_inference': False, 'physics_steps': 0,
              'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'gallery': str(index), 'gallery_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
              'files': sorted(receipts, key=lambda x: x['source'])}
    target = ROOT/'reports/vita17-local-delivery.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'gallery': str(index), 'receipt': str(target), 'verified_files': len(receipts)}), flush=True)


if __name__ == '__main__':
    main()
