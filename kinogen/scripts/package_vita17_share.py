"""Create a SHA-checked, Feishu-ready ZIP of the 17 kinematic demos."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORTS = [
    'reports/VITA17_FEISHU_UPDATE_2026-10-26.md',
    'reports/vita17-input-audit.json',
    'reports/vita17-video-audit.json',
    'reports/vita17-local-delivery.json',
    'reports/vita17-policy-tracking-feasibility.json',
]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT/'reports/vita_motion_demos/manifest.json').read_text(encoding='utf-8'))
    receipt = json.loads((ROOT/'reports/vita17-local-delivery.json').read_text(encoding='utf-8'))
    clips = manifest['clips']
    if len(clips) != 17 or sum(x['source_frames'] for x in clips) != 6415:
        raise ValueError('Expected exactly 17 complete clips and 6415 frames')
    if receipt['status'] != 'all17_kinematic_previews_locally_sha_verified':
        raise ValueError('Local delivery is not verified')
    expected = {}
    for clip in clips:
        if clip['mode'] != 'kinematic_replay' or clip['policy_inference'] or clip['physics_steps']:
            raise ValueError('Cannot package physical results as kinematic demos')
        expected['previews/'+Path(clip['video']).name] = clip['video_sha256']
        expected['previews/'+Path(clip['contactsheet']).name] = clip['contactsheet_sha256']
    if len(expected) != 34:
        raise ValueError('Expected 17 videos and 17 contact sheets')
    for relative, digest in expected.items():
        if sha(ROOT/relative) != digest:
            raise ValueError(f'Local artifact changed: {relative}')
    files = sorted(expected) + ['previews/vita17_index.html'] + REPORTS
    output = ROOT/'vita17_share.zip'
    partial = output.with_name(output.name+'.partial')
    partial.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(partial, 'w', allowZip64=True) as bundle:
            for relative in files:
                bundle.write(ROOT/relative, relative, compress_type=zipfile.ZIP_STORED)
        with zipfile.ZipFile(partial) as bundle:
            bad = bundle.testzip()
            if bad or set(bundle.namelist()) != set(files):
                raise RuntimeError(f'Archive verification failed: {bad}')
        partial.replace(output)
    finally:
        partial.unlink(missing_ok=True)
    result = {'path': str(output), 'bytes': output.stat().st_size, 'sha256': sha(output),
              'videos': 17, 'contact_sheets': 17, 'frames': 6415,
              'all_media_source_sha_verified': True,
              'video_audit_sha256': sha(ROOT/'reports/vita17-video-audit.json'),
              'feishu_report_sha256': sha(ROOT/REPORTS[0])}
    (ROOT/'reports/vita17-share-bundle.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
