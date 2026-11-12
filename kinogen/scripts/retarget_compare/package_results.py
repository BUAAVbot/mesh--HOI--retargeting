"""Make a compact, checksummed handoff for the two-person Vita comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def include(relative: Path) -> bool:
    parts = relative.parts
    if not parts:
        return False
    if parts[0] in {'gmr_smoke', 'hoi_smoke', 'phc_smoke', 'omniretarget_smoke',
                    'umr_setup', 'phc_seed1_noise005', 'phc_seed2_noise005'}:
        return False
    if any(part in {'__pycache__', 'env_overlay'} for part in parts):
        return False
    if relative.suffix in {'.pyc', '.zip', '.tar', '.gz'}:
        return False
    if parts[:2] == ('umr', 'correspondence') and relative.name in {
            'checkpoint-final.pt', 'dataset.npz'}:
        return False
    if parts[0] == 'shared_input' and len(parts) >= 2 and parts[1] == 'rgb_174114_sim_handoff':
        return len(parts) == 3 and relative.name in {
            'input.mp4', 'cameras_world.npz', 'timeline.npz', 'coordinates.npz'}
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path,
                        default=Path('/data/KinoGen/outputs/retarget_compare_20261028'))
    parser.add_argument('--output', type=Path,
                        default=Path('/data/KinoGen/outputs/vita_retarget_comparison_20261028.zip'))
    args = parser.parse_args()
    root = args.root.resolve()
    files = [path for path in root.rglob('*') if path.is_file() and include(path.relative_to(root))]
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    required = [
        'comparison_metrics.json', 'shared_input/source_manifest.json',
        'shared_input/common_observations.npz', 'report/REPORT.zh-CN.md',
        'report/report.html', 'media/overview/comparison.mp4',
        'media/overview/umr_ablation.mp4', 'intermimic/report.json',
        'audit/shared_smplx_gpu0/report.json',
    ]
    included = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(set(required) - included)
    if missing:
        raise RuntimeError(f'Missing required comparison artifacts: {missing}')
    entries = [{'path': path.relative_to(root).as_posix(), 'bytes': path.stat().st_size,
                'sha256': digest(path)} for path in files]
    manifest = {
        'description': 'Two-person Vita52 kinematic retargeting comparison; not a physical-policy rollout',
        'source_root': str(root), 'archive_prefix': root.name,
        'excluded_large_files': [
            'umr/correspondence/checkpoint-final.pt', 'umr/correspondence/dataset.npz',
            'umr/env_overlay', 'shared_input/rgb_174114_sim_handoff/* except input.mp4 and metadata',
        ],
        'server_side_large_file_hashes': {
            name: digest(root / name) for name in (
                'umr/correspondence/checkpoint-final.pt', 'umr/correspondence/dataset.npz')
        },
        'files': entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, 'w', ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, f'{root.name}/{path.relative_to(root).as_posix()}')
        archive.writestr(f'{root.name}/PACKAGE_MANIFEST.json',
                         json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    with ZipFile(args.output) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise RuntimeError(f'ZIP CRC failed: {corrupt}')
    summary = {'archive': str(args.output), 'bytes': args.output.stat().st_size,
               'sha256': digest(args.output), 'files': len(files),
               'manifest_path': f'{root.name}/PACKAGE_MANIFEST.json', 'crc_ok': True}
    args.output.with_suffix('.receipt.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
