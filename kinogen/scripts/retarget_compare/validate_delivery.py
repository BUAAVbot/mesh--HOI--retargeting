"""Check local comparison delivery links and method coverage."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    metrics = json.loads((root / 'comparison_metrics.json').read_text(encoding='utf-8'))
    names = {row['method'] for row in metrics['results']}
    expected = {'gmr', 'custom_ik', 'custom_ik_contact', 'umr', 'umr_refined',
                'umr_contact_adapted', 'umr_postprocessed', 'omniretarget',
                'hoi_retarget', 'phc'}
    if names != expected:
        raise RuntimeError(f'Method coverage differs: {names ^ expected}')
    report = root / 'report/REPORT.zh-CN.md'
    markdown = report.read_text(encoding='utf-8')
    html = (root / 'report/report.html').read_text(encoding='utf-8')
    links = re.findall(r'(?<!!)\[[^]]+\]\(([^)]+)\)|!\[[^]]*\]\(([^)]+)\)', markdown)
    targets = [link[0] or link[1] for link in links]
    targets += re.findall(r'(?:src|href)="([^"]+)"', html)
    broken = []
    for target in targets:
        if target.startswith(('http:', 'https:', '#')):
            continue
        path = (report.parent / unquote(target)).resolve()
        if root not in path.parents and path != root or not path.exists():
            broken.append(target)
    if broken:
        raise RuntimeError(f'Broken report links: {broken}')
    primary = ['source', 'gmr', 'custom_ik', 'custom_ik_contact', 'umr',
               'omniretarget', 'hoi_retarget', 'phc']
    main_video = json.loads((root / 'media/overview/comparison.manifest.json').read_text())
    ablation = json.loads((root / 'media/overview/umr_ablation.manifest.json').read_text())
    if [item['method'] for item in main_video['methods']] != primary or main_video['frames'] != 172:
        raise RuntimeError('Main video pane or frame mismatch')
    if [item['method'] for item in ablation['methods']] != [
            'source', 'umr', 'umr_refined', 'umr_contact_adapted', 'umr_postprocessed']:
        raise RuntimeError('UMR ablation pane mismatch')
    for row in metrics['results']:
        directory = root / row['method']
        for name in ('evaluation.json', 'evaluation_arrays.npz', 'interval_rate_audit.json'):
            if not (directory / name).exists():
                raise RuntimeError(f'Missing {directory / name}')
    result = {'method_rows': len(names), 'distinct_methods': 6,
              'main_video_panes': len(primary), 'umr_ablation_panes': 5,
              'frames': 172, 'report_links_checked': len(targets), 'broken_links': 0,
              'intermimic_status': json.loads((root / 'intermimic/report.json').read_text())['status']}
    (root / 'delivery_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
