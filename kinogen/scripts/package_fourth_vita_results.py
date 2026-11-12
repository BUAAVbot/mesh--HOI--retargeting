"""Package the verified short-demo results; original meshes/data stay in the workspace."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
base = ROOT/'outputs/fourth_vita_demo'
files = list((base/'physics_source_slide_v12_final').glob('*'))
files += list((base/'hoi_v12_final').glob('*'))
files += list((ROOT/'previews').glob('fourth_vita_*v12*'))
files += [base/'retarget_source_slide_v12.npz', base/'retarget_source_slide_v12.json',
          base/'retarget_source_slide_v12.metadata.json',
          base/'assets/vita_object_05_dynamic_ankle100.xml',
          base/'assets/vita_object_05_assets_report.json',
          base/'assets/vita_object_05_dynamic_ankle100_report.json']
files += [ROOT/'scripts'/n for n in [
    'fourth_vita_physics.py', 'fourth_vita_retarget.py',
    'fourth_vita_contact_slide_reference.py', 'fourth_vita_sim_assets.py',
    'fourth_vita_gain_variant.py', 'audit_fourth_vita_physics.py',
    'render_fourth_vita_trace.py', 'plot_fourth_vita_metrics.py',
    'export_fourth_vita_hoi.py', 'verify_fourth_vita_delivery.py']]
files += [ROOT/'reports'/n for n in [
    'FOURTH_VITA_DEMO_RESULT_2026-10-27.md', 'FEISHU_FOURTH_VITA_2026-10-27.md',
    'fourth_vita_v12_source_audit.json', 'fourth_vita_v12_delivery_verification.json']]
files = sorted(set(p for p in files if p.is_file()))
manifest = {str(p.relative_to(ROOT)).replace('\\','/'):
            {'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
folder = ROOT/'deliveries'
folder.mkdir(exist_ok=True)
out = folder/'fourth_vita_v12_results_20261027.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files:
        z.write(p, str(p.relative_to(ROOT)).replace('\\','/'))
    z.writestr('delivery_manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    z.writestr('README.txt',
        'Vita52 + bowl5 short physical demo results, 2026-10-27.\n'
        'Start with reports/FEISHU_FOURTH_VITA_2026-10-27.md and previews/*.mp4.\n'
        'This results archive includes frozen traces, contact forces, joint torques, reference and scripts.\n'
        'Original source archive, full robot mesh library and Python environment are not duplicated here.\n'
        'The frozen MJCF retains server mesh paths under /data/KinoGen. See the result report for reproduction.\n'
        'Hand targets are adapted; original object targets/timing are preserved; physical parameters assumed.\n')
with zipfile.ZipFile(out) as z:
    if z.testzip() is not None:
        raise ValueError('Archive CRC validation failed')
    for name, item in manifest.items():
        if hashlib.sha256(z.read(name)).hexdigest() != item['sha256']:
            raise ValueError('Archive member hash mismatch: '+name)
receipt = {'archive':str(out), 'files':len(manifest), 'bytes':out.stat().st_size,
           'sha256':hashlib.sha256(out.read_bytes()).hexdigest(), 'members':manifest}
out.with_suffix('.manifest.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k != 'members'}))
