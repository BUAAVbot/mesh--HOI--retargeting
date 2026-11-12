"""Audit, export and render one accepted full-sequence trial into a portable bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb')as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/rgb_261026_near_handoff'))
    p.add_argument('--source-targets',type=Path,default=Path('outputs/rgb_near_pipeline/source_joint_targets.npz'))
    p.add_argument('--skip-video',action='store_true')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    a.trial=a.trial.resolve();a.output=a.output.resolve()
    report=json.loads((a.trial/'report.json').read_text())
    checks=report.get('checks',{})
    if report.get('simulation_passed')is not True or len(checks)<12 or any(v is not True for v in checks.values()):
        raise ValueError('This trial has not passed the complete physical acceptance checks')
    if a.output.exists():raise FileExistsError('Choose a new delivery directory; frozen delivery will not be overwritten')
    a.output.mkdir(parents=True)
    commands=[]
    def run(script,*args):
        cmd=[sys.executable,str(root/'scripts'/script),*map(str,args)]
        commands.append(cmd)
        print(json.dumps({'stage':script}),flush=True)
        subprocess.run(cmd,cwd=root,env={**os.environ,'MUJOCO_GL':'egl'},check=True)
    ref=Path(report['input']['reference']['path'])
    run('audit_rgb_near_physics.py','--trial',a.trial)
    run('audit_rgb_near_source.py','--data',a.data.resolve(),'--reference',ref,
        '--source-targets',a.source_targets.resolve(),'--output',a.trial/'source_provenance_audit.json')
    run('export_rgb_near_hoi.py','--trial',a.trial,'--output',a.output/'dataset','--data',a.data.resolve())
    videos=[]
    if not a.skip_video:
        for view in ('shoe','ego'):
            video=a.output/'videos'/f'dual_vita_{view}.mp4'
            run('render_rgb_near_pipeline.py','--trial',a.trial,'--output',video,'--right-view',view,'--shoe-proxy-overlay')
            videos.append(video)
    run('plot_rgb_near_quality.py','--trial',a.trial,'--output',a.output/'reports/quality.png')
    run('summarize_rgb_near_trials.py','--base',a.trial.parent,'--output',a.output/'reports/iteration_history.json')
    run('verify_rgb_near_delivery.py','--trial',a.trial,'--export',a.output/'dataset','--reference',ref,
        *[s for video in videos for s in ('--video',video)],'--output',a.output/'reports/delivery_verification.json')
    dest=a.output/'frozen_trial';dest.mkdir()
    for name in ('model.mjb','trace.npz','contacts.npz','report.json','run_config.json',
                 'independent_replay_audit.json','source_provenance_audit.json'):
        shutil.copy2(a.trial/name,dest/name)
    for path in a.trial.glob('*.source.py'):shutil.copy2(path,dest/path.name)
    shutil.copy2(ref,dest/'reference.npz')
    for path in (ref.with_suffix('.json'),ref.with_suffix('.generator.py'),a.trial/'model.xml'):
        if path.exists():shutil.copy2(path,dest/path.name)
    for key in ('model_xml','proxy_xml'):
        source=Path(report['input'][key]['path'])
        shutil.copy2(source,dest/(key+'.xml'))
        meta=source.with_suffix('.json')
        if meta.exists():shutil.copy2(meta,dest/(key+'.json'))
    code=a.output/'scripts';code.mkdir()
    paths=set((root/'scripts').glob('rgb_near*.py'))
    paths.update((root/'scripts').glob('*rgb_near*.py'))
    paths.update([root/'scripts/fourth_sonic_controller.py',root/'scripts/o6_calibration.py'])
    for path in sorted(paths):shutil.copy2(path,code/path.name)
    src_receipts=a.output/'reports'
    for name in ('rgb_near_source_preparation_server.json','o6_06_source_audit_server.json',
                 'O6_06_HAND_INTEGRATION_2026-10-27.md','RGB_NEAR_SEATED_COLLISION_AUDIT_2026-10-27.md'):
        source=root/'reports'/name
        if source.exists():shutil.copy2(source,src_receipts/name)
    model_receipt=root/'outputs/rgb_near_pipeline/assets/O6_v4_build_receipt.json'
    if model_receipt.exists():shutil.copy2(model_receipt,src_receipts/model_receipt.name)
    (a.output/'README.txt').write_text(
        'RGB near dual-Vita HOI demonstration\n\n'
        'This bundle contains one accepted 398-frame source sequence, its recorded dynamic rollout,\n'
        'the compiled MuJoCo plant, source/reference provenance, simulated forces and rendered views.\n'
        'Acceptance uses assumed physical parameters and an explicitly disclosed hybrid SONIC controller.\n'
        'Wearing/grasping use recorded compliant attachments; this is not pure friction grasping.\n'
        'The original source shoe trajectory and the contact-optimized reference are both retained.\n'
        'See dataset/metadata.json for correction magnitudes, controller mix, gates and limitations.\n\n'
        'Recorded physics replay and rendering require Python, NumPy, MuJoCo 3.7.0, Pillow and FFmpeg.\n'
        'Run scripts/audit_rgb_near_physics.py --trial frozen_trial to verify the saved controls/states.\n'
        'Run scripts/render_rgb_near_pipeline.py --trial frozen_trial --output replay.mp4 to render.\n'
        'No policy checkpoint is needed for recorded-control replay. New policy inference or rebuilding\n'
        'references/assets requires the original KinoGen project, checkpoint and source datasets.\n'
        'Source/report paths refer to the producing server and remain provenance records.\n'
        'The ego view uses a synthetic look-at camera, not a calibrated original first-person capture.\n'
        'Simulation force/torque labels are estimates, not measured hardware ground truth.\n',encoding='utf-8')
    manifest={'schema':'rgb_near_delivery.v1','status':'accepted_assumed_physics_full_sequence',
              'source_trial':str(a.trial),'commands':commands,'files':{}}
    for path in sorted(a.output.rglob('*')):
        if path.is_file():manifest['files'][path.relative_to(a.output).as_posix()]={'bytes':path.stat().st_size,'sha256':sha(path)}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    archive=a.output.with_suffix('.zip')
    if archive.exists():raise FileExistsError('Archive already exists: '+str(archive))
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=4)as z:
        for path in sorted(a.output.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(a.output))
    print(json.dumps({'status':manifest['status'],'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive)}),flush=True)


if __name__=='__main__':main()
