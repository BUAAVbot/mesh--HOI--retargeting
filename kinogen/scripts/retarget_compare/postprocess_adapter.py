"""Run the supplied UMR two-person ground/SQP stages with a Vita52 port.

The bundled algorithms retain their objectives, weights, constraints and stopping
rules. Only state dimensions and common-scene geometry names are adapted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/KinoGen/outputs/retarget_compare_20261028'))
    p.add_argument('--output', type=Path)
    p.add_argument('--stage', choices=['prepare', 'ground', 'smooth', 'assemble', 'all'], default='all')
    a = p.parse_args()
    output = a.output or a.root/'umr/postprocessed'
    source = a.root/'umr/source'
    manifest_path = a.root/'shared_input/source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    model_path = Path(manifest['model']).parent/'two_vita_solver20.xml'
    import numpy as np
    import mujoco as mj
    if a.stage in ('prepare', 'all'):
        if output.exists():
            raise FileExistsError('Use a new postprocessing directory: '+str(output))
        output.mkdir(parents=True)
        for name in ['results', 'preview_refined', 'preview_grounded', 'adapted_stages']:
            (output/name).mkdir()
        model = mj.MjModel.from_xml_path(str(model_path))
        if (model.nq, model.nv) != (118, 116):
            raise ValueError('Expected unchanged shared dual Vita52 scene')
        reference_path = a.root/'umr/umr_vita52.npz'
        archive = np.load(reference_path, allow_pickle=False)
        q = archive['qpos']
        if q.shape != (2, 172, 59):
            raise ValueError('Expected full unmodified UMR raw reference')
        times = archive['times']
        names = [str(v) for v in archive['dof_names']]
        if names != manifest['dof_names']:
            raise ValueError('UMR output order differs from shared model')
        for person in (1, 2):
            np.savez_compressed(output/('results/person_%02d_refined.npz' % person),
                                qpos=q[person-1], times=times, dof_names=np.array(names),
                                fps=np.array(30.), frame_ids=np.arange(172),
                                source_reference=np.array(str(reference_path)))
        (output/'baseline_settings.json').write_text(json.dumps({'frames':172, 'fps':30.,
                  'source':'common UMR raw reference, no retiming or seat refitting'}, indent=2)+'\n')
        # Both stages use the exact same scene. Absolute paths preserve mesh refs.
        for folder in ['preview_refined', 'preview_grounded']:
            (output/folder/'dual_scene.xml').write_bytes(model_path.read_bytes())
        bundled = source/'tools/two_person_baseline'
        stage_records = []
        replacements = {35:59, 34:58, 28:52, 68:116, 70:118}
        for filename in ['ground_contacts.py', 'smooth_motion.py']:
            original = bundled/filename
            text = original.read_text()
            text = re.sub(r'\b(35|34|28|68|70)\b',
                          lambda match:str(replacements[int(match.group(0))]), text)
            text = text.replace('"floor"', '"ground_proxy"')
            text = text.replace('f"seat_{i:02d}"', 'f"seat_{i}_proxy"')
            if filename == 'ground_contacts.py':
                # Original module selects five legacy foot-proxy name patterns.
                # Select all seven existing capsule proxies without modifying geometry.
                text = text.replace('for key in [f"foot_plane_{side}",',
                                    'for key in [f"collision_foot_{side}_", f"foot_plane_{side}",')
            else:
                # Original foot-plane proxy center is replaced by the actual
                # central/longest capsule of the shared seven-capsule foot.
                text = text.replace('f"p{person}_collision_foot_plane_{side}_box"',
                                    'f"p{person}_collision_foot_{side}_capsule_3"')
            ast.parse(text)
            target = output/'adapted_stages'/filename
            target.write_text(text)
            stage_records.append({'source':str(original),'source_sha256':sha(original),
                                  'adapted':str(target),'adapted_sha256':sha(target)})
        settings = ('from pathlib import Path\nimport json\n'
                    'REPO=Path('+repr(str(source))+')\n'
                    'def work_dir():return Path('+repr(str(output))+')\n'
                    'def load_settings():return json.loads((work_dir()/"baseline_settings.json").read_text())\n')
        (output/'adapted_stages/settings.py').write_text(settings)
        aux = [i+7 for i,n in enumerate(names) if n.startswith(('lh_','rh_','neck_'))]
        prep = {'schema':'umr_bundled_postprocessing_vita52.preparation.v1',
                'method':'UMR + bundled ground and sparse SQP, local Vita52 adapter',
                'input':str(reference_path),'input_sha256':sha(reference_path),
                'input_qpos_shape':list(q.shape),'frames':172,'times_unchanged':True,
                'common_scene':str(model_path),'common_scene_sha256':sha(model_path),
                'shared_manifest_sha256':sha(manifest_path),'nq':118,'nv':116,
                'source_stages':stage_records,
                'raw_auxiliary_max_abs_rad':float(np.abs(q[...,aux]).max()),
                'frozen_auxiliary_model_used':False,
                'adaptations':['Full state dimensions59/58 per robot replace legacy35/34; all52 joints retained.',
                  'World geom names select the unchanged ground_proxy/seat_1_proxy/seat_2_proxy.',
                  'Foot anchor group selects actual seven capsule proxies; central capsule3 serves as original foot-center proxy.',
                  'Input filenames keep bundled legacy refined names but contain exactly the current raw UMR reference.',
                  'No objective weights, constraints, iteration limits, residual20mm repair gate, time, scene or learned correspondence changed.'],
                'physical_rollout_executed':False}
        (output/'preparation.json').write_text(json.dumps(prep,indent=2)+'\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    env['PYTHONPATH'] = os.pathsep.join([str(a.root/'umr/env_overlay'),
                                      str(source/'scripts'), env.get('PYTHONPATH','')])
    timings = []
    for stage, filename in [('ground','ground_contacts.py'),('smooth','smooth_motion.py')]:
        if a.stage not in (stage, 'all'):
            continue
        started = time.perf_counter()
        command = [sys.executable, str(output/'adapted_stages'/filename)]
        log = output/(stage+'.log')
        with log.open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            print(json.dumps({'stage':stage,'pid':process.pid,'log':str(log),'command':command}),flush=True)
            code = process.wait()
        row = {'stage':stage,'wall_seconds':time.perf_counter()-started,'returncode':code,'log':str(log)}
        timings.append(row)
        with (output/'stage_runs.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if code:
            result = {'status':stage+'_failed_original_acceptance_or_solver',
                      'method':'UMR + bundled postprocessing (local Vita52 adaptation)',
                      'stage_runs':timings,'source_reference_preserved':True,
                      'physics_rollout_executed':False,'physical_success':None,
                      'failure_log_tail':log.read_text()[-9000:]}
            (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
            raise SystemExit(code)
    if a.stage in ('assemble','all'):
        paths = [output/('smoothing_v1/results/person_%02d_smoothed.npz'%i) for i in (1,2)]
        q = np.stack([np.load(path,allow_pickle=False)['qpos'] for path in paths])
        original = np.load(a.root/'umr/umr_vita52.npz',allow_pickle=False)
        target = output/'umr_vita52.npz'
        np.savez_compressed(target,qpos=q,times=original['times'],dof_names=original['dof_names'],
                            frame_ids=original['frame_ids'],
                            method=np.asarray('UMR + bundled postprocessing (local Vita52 adaptation)'))
        smooth = json.loads((output/'smoothing_v1/smoothing_report.json').read_text())
        runs = [json.loads(line)for line in (output/'stage_runs.jsonl').read_text().splitlines()]
        result = {'schema':'umr_bundled_postprocessing_vita52.result.v1',
                  'method':'UMR + bundled postprocessing (local Vita52 adaptation)',
                  'status':'complete_postprocessing_converged' if smooth['converged'] else 'complete_postprocessing_iteration_limit',
                  'sqp_converged':smooth['converged'],'termination':smooth['termination'],
                  'source_reference_preserved':True,'reference':str(target),'reference_sha256':sha(target),
                  'qpos_shape':list(q.shape),'frames':172,'duration_seconds':5.7,
                  'stage_runs':runs,'smoothing_report':smooth,
                  'physics_rollout_executed':False,'physical_success':None,
                  'limitations':['Bundled kinematic postprocessing; convergence and geometry metrics are separate.',
                                 'The full52 state and shared capsule foot proxy are disclosed local adaptations.',
                                 'No policy, physical stability or load-bearing validation.']}
        (output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k]for k in ['status','sqp_converged','qpos_shape']}),flush=True)


if __name__ == '__main__':
    main()
