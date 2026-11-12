"""Small, explicit wrist-path experiments; never edits hand or object geometry."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
OUT = ROOT / 'outputs/fourth_full_sequence/edge_warp_batch_v2'
if OUT.exists():
    raise FileExistsError('Use a new batch name; preserve experiment evidence')
OUT.mkdir()
source_trial = ROOT / 'outputs/fourth_full_sequence/physics_o6_thickrim_B_top_v1'
runner = ROOT / 'scripts/run_fourth_o6_edge_warp_frozen_v2.py'
if runner.exists():
    raise FileExistsError(runner)
shutil.copy2(source_trial / 'source_run_fourth_o6_thickrim_grasp.py', runner)
model = ROOT / 'outputs/fourth_full_sequence/assets/vita_six_objects_newft_o6_thickrim_inward_008_v2.xml'
candidate_path = source_trial / 'source_candidate.json'
base = json.loads(candidate_path.read_text())['best']
m = mj.MjModel.from_xml_path(str(model))
d = mj.MjData(m)
d.qpos[:] = base['qpos']
mj.mj_forward(m, d)
ob = m.body('target_object_01').id
hb = m.body('rh_hand_base_link').id
site = m.site('rh_palm_control_site').id
OR = d.xmat[ob].reshape(3, 3)
HR = d.xmat[hb].reshape(3, 3)
offset = OR.T @ (d.site_xpos[site] - d.xpos[ob])
localR = OR.T @ HR
vertices = []
for g in range(m.ngeom):
    if m.geom_bodyid[g] != ob or not m.geom_contype[g]:
        continue
    if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
        mesh = int(m.geom_dataid[g])
        start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
        world = m.mesh_vert[start:start+count] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        vertices.append((world - d.xpos[ob]) @ OR)
v = np.concatenate(vertices)
radius = float(np.linalg.norm(v[:, :2], axis=1).max())
top = float(v[:, 2].max())
radial = np.r_[offset[:2], 0.]
radial /= np.linalg.norm(radial)
tangent = np.cross([0., 0., 1.], radial)
pivot = radial * (radius-.005) + np.array([0., 0., top-.006])
template = {
    'finger_joint_targets': {m.joint(j).name: float(d.qpos[int(m.jnt_qposadr[j])])
                             for j in range(m.njnt) if (m.joint(j).name or '').startswith('rh_')},
    'controlled_digits': ['thumb', 'index', 'middle', 'ring', 'pinky'],
    'required_contact_digits': ['thumb', 'index', 'middle']}

gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
gpu_uuids = {int(a.strip()): b.strip() for line in gpu_info.splitlines()
             for a, b, c, e in [line.split(',')]}
free = {int(a.strip()) for line in gpu_info.splitlines()
        for a, b, c, e in [line.split(',')] if int(c.strip()) < 256 and int(e.strip()) == 0}
experiments = [('down4', 4, -.004, 0.), ('down6', 6, -.006, 0.), ('tilt24_down4', 7, -.004, 24.)]
if not {x[1] for x in experiments} <= free:
    raise RuntimeError('An assigned GPU became busy; do not use occupied GPUs')
manifest = {'scope': 'Actual full-robot actuator-only comparisons of wrist trajectory warps. '
                     'Candidate FK states are planning only; raw object reset remains unchanged.',
            'source_candidate': str(candidate_path),
            'source_candidate_sha256': hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
            'frozen_runner': str(runner), 'runner_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(),
            'object_model': str(model), 'rim_pivot_object_m': pivot.tolist(),
            'physics_parameters_unchanged': True, 'experiments': []}
jobs = []
for name, gpu, dz, degrees in experiments:
    rot = Rotation.from_rotvec(tangent*np.deg2rad(degrees)).as_matrix()
    changed = dict(base)
    entry = dict(template)
    entry['palm_offset_in_object_m'] = (pivot+rot@(offset-pivot)+np.array([0., 0., dz])).tolist()
    entry['hand_rotation_in_object_wxyz'] = Rotation.from_matrix(rot@localR).as_quat()[[3, 0, 1, 2]].tolist()
    changed['template'] = entry
    cp = OUT / (name+'_candidate.json')
    cp.write_text(json.dumps({'scope': 'Wrist target warp only; qpos is a planning seed, not a physical reset.',
                             'warp': {'object_z_m': dz, 'tangent_rotation_deg': degrees},
                             'best': changed}, indent=2)+'\n')
    trial = ROOT / ('outputs/fourth_full_sequence/physics_o6_edge_warp_'+name+'_v2')
    row = {'name': name, 'gpu': gpu, 'gpu_uuid': gpu_uuids[gpu], 'object_z_m': dz, 'tangent_rotation_deg': degrees,
           'candidate': str(cp), 'trial': str(trial), 'status': 'prepared'}
    manifest['experiments'].append(row)
    jobs.append((row, cp, trial))
mp = OUT / 'manifest.json'
mp.write_text(json.dumps(manifest, indent=2)+'\n')

def run(job):
    row, cp, trial = job
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=row['gpu_uuid'], OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    cmd = [sys.executable, str(runner), '--model', str(model), '--candidate', str(cp),
           '--output', str(trial), '--approach', 'top', '--hold-open-seed',
           '--closure-rate', '.1', '--gain', '.00015', '--seconds', '24']
    start = time.monotonic()
    with (OUT/(row['name']+'.log')).open('w') as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    row.update(status='finished' if result.returncode == 0 else 'execution_failed',
               returncode=result.returncode, elapsed_s=time.monotonic()-start)
    if (trial/'report.json').exists():
        r = json.loads((trial/'report.json').read_text())
        row['result'] = {k: r[k] for k in ['success', 'duration_s', 'termination',
                         'max_mimic_error_rad', 'maximum_object_rise_m'] if k in r}
    return row

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for future in concurrent.futures.as_completed([pool.submit(run, j) for j in jobs]):
        print(json.dumps(future.result()), flush=True)
        mp.write_text(json.dumps(manifest, indent=2)+'\n')
print(str(mp), flush=True)
