"""Low-dimensional reference-PD/actor mixture grid, same HDMI physics contract."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from backend import HdmiDynamics, sha

p=argparse.ArgumentParser()
p.add_argument('--output',required=True,type=Path)
p.add_argument('--clip-targets',action='store_true')
a=p.parse_args()
base=Path('/data/yihang.bai/training/hdmi_vita_mjlab')
out=base/'outputs/gallery08_p01_trajopt_20261030_2205'
motion=base/'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3/mjlab_motion.npz'
ckpt=base/'logs/rsl_rl/hdmi_vita28_exo_exo_20261030_v1_videomimic_gallery_08_v2__person_01/2026-10-30_18-31-07_repair_smooth3_continuity_v1_gpu4/model_2297.pt'
alphas=np.asarray(list(itertools.product((0.,.1,.25,.5,.75,1.),(0.,.5,1.),(0.,.5,1.))))
a.output.mkdir(exist_ok=False)
b=HdmiDynamics(out/'runtime_snapshot',motion,motion.with_name('scene.json'),ckpt,count=len(alphas),clip_targets=a.clip_targets)
try:
    costs,metrics,_=b.rollout(np.zeros((len(alphas),b.frames-1,28),np.float32),blends=alphas)
    ranks=np.lexsort((costs,metrics['normalized_gate_violation'],metrics['failed_gate_count']))
    rows=[dict(rank=i,blend=alphas[k].tolist(),cost=float(costs[k]),metrics={key:np.asarray(val)[k].item() for key,val in metrics.items()}) for i,k in enumerate(ranks)]
    (a.output/'ranked_grid.json').write_text(json.dumps(rows,indent=2))
    for i,k in enumerate(ranks[:5]):
        np.savez_compressed(a.output/f'controls_rank{i}.npz',residual_actions=np.zeros((b.frames-1,28),np.float32),
            reference_blend=alphas[k],clip_control_targets_to_limits=np.array(a.clip_targets),
            source_reference_sha256=np.array(sha(motion)),checkpoint_sha256=np.array(sha(ckpt)),scene_sha256=np.array(sha(motion.with_name('scene.json'))),
            joint_names=np.asarray(b.names),semantics=np.array('frozen_actor_reference_PD_mixture_plus_normalized_residual'))
    print(json.dumps(rows[:5],indent=2),flush=True)
finally:b.close()
