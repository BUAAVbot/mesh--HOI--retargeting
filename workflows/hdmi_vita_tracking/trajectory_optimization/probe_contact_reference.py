"""Small physical probe of a contact-estimation candidate; never starts RL."""
import argparse
import json
from pathlib import Path
import numpy as np
from backend import HdmiDynamics,sha

p=argparse.ArgumentParser();p.add_argument('--motion',type=Path,required=True)
p.add_argument('--scene',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
base=Path('/data/yihang.bai/training/hdmi_vita_mjlab');out=base/'outputs/gallery08_p01_trajopt_20261030_2205'
ckpt=base/'logs/rsl_rl/hdmi_vita28_exo_exo_20261030_v1_videomimic_gallery_08_v2__person_01/2026-10-30_18-31-07_repair_smooth3_continuity_v1_gpu4/model_2297.pt'
blends=np.asarray(((0,0,0),(0,.2,.5),(0,.5,1),(.05,0,0),(.05,.2,.5),(.1,.2,.5),
                   (.2,.2,.5),(.3,.5,1),(.5,.5,1),(.75,.5,1),(1,1,1),(.1,1,1)))
b=HdmiDynamics(out/'runtime_snapshot',a.motion,a.scene,ckpt,count=len(blends))
try:
    costs,metrics,_=b.rollout(np.zeros((len(blends),b.frames-1,28),np.float32),blends=blends)
    ranks=np.lexsort((costs,metrics['normalized_gate_violation'],metrics['failed_gate_count']))
    rows=[]
    for rank,k in enumerate(ranks):
        rows.append(dict(rank=rank,blend=blends[k].tolist(),metrics={key:np.asarray(v)[k].item() for key,v in metrics.items()}))
        if rank<3:
            np.savez_compressed(a.output/f'controls_rank{rank}.npz',residual_actions=np.zeros((b.frames-1,28),np.float32),
                reference_blend=blends[k],clip_control_targets_to_limits=np.array(False),
                source_reference_sha256=np.array(sha(a.motion)),scene_sha256=np.array(sha(a.scene)),checkpoint_sha256=np.array(sha(ckpt)),
                joint_names=np.asarray(b.names),semantics=np.array('frozen_actor_reference_PD_mixture_plus_normalized_residual'))
    (a.output/'probe.json').write_text(json.dumps(dict(status='physical_probe_not_acceptance',
        reference=str(a.motion),reference_sha256=sha(a.motion),scene_sha256=sha(a.scene),ranked=rows),indent=2))
    print(json.dumps(rows,indent=2),flush=True)
finally:b.close()
