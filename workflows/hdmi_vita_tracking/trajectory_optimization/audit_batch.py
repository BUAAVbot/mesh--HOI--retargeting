import json
import numpy as np
from pathlib import Path
from backend import HdmiDynamics

base = Path('/data/yihang.bai/training/hdmi_vita_mjlab')
out = base / 'outputs/gallery08_p01_trajopt_20261030_2205'
motion = base / 'batch_runs/exo_20261030_v1/gallery08_p01_contact_candidate_smooth3/mjlab_motion.npz'
ckpt = base / 'logs/rsl_rl/hdmi_vita28_exo_exo_20261030_v1_videomimic_gallery_08_v2__person_01/2026-10-30_18-31-07_repair_smooth3_continuity_v1_gpu4/model_2297.pt'
results = []
for n in (1, 4):
    b = HdmiDynamics(out/'runtime_snapshot', motion, motion.with_name('scene.json'), ckpt, count=n)
    try:
        cost, metrics, _ = b.rollout(np.zeros((n, b.frames-1, 28), np.float32))
        result = dict(count=n, origins=b.env.scene.env_origins.cpu().numpy().tolist(),
                      cost=cost.tolist(), metrics={k:v.tolist() for k,v in metrics.items()})
        print(json.dumps(result), flush=True)
        results.append(result)
    finally:
        b.close()
(out/'batch_invariance_audit.json').write_text(json.dumps(results, indent=2))
