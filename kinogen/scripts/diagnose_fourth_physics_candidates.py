"""Explicit diagnostic reference variants; never replace the raw source reference."""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/KinoGen'))
    args = p.parse_args()
    output = args.root/'outputs/fourth_vita_demo'
    with np.load(output/'retarget_reference_fingercontact_v5b.npz', allow_pickle=False) as z:
        base = {k: z[k] for k in z.files}
    with np.load(output/'retarget_reference_palm_v6.npz', allow_pickle=False) as z:
        palm = {k: z[k] for k in z.files}
    names = list(base['dof_names'].astype(str))
    right = [7+i for i,n in enumerate(names) if n.startswith(('right_shoulder','right_elbow','right_wrist','rh_'))]
    rows = []
    for start in [581, 584, 587]:
        mask = base['frame_ids'] >= start
        item = {k:(v[mask].copy() if v.ndim and len(v)==len(mask) else v.copy()) for k,v in base.items()}
        item['qpos'][:,right] = palm['qpos'][0,right]
        item['dof_pos'] = item['qpos'][:,7:].copy()
        item['time_s'] -= item['time_s'][0]
        item['diagnostic_variant'] = np.asarray('v5b left reference plus v6 safe right arm; source status1 suffix; no source object trajectory modification')
        path = output/f'retarget_diagnostic_status1_{start}.npz'
        np.savez_compressed(path, **item)
        rows.append({'path': str(path), 'frames':len(item['qpos']), 'duration_s':float(item['time_s'][-1]), 'start':start})
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main()
