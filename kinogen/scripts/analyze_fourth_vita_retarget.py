"""Inspect candidate Vita retarget contact geometry without stepping physics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    a = p.parse_args()
    z = np.load(a.reference, allow_pickle=False)
    model = mujoco.MjModel.from_xml_path(str(a.model))
    data = mujoco.MjData(model)
    palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'lh_hand_base_link')
    hand_geoms = [g for g in range(model.ngeom) if
                  (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                   int(model.geom_bodyid[g])) or '').startswith(('lh_', 'rh_'))]
    wall_geoms = [g for g in range(model.ngeom) if
                  (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '').startswith('target_object_05_wall_')]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'assumed_table')
    if floor < 0:
        print('TABLE_GEOMS', [(g, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g))
              for g in range(model.ngeom) if 'table' in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '')])
    rows = []
    for i in range(len(z['qpos'])):
        data.qpos[:59] = z['qpos'][i]
        data.qpos[59:66] = z['object_target_pose'][i]
        mujoco.mj_forward(model, data)
        support_target = z['palm_support_target_world'][i]
        palm_site = data.xpos[palm] + data.xmat[palm].reshape(3, 3) @ np.asarray([.015, 0, .055])
        best = (1e9, '', '')
        for g in hand_geoms:
            for w in wall_geoms:
                d = float(mujoco.mj_geomDistance(model, data, g, w, .3, np.zeros(6)))
                if d < best[0]:
                    best = (d, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g),
                            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, w))
        row = {'frame': int(z['frame_ids'][i]), 'palm_support_error_m': float(np.linalg.norm(palm_site-support_target)),
               'support_blend': float(z['palm_support_blend'][i]),
               'palm_site': palm_site.tolist(), 'target': support_target.tolist(),
               'hand_bowl_signed_distance_m': best}
        rows.append(row)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
