"""Read-only CPU audit of a serialized mjlab scene and expected static boxes."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    m = mujoco.MjModel.from_binary_path(str(args.model))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    expected = json.loads(args.scene.read_text())["boxes"]
    rows = []
    for i, box in enumerate(expected):
        label = f"seat_{i}"
        body_ids = [bid for bid in range(m.nbody)
                    if label in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or "")]
        geom_ids = [gid for gid in range(m.ngeom)
                    if label in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or "")]
        geoms = []
        for gid in geom_ids:
            geoms.append(dict(name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid),
                              world_pos=d.geom_xpos[gid].tolist(),
                              half_size=m.geom_size[gid].tolist(),
                              expected_pos=box["pos"],
                              position_error_m=float(np.linalg.norm(d.geom_xpos[gid] - box["pos"]))))
        rows.append(dict(index=i, expected_name=box.get("name"), expected_pos=box["pos"],
                         bodies=[dict(name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid),
                                      parent=int(m.body_parentid[bid]), mocap_id=int(m.body_mocapid[bid]),
                                      model_pos=m.body_pos[bid].tolist(), world_pos=d.xpos[bid].tolist())
                                 for bid in body_ids], geoms=geoms))
    result = dict(nmocap=m.nmocap, default_mocap_positions=d.mocap_pos.tolist(), boxes=rows)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
