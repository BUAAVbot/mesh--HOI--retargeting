"""CPU FK/geometry audit of reference support, distinct from penetration checks."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from acceptance import load_npz, names, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--all-frames", action="store_true")
    args = p.parse_args()
    src = load_npz(args.source)
    sj, sb = names(src, "joint_names"), names(src, "body_names")
    model = mujoco.MjModel.from_binary_path(str(args.model))
    data = mujoco.MjData(model)
    mjn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    source_qadr = []
    for name in sj:
        ids = [i for i, other in enumerate(mjn) if other == name or (other and other.endswith("/" + name))]
        if len(ids) != 1:
            raise ValueError(name)
        source_qadr.append(int(model.jnt_qposadr[ids[0]]))
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1:
        raise ValueError("Expected exactly one floating root")
    root_qadr = int(model.jnt_qposadr[free[0]])
    root_index = sb.index("base_link")
    gn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    bn = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    feet = {side: [g for g in range(model.ngeom)
                   if bn[model.geom_bodyid[g]].split("/")[-1] == f"{side}_ankle_roll_link"]
            for side in ("left", "right")}
    surfaces = [g for g in range(model.ngeom)
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or "seat_" in gn[g]]
    frames = range(len(src["joint_pos"])) if args.all_frames else (0, len(src["joint_pos"]) - 1)
    rows = []
    for f in frames:
        mujoco.mj_resetData(model, data)
        data.qpos[root_qadr:root_qadr + 3] = src["body_pos_w"][f, root_index]
        data.qpos[root_qadr + 3:root_qadr + 7] = src["body_quat_w"][f, root_index]
        data.qpos[source_qadr] = src["joint_pos"][f]
        mujoco.mj_forward(model, data)
        row = dict(frame=int(f), root_position=data.qpos[root_qadr:root_qadr + 3].tolist(), feet={})
        for side, geoms in feet.items():
            measurements = []
            for fg in geoms:
                for sg in surfaces:
                    points = np.zeros(6)
                    distance = float(mujoco.mj_geomDistance(model, data, fg, sg, 100., points))
                    measurements.append(dict(foot_geom=gn[fg], surface_geom=gn[sg],
                                             distance_m=distance, foot_point=points[:3].tolist(),
                                             surface_point=points[3:].tolist(),
                                             is_plane=bool(model.geom_type[sg] == mujoco.mjtGeom.mjGEOM_PLANE)))
            plane = min((v for v in measurements if v["is_plane"]), key=lambda v: v["distance_m"])
            closest = min(measurements, key=lambda v: v["distance_m"])
            row["feet"][side] = dict(closest_surface=closest, closest_plane=plane,
                                     geometric_contact_within_5mm=closest["distance_m"] <= .005)
        rows.append(row)
    result = dict(schema="hdmi.trajopt.source_support_audit.v1", source_reference_sha256=sha256(args.source),
                  model_sha256=sha256(args.model), frames=rows,
                  note="FK distance only; absence of penetration is not proof of support or dynamic feasibility")
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(source_reference_sha256=result["source_reference_sha256"],
                          frames=[rows[0], rows[-1]],
                          missing_both_feet_contact_frames=sum(not any(
                              v["geometric_contact_within_5mm"] for v in row["feet"].values()) for row in rows))))


if __name__ == "__main__":
    main()
