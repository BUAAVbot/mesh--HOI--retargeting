"""Minimum distance between two independently simulated robots (seat-swap demo).

Both rollouts share the same env-local frame (same xy_offset in scene.json), so they can be overlaid.
Uses the Vita MJCF collision geoms; reports min distance per frame and overall.

Usage: python scripts/vita/check_pair_clearance.py --a EXPORT_DIR_A --b EXPORT_DIR_B [--stride 4]
"""
import argparse
import json
import os

import mujoco
import numpy as np

DISTMAX = 0.3
XML = "/vita-vepfs-data/fileset1/usr_data/yunhao.xia/code/ScaleBFM/ScaleRetarget/assets/vita02a_shoes/vita02a_gmr.xml"


def load(d):
    r = np.load(os.path.join(d, "rollout.npz"))
    m = json.load(open(os.path.join(d, "meta.json")))
    return r, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--stride", type=int, default=4, help="physics frames between checks (4 = 50 Hz)")
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ra, ma = load(args.a)
    rb, mb = load(args.b)

    spec = mujoco.MjSpec()
    for tag in ("A_", "B_"):
        child = mujoco.MjSpec.from_file(args.xml)
        frame = spec.worldbody.add_frame()
        frame.attach_body(child.worldbody.first_body(), tag, "")
    model = spec.compile()
    data = mujoco.MjData(model)

    def free_adr(tag):
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(tag):
                return model.jnt_qposadr[j]
        raise KeyError(tag)

    def joint_adrs(tag, names):
        return [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tag + n)] for n in names]

    fa, fb = free_adr("A_"), free_adr("B_")
    ja, jb = joint_adrs("A_", ma["joint_names"]), joint_adrs("B_", mb["joint_names"])

    def geoms(tag):
        out = []
        for g in range(model.ngeom):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
            if bname.startswith(tag) and (model.geom_contype[g] or model.geom_conaffinity[g]):
                out.append(g)
        return out

    ga, gb = geoms("A_"), geoms("B_")
    T = min(len(ra["time"]), len(rb["time"]))
    frames = list(range(0, T, args.stride))
    dmin = np.full(len(frames), DISTMAX)
    pair = [None] * len(frames)
    fromto = np.zeros(6)
    for i, t in enumerate(frames):
        data.qpos[:] = model.qpos0
        data.qpos[fa:fa + 3] = ra["root_pos_w"][t]; data.qpos[fa + 3:fa + 7] = ra["root_quat_w"][t]
        data.qpos[fb:fb + 3] = rb["root_pos_w"][t]; data.qpos[fb + 3:fb + 7] = rb["root_quat_w"][t]
        data.qpos[ja] = ra["joint_pos"][t]; data.qpos[jb] = rb["joint_pos"][t]
        mujoco.mj_kinematics(model, data)
        # bounding-sphere prefilter, then exact distances. distmax is kept small on purpose: with large distmax,
        # mj_geomDistance returned 0.0 for some far-apart mesh pairs (false contacts at >0.2 m true distance).
        for g1 in ga:
            for g2 in gb:
                if np.linalg.norm(data.geom_xpos[g1] - data.geom_xpos[g2]) - model.geom_rbound[g1] - model.geom_rbound[g2] > DISTMAX:
                    continue
                d = mujoco.mj_geomDistance(model, data, g1, g2, DISTMAX, fromto)
                if d < dmin[i]:
                    dmin[i] = d
                    pair[i] = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g1]),
                               mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g2]))
    k = int(np.argmin(dmin))
    dt = ma["physics_dt"]
    summary = {
        "frames_checked": len(frames),
        "overlap_seconds": T * dt,
        "min_distance_m": float(dmin[k]),
        "min_distance_time_s": frames[k] * dt,
        "closest_bodies": pair[k],
        "frames_below_5cm": int((dmin < 0.05).sum()),
        "frames_in_contact": int((dmin <= 0.0).sum()),
    }
    print("[PAIR]", json.dumps(summary))
    if args.out:
        np.savez(args.out, time=np.array(frames) * dt, min_distance=dmin)


if __name__ == "__main__":
    main()
