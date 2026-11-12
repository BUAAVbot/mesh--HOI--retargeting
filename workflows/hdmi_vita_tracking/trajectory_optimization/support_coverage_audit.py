"""Reject load-bearing support over unobserved reconstructed terrain.

Preferred input is verified actual contact points. The fallback uses exact
200 Hz physical qpos and a deliberately conservative bounding footprint of
each loaded foot; it can reject a valid edge contact but cannot certify a
large unseen hole by looking only at the foot centre.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from acceptance import load_npz, sha256


def audit(record_path, model_path, heightfield_path):
    r = load_npz(record_path)
    with np.load(heightfield_path, allow_pickle=False) as field:
        x, y = np.asarray(field["x"], float), np.asarray(field["y"], float)
        observed = np.asarray(field["observed_mask"], bool)
        filled = np.asarray(field["small_hole_filled_mask"], bool)
        unknown = np.asarray(field["unknown_mask"], bool)
    checks = dict(mask_shape=observed.shape == filled.shape == unknown.shape == (len(y), len(x)),
                  mask_partition=bool(np.array_equal(unknown, ~(observed | filled))),
                  support_filter_verified=bool(np.asarray(r.get("support_filter_verified", False)).reshape(-1)[0]),
                  support_filter_semantics=str(np.asarray(r.get("support_filter_semantics", "")).reshape(-1)[0]) == "exact_per_static_body_excludes_self")
    dx, dy = x[1] - x[0], y[1] - y[0]
    counts = dict(observed=0, small_hole_filled=0, unknown=0, outside=0, total=0)
    bad = []

    def query(points, substep, foot):
        points = np.asarray(points, float)
        ix = np.round((points[:, 0] - x[0]) / dx).astype(int)
        iy = np.round((points[:, 1] - y[0]) / dy).astype(int)
        inside = (ix >= 0) & (ix < len(x)) & (iy >= 0) & (iy < len(y))
        known = np.zeros(len(points), bool)
        oc = observed[iy[inside], ix[inside]]
        fc = filled[iy[inside], ix[inside]]
        known[inside] = oc | fc
        counts["total"] += len(points)
        counts["observed"] += int(oc.sum())
        counts["small_hole_filled"] += int((fc & ~oc).sum())
        counts["outside"] += int((~inside).sum())
        counts["unknown"] += int((inside & ~known).sum())
        if np.any(~known) and len(bad) < 400:
            bad.append(dict(physics_substep=int(substep), control_frame=int(substep // 4 + 1),
                            foot=foot, unsupported_sample_xy=points[np.flatnonzero(~known)[0], :2].tolist(),
                            unknown_or_outside_samples=int((~known).sum())))

    direct_fields = ("physics_support_contact_points_w", "physics_support_contact_force_N", "physics_support_contact_substep")
    direct = (all(key in r for key in direct_fields)
              and bool(np.asarray(r.get("support_contact_points_verified", False)).reshape(-1)[0]))
    dense_fields = ("physics_foot_contact_pos_w", "physics_foot_contact_normal_force_N",
                    "physics_foot_contact_mask", "physics_foot_contact_overflow")
    dense = all(key in r for key in dense_fields)
    if dense:
        mode = "actual_200hz_static_support_contact_points_dense"
        steps = len(r["joint_pos"]) - 1
        points = np.asarray(r[dense_fields[0]], float)
        force = np.asarray(r[dense_fields[1]], float)
        mask = np.asarray(r[dense_fields[2]], bool)
        overflow = np.asarray(r[dense_fields[3]], bool)
        expected = (steps, 4, 2, 128)
        complete = points.shape == expected + (3,) and force.shape == mask.shape == expected and overflow.shape == expected[:3]
        checks["complete_dense_contact_record"] = complete
        checks["actual_contact_points_verified"] = (
            bool(np.asarray(r.get("support_contact_points_verified", False)).reshape(-1)[0])
            and str(np.asarray(r.get("support_contact_points_semantics", "")).reshape(-1)[0]) == "warp_contact_force_static_pairs_force_gt_1N")
        checks["contact_point_overflow_absent"] = not bool(overflow.any())
        checks["dense_contact_record_finite"] = bool(np.isfinite(points).all() and np.isfinite(force).all())
        if complete:
            for substep, (ps, fs, valid) in enumerate(zip(points.reshape(-1, 2, 128, 3),
                                                        force.reshape(-1, 2, 128), mask.reshape(-1, 2, 128))):
                for fi, side in enumerate(("left", "right")):
                    selected = valid[fi] & (fs[fi] > 1.) & np.isfinite(ps[fi]).all(-1)
                    if np.any(selected):
                        query(ps[fi, selected], substep, side)
    elif direct:
        mode = "actual_200hz_static_support_contact_points"
        points = np.asarray(r[direct_fields[0]], float)
        force = np.asarray(r[direct_fields[1]], float)
        substeps = np.asarray(r[direct_fields[2]], int)
        good = (force > 1.) & np.isfinite(force) & np.isfinite(points).all(-1)
        checks["contact_point_shapes"] = points.ndim == 2 and points.shape[1] == 3 and len(points) == len(force) == len(substeps)
        for point, substep in zip(points[good], substeps[good]):
            query(point[None], substep, "actual_contact")
    else:
        mode = "conservative_loaded_foot_200hz_bounding_footprint"
        force = np.asarray(r.get("physics_foot_support_force_N", []), float)
        qpos = np.asarray(r.get("physics_qpos", []), float)
        steps = len(r["joint_pos"]) - 1
        checks["complete_perfoot_force"] = force.shape == (steps, 4, 2) and bool(np.isfinite(force).all())
        checks["complete_substep_qpos"] = qpos.shape == (steps, 4, 35) and bool(np.isfinite(qpos).all())
        if checks["complete_perfoot_force"] and checks["complete_substep_qpos"]:
            m = mujoco.MjModel.from_binary_path(str(model_path))
            d = mujoco.MjData(m)
            feet = {}
            for side in ("left", "right"):
                feet[side] = [g for g in range(m.ngeom)
                              if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "").split("/")[-1] == f"{side}_ankle_roll_link"]
            signs = np.asarray([(a, b, c) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
            for substep, (q, f) in enumerate(zip(qpos.reshape(-1, 35), force.reshape(-1, 2))):
                if not np.any(f > 1.):
                    continue
                d.qpos[:] = q
                mujoco.mj_kinematics(m, d)
                for fi, side in enumerate(("left", "right")):
                    if f[fi] <= 1.:
                        continue
                    vertices = []
                    for gid in feet[side]:
                        if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
                            local = signs * m.geom_size[gid]
                        elif m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                            a = np.arange(16) * (2 * np.pi / 16)
                            circle = np.column_stack((np.cos(a), np.sin(a))) * m.geom_size[gid, 0]
                            local = np.concatenate([np.column_stack((circle, np.full(16, z)))
                                                    for z in (-m.geom_size[gid, 1], m.geom_size[gid, 1])])
                        else:
                            continue
                        vertices.append(local @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid])
                    if not vertices:
                        checks["foot_geometry_present"] = False
                        continue
                    vertices = np.concatenate(vertices)
                    lo, hi = vertices[:, :2].min(0), vertices[:, :2].max(0)
                    # Check every intersected terrain cell, not just corners.
                    ix = np.arange(int(np.floor((lo[0] - x[0]) / dx - .5)), int(np.ceil((hi[0] - x[0]) / dx + .5)) + 1)
                    iy = np.arange(int(np.floor((lo[1] - y[0]) / dy - .5)), int(np.ceil((hi[1] - y[0]) / dy + .5)) + 1)
                    xx, yy = np.meshgrid(x[0] + ix * dx, y[0] + iy * dy)
                    query(np.column_stack((xx.reshape(-1), yy.reshape(-1))), substep, side)
    checks["loadbearing_evidence_present"] = counts["total"] > 0
    checks["unknown_loadbearing_samples_zero"] = counts["unknown"] == 0
    checks["outside_loadbearing_samples_zero"] = counts["outside"] == 0
    passed = bool(all(checks.values()))
    return dict(schema="hdmi.trajopt.support_coverage.v1", passed=passed,
                record_sha256=sha256(record_path), model_sha256=sha256(model_path),
                heightfield_asset_sha256=sha256(heightfield_path),
                mode=mode, force_threshold_N=1., counts=counts, checks=checks,
                failed_checks=[key for key, value in checks.items() if not value],
                first_unknown_support_intervals=bad,
                limits=["Fallback bounds complete foot footprints and is intentionally conservative near holes and edges.",
                        "Observed and explicitly small-hole-filled cells remain distinct; large unknown holes never count as known support."])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--record", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--heightfield", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.record, args.model, args.heightfield)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("passed", "mode", "counts", "failed_checks")}))


if __name__ == "__main__":
    main()
