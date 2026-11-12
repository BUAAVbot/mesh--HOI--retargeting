"""Convert a UMR retarget result (qpos @ fps, MuJoCo floating base) into an HDMI motion dir.

Output dir layout:
    motion.npz  body_pos_w, body_quat_w (wxyz), body_lin_vel_w, body_ang_vel_w [T,B,*], joint_pos, joint_vel [T,J]
    meta.json   body_names, joint_names, fps
    scene.json  static boxes (env-local frame) + the xy offset applied to everything

Usage:
    python scripts/vita/umr_to_motion.py --umr_npz X_smoothed.npz --out_dir data/motion/vita/stair \
        --scene_json terrain_proxies.json [--xy_offset X Y] [--pad_start 0.5 --pad_end 1.0]
"""
import argparse
import json
import os

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as sRot

DEFAULT_XML = "/vita-vepfs-data/fileset1/usr_data/yunhao.xia/code/ScaleBFM/ScaleRetarget/assets/vita02a_shoes/vita02a_gmr.xml"
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def load_boxes(scene_json, scale=1.0):
    """UMR scene proxies are stored in reconstruction units; UMR places them in the robot world
    multiplied by the sequence smpl_scale (centre and size), same as its preview scene.xml."""
    if scene_json is None:
        return []
    with open(scene_json) as f:
        items = json.load(f)
    boxes = []
    for it in items:
        if "quat_xyzw" in it:
            x, y, z, w = it["quat_xyzw"]
            quat_wxyz = [w, x, y, z]
        else:
            quat_wxyz = [1.0, 0.0, 0.0, 0.0]
        boxes.append({
            "name": it["name"],
            "pos": [float(v) * scale for v in it["center_zup_m"]],
            "size": [float(v) * scale for v in it["size_m"]],
            "quat_wxyz": [float(v) for v in quat_wxyz],
        })
    return boxes


def box_rot(box):
    w, x, y, z = box["quat_wxyz"]
    return sRot.from_quat([x, y, z, w])


def crop_box(box, pts_xy, margin):
    """Shrink a box along its local x/y axes to the extent of pts_xy (+margin). Physics is unchanged
    wherever the robot actually goes, and env spacing can stay small."""
    R = box_rot(box)
    c = np.asarray(box["pos"])
    half = np.asarray(box["size"]) / 2
    local = R.inv().apply(np.c_[pts_xy - c[:2], np.zeros(len(pts_xy))])
    new_c_local = np.zeros(3)
    new_half = half.copy()
    for ax in (0, 1):
        lo = max(-half[ax], local[:, ax].min() - margin)
        hi = min(half[ax], local[:, ax].max() + margin)
        if hi <= lo:  # trajectory never comes near this box along this axis
            return None
        new_c_local[ax] = 0.5 * (lo + hi)
        new_half[ax] = 0.5 * (hi - lo)
    out = dict(box)
    out["pos"] = (c + R.apply(new_c_local)).tolist()
    out["size"] = (2 * new_half).tolist()
    return out


def support_height(pts, boxes):
    """Height of the highest support surface (ground=0 or box top) under each point."""
    h = np.zeros(len(pts))
    for b in boxes:
        R = box_rot(b)
        local = R.inv().apply(pts - np.asarray(b["pos"]))
        half = np.asarray(b["size"]) / 2
        inside = (np.abs(local[:, 0]) <= half[0]) & (np.abs(local[:, 1]) <= half[1])
        top = b["pos"][2] + half[2]
        h = np.where(inside, np.maximum(h, top), h)
    return h


def finite_diff_ang_vel(quat_wxyz, dt):
    """World-frame angular velocity via central differences. quat_wxyz: [T,B,4]."""
    T, B, _ = quat_wxyz.shape
    q = quat_wxyz.reshape(-1, 4)[:, [1, 2, 3, 0]]
    R = sRot.from_quat(q)
    Rs = [sRot.from_quat(quat_wxyz[t][:, [1, 2, 3, 0]]) for t in range(T)]
    w = np.zeros((T, B, 3))
    for t in range(T):
        t0, t1 = max(t - 1, 0), min(t + 1, T - 1)
        if t1 == t0:
            continue
        dR = Rs[t1] * Rs[t0].inv()
        w[t] = dR.as_rotvec() / ((t1 - t0) * dt)
    return w


def fit_box_tops(xml, qpos_full, fadr, qadr, boxes, margin, stride=1):
    """Lower each ground-standing box so no robot collision geom penetrates its top during the clip.

    Box heights from video-based terrain proxies are the least reliable part of the scene; lowering
    (never raising) a box only removes interpenetration and at most leaves a small air gap.
    Returns {box_name: lowered_by_m}."""
    spec = mujoco.MjSpec.from_file(xml)
    for b in boxes:
        g = spec.worldbody.add_geom()
        g.name = "fitbox_" + b["name"]
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = (np.asarray(b["size"]) / 2).tolist()
        g.pos = b["pos"]
        g.quat = b["quat_wxyz"]
    m = spec.compile()
    d = mujoco.MjData(m)
    bg = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "fitbox_" + b["name"]) for b in boxes]
    rg = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]
    dmin = np.full(len(boxes), np.inf)
    ft = np.zeros(6)
    for t in range(0, len(qpos_full), stride):
        d.qpos[fadr:fadr + 7] = qpos_full[t, :7]
        d.qpos[qadr] = qpos_full[t, 7:]
        mujoco.mj_kinematics(m, d)
        for i, g2 in enumerate(bg):
            for g1 in rg:
                dmin[i] = min(dmin[i], mujoco.mj_geomDistance(m, d, g1, g2, 0.1, ft))
    lowered = {}
    for b, dm in zip(boxes, dmin):
        drop = max(0.0, margin - dm) if np.isfinite(dm) else 0.0
        if drop > 0:
            b["size"][2] -= drop
            b["pos"][2] -= drop / 2
        lowered[b["name"]] = float(drop)
    return lowered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umr_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--robot_xml", default=DEFAULT_XML)
    ap.add_argument("--scene_json", default=None)
    ap.add_argument("--xy_offset", type=float, nargs=2, default=None,
                    help="subtracted from all world xy; default = mean root xy")
    ap.add_argument("--pad_start", type=float, default=0.5, help="seconds holding the first frame")
    ap.add_argument("--pad_end", type=float, default=1.0, help="seconds holding the last frame")
    ap.add_argument("--crop_margin", type=float, default=1.0)
    ap.add_argument("--z_lift", type=float, default=0.0,
                    help="raise robot and props (e.g. 0.008 when UMR placed its floor at z=-0.008)")
    ap.add_argument("--fit_box_tops", action="store_true", help="lower boxes until no body penetrates them")
    ap.add_argument("--fit_margin", type=float, default=0.002, help="target clearance for --fit_box_tops (m)")
    ap.add_argument("--pin_feet", action="store_true",
                    help="remove stance-foot sliding by translating the root (cumulative xy correction)")
    ap.add_argument("--pin_contact_h", type=float, default=0.03, help="foot sole height above support counted as contact (m)")
    ap.add_argument("--pin_contact_vz", type=float, default=0.2, help="max |vertical foot speed| counted as contact (m/s)")
    args = ap.parse_args()

    d = np.load(args.umr_npz, allow_pickle=True)
    qpos = d["qpos"].astype(np.float64)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    umr_joint_names = [str(n) for n in d["robot_joint_names"]]
    dt = 1.0 / fps

    n0, n1 = int(round(args.pad_start * fps)), int(round(args.pad_end * fps))
    qpos = np.concatenate([np.repeat(qpos[:1], n0, 0), qpos, np.repeat(qpos[-1:], n1, 0)], 0)
    T = qpos.shape[0]

    xy_offset = np.asarray(args.xy_offset) if args.xy_offset is not None else qpos[:, :2].mean(0)
    qpos[:, :2] -= xy_offset
    qpos[:, 2] += args.z_lift

    m = mujoco.MjModel.from_xml_path(args.robot_xml)
    data = mujoco.MjData(m)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in umr_joint_names]
    assert min(jid) >= 0, "joint name mismatch between UMR result and robot xml"
    qadr = [m.jnt_qposadr[j] for j in jid]
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    assert len(free) == 1
    fadr = m.jnt_qposadr[free[0]]

    # bodies kept = floating root + every body driven by one of the 28 joints (fixed links get merged in Isaac)
    root_body = m.jnt_bodyid[free[0]]
    body_ids = [root_body] + [m.jnt_bodyid[j] for j in jid]
    body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in body_ids]

    foot_geoms = [g for g in range(m.ngeom)
                  if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) in FOOT_BODIES
                  and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_contype[g] != 0]
    if not foot_geoms:  # visual/collision split may use conaffinity only
        foot_geoms = [g for g in range(m.ngeom)
                      if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) in FOOT_BODIES
                      and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]

    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    foot_body_idx = [body_names.index(fb) for fb in FOOT_BODIES]

    def forward_kinematics(qpos):
        body_pos = np.zeros((T, len(body_ids), 3))
        body_com = np.zeros((T, len(body_ids), 3))
        body_quat = np.zeros((T, len(body_ids), 4))
        foot_low = np.zeros((T, 2, 3))  # lowest corner per foot
        for t in range(T):
            data.qpos[:] = m.qpos0
            data.qpos[fadr:fadr + 7] = qpos[t, :7]
            data.qpos[qadr] = qpos[t, 7:]
            mujoco.mj_kinematics(m, data)
            body_pos[t] = data.xpos[body_ids]
            body_com[t] = data.xipos[body_ids]
            body_quat[t] = data.xquat[body_ids]
            for k, fb in enumerate(FOOT_BODIES):
                corners = []
                for g in foot_geoms:
                    if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) != fb:
                        continue
                    c = data.geom_xpos[g] + (signs * m.geom_size[g]) @ data.geom_xmat[g].reshape(3, 3).T
                    corners.append(c)
                corners = np.concatenate(corners, 0)
                foot_low[t, k] = corners[corners[:, 2].argmin()]
        return body_pos, body_com, body_quat, foot_low

    scene_scale = float(np.ravel(d["smpl_scale"])[0]) if "smpl_scale" in d.files else 1.0
    boxes = load_boxes(args.scene_json, scene_scale)
    for b in boxes:
        b["pos"][0] -= xy_offset[0]
        b["pos"][1] -= xy_offset[1]
        if args.z_lift:
            # z_lift moves the whole UMR world (robot and props) up, so props are extended down to the new ground
            b["pos"][2] += args.z_lift / 2
            b["size"][2] += args.z_lift

    body_pos, body_com, body_quat, foot_low = forward_kinematics(qpos)

    pin_shift = np.zeros((T, 2))
    if args.pin_feet:
        # Video reconstructions let stance feet glide (and whole-body jumps while both feet are planted),
        # which moves the person relative to the terrain. Undo it causally: whenever a foot is in contact
        # with the support under its corrected position, cancel that foot's xy motion by shifting the root.
        ankle = body_pos[:, foot_body_idx]
        vz = np.gradient(foot_low[..., 2], dt, axis=0)
        for t in range(1, T):
            moves = []
            for k in range(2):
                pts = np.stack([ankle[t, k], foot_low[t, k]]).copy()
                pts[:, :2] += pin_shift[t - 1]
                sup = support_height(pts, boxes).max()
                if foot_low[t, k, 2] - sup < args.pin_contact_h and abs(vz[t, k]) < args.pin_contact_vz:
                    moves.append(ankle[t, k, :2] - ankle[t - 1, k, :2])
            pin_shift[t] = pin_shift[t - 1] - (np.mean(moves, 0) if moves else 0.0)
        qpos[:, :2] += pin_shift
        body_pos, body_com, body_quat, foot_low = forward_kinematics(qpos)

    # quaternion sign continuity for slerp downstream
    for t in range(1, T):
        flip = (body_quat[t] * body_quat[t - 1]).sum(-1) < 0
        body_quat[t, flip] *= -1

    body_lin_vel = np.gradient(body_com, dt, axis=0)
    body_ang_vel = finite_diff_ang_vel(body_quat, dt)
    joint_pos = qpos[:, 7:]
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    # padded hold frames are static by construction
    for sl in (slice(0, n0), slice(T - n1, T)):
        body_lin_vel[sl] = 0.0
        body_ang_vel[sl] = 0.0
        joint_vel[sl] = 0.0

    traj_xy = body_pos[:, :, :2].reshape(-1, 2)
    cropped = []
    for b in boxes:
        cb = crop_box(b, traj_xy, args.crop_margin)
        if cb is not None:
            cropped.append(cb)
    lowered = {}
    if args.fit_box_tops and cropped:
        lowered = fit_box_tops(args.robot_xml, qpos, fadr, qadr, cropped, args.fit_margin)
        print("box tops lowered (mm): " + ", ".join(f"{k}={v * 1000:.1f}" for k, v in lowered.items()))

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out_dir, "motion.npz"),
        body_pos_w=body_pos.astype(np.float32),
        body_quat_w=body_quat.astype(np.float32),
        body_lin_vel_w=body_lin_vel.astype(np.float32),
        body_ang_vel_w=body_ang_vel.astype(np.float32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
    )
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump({"body_names": body_names, "joint_names": umr_joint_names, "fps": int(round(fps))}, f, indent=2)

    all_xy = np.concatenate([traj_xy] + [np.asarray(b["pos"])[None, :2] for b in cropped], 0)
    scene = {
        "source_umr_npz": os.path.abspath(args.umr_npz),
        "source_scene_json": os.path.abspath(args.scene_json) if args.scene_json else None,
        "xy_offset": xy_offset.tolist(),
        "scene_scale": scene_scale,
        "z_lift": args.z_lift,
        "box_tops_lowered_m": lowered,
        "pin_feet": bool(args.pin_feet),
        "pin_shift_final_xy": pin_shift[-1].tolist(),
        "pin_shift_max_m": float(np.linalg.norm(pin_shift, axis=-1).max()),
        "pad_start_frames": n0,
        "pad_end_frames": n1,
        "fps": fps,
        "boxes": cropped,
        "extent_xy": [all_xy.min(0).tolist(), all_xy.max(0).tolist()],
    }
    with open(os.path.join(args.out_dir, "scene.json"), "w") as f:
        json.dump(scene, f, indent=2)

    # diagnostics: foot clearance vs. support surface (negative = penetration)
    sup = support_height(foot_low.reshape(-1, 3), cropped).reshape(T, 2)
    clear = foot_low[..., 2] - sup
    print(f"T={T} frames @ {fps} fps ({T / fps:.2f}s), bodies={len(body_names)}, joints={len(umr_joint_names)}")
    print(f"xy_offset={xy_offset.round(4).tolist()}  root z range=[{qpos[:, 2].min():.3f}, {qpos[:, 2].max():.3f}]")
    print(f"boxes kept {len(cropped)}/{len(boxes)}: " + ", ".join(
        f"{b['name']} size={np.round(b['size'], 3).tolist()} pos={np.round(b['pos'], 3).tolist()}" for b in cropped))
    print(f"scene extent xy: {np.round(scene['extent_xy'], 3).tolist()}")
    for k, fb in enumerate(FOOT_BODIES):
        c = clear[:, k]
        print(f"{fb}: min clearance {c.min() * 1000:.1f} mm (frame {c.argmin()}), "
              f"frames penetrating >5mm: {(c < -0.005).sum()}, support h range [{sup[:, k].min():.3f},{sup[:, k].max():.3f}]")
    print(f"root quat norm range: [{np.linalg.norm(qpos[:, 3:7], axis=1).min():.4f}, {np.linalg.norm(qpos[:, 3:7], axis=1).max():.4f}]")
    print(f"max |joint_vel| {np.abs(joint_vel).max():.2f} rad/s; max root lin speed {np.linalg.norm(body_lin_vel[:, 0], axis=-1).max():.2f} m/s")


if __name__ == "__main__":
    main()
