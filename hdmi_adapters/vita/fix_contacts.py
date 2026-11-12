"""Make a kinematic reference contact-consistent with its static box scene.

Video reconstructions let stance feet glide and place footholds partly off the edge of steps. This
re-plans the feet and re-solves the legs:
  1. stance detection per foot: sole near a support level (ground or a box top) with low vertical speed
  2. each stance gets one fixed foothold: the touchdown ankle xy, pushed (minimally) so the whole sole
     lies on the matching box top with an inward margin (or off every box when on the ground), and the
     sole height snapped to that surface
  3. swing feet follow the reference plus an offset blended between neighbouring footholds, raised where
     the sole would cut into a box
  4. root xy/z follows the stance-weighted foot offset (smoothed); damped least-squares IK on the root
     translation and the 12 leg joints tracks both foot poses (reference foot orientation kept)
Upper body joints, root orientation and timing are untouched.

Usage:
    python scripts/vita/fix_contacts.py --in_dir data/motion/vita/stair_rgb_145317 \
        --out_dir data/motion/vita/stair_rgb_145317_fixed
"""
import argparse
import json
import os
import shutil

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as sRot

from umr_to_motion import DEFAULT_XML, FOOT_BODIES, box_rot, finite_diff_ang_vel, support_height

LEG_JOINT_KEYS = ("hip", "knee", "ankle")


def load_model(xml, joint_names):
    m = xml if isinstance(xml, mujoco.MjModel) else mujoco.MjModel.from_xml_path(xml)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
    assert min(jid) >= 0
    free = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]
    return m, jid, free


class Robot:
    def __init__(self, xml, joint_names, body_names):
        self.m, self.jid, free = load_model(xml, joint_names)
        m = self.m
        self.d = mujoco.MjData(m)
        self.fadr = m.jnt_qposadr[free]
        self.fdof = m.jnt_dofadr[free]
        self.qadr = np.array([m.jnt_qposadr[j] for j in self.jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in self.jid])
        self.body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
        self.foot_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in FOOT_BODIES]
        self.leg = [i for i, n in enumerate(joint_names) if any(k in n for k in LEG_JOINT_KEYS)]
        self.lo = np.array([m.jnt_range[j][0] for j in self.jid])
        self.hi = np.array([m.jnt_range[j][1] for j in self.jid])
        limited = np.array([m.jnt_limited[j] for j in self.jid]).astype(bool)
        self.lo[~limited], self.hi[~limited] = -np.inf, np.inf
        # sole = bottom-face corners of the foot collision boxes (main block and toe/heel pads differ by ~2 mm),
        # in the foot body frame
        self.sole = []
        for fb in self.foot_ids:
            pts = []
            for g in range(m.ngeom):
                if m.geom_bodyid[g] != fb or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                    continue
                R = sRot.from_quat(m.geom_quat[g][[1, 2, 3, 0]])
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            pts.append(m.geom_pos[g] + R.apply(np.array([sx, sy, sz]) * m.geom_size[g]))
            pts = np.array(pts)
            self.sole.append(pts[pts[:, 2] <= pts[:, 2].min() + 0.006])
        self.collide = [g for g in range(m.ngeom) if m.geom_bodyid[g] != 0 and (m.geom_contype[g] or m.geom_conaffinity[g])]

    def set(self, q):
        self.d.qpos[:] = self.m.qpos0
        self.d.qpos[self.fadr:self.fadr + 7] = q[:7]
        self.d.qpos[self.qadr] = q[7:]
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)

    def foot_pose(self, k):
        b = self.foot_ids[k]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def sole_w(self, k, pos=None, mat=None):
        p, R = self.foot_pose(k)
        p = p if pos is None else pos
        R = R if mat is None else mat
        return p + self.sole[k] @ R.T


def sole_world(sole_local, pos, mat):
    return pos + sole_local @ mat.T


def detect_stance(sole_low_z, levels, fps, h_tol, vz_tol, min_len, fill_gap, foot_xy=None, vxy_tol_raised=np.inf,
                  strict=True):
    """Returns per-frame level index (-1 = swing; level 0 = ground).

    On raised levels a foot gliding faster than vxy_tol_raised is treated as swing: reconstructions often
    turn a step down/along a tier into a glide at tier height. On the ground the same glides are kept as
    stance (they are whole-body slides, removed by pinning)."""
    T = len(sole_low_z)
    vz = np.gradient(sole_low_z, 1.0 / fps)
    vxy = np.zeros(T) if foot_xy is None else np.linalg.norm(np.gradient(foot_xy, 1.0 / fps, axis=0), axis=1)
    lev = np.full(T, -1)
    below = 0.5 * h_tol if strict else h_tol  # a sole clearly below a surface is not standing on it

    def on_level(t, i, tol):
        return (-below < sole_low_z[t] - levels[i] < tol and abs(vz[t]) < vz_tol
                and (i == 0 or not strict or vxy[t] < vxy_tol_raised))

    for t in range(T):
        i = int(np.abs(sole_low_z[t] - levels).argmin())
        if on_level(t, i, h_tol):
            lev[t] = i
    # close short gaps inside a stance on the same level, then drop blips
    t = 0
    while t < T:
        if lev[t] == -1:
            s = t
            while t < T and lev[t] == -1:
                t += 1
            if s > 0 and t < T and t - s <= fill_gap and lev[s - 1] == lev[t]:
                lev[s:t] = lev[t]
        else:
            t += 1
    # hysteresis: grow each stance while the sole stays within 1.5x the tolerance of the same level
    for a, b, l in segments(lev):
        if l < 0:
            continue
        for rng in (range(a - 1, -1, -1), range(b, T)):
            for t in rng:
                if lev[t] != -1 or not on_level(t, l, 1.5 * h_tol):
                    break
                lev[t] = l
    for a, b, l in segments(lev):
        if l >= 0 and b - a < min_len:
            lev[a:b] = -1
    return lev


def segments(lev):
    out, a = [], 0
    for t in range(1, len(lev) + 1):
        if t == len(lev) or lev[t] != lev[a]:
            out.append((a, t, int(lev[a])))
            a = t
    return out


def place_on_box(sole_xy, box, margin, max_iter=50):
    """Smallest xy shift putting every sole point at least `margin` inside the box footprint."""
    R = box_rot(box)
    c = np.asarray(box["pos"])
    half = np.asarray(box["size"])[:2] / 2 - margin
    loc = R.inv().apply(np.c_[sole_xy - c[:2], np.zeros(len(sole_xy))])[:, :2]
    shift = np.zeros(2)
    for ax in (0, 1):
        lo, hi = loc[:, ax].min(), loc[:, ax].max()
        if hi - lo > 2 * half[ax]:
            shift[ax] = -(lo + hi) / 2
        elif lo < -half[ax]:
            shift[ax] = -half[ax] - lo
        elif hi > half[ax]:
            shift[ax] = half[ax] - hi
    return R.apply(np.r_[shift, 0.0])[:2]


def push_off_boxes(sole_xy, boxes, margin):
    """Smallest xy shift moving the sole out of every box footprint (foot on the ground)."""
    total = np.zeros(2)
    for _ in range(3):
        moved = False
        for b in boxes:
            R = box_rot(b)
            c = np.asarray(b["pos"])
            half = np.asarray(b["size"])[:2] / 2 + margin
            loc = R.inv().apply(np.c_[sole_xy + total - c[:2], np.zeros(len(sole_xy))])[:, :2]
            inside = (np.abs(loc) < half).all(1)
            if not inside.any():
                continue
            # candidate exits along each local axis / direction; keep the shortest
            best = None
            for ax in (0, 1):
                for sgn in (-1, 1):
                    d = (half[ax] - sgn * loc[:, ax]).max() if sgn > 0 else (half[ax] + loc[:, ax]).max()
                    v = np.zeros(2)
                    v[ax] = sgn * d
                    if best is None or abs(d) < np.linalg.norm(best):
                        best = v
            total += R.apply(np.r_[best, 0.0])[:2]
            moved = True
        if not moved:
            break
    return total


def solve_ik(robot, q0, root_target, foot_targets, iters, w_rot, w_root, w_reg, damping):
    q = q0.copy()
    m, d = robot.m, robot.d
    leg = robot.leg
    nv = 3 + len(leg)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    cols = np.r_[robot.fdof + np.arange(3), robot.dadr[leg]]
    for _ in range(iters):
        robot.set(q)
        rows, errs = [], []
        for k in range(2):
            p, R = robot.foot_pose(k)
            pt, Rt = foot_targets[k]
            mujoco.mj_jacBody(m, d, jacp, jacr, robot.foot_ids[k])
            rows.append(jacp[:, cols])
            errs.append(pt - p)
            rows.append(w_rot * jacr[:, cols])
            errs.append(w_rot * (sRot.from_matrix(Rt) * sRot.from_matrix(R).inv()).as_rotvec())
        Jr = np.zeros((3, nv))
        Jr[:, :3] = np.eye(3)
        rows.append(w_root * Jr)
        errs.append(w_root * (root_target - q[:3]))
        Jq = np.zeros((len(leg), nv))
        Jq[:, 3:] = np.eye(len(leg))
        rows.append(w_reg * Jq)
        errs.append(w_reg * (q0[7:][leg] - q[7:][leg]))
        J = np.concatenate(rows)
        e = np.concatenate(errs)
        dq = np.linalg.solve(J.T @ J + damping * np.eye(nv), J.T @ e)
        q[:3] += dq[:3]
        qj = q[7:]
        qj[leg] = np.clip(qj[leg] + dq[3:], robot.lo[leg], robot.hi[leg])
        if np.abs(dq).max() < 1e-5:
            break
    robot.set(q)
    res = [np.linalg.norm(robot.foot_pose(k)[0] - foot_targets[k][0]) for k in range(2)]
    return q, max(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--robot_xml", default=DEFAULT_XML)
    ap.add_argument("--h_tol", type=float, default=0.06, help="sole height above a support level counted as stance (m)")
    ap.add_argument("--vz_tol", type=float, default=0.25, help="max |sole vertical speed| in stance (m/s)")
    ap.add_argument("--min_stance", type=int, default=6, help="frames")
    ap.add_argument("--vxy_tol_raised", type=float, default=0.6, help="max horizontal foot speed for stance on a box top (m/s)")
    ap.add_argument("--max_shift", type=float, default=0.4,
                    help="a stance whose foothold would move further than this to reach its surface is treated as swing (m)")
    ap.add_argument("--fill_gap", type=int, default=3, help="frames")
    ap.add_argument("--edge_margin", type=float, default=0.03, help="sole kept this far inside a box top / outside a box (m)")
    ap.add_argument("--swing_clear", type=float, default=0.03, help="min sole clearance over boxes during swing (m)")
    ap.add_argument("--root_sigma", type=float, default=3.0, help="gaussian smoothing of the root offset (frames)")
    ap.add_argument("--ik_iters", type=int, default=40)
    ap.add_argument("--flat_blend", type=int, default=6,
                    help="frames over which a stance foot is rotated to/from flat (yaw kept) at touchdown/liftoff")
    ap.add_argument("--seated_h", type=float, default=0.3,
                    help="root less than this above a box top counts as seated: the root then keeps its reference position")
    args = ap.parse_args()

    motion = np.load(os.path.join(args.in_dir, "motion.npz"))
    meta = json.load(open(os.path.join(args.in_dir, "meta.json")))
    scene = json.load(open(os.path.join(args.in_dir, "scene.json")))
    boxes = scene["boxes"]
    fps = float(meta["fps"])
    dt = 1.0 / fps
    body_names, joint_names = meta["body_names"], meta["joint_names"]
    robot = Robot(args.robot_xml, joint_names, body_names)
    T = motion["joint_pos"].shape[0]
    q_ref = np.concatenate([motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"]], 1).astype(np.float64)

    # reference feet
    fpos = np.zeros((T, 2, 3))
    fmat = np.zeros((T, 2, 3, 3))
    for t in range(T):
        robot.set(q_ref[t])
        for k in range(2):
            fpos[t, k], fmat[t, k] = robot.foot_pose(k)
    sole_low = np.array([[sole_world(robot.sole[k], fpos[t, k], fmat[t, k])[:, 2].min() for k in range(2)] for t in range(T)])

    level_z = np.array([0.0] + [b["pos"][2] + b["size"][2] / 2 for b in boxes])
    level_box = [None] + boxes

    tgt = fpos.copy()
    tmat = fmat.copy()
    offs = np.zeros((T, 2, 3))
    stance = np.zeros((T, 2), bool)
    report = []
    # strict stance (no glide, not below the surface) wherever possible; the looser test only where the other
    # foot has no strict stance, so the reference never loses all support because of the strict rules
    lev_s, lev_l = [], []
    for k in range(2):
        kw = dict(foot_xy=fpos[:, k, :2], vxy_tol_raised=args.vxy_tol_raised)
        lev_s.append(detect_stance(sole_low[:, k], level_z, fps, args.h_tol, args.vz_tol, args.min_stance,
                                   args.fill_gap, strict=True, **kw))
        lev_l.append(detect_stance(sole_low[:, k], level_z, fps, args.h_tol, args.vz_tol, args.min_stance,
                                   args.fill_gap, strict=False, **kw))
    levs = []
    for k in range(2):
        lev = np.where(lev_s[k] >= 0, lev_s[k], np.where(lev_s[1 - k] < 0, lev_l[k], -1))
        for a, b, l in segments(lev):
            if l >= 0 and b - a < args.min_stance:
                lev[a:b] = -1
        levs.append(lev)
    for k in range(2):
        lev = levs[k]
        segs = segments(lev)
        anchors = []  # (a, b, offset at a, offset at b-1)
        for a, b, l in segs:
            if l < 0:
                continue
            stance[a:b, k] = True
            # reference stance feet glide; the median position is the least biased foothold
            anchor = np.median(fpos[a:b, k], axis=0)
            sole_xy = np.stack([sole_world(robot.sole[k], np.r_[anchor[:2], fpos[t, k, 2]], fmat[t, k])[:, :2]
                                for t in range(a, b)]).reshape(-1, 2)
            if level_box[l] is None:
                shift = push_off_boxes(sole_xy, boxes, args.edge_margin)
            else:
                shift = place_on_box(sole_xy, level_box[l], args.edge_margin)
            if np.linalg.norm(shift) > args.max_shift:
                lev[a:b] = -1
                report.append((FOOT_BODIES[k], a, b, "REJECTED", 0.0, float(np.linalg.norm(shift)), 0.0))
                continue
            anchor[:2] += shift
            # stance foot: roll/pitch removed (reference yaw kept, so no twist is introduced), blended in/out so
            # touchdown and toe-off keep their tilt
            for t in range(a, b):
                r_flat = sRot.from_euler("z", np.arctan2(fmat[t, k, 1, 0], fmat[t, k, 0, 0]))
                wa = 1.0 if a == 0 else min(1.0, (t - a + 1) / args.flat_blend)
                wb = 1.0 if b == T else min(1.0, (b - t) / args.flat_blend)
                wgt = min(wa, wb)
                r_ref = sRot.from_matrix(fmat[t, k])
                rel = (r_flat * r_ref.inv()).as_rotvec()
                tmat[t, k] = (sRot.from_rotvec(wgt * rel) * r_ref).as_matrix()
                tgt[t, k, :2] = anchor[:2]
                tgt[t, k, 2] = level_z[l] - (robot.sole[k] @ tmat[t, k].T)[:, 2].min()
            dz = float(np.median(tgt[a:b, k, 2] - fpos[a:b, k, 2]))
            offs[a:b, k] = tgt[a:b, k] - fpos[a:b, k]
            slide = np.linalg.norm(fpos[a:b, k, :2] - fpos[a, k, :2], axis=1).max()
            report.append((FOOT_BODIES[k], a, b, "ground" if l == 0 else boxes[l - 1]["name"], slide,
                           float(np.linalg.norm(shift)), float(dz)))
            anchors.append((a, b))
        # swing: blend offsets between neighbouring stances (smoothstep)
        segs = segments(lev)
        for i, (a, b, l) in enumerate(segs):
            if l >= 0:
                continue
            o0 = offs[a - 1, k] if a > 0 else (offs[b, k] if b < T else np.zeros(3))
            o1 = offs[b, k] if b < T else o0
            s = (np.arange(a, b) - (a - 1)) / (b - a + 1)
            s = s * s * (3 - 2 * s)
            offs[a:b, k] = o0[None] + (o1 - o0)[None] * s[:, None]
            tgt[a:b, k] = fpos[a:b, k] + offs[a:b, k]
        # swing clearance over box edges (sole points sampled with a small inflation)
        raise_z = np.zeros(T)
        for t in range(T):
            if stance[t, k]:
                continue
            pts = sole_world(robot.sole[k], tgt[t, k], tmat[t, k])
            infl = np.concatenate([pts + np.array([dx, dy, 0]) for dx in (-0.02, 0, 0.02) for dy in (-0.02, 0, 0.02)])
            need = (support_height(infl, boxes).max() + args.swing_clear) - pts[:, 2].min()
            raise_z[t] = max(0.0, need) if support_height(infl, boxes).max() > 0 else 0.0
        if raise_z.any():
            from scipy.ndimage import maximum_filter1d
            r = gaussian_filter1d(maximum_filter1d(raise_z, 5), 1.5)
            r[stance[:, k]] = 0.0
            tgt[:, k, 2] += np.maximum(r, raise_z)
            offs[:, k, 2] = tgt[:, k, 2] - fpos[:, k, 2]

    # when sitting, the seat carries the pelvis and the lightly loaded feet may slide: keep the reference root
    # and reference foot xy (heights stay corrected)
    root_sup = support_height(q_ref[:, :3], boxes)
    seated = ((root_sup > 0) & (q_ref[:, 2] - root_sup < args.seated_h)).astype(float)
    seat_w = np.clip(gaussian_filter1d(seated, args.root_sigma, mode="nearest") * 2.0, 0.0, 1.0)
    offs[..., :2] *= (1.0 - seat_w)[:, None, None]
    tgt[..., :2] = fpos[..., :2] + offs[..., :2]

    # root offset: stance feet dominate
    w = np.where(stance, 1.0, 0.25)
    root_off = (offs * w[..., None]).sum(1) / w.sum(1)[:, None]
    root_off *= (1.0 - seat_w)[:, None]
    root_off = gaussian_filter1d(root_off, args.root_sigma, axis=0, mode="nearest")
    root_tgt = q_ref[:, :3] + root_off

    q_new = q_ref.copy()
    q_prev = None
    ik_res = np.zeros(T)
    for t in range(T):
        q0 = q_ref[t].copy()
        if q_prev is not None:
            q0[:3] = q_prev[:3] + (root_tgt[t] - root_tgt[t - 1])
            q0[7:][robot.leg] = q_prev[7:][robot.leg]
        q0_reg = q_ref[t].copy()
        q0_reg[:3] = q0[:3]
        qs, ik_res[t] = solve_ik(robot, q0, root_tgt[t], [(tgt[t, k], tmat[t, k]) for k in range(2)],
                                 args.ik_iters, w_rot=0.3, w_root=0.1, w_reg=0.03, damping=1e-4)
        # keep the regularisation anchor at the reference joints, not the warm start
        q_new[t] = qs
        q_prev = qs

    # FK + velocities (same conventions as umr_to_motion)
    nb = len(body_names)
    body_pos = np.zeros((T, nb, 3))
    body_com = np.zeros((T, nb, 3))
    body_quat = np.zeros((T, nb, 4))
    for t in range(T):
        robot.set(q_new[t])
        body_pos[t] = robot.d.xpos[robot.body_ids]
        body_com[t] = robot.d.xipos[robot.body_ids]
        body_quat[t] = robot.d.xquat[robot.body_ids]
    for t in range(1, T):
        flip = (body_quat[t] * body_quat[t - 1]).sum(-1) < 0
        body_quat[t, flip] *= -1
    body_lin_vel = np.gradient(body_com, dt, axis=0)
    body_ang_vel = finite_diff_ang_vel(body_quat, dt)
    joint_pos = q_new[:, 7:]
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    n0, n1 = scene.get("pad_start_frames", 0), scene.get("pad_end_frames", 0)
    for sl in (slice(0, n0), slice(T - n1, T)):
        body_lin_vel[sl] = 0.0
        body_ang_vel[sl] = 0.0
        joint_vel[sl] = 0.0

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(args.out_dir, "motion.npz"),
                        body_pos_w=body_pos.astype(np.float32), body_quat_w=body_quat.astype(np.float32),
                        body_lin_vel_w=body_lin_vel.astype(np.float32), body_ang_vel_w=body_ang_vel.astype(np.float32),
                        joint_pos=joint_pos.astype(np.float32), joint_vel=joint_vel.astype(np.float32))
    shutil.copy(os.path.join(args.in_dir, "meta.json"), os.path.join(args.out_dir, "meta.json"))
    scene_out = dict(scene)
    scene_out["contact_fix"] = {"source_motion_dir": os.path.abspath(args.in_dir), **{k: v for k, v in vars(args).items()
                                                                                       if k not in ("in_dir", "out_dir")}}
    json.dump(scene_out, open(os.path.join(args.out_dir, "scene.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(args.out_dir, "contact_fix_debug.npz"), stance=stance, foot_target=tgt,
                        foot_ref=fpos, root_offset=root_off, ik_residual=ik_res)

    # diagnostics
    print("stance segments (foot, frames, surface, ref slide m, foothold shift m, dz m):")
    for r in report:
        print(f"  {r[0]:22s} {r[1]:4d}-{r[2]:4d} {r[3]:11s} slide {r[4]:.3f} shift {r[5]:.3f} dz {r[6]:+.3f}")
    print(f"IK foot residual: median {np.median(ik_res) * 1000:.1f} mm, max {ik_res.max() * 1000:.1f} mm (frame {ik_res.argmax()})")
    print(f"root offset max |xy| {np.linalg.norm(root_off[:, :2], axis=1).max():.3f} m, |z| {np.abs(root_off[:, 2]).max():.3f} m")
    print(f"max |joint_vel| {np.abs(joint_vel).max():.2f} rad/s; max root speed {np.linalg.norm(body_lin_vel[:, 0], axis=-1).max():.2f} m/s")


if __name__ == "__main__":
    main()
