"""Kinematic contact-consistency report for an HDMI motion dir (motion.npz + meta.json + scene.json).

Reports, against the scene boxes:
  - penetration: most negative signed distance between any robot collision geom and any box
  - stance support: for frames where a foot sole is within --h_tol of the surface under it and
    nearly still vertically, the fraction of sole corners actually over that surface
  - stance slide: horizontal ankle travel while a foot stays in stance
Usage: python scripts/vita/check_motion_contacts.py DIR [DIR ...]
"""
import argparse
import json
import os

import mujoco
import numpy as np

from fix_contacts import Robot, segments, sole_world
from umr_to_motion import DEFAULT_XML, support_height


def report(path, h_tol, vz_tol):
    mo = np.load(os.path.join(path, "motion.npz"))
    meta = json.load(open(os.path.join(path, "meta.json")))
    boxes = json.load(open(os.path.join(path, "scene.json")))["boxes"]
    fps = float(meta["fps"])
    spec = mujoco.MjSpec.from_file(DEFAULT_XML)
    for b in boxes:
        g = spec.worldbody.add_geom()
        g.name = "chk_" + b["name"]
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = (np.asarray(b["size"]) / 2).tolist()
        g.pos = b["pos"]
        g.quat = b["quat_wxyz"]
    r = Robot(spec.compile(), meta["joint_names"], meta["body_names"])
    m, d = r.m, r.d
    bg = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "chk_" + b["name"]) for b in boxes]
    rg = [g for g in r.collide if g not in bg]
    q = np.concatenate([mo["body_pos_w"][:, 0], mo["body_quat_w"][:, 0], mo["joint_pos"]], 1).astype(np.float64)
    T = len(q)
    pen = np.zeros(T)
    pen_body = [""] * T
    sole_low = np.zeros((T, 2))
    support_frac = np.full((T, 2), np.nan)
    ankle = np.zeros((T, 2, 3))
    ft = np.zeros(6)
    for t in range(T):
        r.set(q[t])
        best = 1.0
        for g2 in bg:
            for g1 in rg:
                dist = mujoco.mj_geomDistance(m, d, g1, g2, 0.05, ft)
                if dist < best:
                    best, pen_body[t] = dist, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g1])
        pen[t] = best
        for k in range(2):
            pts = r.sole_w(k)
            ankle[t, k] = r.foot_pose(k)[0]
            sole_low[t, k] = pts[:, 2].min()
            sup = support_height(pts, boxes)
            surf = sup.max()
            if sole_low[t, k] - surf < h_tol:
                support_frac[t, k] = (np.abs(pts[:, 2] - sup) < h_tol).mean()
    vz = np.gradient(sole_low, 1.0 / fps, axis=0)
    stance = ~np.isnan(support_frac) & (np.abs(vz) < vz_tol)
    print(f"== {path}")
    i = pen.argmin()
    print(f"  penetration: worst {pen[i] * 1000:.1f} mm (frame {i}, {pen_body[i]}); frames < -10 mm: {(pen < -0.01).sum()}/{T}")
    for k, name in enumerate(("left", "right")):
        sf = support_frac[stance[:, k], k]
        slides = []
        for a, b, s in segments(stance[:, k].astype(int)):
            if s == 1 and b - a >= 4:
                slides.append(np.linalg.norm(ankle[a:b, k, :2] - ankle[a, k, :2], axis=1).max())
        print(f"  {name}: stance frames {stance[:, k].sum()}, sole support mean {np.nanmean(sf):.2f}, "
              f"frames <50% supported {(sf < 0.5).sum()}, stance slide max {max(slides) if slides else 0:.3f} m "
              f"(per stance: {', '.join(f'{s:.2f}' for s in slides)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--h_tol", type=float, default=0.03)
    ap.add_argument("--vz_tol", type=float, default=0.25)
    a = ap.parse_args()
    for p in a.dirs:
        report(p, a.h_tol, a.vz_tol)
