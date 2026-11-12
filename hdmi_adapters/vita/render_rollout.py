"""Render HDMI rollout exports (sim robot) over the reference motion (translucent ghost) in the box scene.

Usage:
    MUJOCO_GL=egl python scripts/vita/render_rollout.py --rollout EXPORT_DIR [--rollout EXPORT_DIR2 ...] \
        --out video.mp4 [--forces] [--azimuth 135 --elevation -20 --distance 4.5] [--stills 0,4.5,8]
"""
import argparse
import json
import os

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HDMI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
XML = "/vita-vepfs-data/fileset1/usr_data/yunhao.xia/code/ScaleBFM/ScaleRetarget/assets/vita02a_shoes/vita02a_gmr.xml"
SIM_COLORS = [(0.20, 0.45, 0.85, 1.0), (0.90, 0.50, 0.15, 1.0)]
GHOST_COLORS = [(0.30, 0.85, 0.40, 0.28), (0.85, 0.30, 0.80, 0.28)]


def load_export(d):
    r = dict(np.load(os.path.join(d, "rollout.npz")))
    meta = json.load(open(os.path.join(d, "meta.json")))
    scene_json = meta["scene_json"]
    if not os.path.isabs(scene_json):
        scene_json = os.path.join(HDMI_ROOT, scene_json)
    motion_dir = os.path.dirname(scene_json)
    ref = dict(np.load(os.path.join(motion_dir, "motion.npz")))
    ref_meta = json.load(open(os.path.join(motion_dir, "meta.json")))
    scene = json.load(open(scene_json))
    return r, meta, ref, ref_meta, scene


def recolor(child, rgba):
    for g in child.geoms:
        g.rgba = rgba
        g.material = ""
        if rgba[3] < 0.5:
            g.contype = 0
            g.conaffinity = 0


def build_model(n, scene, xml, show_ghost, sim_alpha=1.0):
    spec = mujoco.MjSpec()
    spec.visual.global_.offwidth = 1920
    spec.visual.global_.offheight = 1080
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.75, 0.85, 0.95], rgb2=[0.95, 0.95, 0.97], width=256, height=256)
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.82, 0.82, 0.82], rgb2=[0.70, 0.70, 0.70], width=512, height=512)
    mat = spec.add_material(name="grid")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    mat.texrepeat = [8, 8]
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[8, 8, 0.05], material="grid")
    spec.worldbody.add_light(pos=[0, 0, 6], dir=[0, 0, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
    spec.worldbody.add_light(pos=[3, -3, 4], dir=[-0.5, 0.5, -0.7], diffuse=[0.35, 0.35, 0.35], castshadow=False)
    for b in scene["boxes"]:
        spec.worldbody.add_geom(name="box_" + b["name"], type=mujoco.mjtGeom.mjGEOM_BOX,
                                size=(np.asarray(b["size"]) / 2).tolist(), pos=b["pos"], quat=b["quat_wxyz"],
                                rgba=[0.62, 0.55, 0.45, 1.0])
    tags = []
    for i in range(n):
        for kind in (["S", "R"] if show_ghost else ["S"]):
            child = mujoco.MjSpec.from_file(xml)
            recolor(child, SIM_COLORS[i % 2][:3] + (sim_alpha,) if kind == "S" else GHOST_COLORS[i % 2])
            frame = spec.worldbody.add_frame()
            tag = f"{kind}{i}_"
            frame.attach_body(child.worldbody.first_body(), tag, "")
            tags.append(tag)
    return spec.compile()


def qpos_adr(model, tag, joint_names):
    free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
            and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(tag)]
    assert len(free) == 1, tag
    fa = model.jnt_qposadr[free[0]]
    ja = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, tag + n)] for n in joint_names])
    return fa, ja


def add_arrow(scn, p0, p1, rgba, width=0.012):
    """Capsule shaft p0->p1 plus an arrow head ending at p1 (MuJoCo's ARROW geom draws only half the connector length)."""
    v = p1 - p0
    L = np.linalg.norm(v)
    if scn.ngeom + 2 > scn.maxgeom or L < 1e-3:
        return
    rgba = np.asarray(rgba, np.float32)
    u = v / L
    head = min(0.08, 0.5 * L)
    for typ, a, b, w in ((mujoco.mjtGeom.mjGEOM_CAPSULE, p0, p1 - u * head, width),
                         (mujoco.mjtGeom.mjGEOM_ARROW, p1 - u * head, p1 + u * head, width * 1.6)):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, typ, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        mujoco.mjv_connector(g, typ, w, a, b)
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", action="append", required=True)
    ap.add_argument("--labels", default=None, help="comma separated labels per rollout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--no_ghost", action="store_true")
    ap.add_argument("--forces", action="store_true", help="draw per-point contact forces")
    ap.add_argument("--force_scale", type=float, default=1.0 / 800.0, help="m per N")
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-20.0)
    ap.add_argument("--distance", type=float, default=4.5)
    ap.add_argument("--lookat", type=float, nargs=3, default=None, help="fixed look-at point (default: follow robots)")
    ap.add_argument("--stills", default=None, help="comma separated times (s): write PNGs instead of a video")
    ap.add_argument("--xml", default=XML)
    args = ap.parse_args()

    ex = [load_export(d) for d in args.rollout]
    labels = args.labels.split(",") if args.labels else [os.path.basename(os.path.normpath(d)) for d in args.rollout]
    n = len(ex)
    # arrows start under the feet and pass through the legs, so the robot is see-through when forces are drawn
    model = build_model(n, ex[0][4], args.xml, not args.no_ghost, sim_alpha=0.55 if args.forces else 1.0)
    data = mujoco.MjData(model)
    adr_s = [qpos_adr(model, f"S{i}_", e[1]["joint_names"]) for i, e in enumerate(ex)]
    adr_r = [qpos_adr(model, f"R{i}_", e[3]["joint_names"]) for i, e in enumerate(ex)] if not args.no_ghost else None

    dt = ex[0][1]["physics_dt"]
    policy_dt = ex[0][1]["policy_dt"]
    T_sim = [len(e[0]["time"]) for e in ex]
    motion_T = [e[1]["motion_len_policy_steps"] * policy_dt for e in ex]
    duration = max(motion_T)

    # force sign: PhysX reports the normal of the pair, orient it so ground reaction on feet points up
    pt_index = []
    for e in ex:
        r = e[0]
        fr = r["contact_point_frame"].astype(int)
        order = np.argsort(fr, kind="stable")
        starts = np.searchsorted(fr[order], np.arange(len(r["time"]) + 1))
        sign = 1.0
        if len(fr):
            nz = (r["contact_point_normal_w"][:, 2] * r["contact_point_normal_force"]).sum()
            sign = 1.0 if nz >= 0 else -1.0
        ffr = r["friction_point_frame"].astype(int) if "friction_point_frame" in r else np.zeros(0, int)
        forder = np.argsort(ffr, kind="stable")
        fstarts = np.searchsorted(ffr[forder], np.arange(len(r["time"]) + 1))
        pt_index.append((order, starts, sign, forder, fstarts))

    renderer = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance
    opt = mujoco.MjvOption()
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    def render_at(t):
        lookat = []
        status = []
        data.qpos[:] = model.qpos0
        for i, e in enumerate(ex):
            r, meta, ref, ref_meta, _ = e
            k = min(int(round(t / dt)), T_sim[i] - 1)
            fell = t / dt > T_sim[i] - 1 and meta["terminated_early"]
            fa, ja = adr_s[i]
            data.qpos[fa:fa + 3] = r["root_pos_w"][k]
            data.qpos[fa + 3:fa + 7] = r["root_quat_w"][k]
            data.qpos[ja] = r["joint_pos"][k]
            lookat.append(r["root_pos_w"][k])
            if adr_r is not None:
                fps = ref_meta["fps"]
                rf = min(int(round(t * fps)), ref["joint_pos"].shape[0] - 1)
                fa2, ja2 = adr_r[i]
                data.qpos[fa2:fa2 + 3] = ref["body_pos_w"][rf, 0]
                data.qpos[fa2 + 3:fa2 + 7] = ref["body_quat_w"][rf, 0]
                data.qpos[ja2] = ref["joint_pos"][rf]
            status.append(f"{labels[i]}: " + ("TERMINATED (fell / lost tracking)" if fell else
                                              ("success" if meta["success"] else "running")))
        mujoco.mj_kinematics(model, data)
        cam.lookat[:] = args.lookat if args.lookat is not None else np.mean(lookat, 0) * [1, 1, 0] + [0, 0, 0.7]
        renderer.update_scene(data, cam, opt)
        if args.forces:
            scn = renderer.scene
            for i, e in enumerate(ex):
                r = e[0]
                k = min(int(round(t / dt)), T_sim[i] - 1)
                order, starts, sign, forder, fstarts = pt_index[i]
                # one arrow per (body, surface) pair, anchored at the force-weighted centre of its contact points
                groups = {}
                for j in order[starts[k]:starts[k + 1]]:
                    fn = float(r["contact_point_normal_force"][j])
                    if fn <= 0.0:
                        continue
                    key = (int(r["contact_point_body"][j]), int(r["contact_point_filter"][j]))
                    g = groups.setdefault(key, [np.zeros(3), np.zeros(3), 0.0, np.zeros(3)])
                    g[0] += sign * r["contact_point_normal_w"][j] * fn
                    g[1] += r["contact_point_pos_w"][j] * fn
                    g[2] += fn
                for j in forder[fstarts[k]:fstarts[k + 1]]:
                    key = (int(r["friction_point_body"][j]), int(r["friction_point_filter"][j]))
                    if key in groups:
                        groups[key][3] += r["friction_point_force_w"][j]
                for fn_vec, pw, w, ff in groups.values():
                    if w < 5.0:
                        continue
                    p = pw / w
                    add_arrow(scn, p, p + fn_vec * args.force_scale, (0.9, 0.05, 0.05, 1), width=0.015)
                    add_arrow(scn, p, p + ff * args.force_scale, (0.95, 0.8, 0.05, 1), width=0.011)
        img = Image.fromarray(renderer.render())
        dr = ImageDraw.Draw(img)
        lines = [f"t = {t:5.2f} s"] + status + ["solid = physics rollout, translucent = reference (UMR)"]
        if args.forces:
            lines.append(f"red = normal force, yellow = friction, summed per body/surface ({1 / args.force_scale:.0f} N per m)")
        for li, s in enumerate(lines):
            dr.text((16, 12 + 28 * li), s, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(0, 0, 0))
        return np.asarray(img)

    if args.stills:
        os.makedirs(args.out, exist_ok=True)
        for s in args.stills.split(","):
            t = float(s)
            Image.fromarray(render_at(t)).save(os.path.join(args.out, f"t{t:05.2f}.png"))
        print("[RENDER] stills ->", args.out)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        w = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=7, macro_block_size=8)
        nframes = int(duration * args.fps) + 1
        for f in range(nframes):
            w.append_data(render_at(f / args.fps))
        w.close()
        print(f"[RENDER] {nframes} frames ({duration:.2f}s) -> {args.out}")
    os._exit(0)


if __name__ == "__main__":
    main()
