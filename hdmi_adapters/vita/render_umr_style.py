"""Render physics rollouts in the original UMR preview scene (RGB reconstruction mesh + fitted props +
synchronized source video), using the same cameras/layout as UMR's kinematic preview videos.

Modes:
  physics  : same layout as UMR's kinematic preview, robots driven by the physics rollout(s)
  compare  : kinematic (UMR result) vs physics side by side, plus source video and live contact metrics
  refs     : two HDMI motion dirs (e.g. raw vs contact-repaired reference) side by side, single person;
             pass them with --motion A --motion B (no rollout needed)

Usage:
  MUJOCO_GL=egl python scripts/vita/render_umr_style.py --preset seatswap \
      --rollout EXPORT_P01 --rollout EXPORT_P02 --mode compare --out out.mp4 [--forces]
"""
import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

UMR_OUT = "/vita-vepfs-data/fileset1/usr_data/yunhao.xia/code/UMR/output"
HDMI_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PRESETS = {
    "seatswap": dict(
        scene_xml=f"{UMR_OUT}/baseline_verification_20261028/preview_grounded/dual_scene.xml",
        source=f"{UMR_OUT}/baseline_verification_20261028/source/input.mp4",
        kinematic=[f"{UMR_OUT}/baseline_verification_20261028/smoothing_v1/results/person_01_smoothed.npz",
                   f"{UMR_OUT}/baseline_verification_20261028/smoothing_v1/results/person_02_smoothed.npz"],
        prefixes=["p1_", "p2_"],
        title="Vita02A x2 | seat exchange",
        people="Person 1: cyan    Person 2: orange",
        follow=False,
        main=dict(distance=4.8, elevation=-17, azimuth=90, lookat=[-.22, .9, .78]),
        side=dict(distance=4.8, elevation=-17, azimuth=125, lookat=[-.22, .9, .78]),
    ),
    "stair": dict(
        scene_xml=f"{UMR_OUT}/rgb_145317_vita02a/preview_person_01_smoothed/scene.xml",
        source=f"{UMR_OUT}/rgb_145317_vita02a/source/input.mp4",
        kinematic=[f"{UMR_OUT}/rgb_145317_vita02a/results/person_01_smoothed.npz"],
        prefixes=["r_"],
        title="Vita02A | rgb_145317 stairs + bench",
        people="Brown boxes: fitted tier/bench collision proxies   RGB mesh: visual only",
        follow=True,
        main=dict(distance=4.2, elevation=-15, azimuth=90),
        side=dict(distance=3.6, elevation=-12, azimuth=20),
    ),
}


def put_text(canvas, text, xy, size=.65, color=(230, 235, 240)):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def load_rollout(d):
    r = dict(np.load(os.path.join(d, "rollout.npz")))
    meta = json.load(open(os.path.join(d, "meta.json")))
    sj = meta["scene_json"] if os.path.isabs(meta["scene_json"]) else os.path.join(HDMI_ROOT, meta["scene_json"])
    scene = json.load(open(sj))
    ref = dict(np.load(os.path.join(os.path.dirname(sj), "motion.npz")))
    fr = r["contact_point_frame"].astype(int)
    order = np.argsort(fr, kind="stable")
    starts = np.searchsorted(fr[order], np.arange(len(r["time"]) + 1))
    ffr = r["friction_point_frame"].astype(int)
    forder = np.argsort(ffr, kind="stable")
    fstarts = np.searchsorted(ffr[forder], np.arange(len(r["time"]) + 1))
    return dict(r=r, meta=meta, scene=scene, ref=ref, idx=(order, starts, forder, fstarts))


def add_arrow(scn, p0, p1, rgba, width):
    v = p1 - p0
    L = np.linalg.norm(v)
    if scn.ngeom + 2 > scn.maxgeom or L < 1e-3:
        return
    u = v / L
    head = min(0.08, 0.5 * L)
    for typ, a, b, w in ((mujoco.mjtGeom.mjGEOM_CAPSULE, p0, p1 - u * head, width),
                         (mujoco.mjtGeom.mjGEOM_ARROW, p1 - u * head, p1 + u * head, width * 1.6)):
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, typ, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, typ, w, a, b)
        scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), required=True)
    ap.add_argument("--rollout", action="append", default=[], help="one export dir per person, in scene order")
    ap.add_argument("--motion", action="append", default=[], help="refs mode: two HDMI motion dirs")
    ap.add_argument("--motion_labels", default="raw UMR reference,contact-repaired reference")
    ap.add_argument("--kinematic_ref", action="store_true",
                    help="compare mode: left panel shows the HDMI reference the policy tracked, not the UMR result")
    ap.add_argument("--mode", choices=["physics", "compare", "refs"], default="physics")
    ap.add_argument("--out", required=True)
    ap.add_argument("--forces", action="store_true")
    ap.add_argument("--force_scale", type=float, default=0.0015, help="m per N")
    ap.add_argument("--label", default="", help="extra text, e.g. checkpoint name")
    ap.add_argument("--frames", default=None, help="comma separated source frames: write JPGs instead of a video")
    args = ap.parse_args()
    P = PRESETS[args.preset]
    if args.mode == "refs":
        assert len(args.motion) == 2 and len(P["prefixes"]) == 1
    else:
        assert len(args.rollout) == len(P["prefixes"])

    model = mujoco.MjModel.from_xml_path(P["scene_xml"])
    data = mujoco.MjData(model)
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""
        if body.startswith(tuple(P["prefixes"])) and model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            model.geom_group[g] = 3
    option = mujoco.MjvOption()
    mujoco.mjv_defaultOption(option)
    option.geomgroup[3:] = 0
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = [.45] * 3
    model.vis.headlight.diffuse[:] = [.65] * 3

    kin, scale = [], None
    for path in P["kinematic"]:
        with np.load(path, allow_pickle=True) as a:
            kin.append(a["qpos"].astype(float))
            scale = float(np.ravel(a["smpl_scale"])[0])
            fps = float(np.ravel(a["fps"])[0])
    kin_q = np.concatenate(kin, axis=1)
    assert kin_q.shape[1] == model.nq
    F = kin_q.shape[0]

    def world_offset(e):
        s = e["scene"]
        return np.array([s["xy_offset"][0], s["xy_offset"][1], -s.get("z_lift", 0.0)])

    def person_adr(pre, joint_names):
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j).startswith(pre)][0]
        ja = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + n)]
                       for n in joint_names])
        return model.jnt_qposadr[free], ja

    def motion_qpos(mdir, base_q):
        """UMR-scene qpos per source frame for an HDMI motion dir (env-local -> UMR world, padding removed)."""
        mo = np.load(os.path.join(mdir, "motion.npz"))
        meta = json.load(open(os.path.join(mdir, "meta.json")))
        sc = {"scene": json.load(open(os.path.join(mdir, "scene.json")))}
        fa, ja = person_adr(P["prefixes"][0], meta["joint_names"])
        n0 = sc["scene"]["pad_start_frames"]
        q = np.repeat(base_q[:1], F, 0).copy()
        for f in range(F):
            rf = min(f + n0, mo["joint_pos"].shape[0] - 1)
            q[f, fa:fa + 3] = mo["body_pos_w"][rf, 0] + world_offset(sc)
            q[f, fa + 3:fa + 7] = mo["body_quat_w"][rf, 0]
            q[f, ja] = mo["joint_pos"][rf]
        return q

    if args.mode == "refs":
        ref_q = [motion_qpos(m, kin_q) for m in args.motion]
        ref_labels = args.motion_labels.split(",")
        args.forces = False

    ro = [load_rollout(d) for d in args.rollout]
    adr = [person_adr(pre, e["meta"]["joint_names"]) for pre, e in zip(P["prefixes"], ro)]
    dt = ro[0]["meta"]["physics_dt"] if ro else 1.0 / fps
    if args.kinematic_ref:
        md = os.path.dirname(ro[0]["meta"]["scene_json"] if os.path.isabs(ro[0]["meta"]["scene_json"])
                             else os.path.join(HDMI_ROOT, ro[0]["meta"]["scene_json"]))
        kin_q = motion_qpos(md, kin_q)

    def sim_index(e, f):
        t = (f + e["scene"]["pad_start_frames"]) / fps
        k = int(round(t / dt))
        n = len(e["r"]["time"])
        return min(k, n - 1), k > n - 1

    def physics_qpos(f):
        q = model.qpos0.copy()
        info = []
        for (fa, ja), e in zip(adr, ro):
            k, ended = sim_index(e, f)
            r = e["r"]
            q[fa:fa + 3] = r["root_pos_w"][k] + world_offset(e)
            q[fa + 3:fa + 7] = r["root_quat_w"][k]
            q[ja] = r["joint_pos"][k]
            rf = min(f + e["scene"]["pad_start_frames"], e["ref"]["body_pos_w"].shape[0] - 1)
            root_err = np.linalg.norm(r["root_pos_w"][k] - e["ref"]["body_pos_w"][rf, 0])
            F_t = r["contact_total_force_w"][k][..., 2]
            mg = e["meta"]["total_mass_kg"] * 9.81
            names = e["meta"]["contact_filter_names"]
            ground = F_t[:, names.index("ground")].sum() / mg
            props = sum(F_t[:, i].sum() for i, n in enumerate(names) if n != "ground") / mg
            info.append(dict(k=k, ended=ended and e["meta"]["terminated_early"], root_err=root_err,
                             ground=ground, props=props))
        return q, info

    def draw_forces(scn, info):
        for e, inf in zip(ro, info):
            r, (order, starts, forder, fstarts) = e["r"], e["idx"]
            k, off = inf["k"], world_offset(e)
            groups = {}
            for j in order[starts[k]:starts[k + 1]]:
                fn = float(r["contact_point_normal_force"][j])
                if fn <= 0:
                    continue
                key = (int(r["contact_point_body"][j]), int(r["contact_point_filter"][j]))
                g = groups.setdefault(key, [np.zeros(3), np.zeros(3), 0.0, np.zeros(3)])
                n = r["contact_point_normal_w"][j]
                g[0] += (n if n[2] >= 0 else -n) * fn if abs(n[2]) > 0.5 else n * fn
                g[1] += r["contact_point_pos_w"][j] * fn
                g[2] += fn
            for j in forder[fstarts[k]:fstarts[k + 1]]:
                key = (int(r["friction_point_body"][j]), int(r["friction_point_filter"][j]))
                if key in groups:
                    groups[key][3] += r["friction_point_force_w"][j]
            for fn_vec, pw, w, ff in groups.values():
                if w < 5.0:
                    continue
                p = pw / w + off
                add_arrow(scn, p, p + fn_vec * args.force_scale, (0.9, 0.05, 0.05, 1), 0.015)
                add_arrow(scn, p, p + ff * args.force_scale, (0.95, 0.8, 0.05, 1), 0.011)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    root_xy = kin_q[:, :2]
    kernel = np.ones(31) / 31
    smooth_xy = np.column_stack([np.convolve(np.pad(root_xy[:, i], 15, mode="edge"), kernel, "valid") for i in range(2)])

    def set_cam(view, f):
        c = P[view]
        cam.distance, cam.elevation, cam.azimuth = c["distance"] * scale, c["elevation"], c["azimuth"]
        if P["follow"]:
            cam.lookat[:] = [smooth_xy[f, 0], smooth_xy[f, 1] + .2 * scale, .75 * scale]
        else:
            cam.lookat[:] = np.asarray(c["lookat"]) * scale

    renderer = mujoco.Renderer(model, width=1280, height=900)

    def render(q, view, f, info=None):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        set_cam(view, f)
        renderer.update_scene(data, camera=cam, scene_option=option)
        if info is not None and args.forces:
            draw_forces(renderer.scene, info)
        return renderer.render()

    source = cv2.VideoCapture(P["source"])
    status_col = (120, 230, 140)

    def status_lines(info):
        out = []
        for i, inf in enumerate(info):
            who = f"P{i + 1}" if len(info) > 1 else "Robot"
            st = "TERMINATED (fell / lost tracking)" if inf["ended"] else "tracking"
            out.append(f"{who}: {st} | root err {inf['root_err'] * 100:4.1f} cm | "
                       f"load ground {inf['ground'] * 100:3.0f}% / props {inf['props'] * 100:3.0f}% mg")
        return out

    def frame_physics(f, image, info):
        canvas = np.full((1080, 1920, 3), (22, 28, 33), np.uint8)
        q, _ = physics_qpos(f)
        canvas[70:970, :1280] = render(q, "main", f, info)
        canvas[430:880, 1280:] = cv2.resize(render(q, "side", f, info), (640, 450))
        canvas[70:430, 1280:] = cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2RGB)
        put_text(canvas, P["title"] + " | physics rollout", (28, 44), .9)
        put_text(canvas, f"Frame {f:03d}/{F - 1} | {f / fps:.2f}s | {fps:.0f} FPS", (1320, 44), .65)
        put_text(canvas, P["people"], (28, 1004), .7)
        put_text(canvas, ("Contacts: red = normal force, yellow = friction (per body/surface)  " if args.forces else "")
                 + args.label, (28, 1045), .6)
        put_text(canvas, "Source RGB (synchronized)", (1300, 100), .6)
        put_text(canvas, "Oblique view", (1300, 460), .6)
        put_text(canvas, "PHYSICS ROLLOUT", (1300, 915), .75, status_col)
        for i, s in enumerate(status_lines(info)):
            put_text(canvas, s, (1290, 945 + 26 * i), .42)
        return canvas

    def frame_compare(f, image, info):
        canvas = np.full((1080, 1920, 3), (22, 28, 33), np.uint8)
        q_phys, _ = physics_qpos(f)
        canvas[70:745, :960] = cv2.resize(render(kin_q[f], "main", f), (960, 675))
        canvas[70:745, 960:] = cv2.resize(render(q_phys, "main", f, info), (960, 675))
        canvas[770:1040, :480] = cv2.cvtColor(cv2.resize(image, (480, 270)), cv2.COLOR_BGR2RGB)
        canvas[770:1040, 480:960] = cv2.resize(render(kin_q[f], "side", f), (480, 270))
        canvas[770:1040, 960:1440] = cv2.resize(render(q_phys, "side", f, info), (480, 270))
        put_text(canvas, "KINEMATIC | " + ("HDMI reference (tracked by the policy)" if args.kinematic_ref
                                           else "UMR retarget (smoothed)"), (25, 45), .85, (255, 191, 100))
        put_text(canvas, "PHYSICS | Isaac Lab rollout of HDMI tracking policy", (985, 45), .85, status_col)
        put_text(canvas, f"Source frame {f:03d}/{F - 1} | {f / fps:.2f}s", (1460, 790), .6)
        for i, s in enumerate(status_lines(info)):
            put_text(canvas, s.split(" | load")[0], (1460, 825 + 52 * i), .45)
            put_text(canvas, "      " + s.split(" | ")[-1], (1460, 825 + 52 * i + 24), .45)
        if args.forces:
            put_text(canvas, "red = normal, yellow = friction", (1460, 1000), .45)
        put_text(canvas, args.label, (1460, 1030), .45)
        put_text(canvas, "source", (8, 790), .5)
        put_text(canvas, "kinematic, oblique", (488, 790), .5)
        put_text(canvas, "physics, oblique", (968, 790), .5)
        return canvas

    def frame_refs(f, image, info):
        canvas = np.full((1080, 1920, 3), (22, 28, 33), np.uint8)
        for i, (q, lab) in enumerate(zip(ref_q, ref_labels)):
            canvas[70:745, 960 * i:960 * (i + 1)] = cv2.resize(render(q[f], "main", f), (960, 675))
            canvas[770:1040, 480 * (i + 1):480 * (i + 2)] = cv2.resize(render(q[f], "side", f), (480, 270))
            put_text(canvas, f"KINEMATIC | {lab}", (25 + 960 * i, 45), .85, ((255, 191, 100), (120, 230, 140))[i])
            put_text(canvas, f"{lab}, oblique", (488 + 480 * i, 790), .5)
        canvas[770:1040, :480] = cv2.cvtColor(cv2.resize(image, (480, 270)), cv2.COLOR_BGR2RGB)
        put_text(canvas, "source", (8, 790), .5)
        put_text(canvas, f"Source frame {f:03d}/{F - 1} | {f / fps:.2f}s", (1460, 790), .6)
        put_text(canvas, args.label, (1460, 1030), .45)
        return canvas

    draw = {"physics": frame_physics, "compare": frame_compare, "refs": frame_refs}[args.mode]
    wanted = None if args.frames is None else set(int(x) for x in args.frames.split(","))
    writer = None
    if wanted is None:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        writer = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8, macro_block_size=1,
                                    ffmpeg_log_level="error")
    else:
        os.makedirs(args.out, exist_ok=True)
    try:
        for f in range(F):
            ok, image = source.read()
            assert ok, f"source ended at {f}"
            if wanted is not None and f not in wanted:
                continue
            info = physics_qpos(f)[1] if ro else []
            canvas = draw(f, image, info)
            if writer is not None:
                writer.append_data(canvas)
            else:
                cv2.imwrite(os.path.join(args.out, f"frame_{f:03d}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        source.release()
        if writer is not None:
            writer.close()
    print(f"[RENDER] {args.mode} {F if wanted is None else len(wanted)} frames -> {args.out}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
