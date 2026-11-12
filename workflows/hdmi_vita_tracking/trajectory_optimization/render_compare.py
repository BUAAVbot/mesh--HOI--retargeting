"""Render two continuous, recorded physics traces with identical cameras.

This program never integrates or edits either trajectory.  The supplied
compiled MuJoCo model is used only for forward kinematics and rendering.
Collision proxies remain visible so missing visual meshes cannot hide gaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_trace(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        names_key = "robot_joint_names" if "robot_joint_names" in z else "joint_names"
        names = tuple(str(x) for x in z[names_key])
        fps = float(np.asarray(z["fps"]).reshape(-1)[0])
        if "qpos" in z:
            q = np.asarray(z["qpos"], dtype=float)
        elif "root_state" in z and "joint_pos" in z:
            q = np.concatenate((z["root_state"][:, :7], z["joint_pos"]), axis=1).astype(float)
        else:
            raise ValueError(f"{path}: expected qpos or root_state + joint_pos")
        if "motion_frame" in z and np.any(np.diff(z["motion_frame"]) < 0):
            raise ValueError(f"{path}: motion-frame restart detected; cannot label this continuous")
    if q.ndim != 2 or q.shape[1] != 35 or len(names) != 28:
        raise ValueError(f"{path}: expected T x 35 and 28 joint names")
    if len(set(names)) != 28 or not np.isfinite(q).all() or fps <= 0:
        raise ValueError(f"{path}: invalid names, values or fps")
    if np.max(np.abs(np.linalg.norm(q[:, 3:7], axis=1) - 1)) > 1e-3:
        raise ValueError(f"{path}: root quaternion must be normalized wxyz")
    return q, names, fps


def load_receipt(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    resets = data.get("reset_count", data.get("resets_during_rollout"))
    writes = data.get("state_writes_after_initialization")
    if resets != 0 or writes != 0:
        raise ValueError(f"{path}: receipt must explicitly record reset_count=0 and state_writes_after_initialization=0")
    if data.get("external_root_force") is not False:
        raise ValueError(f"{path}: receipt must explicitly record external_root_force=false")
    return data


def first_failure(report):
    value = report.get("first_failure_frame")
    if value is not None:
        return int(value)
    failures = report.get("failures", [])
    if failures:
        return min(int(f.get("rollout_frame", f.get("frame", 0))) for f in failures)
    return None


def passed(report):
    return bool(report.get("accepted", report.get("passed", report.get("passed_nominal_gate", False))))


def model_joint_map(model, names):
    import mujoco as mj

    actual = {}
    free = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE:
            free.append(int(model.jnt_qposadr[j]))
        if model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE:
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
            actual[name] = int(model.jnt_qposadr[j])
    if len(free) != 1:
        raise ValueError("comparison supports one dynamic floating-root robot")
    addresses = []
    for name in names:
        candidates = [address for key, address in actual.items()
                      if key == name or key.endswith("/" + name)]
        if len(candidates) != 1:
            raise ValueError(f"joint mapping ambiguous/missing: {name}")
        addresses.append(candidates[0])
    return free[0], addresses


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "before", "after", "before-report", "after-report", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--reference", type=Path, help="Optional original target for fixed camera framing only")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=400)
    p.add_argument("--fps", type=float, default=25.)
    p.add_argument("--sample-only", type=int, default=0)
    p.add_argument("--before-label", default="BEFORE / fixed policy")
    p.add_argument("--after-label", default="AFTER / fixed policy + optimized residual")
    p.add_argument("--single-before", action="store_true", help="Render only before trace, in two fixed views")
    p.add_argument("--single-after", action="store_true", help="Render only after trace, in two fixed views")
    p.add_argument("--after-gate", type=Path, help="Optional independent acceptance report bound to after trace SHA256")
    p.add_argument("--scene-note", help="Explicit scene description printed on every output frame")
    p.add_argument("--contact-calibration-report", type=Path,
                   help="Explicit provenance for comparing original and contact-calibrated commands/initial states")
    p.add_argument("--calibrated-reference", type=Path,
                   help="Calibrated command NPZ, required with contact-calibration-report")
    args = p.parse_args()
    if args.single_before and args.single_after:
        p.error("Choose at most one single-trace view")
    single = args.single_before or args.single_after

    import imageio.v2 as imageio
    import mujoco as mj
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    before, before_names, before_fps = load_trace(args.before)
    after, after_names, after_fps = load_trace(args.after)
    if len(before) != len(after) or abs(before_fps - after_fps) > 1e-7:
        raise ValueError("A/B must have the same physical clock and sample count; retiming comparisons require an explicit source-time mapping")
    reports = [load_receipt(args.before_report), load_receipt(args.after_report)]
    calibration = None
    if args.contact_calibration_report:
        if not args.reference or not args.calibrated_reference:
            raise ValueError("contact-calibrated comparison requires original --reference and --calibrated-reference")
        calibration = json.loads(args.contact_calibration_report.read_text(encoding="utf-8"))
        original_sha = sha(args.reference)
        if (calibration.get("original_source_reference_sha256") != original_sha
                or reports[0].get("source_reference_sha256") != original_sha
                or reports[1].get("source_reference_sha256") != sha(args.calibrated_reference)):
            raise ValueError("contact calibration source/command hashes do not bind both traces")
        if not calibration.get("initial_state_changed") or not calibration.get("source_clock_unchanged"):
            raise ValueError("contact-calibrated comparison must explicitly disclose changed initial state and unchanged source clock")
        if calibration.get("scene_sha256") != reports[0].get("scene_sha256"):
            raise ValueError("contact projection used a different scene")
        actual_delta_z = float(after[0, 2] - before[0, 2])
        expected_delta_z = float(calibration["initial_rootZ_correction_m"])
        if abs(actual_delta_z - expected_delta_z) > 1e-3:
            raise ValueError("physical initial-root correction differs from contact calibration report")
    for field in ("scene_sha256", "checkpoint_sha256", "model_sha256"):
        if not reports[0].get(field) or reports[0].get(field) != reports[1].get(field):
            raise ValueError(f"A/B source or controller contract differs: {field}")
    if calibration is None and reports[0].get("source_reference_sha256") != reports[1].get("source_reference_sha256"):
        raise ValueError("command references differ without explicit contact calibration provenance")
    if reports[0]["model_sha256"] != sha(args.model):
        raise ValueError("render model does not match the recorded physics model hash")
    independent_after = None
    if args.after_gate:
        independent_after = json.loads(args.after_gate.read_text(encoding="utf-8"))
        if independent_after.get("record_sha256") != sha(args.after):
            raise ValueError("independent gate is not bound to this after physical record")
        for field in ("source_reference_sha256", "scene_sha256", "checkpoint_sha256", "model_sha256"):
            if independent_after.get(field) != reports[1].get(field):
                raise ValueError(f"independent gate contract differs from physical report: {field}")
    display_passed = [passed(reports[0]), bool(independent_after.get("accepted", False)) if independent_after else False]
    if args.model.suffix == ".mjb":
        model = mj.MjModel.from_binary_path(str(args.model))
    else:
        model = mj.MjModel.from_xml_path(str(args.model))
    maps = [model_joint_map(model, before_names), model_joint_map(model, after_names)]
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    model.vis.headlight.ambient[:] = .7
    model.vis.headlight.diffuse[:] = .8

    intended_roots = after[:, :3]
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as ref:
            if "body_pos_w" in ref:
                intended_roots = ref["body_pos_w"][:, 0, :]
            elif "root_state" in ref:
                intended_roots = ref["root_state"][:, :3]
    framing_roots = np.concatenate((intended_roots, before[:, :3], after[:, :3]), axis=0)
    lower, upper = framing_roots.min(axis=0), framing_roots.max(axis=0)
    center = (lower + upper) / 2
    center[2] = max(.35, center[2] - .2)
    spread = np.max(np.linalg.norm(framing_roots[:, :2] - center[:2], axis=1))
    distance = float(max(3., 2.0 * spread + 2.5, 2.0 * (upper[2] - lower[2]) + 2.))
    cameras = []
    for azimuth, elevation in ((135., -18.), (90., -10.)):
        camera = mj.MjvCamera()
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = center
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        cameras.append(camera)
    option = mj.MjvOption()
    option.geomgroup[:] = 1
    option.sitegroup[:] = 0
    # Transparent proxies would hide floor/support gaps. Use opaque geometry.
    model.geom_rgba[:, 3] = np.maximum(model.geom_rgba[:, 3], .8)
    hfields = model.geom_type == mj.mjtGeom.mjGEOM_HFIELD
    model.geom_rgba[hfields] = [.30, .43, .50, 1.0]
    model.geom_matid[hfields] = -1
    frames = int(np.ceil(len(before) * args.fps / before_fps))
    if args.sample_only:
        frames = min(frames, args.sample_only)
    data = mj.MjData(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = 40
    footer = 48
    first = [first_failure(r) for r in reports]
    labels = (args.before_label, args.after_label)
    try:
        title_font = ImageFont.truetype("DejaVuSans.ttf", 15)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        title_font = small_font = ImageFont.load_default()
    snapshots = {0, frames - 1}
    snapshots.update(min(frames - 1, int(np.ceil(f / before_fps * args.fps)))
                     for f in first if f is not None)
    with mj.Renderer(model, height=args.height, width=args.width) as renderer, \
            imageio.get_writer(str(args.output), fps=args.fps, codec="libx264", quality=8) as writer:
        for index in range(frames):
            t = index / args.fps
            source_index = min(round(t * before_fps), len(before) - 1)
            canvas = Image.new("RGB", (args.width * (1 if single else 2), (args.height + header) * 2 + footer), (238, 241, 245))
            draw = ImageDraw.Draw(canvas)
            for row, camera in enumerate(cameras):
                panels = ((0, before),) if args.single_before else ((1, after),) if args.single_after else ((0, before), (1, after))
                for panel_col, (col, trace) in enumerate(panels):
                    q = trace[source_index]
                    free, addresses = maps[col]
                    data.qpos[free:free + 7] = q[:7]
                    data.qpos[addresses] = q[7:]
                    mj.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = 0
                    renderer.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION] = 0
                    x = panel_col * args.width
                    y = row * (args.height + header)
                    canvas.paste(Image.fromarray(renderer.render().copy()), (x, y + header))
                    has_failed = first[col] is not None and source_index >= first[col]
                    gate_rejected = col == 1 and (independent_after is not None or calibration is not None) and not display_passed[col]
                    color = (170, 32, 38) if has_failed or gate_rejected else (28, 57, 78)
                    draw.rectangle((x, y, x + args.width, y + header), fill=color)
                    rejection = "NOT ACCEPTED / physical gates failed" if independent_after is not None else "NOT ACCEPTED / physical probe failed" if first[col] is not None and first[col] < len(trace) else "NOT ACCEPTED / certificate pending"
                    status = rejection if gate_rejected else "FAILED GATE (physics continues)" if has_failed else "ACCEPTED / physics gates" if display_passed[col] else "CANDIDATE / not accepted"
                    draw.text((x + 8, y + 3), labels[col] + f"  | view {row + 1}", fill="white", font=title_font)
                    draw.text((x + 8, y + 22), f"t={t:.2f}s  frame={source_index}  {status}", fill="white", font=small_font)
            y = (args.height + header) * 2 + 6
            footer1 = "Continuous recorded physics / no resets or root assistance." if single else "Recorded continuous physics; no reset or floating-root assistance. Same camera, scale and physical clock."
            footer2 = "Collision/support proxies; not a visual-mesh acceptance test." if single else "Collision/support proxies shown; incomplete reconstructed visual mesh is not used to conceal contact gaps."
            if args.scene_note:
                footer2 = args.scene_note
            if calibration:
                shift_mm = calibration["initial_rootZ_correction_m"] * 1000
                footer2 = f"Initial rootZ {shift_mm:+.0f}mm + joint contact calibrated; command/initial state changed."
                if args.single_before:
                    footer2 = "Raw reference / original initial state unchanged; contact-calibrated after is separate."
                elif not single:
                    footer2 = "Mesh-calibrated scene. " + footer2 + " Same physical model, scene, actor and clock."
            draw.text((8, y), footer1, fill=(20, 30, 45), font=small_font)
            draw.text((8, y + 17), footer2, fill=(20, 30, 45), font=small_font)
            if index in snapshots:
                canvas.save(args.output.parent / f"{args.output.stem}_frame{source_index:04d}.jpg", quality=92)
            writer.append_data(np.asarray(canvas))
    meta = {"kind": "single_continuous_physics_trace" if single else "continuous_physics_before_after_comparison", "integrates_physics": False,
            "model": str(args.model), "model_sha256": sha(args.model),
            "before": str(args.before), "after": str(args.after),
            "before_sha256": sha(args.before), "after_sha256": sha(args.after),
            "before_report": str(args.before_report), "after_report": str(args.after_report),
            "physical_fps": before_fps, "output_fps": args.fps, "output_frames": frames,
            "physical_samples": len(before), "camera_center": center.tolist(), "camera_distance": distance,
            "first_failure_frames": first, "passed": display_passed,
            "independent_after_gate": str(args.after_gate) if args.after_gate else None,
            "scene_note": args.scene_note,
            "contact_calibration_report": str(args.contact_calibration_report) if calibration else None,
            "contact_calibration_report_sha256": sha(args.contact_calibration_report) if calibration else None,
            "original_source_reference_sha256": calibration.get("original_source_reference_sha256") if calibration else reports[0].get("source_reference_sha256"),
            "before_command_reference_sha256": reports[0].get("source_reference_sha256"),
            "after_command_reference_sha256": reports[1].get("source_reference_sha256"),
            "initial_state_changed": calibration.get("initial_state_changed") if calibration else False,
            "note": "Rendering recorded states only; visual does not establish independent dynamics acceptance."}
    args.output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
