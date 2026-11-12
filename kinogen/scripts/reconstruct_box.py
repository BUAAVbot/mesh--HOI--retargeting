#!/usr/bin/env python3
"""CPU, self-implemented known-cuboid two-view silhouette reconstruction.

Uses fixed metric box dimensions and supplied camera projections, NOT relative
monocular depth as metres. The generated video may violate the camera/rigidity
assumptions; failed frames and conditional calibration are explicitly recorded.
No GPU, training, model downloads, or official KinoGen integration is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def file_record(path):
    path = Path(path)
    return {"path": str(path.resolve().relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}


def box_corners(size):
    import numpy as np
    return np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)], dtype=float) * size


def project_pose(parameters, corners, projections):
    import numpy as np
    from scipy.spatial.transform import Rotation
    points = corners @ Rotation.from_rotvec(parameters[3:]).as_matrix().T + parameters[:3]
    homogeneous = np.column_stack((points, np.ones(len(points))))
    xyw = np.einsum("vij,nj->vni", projections, homogeneous)
    return xyw[..., :2] / np.maximum(xyw[..., 2:], 1e-8), xyw[..., 2], points


def resample_polygon(polygon, number=96):
    import numpy as np
    segments = np.roll(polygon, -1, axis=0) - polygon
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.r_[0, lengths.cumsum()]
    if cumulative[-1] < 1e-6:
        return np.repeat(polygon[:1], number, axis=0)
    distances = np.arange(number) / number * cumulative[-1]
    indices = np.clip(np.searchsorted(cumulative, distances, side="right") - 1, 0, len(polygon) - 1)
    fractions = (distances - cumulative[indices]) / np.maximum(lengths[indices], 1e-8)
    return polygon[indices] + fractions[:, None] * segments[indices]


def point_polygon_distances(points, polygon):
    import numpy as np
    a = polygon
    ab = np.roll(polygon, -1, axis=0) - a
    ap = points[:, None, :] - a[None]
    fraction = np.clip((ap * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-10), 0, 1)
    distances = np.linalg.norm(ap - fraction[..., None] * ab[None], axis=-1)
    return distances.min(axis=1)


def sample_map(image, xy, outside=100):
    from scipy.ndimage import map_coordinates
    return map_coordinates(image, [xy[:, 1], xy[:, 0]], order=1, mode="constant", cval=outside)


def observation(mask, valid, robot, directions, min_area=600):
    import cv2
    import numpy as np
    raw = np.asarray(mask, bool)
    valid = np.asarray(valid, bool)
    mask = raw & valid
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area or len(contour) < 12:
        return None
    component = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(component, [contour], -1, 1, cv2.FILLED)
    mask = (component > 0) & valid
    occluder = np.asarray(robot, bool) & ~raw
    occluder = cv2.dilate(occluder.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    invalid_border = cv2.dilate((~valid).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    excluded = occluder | invalid_border
    hull = cv2.convexHull(contour.astype(np.float32)).reshape(-1, 2).astype(float)
    hull_points = resample_polygon(hull)
    contour_points = contour[:, 0].astype(float)
    contour_points = contour_points[np.linspace(0, len(contour_points) - 1, 96).astype(int)]
    edge = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(edge, [contour], -1, 1, 1)
    distance = cv2.distanceTransform(1 - edge, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    supports = hull @ directions.T
    indices = supports.argmax(axis=0)
    support_points = hull[indices]
    support_weights = 1 - .65 * sample_map(excluded.astype(float), support_points, outside=1)
    contour_weights = 1 - .8 * sample_map(excluded.astype(float), contour_points, outside=1)
    return {"mask": mask, "valid": valid, "occluder": occluder, "excluded": excluded,
            "hull": hull, "contour": contour_points, "distance": distance,
            "supports": supports.max(axis=0), "support_weights": support_weights,
            "contour_weights": contour_weights, "hull_points": hull_points,
            "area": int(mask.sum()), "raw_area": int(raw.sum()),
            "invalid_removed_fraction": float((raw & ~valid).sum() / max(raw.sum(), 1))}


def data_residual(parameters, corners, projections, observations, directions):
    import cv2
    import numpy as np
    xy, depth, _ = project_pose(parameters, corners, projections)
    result = []
    for view, obs in enumerate(observations):
        if obs is None:
            continue
        if np.any(depth[view] <= .05):
            result.extend(np.full(len(directions) + 192, 1000.))
            continue
        hull = cv2.convexHull(xy[view].astype(np.float32)).reshape(-1, 2).astype(float)
        support = (xy[view] @ directions.T).max(axis=0)
        support_residual = (support - obs["supports"]) * obs["support_weights"]
        observed_distances = point_polygon_distances(obs["contour"], hull) * obs["contour_weights"]
        predicted_points = resample_polygon(hull)
        predicted_distances = sample_map(obs["distance"], predicted_points)
        prediction_weights = 1 - .8 * sample_map(obs["excluded"].astype(float), predicted_points, outside=1)
        result.extend(support_residual / 3.)
        result.extend(observed_distances * .35 / 3.)
        result.extend(predicted_distances * prediction_weights * .35 / 3.)
    return np.asarray(result, dtype=float)


def solve_pose(corners, projections, observations, directions, previous, velocity, initial, frame, max_nfev=90):
    import numpy as np
    from scipy.optimize import least_squares
    lower = np.r_[initial[:3] - [.85, .85, .5], [-.9, -.9, -1.5]]
    upper = np.r_[initial[:3] + [.85, .85, .9], [.9, .9, 1.5]]
    def residual(parameters, regularize=True):
        geometry = data_residual(parameters, corners, projections, observations, directions)
        if not regularize:
            return geometry
        temporal = (parameters - previous) / np.array([.08, .08, .08, .25, .25, .25])
        # Weak temporal regularization, with no target lift or contact height.
        return np.r_[geometry, temporal * .35]
    starts = [previous.copy(), previous + velocity]
    if frame == 0 or frame % 12 == 0:
        for yaw in (-.35, .35):
            candidate = previous.copy()
            candidate[5] += yaw
            starts.append(candidate)
    candidates = []
    for start in starts:
        start = np.minimum(np.maximum(start, lower + 1e-7), upper - 1e-7)
        result = least_squares(residual, start, bounds=(lower, upper), loss="soft_l1", f_scale=2.,
                               max_nfev=max_nfev, diff_step=1e-3,
                               x_scale=np.array([.1, .1, .1, .3, .3, .3]), ftol=2e-5, xtol=2e-5, gtol=1e-5)
        candidates.append(result)
    result = min(candidates, key=lambda item: item.cost)
    plausible = [item.x for item in candidates if item.cost <= result.cost * 1.1 + .5]
    spread = np.ptp(np.asarray(plausible), axis=0) if len(plausible) > 1 else np.zeros(6)
    # Conditional local uncertainty from image residuals only, excluding prior.
    step = np.array([1e-4] * 3 + [1e-3] * 3)
    jacobian = np.column_stack([(residual(result.x + np.eye(6)[j] * step[j], False) - residual(result.x - np.eye(6)[j] * step[j], False)) / (2 * step[j]) for j in range(6)])
    data = residual(result.x, False)
    variance = max(float(np.mean(data * data)), .25)
    covariance = np.linalg.pinv(jacobian.T @ jacobian, rcond=1e-10) * variance
    normalized = jacobian * np.array([.1, .1, .1, .3, .3, .3])[None]
    singular = np.linalg.svd(normalized, compute_uv=False)
    condition = float(singular[0] / max(singular[-1], 1e-12))
    return result.x, {"cost": float(result.cost), "nfev": int(result.nfev), "converged": bool(result.success),
                     "normalized_jacobian_condition": condition, "covariance_proxy": covariance,
                     "multistart_parameter_span": spread, "data_residual_rms_normalized": float(np.sqrt(np.mean(data ** 2)))}


def view_metrics(parameters, corners, projections, observations):
    import cv2
    import numpy as np
    xy, _, _ = project_pose(parameters, corners, projections)
    rows = []
    for view, obs in enumerate(observations):
        if obs is None:
            rows.append(None)
            continue
        polygon = cv2.convexHull(xy[view].astype(np.float32)).reshape(-1, 2).astype(float)
        predicted = np.zeros(obs["mask"].shape, np.uint8)
        cv2.fillConvexPoly(predicted, np.round(polygon).astype(np.int32), 1)
        observable = obs["valid"] & ~obs["occluder"]
        predicted = predicted.astype(bool) & observable
        target = obs["mask"] & observable
        union = (predicted | target).sum()
        points = resample_polygon(polygon)
        include = sample_map(obs["excluded"].astype(float), points, outside=1) < .5
        forward = sample_map(obs["distance"], points)[include]
        backward = point_polygon_distances(obs["contour"], polygon)[obs["contour_weights"] > .5]
        distances = np.r_[forward, backward]
        rows.append({"observable_iou": float((predicted & target).sum() / max(union, 1)),
                     "chamfer_mean_px": float(distances.mean()) if len(distances) else None,
                     "chamfer_median_px": float(np.median(distances)) if len(distances) else None,
                     "chamfer_p95_px": float(np.quantile(distances, .95)) if len(distances) else None,
                     "mask_area_px": obs["area"], "invalid_removed_fraction": obs["invalid_removed_fraction"],
                     "observable_boundary_samples": int(len(distances))})
    return rows


def independent_view_centres(parameters, corners, projections, observations, directions):
    import numpy as np
    from scipy.optimize import least_squares
    centres = []
    for view, obs in enumerate(observations):
        if obs is None:
            centres.append(np.full(3, np.nan))
            continue
        def residual(center):
            value = parameters.copy()
            value[:3] = center
            return data_residual(value, corners, projections[view:view + 1], [obs], directions)
        result = least_squares(residual, parameters[:3], bounds=(parameters[:3] - .4, parameters[:3] + .4),
                               max_nfev=50, diff_step=1e-3, loss="soft_l1", f_scale=2.)
        centres.append(result.x)
    centres = np.asarray(centres)
    disagreement = float(np.linalg.norm(centres[0] - centres[1])) if np.isfinite(centres).all() else None
    return centres, disagreement


def triangulate_visible_corners(parameters, corners, projections, observations):
    """Diagnostic model-guided silhouette-corner matches, not semantic GT."""
    import cv2
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    pixels, _, world = project_pose(parameters, corners, projections)
    matches = []
    for view, obs in enumerate(observations):
        assigned = {}
        if obs is not None:
            predicted_indices = cv2.convexHull(pixels[view].astype(np.float32), returnPoints=False).reshape(-1)
            observed = cv2.approxPolyDP(obs["hull"].astype(np.float32), 1.5, True).reshape(-1, 2)
            cost = np.linalg.norm(pixels[view, predicted_indices, None] - observed[None], axis=-1)
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                point = observed[column]
                if cost[row, column] < 8 and sample_map(obs["excluded"].astype(float), point[None], outside=1)[0] < .5:
                    assigned[int(predicted_indices[row])] = point.astype(float)
        matches.append(assigned)
    common = sorted(set(matches[0]) & set(matches[1]))
    triangulated = np.full((8, 3), np.nan)
    accepted, residuals, reprojections = [], [], []
    if common:
        a = np.asarray([matches[0][i] for i in common]).T
        b = np.asarray([matches[1][i] for i in common]).T
        homogeneous = cv2.triangulatePoints(projections[0], projections[1], a, b)
        points = (homogeneous[:3] / homogeneous[3:]).T
        for corner, point, observed_a, observed_b in zip(common, points, a.T, b.T):
            projected = projections @ np.r_[point, 1]
            uv = projected[:, :2] / projected[:, 2:]
            reprojection = float(np.linalg.norm(uv - [observed_a, observed_b], axis=1).mean())
            if np.isfinite(point).all() and np.all(projected[:, 2] > 0) and reprojection <= 4:
                triangulated[corner] = point
                accepted.append(corner)
                residuals.append(float(np.linalg.norm(point - world[corner])))
                reprojections.append(reprojection)
    diagnostic = {"method": "model-guided visible silhouette-corner assignment; not independent semantic correspondence",
                  "accepted_corner_indices": accepted, "accepted_corner_count": len(accepted),
                  "triangulated_to_fitted_cuboid_corner_rms_m": float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None,
                  "mean_corner_reprojection_px": float(np.mean(reprojections)) if reprojections else None}
    return triangulated, diagnostic


def read_geometry(path):
    import numpy as np
    data = np.load(path, allow_pickle=False)
    projection_key = next((key for key in ("P_video", "projection_video", "camera_P_video", "P") if key in data), None)
    if projection_key is None:
        raise ValueError(f"No video camera projection in {list(data.keys())}")
    matrices = np.asarray(data[projection_key], dtype=float)
    if matrices.shape != (2, 3, 4):
        raise ValueError(f"Expected fixed two-view projections [2,3,4], got {matrices.shape}")
    return matrices, {"projection_key": projection_key, "available_fields": list(data.keys()),
                      "calibration_valid_flag": bool(np.asarray(data["calibration_valid"]).all()) if "calibration_valid" in data else None,
                      "conditional_on_supplied_cameras": True,
                      "note": "Video image homography/camera calibration is provided by a separate check; this fit cannot certify that AI generation preserved a physical camera."}


def self_test():
    import cv2
    import numpy as np
    from scipy.spatial.transform import Rotation
    cv2.setNumThreads(4)
    size = np.array([.28, .34, .24])
    corners = box_corners(size)
    intrinsics = np.array([[700, 0, 320], [0, 700, 360], [0, 0, 1.]])
    projections = []
    for eye in (np.array([2.35, -1.9, 1.52]), np.array([2.35, 1.9, 1.52])):
        z = np.array([.3, 0, .8]) - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.]); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rotation = np.array([x, y, z])
        projections.append(intrinsics @ np.column_stack((rotation, -rotation @ eye)))
    projections = np.asarray(projections)
    initial = np.array([.41, 0, .9, 0, 0, 0.])
    truth = initial + np.array([.04, -.025, .173, .075, -.04, .11])
    pixels, _, _ = project_pose(truth, corners, projections)
    angles = np.arange(64) * 2 * np.pi / 64
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    observations = []
    for points in pixels:
        mask = np.zeros((720, 640), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(np.round(points).astype(np.int32)), 1)
        observations.append(observation(mask, np.ones_like(mask, bool), np.zeros_like(mask, bool), directions))
    estimate, info = solve_pose(corners, projections, observations, directions, initial, np.zeros(6), initial, 0, max_nfev=180)
    position_error = float(np.linalg.norm(estimate[:3] - truth[:3]))
    angle_error = float((Rotation.from_rotvec(estimate[3:]).inv() * Rotation.from_rotvec(truth[3:])).magnitude())
    result = {"mode": "synthetic_numerical_self_test_only", "truth": truth.tolist(), "estimate": estimate.tolist(),
              "position_error_m": position_error, "rotation_error_rad": angle_error,
              "status": "passed" if position_error < .015 and angle_error < .15 else "failed"}
    occluded = []
    for view, obs in enumerate(observations):
        mask = obs["mask"].copy()
        valid = np.ones_like(mask)
        robot = np.zeros_like(mask)
        yy, xx = np.where(mask)
        x0, x1, y0, y1 = int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())
        # Known artificial hand occluder plus an invalid watermark corner.
        robot[y0 + (y1-y0)//3:y0 + 2*(y1-y0)//3, x0 - 4:x0 + 20] = True
        valid[y0 - 3:y0 + 16, x1 - 16:x1 + 4] = False
        mask &= ~robot & valid
        occluded.append(observation(mask, valid, robot, directions))
    occluded_estimate, _ = solve_pose(corners, projections, occluded, directions, initial, np.zeros(6), initial, 0, max_nfev=180)
    occluded_position_error = float(np.linalg.norm(occluded_estimate[:3] - truth[:3]))
    occluded_angle_error = float((Rotation.from_rotvec(occluded_estimate[3:]).inv() * Rotation.from_rotvec(truth[3:])).magnitude())
    result["known_occluder_and_invalid_corner_test"] = {"position_error_m": occluded_position_error,
        "rotation_error_rad": occluded_angle_error,
        "status": "passed" if occluded_position_error < .02 and occluded_angle_error < .2 else "failed"}
    if result["known_occluder_and_invalid_corner_test"]["status"] != "passed":
        result["status"] = "failed"
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Synthetic cuboid optimization did not recover its own known geometry")


def make_preview(video, destination, parameters, valid_frames, metrics, corners, projections, timestamps):
    import av
    import cv2
    import numpy as np
    destination.parent.mkdir(parents=True, exist_ok=True)
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if (a ^ b) in (1, 2, 4)]
    minimum = float(np.nanmin(parameters[:, 2])) - .03
    maximum = float(np.nanmax(parameters[:, 2])) + .03
    with av.open(str(video)) as source, av.open(str(destination), "w") as target:
        rate = source.streams.video[0].average_rate
        stream = target.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 1280, 840, "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        count = 0
        for frame, decoded in enumerate(source.decode(video=0)):
            if frame >= len(parameters):
                break
            image = decoded.to_ndarray(format="bgr24")
            if image.shape[:2] != (720, 1280):
                raise RuntimeError("Preview expects the original 1280x720 two-view video")
            pixels, _, _ = project_pose(parameters[frame], corners, projections)
            colour = (50, 230, 70) if valid_frames[frame] else (30, 150, 255)
            for view in range(2):
                points = np.round(pixels[view]).astype(int) + [view * 640, 0]
                for a, b in edges:
                    cv2.line(image, tuple(points[a]), tuple(points[b]), colour, 2, cv2.LINE_AA)
                row = metrics[frame][view]
                text = f"view {view+1}: IoU {row['observable_iou']:.2f}, edge {row['chamfer_mean_px']:.1f}px" if row and row["chamfer_mean_px"] is not None else f"view {view+1}: no valid box observation"
                cv2.putText(image, text, (view * 640 + 10, 700), cv2.FONT_HERSHEY_SIMPLEX, .45, colour, 1, cv2.LINE_AA)
            canvas = np.zeros((840, 1280, 3), np.uint8)
            canvas[:720] = image
            cv2.putText(canvas, "Self-implemented known-box baseline | supplied-camera metric scale is CONDITIONAL", (12, 744), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"frame {frame+1}/{len(parameters)} t={timestamps[frame]:.2f}s | metric checks {'passed' if valid_frames[frame] else 'INVALID / candidate'} | centre z={parameters[frame,2]:.3f}m", (12, 770), cv2.FONT_HERSHEY_SIMPLEX, .5, colour, 1, cv2.LINE_AA)
            pts = np.column_stack((np.linspace(15, 1260, len(parameters)), 826 - (parameters[:, 2] - minimum) / max(maximum - minimum, .01) * 38)).astype(np.int32)
            cv2.polylines(canvas, [pts], False, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.polylines(canvas, [pts[:frame + 1]], False, colour, 2, cv2.LINE_AA)
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="bgr24")):
                target.mux(packet)
            count += 1
        for packet in stream.encode():
            target.mux(packet)
    with av.open(str(destination)) as check:
        decoded = sum(1 for _ in check.decode(video=0))
    if decoded != len(parameters) or count != len(parameters):
        raise RuntimeError("Reconstruction preview frame count mismatch")
    return decoded


def run(args, report):
    import cv2
    import numpy as np
    from scipy.spatial.transform import Rotation
    cv2.setNumThreads(4)
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    projections, calibration = read_geometry(args.geometry)
    report["camera_assumption"] = calibration
    segmentation = np.load(args.segmentation, allow_pickle=False)
    if "source_video_sha256" in segmentation and str(np.asarray(segmentation["source_video_sha256"]).item()) != digest(args.video):
        raise RuntimeError("Segmentation source-video SHA256 does not match this video")
    masks = segmentation["box"]
    if masks.ndim != 4 or masks.shape[1:] != (2, 720, 640):
        raise RuntimeError(f"Expected box masks [T,2,720,640], got {masks.shape}")
    n = len(masks)
    timestamps = np.asarray(segmentation["timestamps"], dtype=float) if "timestamps" in segmentation else np.arange(n) / 24.
    if timestamps.shape != (n,) or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise RuntimeError("Segmentation timestamps must correspond to all increasing video frames")
    robots = segmentation["robot"] if "robot" in segmentation else np.zeros_like(masks)
    observation_valid = segmentation["observation_valid"] if "observation_valid" in segmentation else np.ones_like(masks)
    if args.invalid:
        extra = np.load(args.invalid, allow_pickle=False)
        invalid = extra["invalid"] if hasattr(extra, "files") else extra
        observation_valid = observation_valid & ~invalid
    size = np.asarray(scene["box"]["size_xyz_m"], float)
    initial = np.r_[scene["box"]["center_xyz_m"], Rotation.from_quat(scene["box"]["quaternion_xyzw"]).as_rotvec()]
    corners = box_corners(size)
    angles = np.arange(64) * 2 * np.pi / 64
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    parameters, covariances, spreads, agreements, centres, frame_valid, measured, image_valid_flags = [], [], [], [], [], [], [], []
    triangulated_corners, corner_diagnostics = [], []
    metrics, frame_reports = [], []
    previous, velocity = initial.copy(), np.zeros(6)
    for frame in range(n):
        observations = [observation(masks[frame, view], observation_valid[frame, view], robots[frame, view], directions) for view in range(2)]
        for view in range(2):
            if "box_valid" in segmentation and not bool(segmentation["box_valid"][frame, view]):
                observations[view] = None
        measurement = any(obs is not None for obs in observations)
        if measurement:
            pose, optimization = solve_pose(corners, projections, observations, directions, previous, velocity, initial, frame, args.max_nfev)
            rows = view_metrics(pose, corners, projections, observations)
            view_centres, disagreement = independent_view_centres(pose, corners, projections, observations, directions)
            covariance = optimization.pop("covariance_proxy")
            span = optimization.pop("multistart_parameter_span")
            reasons = []
            if any(row is None for row in rows):
                reasons.append("missing_two_view_box_observation")
            for view, row in enumerate(rows):
                if row is None:
                    continue
                if row["observable_iou"] < .55:
                    reasons.append(f"view_{view}_low_visible_silhouette_iou")
                if row["chamfer_mean_px"] is None or row["chamfer_mean_px"] > 6 or row["chamfer_p95_px"] > 16:
                    reasons.append(f"view_{view}_large_silhouette_reprojection_error")
                if row["invalid_removed_fraction"] > .4:
                    reasons.append(f"view_{view}_watermark_obscures_box")
            if disagreement is None or disagreement > .1:
                reasons.append("independent_view_metric_centres_disagree")
            if optimization["normalized_jacobian_condition"] > 1e6:
                reasons.append("poor_pose_observability")
            if np.sqrt(np.maximum(np.diag(covariance)[:3], 0)).max() > .05:
                reasons.append("large_local_translation_uncertainty")
            if np.max(span[:3]) > .04:
                reasons.append("multistart_translation_ambiguity")
        else:
            pose = previous.copy()
            rows, disagreement, view_centres = [None, None], None, np.full((2, 3), np.nan)
            covariance, span = np.full((6, 6), np.nan), np.full(6, np.nan)
            optimization = {"status": "no_observations_previous_pose_for_display_only"}
            reasons = ["no_box_measurements"]
        valid = not reasons
        image_reasons = [reason for reason in reasons if reason == "no_box_measurements" or reason == "missing_two_view_box_observation" or "silhouette" in reason or "watermark" in reason]
        image_valid = not image_reasons
        parameters.append(pose)
        covariances.append(covariance)
        spreads.append(span)
        agreements.append(np.nan if disagreement is None else disagreement)
        centres.append(view_centres)
        frame_valid.append(valid)
        image_valid_flags.append(image_valid)
        measured.append(measurement)
        metrics.append(rows)
        stereo_corners, corner_diagnostic = triangulate_visible_corners(pose, corners, projections, observations)
        triangulated_corners.append(stereo_corners)
        corner_diagnostics.append(corner_diagnostic)
        frame_reports.append({"frame": frame, "time_seconds": float(timestamps[frame]), "measurement_backed": measurement,
            "image_fit_valid": image_valid, "strict_geometry_valid": valid, "invalid_reasons": reasons, "image_fit_invalid_reasons": image_reasons, "center_xyz_m": pose[:3].tolist(),
            "quaternion_xyzw": Rotation.from_rotvec(pose[3:]).as_quat().tolist(), "view_metrics": rows,
            "independent_view_center_disagreement_m": disagreement, "optimization": optimization,
            "half_independent_view_disagreement_scale_m": disagreement / 2 if disagreement is not None else None,
            "model_guided_corner_triangulation_diagnostic": corner_diagnostic,
            "translation_std_local_proxy_m": np.sqrt(np.maximum(np.diag(covariance)[:3], 0)).tolist(),
            "multistart_parameter_span": span.tolist()})
        if measurement:
            velocity = np.clip(pose - previous, -.04, .04)
            previous = pose
        if frame % 12 == 0 or frame == n - 1:
            print(f"box frame {frame+1}/{n}: valid={valid}, center={pose[:3].round(4).tolist()}, views={disagreement}", flush=True)
    parameters = np.asarray(parameters)
    frame_valid = np.asarray(frame_valid, bool)
    centers_valid = parameters[:, :3].copy()
    centers_valid[~frame_valid] = np.nan
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    npz = output / "box_trajectory.npz"
    quaternions = Rotation.from_rotvec(parameters[:, 3:]).as_quat()
    np.savez_compressed(npz, timestamps=timestamps, center_xyz_m=parameters[:, :3], quaternion_xyzw=quaternions,
        rotation_vector=parameters[:, 3:], valid=frame_valid, image_fit_valid=np.asarray(image_valid_flags), measurement_backed=np.asarray(measured),
        center_valid_only_xyz_m=centers_valid, covariance_local_proxy=np.asarray(covariances),
        multistart_parameter_span=np.asarray(spreads), independent_view_centers_m=np.asarray(centres),
        independent_view_center_disagreement_m=np.asarray(agreements), box_size_xyz_m=size,
        model_guided_triangulated_corners_world_m=np.asarray(triangulated_corners),
        projection_video=projections, world_corners_m=np.stack([project_pose(p, corners, projections)[2] for p in parameters]))
    trajectory_json = output / "box_trajectory.json"
    trajectory_json.write_text(json.dumps({"coordinate_system": scene["coordinate_system"], "metric_scale_condition": "fixed known cuboid size and supplied camera matrices", "frames": frame_reports}, indent=2) + "\n")
    ply = output / "box_trajectory.ply"
    header = f"ply\nformat ascii 1.0\ncomment Conditional known-box silhouette reconstruction\nelement vertex {n}\nproperty float x\nproperty float y\nproperty float z\nproperty float time\nproperty uchar valid\nend_header\n"
    ply.write_text(header + "".join(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {t:.7f} {int(v)}\n" for p, t, v in zip(parameters, timestamps, frame_valid)))
    z = parameters[frame_valid, 2]
    initial_valid = np.flatnonzero(frame_valid)[:5]
    baseline = float(np.median(parameters[initial_valid, 2])) if len(initial_valid) else None
    lift = float(z.max() - baseline) if baseline is not None else None
    report.update({"status": "conditional_known_box_candidate_reconstructed" if frame_valid.mean() >= .7 else "degraded_geometry_candidate_many_invalid_frames",
        "known_box_size_xyz_m": size.tolist(), "frame_count": n, "valid_frames": int(frame_valid.sum()),
        "image_fit_valid_frames": int(np.sum(image_valid_flags)),
        "invalid_frames": np.flatnonzero(~frame_valid).tolist(), "measurement_backed_frames": int(np.sum(measured)),
        "max_lift_relative_to_first_valid_frames_m": lift, "lift_was_not_prescribed": True,
        "initial_valid_center_z_median": baseline,
        "final_minus_initial_valid_center_z_m": float(z[-1] - baseline) if baseline is not None else None,
        "independent_view_center_disagreement_median_m": float(np.nanmedian(agreements)),
        "half_independent_view_disagreement_median_scale_m": float(np.nanmedian(agreements) / 2),
        "observable_iou_median_per_view": [float(np.median([rows[v]["observable_iou"] for rows in metrics if rows[v] is not None])) for v in range(2)],
        "silhouette_chamfer_mean_median_px_per_view": [float(np.median([rows[v]["chamfer_mean_px"] for rows in metrics if rows[v] is not None and rows[v]["chamfer_mean_px"] is not None])) for v in range(2)],
        "candidate_max_lift_all_measured_frames_m": float(parameters[np.asarray(measured), 2].max() - np.median(parameters[:5, 2])) if any(measured) else None,
        "model_guided_corner_triangulation_frames_with_at_least_two_matches": sum(row["accepted_corner_count"] >= 2 for row in corner_diagnostics),
        "model_guided_corner_triangulation_note": "These model-guided matches are an additional diagnostic and do not override the independent-view size/depth disagreement or establish strict 3D validity.",
        "orientation_symmetry": "Known rectangular cuboid has 180-degree orientation symmetries; nearest initial-pose branch chosen, texture correspondences not used",
        "uncertainty": "Local linearized covariance and multistart spread are conditional proxies, not calibrated confidence intervals; correlated dense contour samples make local covariance optimistic. Camera calibration/model-shape uncertainty is not included. Half the independent-view centre disagreement is separately reported as a conservative diagnostic scale, not a statistical confidence bound.",
        "segmentation_clipping_note": "Input SAM masks may already be clipped by observation_valid. invalid_removed_fraction measures only extra clipping in this script, not the original SAM raw-to-valid area loss. Invalid-region cut boundaries are downweighted.",
        "invalid_pose_policy": "Candidate arrays remain available; valid-only positions are NaN. With no observations, previous pose is carried for display only and flagged invalid.",
        "input_files": [file_record(args.video), file_record(args.scene), file_record(args.geometry), file_record(args.segmentation)],
        "output_files": [file_record(npz), file_record(trajectory_json), file_record(ply)], "frames": frame_reports})
    if not args.no_preview:
        preview = ROOT / "previews/user_box_reconstruction.mp4"
        count = make_preview(args.video, preview, parameters, frame_valid, metrics, corners, projections, timestamps)
        report["preview_decoded_frames"] = count
        report["output_files"].append(file_record(preview))
        export = ROOT / "outputs/previews/user_box_reconstruction.mp4"
        export.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copyfile(preview, export)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=ROOT / "data/video_generation/candidate_scene/generated.mp4")
    parser.add_argument("--scene", type=Path, default=ROOT / "data/video_generation/candidate_scene/scene.json")
    parser.add_argument("--geometry", type=Path, default=ROOT / "outputs/user_reconstruction/geometry.npz")
    parser.add_argument("--segmentation", type=Path, default=ROOT / "outputs/user_reconstruction/segmentation.npz")
    parser.add_argument("--invalid", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/user_reconstruction")
    parser.add_argument("--max-nfev", type=int, default=90)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "status": "running",
        "method": "self-implemented CPU known-cuboid multiview silhouette support + bidirectional edge Chamfer fitting",
        "official_kinogen_pipeline": False, "gpu_used": False, "relative_depth_used_as_metric": False,
        "physical_3d_ground_truth_available": False,
        "limitations": ["Metric result conditional on known cuboid size and supplied cameras", "AI video may deform box or cameras and fail rigid geometry", "Visible SAM mask is not amodal", "No texture corner correspondence, no FoundationPose register", "Temporal regularization is a weak prior; exact lift is estimated rather than supplied", "Not official KinoGen or a physically validated trajectory"]}
    started = time.perf_counter()
    try:
        run(args, report)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        path = ROOT / "reports/user-box-reconstruction.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
