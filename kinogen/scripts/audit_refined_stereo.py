"""Independent held-out foot-track check of the static-only camera refinement."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    import cv2
    import numpy as np
    src = ROOT / "outputs/user_reconstruction"
    out = ROOT / "outputs/user_reconstruction_v2"
    with np.load(src / "tracking.npz", allow_pickle=False) as data:
        tracks, visible, times = data["shared_positions_xy"], data["shared_observation_valid"], data["timestamps"]
    with np.load(src / "geometry.npz", allow_pickle=False) as data:
        baseline_R, baseline_t = data["camera_R_wc"], data["camera_t_wc"]
    camera_path = out / "camera_refinement.npz"
    with np.load(camera_path, allow_pickle=False) as data:
        P, P_old, R, t = data["P_refined"], data["P_baseline"], data["R"], data["t"]

    def reconstruct(projection, rotations, translations):
        count = tracks.shape[2]
        xyz = np.full((121, count, 3), np.nan, np.float32)
        errors = np.full((121, 2, count), np.nan, np.float32)
        valid = np.zeros((121, count), bool)
        _, _, vh = np.linalg.svd(projection[0])
        center = vh[-1]; center /= center[3]
        e = projection[1] @ center
        ex = np.asarray([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
        fundamental = ex @ projection[1] @ np.linalg.pinv(projection[0])
        epipolar = np.full((121, count), np.nan, np.float32)
        for frame in range(121):
            ii = np.flatnonzero(visible[frame].all(axis=0))
            a, b = tracks[frame, 0, ii], tracks[frame, 1, ii]
            hom = cv2.triangulatePoints(projection[0], projection[1], a.T, b.T).T
            p = hom[:, :3] / hom[:, 3:]
            xyz[frame, ii] = p
            ah = np.column_stack((a, np.ones(len(a)))); bh = np.column_stack((b, np.ones(len(b))))
            Fa = ah @ fundamental.T; Ftb = bh @ fundamental
            sampson = np.abs((bh * Fa).sum(1)) / np.sqrt((Fa[:, :2]**2).sum(1) + (Ftb[:, :2]**2).sum(1) + 1e-12)
            epipolar[frame, ii] = sampson
            good = np.isfinite(p).all(axis=1) & (sampson <= 2)
            for view in range(2):
                h = np.column_stack((p, np.ones(len(p)))) @ projection[view].T
                uv = h[:, :2] / h[:, 2:]
                err = np.linalg.norm(uv - tracks[frame, view, ii], axis=1)
                errors[frame, view, ii] = err
                good &= (err <= 3) & ((p @ rotations[view].T + translations[view])[:, 2] > .1)
            valid[frame, ii] = good
        all_observed = np.broadcast_to(visible.all(axis=1)[:, None, :], errors.shape)
        summary = {"valid_point_observations": int(valid.sum()), "available_track_pairs": int(visible.all(axis=1).sum()),
                   "all_observed_reprojection_median_px": float(np.nanmedian(errors[all_observed])),
                   "all_observed_reprojection_p95_px": float(np.nanquantile(errors[all_observed], .95)),
                   "foot_height_median_m": float(np.nanmedian(xyz[:, :, 2][valid])),
                   "scope": "Same 37 independent foot/ankle material-point tracks; no camera fitting on these points"}
        return xyz, errors, valid, epipolar, summary

    _, _, _, _, old = reconstruct(P_old, baseline_R, baseline_t)
    xyz, errors, valid, epipolar, refined = reconstruct(P, R, t)
    np.savez_compressed(out / "sparse_stereo.npz", points_world=xyz, reprojection_error_px=errors,
                        valid=valid, epipolar_sampson_px=epipolar, timestamps=times, camera_sha256=np.asarray(hashlib.sha256(camera_path.read_bytes()).hexdigest()),
                        note=np.asarray("Conditional foot-point stereo; v1 gates: 3px reprojection,2px Sampson,positive Z; validation only, no optimization on these tracks"))
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "independent_track_stereo_audit_completed",
              "baseline": old, "refined": refined, "strict_metric_ground_truth": False,
              "camera_sha256": hashlib.sha256(camera_path.read_bytes()).hexdigest(),
              "warning": "A better static heldout fit does not guarantee better dynamic multi-view geometry; both metrics retained."}
    (ROOT / "reports/user-stereo-independent-audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
