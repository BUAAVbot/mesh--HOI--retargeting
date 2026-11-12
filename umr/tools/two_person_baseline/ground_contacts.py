"""Kinematic contact projection, preserving foot XY and the retargeted walking path."""
from pathlib import Path
import json
import sys

import mujoco
import numpy as np

from settings import REPO, work_dir

HERE = work_dir()
sys.path.insert(0, str(REPO / "scripts"))
from smpl_surface_retarget_common import solve_clarabel_qp_step


def main():
    model = mujoco.MjModel.from_xml_path(str(HERE / "preview_refined/dual_scene.xml"))
    if model.nq != 70 or model.nv != 68:
        raise ValueError("This preset requires two Vita02A robots, each nq=35 and nv=34.")
    data = mujoco.MjData(model)
    files = [np.load(HERE / f"results/person_{i:02d}_refined.npz", allow_pickle=True) for i in (1, 2)]
    original = np.concatenate([a["qpos"] for a in files], axis=1).astype(float)
    output = original.copy()
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    seats = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"seat_{i:02d}") for i in (1, 2)]
    diagnostics = []

    def name(kind, i):
        return mujoco.mj_id2name(model, kind, i) or ""

    def point_jacobian(point, body, dofs):
        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jac, None, point, body)
        return jac[:, dofs]

    for person in (1, 2):
        qslice = slice((person - 1) * 35, person * 35)
        dofs = np.arange((person - 1) * 34, person * 34)
        geoms = [g for g in range(model.ngeom)
                 if name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])).startswith(f"p{person}_")
                 and (model.geom_contype[g] or model.geom_conaffinity[g])]
        foot_groups = [[g for g in geoms if any(key in name(mujoco.mjtObj.mjOBJ_GEOM, g)
                       for key in [f"foot_plane_{side}", f"foot_front_{side}", f"foot_back_{side}",
                                   f"toe_0_{side}", f"toe_1_{side}"])] for side in ("l", "r")]
        weights = np.ones(34)
        weights[:2] = 80
        weights[2] = 3
        weights[3:6] = 30
        weights[18:] = 5
        for frame in range(len(output)):
            data.qpos[:] = original[frame]
            mujoco.mj_forward(model, data)
            contacts = []
            for group in foot_groups:
                distances = []
                for geom in group:
                    ft = np.zeros(6)
                    dist = mujoco.mj_geomDistance(model, data, geom, floor, 1., ft)
                    distances.append((dist, geom, ft[:3].copy()))
                dist, geom, point = min(distances, key=lambda item: item[0])
                body = int(model.geom_bodyid[geom])
                local = data.xmat[body].reshape(3, 3).T @ (point - data.xpos[body])
                contacts.append((dist, body, local, point))
            # This clip is walking and sitting, not jumping: one foot remains supported.
            root_shift = .003 - min(item[3][2] for item in contacts)
            targets = [point + [0, 0, root_shift] for _, _, _, point in contacts]
            goal = original[frame].copy()
            goal[(person - 1) * 35 + 2] += root_shift
            current = goal.copy()
            for iteration in range(24):
                data.qpos[:] = current
                mujoco.mj_forward(model, data)
                error = np.zeros(model.nv)
                mujoco.mj_differentiatePos(model, error, 1., goal, current)
                rows = [np.diag(weights)]
                residuals = [weights * error[dofs]]
                for (_, body, local, _), target in zip(contacts, targets):
                    point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
                    rows.append(200 * point_jacobian(point, body, dofs))
                    residuals.append(200 * (point - target))
                inequalities, bounds = [], []
                min_distance = 1.
                for geom in geoms:
                    body = int(model.geom_bodyid[geom])
                    for obstacle in [floor, *seats]:
                        ft = np.zeros(6)
                        dist = mujoco.mj_geomDistance(model, data, geom, obstacle, .06, ft)
                        min_distance = min(min_distance, dist)
                        if dist >= .055 or abs(dist) < 1e-10:
                            continue
                        # Signed witness separation gives the distance derivative through penetration.
                        normal = (ft[:3] - ft[3:]) / dist
                        jac = normal @ point_jacobian(ft[:3], body, dofs)
                        inequalities.append(-jac)
                        bounds.append(dist - .001)
                lower, upper = np.full(34, -.12), np.full(34, .12)
                for joint in range(model.njnt):
                    dadr = int(model.jnt_dofadr[joint])
                    if dadr not in dofs or not model.jnt_limited[joint]:
                        continue
                    offset = dadr - dofs[0]
                    qadr = int(model.jnt_qposadr[joint])
                    lo, hi = model.jnt_range[joint]
                    lower[offset] = max(lower[offset], lo - current[qadr])
                    upper[offset] = min(upper[offset], hi - current[qadr])
                step = solve_clarabel_qp_step(
                    np.vstack(rows), np.concatenate(residuals), .01, lower, upper,
                    ineq_A=np.asarray(inequalities).reshape(-1, 34),
                    ineq_b=np.asarray(bounds), ineq_soft_cost=1e8)
                fullstep = np.zeros(model.nv)
                fullstep[dofs] = step
                mujoco.mj_integratePos(model, current, fullstep, 1.)
                if np.linalg.norm(step) < 2e-5 and min_distance > -.001:
                    break
            output[frame, qslice] = current[qslice]
            data.qpos[:] = output[frame]
            mujoco.mj_forward(model, data)
            errors = []
            for (_, body, local, _), target in zip(contacts, targets):
                point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
                errors.append(float(np.linalg.norm(point - target)))
            diagnostics.append({"person": person, "frame": frame, "initial_root_shift_m": root_shift,
                                "max_foot_anchor_error_m": max(errors), "iterations": iteration + 1})
            if frame % 40 == 0:
                print("projected", person, frame, "foot_error", max(errors), flush=True)
    # Resolve small nonlinear residuals with a bounded, minimum-distance root translation.
    final_shifts = np.zeros((len(output), 2, 3))
    for person in (1, 2):
        geoms = [g for g in range(model.ngeom)
                 if name(mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])).startswith(f"p{person}_")
                 and (model.geom_contype[g] or model.geom_conaffinity[g])]
        for frame in range(len(output)):
            for _ in range(12):
                data.qpos[:] = output[frame]
                mujoco.mj_forward(model, data)
                rows, bounds = [], []
                worst = 1.
                for geom in geoms:
                    for obstacle in [floor, *seats]:
                        ft = np.zeros(6)
                        dist = mujoco.mj_geomDistance(model, data, geom, obstacle, .025, ft)
                        worst = min(worst, dist)
                        if dist >= .023 or abs(dist) < 1e-10:
                            continue
                        normal = (ft[:3] - ft[3:]) / dist
                        rows.append(-normal)
                        bounds.append(dist - .0012)
                if worst >= .001:
                    break
                shift = solve_clarabel_qp_step(
                    np.eye(3), np.zeros(3), .01, np.full(3, -.015), np.full(3, .015),
                    ineq_A=np.asarray(rows).reshape(-1, 3), ineq_b=np.asarray(bounds))
                root = slice((person - 1) * 35, (person - 1) * 35 + 3)
                output[frame, root] += shift
                final_shifts[frame, person - 1] += shift
            if np.linalg.norm(final_shifts[frame, person - 1]) > .02:
                raise RuntimeError("Residual seat correction exceeds the 20mm preview repair limit")
    for person, original_file in enumerate(files, 1):
        payload = {key: original_file[key] for key in original_file.files}
        payload["qpos"] = output[:, (person - 1) * 35:person * 35].astype(np.float32)
        payload["postprocess"] = np.asarray("ground_contacts.py: constrained kinematic foot-anchor projection")
        payload["postprojection_seat_clearance_translation_m"] = final_shifts[:, person - 1]
        np.savez_compressed(HERE / f"results/person_{person:02d}_grounded.npz", **payload)
    (HERE / "ground_contact_projection.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (HERE / "ground_contact_final_lifts.json").write_text(json.dumps({
        "max_seat_clearance_translation_m": np.linalg.norm(final_shifts, axis=2).max(axis=0).tolist(),
        "frames_adjusted": (np.linalg.norm(final_shifts, axis=2) > 1e-6).sum(axis=0).tolist()}, indent=2) + "\n")
    print("Contact projection complete", flush=True)


if __name__ == "__main__":
    main()
