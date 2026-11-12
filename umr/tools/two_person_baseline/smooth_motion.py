"""Whole-sequence sparse SQP smoothing with contact, speed and collision constraints."""
from settings import work_dir, load_settings
import hashlib
import json

import clarabel
import mujoco
import numpy as np
from scipy import sparse
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation

BASE = work_dir()
OUT = BASE / "smoothing_v1"
FRAMES, FPS = load_settings()["frames"], load_settings()["fps"]
NV = 68


def chart(q):
    y = np.zeros((len(q), NV))
    for i in range(2):
        y[:, 34*i:34*i+3] = q[:, 35*i:35*i+3]
        y[:, 34*i+6:34*i+34] = q[:, 35*i+7:35*i+35]
    return y


def stats(q):
    actors = []
    for i in range(2):
        a = q[:, i*35:(i+1)*35]
        joints = a[:, 7:]
        rot = Rotation.from_quat(a[:, [4, 5, 6, 3]])
        rotation_steps = np.rad2deg((rot[:-1].inv() * rot[1:]).magnitude())
        step = np.rad2deg(np.diff(joints, axis=0))
        accel = np.rad2deg(np.diff(joints, n=2, axis=0)) * FPS**2
        root_accel = np.diff(a[:, :3], n=2, axis=0) * FPS**2
        actors.append({
            "max_joint_step_deg": float(abs(step).max()),
            "joint_acceleration_rms_deg_s2": float(np.sqrt(np.mean(accel**2))),
            "root_acceleration_rms_m_s2": float(np.sqrt(np.mean(root_accel**2))),
            "max_root_step_m": float(np.linalg.norm(np.diff(a[:, :3], axis=0), axis=1).max()),
            "max_root_rotation_step_deg": float(rotation_steps.max()),
        })
    return actors


def main():
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    files = [np.load(BASE / f"results/person_{i:02d}_grounded.npz", allow_pickle=True) for i in (1, 2)]
    raw = np.concatenate([a["qpos"] for a in files], axis=1).astype(float)
    assert raw.shape == (FRAMES, 70) and np.isfinite(raw).all()
    model = mujoco.MjModel.from_xml_path(str(BASE / "preview_grounded/dual_scene.xml"))
    if model.nq != 70 or model.nv != NV:
        raise ValueError("This preset requires two Vita02A robots, each nq=35 and nv=34.")
    data = mujoco.MjData(model)
    seed = raw.copy()
    for i in range(2):
        offset = 35*i
        for j in range(3):
            v = raw[:, offset+j]
            local_median = median_filter(v, size=11, mode="nearest")
            deglitched = np.where(abs(v-local_median) > .06, local_median, v)
            seed[:, offset+j] = gaussian_filter1d(deglitched, 2., mode="nearest")
        seed[:, offset+7:offset+35] = gaussian_filter1d(
            median_filter(raw[:, offset+7:offset+35], size=(5, 1), mode="nearest"),
            1.6, axis=0, mode="nearest")
        rotations = Rotation.from_quat(raw[:, [offset+4, offset+5, offset+6, offset+3]])
        quats = []
        for t in range(FRAMES):
            indices = np.arange(max(0, t-6), min(FRAMES, t+7))
            weights = np.exp(-.5*((indices-t)/2.)**2)
            quat = rotations[indices].mean(weights=weights).as_quat()[[3, 0, 1, 2]]
            if quats and np.dot(quat, quats[-1]) < 0:
                quat *= -1
            quats.append(quat)
        seed[:, offset+3:offset+7] = quats
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            adr = model.jnt_qposadr[j]
            seed[:, adr] = np.clip(seed[:, adr], *model.jnt_range[j])

    def geom_id(name):
        result = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert result >= 0, name
        return result

    floor = geom_id("floor")
    seats = [geom_id(f"seat_{i:02d}") for i in (1, 2)]
    geoms, feet = [], []
    for person in (1, 2):
        geoms.append([g for g in range(model.ngeom)
                      if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                          int(model.geom_bodyid[g])) or "").startswith(f"p{person}_")
                      and (model.geom_contype[g] or model.geom_conaffinity[g])])
        feet.append([geom_id(f"p{person}_collision_foot_plane_{side}_box") for side in ("l", "r")])
    foot_targets = np.empty((FRAMES, 2, 2, 3))
    for t in range(FRAMES):
        data.qpos[:] = seed[t]
        mujoco.mj_forward(model, data)
        foot_targets[t] = [[data.geom_xpos[g].copy() for g in group] for group in feet]
    foot_targets = gaussian_filter1d(foot_targets, 1.5, axis=0, mode="nearest")
    support = np.zeros((FRAMES, 2, 2), dtype=bool)
    for actor in range(2):
        current = int(foot_targets[0, actor, :, 2].argmin())
        for t in range(FRAMES):
            heights = foot_targets[t, actor, :, 2]
            other = 1-current
            if heights[other] < heights[current]-.012:
                current = other
            support[t, actor, current] = True
            if (t < FRAMES*14/172 or t > FRAMES*130/172) and abs(heights[0]-heights[1]) < .04:
                support[t, actor] = True
    np.savez_compressed(OUT / "support_schedule.npz", support=support, foot_center_targets=foot_targets)

    n = FRAMES*NV
    identity = sparse.eye(n, format="csc")
    d1 = sparse.diags([-np.ones(FRAMES-1), np.ones(FRAMES-1)], [0, 1],
                      shape=(FRAMES-1, FRAMES), format="csc")
    d2 = sparse.diags([np.ones(FRAMES-2), -2*np.ones(FRAMES-2), np.ones(FRAMES-2)],
                      [0, 1, 2], shape=(FRAMES-2, FRAMES), format="csc")
    d3 = sparse.diags([-np.ones(FRAMES-3), 3*np.ones(FRAMES-3),
                      -3*np.ones(FRAMES-3), np.ones(FRAMES-3)], [0, 1, 2, 3],
                      shape=(FRAMES-3, FRAMES), format="csc")
    dof_weights = np.tile(np.r_[100., 100., 100., 4., 4., 4., np.ones(28)], 2)
    temporal_weights = dof_weights.copy()
    moving = np.ones(NV, dtype=bool)
    for i in range(2):
        moving[34*i+3:34*i+6] = False
        temporal_weights[34*i+3:34*i+6] = 0
    w = sparse.diags(np.tile(dof_weights, FRAMES), format="csc")
    h = sparse.kron(16*(d2.T@d2)+50*(d3.T@d3),
                    sparse.diags(temporal_weights), format="csc")
    regularizer = w+h
    velocity = sparse.kron(d1, sparse.eye(NV), format="csc")
    velocity = velocity[np.tile(moving, FRAMES-1)]
    speed_limit = np.tile(np.r_[.045, .045, .03, .1, .1, .1,
                                 np.full(28, np.deg2rad(7.))], 2)
    speed_limit *= 30. / FPS
    vcap = np.tile(speed_limit, FRAMES-1)[np.tile(moving, FRAMES-1)]
    target = chart(seed).ravel()
    q = seed.copy()
    logs = []
    for iteration in range(10):
        y = chart(q).ravel()
        lower = np.tile(np.tile(np.r_[np.full(3, -.08), np.full(3, -1e-10),
                                      np.full(28, -.25)], 2), FRAMES)
        upper = -lower
        for joint in range(model.njnt):
            if model.jnt_limited[joint]:
                dof, adr = int(model.jnt_dofadr[joint]), int(model.jnt_qposadr[joint])
                lo, hi = model.jnt_range[joint]
                lower[dof::NV] = np.maximum(lower[dof::NV], lo-q[:, adr])
                upper[dof::NV] = np.minimum(upper[dof::NV], hi-q[:, adr])
        env_r, env_c, env_v, env_b = [], [], [], []
        obj_r, obj_c, obj_v, obj_b = [], [], [], []
        minimum = 1.

        def add_environment(jac, distance, t, margin=.0015):
            row = len(env_b)
            ids = np.flatnonzero(abs(jac) > 1e-12)
            env_r.extend([row]*len(ids)); env_c.extend((t*NV+ids).tolist())
            env_v.extend((-jac[ids]).tolist()); env_b.append(distance-margin)

        def add_objective(jac, residual, t, weight):
            row = len(obj_b)
            ids = np.flatnonzero(abs(jac) > 1e-12)
            obj_r.extend([row]*len(ids)); obj_c.extend((t*NV+ids).tolist())
            obj_v.extend((weight*jac[ids]).tolist()); obj_b.append(weight*residual)

        def jacobian(point, body):
            jac = np.zeros((3, NV))
            mujoco.mj_jac(model, data, jac, None, point, body)
            return jac

        for t in range(FRAMES):
            data.qpos[:] = q[t]
            mujoco.mj_forward(model, data)
            for actor in range(2):
                for geom in geoms[actor]:
                    body = int(model.geom_bodyid[geom])
                    for obstacle in [floor, *seats]:
                        ft = np.zeros(6)
                        distance = mujoco.mj_geomDistance(model, data, geom, obstacle, .055, ft)
                        minimum = min(minimum, distance)
                        if distance >= .05 or abs(distance) < 1e-10:
                            continue
                        normal = (ft[:3]-ft[3:])/distance
                        add_environment(normal@jacobian(ft[:3], body), distance, t)
                for side, geom in enumerate(feet[actor]):
                    body = int(model.geom_bodyid[geom])
                    point = data.geom_xpos[geom].copy()
                    jac = jacobian(point, body)
                    for axis in (0, 1):
                        add_objective(jac[axis], point[axis]-foot_targets[t, actor, side, axis],
                                      t, 12.)
                    if support[t, actor, side]:
                        ft = np.zeros(6)
                        distance = mujoco.mj_geomDistance(model, data, geom, floor, .5, ft)
                        add_objective(jacobian(ft[:3], body)[2], distance-.011, t, 65.)
            # Only nearby cross-robot pairs need linearized constraints.
            for first in geoms[0]:
                for second in geoms[1]:
                    ft = np.zeros(6)
                    distance = mujoco.mj_geomDistance(model, data, first, second, .12, ft)
                    if distance >= .115 or abs(distance) < 1e-10:
                        continue
                    normal = (ft[:3]-ft[3:])/distance
                    jac = normal@(jacobian(ft[:3], int(model.geom_bodyid[first])) -
                                  jacobian(ft[3:], int(model.geom_bodyid[second])))
                    add_environment(jac, distance, t, margin=.06)
        env = sparse.coo_matrix((env_v, (env_r, env_c)), shape=(len(env_b), n)).tocsc()
        objective = sparse.coo_matrix((obj_v, (obj_r, obj_c)), shape=(len(obj_b), n)).tocsc()
        p = regularizer+objective.T@objective+sparse.eye(n, format="csc")*.001
        linear = w@(y-target)+h@y+objective.T@np.asarray(obj_b)
        count = len(env_b)
        p = sparse.block_diag([p, sparse.eye(count, format="csc")*1e7], format="csc")
        z = lambda rows: sparse.csc_matrix((rows, count))
        constraints = sparse.vstack([
            sparse.hstack([identity, z(n)]), sparse.hstack([-identity, z(n)]),
            sparse.hstack([velocity, z(velocity.shape[0])]),
            sparse.hstack([-velocity, z(velocity.shape[0])]),
            sparse.hstack([env, -sparse.eye(count)]),
            sparse.hstack([sparse.csc_matrix((count, n)), -sparse.eye(count)])
        ], format="csc")
        bounds = np.r_[upper, -lower, vcap-velocity@y, vcap+velocity@y,
                       np.asarray(env_b), np.zeros(count)]
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.max_iter = 150
        settings.tol_feas = 1e-8
        solver = clarabel.DefaultSolver(sparse.triu(p, format="csc"),
                                       np.r_[linear, np.zeros(count)],
                                       constraints, bounds,
                                       [clarabel.NonnegativeConeT(len(bounds))], settings)
        solution = solver.solve()
        status = str(solution.status)
        if status not in ("Solved", "AlmostSolved"):
            raise RuntimeError(f"Temporal QP failed: {status}")
        delta = np.asarray(solution.x[:n]).reshape(FRAMES, NV)
        for t in range(FRAMES):
            mujoco.mj_integratePos(model, q[t], delta[t], 1.)
        record = {"iteration": iteration, "status": status,
                  "max_step": float(abs(delta).max()), "current_min_environment_distance_m": minimum}
        logs.append(record)
        print(json.dumps(record), flush=True)
        if iteration >= 5 and abs(delta).max() < 2e-4 and minimum > -.0005:
            break
    for person, original in enumerate(files, 1):
        payload = {key: original[key] for key in original.files
                   if key != "postprojection_seat_clearance_translation_m"}
        payload["qpos"] = q[:, (person-1)*35:person*35].astype(np.float32)
        payload["postprocess"] = np.asarray("smoothing_v1: whole-sequence constrained temporal SQP")
        np.savez_compressed(OUT / f"results/person_{person:02d}_smoothed.npz", **payload)
    before, after = stats(raw), stats(q)
    converged = logs[-1]["max_step"] < 2e-4 and logs[-1]["current_min_environment_distance_m"] > -.0005
    report = {"before": before, "after": after, "iterations": logs,
              "converged": converged, "termination": "converged" if converged else "iteration_limit",
              "input_sha256": [hashlib.sha256((BASE / f"results/person_{i:02d}_grounded.npz").read_bytes()).hexdigest()
                               for i in (1, 2)],
              "frame_count": FRAMES, "fps": FPS, "dynamics_validated": False}
    (OUT / "smoothing_report.json").write_text(json.dumps(report, indent=2)+"\n")
    for i in range(2):
        assert after[i]["max_joint_step_deg"] < 7.1 * 30. / FPS
        assert after[i]["max_root_step_m"] < .08 * 30. / FPS
        assert after[i]["joint_acceleration_rms_deg_s2"] < before[i]["joint_acceleration_rms_deg_s2"]*.5
    print(json.dumps({"before": before, "after": after}, indent=2), flush=True)


if __name__ == "__main__":
    main()
