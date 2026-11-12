"""Shared, frozen data contract for the Stage-2 backend comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

NATURAL_FREQ = 20.0 * np.pi
ARM_RP90S = 21.913**2 * 9.33e-5
ARM_LZ5735 = 35.0**2 * 1.23e-5
ARM_RP70H = 20.0**2 * 3.975e-5
ARM_WRIST = 0.00316508
ANKLE_ROLL_RATIO = 0.02175 / 0.05

EXPECTED_SOURCE_SHA256 = "c7eb855aaa600e77530b85ba8ef10d97b4fafaaa64cf7716398356f0a3a24406"
EXPECTED_MODEL_SHA256 = "a09e59e9ecc150728bc42ed3a6a194a872e9f4f98e6aa6fb481496e29b1f16a0"
EXPECTED_SCENE_SHA256 = "f81d2fafe847d3651df4c5fda67276311be032e1ce85e098824222a51c847f8a"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_namespace(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def classify_robot_static_bodies(model: Any, mujoco: Any, root_name: str = "base_link") -> tuple[set[int], set[int], list[int]]:
    """Classify bodies by MuJoCo kinematic root, including multiple static support bodies."""
    matches = [
        body_id for body_id in range(1, model.nbody)
        if strip_namespace(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "") == root_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one {root_name!r} body, found {matches}")
    robot_root = int(model.body_rootid[matches[0]])
    robot_order = [
        body_id for body_id in range(1, model.nbody)
        if int(model.body_rootid[body_id]) == robot_root
    ]
    robot = set(robot_order)
    static = set(range(model.nbody)) - robot
    if not robot or not static or robot & static:
        raise ValueError("Invalid robot/static body partition")
    return robot, static, robot_order


def pd_parameters(joint_name: str) -> tuple[float, float, float, float]:
    """Return the HDMI MJLab stiffness, damping, armature and effort limit."""
    if "hip_" in joint_name or joint_name.endswith("knee_joint"):
        armature, effort = ARM_RP90S, 120.0
        return armature * NATURAL_FREQ**2, 2.0 * armature * NATURAL_FREQ, armature, effort
    if joint_name.endswith("ankle_pitch_joint"):
        return 40.0, 2.0, 2.0 * ARM_LZ5735, 141.0
    if joint_name.endswith("ankle_roll_joint"):
        ratio2 = ANKLE_ROLL_RATIO**2
        return 40.0 * ratio2, 2.0 * ratio2, 2.0 * ARM_LZ5735, 141.0
    if joint_name.startswith("waist_"):
        return ARM_LZ5735 * NATURAL_FREQ**2, 2.0 * ARM_LZ5735 * NATURAL_FREQ, ARM_LZ5735, 70.5
    if "shoulder_" in joint_name or joint_name.endswith("elbow_joint"):
        return ARM_RP70H * NATURAL_FREQ**2, 2.0 * ARM_RP70H * NATURAL_FREQ, ARM_RP70H, 60.0
    if "wrist_" in joint_name:
        return 20.0, 1.0, ARM_WRIST, 15.0
    raise KeyError(f"No frozen HDMI actuator parameters for {joint_name!r}")


def configure_position_pd(model: Any, mujoco: Any) -> dict[str, Any]:
    """Convert the compiled torque motors to HDMI-equivalent position servos."""
    mapping: list[dict[str, Any]] = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            raise ValueError(f"Actuator {actuator_id} is not attached to a joint")
        full_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        name = strip_namespace(full_name)
        kp, kd, armature, effort = pd_parameters(name)
        already_position_pd=(model.actuator_gaintype[actuator_id] == mujoco.mjtGain.mjGAIN_FIXED
            and model.actuator_biastype[actuator_id] == mujoco.mjtBias.mjBIAS_AFFINE
            and model.actuator_gainprm[actuator_id,0] > 0
            and np.isclose(model.actuator_biasprm[actuator_id,1],-model.actuator_gainprm[actuator_id,0]))
        model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[actuator_id, :] = 0.0
        model.actuator_biasprm[actuator_id, :] = 0.0
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kd
        model.actuator_forcerange[actuator_id] = (-effort, effort)
        model.actuator_forcelimited[actuator_id] = 1
        if not already_position_pd:
            lo, hi = model.jnt_range[joint_id]
            model.actuator_ctrlrange[actuator_id] = (lo, hi)
            model.actuator_ctrllimited[actuator_id] = 1
        dof_id = int(model.jnt_dofadr[joint_id])
        model.dof_armature[dof_id] = armature
        mapping.append({
            "actuator_id": actuator_id,
            "joint_id": joint_id,
            "joint_name": name,
            "qpos_address": int(model.jnt_qposadr[joint_id]),
            "dof_address": dof_id,
            "kp": kp,
            "kd": kd,
            "armature": armature,
            "effort_limit": effort,
        })
    return {"control_semantics": "joint_position_target_rad", "actuators": mapping}


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def physics_substeps(source_dt: float, sim_dt: float) -> int:
    if not np.isfinite([source_dt, sim_dt]).all() or min(source_dt, sim_dt) <= 0:
        raise ValueError("Clock intervals must be finite and positive")
    count = int(round(source_dt / sim_dt))
    if count < 1 or not np.isclose(count * sim_dt, source_dt, rtol=1e-9, atol=1e-12):
        raise ValueError("Physics timestep must exactly divide the source interval")
    return count


def load_canonical(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {
        "qpos_ref": (35,), "qvel_ref": (34,), "ctrl_ref": (28,),
        "contact_ref": (2,), "contact_pos_ref": (2, 3),
    }
    frames = int(data["qpos_ref"].shape[0])
    for key, tail in required.items():
        if key not in data or data[key].shape != (frames, *tail):
            raise ValueError(f"{key} must have shape {(frames, *tail)}, got {data.get(key, np.empty(0)).shape}")
        if not np.isfinite(data[key]).all():
            raise ValueError(f"{key} contains non-finite values")
    if frames < 2:
        raise ValueError("Canonical reference must have at least two frames")
    return data

