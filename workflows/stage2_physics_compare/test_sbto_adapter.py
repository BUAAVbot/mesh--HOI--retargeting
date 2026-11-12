from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("sbto_adapter.py")
SPEC = importlib.util.spec_from_file_location("sbto_adapter", MODULE_PATH)
assert SPEC and SPEC.loader
ADAPTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTER)


def test_reference_resampling_keeps_unit_quaternion() -> None:
    qpos = np.zeros((3, 9), dtype=np.float64)
    qpos[:, 2] = 1.0
    qpos[:, 3] = 1.0
    qpos[1, 3] = -1.0  # equivalent sign must be made continuous
    qvel = np.zeros((3, 8), dtype=np.float64)
    sim_t, qpos_sim, qvel_sim, ctrl_sim = ADAPTER.resample_reference(
        qpos, qvel, None, fps=2.0, dt=0.25, act_qposadr=np.asarray((7, 8))
    )
    assert sim_t.shape == (5,)
    assert qpos_sim.shape == (5, 9)
    assert qvel_sim.shape == (5, 8)
    assert ctrl_sim.shape == (4, 2)
    np.testing.assert_allclose(np.linalg.norm(qpos_sim[:, 3:7], axis=1), 1.0)


def test_hdmi_gain_groups_are_complete() -> None:
    for name in (
        "left_hip_pitch_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
    ):
        kp, kd, armature, effort = ADAPTER._joint_pd(name)
        assert kp > 0 and kd > 0 and armature > 0 and effort > 0
