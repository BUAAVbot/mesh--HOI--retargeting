from __future__ import annotations

from pathlib import Path

import numpy as np

from roll_sbto_windows import _window_reference


def test_window_reference_slices_and_replaces_initial_state(tmp_path: Path) -> None:
    source = {
        "qpos_ref": np.arange(30, dtype=np.float64).reshape(5, 6),
        "qvel_ref": np.arange(25, dtype=np.float64).reshape(5, 5),
        "contact_ref": np.zeros((5, 2), dtype=np.float64),
        "fps": np.asarray([50.0]),
    }
    output = tmp_path / "window.npz"
    qpos = np.full(6, 7.0)
    qvel = np.full(5, 8.0)
    _window_reference(source, 1, 4, qpos, qvel, output)
    with np.load(output, allow_pickle=False) as archive:
        assert archive["qpos_ref"].shape == (3, 6)
        assert archive["qvel_ref"].shape == (3, 5)
        np.testing.assert_array_equal(archive["qpos_ref"][0], qpos)
        np.testing.assert_array_equal(archive["qvel_ref"][0], qvel)
        np.testing.assert_array_equal(archive["qpos_ref"][1], source["qpos_ref"][2])
        np.testing.assert_array_equal(archive["fps"], source["fps"])
