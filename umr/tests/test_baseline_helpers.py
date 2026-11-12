"""Asset-free regression checks for the local retargeting extensions."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import clarabel
import mujoco
import numpy as np
from scipy import sparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from retarget_output_paths import floating_mjcf_path
from source_validity_export import selected_source_validity
from soma_identity_cache import install
from soma_lbs_optimized import lbs_vertex_chunked, lbs_blended_transform
from qp_numerical_retry import cone_violation, retry_same_qp
import smpl_surface_retarget_common as common


class OutputPathTests(unittest.TestCase):
    def test_legacy_name(self):
        self.assertEqual(floating_mjcf_path("/tmp/walk.npz"), Path("/tmp/walk.floating_mjcf.xml"))

    def test_long_names_and_distinct_directories(self):
        name = "换" * 80 + ".npz"
        first = floating_mjcf_path(Path("/tmp/a") / name, "/tmp")
        second = floating_mjcf_path(Path("/tmp/b") / name, "/tmp")
        self.assertLessEqual(len(os.fsencode(first.name)), 255)
        self.assertNotEqual(first, second)
        self.assertEqual(first, floating_mjcf_path(Path("/tmp/a") / name, "/tmp"))


class SourceValidityTests(unittest.TestCase):
    def test_selected_timeline_and_bad_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.npz"
            np.savez(path, source_valid=np.array([True, False, True]))
            np.testing.assert_array_equal(selected_source_validity(path, [2, 1]), [True, False])
            for ids in ([-1], [3], [0.5], [[0]]):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    selected_source_validity(path, ids)
            np.savez(path, source_valid=np.ones(3))
            with self.assertRaises(ValueError):
                selected_source_validity(path, [0])

    def test_optional_mask(self):
        self.assertIsNone(selected_source_validity("/tmp/motion.bvh", [0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motion.npz"
            np.savez(path, poses=np.zeros((1, 3)))
            self.assertIsNone(selected_source_validity(path, [0]))


class SomaTests(unittest.TestCase):
    def test_identity_cache_invalidation_and_grad_mode(self):
        class Layer:
            calls = 0

            def prepare_identity(self, *args):
                self.calls += 1

        layer = Layer()
        original = layer.prepare_identity
        stats, restore = install(layer)
        identity = torch.zeros(2)
        with torch.no_grad():
            layer.prepare_identity(identity, kwargs={"x": (1.,)})
            layer.prepare_identity(identity.clone(), kwargs={"x": (1.,)})
            self.assertEqual(layer.calls, 1)
            identity[0] = 1
            layer.prepare_identity(identity, kwargs={"x": (1.,)})
            self.assertEqual(layer.calls, 2)
        layer.prepare_identity(identity)
        layer.prepare_identity(identity)
        self.assertEqual(layer.calls, 4)
        self.assertEqual(stats, {"prepared": 4, "reused": 1})
        restore()
        self.assertEqual(layer.prepare_identity, original)

    def test_lbs_equivalence(self):
        from soma.geometry.lbs import lbs
        torch.manual_seed(0)
        points = torch.randn(2, 1041, 3, dtype=torch.float64)
        weights = torch.softmax(torch.randn(1041, 4, dtype=torch.float64), dim=1)
        transforms = torch.randn(2, 4, 4, 4, dtype=torch.float64)
        expected = lbs(points, weights, transforms)
        torch.testing.assert_close(lbs_vertex_chunked(points, weights, transforms, 128), expected)
        torch.testing.assert_close(lbs_blended_transform(points, weights, transforms), expected)


class QPRetryTests(unittest.TestCase):
    def test_cone_violation(self):
        self.assertEqual(cone_violation(np.array([-1., 2., 1., 1.]),
                                       [clarabel.NonnegativeConeT(1), clarabel.SecondOrderConeT(3)]), 1.)

    def test_same_feasible_qp(self):
        p = sparse.csc_matrix([[1.]])
        a = sparse.csc_matrix([[1.], [-1.]])
        with patch.dict(os.environ, {"UMR_QP_CAPTURE_ONLY": "", "UMR_QP_STEP_RETRY": ""}):
            result = retry_same_qp(p, np.array([-1.]), a, np.array([2., 2.]),
                                   [clarabel.NonnegativeConeT(2)], [-2.], [2.], 1, "test")
        np.testing.assert_allclose(result, [1.], atol=1e-6)

    def test_infeasible_qp_not_promoted(self):
        with patch.dict(os.environ, {"UMR_QP_CAPTURE_ONLY": "", "UMR_QP_STEP_RETRY": ""}):
            with self.assertRaisesRegex(RuntimeError, "Same-QP numerical retries failed"):
                retry_same_qp(sparse.csc_matrix([[1.]]), np.zeros(1),
                              sparse.csc_matrix([[1.], [-1.]]), np.array([-1., -1.]),
                              [clarabel.NonnegativeConeT(2)], [-2.], [2.], 1, "test")


class GroundCollisionTests(unittest.TestCase):
    def model(self, explicit=False):
        pair = '<contact><pair geom1="floor" geom2="ball"/></contact>' if explicit else ""
        return mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom name="floor" type="plane" size="2 2 .1" contype="0" conaffinity="0"/>'
            '<body name="robot" pos="0 0 1"><freejoint/><geom name="ball" type="sphere" size=".1"/>'
            '</body></worldbody>' + pair + '</mujoco>')

    def args(self):
        return SimpleNamespace(ground_penetration_hard_constraint=True,
                               ground_penetration_hard_constraint_mode="mujoco_collision")

    def test_disabled_floor_rejected(self):
        with self.assertRaisesRegex(ValueError, "no ground/robot pair"):
            common.build_ground_penetration_collision_cache(self.model(), self.args())

    def test_explicit_pair_and_unbounded_activation(self):
        model = self.model(explicit=True)
        cache = common.build_ground_penetration_collision_cache(model, self.args())
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        margins = model.geom_margin.copy()
        rows, distances = common.compute_ground_penetration_collision_rows(model, data, cache, 0., -1.)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(distances[0], .9)
        np.testing.assert_array_equal(model.geom_margin, margins)


if __name__ == "__main__":
    unittest.main()
