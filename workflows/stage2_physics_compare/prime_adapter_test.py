from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import numpy as np

import prime_adapter


class PrimeAdapterTest(unittest.TestCase):
    def test_prepare_converts_wxyz_and_writes_strict_torque_input(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            canonical = root / "canonical.npz"
            urdf = root / "robot.urdf"
            urdf.write_text("<robot name='test'/>", encoding="utf-8")
            q = np.zeros((3, 35))
            q[:, 3] = 1.0
            v = np.zeros((3, 34))
            tau = np.ones((3, 28))
            np.savez(canonical, qpos_ref=q, qvel_ref=v, tau_ref=tau, dt=0.02)
            args = argparse.Namespace(
                canonical=str(canonical), urdf=str(urdf), output=str(root / "out"),
                nq=35, nv=34, dt=0.02, input_quaternion="wxyz",
                contacts="left_ankle_roll_link,right_ankle_roll_link", identify="base_link",
                max_iter=2, threads=1, kappa=10.0, mu=0.8, dry_run=True,
            )
            self.assertEqual(prime_adapter.prepare(args), 0)
            saved_q = np.loadtxt(root / "out" / "q_ref_pinocchio.csv", delimiter=",")
            np.testing.assert_allclose(saved_q[:, 4:7], 0.0)
            np.testing.assert_allclose(saved_q[:, 7], 1.0)
            saved_u = np.loadtxt(root / "out" / "tau_ref.csv", delimiter=",")
            self.assertEqual(saved_u.shape, (3, 29))
            self.assertIn('dry_run="true"', (root / "out" / "prime_vita.xml").read_text())

    def test_prepare_refuses_position_target_as_torque(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            canonical = root / "canonical.npz"
            urdf = root / "robot.urdf"
            urdf.write_text("<robot name='test'/>", encoding="utf-8")
            q = np.zeros((2, 35)); q[:, 3] = 1.0
            np.savez(canonical, qpos_ref=q, qvel_ref=np.zeros((2, 34)), ctrl_ref=np.zeros((2, 28)))
            args = argparse.Namespace(
                canonical=str(canonical), urdf=str(urdf), output=str(root / "out"),
                nq=35, nv=34, dt=0.02, input_quaternion="wxyz", contacts="a,b",
                identify="base_link", max_iter=2, threads=1, kappa=10.0, mu=0.8, dry_run=True,
            )
            with self.assertRaisesRegex(ValueError, "missing required array"):
                prime_adapter.prepare(args)

    def test_collect_requires_declared_force_semantics(self) -> None:
        parser = prime_adapter.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["collect", "--source", "x", "--output", "y"])

    def test_collect_emits_complete_prime_certificate(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            source = root / "official"
            source.mkdir()
            # nq + nv + two 10D parameter blocks for one identified link.
            xs = np.zeros((3, 35 + 34 + 20))
            rollout = xs.copy()
            rollout[-1, 0] = 1e-4
            us = np.zeros((2, 34))
            forces = np.asarray([[0.1, 0.0, 1.0, 0.0, 0.1, 1.0]] * 2)
            np.savetxt(source / "xs_results_fddp.csv", xs, delimiter=",")
            np.savetxt(source / "xs_rollout.csv", rollout, delimiter=",")
            np.savetxt(source / "us_results_fddp.csv", us, delimiter=",")
            np.savetxt(source / "f_rollout.csv", forces, delimiter=",")
            args = argparse.Namespace(
                source=str(source), output=str(root / "normalized"), nq=35, nv=34,
                identified_links=1, mu=0.8, force_layout="xyz",
            )
            self.assertEqual(prime_adapter.collect(args), 0)
            with np.load(root / "normalized" / "physics_certificate.npz") as certificate:
                self.assertIn("dynamics_state_residual", certificate.files)
                self.assertIn("friction_cone_residual", certificate.files)
            with np.load(root / "normalized" / "controls.npz") as controls:
                self.assertEqual(controls["joint_torque"].shape, (2, 28))


if __name__ == "__main__":
    unittest.main()
