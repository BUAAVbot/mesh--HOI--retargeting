"""Training must stay closed for failed, stale, or reset-based optimizations."""
import json
from pathlib import Path
import tempfile
import unittest

from train_verified_reference import sha256, verify_certificate


class TrainingGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.motion = self.root / "motion.npz"
        self.motion.write_bytes(b"accepted physical reference fixture")
        self.artifacts = {"optimized_reference_sha256": self.motion}
        self.report = {
            "schema": "hdmi.trajopt.acceptance.v1", "passed": True, "accepted": True,
            "scope": "continuous_nominal_simulation_trajectory", "failed_checks": [],
            "optimized_reference_sha256": sha256(self.motion),
            "batch_single_world_scene_alignment_verified": True,
            "support_contact_filter_verified": True,
            "fresh_replay": {key: True for key in (
                "performed", "continuous", "no_midrollout_reset", "no_root_state_write",
                "no_external_root_force", "serialized_controls_reloaded")},
        }
        self.certificate = self.root / "acceptance.json"

    def verify(self):
        self.certificate.write_text(json.dumps(self.report), encoding="utf-8")
        return verify_certificate(self.certificate, self.artifacts)

    def test_accepted_unchanged_reference(self):
        self.assertTrue(self.verify()["passed"])

    def test_reject_failed_optimization(self):
        self.report["passed"] = False
        with self.assertRaises(ValueError):
            self.verify()

    def test_reject_postacceptance_reference_edit(self):
        self.motion.write_bytes(b"different reference")
        with self.assertRaises(ValueError):
            self.verify()

    def test_reject_midrollout_reset(self):
        self.report["fresh_replay"]["no_midrollout_reset"] = False
        with self.assertRaises(ValueError):
            self.verify()

    def test_reject_contradictory_certificate(self):
        self.report["failed_checks"] = ["penetration"]
        with self.assertRaises(ValueError):
            self.verify()

    def test_reject_unverified_batch_scene(self):
        self.report["batch_single_world_scene_alignment_verified"] = False
        with self.assertRaises(ValueError):
            self.verify()

    def test_reject_selfcontact_as_support(self):
        self.report["support_contact_filter_verified"] = False
        with self.assertRaises(ValueError):
            self.verify()


if __name__ == "__main__":
    unittest.main()
