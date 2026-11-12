"""A public NR default need not embed a dataset-specific sequence identifier."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import humanoid_retarget_pipeline_nr as nr


class NRDefaultSelectionTests(unittest.TestCase):
    def select(self, override=None):
        config = json.loads((ROOT / "humanoid_retarget_defaults_nr.json").read_text())
        self.assertEqual(config["motion"]["seq_key"], "")
        motions = {"walk": {"num_frames": 5}, "sit": {"num_frames": 6}}
        with patch.object(nr.nr_source, "is_nr_root", return_value=True):
            with patch.object(nr.common, "load_motion_collection", return_value=(motions, "nr_fbx")):
                return nr.resolve_nr_sequence(config, None, override)

    def test_blank_default_selects_first_motion(self):
        _, key, sequence = self.select()
        self.assertEqual(key, "walk")
        self.assertEqual(sequence["num_frames"], 5)

    def test_explicit_bvh_override(self):
        _, key, sequence = self.select("sit.bvh")
        self.assertEqual(key, "sit")
        self.assertEqual(sequence["num_frames"], 6)

    def test_missing_override_is_not_silently_replaced(self):
        with self.assertRaises(KeyError):
            self.select("missing")


if __name__ == "__main__":
    unittest.main()
