"""The orchestration plan is inspectable without body models or rendering."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/two_person_baseline"))
import run as runner
from prepare import timeline_fps


class RunnerTests(unittest.TestCase):
    def test_float32_timeline(self):
        frames = np.arange(172)
        times = (frames / 30.).astype(np.float32)
        self.assertAlmostEqual(timeline_fps(frames, times, np.ones((172, 2), dtype=bool)), 30., places=5)
        times[30] += .01
        with self.assertRaisesRegex(ValueError, "uniform"):
            timeline_fps(frames, times, np.ones(172, dtype=bool))

    def args(self, directory, *extra):
        directory = Path(directory)
        return runner.parser().parse_args([
            "--work-dir", str(directory / "work"),
            "--input-zip", str(directory / "input.zip"),
            "--robot-xml", str(directory / "robot.xml"),
            "--smplx-model-dir", str(directory / "models"),
        ] + list(extra))

    def assets(self, directory):
        directory = Path(directory)
        (directory / "input.zip").touch()
        (directory / "robot.xml").touch()
        (directory / "models").mkdir()

    def test_dry_run_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assets(directory)
            args = self.args(directory)
            argv = ["--work-dir", str(args.work_dir), "--input-zip", str(args.input_zip),
                    "--robot-xml", str(args.robot_xml), "--smplx-model-dir", str(args.smplx_model_dir),
                    "--dry-run"]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                runner.main(argv)
            self.assertIn("--stage build", output.getvalue())
            self.assertIn("--stage train", output.getvalue())
            self.assertIn("smooth_motion.py", output.getvalue())
            self.assertFalse(args.work_dir.exists())

    def test_existing_slots_skip_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assets(directory)
            slots = Path(directory) / "slots.npz"
            slots.touch()
            args = self.args(directory, "--slots", str(slots))
            work, settings, initialize = runner.workspace_settings(args)
            self.assertTrue(initialize)
            self.assertEqual(runner.stage_commands("correspondence", work, settings), [])
            self.assertEqual(len(runner.stage_commands("retarget", work, settings)), 2)

    def test_no_nonempty_or_source_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assets(directory)
            args = self.args(directory)
            args.work_dir.mkdir()
            (args.work_dir / "user.txt").touch()
            with self.assertRaisesRegex(ValueError, "nonempty"):
                runner.workspace_settings(args)
            args.work_dir = ROOT / "scripts"
            with self.assertRaisesRegex(ValueError, "under output"):
                runner.workspace_settings(args)

    def test_asset_identity_pinned(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assets(directory)
            args = self.args(directory)
            work, settings, _ = runner.workspace_settings(args)
            work.mkdir()
            (work / "baseline_settings.json").write_text(json.dumps(settings))
            args.robot_xml = Path(directory) / "other.xml"
            with self.assertRaisesRegex(ValueError, "changed"):
                runner.workspace_settings(args)


if __name__ == "__main__":
    unittest.main()
