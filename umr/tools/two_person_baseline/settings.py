"""Workspace settings shared by the standalone baseline stages."""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def work_dir():
    return Path(os.environ.get(
        "UMR_BASELINE_WORKDIR", REPO / "output/two_person_baseline"
    )).expanduser().resolve()


def load_settings():
    path = work_dir() / "baseline_settings.json"
    if not path.is_file():
        raise FileNotFoundError(f"Initialize this workspace with run.py first: {path}")
    return json.loads(path.read_text())


def write_settings(settings):
    (work_dir() / "baseline_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n"
    )
