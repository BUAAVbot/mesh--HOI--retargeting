"""Portable, explicit stage runner for the two-person Vita02A baseline."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from settings import REPO

TOOLS = Path(__file__).resolve().parent
STAGES = ("prepare", "correspondence", "retarget", "scene", "ground", "smooth", "render")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--work-dir", type=Path, required=True)
    result.add_argument("--stage", choices=("all",) + STAGES, default="all")
    result.add_argument("--input-zip", type=Path)
    result.add_argument("--robot-xml", type=Path)
    result.add_argument("--smplx-model-dir", type=Path)
    result.add_argument("--slots", type=Path, help="Existing compatible neutral correspondence; otherwise learn it.")
    result.add_argument("--archive-prefix", default=None, help="Default: rgb_174114_sim_handoff/")
    result.add_argument("--seat-centers", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                        help="Approximate seat XY centers after Y-up to Z-up conversion.")
    result.add_argument("--overwrite", action="store_true", help="Regenerate this tool's existing stage outputs.")
    result.add_argument("--dry-run", action="store_true", help="Print commands without creating or changing files.")
    return result


def workspace_settings(args):
    work = args.work_dir.expanduser().resolve()
    if work == REPO or work in REPO.parents or (work.is_relative_to(REPO) and
                                                not work.is_relative_to(REPO / "output")):
        raise ValueError("Workspaces inside the repository must be under output/.")
    path = work / "baseline_settings.json"
    if path.exists():
        settings = json.loads(path.read_text())
        for key in ("input_zip", "robot_xml", "smplx_model_dir", "slots"):
            value = getattr(args, key)
            if value is not None and str(value.expanduser().resolve()) != settings[key]:
                raise ValueError(f"{key} changed; use a new workspace.")
        if args.archive_prefix is not None and args.archive_prefix != settings["archive_prefix"]:
            raise ValueError("archive_prefix changed; use a new workspace.")
        if args.seat_centers is not None and args.seat_centers != settings["seat_centers"]:
            raise ValueError("seat_centers changed; use a new workspace.")
        return work, settings, False
    if args.stage not in ("all", "prepare"):
        raise ValueError("Run --stage prepare with the three asset paths first.")
    if work.exists() and any(work.iterdir()):
        raise ValueError("Refusing to initialize a nonempty workspace.")
    settings = {}
    for key in ("input_zip", "robot_xml", "smplx_model_dir"):
        value = getattr(args, key)
        if value is None:
            raise ValueError(f"--{key.replace('_', '-')} is required for a new workspace.")
        value = value.expanduser().resolve()
        if (key == "smplx_model_dir" and not value.is_dir()) or (
                key != "smplx_model_dir" and not value.is_file()):
            raise FileNotFoundError(value)
        settings[key] = str(value)
    slots = args.slots.expanduser().resolve() if args.slots else work / "correspondence/correspondence_slots_final.npz"
    if args.slots and not slots.is_file():
        raise FileNotFoundError(slots)
    settings.update(
        slots=str(slots),
        learn_correspondence=args.slots is None,
        archive_prefix=args.archive_prefix if args.archive_prefix is not None else "rgb_174114_sim_handoff/",
        seat_centers=args.seat_centers if args.seat_centers is not None else [-1.12, 1.16, .76, 1.22],
    )
    return work, settings, True


def stage_commands(stage, work, settings):
    py = [sys.executable, "-u"]
    script = lambda name: py + [str(TOOLS / name)]
    pipeline = py + [str(REPO / "scripts/humanoid_retarget_pipeline.py")]
    if stage == "prepare":
        return [script("prepare.py")]
    if stage == "correspondence":
        if not settings["learn_correspondence"]:
            return []
        return [pipeline + ["--config", str(work / "configs/person_01.json"),
                            "--stage", step, "--skip-view"] for step in ("build", "train")]
    if stage == "retarget":
        return [pipeline + ["--config", str(work / f"configs/person_{i:02d}_refined.json"),
                            "--stage", "retarget", "--force-retarget", "--skip-view"] for i in (1, 2)]
    if stage == "scene":
        return [script("render_preview.py") + ["--refined", "--scene-only"]]
    if stage == "ground":
        return [script("ground_contacts.py"),
                script("render_preview.py") + ["--grounded", "--scene-only"]]
    if stage == "smooth":
        return [script("smooth_motion.py")]
    if stage == "render":
        return [script("render_preview.py") + ["--grounded"], script("render_smooth.py")]
    raise ValueError(stage)


def stage_outputs(stage, work, settings):
    paths = {
        "prepare": ["preparation.json", "configs", "inputs", "objects", "source"],
        "correspondence": ["correspondence"],
        "retarget": [f"results/person_{i:02d}_refined.npz" for i in (1, 2)],
        "scene": ["preview_refined"],
        "ground": ["ground_contact_projection.json", "ground_contact_final_lifts.json",
                   "preview_grounded"] + [f"results/person_{i:02d}_grounded.npz" for i in (1, 2)],
        "smooth": ["smoothing_v1"],
        "render": ["preview_grounded/dual_vita02a_seat_exchange.mp4", "smoothing_v1/preview"],
    }
    return [work / p for p in paths[stage]]


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        work, settings, initialize = workspace_settings(args)
        stages = STAGES if args.stage == "all" else (args.stage,)
        for stage in stages:
            if stage == "correspondence" and not settings["learn_correspondence"]:
                continue
            existing = [str(path) for path in stage_outputs(stage, work, settings) if path.exists()]
            if existing and not args.overwrite:
                raise ValueError(f"{stage} already has outputs: {existing}. Use --overwrite or a new workspace.")
    except (ValueError, OSError, KeyError) as exc:
        p.error(str(exc))
    if not args.dry_run:
        work.mkdir(parents=True, exist_ok=True)
        if initialize:
            (work / "baseline_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    env = os.environ.copy()
    env["UMR_BASELINE_WORKDIR"] = str(work)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(key, "2")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    print(f"Workspace: {work}", flush=True)
    for stage in stages:
        commands = stage_commands(stage, work, settings)
        if not commands:
            print(f"[{stage}] Reusing explicitly supplied correspondence.", flush=True)
        for command in commands:
            print(f"[{stage}] {shlex.join(command)}", flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=REPO, env=env, check=True)


if __name__ == "__main__":
    main()
