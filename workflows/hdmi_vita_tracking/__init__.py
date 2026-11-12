"""Register the Vita28 task when installed under mjlab/tasks/hdmi_vita28.

Set HDMI_VITA_MOTION and HDMI_VITA_SCENE to absolute paths before running mjlab.
"""
import os
from pathlib import Path

motion = os.environ.get("HDMI_VITA_MOTION")
scene = os.environ.get("HDMI_VITA_SCENE")
if motion and scene:
    if not Path(motion).is_file():
        raise FileNotFoundError(f"HDMI_VITA_MOTION does not exist: {motion}")
    if not Path(scene).is_file():
        raise FileNotFoundError(f"HDMI_VITA_SCENE does not exist: {scene}")
    from mjlab.tasks.registry import register_mjlab_task
    from .rl_cfg import make_runner_cfg
    from .task import make_env_cfg

    register_mjlab_task(
        task_id="Mjlab-Tracking-Flat-HDMI-Vita28",
        env_cfg=make_env_cfg(motion, scene),
        play_env_cfg=make_env_cfg(motion, scene, play=True),
        rl_cfg=make_runner_cfg(),
    )
