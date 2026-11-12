"""Local, dependency-light gate for the HDMI→mjlab handoff."""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent
SOURCE = ROOT / "projects/kinogen/outputs/retarget_compare_20261028"


def check() -> dict:
    xml = ET.parse(TASK / "assets/vita28_shoes.xml").getroot()
    names = [x.get("name") for x in xml.iter("joint")]
    assert len(names) == len(set(names)) == 28, "MJCF active joints must be 28 unique names"
    assert len(list(xml.iter("geom"))) == 47, "shoe collision count changed"
    assert len(list(xml.iter("freejoint"))) == 1, "one floating root required"
    assert len(list(xml.iter("inertial"))) == 55, "inertial link count changed"
    result = {"model": {"joint_count": len(names), "collision_count": 47, "floating_roots": 1}}
    result["projected_retargets"] = {}
    for method in ("umr", "omniretarget", "hoi_retarget"):
        for person in ("01", "02"):
            key = f"{method}_person_{person}"
            path = TASK / f"data/{key}_vita28.npz"
            with np.load(path, allow_pickle=False) as d:
                q = d["qpos"]
                j = [str(x) for x in d["robot_joint_names"]]
                fps = float(d["fps"])
                assert q.shape == (172, 35) and set(j) == set(names) and fps > 0
                assert np.isfinite(q).all()
                result["projected_retargets"][key] = {"frames": len(q), "qpos_width": q.shape[1], "fps": fps}
    video = SOURCE / "media/overview"
    result["existing_videos"] = {}
    for name in ("source", "umr", "omniretarget", "hoi_retarget"):
        path = video / f"{name}.mp4"
        assert path.is_file() and path.stat().st_size > 0, f"missing {name} stage video"
        result["existing_videos"][name] = str(path)
    result["mjlab_runtime_verified"] = False
    result["hdmi_roa_teacher_student_ported"] = False
    result["physical_expert_video_available"] = False
    return result


if __name__ == "__main__":
    receipt = check()
    dest = TASK / "preflight.json"
    dest.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
