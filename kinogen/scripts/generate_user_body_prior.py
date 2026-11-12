"""Sample a genuine text-conditioned G1 prior; keep it separate from video evidence."""
from argparse import Namespace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform != "linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise RuntimeError("Run in the coordinated server kimodo-text environment on physical GPU7")
    import numpy as np
    import torch
    from scipy.interpolate import interp1d
    from scipy.spatial.transform import Rotation, Slerp
    from smoke_kimodo_text import run, digest

    torch.cuda.set_per_process_memory_fraction(.83)
    prompt = "A humanoid robot stands with both feet planted, reaches forward with both hands, grasps a box on a table, lifts it slightly, and holds it steadily at chest height."
    args = Namespace(runtime=ROOT / "configs/kimodo-text-runtime.json", prompt=[prompt],
                     frames=151, steps=50, seed=42, cfg_weight=2., preflight=False, encode_only=False)
    report = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "scope": "A sampled weak motion prior, not observed video motion or a fitted result",
              "source_video_sha256": "035c5852f9580cbbf5d612e2ad93a7f80065426a34ce355fb87db43876b20d6b",
              "physical_gpu": 7, "training": False, "api": False}
    out = ROOT / "outputs/user_reconstruction_v2"
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(args, report)
        from kimodo.skeleton import G1Skeleton34
        from kimodo.exports.mujoco import MujocoQposConverter
        model_file = next(x for x in report["motion_samples"][0]["files"] if x["path"].endswith(".npz"))
        with np.load(ROOT / model_file["path"], allow_pickle=False) as data:
            motion = {key: data[key] for key in data.files}
        converter = MujocoQposConverter(G1Skeleton34())
        qpos = np.asarray(converter.dict_to_qpos(motion, device="cpu", root_quat_w_first=False, mujoco_rest_zero=True)).reshape(151, 36)
        times = np.arange(121, dtype=np.float64) / 24
        prior_times = np.arange(151, dtype=np.float64) / 30
        root = interp1d(prior_times, qpos[:, :3], axis=0)(times)
        quat = Slerp(prior_times, Rotation.from_quat(qpos[:, 3:7]))(times).as_quat()
        dof = interp1d(prior_times, np.unwrap(qpos[:, 7:], axis=0), axis=0)(times)
        joint_names = np.asarray([x.replace("_skel", "_joint") for x in converter._mujoco_joint_including_root_list[1:]])
        joints = interp1d(prior_times, motion["posed_joints"], axis=0)(times)[..., [2, 0, 1]]
        path = out / "kimodo_prior.npz"
        np.savez_compressed(path, timestamps=times, joint_names=joint_names, joint_positions_rad=dof,
                            base_xyz_m=root, base_quaternion_xyzw=quat,
                            posed_joints_world_axes_m=joints, skeleton_joint_names=motion["joint_names"],
                            prior_is_observation=np.asarray(False), prompt=np.asarray(prompt), seed=np.asarray(42),
                            coordinate_system=np.asarray("URDF/MuJoCo X forward,Y left,Z up; model origin not video alignment"))
        video = next(x for x in report["motion_samples"][0]["files"] if x["path"].endswith(".mp4"))
        preview = ROOT / "outputs/previews/user_v2_kimodo_prior.mp4"
        shutil.copyfile(ROOT / video["path"], preview)
        report.update(status="genuine_weak_prior_sampled_not_video_fitted", target_times_frames=121,
                      coordinate_transform="[Xurdf,Yurdf,Zurdf]=[Zkimodo,Xkimodo,Ykimodo]; official MujocoQposConverter; rest_zero=True",
                      output={"path": str(path.relative_to(ROOT)), "sha256": digest(path)},
                      preview={"path": str(preview.relative_to(ROOT)), "sha256": digest(preview)},
                      caution="Prior may walk/lean or disagree with video; only use weakly if heldout image evidence improves. No prior angles are measured joints.")
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        (ROOT / "reports/user-body-prior.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k:v for k,v in report.items() if k in ("status", "output", "preview", "error")}), flush=True)


if __name__ == "__main__":
    main()
