# Vita02A on HDMI

Scripts for turning UMR kinematic retargets into HDMI tracking tasks with static scene boxes, training one
policy per clip, and exporting physically consistent trajectories with per-contact forces.

Run everything from the repository root with the HDMI Python environment. Isaac Lab needs
`OMNI_KIT_ACCEPT_EULA=YES`; offline logging uses `wandb.mode=offline`.

## Reference motions

`data/motion/vita/` is not versioned; regenerate it from the UMR outputs:

```bash
UMR=/path/to/UMR/output

# single-person stairs + bench (rgb_145317)
python scripts/vita/umr_to_motion.py \
    --umr_npz $UMR/rgb_145317_vita02a/results/person_01_smoothed.npz \
    --scene_json $UMR/rgb_145317_vita02a/terrain_proxies.json \
    --xy_offset 0.07670424932457766 0.48793397792438165 \
    --out_dir data/motion/vita/stair_rgb_145317
# contact repair: pinned stance feet on the matching tier, legs re-solved by IK
(cd scripts/vita && python fix_contacts.py \
    --in_dir ../../data/motion/vita/stair_rgb_145317 --out_dir ../../data/motion/vita/stair_rgb_145317_fixed)

# two-person seat exchange (one motion dir per person, shared frame)
for p in 01 02; do
  python scripts/vita/umr_to_motion.py \
      --umr_npz $UMR/baseline_verification_20261028/smoothing_v1/results/person_${p}_smoothed.npz \
      --scene_json $UMR/baseline_verification_20261028/seats.json \
      --xy_offset -0.2233 0.825 --z_lift 0.008 \
      --out_dir data/motion/vita/seatswap_p${p}
done
```

`scripts/vita/check_motion_contacts.py DIR...` reports penetration, stance support and stance-foot sliding of a
motion dir against its boxes.

## Train, export, render

```bash
# train then export the final checkpoint to $EXPORTS/<name>
scripts/vita/train_and_export.sh stair_fixed_v1 Vita/stair_rgb_145317_fixed 0 "" stair_fixed_v1_final ""

# UMR-style video: reference vs physics, with contact forces
python scripts/vita/render_umr_style.py --preset stair --rollout $EXPORTS/stair_fixed_v1_final \
    --kinematic_ref --mode compare --forces --out stair_compare.mp4
```

`train_and_export.sh` and the render presets contain absolute paths of the original workstation; adjust them
before use elsewhere.

Exported `rollout.npz` (see `rollout_export.py`): root/joint/body states at the physics rate, per-body
per-surface contact force matrices, and individual contact points with normals, normal forces and friction
forces.
