"""Experimental bounded-batch original LBS; never enabled in production by default."""
import torch


def lbs_frame_chunked(bind_shape, skinning_weights, target_pose_world, chunk_frames=8):
    from soma.geometry.lbs import lbs
    # Restrict the experiment to an explicit single batch axis; preserve the
    # original broadcast behavior for all other shapes.
    if target_pose_world.ndim != 4 or target_pose_world.shape[0] <= chunk_frames:
        return lbs(bind_shape, skinning_weights, target_pose_world)
    frames = target_pose_world.shape[0]
    if bind_shape.ndim not in (2, 3) or (bind_shape.ndim == 3 and bind_shape.shape[0] not in (1, frames)):
        return lbs(bind_shape, skinning_weights, target_pose_world)
    return torch.cat([
        lbs(bind_shape[start:start+chunk_frames] if bind_shape.ndim == 3 and bind_shape.shape[0] == frames else bind_shape,
            skinning_weights, target_pose_world[start:start+chunk_frames])
        for start in range(0, frames, chunk_frames)
    ], dim=0)
