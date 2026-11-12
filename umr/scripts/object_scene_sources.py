"""Combine independent object trajectories for contact and collision constraints."""
import numpy as np


def combine_object_sources(sources):
    if not sources:
        raise ValueError('an object scene must contain at least one object')
    if len(sources) == 1:
        return sources[0]
    distances = np.stack([source['distances'] for source in sources], axis=2)
    closest = distances.argmin(axis=2)
    frames, slots = closest.shape
    frame_index = np.arange(frames)[:, None]
    slot_index = np.arange(slots)[None, :]
    offsets = np.cumsum([0] + [source['retarget_points_world'].shape[1] for source in sources[:-1]])
    ids = np.stack([source['object_ids'] + offset for source, offset in zip(sources, offsets)], axis=2)
    vectors = np.stack([source['pair_vectors'] for source in sources], axis=2)
    result = dict(sources[0])
    result.update(
        name='+'.join(source['name'] for source in sources),
        objects=sources,
        object_names=[source['name'] for source in sources],
        distances=distances[frame_index, slot_index, closest],
        object_ids=ids[frame_index, slot_index, closest],
        pair_vectors=vectors[frame_index, slot_index, closest],
        source_points_world=np.concatenate([source['source_points_world'] for source in sources], axis=1),
        retarget_points_world=np.concatenate([source['retarget_points_world'] for source in sources], axis=1),
        motion_positions=np.stack([source['motion_positions'] for source in sources], axis=1),
        motion_quats_wxyz=np.stack([source['motion_quats_wxyz'] for source in sources], axis=1),
    )
    return result
