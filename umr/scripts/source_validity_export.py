"""Preserve explicit canonical source validity on the selected frame timeline."""
from pathlib import Path
import numpy as np


def selected_source_validity(source, frame_ids):
    path = Path(source)
    if path.suffix.lower() != '.npz':
        return None
    with np.load(path, allow_pickle=False) as data:
        if 'source_valid' not in data:
            return None
        valid = data['source_valid']
    ids = np.asarray(frame_ids)
    if valid.ndim != 1 or valid.dtype != np.bool_:
        raise ValueError('source_valid must be a one-dimensional boolean mask')
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('frame_ids must be one-dimensional integer indices')
    if len(ids) and (ids.min() < 0 or ids.max() >= len(valid)):
        raise ValueError('frame_ids outside source validity timeline')
    return valid[ids].copy()
