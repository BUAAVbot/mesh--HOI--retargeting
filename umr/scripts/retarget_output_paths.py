"""Filesystem-safe paths for retargeting sidecar artifacts."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


def _name_max(directory: Path) -> int:
    """Return the filesystem's maximum filename-component length."""
    try:
        return int(os.pathconf(directory, "PC_NAME_MAX"))
    except (OSError, ValueError):
        # NAME_MAX is 255 on the Linux filesystems used by the retargeting jobs.
        return 255


def floating_mjcf_path(output_path: Path | str, directory: Path | str | None = None) -> Path:
    """Choose a backward-compatible, filesystem-safe floating-MJCF path.

    Normal output names retain the legacy ``<stem>.floating_mjcf.xml`` form.
    When that component exceeds NAME_MAX, use a deterministic short name.  The
    hash includes the complete requested output path to prevent collisions when
    several long motion names share one sidecar directory.
    """
    output = Path(output_path)
    parent = Path(directory) if directory is not None else output.parent
    legacy_name = output.with_suffix(".floating_mjcf.xml").name
    if len(os.fsencode(legacy_name)) <= _name_max(parent):
        return parent / legacy_name

    digest = hashlib.sha256(os.fsencode(str(output))).hexdigest()[:20]
    return parent / f"umr_floating_{digest}.xml"
