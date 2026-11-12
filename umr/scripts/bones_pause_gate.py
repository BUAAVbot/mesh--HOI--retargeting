"""Park newly dispatched BONES workers before heavy imports; preserve live solves."""
import os
from pathlib import Path
import signal


MARKER = Path(__file__).resolve().parents[2] / 'data_umr/work/BONES_PAUSED_FOR_OTHER_DATASETS.json'


def applies(argv):
    for index, arg in enumerate(argv[:-1]):
        if arg == '--config':
            return '/work/bones_seed_proportional_v1/configs/' in str(Path(argv[index + 1]).resolve())
    return False


def park_if_requested(argv):
    if applies(argv):
        while MARKER.exists():
            print(f'[BONES pause] pid={os.getpid()} parked before model loading; SIGCONT after marker removal to resume', flush=True)
            os.kill(os.getpid(), signal.SIGSTOP)
        from bones_shard_gate import wait_for_remote_if_reserved
        wait_for_remote_if_reserved(argv)
