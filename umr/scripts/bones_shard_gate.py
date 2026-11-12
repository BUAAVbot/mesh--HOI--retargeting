"""Lightweight ownership gate for the already-live original BONES dispatcher."""
import hashlib
import json
from pathlib import Path
import time

PLAN=Path(__file__).resolve().parents[2]/'data_umr/work/bones_two_machine_v1/plan.json'


def wait_for_remote_if_reserved(argv):
    if not PLAN.exists():return
    plan=json.loads(PLAN.read_text())
    def arg(name):return argv[argv.index(name)+1]
    source=Path(arg('--data')).resolve()
    key=source.relative_to(plan['source_root']).as_posix()
    if key<plan['remote_first_key']:return
    out=Path(arg('--out')).resolve()
    receipt=Path(plan['receipts'])/(hashlib.sha256(key.encode()).hexdigest()+'.json')
    print(f'[BONES shard] remote-owned {key}; waiting for verified output, not solving locally',flush=True)
    while True:
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r.get('source')!=str(source) or r.get('canonical_output')!=str(out):
                raise RuntimeError('remote receipt identity mismatch')
            if r['status']=='verified':
                if not out.is_file() or hashlib.sha256(out.read_bytes()).hexdigest()!=r['output_sha256']:
                    raise RuntimeError('remote canonical output missing or hash mismatch')
                print(f'[BONES shard] verified remote output {out}',flush=True)
                raise SystemExit(0)
            if r['status']=='failed':
                print(f'[BONES shard] remote failed, retained for full failure recovery: {r.get("error")}',flush=True)
                raise SystemExit(1)
        time.sleep(60)
