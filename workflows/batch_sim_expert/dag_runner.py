"""Resume-safe, eight-GPU DAG runner for SIM processing jobs.

Jobs are explicit argv arrays. This runner never infers a successful retarget,
SONIC result or RL expert from process exit alone: every job declares outputs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_jobs(path: Path) -> tuple[dict, list[dict]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    jobs = config.get("jobs", [])
    if not jobs:
        raise ValueError("manifest needs a nonempty jobs list")
    by_id = {j["id"]: j for j in jobs}
    if len(by_id) != len(jobs):
        raise ValueError("duplicate job id")
    for job in jobs:
        if not isinstance(job.get("argv"), list) or not job["argv"] or not all(
            isinstance(x, str) and x for x in job["argv"]
        ):
            raise ValueError(f"{job['id']}: argv must be a nonempty string list")
        if not isinstance(job.get("outputs"), list) or not job["outputs"]:
            raise ValueError(f"{job['id']}: declare at least one required output")
        if not set(job.get("needs", [])).issubset(by_id):
            raise ValueError(f"{job['id']}: unknown dependency")
        if job["id"] in job.get("needs", []):
            raise ValueError(f"{job['id']}: self dependency")
        if int(job.get("gpus", 0)) < 0:
            raise ValueError(f"{job['id']}: negative GPU count")
        if "CUDA_VISIBLE_DEVICES" in job.get("env", {}):
            raise ValueError(f"{job['id']}: CUDA_VISIBLE_DEVICES is controlled by the scheduler")
    remaining = set(by_id)
    resolved = set()
    while remaining:
        ready = {name for name in remaining if set(by_id[name].get("needs", [])) <= resolved}
        if not ready:
            raise ValueError("dependency cycle")
        remaining -= ready
        resolved |= ready
    return config, jobs


def fingerprint(job: dict) -> str:
    payload = {k: job.get(k) for k in ("argv", "cwd", "outputs", "needs", "gpus", "env")}
    for entry in job.get("inputs", []):
        path = Path(entry)
        if not path.is_file():
            raise FileNotFoundError(f"{job['id']}: missing input {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        payload.setdefault("input_sha256", {})[str(path)] = digest.hexdigest()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def outputs_ready(job: dict) -> bool:
    return all(Path(value).is_file() and Path(value).stat().st_size > 0
               for value in job["outputs"])


def execute(job: dict, gpu_ids: list[int], receipt_dir: Path, resume: bool) -> dict:
    name = job["id"]
    digest = fingerprint(job)
    receipt_path = receipt_dir / f"{name}.json"
    log_path = receipt_dir / f"{name}.log"
    if resume and receipt_path.is_file():
        old = json.loads(receipt_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") == digest and old.get("returncode") == 0 and outputs_ready(job):
            return {**old, "status": "reused"}
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    env.update({str(k): str(v) for k, v in job.get("env", {}).items()})
    began = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(job["argv"], cwd=job.get("cwd"), env=env,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    row = {"id": name, "status": "done" if result.returncode == 0 and outputs_ready(job) else "failed",
           "returncode": result.returncode, "wall_seconds": time.monotonic() - began,
           "gpu_ids": gpu_ids, "fingerprint": digest, "outputs": job["outputs"],
           "log": str(log_path)}
    receipt_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def run(jobs: list[dict], gpu_pool: list[int], workers: int,
        receipt_dir: Path, resume: bool) -> dict:
    if workers < 1 or len(set(gpu_pool)) != len(gpu_pool):
        raise ValueError("workers and GPU pool are invalid")
    if any(int(j.get("gpus", 0)) > len(gpu_pool) for j in jobs):
        raise ValueError("job requests more GPUs than the pool")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    pending = {j["id"]: j for j in jobs}
    done: dict[str, dict] = {}
    available = set(gpu_pool)
    active = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while pending or active:
            made_progress = False
            for name, job in list(pending.items()):
                needs = job.get("needs", [])
                if any(dep in done and done[dep]["status"] not in ("done", "reused") for dep in needs):
                    done[name] = {"id": name, "status": "blocked", "reason": "dependency failed"}
                    del pending[name]
                    made_progress = True
                    continue
                if len(active) >= workers or not all(dep in done for dep in needs):
                    continue
                count = int(job.get("gpus", 0))
                if count > len(available):
                    continue
                chosen = sorted(available)[:count]
                available.difference_update(chosen)
                future = pool.submit(execute, job, chosen, receipt_dir, resume)
                active[future] = (name, chosen)
                del pending[name]
                made_progress = True
            if not active and pending and not made_progress:
                raise RuntimeError("scheduler stalled")
            if active:
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    name, chosen = active.pop(future)
                    available.update(chosen)
                    try:
                        done[name] = future.result()
                    except Exception as error:
                        done[name] = {"id": name, "status": "failed", "error": repr(error)}
                    print(json.dumps(done[name], ensure_ascii=False), flush=True)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--run", action="store_true", help="execute jobs; default only validates/prints plan")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--receipts", type=Path)
    args = parser.parse_args()
    _, jobs = read_jobs(args.manifest)
    gpu_ids = [int(x) for x in args.gpus.split(",") if x]
    if not args.run:
        print(json.dumps({"jobs": [{"id": j["id"], "needs": j.get("needs", []),
                                     "gpus": j.get("gpus", 0), "outputs": j["outputs"]}
                                    for j in jobs], "gpu_pool": gpu_ids}, ensure_ascii=False, indent=2))
        return
    receipts = args.receipts or args.manifest.parent / "receipts"
    result = run(jobs, gpu_ids, args.workers, receipts, args.resume)
    failed = [name for name, row in result.items() if row["status"] not in ("done", "reused")]
    print(json.dumps({"finished": len(result), "failed_or_blocked": failed}, ensure_ascii=False))
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
