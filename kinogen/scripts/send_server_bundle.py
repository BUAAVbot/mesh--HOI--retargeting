"""Binary-safe SSH transport; verify archive and payload on the destination."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import PurePosixPath
import shlex
import subprocess
import time
from prepare_dependencies import ROOT, write_json, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["base", "weights", "metadata", "runtime", "wheels"])
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9), help="Parallel, resumable 256 MiB chunks when greater than 1")
    parser.add_argument("--compress", action="store_true", help="Use SSH compression for slow network links")
    parser.add_argument("--chunk-mib", type=int, default=256, choices=[8, 32, 128, 256])
    args = parser.parse_args()
    description = json.loads((ROOT / f"manifests/deployment/latest-{args.mode}.json").read_text(encoding="utf-8"))
    target = json.loads((ROOT / "configs/execution-targets.json").read_text(encoding="utf-8"))["server"]
    remote_root = target["project_root"]
    if not remote_root.startswith("/data/") or ".." in PurePosixPath(remote_root).parts:
        raise ValueError("Unexpected server project root")
    archive = ROOT / description["archive"]
    incoming = remote_root + "/.incoming/" + archive.name
    ssh = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", target["ssh_host"]]
    if args.compress:
        ssh.insert(1, "-C")
    if args.workers > 1:
        chunks_directory = incoming + ".chunks"
        subprocess.run(ssh + ["mkdir -p " + shlex.quote(chunks_directory)], check=True)
        chunk_size = args.chunk_mib * 2**20
        tasks = [(index, offset, min(chunk_size, archive.stat().st_size - offset)) for index, offset in enumerate(range(0, archive.stat().st_size, chunk_size))]

        def send_chunk_once(task):
            index, offset, size = task
            digest = hashlib.sha256()
            with archive.open("rb") as stream:
                stream.seek(offset)
                remaining = size
                while remaining:
                    block = stream.read(min(2**20, remaining))
                    digest.update(block)
                    remaining -= len(block)
            remote = chunks_directory + f"/{index:05d}"
            line = digest.hexdigest() + "  " + remote
            check = "printf '%s\\n' " + shlex.quote(line) + " | sha256sum -c - >/dev/null 2>&1"
            existing = subprocess.run(ssh + [check], capture_output=True)
            if existing.returncode == 0:
                return index
            temporary_check = "printf '%s\\n' " + shlex.quote(digest.hexdigest() + "  " + remote + ".tmp") + " | sha256sum -c - >/dev/null 2>&1"
            command = "cat > " + shlex.quote(remote + ".tmp") + " && " + temporary_check + " && mv " + shlex.quote(remote + ".tmp") + " " + shlex.quote(remote)
            process = subprocess.Popen(ssh + [command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                with archive.open("rb") as stream:
                    stream.seek(offset)
                    remaining = size
                    while remaining:
                        block = stream.read(min(2**20, remaining))
                        process.stdin.write(block)
                        remaining -= len(block)
                process.stdin.close()
                process.stdin = None
                output, _ = process.communicate()
                if process.returncode:
                    raise RuntimeError(output.decode("utf-8", errors="replace"))
            except BaseException:
                process.kill()
                process.wait()
                raise
            return index

        def send_chunk(task):
            for attempt in range(3):
                try:
                    return send_chunk_once(task)
                except (OSError, RuntimeError, subprocess.CalledProcessError):
                    if attempt == 2:
                        raise
                    print(f"Reconnecting chunk {task[0] + 1}; verified chunks retained", flush=True)
                    time.sleep(attempt + 2)

        print(f"Sending {len(tasks)} resumable chunks with {args.workers} SSH streams", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(send_chunk, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                index = future.result()
                print(f"Verified chunk {index + 1}; progress {count}/{len(tasks)}", flush=True)
        assemble = "cat " + shlex.quote(chunks_directory) + "/[0-9][0-9][0-9][0-9][0-9] > " + shlex.quote(incoming)
        subprocess.run(ssh + [assemble], check=True)
    checksum_line = description["sha256"] + "  " + incoming
    command = " && ".join([
        "mkdir -p " + shlex.quote(remote_root + "/.incoming"),
        *([] if args.workers > 1 else ["cat > " + shlex.quote(incoming)]),
        "printf '%s\\n' " + shlex.quote(checksum_line) + " | sha256sum -c -",
        "tar -xf " + shlex.quote(incoming) + " -C " + shlex.quote(remote_root),
        "cd " + shlex.quote(remote_root),
        "/usr/bin/python3.12 scripts/verify_server_payload.py " + shlex.quote(description["receipt"]),
    ])
    print(f"Sending {args.mode}: {archive.stat().st_size/2**20:.1f} MiB to {target['ssh_host']}:{remote_root}", flush=True)
    # The binary stdin file avoids PowerShell newline/encoding conversions.
    with archive.open("rb") as stream:
        result = subprocess.run(ssh + [command], stdin=stream if args.workers == 1 else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout.decode("utf-8", errors="replace"), flush=True)
    if result.returncode:
        return result.returncode
    write_json(ROOT / f"manifests/deployment/server-{args.mode}-verified.json", {**description, "server": target["ssh_host"], "root": remote_root, "verified_at_utc": now()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
