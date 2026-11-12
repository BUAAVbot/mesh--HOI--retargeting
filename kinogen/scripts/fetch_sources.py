"""Download official public evidence; never execute or clone linked code."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "sources"
URLS = {
    "project.html": "https://kinogen-humanoid.github.io/",
    "paper.html": "https://arxiv.org/html/2610.20552v1",
    "paper.pdf": "https://arxiv.org/pdf/2610.20552v1",
    "github-repositories.json": "https://api.github.com/users/KinoGen-humanoid/repos?per_page=100&type=owner&page=1",
}


def main() -> int:
    SOURCES.mkdir(parents=True, exist_ok=True)
    entries = []
    failures = []
    for name, url in URLS.items():
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "KinoGen-reproduction-preparation/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
                final_url = response.url
                content_type = response.headers.get("Content-Type", "")
                next_page = response.headers.get("Link", "")
            if name.endswith(".pdf") and not body.startswith(b"%PDF-"):
                raise ValueError("Expected a PDF, received different content")
            entry = {
                "file": name, "url": url, "resolved_url": final_url,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                "content_type": content_type,
            }
            if name == "project.html":
                page = body.decode("utf-8")
                entry["code_coming_soon"] = bool(re.search(r"code\s+coming\s+soon", page, re.I))
                entry["github_links"] = sorted(set(re.findall(r'https://github\.com/[^\s"<>]+', page)))
            elif name == "github-repositories.json":
                repos = json.loads(body)
                if not isinstance(repos, list):
                    raise ValueError("Expected a GitHub repository list")
                entry["repositories"] = [repo["full_name"] for repo in repos]
                entry["repository_list_complete"] = 'rel="next"' not in next_page
            target = SOURCES / name
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(body)
            temporary.replace(target)
            entries.append(entry)
            print(f"Saved {name}: {len(body)} bytes")
        except Exception as exc:
            failures.append({"file": name, "url": url, "error": str(exc)})
            print(f"Failed {name}: {exc}", file=sys.stderr)
    manifest = {"files": entries, "failures": failures}
    # This manifest records only this invocation, so old snapshots cannot look fresh.
    (SOURCES / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Review sources/manifest.json for release evidence; no algorithm was installed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
