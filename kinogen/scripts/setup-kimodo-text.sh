#!/usr/bin/env bash
# Offline, project-local compatibility overlay. Never writes to baseline envs.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
BASE="$ROOT/.venvs/kimodo"
TARGET="$ROOT/.venvs/kimodo-text"
WHEELS="$ROOT/downloads/kimodo-text-wheels"
UV="${KINOGEN_UV:-/root/.local/bin/uv}"
if [[ ! -x "$BASE/bin/python" || ! -x "$UV" || ! -f "$WHEELS/receipt.json" ]]; then
  printf 'Need existing project Kimodo environment, uv, and verified wheel bundle.\n' >&2
  exit 2
fi
export KINOGEN_CUDA_DEVICES="${KINOGEN_CUDA_DEVICES:-7}"
if [[ "$KINOGEN_CUDA_DEVICES" != 6 && "$KINOGEN_CUDA_DEVICES" != 7 ]]; then
  printf 'Select physical GPU 6 or 7 only.\n' >&2
  exit 2
fi
export UV_CACHE_DIR="$ROOT/.cache/uv" UV_OFFLINE=true
export KINOGEN_TEXT_WHEELS="$WHEELS"
"$BASE/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
directory = Path(os.environ["KINOGEN_TEXT_WHEELS"])
receipt = json.loads((directory / "receipt.json").read_text())
expected = {"transformers": "4.51.3", "peft": "0.15.2", "tokenizers": "0.21.4", "huggingface-hub": "0.36.0"}
if {row["name"]: row["version"] for row in receipt} != expected or len(receipt) != 4:
    raise RuntimeError("Unexpected compatibility wheel manifest")
for row in receipt:
    if Path(row["file"]).name != row["file"]:
        raise RuntimeError("Wheel filename must be a basename")
    path = directory / row["file"]
    if path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise RuntimeError(f"Wheel integrity mismatch: {path}")
print("All four official-PyPI wheel hashes verified")
PY
if [[ ! -x "$TARGET/bin/python" ]]; then
  "$UV" venv --python "$BASE/bin/python" --system-site-packages "$TARGET"
fi
PYTHON_VERSION="$("$BASE/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SITE="$TARGET/lib/python$PYTHON_VERSION/site-packages"
if [[ ! -d "$SITE" ]]; then
  printf 'Unexpected target environment layout: %s\n' "$SITE" >&2
  exit 2
fi
# Standard venv creation does not inherit packages from an outer venv.
# This adds the project's baseline as a read-only fallback after local packages.
printf '%s\n' "$BASE/lib/python$PYTHON_VERSION/site-packages" > "$SITE/kinogen-kimodo-baseline.pth"
"$UV" pip install --python "$TARGET/bin/python" --no-index --no-deps \
  --find-links "$WHEELS" --require-hashes --requirement "$ROOT/requirements-kimodo-text.lock.txt"
source "$ROOT/scripts/activate-server.sh" kimodo-text
python scripts/smoke_kimodo_text.py --preflight
python - <<'PY'
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
root = Path.cwd()
value = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "python": sys.executable, "base_prefix": sys.base_prefix,
    "packages": {key: importlib.metadata.version(key) for key in ("torch", "transformers", "peft", "tokenizers", "huggingface-hub")},
    "read_only_project_baseline": str(root / ".venvs/kimodo"),
    "wheel_receipt": "downloads/kimodo-text-wheels/receipt.json",
    "overlay_lock": "requirements-kimodo-text.lock.txt",
    "weights_loaded": False, "training_executed": False,
    "note": "Four-package compatibility overlay; not a complete independent dependency lock or official KinoGen environment",
}
(root / "reports/kimodo-text-environment.json").write_text(json.dumps(value, indent=2) + "\n")
print(json.dumps(value, indent=2))
PY
