"""Report the server GPU baseline without loading model checkpoints."""
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    import torch

    kernel_result = None
    if torch.cuda.is_available():
        kernel_result = (torch.tensor([1.0, 2.0], device="cuda").square().sum()).item()
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=True)
    packages = {}
    for name in ("torch", "torchvision", "numpy", "transformers", "timm", "pytorch3d", "nvdiffrast"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "environment_prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "visible_device_count": torch.cuda.device_count(),
        "cuda_kernel_smoke_result": kernel_result,
        "nvidia_smi": query.stdout.strip().splitlines(),
        "training_launched": False,
        "model_load_tested": False,
        "note": "GPU baseline reuses the existing GRAIL packages through system-site-packages; exact KinoGen runtime versions remain unknown.",
    }
    root = Path(__file__).resolve().parents[1]
    (root / "reports").mkdir(exist_ok=True)
    (root / "reports/server-runtime.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["cuda_available"] and kernel_result == 5.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
