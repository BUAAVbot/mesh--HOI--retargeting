"""Small dependency/GPU/resume test; no actual GPU work is launched."""
from pathlib import Path
import sys
import tempfile

from dag_runner import run


ROOT = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix="batch_selftest_", dir=ROOT) as directory:
    tmp = Path(directory).resolve()
    if ROOT.resolve() not in tmp.parents:
        raise RuntimeError("temporary test directory escaped the workflow root")
    a, b = tmp / "a.txt", tmp / "b.txt"
    jobs = [
        {"id": "retarget_a", "argv": [sys.executable, "-c", f"from pathlib import Path; Path({str(a)!r}).write_text('a')"],
         "outputs": [str(a)], "gpus": 1},
        {"id": "expert_b", "argv": [sys.executable, "-c", f"from pathlib import Path; Path({str(b)!r}).write_text('b')"],
         "outputs": [str(b)], "needs": ["retarget_a"], "gpus": 8},
    ]
    first = run(jobs, list(range(8)), 2, tmp / "receipts", resume=False)
    assert all(row["status"] == "done" for row in first.values()), first
    second = run(jobs, list(range(8)), 2, tmp / "receipts", resume=True)
    assert all(row["status"] == "reused" for row in second.values()), second
print("BATCH_DAG_SELFTEST_OK")
