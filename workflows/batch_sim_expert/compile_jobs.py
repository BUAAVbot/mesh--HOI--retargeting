"""Expand a SIM inventory and scenario recipes into one dependency job graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from dag_runner import read_jobs


STAGES = ("prepare", "gmr", "umr", "omniretarget", "hoi_retarget",
          "evaluate", "sonic")
NEEDS = {
    "prepare": (), "gmr": ("prepare",), "umr": ("prepare",),
    "omniretarget": ("prepare",), "hoi_retarget": ("prepare", "gmr"),
    "evaluate": ("umr", "omniretarget", "hoi_retarget"),
    "sonic": ("evaluate",),
}


def expand(spec: dict, values: dict[str, str]) -> dict:
    def one(value):
        if isinstance(value, str):
            return value.format_map(values)
        if isinstance(value, list):
            return [one(item) for item in value]
        if isinstance(value, dict):
            return {key: one(item) for key, item in value.items()}
        return value
    return one(spec)


def compile_inventory(inventory: dict, recipes: dict, output_root: Path) -> list[dict]:
    jobs = []
    ids = set()
    for case in inventory["cases"]:
        case_id = case["id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", case_id) or case_id in ids:
            raise ValueError(f"invalid or duplicate case id: {case_id}")
        ids.add(case_id)
        kind = case["kind"]
        recipe = recipes.get(kind)
        if recipe is None or set(recipe) != set(STAGES):
            raise ValueError(f"{kind}: recipe must define all seven case stages: {STAGES}")
        case_root = output_root / case_id
        values = {"case_id": case_id, "source": case["source"],
                  "case_output": str(case_root), "frames": str(case["frames"]),
                  "actors": str(case["actors"]), "kind": kind}
        for stage in STAGES:
            spec = expand(recipe[stage], values)
            if spec.get("enabled") is False:
                raise ValueError(f"{kind}/{stage}: unavailable adapter; cannot claim automatic pipeline")
            job = {"id": f"{case_id}__{stage}", "needs": [f"{case_id}__{x}" for x in NEEDS[stage]],
                   "argv": spec["argv"], "cwd": spec.get("cwd"),
                   "outputs": spec["outputs"], "inputs": spec.get("inputs", []),
                   "gpus": int(spec.get("gpus", 0)), "env": spec.get("env", {})}
            jobs.append(job)
    global_spec = recipes.get("_global_expert")
    if global_spec is None or global_spec.get("enabled") is False:
        raise ValueError("missing available _global_expert recipe for multi-case training")
    values = {"output_root": str(output_root), "case_count": str(len(ids))}
    spec = expand(global_spec, values)
    jobs.append({"id": "global_expert", "needs": [f"{case_id}__sonic" for case_id in sorted(ids)],
                 "argv": spec["argv"], "cwd": spec.get("cwd"),
                 "outputs": spec["outputs"], "inputs": spec.get("inputs", []),
                 "gpus": int(spec.get("gpus", 8)), "env": spec.get("env", {})})
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    recipes = json.loads(args.recipes.read_text(encoding="utf-8"))
    jobs = compile_inventory(inventory, recipes, args.output_root)
    args.jobs.parent.mkdir(parents=True, exist_ok=True)
    args.jobs.write_text(json.dumps({"schema": "sim_batch.jobs.v1", "jobs": jobs},
                                    ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    read_jobs(args.jobs)
    print(json.dumps({"cases": len(inventory["cases"]), "jobs": len(jobs),
                      "output": str(args.jobs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
