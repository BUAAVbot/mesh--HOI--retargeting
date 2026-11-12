#!/usr/bin/env python3
"""Strict adapter from the shared Stage-2 contract to official PRIME.

PRIME is a Crocoddyl fork with a contact-identification extension.  Its
experiment executable is XML driven and is robot agnostic even though the
checked-in executable is named after G1.  This adapter only prepares inputs,
invokes that executable, and normalizes its outputs; it does not reimplement
PRIME or silently replace it with a trajectory smoother.

Aligator is deliberately *not* used here.  A future Aligator port would be a
separate custom backend because the official PRIME repository calls
``crocoddyl::SolverFDDP``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np


OFFICIAL_RUNNER_SOURCE = "experiments/G1_real_dance_1/src/g1_real_dance_1.cpp"
REQUIRED_TEMPLATE_FILES = (
    "cmake/base.cmake",
    "include/crocoddyl/contact_id/actions/contact-fwddyn-anitescu-id.hxx",
    "include/crocoddyl/contact_id/actions/arrival-fwddyn-id.hxx",
    "include/crocoddyl/contact_id/states/multibody_params.hxx",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _first(npz: np.lib.npyio.NpzFile, names: Iterable[str]) -> tuple[np.ndarray, str]:
    for name in names:
        if name in npz.files:
            return np.asarray(npz[name], dtype=np.float64), name
    raise ValueError(f"missing required array; expected one of {tuple(names)}")


def _finite_2d(name: str, value: np.ndarray) -> np.ndarray:
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError(f"{name} must be a finite 2-D trajectory with at least two rows")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return value


def inspect_repo(repo: Path) -> dict[str, object]:
    missing = [relative for relative in REQUIRED_TEMPLATE_FILES if not (repo / relative).is_file()]
    runner = repo / OFFICIAL_RUNNER_SOURCE
    if not runner.is_file():
        missing.append(OFFICIAL_RUNNER_SOURCE)
    runner_text = runner.read_text(encoding="utf-8", errors="replace") if runner.is_file() else ""
    return {
        "repo": str(repo.resolve()),
        "runner_source": str(runner.resolve()),
        "official_solver_fddp_present": "crocoddyl::SolverFDDP" in runner_text,
        "missing_required_files": missing,
        "ready_to_build": not missing and "crocoddyl::SolverFDDP" in runner_text,
        "aligator_backend": False,
        "aligator_note": "not part of official PRIME; any port must be reported as custom",
    }


def _convert_quaternion(q: np.ndarray, source_order: str) -> np.ndarray:
    result = np.array(q, copy=True)
    if result.shape[1] < 7:
        raise ValueError("q trajectory must contain floating-base xyz + quaternion")
    if source_order == "wxyz":
        # MuJoCo free-joint qpos is wxyz; Pinocchio free-flyer q is xyzw.
        quat = result[:, 3:7].copy()
        result[:, 3:7] = quat[:, [1, 2, 3, 0]]
    norm = np.linalg.norm(result[:, 3:7], axis=1)
    if np.any(norm < 1e-8):
        raise ValueError("q trajectory contains a zero base quaternion")
    result[:, 3:7] /= norm[:, None]
    return result


def _xml_attr(value: os.PathLike[str] | str) -> str:
    return escape(str(value), {'"': "&quot;"})


def prepare(args: argparse.Namespace) -> int:
    canonical = Path(args.canonical).resolve()
    urdf = Path(args.urdf).resolve()
    output = Path(args.output).resolve()
    if not canonical.is_file() or not urdf.is_file():
        raise FileNotFoundError("canonical NPZ and URDF must both exist")
    output.mkdir(parents=True, exist_ok=True)

    with np.load(canonical, allow_pickle=False) as data:
        q, q_key = _first(data, ("qpos_ref", "qpos", "q_ref"))
        v, v_key = _first(data, ("qvel_ref", "qvel", "v_ref"))
        # PRIME needs measured/generalized torque. Position targets are not a
        # valid substitute for the contact-ID dynamics problem.
        tau, tau_key = _first(data, ("tau_ref", "joint_torque_ref", "generalized_force_ref"))
        if "dt" in data.files:
            dt = float(np.asarray(data["dt"]).reshape(-1)[0])
        elif "fps" in data.files:
            dt = 1.0 / float(np.asarray(data["fps"]).reshape(-1)[0])
        else:
            dt = float(args.dt)

    q = _finite_2d(q_key, q)
    v = _finite_2d(v_key, v)
    tau = _finite_2d(tau_key, tau)
    if q.shape[1] != args.nq or v.shape[1] != args.nv:
        raise ValueError(
            f"dimension mismatch: got q={q.shape[1]}, v={v.shape[1]}; "
            f"expected nq={args.nq}, nv={args.nv}"
        )
    if tau.shape[1] == args.nv:
        if np.max(np.abs(tau[:, :6])) > 1e-8:
            raise ValueError("floating-base generalized-force columns must be zero")
        tau = tau[:, 6:]
    if tau.shape[1] != args.nv - 6:
        raise ValueError(f"PRIME torque input must have nv-6={args.nv - 6} columns")
    rows = min(q.shape[0], v.shape[0], tau.shape[0])
    if rows < 2:
        raise ValueError("not enough synchronized rows")
    q = _convert_quaternion(q[:rows], args.input_quaternion)
    v = v[:rows]
    tau = tau[:rows]
    if not (dt > 0.0 and np.isfinite(dt)):
        raise ValueError("dt must be positive and finite")

    timestamps = (np.arange(rows, dtype=np.float64) * dt)[:, None]
    q_csv = output / "q_ref_pinocchio.csv"
    v_csv = output / "v_ref_pinocchio.csv"
    u_csv = output / "tau_ref.csv"
    np.savetxt(q_csv, np.hstack((timestamps, q)), delimiter=",", fmt="%.12g")
    np.savetxt(v_csv, np.hstack((timestamps, v)), delimiter=",", fmt="%.12g")
    np.savetxt(u_csv, np.hstack((timestamps, tau)), delimiter=",", fmt="%.12g")

    contacts = [item.strip() for item in args.contacts.split(",") if item.strip()]
    if not contacts:
        raise ValueError("at least one URDF contact frame is required")
    contact_xml = "\n".join(f'    <frame name="{_xml_attr(name)}" />' for name in contacts)
    result_dir = output / "official_outputs"
    config = output / "prime_vita.xml"
    config.write_text(
        f"""<?xml version="1.0"?>
<config>
  <robot urdf="{_xml_attr(urdf)}" reference_configuration="standing" />
  <contacts>
{contact_xml}
  </contacts>
  <identification><link name="{_xml_attr(args.identify)}" /></identification>
  <data q="{_xml_attr(q_csv)}" v="{_xml_attr(v_csv)}" u="{_xml_attr(u_csv)}"
        q_has_time_column="true" v_has_time_column="true" u_has_time_column="true"
        normalize_base_quaternion="true" shift_base_to_ground="false"
        average_controls="true" />
  <solver horizon="{rows}" interval="{dt:.12g}" down_sample="1" start_idx="0"
          max_iter="{args.max_iter}" n_thread="{args.threads}" alpha0="1.0"
          callbacks="true" dry_run="{'true' if args.dry_run else 'false'}" />
  <weights arrival_alpha="40" arrival_com="40" arrival_diag="40"
        arrival_scale_alpha="10" meas_position="400" meas_position_z="200"
        meas_orientation="30" meas_joint="200" meas_linearVel="20"
        meas_angularVel="150" meas_jointVel="40" meas_params="0"
        dyn_position="20" dyn_orientation="20" dyn_joint="20" dyn_params="0"
        kappa="{args.kappa:.12g}" mu="{args.mu:.12g}" />
  <outputs directory="{_xml_attr(result_dir)}" xs_log="xs_log.csv"
        us_log="us_log.csv" xs_results="xs_results_fddp.csv"
        us_results="us_results_fddp.csv" u0_results="u0_results_fddp.csv"
        inertia_report="inertia_identification.txt" xs_rollout="xs_rollout.csv"
        force_log="f_rollout.csv" log_initial_guess="true" rollout="true" />
</config>
""",
        encoding="utf-8",
    )
    manifest = {
        "schema": "stage2-prime-input/v1",
        "method": "PRIME+Crocoddyl-FDDP",
        "official_backend": "PRIME contact-ID XML runner",
        "aligator_used": False,
        "canonical": {"path": str(canonical), "sha256": _sha256(canonical)},
        "urdf": {"path": str(urdf), "sha256": _sha256(urdf)},
        "source_keys": {"q": q_key, "v": v_key, "tau": tau_key},
        "dimensions": {"rows": rows, "nq": args.nq, "nv": args.nv, "tau": args.nv - 6},
        "dt": dt,
        "input_quaternion": args.input_quaternion,
        "prime_quaternion": "xyzw",
        "contacts": contacts,
        "identified_link_or_joint": args.identify,
        "mu": args.mu,
        "kappa": args.kappa,
        "config": str(config),
    }
    _json_dump(output / "input_manifest.json", manifest)
    print(config)
    return 0


def run(args: argparse.Namespace) -> int:
    executable = Path(args.executable).resolve()
    config = Path(args.config).resolve()
    if not executable.is_file() or not config.is_file():
        raise FileNotFoundError("PRIME executable and generated XML must exist")
    started = time.time()
    process = subprocess.run(
        [str(executable), str(config)], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=args.timeout, check=False,
    )
    log = Path(args.log).resolve()
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(process.stdout, encoding="utf-8")
    _json_dump(
        log.with_suffix(".run.json"),
        {
            "schema": "stage2-prime-run/v1",
            "command": [str(executable), str(config)],
            "returncode": process.returncode,
            "runtime_s": time.time() - started,
            "official_crocoddyl_runner": True,
            "aligator_used": False,
        },
    )
    if process.returncode != 0:
        print(process.stdout[-4000:], file=sys.stderr)
    return process.returncode


def _load_csv(path: Path, required: bool = True) -> np.ndarray | None:
    if not path.is_file() or path.stat().st_size == 0:
        if required:
            raise FileNotFoundError(f"missing PRIME output: {path}")
        return None
    value = np.loadtxt(path, delimiter=",", ndmin=2)
    return _finite_2d(path.name, value)


def collect(args: argparse.Namespace) -> int:
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    xs = _load_csv(source / "xs_results_fddp.csv")
    us = _load_csv(source / "us_results_fddp.csv")
    rollout = _load_csv(source / "xs_rollout.csv")
    forces = _load_csv(source / "f_rollout.csv")
    assert xs is not None and us is not None and rollout is not None and forces is not None

    parameter_width = 10 * args.identified_links
    expected_state = args.nq + args.nv + 2 * parameter_width
    if xs.shape[1] != expected_state or rollout.shape[1] != expected_state:
        raise ValueError(
            f"PRIME state width must be {expected_state} for {args.identified_links} "
            f"identified links; got xs={xs.shape[1]}, rollout={rollout.shape[1]}"
        )
    if us.shape[1] != args.nv:
        raise ValueError(f"PRIME control width must equal nv={args.nv}, got {us.shape[1]}")
    q_slice = slice(0, args.nq)
    v_slice = slice(args.nq + parameter_width, args.nq + parameter_width + args.nv)
    physical_x = np.hstack((xs[:, q_slice], xs[:, v_slice]))
    rollout_x = np.hstack((rollout[:, q_slice], rollout[:, v_slice]))
    common_rows = min(physical_x.shape[0], rollout_x.shape[0])
    dynamics_residual = rollout_x[:common_rows] - physical_x[:common_rows]

    # calc() may append forces during line-search evaluations. The final
    # ShootingProblem rollout happens last, so only the final trajectory tail
    # is a defensible lambda record.
    force_rows = min(us.shape[0], forces.shape[0])
    contact_lambda = forces[-force_rows:]
    if args.force_layout == "xyz":
        if contact_lambda.shape[1] % 3:
            raise ValueError("xyz force layout requires a multiple of three columns")
        vectors = contact_lambda.reshape(contact_lambda.shape[0], -1, 3)
        friction_residual = np.maximum(
            np.linalg.norm(vectors[..., :2], axis=-1) - args.mu * np.maximum(vectors[..., 2], 0.0),
            0.0,
        )
        normal_force_residual = np.maximum(-vectors[..., 2], 0.0)
    elif args.force_layout == "pyramid4":
        if contact_lambda.shape[1] % 4:
            raise ValueError("pyramid4 layout requires a multiple of four columns")
        # In the Anitescu friction-pyramid basis, cone feasibility reduces to
        # non-negative lambda coefficients.
        basis = contact_lambda.reshape(contact_lambda.shape[0], -1, 4)
        friction_residual = np.max(np.maximum(-basis, 0.0), axis=-1)
        normal_force_residual = np.zeros_like(friction_residual)
    else:
        raise ValueError("force layout must be declared; refusing to invent friction semantics")

    np.savez_compressed(
        output / "optimized_reference.npz",
        qpos_pinocchio=xs[:, q_slice],
        qvel_pinocchio=xs[:, v_slice],
        state_with_identified_parameters=xs,
    )
    np.savez_compressed(
        output / "controls.npz",
        generalized_force=us,
        joint_torque=us[:, 6:],
        contact_lambda=contact_lambda,
    )
    np.savez_compressed(
        output / "physics_certificate.npz",
        dynamics_state_residual=dynamics_residual,
        friction_cone_residual=friction_residual,
        negative_normal_force_residual=normal_force_residual,
    )
    max_dyn = float(np.max(np.abs(dynamics_residual)))
    max_friction = float(np.max(friction_residual))
    max_negative_normal = float(np.max(normal_force_residual))
    manifest = {
        "schema": "stage2-normalized-result/v1",
        "method": "PRIME+Crocoddyl-FDDP",
        "official_prime": True,
        "aligator_used": False,
        "status": "solver_outputs_normalized_not_yet_mujoco_validated",
        "source": str(source),
        "dimensions": {
            "states": list(xs.shape), "controls": list(us.shape),
            "contact_lambda": list(contact_lambda.shape),
        },
        "certificate": {
            "tau_available": True,
            "lambda_available": True,
            "dynamics_residual_available": True,
            "friction_residual_available": True,
            "force_layout": args.force_layout,
            "max_abs_euclidean_rollout_state_residual": max_dyn,
            "max_friction_cone_residual": max_friction,
            "max_negative_normal_force_residual": max_negative_normal,
            "note": "quaternion part of rollout residual is component-wise; MuJoCo replay remains required",
        },
        "files": {
            name: _sha256(output / name)
            for name in ("optimized_reference.npz", "controls.npz", "physics_certificate.npz")
        },
    }
    _json_dump(output / "manifest.json", manifest)
    print(output / "manifest.json")
    return 0


def preflight(args: argparse.Namespace) -> int:
    report = inspect_repo(Path(args.repo))
    if args.executable:
        executable = Path(args.executable)
        report["executable"] = str(executable.resolve())
        report["executable_present"] = executable.is_file()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_to_build"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--repo", required=True)
    p.add_argument("--executable")
    p.set_defaults(func=preflight)

    p = sub.add_parser("prepare")
    p.add_argument("--canonical", required=True)
    p.add_argument("--urdf", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--nq", type=int, default=35)
    p.add_argument("--nv", type=int, default=34)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--input-quaternion", choices=("wxyz", "xyzw"), default="wxyz")
    p.add_argument("--contacts", default="left_ankle_roll_link,right_ankle_roll_link")
    p.add_argument("--identify", default="base_link")
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--kappa", type=float, default=1000.0)
    p.add_argument("--mu", type=float, default=0.8)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=prepare)

    p = sub.add_parser("run")
    p.add_argument("--executable", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--timeout", type=float, default=86400.0)
    p.set_defaults(func=run)

    p = sub.add_parser("collect")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--nq", type=int, default=35)
    p.add_argument("--nv", type=int, default=34)
    p.add_argument("--identified-links", type=int, default=1)
    p.add_argument("--mu", type=float, default=0.8)
    p.add_argument("--force-layout", choices=("xyz", "pyramid4"), required=True)
    p.set_defaults(func=collect)
    return parser


def main() -> int:
    try:
        args = build_parser().parse_args()
        return int(args.func(args))
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"prime_adapter: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
