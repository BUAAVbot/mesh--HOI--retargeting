#!/usr/bin/env python3
"""Create a PRIME Vita URDF/config with explicit sole contact frames."""

from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output-urdf", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--sole-z", type=float, default=-0.072)
    parser.add_argument("--contact-mode", choices=("center", "corners"), default="center")
    args = parser.parse_args()

    urdf_out = Path(args.output_urdf).resolve()
    urdf_out.parent.mkdir(parents=True, exist_ok=True)
    robot_tree = ET.parse(args.urdf)
    robot = robot_tree.getroot()
    contact_specs: list[tuple[str, str, tuple[float, float, float]]] = []
    for side in ("left", "right"):
        if args.contact_mode == "center":
            locations = (("contact", 0.0, 0.0, args.sole_z),)
        else:
            # Conservative outline of the complete shoe collision proxy:
            # back/front boxes span x=[-0.111098, 0.119], and the outer toe
            # cylinders span y=[-0.060, 0.060], with their bottom near -0.073.
            locations = (
                ("rear_left", -0.111098, 0.060, -0.073),
                ("rear_right", -0.111098, -0.060, -0.073),
                ("front_left", 0.119, 0.060, -0.073),
                ("front_right", 0.119, -0.060, -0.073),
            )
        for suffix, x, y, z in locations:
            link_name = f"{side}_sole_{suffix}"
            contact_specs.append((side, link_name, (x, y, z)))
            if robot.find(f"./link[@name='{link_name}']") is not None:
                continue
            ET.SubElement(robot, "link", {"name": link_name})
            joint = ET.SubElement(
                robot,
                "joint",
                {"name": f"{link_name}_joint", "type": "fixed"},
            )
            ET.SubElement(joint, "parent", {"link": f"{side}_ankle_roll_link"})
            ET.SubElement(joint, "child", {"link": link_name})
            ET.SubElement(
                joint,
                "origin",
                {"xyz": f"{x:.9g} {y:.9g} {z:.9g}", "rpy": "0 0 0"},
            )
    ET.indent(robot_tree, space="  ")
    robot_tree.write(urdf_out, encoding="utf-8", xml_declaration=True)

    config_out = Path(args.output_config).resolve()
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_tree = ET.parse(args.config)
    config = config_tree.getroot()
    robot_node = config.find("robot")
    data = config.find("data")
    solver = config.find("solver")
    weights = config.find("weights")
    outputs = config.find("outputs")
    contacts = config.find("contacts")
    if any(node is None for node in (robot_node, data, solver, weights, outputs, contacts)):
        raise ValueError("incomplete PRIME XML")
    assert robot_node is not None and data is not None and solver is not None
    assert weights is not None and outputs is not None and contacts is not None
    robot_node.set("urdf", str(urdf_out))
    for frame in list(contacts.findall("frame")):
        contacts.remove(frame)
    for _, link_name, _ in contact_specs:
        ET.SubElement(contacts, "frame", {"name": link_name})
    data.set("shift_base_to_ground", "true")
    data.set("reuse_initial_ground_height", "true")
    solver.set("horizon", str(args.horizon))
    solver.set("start_idx", str(args.start_idx))
    solver.set("max_iter", str(args.max_iter))
    solver.set("n_thread", "1")
    solver.set("dry_run", "false")
    weights.set("kappa", f"{args.kappa:.12g}")
    weights.set("mu", f"{args.mu:.12g}")
    weights.set("meas_linearVel", "0")
    outputs.set("directory", str(Path(args.output_directory).resolve()))
    ET.indent(config_tree, space="  ")
    config_tree.write(config_out, encoding="utf-8", xml_declaration=True)
    print(config_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
