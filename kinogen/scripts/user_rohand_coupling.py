#!/usr/bin/env python3
"""Read-only adapter for the official ROHand nonlinear six-actuator linkage.

Each hand has 25 URDF motions but only six independent controls. Five controls
are slider displacements in metres; thumb root is an angle in radians.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SIDES = ("left", "right")
ACTUATORS = ("th_slider_link", "if_slider_link", "mf_slider_link", "rf_slider_link", "lf_slider_link", "th_root_link")
GROUPS = (
    ("th_proximal_link", "th_slider_link", "th_connecting_link", "th_distal_link"),
    ("if_slider_link", "if_slider_abpart_link", "if_proximal_link", "if_distal_link", "if_connecting_link"),
    ("mf_slider_link", "mf_slider_abpart_link", "mf_proximal_link", "mf_distal_link", "mf_connecting_link"),
    ("rf_slider_link", "rf_slider_abpart_link", "rf_proximal_link", "rf_distal_link", "rf_connecting_link"),
    ("lf_slider_link", "lf_slider_abpart_link", "lf_proximal_link", "lf_distal_link", "lf_connecting_link"),
    ("th_root_link",),
)


class ROHandCoupling:
    def __init__(self, project_root=ROOT):
        self.project_root = Path(project_root)
        self.math_path = self.project_root / "dependencies/rohand-urdf/scripts/FingerMathURDF.py"
        specification = importlib.util.spec_from_file_location("kinogen_official_rohand_math", self.math_path)
        self.math = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(self.math)
        self.actuator_names, self.actuator_units = [], []
        self.lower, self.upper = [], []
        self.joint_names, self.joint_units = [], []
        self.side_joint_names = {}
        for side in SIDES:
            xml = ET.parse(self.project_root / f"dependencies/rohand-urdf/urdf/rohand_{side}.urdf").getroot()
            joints = {j.attrib["name"]: j for j in xml.findall("joint")}
            prefix = side + "_hand_ro_"
            for name in ACTUATORS:
                joint = joints[name]
                limits = joint.find("limit")
                self.actuator_names.append(prefix + name)
                self.actuator_units.append("m" if joint.attrib["type"] == "prismatic" else "rad")
                self.lower.append(float(limits.attrib["lower"]))
                self.upper.append(float(limits.attrib["upper"]))
            self.side_joint_names[side] = [prefix + name for name in joints]
            for name, joint in joints.items():
                self.joint_names.append(prefix + name)
                self.joint_units.append("m" if joint.attrib["type"] == "prismatic" else "rad")
        if len(self.joint_names) != 50:
            raise ValueError(f"Expected 50 coupled URDF motions, got {len(self.joint_names)}")

    def expand_side(self, positions, side, check_bounds=True):
        if side not in SIDES or len(positions) != 6:
            raise ValueError("Expected six actuator positions and side left/right")
        offset = 0 if side == "left" else 6
        state = {}
        prefix = side + "_hand_ro_"
        for finger, (position, names) in enumerate(zip(positions, GROUPS)):
            position = float(position)
            if not math.isfinite(position):
                raise ValueError("Non-finite ROHand actuator")
            if check_bounds and not self.lower[offset+finger]-1e-12 <= position <= self.upper[offset+finger]+1e-12:
                raise ValueError(f"Actuator {self.actuator_names[offset+finger]} out of URDF bounds: {position}")
            angles = self.math.HAND_FingerPosToAngle(finger, position)
            if finger == 0:
                values = (angles[0], position, angles[1], angles[2])
            elif finger == 5:
                values = (position,)
            else:
                values = (position, *angles)
            if len(values) != len(names) or not all(math.isfinite(x) for x in values):
                raise ValueError("Official coupling returned invalid linkage state")
            state.update({prefix+name: value for name, value in zip(names, values)})
        return state

    def expand_one(self, positions, check_bounds=True):
        if len(positions) != 12:
            raise ValueError("Expected actuator positions [12]")
        return {**self.expand_side(positions[:6], "left", check_bounds),
                **self.expand_side(positions[6:], "right", check_bounds)}

    def expand_array(self, positions):
        import numpy as np
        values = np.asarray(positions, dtype=float)
        if values.ndim != 2 or values.shape[1] != 12:
            raise ValueError("Expected actuator trajectory [T,12]")
        return np.asarray([[state[name] for name in self.joint_names]
                           for state in (self.expand_one(row) for row in values)])

    def domain_review(self, samples=1001):
        rows = []
        for control in range(6):
            valid, invalid = [], []
            minimum, maximum = self.lower[control], self.upper[control]
            for index in range(samples):
                position = minimum + (maximum-minimum)*index/(samples-1)
                try:
                    result = self.math.HAND_FingerPosToAngle(control, position)
                    values = (result,) if control == 5 else result
                    if not all(math.isfinite(x) for x in values):
                        raise ValueError("Non-finite math output")
                    valid.append(position)
                except (ValueError, ZeroDivisionError, OverflowError) as exc:
                    invalid.append({"position":position,"error":str(exc)})
            rows.append({"actuator":ACTUATORS[control],"unit":self.actuator_units[control],
                         "urdf_limits":[minimum,maximum],"samples":samples,"valid_samples":len(valid),
                         "invalid_samples":invalid,"sampled_valid_extent":[min(valid),max(valid)] if valid else None,
                         "conservative_effective_limits":[minimum,maximum] if not invalid else None})
        return {"official_math_path":str(self.math_path.relative_to(self.project_root)),
                "official_math_sha256":hashlib.sha256(self.math_path.read_bytes()).hexdigest(),
                "snapshot_modified":False,"controls_per_side":6,"motions_per_side":25,
                "domain_sampling_is_not_a_formal_continuity_proof":True,"controls":rows,
                "actuator_names":self.actuator_names,"actuator_units":self.actuator_units,
                "lower":self.lower,"upper":self.upper,"joint_names":self.joint_names,"joint_units":self.joint_units}


if __name__ == "__main__":
    print(json.dumps(ROHandCoupling().domain_review(), indent=2))
