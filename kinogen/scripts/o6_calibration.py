"""Read O6 six-channel kinematic calibration; this is not a torque calibration.

The source lookup domain is feedback_u8. Simulation can use a desired feedback
value; rounded SDK commands are estimates, not a verified hardware servo model.
JSON quadratic_runtime coupling takes precedence over URDF linear fallback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class O6Calibration:
    def __init__(self, path):
        self.path = Path(path)
        self.profile = json.loads(self.path.read_text(encoding='utf-8'))
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.profile.get('model') != 'O6' or self.profile.get('angle_unit') != 'rad':
            raise ValueError('Expected O6/radian calibration')
        if self.profile.get('curve_input_domain') != 'feedback_u8':
            raise ValueError('Unexpected lookup domain')
        self.commands = list(self.profile['command_names'])
        self.joints = dict(self.profile['joints'])
        self.active = [None]*6
        self.passive = []
        for name, record in self.joints.items():
            motor = int(record['motor_index'])
            if motor not in range(6):
                raise ValueError('Invalid motor index')
            for branch in ('angle_rad','increasing_rad','decreasing_rad'):
                values = np.asarray(record[branch], dtype=float)
                if values.shape != (256,) or not np.isfinite(values).all():
                    raise ValueError('Invalid feedback lookup: '+name+'/'+branch)
            if record['passive']:
                self.passive.append(name)
                if record.get('coupling_model') != 'quadratic_runtime':
                    raise ValueError('Unsupported passive coupling: '+name)
                c = np.asarray(record['coupling_coefficients'], dtype=float)
                if not np.isfinite(c).all() or np.any(np.abs(c[3:]) > 1e-12):
                    raise ValueError('Expected at most quadratic coupling')
            else:
                if self.active[motor] is not None:
                    raise ValueError('More than one active joint per command')
                self.active[motor] = name
        if len(self.commands) != 6 or len(self.joints) != 11 or len(self.passive) != 5 or any(n is None for n in self.active):
            raise ValueError('Expected six driven and five coupled O6 joints')

    @staticmethod
    def branch(direction):
        aliases = {'average':'angle_rad','increasing':'increasing_rad','decreasing':'decreasing_rad'}
        if direction not in aliases:
            raise ValueError('Direction must name the lookup-input branch: average/increasing/decreasing')
        return aliases[direction]

    def joint_angles_for_driver_angles(self, angles):
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (6,) or not np.isfinite(angles).all():
            raise ValueError('Six finite active-joint angles required')
        result = dict(zip(self.active, map(float, angles)))
        pending = set(self.passive)
        while pending:
            progress = False
            for name in list(pending):
                row = self.joints[name]
                source = row['source_joint']
                if source not in result:
                    continue
                result[name] = float(np.polynomial.polynomial.polyval(result[source], row['coupling_coefficients']))
                pending.remove(name)
                progress = True
            if not progress:
                raise ValueError('Passive coupling graph contains a cycle or missing source')
        return result

    def joint_angles_for_feedback(self, feedback_u8, direction='average'):
        value = np.asarray(feedback_u8, dtype=float)
        if value.shape != (6,) or not np.isfinite(value).all() or np.any((value < 0)|(value > 255)):
            raise ValueError('Six feedback values in [0,255] required')
        branch = self.branch(direction)
        angles = [np.interp(value[i], np.arange(256), self.joints[name][branch]) for i,name in enumerate(self.active)]
        return self.joint_angles_for_driver_angles(angles)

    def desired_feedback_for_driver_angles(self, angles, direction='average'):
        """Approximate inverse of the chosen angle lookup, preserving open=255."""
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (6,) or not np.isfinite(angles).all():
            raise ValueError('Six finite active-joint angles required')
        branch = self.branch(direction)
        out = []
        indices = np.arange(256)
        for value, name in zip(angles, self.active):
            curve = np.asarray(self.joints[name][branch])
            order = np.lexsort((-indices, curve))
            x, first = np.unique(curve[order], return_index=True)
            out.append(float(np.interp(value, x, indices[order][first])))
        return np.asarray(out)

    def project_joint_targets(self, targets):
        """Replace dependent target values using the six actual driver targets."""
        result = dict(targets)
        result.update(self.joint_angles_for_driver_angles([targets[n] for n in self.active]))
        return result

    def metadata(self):
        return {'model':'O6', 'path':str(self.path), 'sha256':self.sha256,
                'side':self.profile['side'], 'independent_commands':self.commands,
                'active_joints':self.active, 'passive_joints':self.passive,
                'curve_input_domain':'feedback_u8', 'coupling':'JSON quadratic_runtime',
                'calibration_scope':self.profile['calibration_scope'],
                'per_joint_status':{k:v['calibration_status'] for k,v in self.joints.items()},
                'reported_quality':self.profile['quality'],
                'torque_or_dynamics_calibration':False,
                'direction_semantics':'increasing/decreasing branch selected explicitly for feedback-input motion; average is available when direction is unknown'}
