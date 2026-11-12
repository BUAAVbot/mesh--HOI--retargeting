"""Small numerical checks for residual bounds, scales and saved reconstruction."""
import importlib.util
from pathlib import Path
import sys
import types
import numpy as np

# These basis tests do not instantiate or import GPU physics.
sys.modules['backend'] = types.SimpleNamespace(HdmiDynamics=None,sha=None)
spec=importlib.util.spec_from_file_location('g08_optimizer_math',Path(__file__).with_name('optimizer.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
names=('left_hip_pitch_joint','left_hip_roll_joint','left_hip_yaw_joint','left_knee_joint',
 'left_ankle_pitch_joint','left_ankle_roll_joint','right_hip_pitch_joint','right_hip_roll_joint',
 'right_hip_yaw_joint','right_knee_joint','right_ankle_pitch_joint','right_ankle_roll_joint',
 'waist_yaw_joint','waist_pitch_joint','left_shoulder_pitch_joint','left_shoulder_roll_joint',
 'left_shoulder_yaw_joint','left_elbow_joint','left_wrist_roll_joint','left_wrist_yaw_joint',
 'left_wrist_pitch_joint','right_shoulder_pitch_joint','right_shoulder_roll_joint',
 'right_shoulder_yaw_joint','right_elbow_joint','right_wrist_roll_joint','right_wrist_yaw_joint','right_wrist_pitch_joint')
params=np.full((1,7,28),.1)
out=module.dense_residual(params,289,names,40,240,'all')
assert out.shape==(1,288,28)
assert np.all(out[:,:41]==0) and np.all(out[:,240:]==0)
assert np.allclose(out[0,140,0],.1/.5,atol=1e-6)
assert np.allclose(out[0,140,14],.1/.4,atol=1e-6)
assert np.allclose(out[0,140,18],.1/.3,atol=1e-6)
leg=module.dense_residual(np.full((1,7,14),.1),289,names,-25,310,'legs')
assert np.isfinite(leg).all() and np.all(leg[:,:,14:]==0)
assert np.any(leg[:,0]!=0)
print('basis/scales/zero-support/negative-left checks passed')
