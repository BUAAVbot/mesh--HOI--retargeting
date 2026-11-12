from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'scripts/run_fourth_single_plate_benchmark_v4b.py').read_text()
old=""" for name,value in template['finger_joint_targets'].items():
  j=m.joint(name).id;aid=m.actuator('motor_'+name).id
  if aid not in passive:ctrl[aid]=qstart[int(m.jnt_qposadr[j])]+s*(value-qstart[int(m.jnt_qposadr[j])])"""
new=""" close_fraction=float(np.clip((t-6.)/6.,0,1));close_fraction=close_fraction*close_fraction*(3-2*close_fraction)
 for name,value in template['finger_joint_targets'].items():
  j=m.joint(name).id;aid=m.actuator('motor_'+name).id
  if aid not in passive:ctrl[aid]=qstart[int(m.jnt_qposadr[j])]+close_fraction*(value-qstart[int(m.jnt_qposadr[j])])"""
assert old in text
text=text.replace(old,new)
text=text.replace("report.update(required_load_digit_policy=", "report.update(finger_motion='Hold separately declared open preshape throughout6s outside descent/insertion, then6s smooth closure toward final grasp; actual force controller applies only after insertion',initial_actual_finger_qpos=qstart[48:59].tolist(),required_load_digit_policy=")
target=ROOT/'scripts/run_fourth_single_plate_benchmark_v5b.py'
target.write_text(text)
print(target)
