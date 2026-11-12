"""Bind scene assets, coverage, exact mesh samples, native rays and figure."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();r=a.experiment;d=r/'mesh_calibrated_scene_v1'
    files=('heightfield.npz','scene.json','audit.json','geometry_error_audit.json',
           'compiled_model.mjb','independent_native_heightfield_audit.json',
           'ground_calibration_comparison.png','ground_calibration_comparison.json')
    hashes={n:sha(d/n) for n in files}
    native=json.loads((d/'independent_native_heightfield_audit.json').read_text())
    mesh=json.loads((d/'geometry_error_audit.json').read_text())
    coverage=json.loads((d/'audit.json').read_text())
    checks=dict(native_heightfield_representation_verified=native['passed'] is True,
        native_bound_model=native['model_sha256']==hashes['compiled_model.mjb'],
        native_bound_heightfield=native['heightfield_sha256']==hashes['heightfield.npz'],
        mesh_sample_bound_heightfield=mesh['heightfield_sha256']==hashes['heightfield.npz'],
        coverage_bound_heightfield=coverage['heightfield_sha256']==hashes['heightfield.npz'],
        sampled_mesh_approximation_under_5mm=mesh['absolute_error_percentiles_m']['100']<.005)
    if not all(checks.values()):raise ValueError(f'Geometry binding failed: {checks}')
    report=dict(schema='hdmi.heightfield.audit_binding.v1',checks=checks,files_sha256=hashes,
        source_mesh_sha256=coverage['source_mesh_sha256'],source_motion_sha256=coverage['source_motion_sha256'],
        source_transform_sha256=coverage['transform_sha256'],
        independent_mesh_ground_audit_sha256=sha(r/'independent_mesh_ground_audit.json'),
        geometry_error_audit_sha256=hashes['geometry_error_audit.json'],
        native_ray_audit_sha256=hashes['independent_native_heightfield_audit.json'],
        native_ray_samples=native['ray_samples'],native_ray_max_error_m=native['ray_absolute_error_max_m'],
        mesh_sample_max_error_m=mesh['absolute_error_percentiles_m']['100'],
        foot_source_observed_or_small_hole_fraction=coverage['foot_observed_or_small_hole_fraction'],
        complete_ground_coverage=False,actual_physics_support_coverage_requires_acceptance=True,
        physics_validated=False,
        caption='Figure overlays original reference foot paths; it is not an optimized physical rollout.')
    target=d/'geometry_binding_manifest.json';target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
