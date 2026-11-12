"""Compare raster interpolation with independently ray-tested mesh samples."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();r=a.experiment
    original=json.loads((r/'independent_mesh_ground_audit.json').read_text())
    path=r/'mesh_calibrated_scene_v1/heightfield.npz';z=np.load(path,allow_pickle=False)
    interp=RegularGridInterpolator((z['y'],z['x']),z['height_m'],bounds_error=False,fill_value=np.nan)
    rows=[]
    for frame in original['frames']:
        for foot in frame['feet']:
            for sample in foot['samples']:
                heights=[h['z'] for h in sample['triangle_hits'] if h['horizontal'] and -.05<=h['z']<=.5]
                if not heights:continue
                xy=np.asarray(sample['xy']);ix=int(np.rint((xy[0]-z['x'][0])/(z['x'][1]-z['x'][0])));iy=int(np.rint((xy[1]-z['y'][0])/(z['y'][1]-z['y'][0])))
                unknown=not(0<=ix<len(z['x']) and 0<=iy<len(z['y'])) or bool(z['unknown_mask'][np.clip(iy,0,len(z['y'])-1),np.clip(ix,0,len(z['x'])-1)])
                estimated=float(interp([[xy[1],xy[0]]])[0]);truth=max(heights)
                rows.append(dict(frame=frame['frame'],geom=sample['geom'],xy=xy.tolist(),
                                 exact_mesh_z_m=truth,heightfield_z_m=estimated,error_m=estimated-truth,unknown=unknown))
    errors=np.array([abs(x['error_m']) for x in rows if not x['unknown'] and np.isfinite(x['error_m'])])
    report=dict(schema='hdmi.heightfield.mesh_samples.v1',
        heightfield_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ray_audit_sha256=hashlib.sha256((r/'independent_mesh_ground_audit.json').read_bytes()).hexdigest(),
        total_ray_samples=len(rows),known_samples=len(errors),
        absolute_error_percentiles_m={str(q):float(np.percentile(errors,q)) for q in (0,50,90,95,100)},
        samples=rows,
        limitations='Finite sample audit; does not bound unsampled edges, steep faces or unknown regions. Heightfield collision triangulation still needs runtime QA.')
    target=r/'mesh_calibrated_scene_v1/geometry_error_audit.json';target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
