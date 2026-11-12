"""Explicit, reversible execution-clock scaling with all source samples intact."""
from pathlib import Path
import argparse,json,hashlib


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scale',type=float,required=True);a=p.parse_args()
    import numpy as np
    if a.output.exists()or a.scale<1:raise ValueError('Choose a fresh output and scale >= 1')
    z=np.load(a.input,allow_pickle=False);out={k:z[k]for k in z.files}
    if not np.array_equal(out['time_s'],out['source_time_s']):raise ValueError('Use an original unretimed parent')
    source=out['source_time_s'];out['time_s']=source[0]+(source-source[0])*a.scale
    out['source_to_execution_time_s']=np.column_stack([source,out['time_s']]);out['execution_time_scale']=np.asarray(a.scale)
    out['retiming_parent_sha256']=np.asarray(hashlib.sha256(a.input.read_bytes()).hexdigest())
    np.savez_compressed(a.output,**out)
    report={'mode':'explicit_execution_retiming','frames':len(source),'source_last_pts_s':float(source[-1]),'execution_last_pts_s':float(out['time_s'][-1]),'scale':a.scale,'poses_unchanged':bool(np.array_equal(out['qpos'],z['qpos'])),'source_frame_ids_unchanged':bool(np.array_equal(out['frame_ids'],z['frame_ids'])),'input_sha256':str(out['retiming_parent_sha256']),'output_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest()}
    a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');a.output.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(report),flush=True)


if __name__=='__main__':main()
