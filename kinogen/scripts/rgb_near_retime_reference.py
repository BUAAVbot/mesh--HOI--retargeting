"""Explicitly map immutable source frames to a slower execution timeline.

No poses, contacts, source frame IDs, or raw source samples are resampled.
time_s is the execution clock for the existing controller; source_time_s and
the source-to-execution mapping retain the original video timeline exactly.
"""
from pathlib import Path
import argparse
import hashlib
import json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v26_geometry_repaired_full398.npz')
    p.add_argument('--output',required=True)
    p.add_argument('--scale',type=float,default=2.)
    p.add_argument('--window',type=float,nargs=2,metavar=('SOURCE_START','SOURCE_END'))
    p.add_argument('--ramp',type=float,default=.5)
    a=p.parse_args()
    import numpy as np
    if a.scale<1:raise ValueError('This explicitly authorized helper only slows execution.')
    inp=a.root/a.input;out=a.root/a.output
    if inp.resolve()==out.resolve()or out.exists():raise ValueError('Never overwrite an input or frozen reference.')
    z=np.load(inp,allow_pickle=False);payload={k:z[k]for k in z.files};src=np.asarray(z['time_s']).copy()
    if'source_to_execution_time_s'in z.files:raise ValueError('Input is already retimed; use the frozen source candidate.')
    if len(src)<2 or np.any(np.diff(src)<=0):raise ValueError('A strictly increasing source timeline is required.')
    if a.window:
        start,end=a.window
        if not src[0]<=start<end<=src[-1]or a.ramp<=0:raise ValueError('Invalid source slowdown window or ramp.')
        # Smooth local tempo changes, integrated on the source intervals.
        def smooth(x):x=np.clip(x,0,1);return x*x*(3-2*x)
        weight=smooth((src-(start-a.ramp))/a.ramp)*(1-smooth((src-end)/a.ramp))
        scale=1+(a.scale-1)*weight
        exe=np.r_[src[0],src[0]+np.cumsum(np.diff(src)*.5*(scale[:-1]+scale[1:]))]
    else:
        scale=np.full(len(src),a.scale);exe=src[0]+(src-src[0])*a.scale
    payload['source_time_s']=src;payload['raw_source_time_s']=src.copy();payload['execution_time_s']=exe;payload['time_s']=exe
    payload['source_to_execution_time_s']=np.column_stack([src,exe]);payload['execution_tempo_scale']=scale
    payload['execution_retiming_parent_sha256']=np.array(sha(inp));payload['execution_retiming_description']=np.array('Source398 frames retained one-to-one. All pose/contact samples unchanged. Controller time_s explicitly becomes execution clock; original video time is source_time_s.')
    changed=[]
    for name in('shoe_release_time_s','O6_grasp_start_time_s','O6_grasp_end_time_s'):
        if name in payload:
            source=float(payload[name]);payload['source_'+name]=payload[name].copy();payload[name]=np.array(np.interp(source,src,exe));changed.append({'field':name,'source_time_s':source,'execution_time_s':float(payload[name])})
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**payload)
    report={'schema':'rgb_near_explicit_execution_retiming_v1','input_sha256':sha(inp),'output_sha256':sha(out),'frames':len(src),'scale_requested':a.scale,'source_window_s':a.window,'ramp_source_seconds':a.ramp if a.window else None,'source_duration_s':float(src[-1]-src[0]),'execution_duration_s':float(exe[-1]-exe[0]),'all_qpos_samples_unchanged':bool(np.array_equal(payload['qpos'],z['qpos'])),'all_object_pose_samples_unchanged':bool(np.array_equal(payload['object_target_pose'],z['object_target_pose'])),'frame_ids_unchanged':bool(np.array_equal(payload['frame_ids'],z['frame_ids'])),'event_times':changed,'source_to_execution_time_s':payload['source_to_execution_time_s'].tolist(),'not_a_dynamics_success_claim':True}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));out.with_suffix('.generator.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:v for k,v in report.items()if k!='source_to_execution_time_s'}),flush=True)


if __name__=='__main__':main()
