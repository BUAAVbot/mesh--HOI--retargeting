"""Collect strict replay evidence and render the Gallery08 window comparison."""
import os
os.environ['MUJOCO_GL']='osmesa'
from pathlib import Path
import json
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from PIL import Image,ImageDraw

from common import write_json,sha256


def main():
    root=Path(sys.argv[1]).resolve()
    window=root/'window_0_80'
    spec=json.loads((window/'experiment.json').read_text())
    variants={
        'PD baseline':window/'baseline_replay',
        'DynaRetarget first':window/'sbto_replay_1',
        'DynaRetarget refined':root/'sbto_refine_replay_1',
        'DSMS reference seed':window/'dsms_replay_1',
        'DSMS PD seed':root/'dsms_physical_seed_replay_1',
        'DSMS step nodes':root/'dsms_step_nodes_replay_1'}
    if (root/'sbto_self_guard_replay_1/report.json').exists():
        variants['DynaRetarget self guard']=root/'sbto_self_guard_replay_1'
    rows=[]
    for name,folder in variants.items():
        report=json.loads((folder/'report.json').read_text())
        assert report['reference_sha256']==spec['reference_sha256']
        assert report['model_sha256']==spec['model_sha256']
        assert sha256(folder/'physical_record.npz')==report['physical_record_sha256']
        second=folder.with_name(folder.name[:-1]+'2') if folder.name.endswith('1') else None
        repeated=None
        if second:
            other=json.loads((second/'report.json').read_text())
            assert sha256(second/'physical_record.npz')==other['physical_record_sha256']
            repeated=report['physical_record_sha256']==other['physical_record_sha256']
            assert repeated
        rows.append(dict(variant=name,passed=report['passed'],metrics=report['metrics'],
            failed_gates=[k for k,v in report['checks'].items() if not v],
            repeated_record_hash_equal=repeated,report=str(folder/'report.json'),
            physical_record_sha256=report['physical_record_sha256'],
            dynamics_residual_max_abs=report['dynamics_residual_max_abs']))
        if name=='DynaRetarget self guard':
            exported=json.loads((root/'dynamic_window_self_guard.json').read_text())
            rows[-1].update(self_penetration_max_m=exported['self_penetration_max_m'],
                self_penetration_5mm_passed=exported['self_penetration_5mm_passed'])
    write_json(root/'summary.json',dict(experiment=spec,results=rows))
    # Show every method attempt in the numeric report; select a passing/best SBTO replay for video.
    sbto=min((r for r in rows if r['variant'].startswith('DynaRetarget')),
        key=lambda r:(len(r['failed_gates']),r['metrics']['root_rmse_m']))
    guarded=next((r for r in rows if r['variant']=='DynaRetarget self guard'),None)
    if guarded and guarded['passed'] and (root/'dynamic_window_self_guard.json').exists():
        guard=json.loads((root/'dynamic_window_self_guard.json').read_text())
        if guard['self_penetration_5mm_passed']:sbto=guarded
    selected=[rows[0],sbto,next(r for r in rows if r['variant']=='DSMS step nodes')]
    ref=np.load(window/'reference.npz',allow_pickle=False)
    records=[np.load(Path(r['report']).parent/'physical_record.npz',allow_pickle=False) for r in selected]
    time=np.arange(len(ref['qpos_ref']))*.02
    fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
    for row,rec in zip(selected,records):
        axes[0].plot(time,np.linalg.norm(rec['qpos'][:,:3]-ref['qpos_ref'][:,:3],axis=1),label=row['variant'])
        axes[1].plot(time,rec['qpos'][:,2],label=row['variant'])
        axes[2].plot(time[1:],rec['physics_penetration_m'].max(axis=1)*1000,label=row['variant'])
    axes[0].axhline(.25,color='k',linestyle='--',label='root max gate')
    axes[1].plot(time,ref['qpos_ref'][:,2],color='k',linestyle='--',label='kinematic reference')
    axes[2].axhline(12,color='k',linestyle='--',label='penetration gate')
    for ax,y in zip(axes,('Root error (m)','Root height (m)','Penetration (mm)')):
        ax.set_ylabel(y);ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Source time (s)');fig.tight_layout()
    fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    model=mj.MjModel.from_binary_path(spec['model'])
    renderer=mj.Renderer(model,height=360,width=480)
    data=mj.MjData(model)
    camera=mj.MjvCamera()
    camera.lookat[:]=ref['qpos_ref'][0,:3];camera.lookat[2]=.6
    camera.distance=3.2;camera.azimuth=135;camera.elevation=-15
    frames=[]
    for frame in range(len(time)):
        panels=[]
        for row,rec in zip(selected,records):
            data.qpos[:]=rec['qpos'][frame];data.qvel[:]=rec['qvel'][frame]
            mj.mj_forward(model,data);renderer.update_scene(data,camera=camera)
            panel=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(panel)
            draw.rectangle((0,0,480,40),fill=(15,20,28))
            draw.text((10,5),row['variant']+' | '+('PASS' if row['passed'] else 'FAIL'),fill='white')
            draw.text((10,22),f'Source frame {frame:02d} | {time[frame]:.2f}s | 0.5x speed',fill='white')
            panels.append(np.asarray(panel))
        frames.append(np.concatenate(panels,axis=1))
    renderer.close()
    imageio.mimsave(root/'comparison.mp4',frames,fps=25,macro_block_size=1)
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
