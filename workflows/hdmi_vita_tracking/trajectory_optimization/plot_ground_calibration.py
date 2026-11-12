"""Scientific 2D evidence view; does not certify a physical trajectory."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();r=a.experiment
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.is_file():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    z=np.load(r/'mesh_calibrated_scene_v1/heightfield.npz',allow_pickle=False)
    scene=json.loads((r/'source/scene.json').read_text())
    original=json.loads((r/'independent_mesh_ground_audit.json').read_text())
    audit=json.loads((r/'mesh_calibrated_scene_v1/audit.json').read_text())
    xx,yy=np.meshgrid(z['x'],z['y']);height=z['height_m'];unknown=z['unknown_mask']
    legacy=np.zeros_like(height)
    for b in scene['boxes']:
        q=np.array(b['pos'][:2]);s=np.array(b['size'][:2])/2
        hit=(abs(xx-q[0])<=s[0])&(abs(yy-q[1])<=s[1])
        legacy[hit]=np.maximum(legacy[hit],b['pos'][2]+b['size'][2]/2)
    foot=z['foot_samples_xy'].reshape(289,6,4,2)
    paths=[foot[:,:3].mean(axis=(1,2)),foot[:,3:].mean(axis=(1,2))]
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.5),layout='constrained',sharex=True,sharey=True)
    extent=[z['x'][0],z['x'][-1],z['y'][0],z['y'][-1]]
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#dadada')
    im=axes[0].imshow(np.ma.masked_where(unknown,height),origin='lower',extent=extent,cmap=cmap,vmin=0,vmax=.22,interpolation='none')
    fig.colorbar(im,ax=axes[0],label='原 mesh 地面 / 台面高度（米）',shrink=.8)
    cmap2=plt.get_cmap('RdBu_r').copy();cmap2.set_bad('#dadada')
    diff=legacy-height
    im2=axes[1].imshow(np.ma.masked_where(unknown,diff),origin='lower',extent=extent,cmap=cmap2,norm=TwoSlopeNorm(vmin=-.04,vcenter=0,vmax=.12),interpolation='none')
    fig.colorbar(im2,ax=axes[1],label='旧碰撞代理高度 − 原 mesh 高度（米）',shrink=.8)
    for axis in axes:
        for i,b in enumerate(scene['boxes']):
            x,y=b['pos'][:2];sx,sy=b['size'][:2]
            axis.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,fill=False,edgecolor='black',lw=1.25,linestyle='--',label='旧 0.35m 碰撞方块' if i==0 else None))
            axis.text(x,y,b['name'].replace('terrain_',''),ha='center',va='center',fontsize=8,color='black',bbox=dict(facecolor='white',alpha=.7,pad=1,edgecolor='none'))
        for path,color,label in zip(paths,('#fb5b57','#07c3d5'),('左脚轨迹','右脚轨迹')):
            axis.plot(path[:,0],path[:,1],color=color,lw=1.7,label=label)
            axis.scatter(path[0,0],path[0,1],s=35,color=color,marker='o',edgecolor='black',zorder=6)
            axis.scatter(path[-1,0],path[-1,1],s=55,color=color,marker='^',edgecolor='black',zorder=6)
        axis.set_aspect('equal');axis.set_xlabel('当前 HDMI 场景 X（米）');axis.set_ylabel('当前 HDMI 场景 Y（米）')
        axis.grid(alpha=.2);axis.legend(loc='lower right',fontsize=8,framealpha=.95)
    axes[0].set_title('真实三角面栅格：保留局部凸台边界')
    axes[1].set_title('红色区域：旧方块将本来的低地面抬高')
    fig.suptitle('③ 场景接触几何校准证据：原 mesh 与旧碰撞代理',fontsize=15)
    fig.text(.5,-.015,'灰色为原 mesh 未覆盖区域，未凭空重建；轨迹为原参考足印，非物理优化成功结果。',ha='center',fontsize=9)
    target=r/'mesh_calibrated_scene_v1/ground_calibration_comparison.png';fig.savefig(target,dpi=170,bbox_inches='tight');plt.close(fig)
    summaries=[dict(name=b['name'],**b['grid_summary']) for b in original['box_center_mesh']]
    report=dict(schema='hdmi.ground_calibration.figure.v1',figure=str(target),
        quantity='Observed mesh triangle height and original collision-proxy height difference',
        original_box_grid_audit=summaries,observed_fraction=audit['observed_grid_fraction'],
        source_foot_observed_or_small_hole_fraction=audit['foot_observed_or_small_hole_fraction'],
        physics_validated=False)
    (target.with_suffix('.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(target)


if __name__=='__main__':main()
