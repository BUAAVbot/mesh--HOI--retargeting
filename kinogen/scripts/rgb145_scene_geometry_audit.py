"""Inspect reconstructed stair planes and source foot support, read-only."""
from pathlib import Path
import argparse,json,hashlib
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    import numpy as np
    from scipy.signal import find_peaks
    from scipy.spatial import cKDTree,ConvexHull
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    U=np.asarray([[1.,0,0],[0,0,-1],[0,1,0]]);src=np.load(a.source/'scene/pi3_mesh.npz');hu=np.load(a.source/'human/person_01/smplx_meshes_world.npz')
    v=src['vertices']@U.T;f=src['faces'];j=hu['joints_world']@U.T;hv=hu['vertices_world']@U.T;tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)/2;normal=cross/(2*area[:,None]+1e-30);cen=tri.mean(1)
    roi=(cen[:,0]>-1.3)&(cen[:,0]<1.3)&(cen[:,1]>-.6)&(cen[:,1]<3.2)&(cen[:,2]>-.12)&(cen[:,2]<1.7)
    horizontal=roi&(abs(normal[:,2])>np.cos(np.deg2rad(15)))&(area>1e-8)
    bins=np.arange(-.12,1.71,.008);hist,edges=np.histogram(cen[horizontal,2],bins,weights=area[horizontal]);peaks,_=find_peaks(hist,distance=12,prominence=.025);order=peaks[np.argsort(hist[peaks])[::-1]];planes=[]
    for k in order:
        z=float((edges[k]+edges[k+1])/2);mask=horizontal&(abs(cen[:,2]-z)<.035);pts=cen[mask];weights=area[mask]
        if len(pts)<20:continue
        X=np.c_[pts[:,:2],np.ones(len(pts))];beta=np.linalg.lstsq(X*np.sqrt(weights[:,None]),pts[:,2]*np.sqrt(weights),rcond=None)[0]
        for _ in range(4):
            error=pts[:,2]-X@beta;keep=abs(error)<max(.008,3*np.median(abs(error)))
            beta=np.linalg.lstsq(X[keep]*np.sqrt(weights[keep,None]),pts[keep,2]*np.sqrt(weights[keep]),rcond=None)[0]
        error=pts[:,2]-X@beta;keep=abs(error)<.02;xy=pts[keep,:2];hull=ConvexHull(xy)
        planes.append({'mode_z_m':z,'z_eq_ax_by_c':beta.tolist(),'slope_deg':float(np.degrees(np.arctan(np.linalg.norm(beta[:2])))),'samples':int(len(pts)),'triangle_area_m2':float(weights.sum()),'median_abs_residual_m':float(np.median(abs(error))),'p95_abs_residual_m':float(np.quantile(abs(error),.95)),
                       'xy_bounds_m':[xy.min(0).tolist(),xy.max(0).tolist()],'xy_convex_hull':xy[hull.vertices].tolist(),'note':'Per-height hull is a diagnostic only; not used as a single scene convex collider.'})
    planes.sort(key=lambda p:p['mode_z_m'])
    inds=np.flatnonzero(horizontal);ht=tri[inds];tree=cKDTree(ht[:,:,:2].mean(1));supports=[]
    for frame in range(len(hv)):
        for side,ankle in [('left',7),('right',8)]:
            centre=j[frame,ankle];foot=hv[frame][(np.linalg.norm(hv[frame,:,:2]-centre[:2],axis=1)<.16)&(hv[frame,:,2]<centre[2]+.02)&(hv[frame,:,2]>centre[2]-.16)]
            if not len(foot):continue
            low=foot[foot[:,2]<np.quantile(foot[:,2],.1)];pos=low.mean(0);_,which=tree.query(pos[:2],k=min(400,len(ht)));t=ht[which];A=t[:,0,:2];B=t[:,1,:2];C=t[:,2,:2];AB=B-A;AC=C-A;D=pos[:2]-A;det=AB[:,0]*AC[:,1]-AB[:,1]*AC[:,0]
            u=(D[:,0]*AC[:,1]-D[:,1]*AC[:,0])/(det+1e-30);w=(AB[:,0]*D[:,1]-AB[:,1]*D[:,0])/(det+1e-30);inside=(u>=-1e-6)&(w>=-1e-6)&(u+w<=1+1e-6)&(abs(det)>1e-9)
            zz=t[:,0,2]+u*(t[:,1,2]-t[:,0,2])+w*(t[:,2,2]-t[:,0,2]);hits=zz[inside];height=float(hits.max())if len(hits)else None
            supports.append({'frame':frame,'side':side,'source_foot_lowpoint_world_m':pos.tolist(),'raw_horizontal_mesh_height_m':height,'signed_foot_gap_m':float(pos[2]-height)if height is not None else None,'ray_triangle_hit_count':int(len(hits))})
    fig,axs=plt.subplots(2,2,figsize=(15,11),constrained_layout=True)
    vv=v[(v[:,0]>-1.3)&(v[:,0]<1.3)&(v[:,1]>-.6)&(v[:,1]<3.2)&(v[:,2]<1.7)]
    im=axs[0,0].scatter(vv[::4,0],vv[::4,1],c=vv[::4,2],s=2,cmap='terrain',vmin=0,vmax=1.5);fig.colorbar(im,ax=axs[0,0],label='Z height [m]');axs[0,0].plot(j[:,0,0],j[:,0,1],c='black',lw=1,label='pelvis path');axs[0,0].legend();axs[0,0].set_title('Raw scene plan and human path (Z-up)');axs[0,0].set_aspect('equal')
    band=vv[(vv[:,0]>-.45)&(vv[:,0]<.55)];axs[0,1].scatter(band[::2,1],band[::2,2],s=1,c='gray',alpha=.3)
    for side,idx,color in [('left',10,'tab:blue'),('right',11,'tab:red')]:axs[0,1].plot(j[:,idx,1],j[:,idx,2],color=color,lw=1,label=side+' source foot joint')
    axs[0,1].set_title('Y-Z profile near actual trajectory');axs[0,1].set_aspect('equal');axs[0,1].legend()
    axs[1,0].plot((edges[:-1]+edges[1:])/2,hist);axs[1,0].set_title('Horizontal triangle area histogram, local ROI');axs[1,0].set_xlabel('height Z [m]');axs[1,0].set_ylabel('surface area [m2] per8mm bin')
    for side,color in [('left','tab:blue'),('right','tab:red')]:
        rows=[x for x in supports if x['side']==side and x['signed_foot_gap_m']is not None];axs[1,1].plot([x['frame']/30 for x in rows],[x['signed_foot_gap_m']for x in rows],c=color,label=side)
    axs[1,1].axhline(0,c='black');axs[1,1].set_title('Source foot low-surface minus raw horizontal mesh');axs[1,1].set_xlabel('time [s]');axs[1,1].set_ylabel('gap [m]');axs[1,1].legend()
    for ax in axs.ravel():ax.grid(alpha=.2)
    a.output_dir.mkdir(parents=True,exist_ok=True);fig.savefig(a.output_dir/'scene_support_geometry_audit.jpg',dpi=145);plt.close(fig)
    report={'source':str(a.source),'source_scene_sha256':hashlib.sha256((a.source/'scene/pi3_mesh.npz').read_bytes()).hexdigest(),'U_yup_to_zup':U.tolist(),'local_planes':planes,'foot_support_samples':supports,
            'limitations':['Scene and human are approximate metric reconstruction with missing/unobserved surfaces.','Only downward rays to observed near-horizontal triangles in the local ROI are used; missing hits are unknown, not proof of a hole or unsupported foot.','Surface footprints are not converted to a single convex hull collider.','Source foot lows are local SMPLX surface estimates; they do not establish robot sole contact.']}
    (a.output_dir/'scene_support_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'planes':planes,'ray_hits':sum(x['raw_horizontal_mesh_height_m']is not None for x in supports),'support_queries':len(supports)}))
if __name__=='__main__':main()
