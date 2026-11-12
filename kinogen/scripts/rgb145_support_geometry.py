"""Fit local observed terrace footprints, retaining separate tread/seat levels."""
from pathlib import Path
import argparse,json,hashlib
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np
    from scipy.spatial import ConvexHull,cKDTree
    from rgb145_support import height_at_xy,inside_convex_polygon
    src=np.load(a.source/'scene/pi3_mesh.npz');U=np.asarray([[1.,0,0],[0,0,-1],[0,1,0]]);v=src['vertices']@U.T;tri=v[src['faces']];cen=tri.mean(1);cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)/2;nz=abs(cross[:,2])/(2*area+1e-30)
    audit=json.loads(a.audit.read_text());roi=(cen[:,0]>-1.3)&(cen[:,0]<1.3)&(cen[:,1]>-.6)&(cen[:,1]<3.2);base=roi&(nz>np.cos(np.deg2rad(15)))&(area>1e-8)
    specs=[]
    for plane in audit['local_planes']:
        if plane['mode_z_m']<.1:ground={'name':'ground_support','z_eq_ax_by_c':plane['z_eq_ax_by_c'],'friction':[.9,.01,.001]};continue
        b=np.asarray(plane['z_eq_ax_by_c']);pred=cen[:,:2]@b[:2]+b[2];mask=base&(abs(cen[:,2]-plane['mode_z_m'])<.035)&(abs(cen[:,2]-pred)<.02)
        pts=tri[mask].reshape(-1,3);pts=pts[abs(pts[:,2]-(pts[:,:2]@b[:2]+b[2]))<.025];xy=pts[:,:2]
        # Clipping the local collision ROI is explicit, not a change to raw mesh.
        xy=np.clip(xy,[-1.3,-.6],[1.3,3.2]);h=ConvexHull(xy);poly=xy[h.vertices]
        is_seat=abs(plane['mode_z_m']-.436)<.06 or abs(plane['mode_z_m']-1.052)<.06
        name=('seat_support_'if is_seat else'tier_support_')+str(sum(x['kind']==('seat'if is_seat else'tread')for x in specs)).zfill(2)
        # Count observed near-horizontal coverage within this local convex patch.
        xx=np.arange(poly[:,0].min()+.02,poly[:,0].max(),.04);yy=np.arange(poly[:,1].min()+.02,poly[:,1].max(),.04);grid=np.stack(np.meshgrid(xx,yy),axis=-1).reshape(-1,2);grid=grid[inside_convex_polygon(grid,poly)]
        tt=tri[mask];tree=cKDTree(tt[:,:,:2].mean(1));hits=0
        for pt in grid:
            _,idx=tree.query(pt,k=min(96,len(tt)));t=tt[np.atleast_1d(idx)];A=t[:,0,:2];B=t[:,1,:2]-A;C=t[:,2,:2]-A;D=pt-A;det=B[:,0]*C[:,1]-B[:,1]*C[:,0];u=(D[:,0]*C[:,1]-D[:,1]*C[:,0])/(det+1e-30);w=(B[:,0]*D[:,1]-B[:,1]*D[:,0])/(det+1e-30)
            hits+=int(np.any((u>=0)&(w>=0)&(u+w<=1)&(abs(det)>1e-10)))
        specs.append({'name':name,'kind':'seat'if is_seat else'tread','top_z_eq_ax_by_c':b.tolist(),'xy_polygon_m':poly.tolist(),'bottom_z_m':-.08,
                      'friction':[.8 if is_seat else .9,.01,.001],'mode_height_m':plane['mode_z_m'],'top_surface_fit_median_error_m':plane['median_abs_residual_m'],
                      'top_surface_fit_p95_error_m':plane['p95_abs_residual_m'],'footprint_area_m2':float(h.volume),'observed_surface_grid_coverage_fraction':hits/max(1,len(grid)),
                      'coverage_grid_spacing_m':.04,'coverage_grid_samples':len(grid),'assumption':'Separate local solid terrace extrusion below observed top. Interior/occluded gaps completed; no entire-scene convex hull.'})
    spec={'schema':'rgb145.support_geometry.v1','source_scene':str(a.source/'scene/pi3_mesh.npz'),'source_scene_sha256':sha(a.source/'scene/pi3_mesh.npz'),'source_audit_sha256':sha(a.audit),'U_yup_to_zup':U.tolist(),
          'collision_roi_xy_m':[[-1.3,-.6],[1.3,3.2]],'ground':ground,'supports':specs,
          'limitations':['Metric scale is inherited from monocular reconstruction and is not measured.','Tread and seat collision footprints are individual local convex patches; no hull spans different step levels.',
                        'Raw full scene mesh remains a separate noncolliding visual layer. Collision ROI covers source actor trajectory with margin; distant railings/plants remain visual-only.',
                        'Each terrain patch is completed downward as static solid fill. Occluded interior geometry and friction are engineering assumptions, not measured CAD.']}
    rows=[x for x in audit['foot_support_samples']if x['raw_horizontal_mesh_height_m']is not None];xy=np.asarray([x['source_foot_lowpoint_world_m'][:2]for x in rows]);proxy,names=height_at_xy(spec,xy,True);raw=np.asarray([x['raw_horizontal_mesh_height_m']for x in rows]);err=proxy-raw
    spec['source_foot_ray_proxy_consistency']={'samples':len(rows),'median_abs_height_error_m':float(np.median(abs(err))),'p95_abs_height_error_m':float(np.quantile(abs(err),.95)),'max_abs_height_error_m':float(np.max(abs(err))),'samples_over_30mm':int(sum(abs(err)>.03))}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps({'output':str(a.output),'supports':[{k:s[k]for k in ['name','kind','mode_height_m','observed_surface_grid_coverage_fraction']}for s in specs],'foot_proxy_raw':spec['source_foot_ray_proxy_consistency']}))
if __name__=='__main__':main()
