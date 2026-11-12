"""Read-only Z-up scene/human support statistics; never execute handoff scripts."""
from pathlib import Path
import argparse,json,hashlib
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np
    U=np.asarray([[1.,0,0],[0,0,-1],[0,1,0]])
    scene=np.load(a.source/'scene/pi3_mesh.npz');human=np.load(a.source/'human/person_01/smplx_meshes_world.npz');root=np.load(a.source/'human/person_01/root_poses_world.npz')
    print('SCENE',[(k,scene[k].shape,str(scene[k].dtype))for k in scene.files],flush=True);print('HUMAN',[(k,human[k].shape,str(human[k].dtype))for k in human.files],flush=True);print('ROOT',[(k,root[k].shape)for k in root.files],flush=True)
    v=scene['vertices']@U.T;f=scene['faces'];tri=v[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)/2;normal=cross/(2*area[:,None]+1e-30);cen=tri.mean(1);horizontal=abs(normal[:,2])>np.cos(np.deg2rad(20))
    js=human['joints_world']@U.T;vs=human['vertices_world']@U.T
    bins=np.arange(np.nanpercentile(v[:,2],1),np.nanpercentile(v[:,2],99)+.01,.01);hist,edges=np.histogram(cen[horizontal,2],bins,weights=area[horizontal]);top=np.argsort(hist)[-30:][::-1]
    records=[]
    for i in[0,50,100,150,200,250,300,349]:
        records.append({'frame':i,'pelvis':js[i,0].tolist(),'left_ankle':js[i,7].tolist(),'right_ankle':js[i,8].tolist(),'left_foot':js[i,10].tolist(),'right_foot':js[i,11].tolist(),'mesh_min_z':float(vs[i,:,2].min())})
    report={'source':str(a.source),'U_yup_to_zup':U.tolist(),'scene_bounds_zup':[v.min(0).tolist(),v.max(0).tolist()],'scene_percentile_bounds_zup':[np.quantile(v,.01,axis=0).tolist(),np.quantile(v,.99,axis=0).tolist()],
            'human_all_bounds':[vs.reshape(-1,3).min(0).tolist(),vs.reshape(-1,3).max(0).tolist()],'human_root_xy_bounds':[js[:,0,:2].min(0).tolist(),js[:,0,:2].max(0).tolist()],
            'horizontal_triangle_fraction':float(horizontal.mean()),'horizontal_area_m2':float(area[horizontal].sum()),'height_area_modes':[{'z_center':float((edges[k]+edges[k+1])/2),'area_m2':float(hist[k])}for k in top],
            'sampled_human_poses':records}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
