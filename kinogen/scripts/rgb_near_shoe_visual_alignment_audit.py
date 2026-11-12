"""Read-only source-vs-compiled shoe alignment and hand proximity diagnostics."""
from pathlib import Path
import argparse,json,hashlib,xml.etree.ElementTree as ET
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    import numpy as np,mujoco as mj
    from scipy.spatial import cKDTree,ConvexHull
    from itertools import product
    m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);ref=np.load(a.reference,allow_pickle=False);x=ET.parse(a.model)
    vg=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'shoe_visual');sb=int(m.geom_bodyid[vg]);mid=int(m.geom_dataid[vg]);mn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_MESH,mid)
    meshfile=Path(x.find(f"asset/mesh[@name='{mn}']").get('file'));raw=[]
    for line in meshfile.read_text().splitlines():
        if line.startswith('v '):raw.append([float(z)for z in line.split()[1:4]])
    raw=np.asarray(raw)
    def mesh(g):
        mid=int(m.geom_dataid[g]);vs=int(m.mesh_vertadr[mid]);vn=int(m.mesh_vertnum[mid]);fs=int(m.mesh_faceadr[mid]);fn=int(m.mesh_facenum[mid])
        return m.mesh_vert[vs:vs+vn].copy(),m.mesh_face[fs:fs+fn].copy()
    vertices,faces=mesh(vg);meshR=np.empty(9);mj.mju_quat2Mat(meshR,m.geom_quat[vg]);meshR=meshR.reshape(3,3)
    compiled_body=vertices@meshR.T+m.geom_pos[vg]
    localerr=cKDTree(raw).query(compiled_body)[0]
    def bbox(v):return[v.min(axis=0).tolist(),v.max(axis=0).tolist()]
    rng=np.random.default_rng(260)
    def sample(v,f,n):
        tri=v[f];areas=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
        ids=rng.choice(len(f),n,p=areas/areas.sum());u=rng.random((n,2));s=np.sqrt(u[:,0]);w=np.c_[1-s,s*(1-u[:,1]),s*u[:,1]]
        return np.r_[v,np.einsum('ij,ijk->ik',w,tri[ids])]
    shoe_surface=sample(vertices,faces,60000)
    proxies=[g for g in range(m.ngeom)if m.geom_bodyid[g]==sb and m.geom_contype[g]]
    hand=[g for g in range(m.ngeom)if(mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))or'').startswith('p1_rh_')and m.geom_contype[g]]
    report={'model':str(a.model),'model_sha256':sha(a.model),'reference':str(a.reference),'reference_sha256':sha(a.reference),'raw_mesh':str(meshfile),'raw_mesh_sha256':sha(meshfile),
            'raw_OBJ_vertices_bbox_body_local_m':bbox(raw),'compiled_visual_vertices_bbox_body_local_m':bbox(compiled_body),'raw_vs_compiled_max_nearest_vertex_error_m':float(localerr.max()),
            'visual_compiled_geom_pos':m.geom_pos[vg].tolist(),'visual_compiled_geom_quat_wxyz':m.geom_quat[vg].tolist(),'frames':[]}
    # Snapshot the actual compiled geometry, not a re-created mesh transform.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    for row,frame in enumerate([300,397]):
        d.qpos[:59]=ref['qpos'][frame,0];d.qpos[59:118]=ref['qpos'][frame,1];d.qpos[118:]=ref['object_target_pose'][frame];mj.mj_fwdPosition(m,d)
        vv=vertices@d.geom_xmat[vg].reshape(3,3).T+d.geom_xpos[vg]
        surface=shoe_surface@d.geom_xmat[vg].reshape(3,3).T+d.geom_xpos[vg]
        direct=raw@d.xmat[sb].reshape(3,3).T+d.xpos[sb]
        worlderr=cKDTree(direct).query(vv)[0]
        hsurfaces=[];hulls=[];hrows=[]
        for g in hand:
            hv,hf=mesh(g);hs=sample(hv,hf,7000);hs=hs@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];hsurfaces.append(hs)
            hull=sample(hv,ConvexHull(hv).simplices,10000);hulls.append(hull@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])
            dist,ids=cKDTree(surface).query(hs);k=int(np.argmin(dist));hrows.append({'hand_geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),'min_raw_surface_sample_distance_m':float(dist[k]),'hand_witness_world':hs[k].tolist(),'raw_witness_world':surface[ids[k]].tolist()})
        hs=np.concatenate(hsurfaces);hh=np.concatenate(hulls);proxyrows=[];proxycorners=[]
        for g in proxies:
            R=d.geom_xmat[g].reshape(3,3);pos=d.geom_xpos[g];sz=m.geom_size[g];corn=np.asarray(list(product(*[[-a,a]for a in sz])))@R.T+pos;proxycorners.append(corn)
            local=(hs-pos)@R;delta=abs(local)-sz;outside=np.linalg.norm(np.maximum(delta,0),axis=1);inside=np.minimum(np.max(delta,axis=1),0);signed=outside+inside
            hl=(hh-pos)@R;hd=abs(hl)-sz;hdist=np.linalg.norm(np.maximum(hd,0),axis=1)+np.minimum(np.max(hd,axis=1),0)
            queries=[]
            for handgeom in hand:
                witness=np.zeros(6);dist=mj.mj_geomDistance(m,d,g,handgeom,.5,witness)
                # Known unresolved MuJoCo mesh GJK queries can report zero
                # with stale/nonzero witnesses; never count bare zero as contact.
                if dist==0:continue
                queries.append({'hand_geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,handgeom),'signed_distance_m':float(dist),'witness':witness.tolist()})
            proxyrows.append({'geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),'bbox_world_m':bbox(corn),'min_hand_mesh_surface_sample_signed_distance_m':float(signed.min()),
                              'min_hand_collision_hull_sample_signed_distance_m':float(hdist.min()),'nearest_valid_mujoco_hand_hull_query':min(queries,key=lambda z:z['signed_distance_m'])if queries else None})
        pc=np.concatenate(proxycorners);mind=min(hrows,key=lambda v:v['min_raw_surface_sample_distance_m'])
        rel=surface@d.xmat[sb].reshape(3,3)-d.xpos[sb]@d.xmat[sb].reshape(3,3)
        rawtoproxy=[]
        for g in proxies:
            local=(surface-d.geom_xpos[g])@d.geom_xmat[g].reshape(3,3);de=abs(local)-m.geom_size[g];rawtoproxy.append(np.linalg.norm(np.maximum(de,0),axis=1)+np.minimum(np.max(de,axis=1),0))
        near=np.min(np.abs(np.stack(rawtoproxy)),axis=0)
        rec={'frame':frame,'time_s':float(ref['time_s'][frame]),'visual_bbox_world_m':bbox(vv),'proxy_union_bbox_world_m':bbox(pc),'hand_collision_samples_bbox_world_m':bbox(hs),
             'direct_raw_body_transform_vs_compiled_world_max_nearest_vertex_error_m':float(worlderr.max()),'nearest_raw_hand_surface_sample':mind,'raw_hand_distances_by_link':hrows,'proxy_hand_distances':proxyrows,
             'raw_to_hand_collision_hull_sample_min_distance_m':float(cKDTree(surface).query(hh)[0].min()),
             'raw_surface_distance_to_nearest_proxy_surface_quantiles_m':np.quantile(near,[0,.05,.5,.95,1]).tolist()}
        report['frames'].append(rec)
        for col,axes in enumerate([(0,1),(1,2)]):
            ax=axs[row,col];i,j=axes;ax.scatter(surface[::30,i],surface[::30,j],s=1,c='black',alpha=.35,label='raw shoe surface');ax.scatter(hs[::40,i],hs[::40,j],s=1,c='tab:blue',alpha=.35,label='right hand surface')
            for corners in proxycorners:
                for aa in range(8):
                    for bb in range(aa+1,8):
                        if bin(aa^bb).count('1')==1:ax.plot(corners[[aa,bb],i],corners[[aa,bb],j],c='tab:orange',lw=.7)
            ax.scatter([mind['hand_witness_world'][i],mind['raw_witness_world'][i]],[mind['hand_witness_world'][j],mind['raw_witness_world'][j]],c='red',s=14)
            ax.set_title('frame '+str(frame)+' | '+'XYZ'[i]+'-'+'XYZ'[j]);ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
    a.output_dir.mkdir(parents=True,exist_ok=True);fig.savefig(a.output_dir/'shoe_raw_proxy_hand_alignment.jpg',dpi=150);plt.close(fig)
    report['interpretation']=['Compiled mesh recentering/rotation is checked by restoring original OBJ body-local vertices, then comparing direct body transform with compiled geom transform.',
                              'Surface proximity uses deterministic area samples plus original vertices; min surface sample distances are approximations, not signed triangle intersection distances.',
                              'Proxy proximity is checked against oriented box surfaces using both original hand mesh and actual collision convex-hull surface samples. Bare zero mj_geomDistance results are excluded as unresolved, not treated as contact.',
                              'Raw partial surface is not a complete shell and need not coincide with all inferred proxy walls. Approximate sample separation does not certify force-bearing grasp.']
    (a.output_dir/'shoe_visual_alignment_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'local_max_vertex_error_m':report['raw_vs_compiled_max_nearest_vertex_error_m'],'frames':[{k:v for k,v in f.items()if k not in['raw_hand_distances_by_link','proxy_hand_distances']}for f in report['frames']]}))
if __name__=='__main__':main()
