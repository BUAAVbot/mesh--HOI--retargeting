"""Estimate table footprint from static scene evidence, never held trajectories."""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.spatial import ConvexHull,cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation


R=Path(__file__).resolve().parents[1]
D=R/'data/fourth_sim_handoff'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,default=R/'outputs/fourth_full_sequence/table_surface_audit')
    a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    scene=np.load(D/'scene/visual_mesh.npz',allow_pickle=False)
    points=scene['vertices'].astype(float);faces=scene['faces'].astype(int)
    plane=json.loads((D/'provenance/object_pose_qa.json').read_text())['table_plane']
    origin0=np.asarray(plane['origin_world']);n0=np.asarray(plane['normal_world'])
    origin=np.asarray([origin0[0],-origin0[2],origin0[1]])
    normal=np.asarray([n0[0],-n0[2],n0[1]]);normal/=np.linalg.norm(normal)
    raw=np.load(D/'scene/source_points_arkit.npz',allow_pickle=False)['points_world'].astype(float)
    raw=np.column_stack([raw[:,0],-raw[:,2],raw[:,1]])
    residual=points@normal-origin@normal
    raw_residual=raw@normal-origin@normal
    triangles=points[faces];cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    area=.5*np.linalg.norm(cross,axis=1);unit=cross/np.maximum(2*area[:,None],1e-15)
    plane_faces=(np.max(np.abs(residual[faces]),axis=1)<.018)&(np.abs(unit@normal)>.65)&(area>1e-8)
    candidate_vertices=np.unique(faces[plane_faces])
    near=points[candidate_vertices]
    # Mesh holes and view triangulation can disconnect tiny flat triangles;
    # cluster the measured plane vertices at the scene's 3cm support scale.
    pairs=cKDTree(near).query_pairs(.035,output_type='ndarray')
    adj=coo_matrix((np.ones(2*len(pairs)),(np.r_[pairs[:,0],pairs[:,1]],np.r_[pairs[:,1],pairs[:,0]])),shape=(len(near),len(near)))
    nc,labels=connected_components(adj,directed=False)
    ob=np.load(D/'objects/poses_sim.npz',allow_pickle=False);tl=np.load(D/'timeline.npz',allow_pickle=False)['pts_seconds']
    initial=ob['position_m'][0]
    components=[]
    for k in range(nc):
        selected=near[labels==k]
        if len(selected)<30:continue
        distance=cKDTree(selected[:,:2]).query(initial[:,:2])[0]
        components.append({'id':k,'vertices':len(selected),'distance_initial_objects_xy_m':distance.tolist(),
                           'score':int((distance<.15).sum())*100000+len(selected)})
    chosen=max(components,key=lambda c:c['score'])
    ids=candidate_vertices[labels==chosen['id']]
    patch=points[ids]
    hull=ConvexHull(patch[:,:2]);polygon=patch[hull.vertices,:2]
    hullEq=hull.equations
    def inside(xy,tolerance=0):return np.all(xy@hullEq[:,:2].T+hullEq[:,2]<=tolerance,axis=1)
    # Plane-face boundaries with a steep adjacent measured face are stronger
    # physical-edge evidence than holes/view-frustum boundaries alone.
    selected_set=np.zeros(len(points),bool);selected_set[ids]=True
    selected_faces=plane_faces&np.all(selected_set[faces],axis=1)
    edge_faces={}
    for fi,tri in enumerate(faces):
        for e in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])]:
            key=tuple(sorted(map(int,e)));edge_faces.setdefault(key,[]).append(fi)
    boundary=[];physical=[];visibility=[]
    for edge,neighbours in edge_faces.items():
        count=sum(bool(selected_faces[f])for f in neighbours)
        if count!=1:continue
        midpoint=points[list(edge)].mean(axis=0)
        boundary.append(midpoint)
        others=[f for f in neighbours if not selected_faces[f]]
        side=any(abs(float(unit[f]@normal))<.55 and float(residual[faces[f]].min())<-.015 for f in others)
        if side:physical.append(midpoint)
        elif len(neighbours)==1:visibility.append(midpoint)
    physical=np.asarray(physical).reshape(-1,3);boundary=np.asarray(boundary).reshape(-1,3);visibility=np.asarray(visibility).reshape(-1,3)
    below=raw[(raw_residual<-.015)&(raw_residual>-.15)]
    below_residual=below@normal-origin@normal
    cloud_side=[]
    if len(below):
        neighbours=cKDTree(below[:,:2]).query_ball_point(boundary[:,:2],.020)
        for point,ids_below in zip(boundary,neighbours):
            if len(ids_below)>=3 and float(below_residual[ids_below].min())<-.035 and float(np.ptp(below_residual[ids_below]))>.020:
                cloud_side.append(point)
    cloud_side=np.asarray(cloud_side).reshape(-1,3)
    reference=np.load(R/'outputs/fourth_full_sequence/reference_v6.npz',allow_pickle=False)
    hand_near=np.asarray(reference['contact_candidate_observed'],bool).any(axis=1)
    hand_observed=np.asarray(reference['hand_observed_mask'],bool).all(axis=1)
    source_qa=json.loads((D/'provenance/object_pose_qa.json').read_text())['objects']
    # Also expose measured off-plane points immediately below the right/front
    # outline. A true tabletop side continues below the top, unlike cropping.
    static=[];initial_support=[];static_footprints=[]
    for oi in range(6):
        v=np.load(D/f'objects/object_{oi+1:02d}/mesh.npz',allow_pickle=False)['vertices'].astype(float)
        valid=np.asarray(ob['valid'][:,oi],bool)
        q=ob['quaternion_wxyz'][:,oi]
        good=valid&np.isfinite(q).all(axis=1)&np.isfinite(ob['position_m'][:,oi]).all(axis=1)
        frames=np.flatnonzero(good)
        rotations=Rotation.from_quat(q[frames][:,[1,2,3,0]]).as_matrix()
        local_normals=rotations.transpose(0,2,1)@normal
        bottom=(v@local_normals.T).min(axis=0)+ob['position_m'][frames,oi]@normal-origin@normal
        velocity=np.full(len(tl),np.inf)
        for k in range(1,len(frames)):
            current,previous=int(frames[k]),int(frames[k-1]);dt=float(tl[current]-tl[previous])
            if dt<.07:
                speed=float(np.linalg.norm(ob['position_m'][current,oi]-ob['position_m'][previous,oi])/dt)
                velocity[current]=min(velocity[current],speed);velocity[previous]=min(velocity[previous],speed)
        support=(np.abs(bottom)<.02)&(velocity[frames]<.03)
        selected_frames=frames[support]
        unheld=selected_frames[(~hand_near[selected_frames,oi])&hand_observed[selected_frames]]
        action_start=int(source_qa[oi]['action_frame_range'][0])
        pre_action=selected_frames[selected_frames<action_start]
        centres=ob['position_m'][selected_frames,oi,:2]
        footprint=[]
        for frame in selected_frames[::max(1,len(selected_frames)//100)]:
            qq=ob['quaternion_wxyz'][frame,oi][[1,2,3,0]];RR=Rotation.from_quat(qq).as_matrix()
            world=v@RR.T+ob['position_m'][frame,oi]
            contact=world[(world@normal-origin@normal)<.01]
            if len(contact):footprint.append(contact[:,:2])
        footprint=np.vstack(footprint)if footprint else np.empty((0,2));static_footprints.append(footprint)
        world0=v@Rotation.from_quat(q[0][[1,2,3,0]]).as_matrix().T+ob['position_m'][0,oi]
        initial_bottom=float((world0@normal-origin@normal).min())
        bottom_support=world0[(world0@normal-origin@normal)<initial_bottom+.004,:2]
        initial_support.append({'object_id':oi+1,'initial_center_xy':initial[oi,:2].tolist(),
            'initial_lowest_point_plane_residual_m':initial_bottom,
            'initial_center_inside_visible_patch_hull':bool(inside(initial[oi:oi+1,:2],.01)[0]),
            'initial_bottom_support_vertices_inside_hull_fraction':float(inside(bottom_support,.01).mean()),
            'nearest_measured_top_vertex_xy_distance_m':float(cKDTree(patch[:,:2]).query(initial[oi,:2])[0])})
        static.append({'object_id':oi+1,'frames':selected_frames.tolist(),'count':len(selected_frames),
            'unheld_observed_frames':unheld.tolist(),'unheld_observed_count':len(unheld),
            'pre_action_static_frames':pre_action.tolist(),'pre_action_static_count':len(pre_action),
            'source_action_frame_range':source_qa[oi]['action_frame_range'],
            'unheld_static_center_xy_min':ob['position_m'][unheld,oi,:2].min(0).tolist()if len(unheld)else None,
            'unheld_static_center_xy_max':ob['position_m'][unheld,oi,:2].max(0).tolist()if len(unheld)else None,
            'condition':'valid source pose, lowest source-mesh signed plane distance abs<2cm and local translational speed<3cm/s',
            'center_xy_min':centres.min(0).tolist()if len(centres)else None,
            'center_xy_max':centres.max(0).tolist()if len(centres)else None,
            'contact_footprint_xy_min':footprint.min(0).tolist()if len(footprint)else None,
            'contact_footprint_xy_max':footprint.max(0).tolist()if len(footprint)else None})
    summary={'schema':'fourth.source_static_table_footprint_audit.v1',
        'source_mesh_sha256':sha(D/'scene/visual_mesh.npz'),'source_points_sha256':sha(D/'scene/source_points_arkit.npz'),
        'plane_origin_sim':origin.tolist(),'plane_normal_sim':normal.tolist(),
        'plane_vertex_residual_abs_percentiles_m':np.percentile(np.abs(residual),[1,5,10,25,50]).tolist(),
        'raw_points_within_18mm_of_table_plane':int((np.abs(raw_residual)<.018).sum()),
        'planar_mesh_face_count':int(plane_faces.sum()),'plane_components':components,
        'chosen_component':chosen,'observed_patch_xy_min':patch[:,:2].min(0).tolist(),'observed_patch_xy_max':patch[:,:2].max(0).tolist(),
        'observed_convex_hull_xy':polygon.tolist(),'convex_hull_area_m2':float(hull.volume),
        'boundary_edges_count':len(boundary),'physical_side_supported_boundary_count':len(physical),'pure_visibility_boundary_count':len(visibility),
        'physical_side_supported_boundary_xy':physical[:,:2].tolist(),
        'physical_side_supported_xy_min':physical[:,:2].min(0).tolist()if len(physical)else None,
        'physical_side_supported_xy_max':physical[:,:2].max(0).tolist()if len(physical)else None,
        'raw_cloud_vertical_side_boundary_count':len(cloud_side),'raw_cloud_vertical_side_boundary_xy':cloud_side[:,:2].tolist(),
        'raw_cloud_vertical_side_test':'At least 3 original points within 2cm XY of planar boundary, depths 1.5-15cm below plane, minimum depth>3.5cm and vertical span>2cm.',
        'initial_object_support_evidence':initial_support,'static_contact_object_frames':static,
        'limitations':['Scene mesh is view-constrained and not watertight; a visibility boundary alone is not a physical table edge.',
                       'The convex hull fills unseen holes and is only a footprint estimate.',
                       'Static object support estimates use reconstructed object meshes and poses, not force measurements.',
                       'Lifted/held trajectories are excluded from static support inference.']}
    np.savez_compressed(a.output_dir/'table_surface_evidence.npz',patch_vertices=patch,hull_xy=polygon,
                        boundary_points=boundary,physical_side_boundary_points=physical,visibility_boundary_points=visibility,
                        raw_cloud_vertical_side_boundary_points=cloud_side,
                        plane_origin=origin,plane_normal=normal,static_support_points_xy=np.vstack([x for x in static_footprints if len(x)]))
    (a.output_dir/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,9),layout='constrained')
    ax.scatter(points[::3,0],points[::3,1],s=.4,c='#bdbdbd',alpha=.25,label='static scene projected')
    ax.scatter(patch[::2,0],patch[::2,1],s=2,c='#2b8cbe',alpha=.7,label='measured tabletop patch')
    ax.plot(*np.vstack([polygon,polygon[0]]).T,color='#08589e',lw=1.5,label='visible patch convex hull')
    if len(physical):ax.scatter(physical[:,0],physical[:,1],s=12,c='#e6550d',label='boundary with measured vertical side')
    if len(cloud_side):ax.scatter(cloud_side[:,0],cloud_side[:,1],s=15,c='#e6550d',label='raw-cloud vertical-side evidence')
    if len(visibility):ax.scatter(visibility[:,0],visibility[:,1],s=5,c='#999999',label='visibility-only boundary')
    colors=plt.cm.tab10(np.arange(6))
    for oi,rows in enumerate(static):
        frames=rows['frames']
        unheld=rows['unheld_observed_frames']
        if unheld:ax.scatter(ob['position_m'][unheld,oi,0],ob['position_m'][unheld,oi,1],s=8,color=colors[oi],label=f'object{oi+1} static, no hand candidate')
        held=np.flatnonzero(np.asarray(ob['valid'][:,oi],bool)&hand_near[:,oi])
        if len(held):ax.scatter(ob['position_m'][held,oi,0],ob['position_m'][held,oi,1],s=2,color=colors[oi],alpha=.25)
        ax.scatter(initial[oi,0],initial[oi,1],s=70,marker='x',color=colors[oi]);ax.text(initial[oi,0],initial[oi,1],str(oi+1))
    human=np.load(D/'human/root_poses_sim.npz',allow_pickle=False)['position_m'][:838]
    ax.plot(human[:,0],human[:,1],color='#333333',lw=1,alpha=.6,label='raw observed human pelvis path')
    oldlow=np.asarray([-1.0066061807,1.3451216030]);oldhigh=np.asarray([-.1370155323,2.2665904951])
    box=np.array([oldlow,[oldhigh[0],oldlow[1]],oldhigh,[oldlow[0],oldhigh[1]],oldlow])
    ax.plot(*box.T,'--',color='#bd0026',lw=1.8,label='old swept-object support box')
    ax.scatter([.08],[1.8],s=100,marker='*',c='black',label='current robot base (.08,1.8)')
    ax.set(xlabel='world X (m)',ylabel='world Y (m)',title='Source tabletop geometry vs object-swept box')
    ax.set_aspect('equal');ax.set_xlim(-1.35,.45);ax.set_ylim(1.1,2.7);ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.2)
    fig.savefig(a.output_dir/'table_footprint_evidence.png',dpi=180);plt.close(fig)
    print(json.dumps({k:summary[k]for k in ['chosen_component','observed_patch_xy_min','observed_patch_xy_max','convex_hull_area_m2',
         'boundary_edges_count','physical_side_supported_boundary_count','physical_side_supported_xy_min','physical_side_supported_xy_max','raw_cloud_vertical_side_boundary_count','initial_object_support_evidence']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
