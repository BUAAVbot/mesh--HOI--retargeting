"""Read-only audit of pelvis/hip seated support against actual visual surfaces."""
from pathlib import Path
import argparse,json,hashlib
import xml.etree.ElementTree as ET

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    import numpy as np,mujoco as mj
    m=mj.MjModel.from_binary_path(str(a.model));d=mj.MjData(m);ref=np.load(a.reference,allow_pickle=False);trace=np.load(a.trace,allow_pickle=False)
    print('TRACE_KEYS',[(k,trace[k].shape)for k in trace.files],flush=True)
    print('REF_KEYS',[(k,ref[k].shape)for k in ref.files],flush=True)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    bodies=['p1_base_link','p1_left_hip_pitch_link','p1_right_hip_pitch_link','p1_left_hip_roll_link','p1_right_hip_roll_link','p1_waist_yaw_link','p1_waist_pitch_link']
    bn=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,b)or'world'for b in range(m.nbody)]
    geoms=[g for g in range(m.ngeom)if bn[int(m.geom_bodyid[g])]in bodies]
    def vertices(g):
        typ=m.geom_type[g]
        if typ==mj.mjtGeom.mjGEOM_MESH:
            mid=m.geom_dataid[g];i=m.mesh_vertadr[mid];n=m.mesh_vertnum[mid];local=m.mesh_vert[i:i+n].copy()
        elif typ==mj.mjtGeom.mjGEOM_CAPSULE:
            radius,half=m.geom_size[g,:2];theta=np.linspace(0,np.pi,33);phi=np.linspace(0,2*np.pi,64,endpoint=False)
            local=np.asarray([[radius*np.sin(t)*np.cos(p),radius*np.sin(t)*np.sin(p),radius*np.cos(t)+(half if np.cos(t)>=0 else-half)]for t in theta for p in phi])
        elif typ==mj.mjtGeom.mjGEOM_BOX:
            from itertools import product
            local=np.asarray(list(product(*[[-x,x]for x in m.geom_size[g]])))
        elif typ==mj.mjtGeom.mjGEOM_CYLINDER:
            r,h=m.geom_size[g,:2];local=np.asarray([[r*np.cos(t),r*np.sin(t),z]for t in np.linspace(0,2*np.pi,128,endpoint=False)for z in[-h,h]])
        else:raise ValueError('Unexpected geometry '+str(typ))
        return local@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
    sk=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'bench_seat_proxy');seat_top=float(m.geom_pos[sk,2]+m.geom_size[sk,2])
    qkey=next((k for k in ['qpos','actual_qpos','qpos_actual','sim_qpos']if k in trace.files),None)
    if qkey is None:raise ValueError('Need actual full-state key in trace '+str(trace.files))
    times=trace['time_s']if'time_s'in trace.files else trace['time']
    poses=[]
    q=np.empty(m.nq);q[:59]=ref['qpos'][0,0];q[59:118]=ref['qpos'][0,1];q[118:]=ref['object_target_pose'][0];poses.append(('reference_frame0',q))
    for t in [0.,1.25,2.]:
        i=int(np.argmin(abs(times-t)));poses.append(('actual_t'+format(float(times[i]),'.3f'),trace[qkey][i]))
    records=[];surface_audits=[]
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(len(poses),2,figsize=(13,4*len(poses)),constrained_layout=True)
    for row,(label,q)in enumerate(poses):
        d.qpos[:]=q;mj.mj_fwdPosition(m,d);rows=[]
        for g in geoms:
            v=vertices(g);active=bool(m.geom_contype[g]or m.geom_conaffinity[g]);name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or('visual_'+bn[int(m.geom_bodyid[g])]);body=bn[int(m.geom_bodyid[g])]
            cylinder_candidate=None
            if row==0 and active and 'hip_roll_cylinder'in name:
                axisz=float(d.geom_xmat[g].reshape(3,3)[2,2]);radius,half=m.geom_size[g,:2]
                lower=float(d.geom_xpos[g,2]-half*abs(axisz)-radius*np.sqrt(max(0.,1-axisz**2)))
                cylinder_candidate={'radius_m':float(radius),'half_length_m':float(half),'axis_world_z':axisz,'capped_cylinder_bottom_world_z_m':lower,'capped_cylinder_gap_to_seat_m':lower-seat_top,'capsule_extra_extension_down_m':float(lower-v[:,2].min())}
            rows.append({'geom':name,'body':body,'active_collision':active,'type':int(m.geom_type[g]),'vertex_count':len(v),'world_bounds_m':[v.min(axis=0).tolist(),v.max(axis=0).tolist()],
                         'original_URDF_capped_cylinder_candidate':cylinder_candidate,
                         'lowest_gap_above_seat_m':float(v[:,2].min()-seat_top),'lowest_vertices_mean_m':v[v[:,2]<v[:,2].min()+.004].mean(axis=0).tolist()})
            if row==0 and not active and 'hip_roll_link'in body:
                from scipy.spatial import ConvexHull
                hull=ConvexHull(v);tri=v[hull.simplices];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)/2
                near=np.mean(tri[:,:,2],axis=1)<v[:,2].min()+.010
                down=hull.equations[:,2]<-.8
                patch=near&down;flat=patch&(hull.equations[:,2]<-np.cos(np.deg2rad(5)))
                floorpts=v[v[:,2]<v[:,2].min()+.004]
                surface_audits.append({'body':body,'visual_lowest_4mm_patch_xy_bounds_m':[floorpts[:,:2].min(axis=0).tolist(),floorpts[:,:2].max(axis=0).tolist()],
                                       'convex_hull_lowest10mm_downward_surface_area_m2':float(area[patch].sum()),'within5deg_horizontal_area_m2':float(area[flat].sum()),
                                       'candidate':'Per-link true visual mesh hull with unchanged local transform; recompute seated posture before use, rather than extending a new seat pad below the visual shell.'})
            color='tab:orange'if active else'tab:blue'
            stride=max(1,len(v)//2500)
            for col,axis in enumerate([1,0]):
                axs[row,col].scatter(v[::stride,axis],v[::stride,2],s=3 if active else 1,alpha=.6 if active else .18,c=color,label=('collision'if active else'visual')if g==geoms[0]else None)
        for col,axis in enumerate([1,0]):
            ax=axs[row,col];ax.axhline(seat_top,c='black',ls='--',label='bench seat top');ax.set_aspect('equal');ax.set_title(label+(' | Y-Z side'if axis==1 else' | X-Z transverse'));ax.set_xlabel('world '+'XYZ'[axis]+' [m]');ax.set_ylabel('world Z [m]');ax.grid(alpha=.2)
        records.append({'pose':label,'geometries':rows})
    fig.savefig(a.output_dir/'seated_support_visual_collision_sections.jpg',dpi=140);plt.close(fig)
    source_path=Path('/data/yunhao.xia/GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/vita02a/urdf/vita02a_shoes.urdf')
    source_collisions=[]
    if source_path.exists():
        ur=ET.parse(source_path)
        for bn0 in ['base_link','left_hip_pitch_link','right_hip_pitch_link','left_hip_roll_link','right_hip_roll_link']:
            for c in ur.findall("link[@name='"+bn0+"']/collision"):source_collisions.append({'body':bn0,'xml':ET.tostring(c,encoding='unicode')})
    report={'model':str(a.model),'model_sha256':sha(a.model),'trace':str(a.trace),'trace_sha256':sha(a.trace),'reference':str(a.reference),'reference_sha256':sha(a.reference),'bench_seat_top_m':seat_top,'poses':records,'reference_hip_surface_audits':surface_audits,'near_source_URDF':str(source_path),'near_source_URDF_sha256':sha(source_path)if source_path.exists()else None,'near_source_collisions':source_collisions,'notes':['Read-only FK of recorded physical states and reference. No model modification or simulation is performed.','Orange = active collision capsule; blue = source visual surface vertices. Capsule surfaces are sampled analytically.','A mesh visual minimum does not by itself establish a safe flat support surface; geometry candidates need containment and dynamics checks.']}
    (a.output_dir/'seated_support_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'seat_top':seat_top,'rows':records[0]}),flush=True)
if __name__=='__main__':main()
