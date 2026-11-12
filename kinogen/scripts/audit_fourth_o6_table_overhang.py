"""Read-only finite table-prism / initial plate overhang geometry.

The table is the documented source-visible support-completion collision proxy,
not a claim of a measured real boundary. Proposed slides are arithmetic only;
no source/actual object state or physics parameter is changed.
"""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from scipy.spatial import ConvexHull

R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--model',type=Path,default=R/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_o6_calibrated_mimic_v3.xml')
p.add_argument('--candidate',type=Path,default=R/'reports/fourth_o6_static_grasp_refined_v1.json')
p.add_argument('--output',type=Path,default=R/'reports/fourth_o6_table_edge_overhang_geometry.json');a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m);d.qpos[:]=json.loads(a.candidate.read_text())['best']['qpos'];mj.mj_kinematics(m,d)
def vertices(g):
    mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
    return m.mesh_vert[start:start+count]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
table=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'table_support');tv=vertices(table)
numeric=mj.mj_name2id(m,mj.mjtObj.mjOBJ_NUMERIC,'source_table_plane_abcd');adr=int(m.numeric_adr[numeric]);plane=m.numeric_data[adr:adr+4].copy();plane/=np.linalg.norm(plane[:3])
sd=tv@plane[:3]+plane[3];top=tv[sd>=sd.max()-1e-6];xy=np.unique(np.round(top[:,:2],7),axis=0);hull=ConvexHull(xy);polygon=xy[hull.vertices]
body=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'target_object_01');centre=d.xpos[body].copy();edges=[]
for i,start in enumerate(polygon):
    end=polygon[(i+1)%len(polygon)];direction=end-start;out=np.asarray([direction[1],-direction[0]])/np.linalg.norm(direction)
    signed=float(out@(centre[:2]-start));edges.append({'start_xy':start.tolist(),'end_xy':end.tolist(),'outward_horizontal_unit_normal':out.tolist(),'object_origin_signed_edge_clearance_m':signed})
edge=max(edges,key=lambda x:x['object_origin_signed_edge_clearance_m']);normal2=np.asarray(edge['outward_horizontal_unit_normal']);anchor=np.asarray(edge['start_xy']);normal3=np.r_[normal2,0.]
bottom=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'target_object_01_bottom_collision');Rbottom=d.geom_xmat[bottom].reshape(3,3);axis=Rbottom[:,2]
extent=float(m.geom_size[bottom,0])*np.sqrt(max(0.,1-float(normal3@axis)**2))+float(m.geom_size[bottom,1])*abs(float(normal3@axis))
bottom_centre=d.geom_xpos[bottom].copy();bottom_side=float(normal2@(bottom_centre[:2]-anchor));wall=[]
for g in range(m.ngeom):
    if int(m.geom_bodyid[g])==body and m.geom_contype[g]>0 and m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:wall.append(vertices(g))
wv=np.vstack(wall);wall_sides=(wv[:,:2]-anchor)@normal2
out_tangent=np.r_[normal2,-float(plane[:2]@normal2)/plane[2]];out_tangent/=np.linalg.norm(out_tangent)
sliding=[]
for distance in [0.,.025,.05]:
    delta=distance*out_tangent;normal_gain=float(normal2@delta[:2])
    sliding.append({'physical_slide_length_m':distance,'world_delta_xyz_m':delta.tolist(),
                    'source_raw_origin_displacement_norm_m':float(np.linalg.norm(delta)),
                    'hypothetical_object_origin_xyz_m':(centre+delta).tolist(),
                    'centre_inset_from_proxy_edge_m':-float(edge['object_origin_signed_edge_clearance_m'])-normal_gain,
                    'outer_rim_overhang_positive_m':float(wall_sides.max()+normal_gain),
                    'bottom_disc_overhang_positive_m':bottom_side+extent+normal_gain,
                    'bottom_disc_fully_inside_proxy_footprint_for_this_edge':bool(bottom_side+extent+normal_gain<=0),
                    'table_plane_distance_unchanged':bool(abs(float(plane[:3]@delta))<1e-12)})
report={'schema':'fourth.o6_finite_table_plate_overhang_readonly.v1','model':str(a.model),'object_id':1,
        'table_boundary_kind':'Source-visible convex support-completion collision proxy, not a uniquely measured true edge.',
        'table_plane_abcd':plane.tolist(),'table_top_polygon_xy_m':polygon.tolist(),'table_thickness_m':float(sd.max()-sd.min()),
        'initial_object_origin_world_m':centre.tolist(),'nearest_proxy_edge':edge,
        'bottom_disc_radius_m':float(m.geom_size[bottom,0]),'bottom_disc_half_thickness_m':float(m.geom_size[bottom,1]),
        'initial_bottom_projection_interval_signed_edge_m':[bottom_side-extent,bottom_side+extent],
        'initial_hollow_wall_projection_interval_signed_edge_m':[float(wall_sides.min()),float(wall_sides.max())],
        'hypothetical_tangent_slides':sliding,'actual_model_reference_pose_or_controller_modified':False,
        'practical_constraint':'A hand below the top plane is allowed outside the finite prism; use real table_support mesh distance/SAT, not an infinite-plane exclusion.',
        'runtime_guard_note':'Current guard uses source plane only as sufficient above-table proof, then actual finite-prism distance fallback for below-plane geometry.',
        'limitations':['Slides are candidate geometry only; a real pre-slide must be executed by physical contacts, not object state assignment.',
                       'A 5cm slide can expose bottom near the edge but does not by itself prove under-COM access or wrench-feasible grasp.']}
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report),flush=True)
