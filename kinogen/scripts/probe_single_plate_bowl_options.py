"""Read-only hand-opening and original bowl geometry screen; not a grasp demo."""
from pathlib import Path
import json
import numpy as np
import mujoco as mj
from fourth_convex_separation import ConvexSeparationOracle

R=Path(__file__).resolve().parents[1];m=mj.MjModel.from_xml_path(str(R/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_o6_calibrated_mimic_v3.xml'));d=mj.MjData(m);o=ConvexSeparationOracle(m)
z=np.load(R/'outputs/fourth_full_sequence/physics_sonic15225_o6_support_absolute_hold_v1_5s/physical_trace.npz');d.qpos[:]=z['qpos'][0];mj.mj_forward(m,d)
digits=['thumb','index','middle','ring','pinky'];gs={v:[g for g in range(m.ngeom)if m.geom_contype[g]and(m.body(int(m.geom_bodyid[g])).name or'').startswith('rh_'+v)]for v in digits};hb=m.body('rh_hand_base_link').id
def vertices(g):
 if m.geom_type[g]!=mj.mjtGeom.mjGEOM_MESH:return None
 return o._shape(g)['vertices']@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
def local(v):return(v-d.xpos[hb])@d.xmat[hb].reshape(3,3)
rows=[]
for yaw in [.5,.9,1.24]:
 for pitch in [0.,.15,.3]:
  for finger in [0.,.2,.4,.6,.8]:
   for name,value in zip(['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch'],[yaw,pitch,finger,finger,finger,finger]):d.qpos[int(m.joint('rh_'+name).qposadr[0])]=value
   for e in range(m.neq):
    if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT:d.qpos[int(m.jnt_qposadr[int(m.eq_obj1id[e])])]=m.eq_data[e,1]*d.qpos[int(m.jnt_qposadr[int(m.eq_obj2id[e])])]
   mj.mj_forward(m,d);vs={v:np.concatenate([local(vertices(g))for g in gg if vertices(g)is not None])for v,gg in gs.items()}
   tips={v:local(d.xpos[m.body('rh_'+v+'_distal').id])for v in digits};axis=np.mean([tips[v]for v in digits[1:]],axis=0)-tips['thumb'];axis/=np.linalg.norm(axis)
   gap=[float(np.min(vs[v]@axis)-np.max(vs['thumb']@axis))for v in digits[1:]];rows.append({'commands_rad':[yaw,pitch,finger,finger,finger,finger],'directional_plane_gap_m':gap,'minimum_common_gap_m':min(gap),'axis_hand_base':axis.tolist()})
objects=[]
for oid in [1,5,6]:
 b=m.body(f'target_object_{oid:02d}').id;world=np.concatenate([vertices(g)for g in range(m.ngeom)if m.geom_bodyid[g]==b and m.geom_contype[g]and vertices(g)is not None]);v=(world-d.xpos[b])@d.xmat[b].reshape(3,3);zz=v[:,2];rr=np.linalg.norm(v[:,:2],axis=1)
 objects.append({'object_id':oid,'mass_kg':float(m.body_mass[b]),'outer_radius_m':float(rr.max()),'collision_mesh_height_m':float(zz.max()-zz.min()),'top_z_m':float(zz.max()),'bottom_mesh_z_m':float(zz.min()),'base_outer_radius_m':float(rr[zz<zz.min()+1e-5].max()),'gravity_eccentric_rim_moment_Nm':float(m.body_mass[b]*9.81*rr.max())})
report={'scope':'Directional support-plane opening of original O6 and geometric comparison only; not exact finger Euclidean separation or feasible arm/path proof. No geometry/model mutation beyond private FK qpos.', 'best_common_opening':max(rows,key=lambda x:x['minimum_common_gap_m']),'objects':objects,'samples':rows};p=R/'reports/single_plate_bowl_geometry_screen.json';p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k!='samples'}),flush=True)
