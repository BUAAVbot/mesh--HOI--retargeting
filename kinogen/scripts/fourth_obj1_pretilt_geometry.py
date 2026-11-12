"""Finite-table, settled-pose plate-underbody planning evidence; no integration.

Slide/tilt cases live only in disposable query state. Neither source data nor
physical initial object states are written by this program.
"""
from pathlib import Path
import argparse,hashlib,json,os


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--stand',type=Path,required=True);p.add_argument('--settled-candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if os.name!='posix':p.error('Server-only private geometry')
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np
    import mujoco as mj
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation
    from scipy.optimize import least_squares
    from fourth_arm_object_feedback import ArmObjectFeedback
    ref=dict(np.load(a.reference,allow_pickle=False));stand=dict(np.load(a.stand,allow_pickle=False));saved=json.loads(a.settled_candidate.read_text())['best']
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);h=ArmObjectFeedback(m,ref)
    names=list(map(str,ref['dof_names']));jids=np.array([m.joint(name).id for name in names]);qa=m.jnt_qposadr[jids]
    snames=list(map(str,stand['dof_names']));qstand=np.array([stand['qpos'][0,7+snames.index(name)]for name in names])
    base=np.array(saved['qpos']);base[qa]=qstand;base[qa[[i for i,name in enumerate(names)if name.startswith('waist_')]]]=0
    relations=[(int(m.eq_obj1id[e]),int(m.eq_obj2id[e]),m.eq_data[e,:5].copy())for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT]
    def expand():
        for child,master,coef in relations:
            value=d.qpos[m.jnt_qposadr[master]]-m.qpos0[m.jnt_qposadr[master]]
            d.qpos[m.jnt_qposadr[child]]=m.qpos0[m.jnt_qposadr[child]]+sum(c*value**i for i,c in enumerate(coef))
    def forward():mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
    d.qpos[:]=base;expand();forward()
    ni=m.numeric('source_table_plane_abcd').id;adr=int(m.numeric_adr[ni]);plane=m.numeric_data[adr:adr+4].copy();plane/=np.linalg.norm(plane[:3]);up=plane[:3]
    def mesh_local(g):
        mid=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mid]);count=int(m.mesh_vertnum[mid]);return m.mesh_vert[start:start+count].astype(float)
    def geometry_world(g):
        if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:v=mesh_local(g)
        elif m.geom_type[g]==mj.mjtGeom.mjGEOM_CYLINDER:
            theta=np.arange(512)*2*np.pi/512;r,z=m.geom_size[g,:2];v=np.array([[r*np.cos(t),r*np.sin(t),zz]for t in theta for zz in [-z,z]])
        else:raise ValueError('Unexpected object primitive')
        return v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
    table=h.table;tv=geometry_world(table);top=tv[(tv@up+plane[3])>np.max(tv@up+plane[3])-1e-5]
    hull=ConvexHull(top[:,:2]);poly=top[hull.vertices,:2];equations=hull.equations;prism=ConvexHull(tv).equations
    pobj=d.xpos[h.object_bids[0]].copy();Robj=d.xmat[h.object_bids[0]].reshape(3,3).copy();comlocal=m.body_ipos[h.object_bids[0]].copy()
    obj_local=np.concatenate([(geometry_world(g)-pobj)@Robj for g in h.object_geoms[0]])
    bottom=next(g for g in h.object_geoms[0]if m.geom_type[g]==mj.mjtGeom.mjGEOM_CYLINDER)
    radius,half=map(float,m.geom_size[bottom,:2]);Rbottom=Rotation.from_quat(m.geom_quat[bottom,[1,2,3,0]]).as_matrix()
    bottom_centre_local=m.geom_pos[bottom]+Rbottom[:,2]*(-half)
    rimlocal=obj_local[obj_local[:,2]>obj_local[:,2].max()-1e-5]
    origin_insets=-(equations[:,:2]@pobj[:2]+equations[:,2]);edge=int(np.argmin(origin_insets));normal=equations[edge,:2].copy()
    edges=[]
    for i,point in enumerate(poly):
        end=poly[(i+1)%len(poly)];delta=end-point;out=np.array([delta[1],-delta[0]]);out/=np.linalg.norm(out)
        edges.append({'start_xy':point.tolist(),'end_xy':end.tolist(),'length_m':float(np.linalg.norm(delta)),'normal':out.tolist()})
    edge_segment=min(edges,key=lambda e:np.linalg.norm(np.array(e['normal'])-normal))
    long_edge=max(edges,key=lambda e:e['length_m']if np.dot(e['normal'],[.7,.7])>.95 else 0);long_normal=np.array(long_edge['normal'])
    robot_direction=base[:2]-pobj[:2];robot_direction/=np.linalg.norm(robot_direction)
    def tangent3(direction):
        result=np.r_[direction,-(up[:2]@direction)/up[2]];return result/np.linalg.norm(result)
    def bottom_info(pos,rot):
        centre=pos+rot@bottom_centre_local;axes=rot@Rbottom
        return centre,axes
    def plane_gap(pos,rot):
        centre=pos+rot@m.geom_pos[bottom];axis=(rot@Rbottom)[:,2];projection=float(axis@up)
        return float(centre@up+plane[3]-half*abs(projection)-radius*np.sqrt(max(0.,1-projection**2)))
    def ray_table_z(xy):
        constant=xy@prism[:,:2].T+prism[:,3];nz=prism[:,2];lower=np.full(len(xy),-np.inf);upper=np.full(len(xy),np.inf);allowed=np.ones(len(xy),bool)
        positive=nz>1e-8;negative=nz< -1e-8;flat=~(positive|negative)
        upper=np.min(-constant[:,positive]/nz[positive],axis=1);lower=np.max(-constant[:,negative]/nz[negative],axis=1)
        if flat.any():allowed&=np.all(constant[:,flat]<=1e-8,axis=1)
        allowed&=lower<=upper+1e-8
        return np.where(allowed,upper,np.nan)
    def clipped(points):
        out=points.copy()
        for row in equations:
            result=[]
            for i,current in enumerate(out):
                previous=out[i-1];dc=current@row[:2]+row[2];dp=previous@row[:2]+row[2]
                if (dc<=0)!=(dp<=0):result.append(previous+(current-previous)*dp/(dp-dc))
                if dc<=0:result.append(current)
            out=np.asarray(result).reshape(-1,2)
            if not len(out):break
        return out
    def area(points):return float(abs(np.sum(points[:,0]*np.roll(points[:,1],-1)-points[:,1]*np.roll(points[:,0],-1)))/2)if len(points)>=3 else 0.
    # Actual rigid distal-mesh normal thickness, not an invented finger sphere.
    finger_dimensions=[]
    for side,prefix in enumerate(['lh_','rh_']):
        hb=h.hand_bids[side];local_frame=ref['robot_palm_anatomical_frame_in_hand_base'][side];HR=d.xmat[hb].reshape(3,3)
        for digit in ['index','middle']:
            g=m.geom('collision_'+prefix+digit+'_distal_mesh').id;v=(geometry_world(g)-d.xpos[hb])@HR
            projected=v@local_frame;span=np.ptp(projected,axis=0)
            profile=[]
            for depth in [.005,.010,.015]:
                tip=projected[projected[:,0]>=projected[:,0].max()-depth]
                profile.append({'tip_depth_m':depth,'normal_thickness_m':float(np.ptp(tip[:,2])),'lateral_width_m':float(np.ptp(tip[:,1]))})
            finger_dimensions.append({'hand':['left','right'][side],'digit':digit,'length_m':float(span[0]),'lateral_width_m':float(span[1]),'normal_thickness_m':float(span[2]),'tip_sections':profile})
    thickness=max(row['normal_thickness_m']for row in finger_dimensions);pad_radius=thickness/2
    theta=np.arange(720)*2*np.pi/720;disk=np.column_stack([radius*np.cos(theta),radius*np.sin(theta),np.zeros(len(theta))])
    other_bounds=[]
    for gs in h.object_geoms[1:]:
        vertices=np.concatenate([geometry_world(g)for g in gs]);other_bounds.append((vertices.min(0),vertices.max(0)))
    scenarios=[]
    raw=ref['object_target_pose'][0,0];rawR=Rotation.from_quat(raw[[4,5,6,3]]).as_matrix()
    definitions=[('raw0_comparison',raw[:3],rawR,{'actual_raw_unchanged':True}),('settled_in_place',pobj,Robj,{})]
    for label,direction,length in [('slide25mm_nearest',normal,.025),('slide50mm_nearest',normal,.05),('slide50mm_long_front',long_normal,.05)]:
        delta=tangent3(direction)*length;definitions.append((label,pobj+delta,Robj,{'hypothetical_supported_slide_delta_world_m':delta.tolist(),'requires_actuator_contact_execution':True}))
    for degrees in [6.,8.,10.,12.]:
        toward=tangent3(robot_direction);local_toward=Robj.T@toward;local_toward[2]=0;local_toward/=np.linalg.norm(local_toward)
        pivot=pobj+Robj@(bottom_centre_local-radius*local_toward);axis=-np.cross(up,toward);axis/=np.linalg.norm(axis)
        turn=Rotation.from_rotvec(axis*np.deg2rad(degrees)).as_matrix();rot=turn@Robj;pos=pivot+turn@(pobj-pivot)
        support_correction=-plane_gap(pos,rot)*up;pos+=support_correction
        definitions.append((f'pretilt_robot_side_{int(degrees)}deg',pos,rot,{'hypothetical_pretilt_degrees':degrees,'pivot_world_m':pivot.tolist(),
            'pivot_kind':'far bottom-disc circumference, not outerrim','tilt_toward_robot_xy':robot_direction.tolist(),'quasistatic_support_normal_adjustment_m':support_correction.tolist(),'requires_actuator_contact_execution':True}))
    for label,pos,rot,extra in definitions:
        centre,axes=bottom_info(pos,rot);bottom_points=disk@axes.T+centre;rim=rimlocal@rot.T+pos;com=pos+rot@comlocal
        bottompoly=bottom_points[ConvexHull(bottom_points[:,:2]).vertices,:2];rimpoly=rim[ConvexHull(rim[:,:2]).vertices,:2]
        signed_bottom=bottom_points[:,:2]@normal+equations[edge,2];signed_rim=rim[:,:2]@normal+equations[edge,2]
        xyz=obj_local@rot.T+pos;low,high=xyz.min(0),xyz.max(0)
        obstacle=[{'object_id':oi+2,'aabb_separation_lower_bound_m':float(np.linalg.norm(np.maximum(np.maximum(low-hi,lo-high),0.)))}for oi,(lo,hi)in enumerate(other_bounds)]
        xx,yy=np.meshgrid(np.linspace(-radius,radius,101),np.linspace(-radius,radius,101));inside=xx*xx+yy*yy<=(radius-.006)**2
        local=np.column_stack([xx[inside],yy[inside],np.zeros(inside.sum())]);points=local@axes.T+centre
        tabletop=ray_table_z(points[:,:2]);gap=points[:,2]-tabletop
        pad_centres=points-axes[:,2]*(pad_radius+.001);certificate=np.max(pad_centres@prism[:,:3].T+prism[:,3],axis=1)
        safe=certificate>=pad_radius+.001
        selected=[]
        for target_t in [-.012,.012]:
            direction=robot_direction if label.startswith('pretilt')else normal;tangent=np.array([-direction[1],direction[0]])
            delta_xy=points[:,:2]-com[:2];tcoord=delta_xy@tangent;eligible=safe&(delta_xy@direction>0)&(abs(tcoord-target_t)<.004)
            if eligible.any():
                ids=np.flatnonzero(eligible);chosen=ids[np.argmin(np.linalg.norm(delta_xy[ids],axis=1))]
                selected.append({'object_surface_point_world_m':points[chosen].tolist(),'object_outward_normal_world':(-axes[:,2]).tolist(),'desired_force_on_object_world':axes[:,2].tolist(),
                    'relative_com_normal_m':float(delta_xy[chosen]@direction),'relative_com_tangent_m':float(tcoord[chosen]),
                    'table_vertical_gap_m':float(gap[chosen])if np.isfinite(gap[chosen])else None,'outside_actual_vertical_prism':bool(not np.isfinite(tabletop[chosen])),
                    'distal_normal_thickness_clearance_certificate_m':float(certificate[chosen]-pad_radius)})
        radial_gaps=[]
        for r in [.05,.06,.07]:
            direction=robot_direction if label.startswith('pretilt')else normal;xy=centre[:2]+direction*r
            zz=centre[2]-(axes[:2,2]@(xy-centre[:2]))/axes[2,2];tz=ray_table_z(xy[None,:])[0]
            radial_gaps.append({'radius_toward_approach_m':r,'point_world_m':[float(xy[0]),float(xy[1]),float(zz)],'bottom_table_vertical_gap_m':float(zz-tz)if np.isfinite(tz)else None,'outside_finite_table':bool(not np.isfinite(tz))})
        row={'case':label,'private_planning_only':True,'object_pose_world_xyz_wxyz':np.r_[pos,Rotation.from_matrix(rot).as_quat()[[3,0,1,2]]].tolist(),
            'object_com_world_m':com.tolist(),'com_rise_from_settled_m':float((com-(pobj+Robj@comlocal))@up),
            'com_inset_from_nearest_original_edge_m':float(-(normal@com[:2]+equations[edge,2])),
            'bottom_disc_signed_overhang_nearest_edge_m':float(signed_bottom.max()),'outerrim_signed_overhang_nearest_edge_m':float(signed_rim.max()),
            'bottom_disc_exposed_area_fraction':1-area(clipped(bottompoly))/area(bottompoly),'outerrim_footprint_exposed_area_fraction':1-area(clipped(rimpoly))/area(rimpoly),
            'other_five_obstacle_clearance_lower_bounds':obstacle,'bottom_disc_support_plane_gap_m':plane_gap(pos,rot),
            'pad_cross_section_safe_grid_fraction':float(safe.mean()),'pad_normal_thickness_used_m':thickness,
            'candidate_under_support_contact_points':selected,'r50_60_70mm_bottom_clearances':radial_gaps,**extra}
        scenarios.append(row)
    # Endpoint-only seven-arm reach test of the Sonic below-table palm seed.
    closure_sign=[]
    for side,prefix in enumerate(['lh_','rh_']):
        index=names.index(prefix+'index_mcp_pitch');HR=d.xmat[h.hand_bids[side]].reshape(3,3);frame=HR@ref['robot_palm_anatomical_frame_in_hand_base'][side]
        closure_sign.append(float(np.sign(frame[:,2]@np.cross(d.xaxis[jids[index]],frame[:,0]))))
    reach=[];slide_case=next(row for row in scenarios if row['case']=='slide50mm_nearest');slidepos=np.array(slide_case['object_pose_world_xyz_wxyz'][:3]);toward=tangent3(normal)
    for side in range(2):
        d.qpos[:]=base;expand();forward();chain=h.chains[side];shoulder=m.body(['left','right'][side]+'_shoulder_pitch_link').id
        points=np.asarray([d.xpos[int(m.jnt_bodyid[j])]for j in chain['joint_ids']]+[h.palm(d,side)]);length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum());shoulder_pos=d.xpos[shoulder].copy()
        angle=np.deg2rad(20.);f=-np.cos(angle)*toward-np.sin(angle)*up;palmar=-np.sin(angle)*toward+np.cos(angle)*up;nn=palmar*closure_sign[side];desired=np.column_stack([f,np.cross(nn,f),nn])@ref['robot_palm_anatomical_frame_in_hand_base'][side].T
        goal=slidepos+.15*toward;goal[2]=-(plane[0]*goal[0]+plane[1]*goal[1]+plane[3])/plane[2]-.048
        qstart=d.qpos[chain['qa']].copy()
        def residual(q):
            d.qpos[chain['qa']]=q;forward();return np.r_[30*(h.palm(d,side)-goal),Rotation.from_matrix(d.xmat[h.hand_bids[side]].reshape(3,3)@desired.T).as_rotvec()]
        fit=least_squares(residual,qstart,bounds=(chain['lower']+1e-6,chain['upper']-1e-6),max_nfev=120)
        residual(fit.x);reach.append({'hand':['left','right'][side],'waist_locked_zero':True,'base_xyz_wxyz':base[:7].tolist(),'palm_target_world_m':goal.tolist(),
            'shoulder_to_target_distance_m':float(np.linalg.norm(goal-shoulder_pos)),'arm_chain_triangle_upper_bound_m':length,
            'position_error_m':float(np.linalg.norm(h.palm(d,side)-goal)),'orientation_error_rad':float(np.linalg.norm(fit.fun[3:])),
            'necessary_position_shortfall_lower_bound_m':max(0.,float(np.linalg.norm(goal-shoulder_pos))-length),
            'joint_targets':{name:float(d.qpos[adr])for name,adr in zip(chain['names'],chain['qa'])},
            'scope':'Endpoint kinematics only; no full hand path/mesh/wrench/physics claim. SONIC owns six-command whole-hand search.'})
    mass=float(m.body_mass[h.object_bids[0]]);weight=mass*float(np.linalg.norm(m.opt.gravity));rimradius=float(np.linalg.norm(rimlocal[:,:2],axis=1).max());partial=weight*radius/(radius+rimradius)
    report={'schema':'fourth_obj1_finite_table_underplate_pretilt_geometry.v1','model':str(a.model),'model_sha256':sha(a.model),'source_reference':str(a.reference),'source_reference_sha256':sha(a.reference),
        'settled_snapshot':str(a.settled_candidate),'source_script_sha256':sha(__file__),'actual_source_or_execution_modified':False,'dynamics_run':False,
        'source_geometry_note':'Convex source-visible/support-completion table proxy, not uniquely measured full real edge. Raw0 and 2s settled object poses are distinguished.',
        'plane_abcd':plane.tolist(),'top_polygon_world_xy':poly.tolist(),'prism_halfspaces_world':prism.tolist(),'settled_object_origin_world_m':pobj.tolist(),
        'raw0_origin_world_m':raw[:3].tolist(),'raw0_minus_settled_origin_m':(raw[:3]-pobj).tolist(),
        'nearest_edge':dict(edge_segment,outward_normal_xy=normal.tolist(),settled_origin_inset_m=float(origin_insets[edge])),
        'long_front_edge':long_edge,'bottom_disc_radius_m':radius,'bottom_disc_half_thickness_m':half,'actual_O6_distal_mesh_dimensions':finger_dimensions,
        'scenarios':scenarios,'fixed_base_below_table_seed_endpoint_reach':reach,
        'partial_support_simple_lever_estimate':{'object_mass_kg':mass,'gravity_weight_N':weight,'far_bottom_pivot_radius_m':radius,'near_outerrim_radius_m':rimradius,
            'vertical_finger_force_N':partial,'fraction_of_weight':partial/weight,'scope':'Fz*(bottomRadius+rimRadius)=mg*bottomRadius only. Table retains remaining load; not a full friction-cone/motor-wrench or dynamic feasibility proof.'},
        'new_grasp_constraints':['Use full finite table_support prism, not an infinite top plane below-table exclusion.','Temporary single-finger lift-edge mode may use table support; it is not yet an unsupported load-bearing grasp.','After opening the gap, target 2+ underside patches with force-on-object positiveZ and a thumb/rim counter moment; validate full gravity+wrench with true .4Nm master limits.','Hypothetical tilted/slid object cases are not valid physical initializations or successful demonstrations. Execute object motion only through actual actuators and contacts.','Pad cross-section certificates are necessary local geometry only; actual whole O6 meshes, ten mimics, robot reach, all obstacles and path still require validation.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output.with_suffix('.npz'),table_top_polygon_xy=poly,table_prism_halfspaces=prism,source_raw_initial_object_pose=raw,settled_object_pose=np.r_[pobj,Rotation.from_matrix(Robj).as_quat()[[3,0,1,2]]])
    print(json.dumps({'output':str(a.output),'nearest_edge':report['nearest_edge'],'finger_dimensions':finger_dimensions,'partial_support':report['partial_support_simple_lever_estimate'],
        'cases':[{'case':r['case'],'bottom_overhang_m':r['bottom_disc_signed_overhang_nearest_edge_m'],'exposed_fraction':r['bottom_disc_exposed_area_fraction'],'COM_rise_m':r['com_rise_from_settled_m'],'gaps':r['r50_60_70mm_bottom_clearances']}for r in scenarios],'reach':reach}),flush=True)


if __name__=='__main__':main()
