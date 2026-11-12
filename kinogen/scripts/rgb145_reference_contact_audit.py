"""Audit reference foot/terrain geometry, without integrating any physics step.

Geometric contacts are not measured load-bearing contacts. A dynamically valid
motion may transiently place COM outside its contact hull; this scan diagnoses
the difference between a planned whole footprint and the actual touching edge.
"""
import argparse,hashlib,json,sys
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def convex_hull(points):
    import numpy as np
    pts=np.unique(np.asarray(points,float).reshape(-1,2),axis=0)
    if len(pts)<3:return pts
    pts=pts[np.lexsort((pts[:,1],pts[:,0]))]
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo=[];hi=[]
    for seq,out in ((pts,lo),(pts[::-1],hi)):
        for p in seq:
            while len(out)>=2 and cross(out[-2],out[-1],p)<=1e-12:out.pop()
            out.append(p)
    return np.asarray(lo[:-1]+hi[:-1])


def hull_measure(hull,xy):
    """Return horizontal area, outside distance, signed inside margin."""
    import numpy as np
    hull=np.asarray(hull,float).reshape(-1,2);xy=np.asarray(xy,float)
    if not len(hull):return 0.,float('nan'),float('nan')
    if len(hull)==1:
        dist=float(np.linalg.norm(xy-hull[0]));return 0.,dist,-dist
    ends=np.roll(hull,-1,axis=0)if len(hull)>2 else hull[::-1]
    edge=ends-hull;frac=np.clip(np.sum((xy-hull)*edge,axis=1)/np.maximum(np.sum(edge**2,axis=1),1e-20),0,1)
    dist=float(np.min(np.linalg.norm(xy-(hull+frac[:,None]*edge),axis=1)))
    if len(hull)==2:return 0.,dist,-dist
    area=float(abs(np.sum(hull[:,0]*ends[:,1]-hull[:,1]*ends[:,0]))/2)
    signed=((edge[:,0]*(xy-hull)[:,1]-edge[:,1]*(xy-hull)[:,0])/np.maximum(np.linalg.norm(edge,axis=1),1e-20))
    margin=float(np.min(signed));return area,0. if margin>=-1e-9 else dist,margin


def pack_ragged(items,dimension):
    import numpy as np
    offsets=np.r_[0,np.cumsum([len(x)for x in items])]
    values=np.concatenate(items,axis=0)if offsets[-1]else np.empty((0,dimension))
    return values,offsets


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--support',type=Path,default=Path('outputs/rgb145_pipeline/support_geometry.json'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worst-count',type=int,default=20)
    p.add_argument('--sole-lift-threshold-mm',type=float,default=3.)
    a=p.parse_args()
    if sys.platform!='linux':p.error('Run the model geometry scan on the authorized server; no local simulation')
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np,mujoco as mj
    from rgb145_support import inside_convex_polygon
    z=dict(np.load(a.reference,allow_pickle=False));q=np.asarray(z['qpos']);n=len(q)
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m)
    if (m.nq,m.nv,m.nu)!=(59,58,52)or q.shape!=(n,59):raise ValueError('Expected single Vita 59/58/52 reference')
    names=list(map(str,z['dof_names']));jids=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,name)for name in names]);qa=m.jnt_qposadr[jids]
    root=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'vita_demo_free_root');rootqa=int(m.jnt_qposadr[root]);base=int(m.jnt_bodyid[root])
    footbody=np.array([mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in ('left','right')])
    gnames=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g)or str(g)for g in range(m.ngeom)]
    footgeom=[set(g for g,name in enumerate(gnames)if name.startswith('collision_foot_'+s+'_'))for s in ('l','r')]
    if any(not g for g in footgeom)or min(footbody)<0:raise ValueError('Missing foot capsule geometry/body')
    spec=json.loads(a.support.read_text());patch={s['name']:s for s in spec['supports']}
    ground_name=spec['ground']['name'];ground=np.asarray(spec['ground']['z_eq_ax_by_c'],float)
    roi=np.asarray(spec['collision_roi_xy_m'],float)
    ground_poly=np.array([[roi[0,0],roi[0,1]],[roi[1,0],roi[0,1]],[roi[1,0],roi[1,1]],[roi[0,0],roi[1,1]]])
    support_ids={g for g,name in enumerate(gnames)if name in patch or name==ground_name}
    expected=np.asarray(z['foot_support_expected'],bool);planned_names=np.asarray(z['foot_support_geom_name']).astype(str)
    if expected.shape!=(n,2)or planned_names.shape!=(n,2):raise ValueError('Invalid support plan')
    planned_ids=np.array([[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,name)for name in row]for row in planned_names])
    if np.any(expected&(planned_ids<0)):raise ValueError('Unknown support geometry for an expected stance foot')
    arrays={name:np.full((n,2),np.nan)for name in (
        'ankle_up_vs_support_normal_deg','sole_plane_clearance_min_m','sole_plane_clearance_max_m',
        'sole_plane_clearance_spread_m','toe_minus_heel_plane_clearance_m','toe_minus_heel_world_z_m',
        'correct_contact_fore_aft_span_m','correct_contact_lateral_span_m','correct_contact_hull_area_xy_m2',
        'correct_contact_hull_com_outside_distance_xy_m','correct_contact_hull_com_signed_margin_xy_m')}
    arrays.update({name:np.zeros((n,2),dtype=int)for name in ('correct_top_contact_count','all_terrain_top_contact_count','wrong_top_support_contact_count','near_plane_sole_endpoint_count')})
    arrays['whole_capsule_footprint_inside_planned_patch']=np.zeros((n,2),bool)
    com=np.zeros((n,3));max_pen=np.zeros(n)
    global_metric=np.full((n,3,3),np.nan)
    correct_points=[];all_points=[];actual_footprints=[];sole_points=[];global_hulls=[];rows=[]
    circles=np.c_[np.cos(np.arange(32)*2*np.pi/32),np.sin(np.arange(32)*2*np.pi/32)]
    for f in range(n):
        d.qpos[:]=m.qpos0;d.qpos[rootqa:rootqa+7]=q[f,:7];d.qpos[qa]=q[f,7:];d.qvel[:]=0.;d.ctrl[:]=0.
        mj.mj_forward(m,d);mj.mj_collision(m,d);com[f]=d.subtree_com[base]
        max_pen[f]=max([max(0.,-float(c.dist))for c in d.contact[:d.ncon]]or[0.])
        perfoot_correct=[[],[]];perfoot_all=[[],[]];seat_points=[]
        for c in d.contact[:d.ncon]:
            if c.dist>1e-6:continue
            gs=(int(c.geom1),int(c.geom2))
            for k in range(2):
                if gs[0]in footgeom[k]and gs[1]in support_ids:other=gs[1];normal=-np.asarray(c.frame[:3])
                elif gs[1]in footgeom[k]and gs[0]in support_ids:other=gs[0];normal=np.asarray(c.frame[:3])
                else:continue
                if normal[2]<.7:continue
                perfoot_all[k].append(np.asarray(c.pos).copy())
                if other==planned_ids[f,k]:perfoot_correct[k].append(np.asarray(c.pos).copy())
                else:arrays['wrong_top_support_contact_count'][f,k]+=1
            for robotidx in (0,1):
                other=gs[1-robotidx];body=int(m.geom_bodyid[gs[robotidx]])
                bname=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,body)or''
                normal=np.asarray(c.frame[:3])*(1 if robotidx==1 else-1)
                if other in support_ids and gnames[other].startswith('seat_support_')and('hip_'in bname or bname=='base_link')and normal[2]>.7:seat_points.append(np.asarray(c.pos).copy())
        for k in range(2):
            name=planned_names[f,k]
            # Unlabelled swing feet have no planned support. Use the ground
            # only for a clearance diagnostic; their planned contact ID stays
            # -1, so this cannot manufacture a correct supporting contact.
            if planned_ids[f,k]<0:name=ground_name
            beta=ground if name==ground_name else np.asarray(patch[name]['top_z_eq_ax_by_c'])
            polygon=ground_poly if name==ground_name else np.asarray(patch[name]['xy_polygon_m'])
            normal=np.r_[-beta[:2],1.];normal_length=float(np.linalg.norm(normal));normal/=normal_length;offset=float(beta[2]/normal_length)
            R=d.xmat[footbody[k]].reshape(3,3);angle=float(np.degrees(np.arccos(np.clip(R[:,2]@normal,-1,1))))
            lowpoints=[];shadow=[]
            for g in sorted(footgeom[k]):
                center=d.geom_xpos[g];axis=d.geom_xmat[g].reshape(3,3)[:,2];radius=float(m.geom_size[g,0]);typ=int(m.geom_type[g])
                if typ==int(mj.mjtGeom.mjGEOM_CAPSULE):ends=center+np.array([-1.,1.])[:,None]*float(m.geom_size[g,1])*axis
                elif typ==int(mj.mjtGeom.mjGEOM_SPHERE):ends=np.asarray([center])
                else:raise ValueError('Expected real capsule/sphere sole geometry: '+gnames[g])
                lowpoints.extend(ends-radius*normal)
                shadow.extend((ends[:,None,:2]+radius*circles[None,:,:]).reshape(-1,2))
            lowpoints=np.asarray(lowpoints);clearance=lowpoints@normal-offset
            localx=(lowpoints-d.xpos[footbody[k]])@R[:,0]
            toe=localx>=np.max(localx)-.02;heel=localx<=np.min(localx)+.02
            footprint=convex_hull(shadow);fit=bool(np.all(inside_convex_polygon(footprint,polygon)))
            cp=np.asarray(perfoot_correct[k]).reshape(-1,3);ap=np.asarray(perfoot_all[k]).reshape(-1,3)
            ch=convex_hull(cp[:,:2]);area,distance,margin=hull_measure(ch,com[f,:2])
            arrays['ankle_up_vs_support_normal_deg'][f,k]=angle
            arrays['sole_plane_clearance_min_m'][f,k]=float(np.min(clearance));arrays['sole_plane_clearance_max_m'][f,k]=float(np.max(clearance))
            arrays['sole_plane_clearance_spread_m'][f,k]=float(np.ptp(clearance))
            arrays['toe_minus_heel_plane_clearance_m'][f,k]=float(np.mean(clearance[toe])-np.mean(clearance[heel]))
            arrays['toe_minus_heel_world_z_m'][f,k]=float(np.mean(lowpoints[toe,2])-np.mean(lowpoints[heel,2]))
            arrays['near_plane_sole_endpoint_count'][f,k]=int(np.count_nonzero(abs(clearance)<=a.sole_lift_threshold_mm/1000))
            arrays['correct_top_contact_count'][f,k]=len(cp);arrays['all_terrain_top_contact_count'][f,k]=len(ap)
            arrays['whole_capsule_footprint_inside_planned_patch'][f,k]=fit
            arrays['correct_contact_hull_area_xy_m2'][f,k]=area
            arrays['correct_contact_hull_com_outside_distance_xy_m'][f,k]=distance
            arrays['correct_contact_hull_com_signed_margin_xy_m'][f,k]=margin
            arrays['correct_contact_fore_aft_span_m'][f,k]=float(np.ptp(cp@R[:,0]))if len(cp)else 0.
            arrays['correct_contact_lateral_span_m'][f,k]=float(np.ptp(cp@R[:,1]))if len(cp)else 0.
            correct_points.append(cp);all_points.append(ap);actual_footprints.append(footprint);sole_points.append(lowpoints)
            if expected[f,k]:
                rows.append({'frame_index':f,'source_frame_id':int(z['frame_ids'][f]),'source_time_s':float(z['source_time_s'][f]),'execution_time_s':float(z['time_s'][f]),'foot':('left','right')[k],'planned_support':name,
                    'ankle_up_error_deg':angle,'whole_footprint_inside_patch':fit,'correct_top_contact_count':len(cp),
                    'contact_fore_aft_span_mm':1000*arrays['correct_contact_fore_aft_span_m'][f,k],
                    'contact_lateral_span_mm':1000*arrays['correct_contact_lateral_span_m'][f,k],
                    'contact_hull_area_cm2':area*10000,'sole_clearance_min_mm':float(np.min(clearance)*1000),
                    'sole_clearance_max_mm':float(np.max(clearance)*1000),'toe_minus_heel_plane_mm':1000*arrays['toe_minus_heel_plane_clearance_m'][f,k],
                    'com_outside_this_foot_contact_hull_mm':None if not np.isfinite(distance)else 1000*distance})
        planned=np.concatenate([np.asarray(perfoot_correct[k]).reshape(-1,3)for k in range(2)if expected[f,k]],axis=0)if expected[f].any()else np.empty((0,3))
        every=np.concatenate([np.asarray(v).reshape(-1,3)for v in perfoot_all],axis=0)
        withseat=np.concatenate([every,np.asarray(seat_points).reshape(-1,3)],axis=0)
        for j,points in enumerate((planned,every,withseat)):
            hull=convex_hull(points[:,:2]);global_hulls.append(hull);global_metric[f,j]=hull_measure(hull,com[f,:2])
    values={**arrays,'source_frame_ids':z['frame_ids'],'source_time_s':z['source_time_s'],'execution_time_s':z['time_s'],
        'foot_support_expected':expected,'foot_support_geom_name':planned_names,'robot_com_world':com,'max_geometric_penetration_m':max_pen,
        'global_hull_metrics':global_metric,'global_hull_metric_names':np.array(['area_xy_m2','com_outside_distance_xy_m','com_signed_margin_xy_m']),
        'global_hull_group_names':np.array(['planned_stance_feet_correct_support','all_feet_terrain_contacts','all_feet_plus_hip_seat_contacts'])}
    for key,items,dim in (('correct_contact',correct_points,3),('all_foot_terrain_contact',all_points,3),('actual_footprint',actual_footprints,2),('capsule_plane_support_point',sole_points,3),('global_contact_hull',global_hulls,2)):
        val,offs=pack_ragged(items,dim);values[key+'_points']=val;values[key+'_offsets']=offs
    stance=expected;angle=arrays['ankle_up_vs_support_normal_deg'];spread=arrays['sole_plane_clearance_spread_m'];count=arrays['correct_top_contact_count'];fit=arrays['whole_capsule_footprint_inside_planned_patch']
    summary={'frames':n,'stance_foot_frames':int(stance.sum()),'stance_without_correct_geometric_contact':int(np.count_nonzero(stance&(count==0))),
        'stance_whole_footprint_fits_but_contact_is_missing':int(np.count_nonzero(stance&fit&(count==0))),
        'stance_whole_footprint_fits_but_contact_fore_aft_span_below_20mm':int(np.count_nonzero(stance&fit&(count>0)&(arrays['correct_contact_fore_aft_span_m']<.02))),
        'stance_ankle_normal_error_deg_p50_p95_max':np.quantile(angle[stance],[.5,.95,1]).tolist(),
        'stance_sole_clearance_spread_mm_p50_p95_max':(1000*np.quantile(spread[stance],[.5,.95,1])).tolist(),
        'stance_sole_spread_above_threshold_count':int(np.count_nonzero(stance&(spread>a.sole_lift_threshold_mm/1000))),
        'stance_whole_footprint_outside_patch':int(np.count_nonzero(stance&~fit)),
        'frames_without_planned_foot_contact_hull':int(np.count_nonzero(~np.isfinite(global_metric[:,0,1]))),
        'frames_COM_outside_planned_foot_contact_hull_over_10mm':int(np.count_nonzero(global_metric[:,0,1]>.01)),
        'frames_COM_outside_all_foot_plus_seat_contact_hull_over_10mm':int(np.count_nonzero(global_metric[:,2,1]>.01)),
        'max_geometric_penetration_mm':float(1000*np.max(max_pen))}
    def score(r):return (1000. if r['correct_top_contact_count']==0 else 0.)+abs(r['toe_minus_heel_plane_mm'])+r['ankle_up_error_deg']+(100. if not r['whole_footprint_inside_patch']else 0.)
    worst=sorted(rows,key=score,reverse=True)[:a.worst_count]
    a.output.mkdir(parents=True);np.savez_compressed(a.output/'arrays.npz',**values)
    report={'schema':'rgb145.reference_actual_contact_geometry.v1','mode':'per_frame_kinematic_geometry_scan_no_mj_step','simulation_passed':False,
        'model_sha256':sha(a.model),'reference_sha256':sha(a.reference),'support_sha256':sha(a.support),'mujoco_version':mj.__version__,
        'summary':summary,'top_worst_stance_foot_frames':worst,
        'array_layout_notes':{'foot_ragged_offsets':'flatten frame-major then left/right; offsets length2*T+1','global_hull_offsets':'frame-major then the3 groups; offsets length3*T+1','sole_points':'Each real capsule endpoint minus radius*unit_support_normal; not an assumed flat sole rectangle'},
        'limitations':['No physics integration; these are geometric touching candidates, not measured supporting loads.','COM outside instantaneous contact hull is a diagnostic, not by itself proof of dynamic infeasibility.','Unlabelled swing feet use the ground plane for clearance diagnostics only; their missing planned support is never counted as a correct contact.','The ground footprint containment audit uses the observed ROI; the collision plane may extend farther.','Contact hulls use strict nonpositive-distance top-facing MuJoCo contacts; a positive sole gap can leave the planned footprint unsupported.','No model geometry, mass, inertia, effort, joint limits or contact parameters changed.']}
    (a.output/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({'summary':summary,'top_worst_stance_foot_frames':worst},allow_nan=False),flush=True)


if __name__=='__main__':main()
