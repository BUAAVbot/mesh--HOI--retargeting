"""Audit kinematic COM/capture-point support feasibility, without simulation."""
from pathlib import Path
import argparse,json


def polygon_distance(point,poly):
    import numpy as np
    edges=np.roll(poly,-1,axis=0)-poly;rel=point-poly
    crosses=edges[:,0]*rel[:,1]-edges[:,1]*rel[:,0]
    if np.all(crosses>=-1e-9):return 0.
    t=np.clip(np.sum(rel*edges,axis=1)/np.maximum(np.sum(edges*edges,axis=1),1e-12),0,1)
    return float(np.min(np.linalg.norm(rel-t[:,None]*edges,axis=1)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import numpy as np
    import mujoco as mj
    from scipy.spatial import ConvexHull
    z=np.load(a.reference,allow_pickle=False);m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m)
    names=[str(n)for n in z['dof_names']];qa=m.jnt_qposadr[[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]]
    rb=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'base_link');floorid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'assumed_floor');floor=float(z['physics_floor_z_m'])
    feet=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in ['left','right']]
    footgeoms=[[g for g in range(m.ngeom)if int(m.geom_bodyid[g])==bid and m.geom_contype[g]>0]for bid in feet]
    com=[];footpos=[];clearance=[];polygons=[];stance=[]
    for q in z['qpos']:
        d.qpos[:7]=q[:7];d.qpos[qa]=q[7:];mj.mj_forward(m,d)
        com.append(d.subtree_com[rb].copy());footpos.append(d.xpos[feet].copy());points=[];clears=[];support=[]
        for side,geoms in enumerate(footgeoms):
            distances=[float(mj.mj_geomDistance(m,d,g,floorid,3.,np.zeros(6)))for g in geoms]
            clears.append(min(distances));support.append(min(distances)<=.025)
            for g,dist in zip(geoms,distances):
                if dist>.025:continue
                rot=d.geom_xmat[g].reshape(3,3);pos=d.geom_xpos[g].copy();size=m.geom_size[g];typ=int(m.geom_type[g])
                if typ==mj.mjtGeom.mjGEOM_CAPSULE:extent=size[0]+size[1]*np.abs(rot[:,2])
                elif typ==mj.mjtGeom.mjGEOM_SPHERE:extent=np.repeat(size[0],3)
                elif typ==mj.mjtGeom.mjGEOM_BOX:extent=np.abs(rot)@size
                elif typ==mj.mjtGeom.mjGEOM_CYLINDER:extent=size[0]*np.sqrt(np.maximum(0.,1-rot[:,2]**2))+size[1]*np.abs(rot[:,2])
                else:
                    bbox=m.geom_aabb[g];pos+=rot@bbox[:3];extent=np.abs(rot)@bbox[3:]
                for sx,sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:points.append(pos[:2]+np.array([sx,sy])*extent[:2])
        if len(points)<3:
            center=d.xpos[feet[np.argmin(clears)]][:2];points=[center+[x,y]for x,y in [(-.02,-.02),(.02,-.02),(.02,.02),(-.02,.02)]]
        points=np.unique(np.asarray(points),axis=0);poly=points[ConvexHull(points).vertices]
        polygons.append(poly);clearance.append(clears);stance.append(support)
    com=np.asarray(com);footpos=np.asarray(footpos);clearance=np.asarray(clearance);stance=np.asarray(stance);times=z['time_s']
    velocity=np.gradient(com,times,axis=0);omega=np.sqrt(9.81/np.maximum(com[:,2]-floor,.1));capture=com[:,:2]+velocity[:,:2]/omega[:,None]
    comoutside=np.asarray([polygon_distance(c[:2],p)for c,p in zip(com,polygons)]);cpoutside=np.asarray([polygon_distance(c,p)for c,p in zip(capture,polygons)])
    footvel=np.gradient(footpos,times,axis=0);stance_slip=np.linalg.norm(footvel[:,:,:2],axis=2)*stance
    pad=np.full((len(times),64,2),np.nan);counts=[]
    for i,poly in enumerate(polygons):pad[i,:len(poly)]=poly;counts.append(len(poly))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output.with_suffix('.npz'),time_s=times,com_world_m=com,com_velocity_world_m_s=velocity,capture_point_world_xy_m=capture,support_polygon_world_xy_m=pad,support_polygon_vertex_count=np.asarray(counts),foot_clearance_m=clearance,stance=stance,stance_foot_slip_m_s=stance_slip,com_outside_support_m=comoutside,capture_point_outside_support_m=cpoutside)
    window=times<=5
    report={'schema':'fourth_kinematic_com_support.v1','model':str(a.model),'reference':str(a.reference),'frames':len(times),'dynamics_run':False,
      'method':'Mass-weighted base_link subtree COM and finite-difference reference velocity; capture point LIP approximation; support polygon hull of active foot geometry XY bounds within25mm of floor. Conservative bounds, not ZMP/force proof.',
      'first5s':{'com_outside_max_m':float(comoutside[window].max()),'capture_outside_max_m':float(cpoutside[window].max()),'stance_foot_speed_max_m_s':float(stance_slip[window].max()),'both_feet_stance_fraction':float(np.mean(np.all(stance[window],axis=1)))},
      'full':{'com_outside_max_m':float(comoutside.max()),'capture_outside_max_m':float(cpoutside.max()),'stance_foot_speed_max_m_s':float(stance_slip.max()),'both_feet_stance_fraction':float(np.mean(np.all(stance,axis=1)))},
      'prefix_samples':[{'frame':int(i),'time_s':float(times[i]),'root_xyz':z['qpos'][i,:3].tolist(),'com_xyz':com[i].tolist(),'com_outside_m':float(comoutside[i]),'capture_outside_m':float(cpoutside[i]),'foot_clearance_m':clearance[i].tolist(),'stance_foot_speed_m_s':stance_slip[i].tolist()}for i in range(0,min(151,len(times)),15)],
      'worst_com_frames':np.argsort(-comoutside)[:10].tolist(),'worst_capture_frames':np.argsort(-cpoutside)[:10].tolist()}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
