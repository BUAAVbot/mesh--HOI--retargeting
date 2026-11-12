"""Synthesize a bounded six-driver O6 grip using its real polynomial mimics.

This is inferred geometry-based hand detail. The declared soft grasp remains;
no claim of pure friction grasping or observed source finger motion is made.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--input',default='outputs/rgb_near_pipeline/reference_v12_O6_left_clearance.npz')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/reference_v13_O6_grasp.npz')
    p.add_argument('--scene',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v4_self.xml')
    p.add_argument('--profile',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_O6_v4.json')
    p.add_argument('--joint-speed-limit',type=float,default=1.)
    a=p.parse_args();generator=Path(__file__).read_bytes()
    import numpy as np
    import mujoco as mj
    from scipy.optimize import least_squares
    root=a.root;inp=root/a.input;out=root/a.output;mp=root/a.scene;profile=root/a.profile
    z=np.load(inp,allow_pickle=False);payload={k:z[k]for k in z.files};q=z['qpos'].copy();names=[str(v)for v in z['dof_names']]
    meta=json.loads(profile.read_text());m=mj.MjModel.from_xml_path(str(mp));d=mj.MjData(m)
    calibration_path=Path(meta['source_profiles']['RIGHT']['calibration'])
    calibration=json.loads(calibration_path.read_text())
    if sha(calibration_path)!=meta['source_profiles']['RIGHT']['calibration_sha256']:
        raise ValueError('O6 right calibration hash mismatch')
    drivers=sorted([v for v in meta['drivers']if v['joint'].startswith('p1_rh_')],key=lambda v:v['motor_index'])
    if len(drivers)!=6:raise ValueError('Expected exactly six independent right O6 drivers')
    driver_names=[v['joint']for v in drivers];driver_ids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,v)for v in driver_names]
    lo=np.asarray([v['range_rad'][0]for v in drivers]);hi=np.asarray([v['range_rad'][1]for v in drivers])
    coupled=[]
    for node in ET.parse(mp).getroot().find('equality').findall('joint'):
        if (node.get('joint1')or'').startswith('p1_rh_'):
            driver=node.get('joint2');coef=np.asarray([float(x)for x in node.get('polycoef').split()])
            if driver not in driver_names:raise ValueError('Mimic driver missing')
            coupled.append((node.get('joint1'),driver_names.index(driver),coef))
    if len(coupled)!=5:raise ValueError('Expected five real O6 mimic relations')
    hand_names=[*driver_names,*[x[0]for x in coupled]]
    hand_ids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,v)for v in hand_names]
    qa=m.jnt_qposadr[hand_ids]
    refcols=np.asarray([7+names.index(v.removeprefix('p1_'))for v in hand_names])
    frame=int(z['shoe_release_frame']);times=z['time_s']
    for actor,pre in enumerate(('p1_','p2_')):
        r=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root');rq=int(m.jnt_qposadr[r])
        ids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+nm)for nm in names]
        if r<0 or min(ids)<0:raise ValueError('Scene/reference joint ABI mismatch')
        d.qpos[rq:rq+7]=q[frame,actor,:7];d.qpos[m.jnt_qposadr[ids]]=q[frame,actor,7:]
    sj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free');sq=int(m.jnt_qposadr[sj]);d.qpos[sq:sq+7]=z['object_target_pose'][frame]
    fixed=d.qpos.copy()
    def bn(b):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(b))or''
    def gn(g):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,int(g))or''
    hg=[g for g in range(m.ngeom)if bn(m.geom_bodyid[g]).startswith('p1_rh_')and m.geom_contype[g]]
    sg=[g for g in range(m.ngeom)if gn(g).startswith('shoe_')and m.geom_contype[g]]
    self_pairs={tuple(sorted((x,y))):i for i,(x,y)in enumerate((x,y)for j,x in enumerate(hg)for y in hg[j+1:])}
    fingers=['thumb','index','middle']
    distal=[]
    for finger in fingers:
        candidates=[g for g in hg if finger in bn(m.geom_bodyid[g])and'distal'in bn(m.geom_bodyid[g])]
        if not candidates:raise ValueError('No independent distal collision geom for '+finger)
        distal.append(candidates)
    def expand(x):return np.concatenate([np.asarray(x),np.asarray([np.polynomial.polynomial.polyval(x[j],c)for _,j,c in coupled])])
    def distances(x):
        d.qpos[:]=fixed;d.qpos[qa]=expand(x);mj.mj_kinematics(m,d)
        ds=np.asarray([[float(mj.mj_geomDistance(m,d,g,w,.3,np.zeros(6)))for w in sg]for g in hg])
        tip=np.asarray([min(ds[hg.index(g)].min()for g in subset)for subset in distal])
        mj.mj_collision(m,d)
        selfdist=np.full(len(self_pairs),.3)
        for c in d.contact:
            key=tuple(sorted((int(c.geom1),int(c.geom2))))
            if key in self_pairs:selfdist[self_pairs[key]]=min(selfdist[self_pairs[key]],float(c.dist))
        return ds,tip,selfdist
    nominal=.2*hi
    def residual(x):
        ds,tip,selfdist=distances(x);r=[3*(tip-.001),.005*(x-nominal)/hi,60*np.minimum(ds+.0005,0).ravel(),200*np.minimum(selfdist-.0005,0)]
        # Avoid a collision-free final pose reached by passing a finger through
        # the shoe earlier in the calibrated monotonic closing trajectory.
        for u in(.2,.4,.6,.8):
            ps,_,ss=distances(x*u);r.extend([40*np.minimum(ps+.0005,0).ravel(),200*np.minimum(ss-.0005,0)])
        return np.concatenate(r)
    best=None;attempts=[]
    seeds=[np.full(6,v)for v in(.02,.15,.35,.55)]+[np.asarray([.05,.75,.1,.1,.2,.2]),np.asarray([.35,.85,.2,.2,.2,.2])]
    for frac in seeds:
        result=least_squares(residual,lo+(hi-lo)*frac,bounds=(lo+1e-7,hi-1e-7),max_nfev=160,ftol=1e-7,xtol=1e-7,gtol=1e-7,diff_step=1e-4)
        ds,tip,selfdist=distances(result.x)
        samples=[distances(result.x*u)for u in np.linspace(0,1,65)]
        path_min=min(float(v[0].min())for v in samples);self_min=min(float(v[2].min())for v in samples)
        handq=expand(result.x)
        limits_ok=bool(np.all(handq>=m.jnt_range[hand_ids,0]-1e-7)and np.all(handq<=m.jnt_range[hand_ids,1]+1e-7))
        changed=int(np.sum(result.x>.05))
        passed=bool(np.max(tip)<=.03 and path_min>=-.002 and self_min>=-.001 and changed>=2 and limits_ok)
        row={'seed_fraction':frac.tolist(),'cost':float(result.cost),'drivers_rad':result.x.tolist(),
             'distal_gap_m':dict(zip(fingers,tip.tolist())),'closing_path_min_gap_m':path_min,
             'closing_path_hand_self_min_gap_m':self_min,
             'changed_driver_count':changed,'limits_ok':limits_ok,'geometry_passed':passed}
        attempts.append(row)
        score=(0 if passed else 1,float(result.cost))
        if best is None or score<best[0]:best=(score,result.x,row)
    theta=best[1];u=np.linspace(0,1,4097);smooth=3*u*u-2*u*u*u;dsmooth=6*u*(1-u)
    driver_curve=smooth[:,None]*theta[None,:]
    joint_curve=np.column_stack([driver_curve,*[np.polynomial.polynomial.polyval(driver_curve[:,j],c)for _,j,c in coupled]])
    driver_deriv=dsmooth[:,None]*theta[None,:]
    joint_deriv=np.column_stack([driver_deriv,*[np.polynomial.polynomial.polyval(driver_curve[:,j],np.polynomial.polynomial.polyder(c))*driver_deriv[:,j]for _,j,c in coupled]])
    duration=max(.5,float(np.max(np.abs(joint_deriv)))/a.joint_speed_limit*1.25)
    start=float(times[frame]);end=start+duration
    if end>times[-1]:raise ValueError('Insufficient sequence duration for URDF-limited O6 closure')
    phase=np.clip((times-start)/duration,0,1);progress=3*phase*phase-2*phase*phase*phase
    curve=np.asarray([expand(theta*v)for v in progress])
    if np.max(np.abs(q[:,0,refcols]))>1e-7:raise ValueError('Input hand must start from calibrated open255 pose')
    q[:,0,refcols]=curve
    maxvel=float(np.max(np.abs(np.diff(curve,axis=0))/np.diff(times)[:,None]))
    if maxvel>a.joint_speed_limit+1e-6:raise ValueError('Physical mimic-expanded O6 reference velocity limit exceeded')
    # Invert the supplied 256-entry calibration, not a linear range scaling.
    # Keep fractional angle targets for the smooth simulated motion; rounded
    # SDK commands are an explicitly quantified export, not a hardware test.
    commands=np.empty((len(times),6));closing_commands=np.empty_like(commands)
    table_roundtrip=np.empty_like(commands)
    for j,name in enumerate(driver_names):
        entry=calibration['joints'][name.removeprefix('p1_')]
        table=np.asarray(entry['angle_rad']);closing=np.asarray(entry['decreasing_rad'])
        if table.shape!=(256,)or np.any(np.diff(table)>1e-8):raise ValueError('Expected monotonic calibrated mean driver table')
        commands[:,j]=np.interp(curve[:,j],table[::-1],np.arange(256)[::-1])
        closing_commands[:,j]=np.interp(curve[:,j],closing[::-1],np.arange(256)[::-1])
        table_roundtrip[:,j]=table[np.rint(commands[:,j]).astype(int)]
    payload.update(qpos=q,qpos_p1=q[:,0],qpos_p2=q[:,1],
        O6_grasp_driver_names=np.asarray(driver_names),O6_grasp_joint_names=np.asarray(hand_names),
        O6_driver_target_rad=curve[:,:6],O6_driver_command_u8=np.rint(commands).astype(np.uint8),
        O6_driver_command_fractional=commands,O6_driver_command_closing_curve_fractional=closing_commands,
        O6_rounded_command_mean_angle_roundtrip_rad=table_roundtrip,
        O6_mimic_polycoef=np.asarray([x[2]for x in coupled]),
        O6_grasp_start_time_s=np.asarray(start),O6_grasp_end_time_s=np.asarray(end),
        O6_physical_joint_velocity_limit_rad_s=np.asarray(a.joint_speed_limit),
        O6_grasp_geometric_quality_passed=np.asarray(best[2]['geometry_passed']),
        O6_grasp_hand_synthetic=np.asarray(True),O6_friction_only_grasp=np.asarray(False))
    out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**payload)
    report={'schema':'rgb_near.O6_grasp_reference.v1','status':'geometry_grip_passed'if best[2]['geometry_passed']else'geometry_grip_failed',
        'input_reference_sha256':sha(inp),'reference_sha256':sha(out),'scene_sha256':sha(mp),'profile_sha256':sha(profile),
        'script_sha256':hashlib.sha256(generator).hexdigest(),'capture_frame':frame,'start_time_s':start,'end_time_s':end,
        'ramp_duration_s':duration,'max_expanded_joint_reference_speed_rad_s':maxvel,
        'velocity_time_safety_factor':1.25,
        'independent_drivers':6,'mimic_followers':5,'best':best[2],'attempts':attempts,
        'calibration_sha256':sha(calibration_path),
        'command_lookup':'Inverse supplied256-entry mean angle_rad table; decreasing_rad inverse included separately; hysteresis not compensated.',
        'rounded_command_angle_error_max_rad':float(np.max(np.abs(table_roundtrip-curve[:,:6]))),
        'limitations':['Source finger observations invalid; grasp hand shape is synthesized.',
            'Measured/transferred partial O6 profile polynomials are preserved.',
            'Soft grasp equality remains active; this is not a friction-only grasp proof.',
            'Geometry-only construction must pass the subsequent dynamic trial.']}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');out.with_suffix('.generator.py').write_bytes(generator)
    print(json.dumps(report),flush=True)
    if not best[2]['geometry_passed']:raise SystemExit(3)


if __name__=='__main__':main()
