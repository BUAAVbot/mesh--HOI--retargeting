"""Bounded kinematic contact hypothesis; no claim of dynamic feasibility.

Uses the native HDMI geometry and the contact-normal Jacobian projection
already used by getup/pipline/mocap/project_collisions.py.  The scene is never
moved here; root-Z correction is an explicitly labelled alternative hypothesis
pending independent mesh-ground calibration.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import lsq_linear, linprog
from scipy.sparse import diags, vstack, hstack, eye, csr_matrix
from scipy.spatial.transform import Rotation
from scipy.interpolate import RegularGridInterpolator


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path)
    p.add_argument('--scene',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--joint-project',action='store_true')
    p.add_argument('--ground-audit',type=Path)
    p.add_argument('--soft-root',action='store_true')
    p.add_argument('--deduplicate-contacts',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.source,allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
    meta=json.loads(a.source.with_suffix('.json').read_text())
    m=mujoco.MjModel.from_binary_path(str(a.model));d=mujoco.MjData(m)
    joint_names=meta['joint_names'];body_names=meta['body_names']
    jids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in joint_names])
    qids=m.jnt_qposadr[jids];vids=m.jnt_dofadr[jids]
    bids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot/'+n) for n in body_names]
    geoms=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in range(m.ngeom)]
    scene_ids=[i for i,n in enumerate(geoms) if not n.startswith('robot/')]
    feet=[[i for i,n in enumerate(geoms) if n.startswith('robot/') and ('foot_' in n or 'toe_' in n)
           and ('_l_' in n if side==0 else '_r_' in n)] for side in range(2)]
    foot_set=set(feet[0]+feet[1]);scene_set=set(scene_ids)
    scene_config=json.loads(a.scene.read_text())
    field=scene_config.get('ground_heightfield')
    height_interpolator=None;unknown_queries=[]
    if field:
        with np.load(field['path'],allow_pickle=False) as z:
            hx,hy,hh,unknown=z['x'].copy(),z['y'].copy(),z['height_m'].copy(),z['unknown_mask'].copy()
        height_interpolator=RegularGridInterpolator((hy,hx),hh,bounds_error=False,fill_value=np.nan)
        local_points={}
        for gid in foot_set:
            size=m.geom_size[gid]
            if m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:
                local_points[gid]=np.asarray([[u*size[0],v*size[1],w*size[2]] for u in (-1,0,1)
                    for v in (-1,0,1) for w in (-1,0,1) if abs(u)+abs(v)+abs(w)>0])
            elif m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_CYLINDER:
                angles=np.linspace(0,2*np.pi,32,endpoint=False)
                local_points[gid]=np.asarray([[size[0]*np.cos(theta),size[0]*np.sin(theta),h*size[1]]
                    for h in (-1,0,1) for theta in angles])
            else:raise ValueError('Unsupported foot collider for heightfield sampling')

        def sampled_gap():
            points=np.concatenate([local_points[g]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g] for g in foot_set])
            height=height_interpolator(points[:,[1,0]])
            ix=np.clip(np.searchsorted(hx,points[:,0])-1,0,len(hx)-2)
            iy=np.clip(np.searchsorted(hy,points[:,1])-1,0,len(hy)-2)
            known=~(unknown[iy,ix]|unknown[iy+1,ix]|unknown[iy,ix+1]|unknown[iy+1,ix+1])
            known&=np.isfinite(height)
            if not known.any():raise ValueError('No observed mesh support beneath either foot')
            return float((points[known,2]-height[known]).min())
    roots=np.concatenate((source['body_pos_w'][:,0],source['body_quat_w'][:,0]),axis=-1).astype(float)
    original_roots=roots.copy();q=source['joint_pos'].astype(float).copy();original_q=q.copy()
    lo=np.maximum(m.jnt_range[jids,0]+.002,original_q-.25)
    hi=np.minimum(m.jnt_range[jids,1]-.002,original_q+.25)
    if a.joint_project:q=np.clip(q,lo,hi)

    def forward(t,angles,zdelta):
        d.qpos[:7]=original_roots[t];d.qpos[2]+=zdelta;d.qpos[qids]=angles
        mujoco.mj_forward(m,d)

    def gap(t,angles,zdelta):
        forward(t,angles,zdelta)
        if height_interpolator is not None:return sampled_gap()
        distances=[mujoco.mj_geomDistance(m,d,g,s,10.,None) for g in foot_set for s in scene_ids]
        penetrating=[float(c.dist) for c in d.contact[:d.ncon]
                     if ((int(c.geom1) in foot_set and int(c.geom2) in scene_set) or
                         (int(c.geom2) in foot_set and int(c.geom1) in scene_set))]
        return min(distances+penetrating)

    def project_z(angles):
        raw=[]
        for t in range(len(q)):
            if height_interpolator is not None:
                raw.append(float(np.clip(.001-gap(t,angles[t],0),-.15,.15)));continue
            bottom,top=-.15,.15
            if gap(t,angles[t],bottom)>.001:
                raw.append(bottom);continue
            for _ in range(15):
                middle=(bottom+top)/2
                if gap(t,angles[t],middle)<.001:bottom=middle
                else:top=middle
            raw.append(top)
        raw=np.asarray(raw)
        n=len(raw);d2=diags([np.ones(n-2),-2*np.ones(n-2),np.ones(n-2)],[0,1,2],shape=(n-2,n))
        solution=lsq_linear(vstack((10*eye(n),12*d2)).tocsr(),np.r_[10*raw,np.zeros(n-2)],
            bounds=(raw,np.full(n,.15)),tol=1e-9,lsmr_tol='auto',max_iter=100).x
        if height_interpolator is not None:
            d1=diags([-np.ones(n-1),np.ones(n-1)],[0,1],shape=(n-1,n)).tocsr()
            # z >= raw makes sum(z-raw) a linear contact-gap objective. HiGHS
            # solves the same hard height/velocity/acceleration bounds without
            # a nonlinear optimizer's slow infeasible-start iterations.
            if a.soft_root:
                constraints=vstack((d1,-d1,d2,-d2)).tocsr()
                matrix=vstack((hstack((constraints,csr_matrix((constraints.shape[0],n)))),
                    hstack((eye(n),-eye(n))),hstack((-eye(n),-eye(n))))).tocsr()
                result=linprog(np.r_[np.zeros(n),np.ones(n)],A_ub=matrix,
                    b_ub=np.r_[np.full(2*(n-1),.010),np.full(2*(n-2),.0012),raw,-raw],
                    bounds=[(-.15,.15)]*n+[(0,None)]*n,method='highs')
            else:
                result=linprog(np.ones(n),A_ub=vstack((d1,-d1,d2,-d2)).tocsr(),
                    b_ub=np.r_[np.full(2*(n-1),.010),np.full(2*(n-2),.0012)],
                    bounds=list(zip(raw,np.full(n,.15))),method='highs')
            if not result.success:raise ValueError(f'Height profile continuity solver failed: {result.message}')
            solution=result.x[:n]
        return raw,solution

    raw,zdelta=project_z(q)
    root_only_z=zdelta.copy()
    jac1=np.zeros((3,m.nv));jac2=jac1.copy();jacr=jac1.copy();identity=np.eye(28)
    if a.joint_project:
        for pass_index in range(3):
            if pass_index:
                q=np.clip(original_q+gaussian_filter1d(q-original_q,1.,axis=0,mode='nearest'),lo,hi)
            for t in range(len(q)):
                for iteration in range(45):
                    forward(t,q[t],zdelta[t]);rows=[];residual=[]
                    contacts=list(d.contact[:d.ncon])
                    if a.deduplicate_contacts:
                        deepest={}
                        for c in contacts:
                            key=tuple(sorted(map(int,c.geom)))
                            if key not in deepest or c.dist<deepest[key].dist:deepest[key]=c
                        contacts=sorted(deepest.values(),key=lambda c:c.dist)
                    for c in contacts:
                        if c.dist>=-.001:continue
                        b1,b2=m.geom_bodyid[c.geom]
                        mujoco.mj_jac(m,d,jac1,jacr,c.pos,int(b1))
                        mujoco.mj_jac(m,d,jac2,jacr,c.pos,int(b2))
                        gradient=c.frame[:3]@(jac2-jac1)[:,vids]
                        rows.append(3*gradient[None]);residual.append(np.array([.003-c.dist])*3)
                    if not rows:break
                    rows.append(.012*identity);residual.append(.012*(original_q[t]-q[t]))
                    if t:
                        rows.append(.008*identity);residual.append(.008*((q[t-1]-original_q[t-1])-(q[t]-original_q[t])))
                    lower=np.maximum(lo[t]-q[t],-.06)-1e-10
                    upper=np.minimum(hi[t]-q[t],.06)+1e-10
                    delta=lsq_linear(np.vstack(rows),np.concatenate(residual),bounds=(lower,upper),
                                     method='bvls',tol=1e-7).x
                    q[t]=np.clip(q[t]+delta,lo[t],hi[t])
            # The final pass fixes root motion and clears remaining joint
            # collisions; do not move root again after that last correction.
            if not (a.soft_root and pass_index==2):raw,zdelta=project_z(q)
    position=[];quaternion=[];self_depth=[];scene_depth=[];support_gap=[]
    for t in range(len(q)):
        forward(t,q[t],zdelta[t]);position.append(d.xpos[bids].copy());quaternion.append(d.xquat[bids].copy())
        self_values=[];scene_values=[]
        for c in d.contact[:d.ncon]:
            bucket=self_values if all(geoms[int(g)].startswith('robot/') for g in c.geom) else scene_values
            bucket.append(max(-float(c.dist),0))
        self_depth.append(max(self_values or [0]));scene_depth.append(max(scene_values or [0]))
        support_gap.append(gap(t,q[t],zdelta[t]))
    position=np.asarray(position);quaternion=np.asarray(quaternion)
    velocities=np.gradient(position,.02,axis=0)
    angular=np.zeros_like(velocities)
    for t in range(len(q)):
        left,right=max(0,t-1),min(len(q)-1,t+1)
        rl=Rotation.from_quat(quaternion[left][:,[1,2,3,0]])
        rr=Rotation.from_quat(quaternion[right][:,[1,2,3,0]])
        angular[t]=(rr*rl.inv()).as_rotvec()/((right-left)*.02)
    output={**source,'joint_pos':q.astype(np.float32),'joint_vel':np.gradient(q,.02,axis=0).astype(np.float32),
        'body_pos_w':position.astype(np.float32),'body_quat_w':quaternion.astype(np.float32),
        'body_lin_vel_w':velocities.astype(np.float32),'body_ang_vel_w':angular.astype(np.float32),
        'joint_names':np.asarray(joint_names),'body_names':np.asarray(body_names),'fps':np.array(50.),
        'original_source_reference_sha256':np.array(sha(a.source)),
        'dynamics_verified':np.array(False),'initial_state_changed':np.array(True)}
    np.savez_compressed(a.output/'mjlab_motion.npz',**output)
    report=dict(status='kinematic_contact_hypothesis_not_dynamics_verified',hypothesis='robot_rootZ_relative_to_current_collision_scene',
        ground_mesh_calibration_status='confirmed_robot_hover_by_independent_mesh_audit' if a.ground_audit else 'pending_independent_audit',
        ground_mesh_audit_sha256=sha(a.ground_audit) if a.ground_audit else None,
        original_source_reference_sha256=sha(a.source),
        model_sha256=sha(a.model),scene_sha256=sha(a.scene),initial_state_changed=True,
        initial_rootZ_correction_m=float(zdelta[0]),rootZ_min_m=float(zdelta.min()),rootZ_max_m=float(zdelta.max()),
        rootZ_rmse_m=float(np.sqrt(np.mean(zdelta**2))),joint_delta_max_rad=float(abs(q-original_q).max()),
        joint_delta_rmse_rad=float(np.sqrt(np.mean((q-original_q)**2))),
        joint_step_max_rad=float(abs(np.diff(q,axis=0)).max()),joint_speed_max_rad_s=float(abs(output['joint_vel']).max()),
        root_step_max_m=float(abs(np.diff(zdelta)).max()),root_vertical_speed_max_m_s=float(abs(np.gradient(zdelta,.02)).max()),
        root_correction_acceleration_max_m_s2=float(abs(np.diff(zdelta,n=2)).max()/.02**2),
        max_self_penetration_m=float(max(self_depth)),max_scene_penetration_m=float(max(scene_depth)),
        supported_within_5mm_fraction=float(np.mean(np.asarray(support_gap)<=.005)),
        support_gap_initial_m=float(support_gap[0]),support_gap_final_m=float(support_gap[-1]),
        joint_projection=bool(a.joint_project),dynamics_verified=False,source_clock_unchanged=True,
        soft_root_fit=bool(a.soft_root),
        contact_rows_deduplicated=bool(a.deduplicate_contacts),
        scene_unchanged=True,rootXY_unchanged=True,root_quaternion_unchanged=True)
    report['kinematic_geometry_valid']=max(self_depth)<=.012 and max(scene_depth)<=.012
    (a.output/'projection_report.json').write_text(json.dumps(report,indent=2))
    (a.output/'mjlab_motion.json').write_text(json.dumps({**meta,'source':str(a.source),
        'joint_names':joint_names,'body_names':body_names,'frames':len(q),'fps':50.,'contact_projection':report},indent=2))
    np.savez_compressed(a.output/'contact_projection.npz',raw_rootZ_correction=raw,rootZ_correction=zdelta,
        root_only_rootZ_correction=root_only_z,support_gap_m=np.asarray(support_gap),
        self_penetration_m=np.asarray(self_depth),scene_penetration_m=np.asarray(scene_depth),joint_delta=q-original_q)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
