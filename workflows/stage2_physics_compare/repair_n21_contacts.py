"""Project the latest N21 reference onto the fixed new stairs, retaining the initial state."""
import argparse
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
from scipy import sparse
sys.path.append('/data/KinoGen/envs/retarget_compare_external/lib/python3.10/site-packages')
sys.path.append('/data/grail-envs/retarget/lib/python3.10/site-packages')
from qpsolvers import solve_qp,available_solvers

from common import classify_robot_static_bodies,sha256,write_json
from prepare_n21_dynamics import canonical
from temporal_objective import derivative_operators


def support_height(points,triangles):
    height=np.full(len(points),np.nan)
    for triangle in triangles:
        a,b,c=triangle[:,:2]
        uv=np.linalg.solve(np.column_stack((b-a,c-a)),(points[:,:2]-a).T).T
        inside=np.all(uv>=-1e-9,axis=1)&(uv.sum(axis=1)<=1+1e-9)
        height[inside]=np.fmax(height[inside],triangle[0,2])
    return height


def collision_residual(gradient,lower):
    # Opposing prism normals can make a linearized hard constraint infeasible.
    active=lower>0
    return -1000.*gradient[active],-1000.*lower[active]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output
    out=root/'contact_projection';out.mkdir(exist_ok=True)
    model=mj.MjModel.from_binary_path(str(root/'input/compiled_model.mjb'))
    with np.load(root/'input/reference_raw.npz',allow_pickle=False) as archive:ref={k:archive[k] for k in archive.files}
    with np.load(root/'input/candidate_geometry.npz',allow_pickle=False) as geometry:
        triangles=geometry['support_vertices'][geometry['support_faces']]
    source=ref['qpos_ref'];n=len(source);q=source.copy();nv=15;size=n*nv
    active=np.r_[np.arange(3),np.arange(6,18)]
    robot,static,_=classify_robot_static_bodies(model,mj)
    feet=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link') for side in ('left','right')]
    assert min(feet)>=0
    data=mj.MjData(model)
    targets=[];unknown=[]
    for t,pose in enumerate(source):
        data.qpos[:]=pose;mj.mj_forward(model,data)
        for side,body in enumerate(feet):
            phase=str(ref['foot_phase'][t,side])
            if phase not in ('flat','heel','toe'):continue
            local=(np.array([[0.,-.03,-.072],[.075,.03,-.072],[-.075,.03,-.072]]) if phase=='flat'
                   else np.array([[(-.096 if phase=='heel' else .1),0.,-.0702 if phase=='heel' else -.070]]))
            points=data.xpos[body]+local@data.xmat[body].reshape(3,3).T
            heights=support_height(points,triangles)
            for point,target_z,local_point in zip(points,heights,local):
                if not np.isfinite(target_z):unknown.append([t,side]);continue
                target=point.copy();target[2]=target_z-.0005
                targets.append((t,body,local_point,target))
    _,d2,d3=derivative_operators(ref['source_times_s'])
    temporal=sparse.vstack([sparse.kron(d2,sparse.diags(np.r_[np.full(3,.006),np.full(12,.003)])),
                            sparse.kron(d3,sparse.diags(np.r_[np.full(3,.00015),np.full(12,.00008)]))],format='csr')
    prior=np.tile(np.r_[20.,20.,10.,np.full(12,2.)],n)
    original=np.c_[source[:,:3],source[:,7:19]]
    by_frame={t:[] for t in range(n)}
    for record in targets:by_frame[record[0]].append(record)
    jp=np.zeros((3,model.nv));jr=np.zeros_like(jp)

    def jac(point,body):
        mj.mj_jac(model,data,jp,jr,point,body)
        return jp[:,active].copy()

    def inspect(poses,linear):
        residual=[];rr=[];cc=[];vv=[];gg=[];gc=[];gv=[];limits=[];max_depth=0.;penalty=0.
        for t,pose in enumerate(poses):
            data.qpos[:]=pose;mj.mj_forward(model,data)
            for _,body,local,target in by_frame[t]:
                point=data.xpos[body]+data.xmat[body].reshape(3,3)@local
                weight=np.array([15.,15.,300.]);error=weight*(point-target)
                start=len(residual);residual.extend(error)
                if linear:
                    matrix=weight[:,None]*jac(point,body)
                    row,column=np.nonzero(abs(matrix)>1e-14)
                    rr.extend(start+row);cc.extend(t*nv+column);vv.extend(matrix[row,column])
            for c in data.contact:
                b1,b2=map(int,model.geom_bodyid[[c.geom1,c.geom2]])
                if b1 not in robot and b2 not in robot:continue
                distance=float(c.dist);max_depth=max(max_depth,-distance)
                penalty+=max(0.,-distance-.0045)**2*1e6
                if linear and distance<.001:
                    gradient=c.frame[:3]@(jac(c.pos,b2)-jac(c.pos,b1))
                    nz=np.flatnonzero(abs(gradient)>1e-14);row=len(limits)
                    gg.extend([row]*len(nz));gc.extend(t*nv+nz);gv.extend(gradient[nz])
                    # Fixed first pose and invariant upper-body pairs retain the 5 mm gate.
                    cap=.005 if t==0 or np.linalg.norm(gradient)<1e-12 else .0045
                    limits.append(-cap-distance)
        x=np.c_[poses[:,:3],poses[:,7:19]]
        cost=float(np.sum((prior*(x-original).ravel())**2)+np.sum((temporal@x.ravel())**2)
                   +np.sum(np.square(residual))+penalty)
        a=sparse.coo_matrix((vv,(rr,cc)),shape=(len(residual),size)).tocsr() if linear else None
        g=sparse.coo_matrix((gv,(gg,gc)),shape=(len(limits),size)).tocsc() if linear else None
        return cost,max_depth,np.array(residual),a,g,np.array(limits)

    trace=[];solver=next(s for s in ('proxqp','clarabel','osqp') if s in available_solvers)
    for iteration in range(16):
        cost,depth,residual,contact,g,lower=inspect(q,True)
        x=np.c_[q[:,:3],q[:,7:19]]
        collision,collision_target=collision_residual(g,lower)
        a=sparse.vstack([sparse.diags(prior),temporal,contact,collision],format='csc')
        b=np.r_[-prior*(x-original).ravel(),-temporal@x.ravel(),-residual,collision_target]
        h=(a.T@a+sparse.eye(size)*1e-6).tocsc();gradient=-np.asarray(a.T@b).ravel()
        lo=np.tile(np.r_[np.full(3,-.04),np.full(12,-.12)],(n,1));hi=-lo.copy()
        delta=x-original;radius=np.r_[.06,.06,.12,np.full(12,.4)]
        lo=np.maximum(lo,-radius-delta);hi=np.minimum(hi,radius-delta)
        lo[:,3:]=np.maximum(lo[:,3:],model.jnt_range[1:13,0]-q[:,7:19])
        hi[:,3:]=np.minimum(hi[:,3:],model.jnt_range[1:13,1]-q[:,7:19])
        lo[0]=hi[0]=0.
        step=solve_qp(h,gradient,lb=lo.ravel(),ub=hi.ravel(),solver=solver)
        if step is None:
            trace.append(dict(iteration=iteration,status='infeasible_linearization',depth_m=depth));break
        step=step.reshape(n,nv);accepted=False
        for alpha in (1.,.5,.25,.125,.0625,.03125):
            trial=q.copy();trial[:,:3]+=alpha*step[:,:3];trial[:,7:19]+=alpha*step[:,3:]
            trial[0]=source[0]
            value,new_depth,*_=inspect(trial,False)
            if value<cost-1e-7 and new_depth<=max(.005,depth)+1e-6:
                q=trial;accepted=True;break
        trace.append(dict(iteration=iteration,accepted=accepted,alpha=alpha,cost_before=cost,
            cost_after=value if accepted else cost,depth_m=new_depth if accepted else depth))
        write_json(out/'progress.json',dict(trace=trace));print(json.dumps(trace[-1]),flush=True)
        if not accepted:break
    np.testing.assert_array_equal(q[0],source[0])
    np.testing.assert_array_equal(q[:,3:7],source[:,3:7])
    np.testing.assert_array_equal(q[:,19:],source[:,19:])
    arrays=canonical(model,q,ref['source_times_s'],ref['foot_phase'])
    arrays['stationary_mask']=ref['stationary_mask'];arrays['source_frame_ids']=ref['source_frame_ids']
    np.savez_compressed(out/'reference.npz',**arrays)
    write_json(out/'result.json',dict(source_sha256=sha256(root/'input/reference_raw.npz'),
        model_sha256=sha256(root/'input/compiled_model.mjb'),reference_sha256=sha256(out/'reference.npz'),
        code_sha256=sha256(__file__),solver=solver,trace=trace,unknown_targets=unknown,
        initial_state_preserved=True,source_clock_preserved=True,geometry_unchanged=True,
        root_change_max_m=float(np.linalg.norm(q[:,:3]-source[:,:3],axis=1).max()),
        leg_change_max_rad=float(np.abs(q[:,7:19]-source[:,7:19]).max()),
        scene_penetration_max_m=float(arrays['scene_penetration_m'].max()),
        stance_abs_gap_p95_m=float(np.quantile(np.abs(arrays['foot_min_distance_m'][arrays['contact_ref']>.5]),.95)),
        dynamics_verified=False))


if __name__=='__main__':main()
