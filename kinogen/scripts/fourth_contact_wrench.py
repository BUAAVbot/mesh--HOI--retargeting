"""Pure static contact-wrench LP utilities; never mutate a model or state.

Inputs are hypothetical/actual contact patches explicitly supplied by callers.
Normal caps are engineering budgets, not hardware ratings. Joint projection P
maps independent motor coordinates into all native velocity coordinates.
"""
import numpy as np
import mujoco as mj
from scipy.optimize import linprog


def pyramid_rays(dim,friction):
    """Native MuJoCo pyramidal basis, verified against real efc_force decode."""
    if dim==1:return[np.asarray([1.,0,0,0,0,0])]
    result=[]
    for k in range(dim-1):
        for sign in [1.,-1.]:
            ray=np.zeros(6);ray[0]=1.;ray[k+1]=sign*float(friction[k]);result.append(ray)
    return result


def contact_columns(model,data,object_body_id,digit_prefixes=('rh_thumb','rh_index'),projection=None,max_gap_m=0.):
    """Use data.contact; caller must obtain any prospective gap contacts privately.

    Returns columns/normal_rows lists suitable for wrench_lp(). Contact frame
    and material parameters come from MuJoCo's collision result, not guesses.
    projection (nv,nmotor) yields required contact load on independent motors.
    """
    columns=[];normal_rows=[];motor=[];patches=[]
    for c in data.contact[:data.ncon]:
        if c.dist>max_gap_m:continue
        g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(model.geom_bodyid[g1]),int(model.geom_bodyid[g2])
        if object_body_id not in [b1,b2]:continue
        other=b2 if b1==object_body_id else b1;name=model.body(other).name or''
        digit=next((i for i,prefix in enumerate(digit_prefixes)if name.startswith(prefix)),None)
        if digit is None:continue
        sign=1. if b2==object_body_id else-1.;frame=np.asarray(c.frame).reshape(3,3).T;point=np.asarray(c.pos)
        jac=np.zeros((3,model.nv));jacr=np.zeros_like(jac)
        if projection is not None:mj.mj_jac(model,data,jac,jacr,point,other)
        for ray in pyramid_rays(int(c.dim),c.friction):
            force=sign*(frame@ray[:3]);pure_torque=sign*(frame@ray[3:]);torque=pure_torque+np.cross(point-data.xipos[object_body_id],force)
            columns.append(np.r_[force,torque]);normal=np.zeros(len(digit_prefixes));normal[digit]=ray[0];normal_rows.append(normal)
            if projection is not None:motor.append(projection.T@(-jac.T@force-jacr.T@pure_torque))
        patches.append({'digit':digit,'point_world_m':point.tolist(),'normal_on_object':(sign*frame[:,0]).tolist(),
                        'dim':int(c.dim),'friction':c.friction.tolist(),'gap_m':float(c.dist)})
    return {'columns':columns,'normal_rows':normal_rows,'motor_torque_matrix':None if projection is None else np.asarray(motor).T,'patches':patches}


def wrench_lp(columns,normal_rows,gravity_wrench,normal_cap_N=None,motor_torque_matrix=None,motor_gravity_bias_Nm=None,motor_cap_Nm=.4):
    """Full gravity+wrench feasibility and minimum required normal budget.

    motor_torque_matrix maps ray weights to generalized contact torque on the
    robot; required actuator torque is gravity_bias - contact_torque.
    residual_force_equivalent_L1 is a continuous candidate-search score.
    """
    if not columns:return {'feasible':False,'reason':'No supplied contacts','residual_force_equivalent_L1':float(np.linalg.norm(gravity_wrench[:3]))}
    A=np.asarray(columns,float).T;N=np.asarray(normal_rows,float).T;K=A.shape[1];G=N.shape[0]
    external=np.asarray(gravity_wrench,float);scale=np.asarray([1,1,1,10,10,10.]);ineq=[];bounds=[]
    if normal_cap_N is not None:
        ineq.extend(N);bounds.extend(np.broadcast_to(normal_cap_N,(G,)))
    if motor_torque_matrix is not None:
        M=np.asarray(motor_torque_matrix,float);bias=np.asarray(motor_gravity_bias_Nm,float);caps=np.broadcast_to(motor_cap_Nm,bias.shape)
        ineq.extend(M);bounds.extend(bias+caps);ineq.extend(-M);bounds.extend(caps-bias)
    U=np.asarray(ineq)if ineq else None;B=np.asarray(bounds)if bounds else None
    fixed=linprog(N.sum(0),A_ub=U,b_ub=B,A_eq=scale[:,None]*A,b_eq=-scale*external,bounds=(0,None),method='highs')
    # Minimax per-digit normal budget, preserving any motor constraints.
    mmU=[np.r_[row,-1.]for row in N];mmB=[0.]*G
    if motor_torque_matrix is not None:
        mmU.extend(np.column_stack([M,np.zeros(M.shape[0])]))
        mmB.extend(bias+caps);mmU.extend(np.column_stack([-M,np.zeros(M.shape[0])]))
        mmB.extend(caps-bias)
    minimax=linprog(np.r_[np.zeros(K),1.],A_ub=np.asarray(mmU),b_ub=np.asarray(mmB),A_eq=np.column_stack([scale[:,None]*A,np.zeros(6)]),b_eq=-scale*external,bounds=(0,None),method='highs')
    slackA=np.column_stack([scale[:,None]*A,np.eye(6),-np.eye(6)])
    slackU=None if U is None else np.column_stack([U,np.zeros((len(U),12))])
    slack=linprog(np.r_[np.zeros(K),np.ones(12)],A_ub=slackU,b_ub=B,A_eq=slackA,b_eq=-scale*external,bounds=(0,None),method='highs')
    result={'feasible':bool(fixed.success),'solver_status':int(fixed.status),'normal_cap_N':normal_cap_N,
            'minimum_maximum_digit_normal_N':float(minimax.x[-1])if minimax.success else None,
            'unlimited_normal_budget_full_wrench_feasible':bool(minimax.success),
            'residual_force_equivalent_L1':float(slack.fun)if slack.success else float('inf'),
            'wrench_matrix_rank':int(np.linalg.matrix_rank(A))}
    if fixed.success:
        result['normal_per_digit_N']=(N@fixed.x).tolist();result['equilibrium_max_error']=float(np.max(abs(A@fixed.x+external)))
        if motor_torque_matrix is not None:result['required_motor_torque_Nm']=(bias-M@fixed.x).tolist()
    return result


def two_point_gravity_axis_certificate(points_world,com_world,mass_kg,gravity_world):
    """No point force can supply torque about the line through its two points."""
    points=np.asarray(points_world,float)
    if len(points)!=2:return None
    axis=points[1]-points[0];length=float(np.linalg.norm(axis))
    if length<1e-9:return None
    axis/=length;moment=np.cross(np.asarray(com_world)-points[0],mass_kg*np.asarray(gravity_world))
    return {'contact_line_axis_world':axis.tolist(),'contact_separation_m':length,
            'gravity_torque_about_contact_line_Nm':float(axis@moment),
            'absolute_unbalanced_axis_torque_Nm':float(abs(axis@moment)),
            'scope':'Certificate applies to exactly two point-force contacts without contact torques (condim3).'}
