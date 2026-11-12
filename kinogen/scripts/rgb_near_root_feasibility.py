"""Frozen-state actuator-only local reachability analysis; never rolls out plant."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.optimize import lsq_linear

p=argparse.ArgumentParser();p.add_argument('trial',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_binary_path(str(a.trial/'model.mjb'));z=np.load(a.trial/'trace.npz');b=np.load(a.trial/'root_acceleration_feedback.npz');d=mj.MjData(m)
rows=[];rv=m.jnt_dofadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'p1_vita_demo_free_root')]
for t in [.2,.5,1.,1.5]:
    i=int(np.argmin(abs(b['time_s']-t)));step=int(np.argmin(abs(z['time'][:-1]-b['time_s'][i])))
    d.qpos[:]=z['qpos'][step];d.qvel[:]=z['qvel'][step];d.eq_active[:]=z['eq_active'][step];m.eq_data[:]=z['eq_data'][step];d.time=float(z['time'][step])
    u=z['ctrl'][step].copy();ids=np.asarray([x for x in b['actuator_ids'][i]if int(x)<52],dtype=int);jids=m.actuator_trnid[ids,0];limits=m.jnt_range[jids]
    def f(ctrl):
        d.ctrl[:]=ctrl;d.qacc_warmstart[:]=0.;mj.mj_forward(m,d);return d.qacc[rv:rv+6].copy()
    baseline=f(u);jac=np.zeros((6,len(ids)))
    for k,aid in enumerate(ids):
        delta=.005 if u[aid]+.005<=limits[k,1]else-.005;v=u.copy();v[aid]+=delta;jac[:,k]=(f(v)-baseline)/delta
    goal=b['root_acceleration_desired'][i,:6];cases=[]
    for radius in [.04,.35]:
        lo=np.maximum(-radius,limits[:,0]-u[ids]);hi=np.minimum(radius,limits[:,1]-u[ids])
        for k,aid in enumerate(ids):
            name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,int(aid))or''
            if name.startswith('p1_motor_left_'):lo[k]=max(lo[k],-.05);hi[k]=min(hi[k],.05)
        fit=lsq_linear(np.vstack([jac,2.*np.eye(len(ids))]),np.r_[goal-baseline,np.zeros(len(ids))],bounds=(lo,hi),tol=1e-8,max_iter=100)
        actual=u.copy();actual[ids]+=fit.x;pred=baseline+jac@fit.x;nonlinear=f(actual)
        # Independent component extrema are upper/lower bounds, not jointly attainable vectors.
        maxacc=baseline+np.sum(np.where(jac>=0,jac*hi,jac*lo),axis=1)
        minacc=baseline+np.sum(np.where(jac>=0,jac*lo,jac*hi),axis=1)
        cases.append({'target_increment_radius_rad':radius,'independent_component_min':minacc.tolist(),'independent_component_max':maxacc.tolist(),
                      'least_squares_prediction':pred.tolist(),'nonlinear_forward_recheck':nonlinear.tolist(),'joint_increment_rad':fit.x.tolist(),
                      'linearization_error_norm':float(np.linalg.norm(pred-nonlinear))})
    rows.append({'time_s':float(z['time'][step]),'root_acceleration_layout':'linearXYZ m/s2; local angularXYZ rad/s2','baseline':baseline.tolist(),'desired':goal.tolist(),
                 'actuator_names':[mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,int(x))for x in ids],
                 'saturated_effort_names':[mj.mj_id2name(m,mj.mjtObj.mjOBJ_ACTUATOR,int(x))for x in ids if abs(z['actuator_force'][step,x])>=.98*max(abs(m.actuator_forcerange[x]))],
                 'cases':cases})
out={'trial':str(a.trial),'type':'Frozen-state local counterfactual actuator targets; not physical rollout or reachable posture guarantee','samples':rows}
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{'time':r['time_s'],'baseline_ang':r['baseline'][3:],'desired_ang':r['desired'][3:],'local_max_ang':r['cases'][0]['independent_component_max'][3:],'local_pred_ang':r['cases'][0]['least_squares_prediction'][3:],'local_actual_ang':r['cases'][0]['nonlinear_forward_recheck'][3:],'local_linearization_error':r['cases'][0]['linearization_error_norm'],'effort_saturated':r['saturated_effort_names']}for r in rows],indent=2))
