"""Kinematic gravity-load proxy and analytic Jacobian for one supporting knee.

The proximal free body has no other scene support. This is a static cost proxy,
not a dynamic feasibility test or an applied torque. No state/model is changed.

Retarget use, only for a non-seated single-support phase::

    proxy = KneeLoadProxy(model)
    residual, jacobian, info = proxy.single_stance_cost(
        data, expected_feet, seated=False, velocity_map=H,
        ratio_threshold=.8)

H maps optimizer-coordinate increments into MuJoCo generalized velocities.
The caller must refresh mj_kinematics/mj_comPos before evaluating this helper.
"""
import numpy as np


def skew(v):
    return np.array([[0.,-v[2],v[1]],[v[2],0.,-v[0]],[-v[1],v[0],0.]])


class KneeLoadProxy:
    def __init__(self,model,root_body_name='base_link'):
        import mujoco as mj
        self.mj=mj;self.model=model
        self.base=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,root_body_name)
        if self.base<0:raise ValueError('Missing robot base')
        self.robot_mass=float(model.body_subtreemass[self.base]);self.knees={}
        for side in ('left','right'):
            joint=mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,side+'_knee_joint')
            actuator=mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+side+'_knee_joint')
            if joint<0 or actuator<0:raise ValueError('Missing knee joint/actuator')
            if model.jnt_type[joint]!=int(mj.mjtJoint.mjJNT_HINGE)or model.actuator_trnid[actuator,0]!=joint:raise ValueError('Expected original hinge-knee motor mapping')
            child=int(model.jnt_bodyid[joint]);distal_mass=float(model.body_subtreemass[child]);gear=float(model.actuator_gear[actuator,0])
            if not model.actuator_forcelimited[actuator]or abs(gear)<1e-12:raise ValueError('Knee requires real bounded motor transmission')
            self.knees[side]={'joint':joint,'actuator':actuator,'child':child,'distal_mass':distal_mass,
                'proximal_mass':self.robot_mass-distal_mass,'gear':gear,
                'force_range':np.asarray(model.actuator_forcerange[actuator],float).copy(),
                'joint_static_friction':float(model.dof_frictionloss[model.jnt_dofadr[joint]])}

    def evaluate(self,data,side,ratio_threshold=.8,compute_jacobian=True):
        """Return raw gravity knee load, force ratio, excess cost and nv Jacobian.

        Signed actuator force is joint torque divided by the unchanged gear.
        The cost ignores static-friction assistance and is slightly conservative.
        A separate best-case friction-assisted diagnostic is returned.
        """
        if side not in self.knees:raise ValueError('Expected left/right supporting side')
        if ratio_threshold<0 or not np.isfinite(ratio_threshold):raise ValueError('Invalid soft load target')
        mj=self.mj;m=self.model;k=self.knees[side];j=k['joint'];child=k['child']
        axis=np.asarray(data.xaxis[j]);anchor=np.asarray(data.xanchor[j]);gravity=np.asarray(m.opt.gravity)
        r_mass=self.robot_mass*data.subtree_com[self.base]-k['distal_mass']*data.subtree_com[child]-k['proximal_mass']*anchor
        moment=np.cross(r_mass,gravity);joint_tau=float(axis@moment);actuator_force=joint_tau/k['gear']
        limit=float(k['force_range'][1]if actuator_force>=0 else-k['force_range'][0])
        if limit<=0:raise ValueError('No motor capacity in the required signed direction')
        rho=abs(actuator_force)/limit;excess=max(rho-ratio_threshold,0.)
        assisted_joint=np.sign(joint_tau)*max(abs(joint_tau)-k['joint_static_friction'],0.)
        out={'side':side,'joint_torque_Nm':joint_tau,'actuator_force':actuator_force,
            'signed_actuator_force_limit_magnitude':limit,'original_actuator_force_range':k['force_range'].copy(),
            'effort_ratio':float(rho),'ratio_threshold':float(ratio_threshold),'residual':float(excess),
            'best_case_static_friction_assisted_actuator_force':float(assisted_joint/k['gear']),
            'robot_mass_kg':self.robot_mass,'distal_mass_kg':k['distal_mass'],'proximal_mass_kg':k['proximal_mass'],
            'assumption':'One supporting foot distal to this knee; no other scene support on proximal body; zero acceleration'}
        if compute_jacobian:
            Jcom=np.zeros((3,m.nv));Jdist=np.zeros_like(Jcom);Janchor=np.zeros_like(Jcom);Jrot=np.zeros_like(Jcom)
            mj.mj_jacSubtreeCom(m,data,Jcom,self.base);mj.mj_jacSubtreeCom(m,data,Jdist,child)
            mj.mj_jac(m,data,Janchor,Jrot,anchor,child)
            dR=self.robot_mass*Jcom-k['distal_mass']*Jdist-k['proximal_mass']*Janchor
            daxis=-skew(axis)@Jrot
            Jtau=daxis.T@moment+np.cross(dR.T,gravity)@axis
            Jforce=Jtau/k['gear'];Jrho=np.sign(actuator_force)*Jforce/limit
            out.update(jacobian_joint_torque=Jtau,jacobian_actuator_force=Jforce,
                jacobian_effort_ratio=Jrho,jacobian_residual=Jrho if excess>0 else np.zeros(m.nv))
        return out

    def single_stance_cost(self,data,foot_support_expected,seated=False,velocity_map=None,ratio_threshold=.8):
        """Always return one cost row; zero it outside non-seated single support."""
        expected=np.asarray(foot_support_expected,bool)
        if expected.shape!=(2,):raise ValueError('Expected left/right stance flags')
        columns=self.model.nv if velocity_map is None else np.asarray(velocity_map).shape[1]
        if seated or np.count_nonzero(expected)!=1:return np.zeros(1),np.zeros((1,columns)),None
        side='left'if expected[0]else'right';result=self.evaluate(data,side,ratio_threshold)
        J=result['jacobian_residual'][None,:]
        if velocity_map is not None:
            H=np.asarray(velocity_map,float)
            if H.shape[0]!=self.model.nv:raise ValueError('Optimizer velocity map must have nv rows')
            J=J@H
        return np.array([result['residual']]),J,result


def validate_cli():
    import argparse,hashlib,json,sys
    from pathlib import Path
    p=argparse.ArgumentParser(description='Server-only pure-FK finite-difference validation; no integration steps')
    p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--lp-screen',type=Path);p.add_argument('--frames',nargs='+',type=int,default=list(range(120,125))+list(range(143,148)))
    p.add_argument('--epsilon',type=float,default=1e-6);p.add_argument('--output',type=Path);a=p.parse_args()
    if sys.platform!='linux':p.error('Run FK validation on the authorized server only')
    import mujoco as mj
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));z=dict(np.load(a.reference,allow_pickle=False));proxy=KneeLoadProxy(m)
    d=mj.MjData(m);plus=mj.MjData(m);minus=mj.MjData(m);rows=[]
    lp=None
    if a.lp_screen:
        rp=a.lp_screen/'screen.json';proof=json.loads(rp.read_text())
        for name,path in (('model_sha256',a.model),('reference_sha256',a.reference)):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=proof[name]:raise ValueError('LP/'+name+' mismatch')
        lp=dict(np.load(a.lp_screen/'arrays.npz',allow_pickle=False))
    def fk(data,q):data.qpos[:]=q;mj.mj_kinematics(m,data);mj.mj_comPos(m,data)
    for f in a.frames:
        expected=np.asarray(z['foot_support_expected'][f],bool)
        if expected.sum()!=1:raise ValueError('Validation frame must be single support: '+str(f))
        side='left'if expected[0]else'right';q=z['qpos'][f].copy();fk(d,q);out=proxy.evaluate(d,side)
        fd=np.zeros(m.nv);fd_res=np.zeros(m.nv)
        for col in range(m.nv):
            direction=np.zeros(m.nv);direction[col]=1.;qp=q.copy();qm=q.copy()
            mj.mj_integratePos(m,qp,direction,a.epsilon);mj.mj_integratePos(m,qm,direction,-a.epsilon)
            fk(plus,qp);fk(minus,qm)
            op=proxy.evaluate(plus,side,compute_jacobian=False);om=proxy.evaluate(minus,side,compute_jacobian=False)
            fd[col]=(op['joint_torque_Nm']-om['joint_torque_Nm'])/(2*a.epsilon)
            fd_res[col]=(op['residual']-om['residual'])/(2*a.epsilon)
        row={'frame':f,'source_time_s':float(z['source_time_s'][f]),'side':side,'proxy_joint_torque_Nm':out['joint_torque_Nm'],
            'proxy_actuator_force':out['actuator_force'],'effort_ratio':out['effort_ratio'],
            'torque_jacobian_FD_max_abs_error':float(np.max(abs(fd-out['jacobian_joint_torque']))),
            'residual_jacobian_FD_max_abs_error':float(np.max(abs(fd_res-out['jacobian_residual'])))}
        if lp is not None:
            name='motor_'+side+'_knee_joint';aid=list(map(str,lp['powered_actuator_names'])).index(name)
            value=float(lp['counterfactual_actuator_force'][f,1,aid])
            row.update(LP_effort_unbounded_actuator_force=value,
                proxy_minus_LP=out['actuator_force']-value,
                best_case_friction_assisted_proxy_minus_LP=out['best_case_static_friction_assisted_actuator_force']-value)
        rows.append(row)
    report={'schema':'rgb145.kinematic_knee_load_proxy_FK_validation.v1','mode':'pure_kinematics_no_mj_step',
        'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),
        'torque_jacobian_FD_max_abs_error':max(r['torque_jacobian_FD_max_abs_error']for r in rows),
        'residual_jacobian_FD_max_abs_error':max(r['residual_jacobian_FD_max_abs_error']for r in rows),
        'max_friction_assisted_proxy_LP_difference':max(abs(r['best_case_friction_assisted_proxy_minus_LP'])for r in rows)if lp is not None else None,
        'rows':rows,'state_or_model_changes':'Only private FK evaluation qpos buffers; no live state or physical parameters modified',
        'dynamic_feasibility_claimed':False}
    if a.output:
        if a.output.exists():raise FileExistsError(a.output)
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return 0 if report['torque_jacobian_FD_max_abs_error']<3e-5 and report['residual_jacobian_FD_max_abs_error']<3e-7 else 3


if __name__=='__main__':raise SystemExit(validate_cli())
