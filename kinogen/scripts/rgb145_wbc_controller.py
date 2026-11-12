"""Experimental floating-base inverse-dynamics QP controller for Vita.

This is a new controller, not SONIC. QP contact forces are predictions only;
only position-actuator ctrl is returned to the real simulator. The controller
never writes the live qpos/qvel, applied forces, model parameters or contacts.
Requires qpsolvers with proxqp (already available in the retarget environment).
"""
from collections import Counter,deque
import time
import numpy as np


def project_xy_to_contact_hull(points,target,inset=0.):
    """Project onto a measured contact polygon; return target and hull diagnostics.

    The polygon is in world XY, so this is a gravity-support heuristic rather
    than a frictional wrench-cone feasibility proof. Degenerate hulls use their
    segment/point. No planned or extrapolated foot contacts are introduced.
    """
    pts=np.unique(np.asarray(points,float).reshape(-1,2),axis=0)
    target=np.asarray(target,float).reshape(2)
    if not len(pts):return target.copy(),{'vertices':0,'resolved_inset_m':0.}
    def cross(o,a,b):return float(np.cross(a-o,b-o))
    ordered=sorted(map(tuple,pts));lower=[];upper=[]
    for value in ordered:
        p=np.asarray(value)
        while len(lower)>=2 and cross(lower[-2],lower[-1],p)<=1e-12:lower.pop()
        lower.append(p)
    for value in reversed(ordered):
        p=np.asarray(value)
        while len(upper)>=2 and cross(upper[-2],upper[-1],p)<=1e-12:upper.pop()
        upper.append(p)
    hull=np.asarray(lower[:-1]+upper[:-1])if len(pts)>1 else pts.copy()
    if not len(hull):hull=pts[:1]
    resolved=0.
    if len(hull)>=3 and inset>0:
        center=hull.mean(axis=0);edges=np.roll(hull,-1,axis=0)-hull
        distance=np.abs(np.cross(edges,center-hull))/np.maximum(np.linalg.norm(edges,axis=1),1e-12)
        margin=float(distance.min())
        if margin>1e-12:
            fraction=min(float(inset)/margin,.45);hull=hull+fraction*(center-hull);resolved=fraction*margin
    if len(hull)==1:projected=hull[0].copy()
    else:
        edges=np.roll(hull,-1,axis=0)-hull
        inside=len(hull)>=3 and np.all(np.cross(edges,target-hull)>=-1e-12)
        if inside:projected=target.copy()
        else:
            u=np.clip(np.sum((target-hull)*edges,axis=1)/np.maximum(np.sum(edges*edges,axis=1),1e-20),0.,1.)
            candidates=hull+u[:,None]*edges;projected=candidates[np.argmin(np.sum((candidates-target)**2,axis=1))]
    return projected,{'vertices':len(hull),'resolved_inset_m':resolved}


def limited_affine_force_interval(gain,bias,ctrl_lower,ctrl_upper,force_lower,force_upper):
    """Exact image of a ctrl interval under a force-limited affine actuator."""
    if np.any(np.asarray(gain)<=0)or np.any(np.asarray(ctrl_lower)>ctrl_upper):raise ValueError('Invalid positive-gain position-actuator interval')
    return (np.clip(gain*ctrl_lower+bias,force_lower,force_upper),
            np.clip(gain*ctrl_upper+bias,force_lower,force_upper))


def position_ctrl_for_limited_force(force,gain,bias,ctrl_lower,ctrl_upper):
    """Pick an allowed preimage; clipping is exact in a saturation plateau."""
    return np.clip((force-bias)/gain,ctrl_lower,ctrl_upper)


def implicit_velocity_damping_diagonal(model,data,powered,actuator_mode='all'):
    """Negative smooth-force velocity derivative for these joint actuators.

    This returns an algorithm vector; it never changes model.dof_damping.
    'unsaturated' linearizes the force clamp at the supplied ctrl/state;
    'all' retains every affine velocity derivative as a comparison mode.
    """
    if actuator_mode not in ('all','unsaturated','none'):raise ValueError('Invalid implicit actuator derivative mode')
    damping=np.asarray(model.dof_damping,float).copy();active=0
    if actuator_mode=='none':return damping,active
    for a in powered:
        j=int(model.actuator_trnid[a,0]);va=int(model.jnt_dofadr[j]);gear=float(model.actuator_gear[a,0])
        if actuator_mode=='unsaturated'and model.actuator_forcelimited[a]:
            raw=model.actuator_gainprm[a,0]*data.ctrl[a]+model.actuator_biasprm[a,0]+model.actuator_biasprm[a,1]*data.actuator_length[a]+model.actuator_biasprm[a,2]*data.actuator_velocity[a]
            if raw<=model.actuator_forcerange[a,0]+1e-8 or raw>=model.actuator_forcerange[a,1]-1e-8:continue
        damping[va]-=model.actuator_biasprm[a,2]*gear*gear;active+=1
    return damping,active


class WBCController:
    def __init__(self,model,reference,config=None):
        import mujoco as mj
        from qpsolvers import solve_qp
        self.mj=mj;self.solve_qp=solve_qp;self.model=model;self.config=config or {}
        self.z=dict(reference);self.times=np.asarray(self.z['time_s'],float)
        self.qref=np.asarray(self.z['qpos'],float).copy()
        self.names_all=list(map(str,self.z['dof_names']))
        self.jref=np.array([i for i,n in enumerate(self.names_all)if not n.startswith(('lh_','rh_','neck_'))])
        self.names=[self.names_all[i]for i in self.jref]
        self.jids=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n)for n in self.names])
        self.qa=model.jnt_qposadr[self.jids];self.va=model.jnt_dofadr[self.jids]
        self.aids=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in self.names])
        self.all_jids=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n)for n in self.names_all])
        self.all_qa=model.jnt_qposadr[self.all_jids];self.all_va=model.jnt_dofadr[self.all_jids]
        self.all_aids=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in self.names_all])
        self.wrist_target_key='wrist_reference_joint_targets_rad'if'wrist_reference_joint_targets_rad'in self.z else'wrist_reference_targets'
        wrist_fields=('wrist_reference_joint_names',self.wrist_target_key)
        if any(k in self.z for k in wrist_fields)and not all(k in self.z for k in wrist_fields):raise ValueError('Incomplete wrist reference metadata')
        self.wrist_cols=np.array([],dtype=int);self.wrist_targets=None
        self.wrist_window=float(self.config.get('wrist_ctrl_window_rad',.20))
        if not np.isfinite(self.wrist_window)or self.wrist_window<0:raise ValueError('Invalid wrist ctrl software window')
        if all(k in self.z for k in wrist_fields):
            self.wrist_names=list(map(str,self.z[wrist_fields[0]]))
            if len(self.wrist_names)!=6 or len(set(self.wrist_names))!=6 or any('wrist_'not in n or n not in self.names for n in self.wrist_names):raise ValueError('Expected six named Vita wrist body joints')
            self.wrist_cols=np.array([self.names.index(n)for n in self.wrist_names])
            raw=np.asarray(self.z[wrist_fields[1]],float)
            if raw.shape==(6,):raw=np.broadcast_to(raw,(len(self.times),6)).copy()
            if raw.shape!=(len(self.times),6)or not np.isfinite(raw).all():raise ValueError('Invalid wrist reference targets')
            wl=model.jnt_range[self.jids[self.wrist_cols]]
            self.wrist_targets=np.clip(raw,wl[:,0],wl[:,1])
            self.wrist_target_clipping_max=float(np.max(abs(self.wrist_targets-raw)))
            self.wrist_vel=np.gradient(self.wrist_targets,self.times,axis=0)
            self.wrist_acc=np.gradient(self.wrist_vel,self.times,axis=0)
        self.weights={
            'root_position_weight':120.,'root_orientation_weight':90.,'body_posture_weight':2.,'auxiliary_acceleration_weight':0.,
            'stance_foot_position_weight':160.,'swing_foot_position_weight':70.,
            'stance_foot_orientation_weight':20.,'swing_foot_orientation_weight':4.,
            'stance_contact_no_slip_weight':250.,'unplanned_contact_no_slip_weight':5.,
            'root_position_kp':100.,'root_position_kd':20.,'root_orientation_kp':120.,'root_orientation_kd':22.,
            'body_posture_kp':70.,'body_posture_kd':15.,'foot_position_kp':250.,'foot_position_kd':30.,
            'foot_orientation_kp':120.,'foot_orientation_kd':20.,'contact_velocity_kd':25.}
        self.weights.update({k:float(self.config[k])for k in self.weights if k in self.config})
        if any(not np.isfinite(v)or v<0 for v in self.weights.values()):raise ValueError('Invalid nonnegative WBC weight/gain')
        self.com_weight=float(self.config.get('com_position_weight',0.))
        self.com_kp=float(self.config.get('com_position_kp',60.));self.com_kd=float(self.config.get('com_position_kd',16.))
        self.com_inset=float(self.config.get('com_support_inset_m',.005))
        self.com_max_shift=float(self.config.get('com_max_reference_shift_m',.10))
        self.com_activation_delay=float(self.config.get('com_activation_delay_s',0.))
        if any(not np.isfinite(x)or x<0 for x in(self.com_weight,self.com_kp,self.com_kd,self.com_inset,self.com_max_shift,self.com_activation_delay)):raise ValueError('Invalid COM-task configuration')
        self.robot_base=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'base_link')
        if self.robot_base<0:raise ValueError('Missing robot subtree root')
        self.qp_solver=str(self.config.get('qp_solver','proxqp'))
        self.qp_solver_options=dict(self.config.get('qp_solver_options',{}))
        self.implicit_compensation=bool(self.config.get('implicit_damping_compensation',False))
        requested_mode=str(self.config.get('implicit_damping_actuator_mode','auto'))
        implicit_integrators=(int(mj.mjtIntegrator.mjINT_IMPLICIT),int(mj.mjtIntegrator.mjINT_IMPLICITFAST))
        if requested_mode=='auto':self.implicit_actuator_mode='all'if int(model.opt.integrator)in implicit_integrators else'none'
        else:self.implicit_actuator_mode=requested_mode
        if self.implicit_actuator_mode not in ('all','unsaturated','none'):raise ValueError('Invalid implicit damping actuator mode')
        if self.implicit_compensation and int(model.opt.integrator)==int(mj.mjtIntegrator.mjINT_RK4):raise ValueError('Implicit damping compensation is inapplicable to RK4')
        self.powered=np.flatnonzero(np.any(model.actuator_gear!=0,axis=1)&(model.actuator_gainprm[:,0]!=0))
        self.aux=np.array([i for i in self.powered if i not in self.aids])
        self.passive=np.array([i for i in range(model.nu)if i not in self.powered])
        if (model.nq,model.nv,model.nu)!=(59,58,52)or len(self.aids)!=28:raise ValueError('Vita ABI required')
        if np.any(model.actuator_gaintype[self.powered])or np.any(model.actuator_biastype[self.powered]!=1):raise ValueError('Require fixed gain/affine bias position actuators')
        if np.any(model.actuator_trntype[self.powered]!=int(mj.mjtTrn.mjTRN_JOINT)):raise ValueError('Require joint transmission')
        self.gear=model.actuator_gear[:,0].copy()
        self.B=np.zeros((model.nv,28))
        self.B[model.jnt_dofadr[model.actuator_trnid[self.aids,0]],np.arange(28)]=self.gear[self.aids]
        self.feet=np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in ('left','right')])
        self.foot_desc=[]
        for foot in self.feet:
            desc=set()
            for body in range(1,model.nbody):
                b=body
                while b>0 and b!=foot:b=int(model.body_parentid[b])
                if b==foot:desc.add(body)
            self.foot_desc.append(desc)
        self.support=set(g for g in range(model.ngeom)if model.geom_bodyid[g]==0 and model.geom_contype[g])
        self.work=mj.MjData(model);self.next=mj.MjData(model);self.teacher=mj.MjData(model)
        self.dqref=np.zeros((len(self.qref),model.nv))
        for i in range(1,len(self.qref)):
            if np.dot(self.qref[i,3:7],self.qref[i-1,3:7])<0:self.qref[i,3:7]*=-1
        for i in range(len(self.qref)-1):mj.mj_differentiatePos(model,self.dqref[i],self.times[i+1]-self.times[i],self.qref[i],self.qref[i+1])
        self.dqref[-1]=0.;self.ddqref=np.gradient(self.dqref,self.times,axis=0)
        self.arm_feedback=None;self.policy_dt=float(model.opt.timestep);self.physics_dt=self.policy_dt
        self.target_history=deque(maxlen=1);self.last=np.zeros(28);self.target=np.zeros(28)
        self.inference_count=0;self.status=Counter();self.diagnostics=[];self.previous_solution=None
        self.last_qp_ddq=np.full(model.nv,np.nan);self.last_qp_contact_force_world=np.zeros(3)
        self.last_algorithm_mass_diagonal=np.zeros(model.nv)

    def reset(self,data):
        self.target_history.clear();self.inference_count=0;self.previous_solution=None
        self.status.clear();self.diagnostics=[];self.last[:]=0.;self.target[:]=0.
        self.last_qp_ddq[:]=np.nan;self.last_qp_contact_force_world[:]=0.
        self.last_algorithm_mass_diagonal[:]=0.

    def sample(self,t):
        j=int(np.clip(np.searchsorted(self.times,t,side='right'),1,len(self.times)-1));i=j-1
        w=float(np.clip((t-self.times[i])/(self.times[j]-self.times[i]),0,1))
        q=(1-w)*self.qref[i]+w*self.qref[j];q[3:7]/=np.linalg.norm(q[3:7])
        v=(1-w)*self.dqref[i]+w*self.dqref[j];acc=(1-w)*self.ddqref[i]+w*self.ddqref[j]
        if self.wrist_targets is not None:
            q[self.qa[self.wrist_cols]]=(1-w)*self.wrist_targets[i]+w*self.wrist_targets[j]
            v[self.va[self.wrist_cols]]=(1-w)*self.wrist_vel[i]+w*self.wrist_vel[j]
            acc[self.va[self.wrist_cols]]=(1-w)*self.wrist_acc[i]+w*self.wrist_acc[j]
        if t>=self.times[-1]:v=np.zeros_like(v);acc=np.zeros_like(acc)
        return q,v,acc,i

    def _jac(self,data,body,pos):
        jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
        self.mj.mj_jac(self.model,data,jp,jr,np.asarray(pos,float),int(body))
        return jp,jr

    def _jac_drift(self,body,pos,J):
        d=self.work;n=self.next;R=d.xmat[body].reshape(3,3)
        local=R.T@(pos-d.xpos[body]);p2=n.xpos[body]+n.xmat[body].reshape(3,3)@local
        J2,_=self._jac(n,body,p2)
        return ((J2-J)/self.derivative_dt)@d.qvel

    def _equalities(self,data):
        m=self.model;rows=[];rhs=[]
        for eq in range(m.neq):
            if not data.eq_active[eq]:continue
            if m.eq_type[eq]!=int(self.mj.mjtEq.mjEQ_JOINT):raise ValueError('Only supplied joint couplings permitted')
            j1,j2=int(m.eq_obj1id[eq]),int(m.eq_obj2id[eq]);v1=int(m.jnt_dofadr[j1])
            q1=float(data.qpos[m.jnt_qposadr[j1]]-m.qpos0[m.jnt_qposadr[j1]])
            q2=float(data.qpos[m.jnt_qposadr[j2]]-m.qpos0[m.jnt_qposadr[j2]])if j2>=0 else 0.
            v2=float(data.qvel[m.jnt_dofadr[j2]])if j2>=0 else 0.;coef=m.eq_data[eq,:5]
            poly=sum(coef[k]*q2**k for k in range(5));der=sum(k*coef[k]*q2**(k-1)for k in range(1,5))
            der2=sum(k*(k-1)*coef[k]*q2**(k-2)for k in range(2,5))
            row=np.zeros(m.nv);row[v1]=1.
            if j2>=0:row[m.jnt_dofadr[j2]]=-der
            rows.append(row);rhs.append(der2*v2*v2-150.*(q1-poly)-25.*(row@data.qvel))
        return np.asarray(rows).reshape(-1,m.nv),np.asarray(rhs)

    def action(self,data,t):
        mj=self.mj;m=self.model;started=time.monotonic();d=self.work
        if np.any(data.qfrc_applied)or np.any(data.xfrc_applied):raise ValueError('External forces forbidden')
        # All assignments below target private planning data, never live state.
        d.qpos[:]=data.qpos;d.qvel[:]=data.qvel;d.eq_active[:]=data.eq_active;d.ctrl[:]=data.ctrl
        mj.mj_forward(m,d)
        q,v,ddq,frame=self.sample(t)
        if self.config.get('hold_reference',False):v=np.zeros_like(v);ddq=np.zeros_like(ddq)
        teacher=self.teacher;teacher.qpos[:]=q;teacher.qvel[:]=v
        mj.mj_forward(m,teacher)
        nominal=np.zeros(m.nu);nominal[self.all_aids]=q[self.all_qa];nominal[self.passive]=0.
        for a in self.powered:
            if m.actuator_ctrllimited[a]:nominal[a]=np.clip(nominal[a],*m.actuator_ctrlrange[a])
        known=np.zeros(m.nv);aux_force=[]
        for a in self.aux:
            j=int(m.actuator_trnid[a,0]);qa=int(m.jnt_qposadr[j]);va=int(m.jnt_dofadr[j])
            f=m.actuator_gainprm[a,0]*nominal[a]+m.actuator_biasprm[a,0]+m.actuator_biasprm[a,1]*d.qpos[qa]+m.actuator_biasprm[a,2]*d.qvel[va]
            if m.actuator_forcelimited[a]:f=np.clip(f,*m.actuator_forcerange[a])
            aux_force.append(float(f))
            known[va]+=self.gear[a]*f
        self.derivative_dt=1e-5;n=self.next;n.qpos[:]=d.qpos;n.qvel[:]=d.qvel
        mj.mj_integratePos(m,n.qpos,d.qvel,self.derivative_dt);mj.mj_kinematics(m,n);mj.mj_comPos(m,n)
        contacts=[]
        for contact_id,c in enumerate(d.contact[:d.ncon]):
            g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(m.geom_bodyid[g1]),int(m.geom_bodyid[g2])
            if g1 in self.support and b2>0:body=b2;sign=1.
            elif g2 in self.support and b1>0:body=b1;sign=-1.
            else:continue
            if c.dist>.003:continue
            side=next((k for k,s in enumerate(self.foot_desc)if body in s),None)
            stance=side is None or bool(self.z.get('foot_support_expected',np.ones((len(self.times),2),bool))[frame,side])
            # A source label must not delete a contact that physically exists.
            # Unplanned contacts remain unilateral force candidates, but their
            # no-slip task is weak so the swing-foot task can unload/lift them.
            if self.config.get('planned_contacts_only',False)and not stance:continue
            J,_=self._jac(d,body,c.pos);W=sign*np.asarray(c.frame).reshape(3,3).T
            contacts.append((J,W,float(c.friction[0]),body,np.asarray(c.pos).copy(),stance))
        nc=len(contacts);Je,be=self._equalities(d);ne=len(Je);nv=m.nv;nt=28
        tau_slice=slice(nv,nv+nt);lambda_start=nv+nt;eta_start=lambda_start+3*nc;nx=eta_start+ne
        P=np.eye(nx)*2e-7;linear=np.zeros(nx)
        delta=np.zeros(nv);mj.mj_differentiatePos(m,delta,1.,d.qpos,q)
        cfg=self.weights
        desired=ddq+cfg['body_posture_kp']*delta+cfg['body_posture_kd']*(v-d.qvel)
        desired[:3]=ddq[:3]+cfg['root_position_kp']*delta[:3]+cfg['root_position_kd']*(v[:3]-d.qvel[:3])
        desired[3:6]=ddq[3:6]+cfg['root_orientation_kp']*delta[3:6]+cfg['root_orientation_kd']*(v[3:6]-d.qvel[3:6])
        desired=np.clip(desired,-200.,200.)
        weight=np.ones(nv)*cfg['auxiliary_acceleration_weight'];weight[:3]=cfg['root_position_weight'];weight[3:6]=cfg['root_orientation_weight'];weight[self.va]=cfg['body_posture_weight']
        P[:nv,:nv]+=np.diag(2*weight);linear[:nv]-=2*weight*desired
        com_log=None
        if self.com_weight>0 and t>=self.com_activation_delay:
            # Only upward-facing contacts present in the actual planning state
            # define the support polygon. Source stance labels add no contacts.
            points=[pos[:2]for J,W,mu,body,pos,stance in contacts if W[2,0]>.5]
            actual=d.subtree_com[self.robot_base].copy();nominal_com=teacher.subtree_com[self.robot_base].copy()
            target_xy,hull_info=project_xy_to_contact_hull(points,nominal_com[:2],self.com_inset)
            shift=target_xy-nominal_com[:2];required=float(np.linalg.norm(shift));limited=required>self.com_max_shift
            if limited:target_xy=nominal_com[:2]+shift*(self.com_max_shift/max(required,1e-12))
            J=np.zeros((3,nv));Jt=np.zeros_like(J);Jn=np.zeros_like(J)
            mj.mj_jacSubtreeCom(m,d,J,self.robot_base);mj.mj_jacSubtreeCom(m,teacher,Jt,self.robot_base);mj.mj_jacSubtreeCom(m,n,Jn,self.robot_base)
            # Projection target changes with contact topology. Its feed-forward
            # velocity is deliberately zero when projected; otherwise use the
            # teacher COM velocity. Contact changes are logged for inspection.
            projected=required>1e-10
            velocity=np.zeros(2)if projected else(Jt@v)[:2]
            acceleration=np.zeros(2)if projected else(Jt@ddq)[:2]
            acc=acceleration+self.com_kp*(target_xy-actual[:2])+self.com_kd*(velocity-(J@d.qvel)[:2])-(((Jn-J)/self.derivative_dt)@d.qvel)[:2]
            Jxy=J[:2];P[:nv,:nv]+=2*self.com_weight*(Jxy.T@Jxy);linear[:nv]-=2*self.com_weight*Jxy.T@acc
            com_log={'actual_xy_m':actual[:2].tolist(),'teacher_xy_m':nominal_com[:2].tolist(),'target_xy_m':target_xy.tolist(),'required_reference_shift_m':required,'applied_reference_shift_m':float(np.linalg.norm(target_xy-nominal_com[:2])),'reference_shift_capped':limited,'contact_points':len(points),**hull_info}
        effort=np.maximum(abs(m.actuator_forcerange[self.aids]).max(axis=1),1.)
        P[tau_slice,tau_slice]+=np.diag(.01/effort**2)
        for k,body in enumerate(self.feet):
            J,Jr=self._jac(d,body,d.xpos[body]);Jt,Jrt=self._jac(teacher,body,teacher.xpos[body])
            stance=bool(self.z.get('foot_support_expected',np.ones((len(self.times),2),bool))[frame,k])
            acc=cfg['foot_position_kp']*(teacher.xpos[body]-d.xpos[body])+cfg['foot_position_kd']*(Jt@v-J@d.qvel)+Jt@ddq-self._jac_drift(body,d.xpos[body],J)
            w=cfg['stance_foot_position_weight']if stance else cfg['swing_foot_position_weight'];P[:nv,:nv]+=2*w*(J.T@J);linear[:nv]-=2*w*J.T@acc
            current=np.empty(4);target=np.empty(4);err=np.empty(3)
            mj.mju_mat2Quat(current,d.xmat[body]);mj.mju_mat2Quat(target,teacher.xmat[body]);mj.mju_subQuat(err,target,current)
            rot_acc=cfg['foot_orientation_kp']*(d.xmat[body].reshape(3,3)@err)+cfg['foot_orientation_kd']*(Jrt@v-Jr@d.qvel)
            wr=cfg['stance_foot_orientation_weight']if stance else cfg['swing_foot_orientation_weight'];P[:nv,:nv]+=2*wr*(Jr.T@Jr);linear[:nv]-=2*wr*Jr.T@rot_acc
        M=np.zeros((nv,nv))
        try:mj.mj_fullM(m,M,d.qM)
        except TypeError:mj.mj_fullM(m,d,M)
        active_damping_actuators=0;self.last_algorithm_mass_diagonal[:]=0.
        if self.implicit_compensation:
            damping,active_damping_actuators=implicit_velocity_damping_diagonal(m,d,self.powered,self.implicit_actuator_mode)
            self.last_algorithm_mass_diagonal=float(m.opt.timestep)*damping
            M+=np.diag(self.last_algorithm_mass_diagonal)
        A=np.zeros((nv+ne,nx));A[:nv,:nv]=M;A[:nv,tau_slice]=-self.B
        b=np.r_[known+d.qfrc_passive-d.qfrc_bias,be]
        G=[];upper=[]
        for k,(J,W,mu,body,pos,planned_stance)in enumerate(contacts):
            sl=slice(lambda_start+3*k,lambda_start+3*k+3);A[:nv,sl]=-J.T@W
            for s1,s2 in ((1,1),(1,-1),(-1,1),(-1,-1)):
                row=np.zeros(nx);row[sl]=[-mu,s1,s2];G.append(row);upper.append(0.)
            # Stance point no-slip is a soft task. A contact may separate under
            # true dynamics; no predicted QP force is applied to the simulator.
            acc=-cfg['contact_velocity_kd']*(J@d.qvel)-self._jac_drift(body,pos,J)
            w=cfg['stance_contact_no_slip_weight']if planned_stance else cfg['unplanned_contact_no_slip_weight'];P[:nv,:nv]+=2*w*J.T@J;linear[:nv]-=2*w*J.T@acc
        if ne:A[:nv,eta_start:]=-Je.T;A[nv:,:nv]=Je
        lb=np.full(nx,-np.inf);ub=np.full(nx,np.inf)
        # The auxiliary fingers have tiny inertia and can legitimately exceed
        # 500 rad/s^2 while their bounded PD and mimic constraints settle. An
        # artificial all-DOF acceleration box made the QP infeasible even when
        # the real plant and all motor force limits remained valid.
        bounded_acc=np.arange(nv)if self.config.get('all_dof_acceleration_bounds',False)else np.r_[np.arange(6),self.va]
        lb[bounded_acc]=-500.;ub[bounded_acc]=500.
        bias=m.actuator_biasprm[self.aids,0]+m.actuator_biasprm[self.aids,1]*d.qpos[self.qa]+m.actuator_biasprm[self.aids,2]*d.qvel[self.va]
        gain=m.actuator_gainprm[self.aids,0]
        effort_lo=m.actuator_forcerange[self.aids,0].copy();effort_hi=m.actuator_forcerange[self.aids,1].copy()
        ctrl_lo=np.full(nt,-np.inf);ctrl_hi=np.full(nt,np.inf)
        for k,a in enumerate(self.aids):
            if m.actuator_ctrllimited[a]:ctrl_lo[k],ctrl_hi[k]=m.actuator_ctrlrange[a]
        if self.wrist_targets is not None:
            cols=self.wrist_cols;target_wrist=q[self.qa[cols]];actual_joint_limits=m.jnt_range[self.jids[cols]]
            ctrl_lo[cols]=np.maximum(ctrl_lo[cols],np.maximum(actual_joint_limits[:,0],target_wrist-self.wrist_window))
            ctrl_hi[cols]=np.minimum(ctrl_hi[cols],np.minimum(actual_joint_limits[:,1],target_wrist+self.wrist_window))
        lo,hi=limited_affine_force_interval(gain,bias,ctrl_lo,ctrl_hi,effort_lo,effort_hi)
        lb[tau_slice]=lo;ub[tau_slice]=hi
        for k in range(nc):lb[lambda_start+3*k]=0.;ub[lambda_start+3*k]=3000.
        A_scale=np.maximum(np.linalg.norm(A,axis=1),1.);Ae=A/A_scale[:,None];beq=b/A_scale
        sol=None;status='exception'
        try:
            if self.config.get('qp_variable_scaling',False):
                # x_physical = scale*x_solver is an exact change of variables;
                # every original dynamic, friction and motor constraint remains.
                scale=np.ones(nx);scale[:nv]=1000.;scale[:3]=10.;scale[3:6]=20.;scale[self.va]=50.
                scale[tau_slice]=effort;scale[lambda_start:eta_start]=max(float(m.body_mass.sum()*9.81/max(nc,1)),10.)
                As=A*scale[None,:];ars=np.maximum(np.linalg.norm(As,axis=1),1.)
                Gs=np.asarray(G)*scale[None,:]if G else None
                if G:
                    grs=np.maximum(np.linalg.norm(Gs,axis=1),1.);Gs/=grs[:,None];hs=np.asarray(upper)/grs
                else:hs=None
                answer=self.solve_qp(P*scale[:,None]*scale[None,:],linear*scale,Gs,hs,As/ars[:,None],b/ars,lb/scale,ub/scale,solver=self.qp_solver,**self.qp_solver_options)
                sol=None if answer is None else answer*scale
            else:sol=self.solve_qp(P,linear,np.asarray(G)if G else None,np.asarray(upper)if G else None,Ae,beq,lb,ub,solver=self.qp_solver,**self.qp_solver_options)
            status='solved'if sol is not None and np.isfinite(sol).all()else'no_solution'
        except Exception as ex:status=type(ex).__name__+':'+str(ex)[:100]
        dyn_error=None
        if status=='solved':
            dyn_error=float(np.max(abs(A@sol-b)))
            if dyn_error>2e-2:status='rejected_dynamics_residual'
        ctrl=nominal.copy()
        if status=='solved':
            tau=sol[tau_slice];ctrl[self.aids]=position_ctrl_for_limited_force(tau,gain,bias,ctrl_lo,ctrl_hi)
            conversion_error=float(np.max(abs(np.clip(gain*ctrl[self.aids]+bias,effort_lo,effort_hi)-tau)))
            if conversion_error>1e-3:status='rejected_ctrl_force_conversion'
        else:conversion_error=None
        if status=='solved':
            self.previous_solution=sol
            self.last_qp_ddq=sol[:nv].copy()
            self.last_qp_contact_force_world=np.sum([c[1]@sol[lambda_start+3*k:lambda_start+3*k+3]for k,c in enumerate(contacts)],axis=0)if contacts else np.zeros(3)
        else:
            # Explicitly counted physical fallback; never claimed as a solved QP.
            ctrl[self.aids]=np.clip(nominal[self.aids],ctrl_lo,ctrl_hi)
            tau=np.clip(gain*ctrl[self.aids]+bias,effort_lo,effort_hi)
            self.last_qp_ddq=np.full(nv,np.nan);self.last_qp_contact_force_world=np.zeros(3)
        ctrl[self.passive]=0.;self.last=tau.copy();self.target=ctrl[self.aids].copy()
        self.target_history.append(self.target.copy());self.inference_count+=1;self.status[status]+=1
        auxiliary_acc_peak=None;bounded_acc_peak=None;explicit_candidate_residual=None;explicit_candidate_feasible=None
        if status!='solved':
            try:
                K=np.block([[M,-Je.T],[Je,np.zeros((ne,ne))]])
                rhs=np.r_[known+d.qfrc_passive-d.qfrc_bias+self.B@np.clip(np.zeros(nt),lo,hi),be]
                candidate=np.linalg.solve(K,rhs);free=candidate[:nv]
                auxiliary=np.setdiff1d(np.arange(nv),np.r_[np.arange(6),self.va])
                auxiliary_acc_peak=float(np.max(abs(free[auxiliary])))if len(auxiliary)else 0.
                bounded_acc_peak=float(np.max(abs(free[bounded_acc])))
                explicit_candidate_residual=float(np.max(abs(K@candidate-rhs)))
                explicit_candidate_feasible=bool(explicit_candidate_residual<1e-5 and np.all(free[bounded_acc]>=lb[bounded_acc]-1e-8)and np.all(free[bounded_acc]<=ub[bounded_acc]+1e-8))
            except np.linalg.LinAlgError:pass
        self.diagnostics.append({'time_s':float(t),'qp_status':status,'contact_points':nc,'planned_contact_points':sum(c[-1]for c in contacts),'auxiliary_pd_force_abs_peak':max(abs(f)for f in aux_force)if aux_force else 0.,'unconstrained_auxiliary_acceleration_abs_peak':auxiliary_acc_peak,'unconstrained_bounded_acceleration_abs_peak':bounded_acc_peak,'explicit_freefall_candidate_residual':explicit_candidate_residual,'explicit_freefall_candidate_feasible':explicit_candidate_feasible,'dynamics_residual_N_or_Nm_max':dyn_error,'ctrl_to_force_conversion_error':conversion_error,'wrist_ctrl_target_error_rad_max':float(np.max(abs(ctrl[self.aids[self.wrist_cols]]-q[self.qa[self.wrist_cols]])))if self.wrist_targets is not None else None,'solve_wall_s':time.monotonic()-started})
        self.diagnostics[-1].update(implicit_damping_compensation=self.implicit_compensation,
            implicit_active_affine_actuators=active_damping_actuators,
            algorithm_added_generalized_inertia_diagonal_max=float(np.max(self.last_algorithm_mass_diagonal)),
            com_task=com_log)
        return ctrl

    def report(self):
        errors=[r['dynamics_residual_N_or_Nm_max']for r in self.diagnostics if r['qp_status']=='solved']
        return {
            'controller':'Experimental new floating-base whole-body inverse-dynamics QP, not SONIC',
            'policy_dt':self.policy_dt,'physics_dt':self.physics_dt,'inference_count':self.inference_count,
            'body_joint_names':self.names,'qp_solver':'qpsolvers/'+self.qp_solver,'config':dict(self.config),
            'objective_weights_and_PD_gains':dict(self.weights),
            'optional_com_task':{'enabled':self.com_weight>0,'activation_delay_s':self.com_activation_delay,'world_xy_weight':self.com_weight,'kp':self.com_kp,'kd':self.com_kd,'requested_support_inset_m':self.com_inset,'maximum_reference_shift_m':self.com_max_shift,'target':'Teacher robot-subtree COM projected onto the inset XY hull of currently existing upward-facing static support contacts; projection capped at the declared distance','assumptions':'Gravity-support heuristic, not a frictional wrench-cone proof. Projected targets use zero velocity/acceleration feed-forward; unprojected targets use teacher COM kinematics. Root/force/model are never written.','max_required_reference_shift_m':max((r['com_task']['required_reference_shift_m']for r in self.diagnostics if r.get('com_task')),default=0.),'capped_steps':sum(bool(r['com_task']['reference_shift_capped'])for r in self.diagnostics if r.get('com_task'))},
            'implicit_damping_discrete_prediction':{
                'enabled':self.implicit_compensation,'model_integrator_code':int(self.model.opt.integrator),
                'resolved_affine_actuator_derivative_mode':self.implicit_actuator_mode,
                'formula':'M_algorithm = original_mj_fullM + dt*diag(model.dof_damping - affine_bias_velocity*gear_squared); only the local QP matrix changes',
                'force_variable':'Actual current-time actuator force, bounded by unchanged real force limits and wrist ctrl window',
                'saturation_linearization':'all includes every affine derivative; unsaturated uses the current planning state and previous applied ctrl, so transitions are approximate',
                'physical_model_parameters_modified':False,
                'residual_interpretation':'discrete implicit velocity increment equation'if self.implicit_compensation else'continuous inverse dynamics equation'},
            'wrist_ctrl_software_window':None if self.wrist_targets is None else {
                'joint_names':list(self.wrist_names),'half_width_rad':self.wrist_window,
                'reference_target_clipping_to_true_joint_limits_rad_max':self.wrist_target_clipping_max,
                'target_source':'reference.wrist_reference_joint_names / '+self.wrist_target_key,
                'enforcement':'Software ctrl interval, intersected with actual joint/ctrl limits and mapped through the exact saturated affine force law before solving the QP; no model effort or joint limits changed',
                'maximum_logged_ctrl_deviation_rad':max(r['wrist_ctrl_target_error_rad_max']for r in self.diagnostics)if self.diagnostics else None},
            'acceleration_box':'all DOFs'if self.config.get('all_dof_acceleration_bounds',False)else'free root and28 body DOFs only; auxiliary fingers/neck retain true motor limits without artificial ddq bounds',
            'qp_status_counts':dict(self.status),'max_solved_dynamics_residual_N_or_Nm':max(errors)if errors else None,
            'mean_solve_wall_s':float(np.mean([r['solve_wall_s']for r in self.diagnostics]))if self.diagnostics else None,
            'live_root_state_writes':0,'external_wrenches':False,'predicted_contact_forces_applied':False,
            'privileged_simulator_observations':True,
            'limitations':['Experimental controller requiring independent physical validation.',
                'Contact forces in the QP are predictions; measured forces must come from MuJoCo after stepping.',
                'Friction uses an inscribed four-facet pyramid; current active contacts define support.',
                'No-slip contact acceleration is a soft objective; no hidden kinematic attachment exists.',
                'The PD-force conversion is exact at the sampled state; implicit integration may alter within-step torque.']}
