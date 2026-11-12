"""Bounded actual-plant arm IK and contact acquisition; actuator commands only."""
from __future__ import annotations
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from scipy.optimize import lsq_linear


class RootAccelerationFeedback:
    """Contact-model finite-difference attitude feedback through real joint motors.

    mj_forward evaluations use private copies of the current plant state. Only
    bounded position-actuator target offsets are returned to the actual plant.
    """
    def __init__(self,model,reference,parts,objective='root',preserve_worn_leg=False,support_arm_stabilizer=False):
        self.model=model;self.parts=parts;self.times=np.asarray(reference['time_s'])
        self.source_times=np.asarray(reference.get('source_time_s',self.times))
        self.objective=objective
        self.support_arm_stabilizer=support_arm_stabilizer
        planted=np.asarray(reference.get('support_phase_planted_inferred',np.ones((len(self.times),2),bool)))
        self.right_support_only=planted[:,1]&~planted[:,0]
        edge=np.diff(np.r_[False,self.right_support_only,False].astype(int));self.right_support_intervals=list(zip(self.times[np.flatnonzero(edge==1)],np.r_[self.times,self.times[-1]+.04][np.flatnonzero(edge==-1)]))
        self.preserve_worn_leg=preserve_worn_leg;self.phases=np.asarray(reference.get('shoe_attachment_name',np.full(len(self.times),''))).astype(str)
        self.scratch=mj.MjData(model);self.next_update=0.;self.records=[];self.maps=[]
        self.desired=mj.MjData(model);self.jacfoot=[];self.feet=[]
        for part in parts:
            sm=part['sm'];names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,int(j))or''for j in sm['jids']]
            keep=np.asarray([any(x in n for x in ('_hip_','_knee_','_ankle_','waist_'))and'yaw'not in n for n in names])
            if support_arm_stabilizer and part['prefix']=='p1_':
                keep|=np.asarray([n.startswith(('p1_left_shoulder','p1_left_elbow'))for n in names])
            self.maps.append({'aids':sm['aids'][keep],'jids':sm['jids'][keep],'offset':np.zeros(int(np.sum(keep)))})
            self.feet.extend([mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,part['prefix']+side+'_ankle_roll_link')for side in ['left','right']])
        self.previous=np.zeros(model.nu)

    def forward(self,data,ctrl):
        s=self.scratch;s.qpos[:]=data.qpos;s.qvel[:]=data.qvel;s.act[:]=data.act
        s.eq_active[:]=data.eq_active;s.qacc_warmstart[:]=data.qacc_warmstart;s.time=data.time;s.ctrl[:]=ctrl
        mj.mj_forward(self.model,s)
        root=[s.qacc[p['sm']['rv']:p['sm']['rv']+6].copy()for p in self.parts]
        feet=[jac@s.qacc for jac in self.jacfoot]if self.objective=='root_feet'else[]
        return np.concatenate(root+feet)

    def apply(self,data,ctrl,t):
        output=ctrl.copy()
        if t+1e-9>=self.next_update:
            self.desired.qpos[:]=data.qpos
            for part in self.parts:
                q=np.asarray([np.interp(min(t,self.times[-1]),self.times,part['reference'][:,j])for j in range(59)])
                sm=part['sm'];self.desired.qpos[sm['rq']:sm['rq']+7]=q[:7];self.desired.qpos[sm['qa']]=q[7:]
            mj.mj_kinematics(self.model,self.desired);mj.mj_comPos(self.model,self.desired)
            self.jacfoot=[]
            for body in self.feet:
                jac=np.zeros((3,self.model.nv));mj.mj_jacBody(self.model,data,jac,None,body);self.jacfoot.append(jac)
            aids=np.concatenate([x['aids']for x in self.maps]);offset=np.concatenate([x['offset']for x in self.maps])
            jids=np.concatenate([x['jids']for x in self.maps]);limits=self.model.jnt_range[jids]
            offset=np.clip(ctrl[aids]+offset,limits[:,0],limits[:,1])-ctrl[aids]
            nominal=ctrl.copy();nominal[aids]+=offset;baseline=self.forward(data,nominal)
            jac=np.zeros((24 if self.objective=='root_feet'else 12,len(aids)));eps=.01
            for k,aid in enumerate(aids):
                delta=eps if nominal[aid]+eps<=limits[k,1]else-eps
                candidate=nominal.copy();candidate[aid]+=delta
                jac[:,k]=(self.forward(data,candidate)-baseline)/delta
            goal=[];errors=[];vel=[]
            for part in self.parts:
                sm=part['sm'];q=np.asarray([np.interp(min(t,self.times[-1]),self.times,part['reference'][:,j])for j in range(3,7)])
                q/=np.linalg.norm(q);r=np.empty(9);mj.mju_quat2Mat(r,q)
                rotation=data.xmat[sm['base']].reshape(3,3)
                error=Rotation.from_matrix(rotation.T@r.reshape(3,3)).as_rotvec()
                omega=data.qvel[sm['rv']+3:sm['rv']+6]
                targetpos=np.asarray([np.interp(min(t,self.times[-1]),self.times,part['reference'][:,j])for j in range(3)])
                targetacc=np.clip(40.*(targetpos-data.qpos[sm['rq']:sm['rq']+3])-12.*data.qvel[sm['rv']:sm['rv']+3],-10.,10.)
                goal.extend(targetacc);goal.extend(np.clip(80.*error-16.*omega,-40.,40.));errors.append(error);vel.append(omega.copy())
            if self.objective=='root_feet':
                for body,jp in zip(self.feet,self.jacfoot):
                    goal.extend(np.clip(80.*(self.desired.xpos[body]-data.xpos[body])-16.*(jp@data.qvel),-20.,20.))
            goal=np.asarray(goal)
            # Tikhonov damping prevents high-gain commands at weak or changing contacts.
            weight=np.r_[np.tile([10.,10.,5.,10.,10.,10.],2),np.ones(12)]if self.objective=='root_feet'else np.tile([2.,2.,1.,1.,1.,1.],2)
            lower=np.maximum(np.maximum(-.35,offset-.04),limits[:,0]-ctrl[aids])
            upper=np.minimum(np.minimum(.35,offset+.04),limits[:,1]-ctrl[aids])
            if self.preserve_worn_leg:
                wear=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_EQUALITY,'shoe_wear_left')
                if wear>=0 and data.eq_active[wear]:
                    for k,aid in enumerate(aids):
                        name=mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_ACTUATOR,int(aid))or''
                        if name.startswith('p1_motor_left_')and any(x in name for x in ('hip','knee','ankle')):
                            lower[k]=max(lower[k],-.05);upper[k]=min(upper[k],.05)
            if self.support_arm_stabilizer:
                priority=0.
                for start,end in self.right_support_intervals:
                    if start-.5<=t<=end+.5:
                        u=min(np.clip((t-start+.5)/.5,0,1),np.clip((end+.5-t)/.5,0,1));priority=max(priority,float(u*u*(3-2*u)))
                source_t=float(np.interp(t,self.times,self.source_times))
                arm_u=float(np.clip((source_t-7.5)/.5,0,1));arm_radius=max(1e-9,.35*arm_u*arm_u*(3-2*arm_u))
                for k,aid in enumerate(aids):
                    name=mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_ACTUATOR,int(aid))or''
                    if name.startswith('p1_motor_right_')and any(x in name for x in ('hip','knee','ankle')):
                        radius=.35-.30*priority;lower[k]=max(lower[k],-radius);upper[k]=min(upper[k],radius)
                    if name.startswith(('p1_motor_left_shoulder','p1_motor_left_elbow')):
                        lower[k]=max(lower[k],-arm_radius);upper[k]=min(upper[k],arm_radius)
            new=offset.copy();cursor=0;scales=[];cross_peak=0.
            for actor,chain in enumerate(self.maps):
                count=len(chain['aids']);cols=np.arange(cursor,cursor+count);cursor+=count
                rows=np.arange(actor*6,actor*6+6)
                if self.objective=='root_feet':rows=np.r_[rows,np.arange(12+actor*6,18+actor*6)]
                other=np.setdiff1d(np.arange(jac.shape[0]),rows)
                cross_peak=max(cross_peak,float(np.max(abs(jac[np.ix_(other,cols)]))))
                block=jac[np.ix_(rows,cols)];w=weight[rows];wb=w[:,None]*block
                fit=lsq_linear(np.vstack([wb,2.*np.eye(count)]),
                               np.r_[w*(goal[rows]-baseline[rows]+block@offset[cols]),np.zeros(count)],
                               bounds=(lower[cols],upper[cols]),tol=1e-7,max_iter=80)
                bestcost=np.inf;bestalpha=0.;best=offset[cols].copy()
                anchor=np.clip(offset[cols],lower[cols],upper[cols])
                for alpha in [1.,.5,.25,0.]:
                    candidate=nominal.copy();candidate[aids[cols]]=ctrl[aids[cols]]+anchor+alpha*(fit.x-anchor)
                    acceleration=self.forward(data,candidate)[rows]
                    test=candidate[aids[cols]]-ctrl[aids[cols]]
                    cost=float(np.sum((w*(goal[rows]-acceleration))**2)+4.*np.sum(test**2))
                    if cost<bestcost:bestcost=cost;bestalpha=alpha;best=test
                new[cols]=best;scales.append(bestalpha)
                # A robot's actuator allocation must not use unrelated actor
                # acceleration noise from the global contact solver.
                jac[np.ix_(other,cols)]=0.
            validated=ctrl.copy();validated[aids]+=new;nonlinear=self.forward(data,validated)
            cursor=0
            for m in self.maps:
                count=len(m['aids']);m['offset']=new[cursor:cursor+count];cursor+=count
            self.records.append({'time_s':float(t),'root_error_local_rad':np.asarray(errors),'root_angular_velocity_local_rad_s':np.asarray(vel),
                                 'root_acceleration_desired':goal,'root_acceleration_nominal':baseline,
                                 'root_acceleration_linearized':baseline+jac@(new-offset),
                                 'root_acceleration_forward_validated':nonlinear,'line_search_scale':np.asarray(scales),
                                 'ignored_cross_actor_sensitivity_peak':cross_peak,
                                 'target_offset_rad':new.copy(),'actuator_ids':aids.copy(),'sensitivity_per_target_rad':jac.copy()})
            self.next_update+=.02
        for m in self.maps:
            ids=m['aids'];old=self.previous[ids];new=np.clip(m['offset'],old-2.*self.model.opt.timestep,old+2.*self.model.opt.timestep)
            self.previous[ids]=new;limits=self.model.jnt_range[m['jids']]
            output[ids]=np.clip(output[ids]+new,limits[:,0],limits[:,1])
        return output

    def report(self):
        return {'type':'Contact-model finite-difference 6D root acceleration DLS via hip/knee/ankle/waist motor targets',
                'objective':self.objective,
                'preserve_worn_leg_offset_bound_rad':.05 if self.preserve_worn_leg else None,
                'support_arm_stabilizer':{'enabled':self.support_arm_stabilizer,'non_task_arm':'p1 left shoulder3+elbow','arm_offset_bound_rad':.35,'arm_blend_interval_source_s':[7.5,8.0],'right_stance_offset_bound_rad':.05,'stance_lookahead_execution_s':.5},
                'attitude_kp_s2':80.,'angular_velocity_kd_s':16.,'desired_acceleration_bound_rad_s2':40.,
                'position_kp_s2':40.,'linear_velocity_kd_s':12.,'desired_linear_acceleration_bound_m_s2':10.,
                'root_acceleration_layout':'Two actors [linear XYZ m/s2, local angular XYZ rad/s2]'+('; then four foot XYZ m/s2'if self.objective=='root_feet'else''),
                'foot_position_kp_s2':80.,'foot_velocity_kd_s':16.,'foot_acceleration_bound_m_s2':20.,
                'least_squares_priority_weights':'Root linear XY10 Z5/angularXYZ10; footXYZ1; absolute offset regularizer2'if self.objective=='root_feet'else'Root linear XY2 Z1/angularXYZ1; absolute offset regularizer2',
                'target_offset_bound_rad':.35,'target_offset_speed_bound_rad_s':2.,'control_dt_s':.02,
                'joint_target_limits':'Included in finite-difference nominal, directional perturbations, and least-squares box bounds; no impossible unclipped private commands.',
                'dls_damping':4.,'finite_difference_target_rad':.01,'calls':len(self.records),
                'actor_allocation':'Independent bounded least squares per actor; unrelated cross-actor solver sensitivities ignored and their peak logged.',
                'nonlinear_candidate_verification':'Shared private-plant mj_forward at scales 1,.5,.25,0; each actor selects least actual root acceleration error plus offset regularization.',
                'active_state_writes':0,'external_force_writes':0,'base_welds':0}


class WorldLegFeedback:
    """Track moving source 6D feet through joint targets under actual free-root pose.

    Neither root nor source swing-foot targets are pinned to the world. A foot
    which lifts in the reference follows its declared moving source target.
    """
    def __init__(self,model,reference,parts,gravity_comp=0.):
        self.model=model;self.ref=reference;self.parts=parts;self.times=np.asarray(reference['time_s'])
        self.gravity_comp=gravity_comp;self.kin=mj.MjData(model);self.desired=mj.MjData(model)
        self.jac=np.zeros((3,model.nv));self.jacr=np.zeros((3,model.nv));self.chains=[]
        for actor,part in enumerate(parts):
            prefix=part['prefix']
            for side in ['left','right']:
                js=[j for j in range(model.njnt)if(mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j)or'').startswith(tuple(prefix+side+'_'+token for token in ['hip','knee','ankle']))]
                names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j)for j in js]
                aids=np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,prefix+'motor_'+name[len(prefix):])for name in names])
                if len(js)!=6:raise ValueError('Expected six-joint leg chain')
                self.chains.append({'actor':actor,'side':side,'qa':model.jnt_qposadr[js],'va':model.jnt_dofadr[js],'aids':aids,
                                    'lower':model.jnt_range[js,0],'upper':model.jnt_range[js,1],
                                    'body':mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+side+'_ankle_roll_link')})
        self.last=[None]*4;self.previous_output=[None]*4;self.next_update=0.;self.targets=None;self.records=[]

    def ramp(self,data,ctrl):
        out=ctrl.copy()
        if self.targets is None:return out
        for i,(chain,target)in enumerate(zip(self.chains,self.targets)):
            previous=data.qpos[chain['qa']]if self.previous_output[i]is None else self.previous_output[i]
            command=np.clip(target,previous-6.*self.model.opt.timestep,previous+6.*self.model.opt.timestep)
            self.previous_output[i]=command.copy();out[chain['aids']]=command
        return out

    def apply(self,data,ctrl,t):
        output=ctrl.copy()
        if t+1e-9<self.next_update:
            return self.ramp(data,output)
        self.desired.qpos[:]=data.qpos
        for actor,part in enumerate(self.parts):
            q=np.asarray([np.interp(min(t,self.times[-1]),self.times,part['reference'][:,j])for j in range(59)])
            sm=part['sm'];self.desired.qpos[sm['rq']:sm['rq']+7]=q[:7];self.desired.qpos[sm['qa']]=q[7:]
        mj.mj_kinematics(self.model,self.desired);mj.mj_comPos(self.model,self.desired)
        self.targets=[]
        for ci,chain in enumerate(self.chains):
            goal=self.desired.xpos[chain['body']].copy();rotation=self.desired.xmat[chain['body']].reshape(3,3).copy()
            kp=self.model.actuator_gainprm[chain['aids'],0]
            bias=np.clip(self.gravity_comp*data.qfrc_bias[chain['va']]/np.maximum(kp,1e-9),-.2,.2)
            seed=np.asarray(ctrl[chain['aids']]-bias).copy()if self.last[ci]is None else self.last[ci].copy()
            lo=np.maximum(chain['lower']+1e-5,seed-.12);hi=np.minimum(chain['upper']-1e-5,seed+.12)
            candidate=np.clip(seed,lo,hi);self.kin.qpos[:]=data.qpos
            for _ in range(12):
                self.kin.qpos[chain['qa']]=candidate;mj.mj_kinematics(self.model,self.kin);mj.mj_comPos(self.model,self.kin)
                pos=self.kin.xpos[chain['body']];r=self.kin.xmat[chain['body']].reshape(3,3)
                mj.mj_jac(self.model,self.kin,self.jac,self.jacr,pos,chain['body'])
                jac=np.vstack([self.jac[:,chain['va']],.08*self.jacr[:,chain['va']]])
                error=np.r_[goal-pos,.08*Rotation.from_matrix(rotation@r.T).as_rotvec()]
                delta=np.linalg.solve(jac.T@jac+np.eye(6)*.0001,jac.T@error)
                update=np.clip(candidate+np.clip(delta,-.12,.12),lo,hi)
                if np.linalg.norm(update-candidate)<1e-5:break
                candidate=update
            self.last[ci]=candidate.copy();command=np.clip(candidate+bias,chain['lower'],chain['upper'])
            self.targets.append(command);output[chain['aids']]=command
            self.records.append({'time_s':float(t),'actor':chain['actor'],'side':chain['side'],'target_position_world_m':goal,
                                 'solved_position_error_m':float(np.linalg.norm(goal-self.kin.xpos[chain['body']]))})
        self.next_update+=.02
        return self.ramp(data,output)

    def report(self):
        return {'type':'Actual free-root DLS moving source foot-pose joint-target feedback',
                'control_dt_s':.02,'max_joint_command_speed_rad_s':6.,'gravity_compensation_fraction':self.gravity_comp,
                'source_swing_foot_policy':'Follows declared moving reference pose; no static floor projection or world-root weld.',
                'calls':len(self.records)//4,'median_solved_position_error_m':float(np.median([x['solved_position_error_m']for x in self.records]))if self.records else None,
                'active_state_writes':0,'external_force_writes':0}


class WorldArmFeedback:
    def __init__(self,model,reference,parts,gravity_comp=0.):
        self.model=model;self.ref=reference;self.parts=parts;self.times=np.asarray(reference['time_s'])
        self.kin=mj.MjData(model);self.desired=mj.MjData(model);self.chains=[]
        self.jac=np.zeros((3,model.nv));self.jacr=np.zeros((3,model.nv))
        self.scene=[g for g in range(model.ngeom)if model.geom_contype[g]and any(t in(mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or'')for t in ['bench_','ground','floor'])]
        self.shoe=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'shoe')
        self.gravity_comp=gravity_comp
        for actor,part in enumerate(parts):
            prefix=part['prefix']
            for side,short in [('left','lh_'),('right','rh_')]:
                js=[j for j in range(model.njnt)if(mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j)or'').startswith(tuple(prefix+side+'_'+x for x in ['shoulder','elbow','wrist']))]
                names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j)for j in js]
                bids=[b for b in range(model.nbody)if(mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,b)or'').startswith((prefix+short,prefix+side+'_wrist'))]
                geoms=[g for g in range(model.ngeom)if int(model.geom_bodyid[g])in bids and model.geom_contype[g]]
                self.chains.append({'actor':actor,'side':side,'qa':model.jnt_qposadr[js],'va':model.jnt_dofadr[js],
                                    'aids':np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,prefix+'motor_'+n[len(prefix):])for n in names]),
                                    'lower':model.jnt_range[js,0],'upper':model.jnt_range[js,1],
                                    'body':mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+short+'hand_base_link'),'geoms':geoms})
                if len(js)!=7:raise ValueError('Expected seven-joint arm feedback chain')
        phases=np.asarray(reference.get('shoe_attachment_name',np.full(len(self.times),''))).astype(str)
        ix=np.flatnonzero(phases=='shoe_grasp_right');self.grasp_frame=int(ix[0])if len(ix)else len(self.times)-1
        self.grasp_time=float(self.times[self.grasp_frame]);self.grasp_rel=None
        self.last=[None]*4;self.previous_output=[None]*4;self.next_update=0.;self.targets=None;self.records=[]

    def ramp(self,data,ctrl):
        out=ctrl.copy()
        if self.targets is None:return out
        for i,(chain,target)in enumerate(zip(self.chains,self.targets)):
            previous=data.qpos[chain['qa']]if self.previous_output[i]is None else self.previous_output[i]
            command=np.clip(target,previous-4.*self.model.opt.timestep,previous+4.*self.model.opt.timestep)
            self.previous_output[i]=command.copy();out[chain['aids']]=command
        return out

    def set_reference(self,data,t):
        self.desired.qpos[:]=data.qpos
        for actor,part in enumerate(self.parts):
            q=np.asarray([np.interp(t,self.times,part['reference'][:,j])for j in range(59)])
            sm=part['sm'];self.desired.qpos[sm['rq']:sm['rq']+7]=q[:7];self.desired.qpos[sm['qa']]=q[7:]
        mj.mj_kinematics(self.model,self.desired);mj.mj_comPos(self.model,self.desired)

    def apply(self,data,ctrl,t):
        output=ctrl.copy()
        if t+1e-9<self.next_update:
            return self.ramp(data,output)
        if self.grasp_rel is None:
            self.set_reference(data,self.grasp_time);chain=self.chains[1];b=chain['body'];Rh=self.desired.xmat[b].reshape(3,3)
            pose=np.asarray(self.ref['object_target_pose'][self.grasp_frame]);quat=pose[3:]/np.linalg.norm(pose[3:]);mat=np.empty(9);mj.mju_quat2Mat(mat,quat)
            self.grasp_rel=(Rh.T@(pose[:3]-self.desired.xpos[b]),Rh.T@mat.reshape(3,3))
        self.set_reference(data,min(float(t),self.times[-1]));self.targets=[]
        actual_grasp=any(data.eq_active[i]and'shoe_grasp_'in(mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_EQUALITY,i)or'')for i in range(self.model.neq))
        for ci,chain in enumerate(self.chains):
            body=chain['body'];goal=self.desired.xpos[body].copy();rotation=self.desired.xmat[body].reshape(3,3).copy();acquire=False
            if chain['actor']==0 and chain['side']=='right' and t>=self.grasp_time-1. and not actual_grasp:
                shoe_r=data.xmat[self.shoe].reshape(3,3);rotation_acquire=shoe_r@self.grasp_rel[1].T
                goal_acquire=data.xpos[self.shoe]-rotation_acquire@self.grasp_rel[0]
                blend=float(np.clip((t-(self.grasp_time-1.))/.5,0,1));goal=(1-blend)*goal+blend*goal_acquire
                rotation=rotation_acquire;acquire=True
            kp=self.model.actuator_gainprm[chain['aids'],0]
            bias=np.clip(self.gravity_comp*data.qfrc_bias[chain['va']]/np.maximum(kp,1e-9),-.2,.2)
            seed=np.asarray(ctrl[chain['aids']]-bias).copy()if self.last[ci]is None else self.last[ci].copy()
            lo=np.maximum(chain['lower']+1e-5,seed-.08);hi=np.minimum(chain['upper']-1e-5,seed+.08)
            candidate=np.clip(seed,lo,hi);self.kin.qpos[:]=data.qpos
            for _ in range(10):
                self.kin.qpos[chain['qa']]=candidate;mj.mj_kinematics(self.model,self.kin);mj.mj_comPos(self.model,self.kin)
                pos=self.kin.xpos[body];r=self.kin.xmat[body].reshape(3,3);mj.mj_jac(self.model,self.kin,self.jac,self.jacr,pos,body)
                jac=np.vstack([self.jac[:,chain['va']],.06*self.jacr[:,chain['va']]])
                error=np.r_[goal-pos,.06*Rotation.from_matrix(rotation@r.T).as_rotvec()]
                rows=[];rhs=[]
                for g in chain['geoms']:
                    best=(.1,None)
                    for scene in self.scene:
                        witness=np.zeros(6);distance=float(mj.mj_geomDistance(self.model,self.kin,g,scene,.1,witness))
                        if distance<best[0]and np.isfinite(witness).all()and abs(np.linalg.norm(witness[:3]-witness[3:])-abs(distance))<1e-4:best=(distance,witness)
                    distance,witness=best
                    if witness is None or distance>=.004:continue
                    normal=(witness[:3]-witness[3:])/distance if abs(distance)>1e-8 else np.array([0.,0.,1.])
                    normal/=max(np.linalg.norm(normal),1e-9);mj.mj_jac(self.model,self.kin,self.jac,None,witness[:3],int(self.model.geom_bodyid[g]))
                    rows.append(3.*(normal@self.jac[:,chain['va']]));rhs.append(3.*(.004-distance))
                if rows:jac=np.vstack([jac,np.asarray(rows)]);error=np.r_[error,rhs]
                delta=np.linalg.solve(jac.T@jac+np.eye(7)*.001,jac.T@error)
                update=np.clip(candidate+np.clip(delta,-.1,.1),lo,hi)
                if np.linalg.norm(update-candidate)<1e-5:break
                candidate=update
            self.last[ci]=candidate.copy();command=np.clip(candidate+bias,chain['lower'],chain['upper']);self.targets.append(command);output[chain['aids']]=command
            self.records.append({'time_s':float(t),'actor':chain['actor'],'side':chain['side'],'acquiring_shoe':acquire,
                                 'target_position_world_m':goal,'solved_error_m':float(np.linalg.norm(goal-self.kin.xpos[body]))})
        self.next_update+=.02
        return self.ramp(data,output)

    def report(self):
        return {'type':'Actual-state bounded world arm IK with inferred actual-shoe acquisition and bench/floor clearance',
                'control_dt_s':.02,'max_joint_command_speed_rad_s':4.,'collision_margin_m':.004,
                'gravity_compensation_fraction':self.gravity_comp,
                'grasp_time_s':self.grasp_time,'acquire_lookahead_s':1.,'calls':len(self.records)//4,
                'median_solved_position_error_m':float(np.median([x['solved_error_m']for x in self.records]))if self.records else None,
                'active_state_writes':0,'external_force_writes':0}
