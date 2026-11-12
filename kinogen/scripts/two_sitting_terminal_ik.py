"""Small terminal seat-acquisition IK through leg motor targets only.

An already weight-bearing actor is left alone. For an eligible unsupported
actor, retract each leg by raising its foot target relative to its actual root;
gravity and physical seat contact, rather than root actuation, lower the body.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco as mj
import numpy as np

SOURCE_BYTES = Path(__file__).read_bytes()
SOURCE_SHA256 = hashlib.sha256(SOURCE_BYTES).hexdigest()


class TerminalIK:
    def __init__(self, model, reference, config=None):
        self.model = model
        self.reference = dict(np.load(reference, allow_pickle=False)) if isinstance(reference,(str,Path)) else dict(reference)
        self.config = dict(config or {})
        self.times = np.asarray(self.reference['time_s'],float)
        self.qref = np.asarray(self.reference['qpos'],float)
        if self.qref.shape != (len(self.times),2,59):
            raise ValueError('Expected qpos[T,2,59] terminal reference')
        self.end = float(self.times[-1])
        self.begin = self.end-.3
        self.control_dt = .02
        self.step_z = .002
        self.max_joint_delta = .02
        self.max_command_speed = .25
        self.ramp_s = .5
        self.damping = float(self.config.get('terminal_ik_damping',.001))
        if self.damping<=0:raise ValueError('IK damping must be positive')
        self.kin = mj.MjData(model)
        self.jac = np.zeros((3,model.nv));self.jacr = np.zeros((3,model.nv))
        self.records=[];self.parts=[];self.calls=0
        def named(kind,name):
            value=mj.mj_name2id(model,kind,name)
            if value<0:raise ValueError('Missing '+name)
            return value
        self.geom_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or''for g in range(model.ngeom)]
        self.geom_bodies=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g]))or''for g in range(model.ngeom)]
        self.seat_geom=np.asarray(['seat'in name.lower()or'stool'in name.lower()for name in self.geom_names])
        self.hip_geom=np.asarray([any(token in name for token in('base_link','hip_','pelvis','thigh'))for name in self.geom_bodies])
        for actor,prefix in enumerate(('p1_','p2_')):
            root=next((mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,prefix+suffix)
                       for suffix in('vita_demo_free_root','vita_ik_free_root')
                       if mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,prefix+suffix)>=0),-1)
            if root<0 or model.jnt_type[root]!=mj.mjtJoint.mjJNT_FREE:raise ValueError('Expected free root '+prefix)
            base=named(mj.mjtObj.mjOBJ_BODY,prefix+'base_link')
            chains=[]
            for side in('left','right'):
                ids=np.asarray([j for j in range(model.njnt)
                    if(mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j)or'').startswith(prefix+side+'_')
                    and any(token in(mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j)or'')for token in('_hip_','_knee_','_ankle_'))])
                if len(ids)!=6:raise ValueError('Expected6-joint leg '+prefix+side)
                names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,int(j))for j in ids]
                aids=np.asarray([named(mj.mjtObj.mjOBJ_ACTUATOR,prefix+'motor_'+name[len(prefix):])for name in names])
                chains.append({'side':side,'jids':ids,'qa':model.jnt_qposadr[ids],
                               'va':model.jnt_dofadr[ids],'aids':aids,
                               'foot':named(mj.mjtObj.mjOBJ_BODY,prefix+side+'_ankle_roll_link')})
            self.parts.append({'actor':actor,'prefix':prefix,'root':root,
                'rq':int(model.jnt_qposadr[root]),'base':base,'chains':chains,
                'aids':np.concatenate([x['aids']for x in chains]),
                'qa':np.concatenate([x['qa']for x in chains]),
                'weight':float(model.body_subtreemass[base]*np.linalg.norm(model.opt.gravity)),
                'minimum_z':float(self.qref[-1,actor,2]-.05),
                'stage':'untouched','activation_time':None,'last_time':None,
                'last_update':None,'last_command':None,'ik_target':None,
                'activation_count':0,'frozen_time':None})
        if any(int(j)in{p['root']for p in self.parts}for j in model.actuator_trnid[:,0]):
            raise ValueError('Root actuator forbidden')

    def seat_forces(self,data):
        force=np.zeros(2)
        for index in range(data.ncon):
            c=data.contact[index];gs=[int(c.geom1),int(c.geom2)]
            for side in range(2):
                other=1-side
                if not self.seat_geom[gs[other]] or not self.hip_geom[gs[side]]:continue
                body=self.geom_bodies[gs[side]]
                actor=0 if body.startswith('p1_')else 1 if body.startswith('p2_')else -1
                if actor<0:continue
                wrench=np.zeros(6);mj.mj_contactForce(self.model,data,index,wrench)
                world=np.asarray(c.frame).reshape(3,3).T@wrench[:3]
                upward=float(world[2])*(1 if side==1 else -1)
                if upward>.5 and wrench[0]>.5:force[actor]+=upward
        return force

    def solve(self,part,data):
        # This is a private kinematic evaluation, never an executing-state edit.
        self.kin.qpos[:]=data.qpos;self.kin.qvel[:]=data.qvel
        mj.mj_kinematics(self.model,self.kin);mj.mj_comPos(self.model,self.kin)
        target=[];deltas=[]
        for chain in part['chains']:
            mj.mj_jacBody(self.model,self.kin,self.jac,self.jacr,chain['foot'])
            matrix=np.vstack([self.jac[:,chain['va']],.1*self.jacr[:,chain['va']]])
            desired=np.asarray([0.,0.,self.step_z,0.,0.,0.])
            delta=np.linalg.solve(matrix.T@matrix+self.damping*np.eye(6),matrix.T@desired)
            delta=np.clip(delta,-self.max_joint_delta,self.max_joint_delta)
            actual=np.asarray(data.qpos[chain['qa']])
            raw=actual+delta
            limited=self.model.jnt_limited[chain['jids']].astype(bool)
            limits=self.model.jnt_range[chain['jids']]
            raw[limited]=np.clip(raw[limited],limits[limited,0],limits[limited,1])
            # A source pose outside a limit is never repaired by a larger step.
            raw=actual+np.clip(raw-actual,-self.max_joint_delta,self.max_joint_delta)
            target.extend(raw);deltas.extend(raw-actual)
        return np.asarray(target),np.asarray(deltas)

    def apply(self,packet,data,t):
        output=np.asarray(packet,float).copy()
        if output.shape!=(self.model.nu,):raise ValueError('Expected full actuator control packet')
        t=float(t)
        if t<self.begin:return output
        self.calls+=1
        support=self.seat_forces(data)
        for part in self.parts:
            actor=part['actor'];rq=part['rq'];stage=part['stage']
            threshold=.10*part['weight']
            # At first terminal observation, already seated actors are exempt
            # permanently. This preserves the independently successful actor.
            if stage=='untouched'and support[actor]>=threshold:
                part['stage']='pre_supported'
                self.records.append({'time_s':t,'actor':actor+1,'event':'pre_supported_no_intervention',
                                     'seat_force_N':float(support[actor]),'seat_threshold_N':threshold})
                continue
            if part['stage']=='pre_supported':continue
            if part['last_update']is not None and t-part['last_update']<self.control_dt-1e-8:
                if part['last_command']is not None:output[part['aids']]=part['last_command']
                continue
            source=np.asarray([np.interp(t,self.times,self.qref[:,actor,j])for j in range(3)])
            actual=np.asarray(data.qpos[rq:rq+3])
            error=float(np.linalg.norm(source[:2]-actual[:2]))
            q=np.asarray(data.qpos[rq+3:rq+7]);q=q/np.linalg.norm(q)
            tilt=float(np.arccos(np.clip(1.-2.*(q[1]**2+q[2]**2),-1.,1.)))
            eligible=error<.3 and tilt<.5
            if part['stage']=='untouched':
                if not eligible:continue
                part['stage']='descending';part['activation_time']=t
                part['activation_count']+=1
                part['last_command']=output[part['aids']].copy()
                part['ik_target']=np.asarray(data.qpos[part['qa']]).copy()
            if support[actor]>=threshold and part['stage']!='frozen':
                part['stage']='frozen';part['frozen_time']=t
            deltas=np.zeros(12);guard=''
            if part['stage']!='frozen':
                if actual[2]<=part['minimum_z']:
                    part['stage']='depth_paused';guard='root_below_reference_final_z_minus_0p05'
                    part['ik_target']=np.asarray(data.qpos[part['qa']]).copy()
                elif not eligible:
                    part['stage']='guard_paused';guard='horizontal_error_or_tilt_guard'
                    part['ik_target']=np.asarray(data.qpos[part['qa']]).copy()
                else:
                    part['stage']='descending'
                    part['ik_target'],deltas=self.solve(part,data)
            elapsed=max(0.,t-part['activation_time'])
            u=float(np.clip(elapsed/self.ramp_s,0.,1.))
            beta=u*u*(3.-2.*u)
            desired=(1-beta)*output[part['aids']]+beta*part['ik_target']
            dt=self.control_dt if part['last_time']is None else max(0.,t-part['last_time'])
            previous=part['last_command'].copy()
            command=previous+np.clip(desired-previous,-self.max_command_speed*dt,self.max_command_speed*dt)
            output[part['aids']]=command
            part['last_command']=command.copy();part['last_time']=t;part['last_update']=t
            self.records.append({'time_s':t,'actor':actor+1,'stage':part['stage'],'guard':guard,
                'root_xy_error_m':error,'root_z_m':float(actual[2]),'root_tilt_rad':tilt,
                'reference_final_root_z_m':float(self.qref[-1,actor,2]),'minimum_descent_root_z_m':part['minimum_z'],
                'seat_force_N':float(support[actor]),'seat_threshold_N':threshold,'blend_beta':beta,
                'max_ik_joint_delta_rad':float(np.max(abs(deltas))),
                'max_applied_command_speed_rad_s':float(np.max(abs(command-previous))/max(dt,1e-12)),
                'ik_leg_target_rad':part['ik_target'].tolist(),'applied_leg_target_rad':command.tolist()})
        return output

    def report(self):
        return {'controller':'Terminal unsupported-actor leg-retraction IK motor-target replacement',
                'source_file':str(Path(__file__).resolve()),'source_sha256':SOURCE_SHA256,
                'enabled_from_execution_time_s':self.begin,'control_dt_s':self.control_dt,
                'relative_foot_goal_increment_world_z_m':self.step_z,
                'relative_foot_goal_xy_shift_m':0.,'foot_rotation_goal':'preserve current orientation',
                'max_ik_joint_delta_rad':self.max_joint_delta,'max_command_speed_rad_s':self.max_command_speed,
                'full_replacement_ramp_s':self.ramp_s,'ik_damping':self.damping,
                'activation_guards':{'root_xy_error_m_below':.3,'root_tilt_rad_below':.5,
                                     'source_final_z_descent_margin_m':.05},
                'seat_support_threshold_fraction_of_body_weight':.10,
                'freeze_rule':'Freeze last IK targets once actual upward seat force reaches threshold; continue rate-limited ramp and hold.',
                'already_supported_rule':'Actors supported before their first intervention are never modified.',
                'actors':[{'actor':p['actor']+1,'stage':p['stage'],'weight_N':p['weight'],
                           'activation_time_s':p['activation_time'],'activation_count':p['activation_count'],
                           'frozen_time_s':p['frozen_time']}for p in self.parts],
                'calls':self.calls,'records':len(self.records),
                'live_state_writes':0,'equality_writes':0,'root_actuation':False,'external_force_writes':0,
                'limitations':['Explicit terminal IK replacement of learned leg control; not an unmodified learned-policy result.',
                               'Physical root descent and seat acquisition must pass independent replay and actual support/stability checks.']}
