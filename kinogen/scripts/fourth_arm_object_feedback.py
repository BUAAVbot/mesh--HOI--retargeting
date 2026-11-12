"""Optional actual-object acquire/transport/release arm IK for SONIC.

Only fourteen arm actuator targets are replaced. Candidate FK uses a separate
MjData; actual simulation state, object poses, constraints and external forces
are never written. All source targets and adapted plans are logged separately.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation


class ArmObjectFeedback:
    def __init__(self,model,reference,config=None):
        self.model=model;self.ref=reference;self.config=dict(config or {});self.data=mj.MjData(model);self.gravity_data=mj.MjData(model)
        self.times=np.asarray(reference['time_s']);self.names=list(map(str,reference['dof_names']))
        self.object_ids=np.asarray(reference.get('object_ids',np.arange(1,7)),dtype=int)
        self.object_bids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,f'target_object_{o:02d}')for o in self.object_ids]
        self.object_qadr=[int(model.jnt_qposadr[mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,f'target_object_{o:02d}_free')])for o in self.object_ids]
        self.object_geoms=[[g for g in range(model.ngeom)if model.geom_bodyid[g]==b and model.geom_contype[g]]for b in self.object_bids]
        self.hand_bids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,s+'h_hand_base_link')for s in ('l','r')]
        self.palm_geoms=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,f'collision_{s}h_palm_mesh')for s in ('l','r')]
        self.palm_sites=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,f'{s}h_palm_control_site')for s in ('l','r')]
        self.table_geoms=[g for g in range(model.ngeom)if model.geom_contype[g]and(mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('table_support')]
        self.table=self.table_geoms[0]if self.table_geoms else-1
        if min(self.object_bids+self.hand_bids+self.palm_geoms+[self.table])<0:raise ValueError('Actual object/hand/table geometry missing')
        self.table_plane=self.config.get('table_plane_world_abcd',reference.get('table_plane_world_abcd'))
        if self.table_plane is not None:
            self.table_plane=np.asarray(self.table_plane,dtype=float).reshape(4)
            if abs(self.table_plane[2])<.5:raise ValueError('Expected near-horizontal table plane')
        self.table_plane_z=self.config.get('table_plane_z_m',reference.get('table_plane_z_m'))
        self.chains=[];self.hand_geoms=[]
        for side,short in [('left','lh_'),('right','rh_')]:
            joints=[j for j in range(model.njnt)if(mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j)or'').startswith((side+'_shoulder',side+'_elbow',side+'_wrist'))]
            names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j)for j in joints]
            self.chains.append({'joint_ids':joints,'names':names,'qa':model.jnt_qposadr[joints].copy(),'va':model.jnt_dofadr[joints].copy(),'aids':np.array([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in names]),'lower':model.jnt_range[joints,0].copy(),'upper':model.jnt_range[joints,1].copy()})
            self.hand_geoms.append([g for g in range(model.ngeom)if model.geom_contype[g]and(mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g]))or'').startswith((short,side+'_wrist'))])
        if any(len(c['names'])!=7 for c in self.chains):raise ValueError('Expected two seven-joint arm chains')
        self.contact_geoms=[[g for g in gs if (mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g]))or'').startswith(prefix)]for gs,prefix in zip(self.hand_geoms,['lh_','rh_'])]
        observed=np.asarray(reference.get('contact_candidate_observed',reference['contact_candidate']),dtype=bool).copy()
        confidence=np.asarray(reference.get('contact_candidate_confidence',np.ones_like(observed,dtype=float)))
        observed &= confidence>=float(self.config.get('minimum_source_contact_confidence',.3))
        self.events=[[],[]]
        for side in range(2):
            for oi,obj in enumerate(self.object_ids):
                ids=np.flatnonzero(observed[:,side,oi])
                if not len(ids):continue
                splits=np.flatnonzero(np.diff(self.times[ids])>.55)+1
                for group in np.split(ids,splits):
                    self.events[side].append({'object_index':oi,'object_id':int(obj),'start':float(self.times[group[0]]-self.times[0]),'end':float(self.times[group[-1]]-self.times[0]),'first_frame':int(group[0])})
            self.events[side].sort(key=lambda x:x['start'])
        self.source_event_counts=[len(v)for v in self.events]
        allowed_ids=self.config.get('active_object_ids');allowed_hands=self.config.get('active_hands')
        for side in range(2):
            if allowed_hands is not None and side not in allowed_hands:self.events[side]=[]
            if allowed_ids is not None:self.events[side]=[v for v in self.events[side]if v['object_id']in allowed_ids]
        self.state=[None,None];self.completed=[set(),set()];self.last_targets=[None,None];self.held_ctrl_targets=[None,None];self.next_update=0.;self.last_time=None;self.records=[];self.absolute_hold_targets={}
        self.control_dt=float(self.config.get('control_dt',.02));self.lookahead=float(self.config.get('lookahead_s',.8));self.maxspeed=float(self.config.get('arm_speed_rad_s',3.));self.contact_plan={'active_object_ids_by_hand':[-1,-1],'phase_by_hand':['idle','idle']}
        self.jpos=np.zeros((3,model.nv));self.jrot=np.zeros((3,model.nv));self.preshape_previous={};self.preshape_previous_time=None;self.preshape_records=[]

    def reset(self):
        self.state=[None,None];self.completed=[set(),set()];self.last_targets=[None,None];self.held_ctrl_targets=[None,None];self.next_update=0.;self.last_time=None;self.records=[]
        self.preshape_previous={};self.preshape_previous_time=None;self.preshape_records=[];self.absolute_hold_targets={}

    def interp(self,array,t):
        x=float(t)+self.times[0]
        return np.array([np.interp(x,self.times,array[:,j])for j in range(array.shape[1])])

    def source_object_pose(self,oi,t):
        p=self.interp(self.ref['object_target_pose'][:,oi],t);p[3:]/=np.linalg.norm(p[3:]);return p

    def actual_pose(self,data,oi):
        return data.qpos[self.object_qadr[oi]:self.object_qadr[oi]+7].copy()

    def grasp_template(self,side,oi):
        value=self.config.get('object_grasp_templates',{}).get(str(int(self.object_ids[oi])))
        if value is None:return None
        side_name='left'if side==0 else'right'
        if 'by_hand'in value:return value['by_hand'].get(side_name)
        if value.get('hand',side_name)!=side_name:return None
        names=value.get('finger_joint_targets',{})
        prefix='lh_'if side==0 else'rh_'
        if names and not all(n.startswith(prefix)for n in names):return None
        return value

    @staticmethod
    def rot(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()

    def palm(self,data,side):
        if self.palm_sites[side]>=0:return data.site_xpos[self.palm_sites[side]].copy()
        g=self.palm_geoms[side]
        return data.geom_xpos[g]+data.geom_xmat[g].reshape(3,3)@self.model.geom_aabb[g,:3]

    def bounds(self,data,geoms):
        low=[];high=[]
        for g in geoms:
            r=data.geom_xmat[g].reshape(3,3);c=data.geom_xpos[g]+r@self.model.geom_aabb[g,:3];ext=abs(r)@self.model.geom_aabb[g,3:]
            low.append(c-ext);high.append(c+ext)
        return np.min(low,axis=0),np.max(high,axis=0)

    def table_height(self,data,xy):
        if self.table_plane is not None:
            p=self.table_plane;return float(-(p[0]*xy[0]+p[1]*xy[1]+p[3])/p[2])
        if self.table_plane_z is not None:return float(self.table_plane_z)
        # Supports a box or several convex support-mesh geoms. This fallback
        # is explicitly collider-derived, not a claim of a measured full edge.
        _,high=self.bounds(data,self.table_geoms);return float(high[2])

    def contact(self,data,side,oi):
        force=0.;digits=set();distance=1.
        target=self.object_bids[oi];hand=set(self.contact_geoms[side])
        for ci in range(data.ncon):
            c=data.contact[ci];g1,g2=int(c.geom1),int(c.geom2)
            if target not in (self.model.geom_bodyid[g1],self.model.geom_bodyid[g2]):continue
            hg=g1 if g1 in hand else g2 if g2 in hand else None
            if hg is None:continue
            f=np.zeros(6);mj.mj_contactForce(self.model,data,ci,f);force+=max(float(f[0]),0);distance=min(distance,float(c.dist))
            name=mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,hg)or''
            for token in ['thumb','index','middle','ring','pinky','palm']:
                if token in name:digits.add(token)
        return force,len(digits),distance

    def grasp_evidence(self,data,side,oi):
        """Measure opposed real contact normals; never infer grip from pose."""
        target=self.object_bids[oi];hand=set(self.contact_geoms[side]);entries=[];normal_sum=0.;up_force=0.;table_force=0.;mu=[]
        template=self.grasp_template(side,oi)or{};controlled=set(template.get('controlled_digits',['thumb','index','middle','ring','pinky']))
        up=-np.asarray(self.model.opt.gravity);up=up/max(np.linalg.norm(up),1e-9)
        for ci in range(data.ncon):
            c=data.contact[ci];g1,g2=int(c.geom1),int(c.geom2);b1,b2=int(self.model.geom_bodyid[g1]),int(self.model.geom_bodyid[g2])
            if target not in (b1,b2):continue
            force=np.zeros(6);mj.mj_contactForce(self.model,data,ci,force)
            if g1 in self.table_geoms or g2 in self.table_geoms:table_force+=max(float(force[0]),0.)
            hg=g1 if g1 in hand else g2 if g2 in hand else None
            if hg is None:continue
            sign=1. if b2==target else-1.;frame=np.asarray(c.frame).reshape(3,3)
            name=mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,hg)or''
            digit=next((s for s in ['thumb','index','middle','ring','pinky']if s in name),None)
            if digit not in controlled:continue
            up_force+=float(up@(sign*frame.T@force[:3]))
            if digit is None or force[0]<=.03:continue
            normal_sum+=float(force[0]);mu.append(float(c.friction[0]));entries.append((digit,sign*frame[0],float(force[0])))
        best=1.;opposed=False
        for digit,norm,force in entries:
            if digit!='thumb' or force<.05:continue
            for digit2,norm2,force2 in entries:
                if digit2=='thumb' or force2<.05:continue
                dot=float(norm@norm2);best=min(best,dot)
                if dot<=float(self.config.get('opposed_normal_dot_max',-.25)):opposed=True
        weight=float(self.model.body_mass[target])*float(np.linalg.norm(self.model.opt.gravity))
        friction=max(min(mu,default=.5),.1)
        budget=weight*float(self.config.get('grasp_normal_safety_factor',1.3))/friction
        digit_names=['thumb','index','middle','ring','pinky'];digit_force={name:sum(v[2]for v in entries if v[0]==name)for name in digit_names};required=template.get('required_contact_digits',[]);targets=template.get('digit_force_targets_N',{})
        if set(required)-set(digit_names)or set(targets)-set(digit_names):raise ValueError('Unknown required/force-target digit')
        if any(not np.isfinite(value)or value<0 for value in targets.values()):raise ValueError('Invalid planned digit force target')
        required_ready=all(digit_force[name]>=.05 for name in required);fraction=float(self.config.get('grasp_target_force_fraction',.9));target_ready=all(digit_force[name]>=fraction*float(value)for name,value in targets.items())
        return {'opposed_thumb_finger':opposed,'minimum_opposed_normal_dot':best,'normal_force_N':normal_sum,'required_normal_force_N':budget,'upward_hand_force_N':up_force,'table_normal_force_N':table_force,'object_weight_N':weight,'contact_digits':len(set(v[0]for v in entries)),'digit_normal_force_N':[digit_force[name]for name in digit_names],'required_contact_ready':required_ready,'target_force_ready':target_ready}

    def acquire_goal(self,data,side,oi):
        template=self.grasp_template(side,oi)
        if template is not None:
            pose=self.actual_pose(data,oi);object_r=self.rot(pose[3:])
            return pose[:3]+object_r@np.asarray(template['palm_offset_in_object_m']),object_r@self.rot(template['hand_rotation_in_object_wxyz'])
        low,high=self.bounds(data,self.object_geoms[oi]);center=(low+high)/2;radius=(high-low)/2
        base=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,'base_link');n=center-data.xpos[base];n[2]=0
        n/=max(np.linalg.norm(n),1e-6)
        down=np.array([0.,0.,-1.]);transverse=np.cross(down,n)
        # Positive MCP pitch about local +Y bends local +Z towards local +X.
        # Thus actual finger closure defines +X as the inner palm normal.
        roll=float(self.config.get('palm_roll_rad_by_hand',[0.,0.])[side])
        target_rotation=np.column_stack([n,transverse,down])@Rotation.from_euler('x',roll).as_matrix()
        g=self.palm_geoms[side];bid=self.hand_bids[side]
        current_hand_r=data.xmat[bid].reshape(3,3);rel_geom_r=current_hand_r.T@data.geom_xmat[g].reshape(3,3)
        palm_radius=float(abs((target_rotation@rel_geom_r).T@n)@self.model.geom_aabb[g,3:])
        gap=float(self.config.get('palm_surface_gap_m',.002))
        point=center-n*(float(abs(n)@radius)+palm_radius+gap)
        table_top=self.table_height(data,point[:2])
        point[2]=max(float(low[2]+.65*(high[2]-low[2])),table_top+float(self.config.get('palm_table_height_m',.085)))
        return point,target_rotation

    def solve(self,data,side,target,target_rotation,current_ctrl):
        c=self.chains[side];ik=self.data;ik.qpos[:]=data.qpos
        if self.last_targets[side]is None:
            seed=data.qpos[c['qa']].copy()if self.config.get('initial_geometric_seed','policy_control')=='actual_state'else np.asarray(current_ctrl[c['aids']]).copy()
        else:seed=self.last_targets[side].copy()
        seed=np.clip(seed,c['lower']+1e-6,c['upper']-1e-6)
        lower=np.maximum(c['lower']+1e-6,seed-self.maxspeed*self.control_dt);upper=np.minimum(c['upper']-1e-6,seed+self.maxspeed*self.control_dt)
        candidate=np.clip(seed,lower,upper);rotweight=float(self.config.get('orientation_weight_m',.06))
        for _ in range(int(self.config.get('ik_iterations',10))):
            ik.qpos[c['qa']]=candidate;mj.mj_kinematics(self.model,ik);mj.mj_comPos(self.model,ik)
            site=self.palm(ik,side);r=ik.xmat[self.hand_bids[side]].reshape(3,3);mj.mj_jac(self.model,ik,self.jpos,self.jrot,site,self.hand_bids[side])
            error=np.r_[target-site,rotweight*Rotation.from_matrix(target_rotation@r.T).as_rotvec()]
            jac=np.vstack([self.jpos[:,c['va']],rotweight*self.jrot[:,c['va']]])
            # All-arm/table collision correction, including active finger proxies.
            rows=[];corrections=[]
            for g in self.hand_geoms[side]:
                dist=1.;witness=np.zeros(6)
                for tg in self.table_geoms:
                    candidate_witness=np.zeros(6);value=float(mj.mj_geomDistance(self.model,ik,g,tg,.1,candidate_witness))
                    if value<dist:dist=value;witness=candidate_witness
                if dist<.001:
                    point=witness[:3];mj.mj_jac(self.model,ik,self.jpos,None,point,int(self.model.geom_bodyid[g]));rows.append(self.jpos[2,c['va']]*2);corrections.append((.002-dist)*2)
            if rows:jac=np.vstack([jac,*rows]);error=np.r_[error,corrections]
            delta=np.linalg.solve(jac.T@jac+np.eye(7)*.003,jac.T@error)
            next_q=np.clip(candidate+np.clip(delta,-.15,.15),lower,upper)
            if np.linalg.norm(next_q-candidate)<1e-5:break
            candidate=next_q
        ik.qpos[c['qa']]=candidate;mj.mj_kinematics(self.model,ik)
        return candidate,float(np.linalg.norm(self.palm(ik,side)-target))

    def finger_preshape(self,data,ctrl):
        """Optional optimized grasp template, upstream of force/table guards."""
        templates=self.config.get('object_grasp_templates',{})
        if not templates:return np.asarray(ctrl).copy()
        result=np.asarray(ctrl).copy();expanded_intent=np.asarray(ctrl).copy();now=float(data.time)
        dt=float(self.model.opt.timestep)if self.preshape_previous_time is None else min(max(now-self.preshape_previous_time,0.),.02)
        self.preshape_previous_time=now;rate=float(self.config.get('finger_preshape_rate_rad_s',.8));active=set();ids=[-1,-1]
        def native_rate_for_master(jid):
            factor=1.
            for e in range(self.model.neq):
                if self.model.eq_type[e]==mj.mjtEq.mjEQ_JOINT and int(self.model.eq_obj2id[e])==jid:factor=max(factor,abs(float(self.model.eq_data[e,1])))
            return rate/factor
        for side,state in enumerate(self.state):
            if state is None or state['phase']not in ['acquire','grasp','transport']:continue
            obj=int(self.object_ids[state['oi']]);template=self.grasp_template(side,state['oi'])or{};ids[side]=obj
            for name,angle in template.get('finger_joint_targets',{}).items():
                prefix='lh_'if side==0 else'rh_'
                if not name.startswith(prefix):raise ValueError('Grasp template can modify only its matching hand fingers')
                jid=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_JOINT,name);aid=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+name)
                if jid<0 or aid<0:raise ValueError('Unknown grasp-template finger joint '+name)
                lower,upper=self.model.jnt_range[jid];desired=float(np.clip(angle,lower,upper));previous=self.preshape_previous.get(aid,float(data.qpos[self.model.jnt_qposadr[jid]]))
                reduced_rate=native_rate_for_master(jid)
                value=previous+float(np.clip(desired-previous,-reduced_rate*dt,reduced_rate*dt));expanded_intent[aid]=value
                if np.all(self.model.actuator_gear[aid]==0):result[aid]=0;continue
                result[aid]=value;self.preshape_previous[aid]=value;active.add(aid)
        for aid,previous in list(self.preshape_previous.items()):
            if aid in active:continue
            jid=int(self.model.actuator_trnid[aid,0]);reduced_rate=native_rate_for_master(jid);wanted=float(ctrl[aid]);value=previous+float(np.clip(wanted-previous,-reduced_rate*dt,reduced_rate*dt))
            result[aid]=value;expanded_intent[aid]=value;self.preshape_previous[aid]=value
            if abs(value-wanted)>1e-7:active.add(aid)
        self.preshape_previous={k:v for k,v in self.preshape_previous.items()if k in active}
        for e in range(self.model.neq):
            if self.model.eq_type[e]!=mj.mjtEq.mjEQ_JOINT:continue
            child,master=int(self.model.eq_obj1id[e]),int(self.model.eq_obj2id[e])
            child_name=mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_JOINT,child)or''
            if not child_name.startswith(('lh_','rh_')):continue
            master_name=mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_JOINT,master)
            ca=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+child_name);ma=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+master_name)
            if ca<0 or ma<0:raise ValueError('O6 mimic needs native compatibility actuator names')
            coefficient=self.model.eq_data[e,:5];x=expanded_intent[ma]-self.model.qpos0[self.model.jnt_qposadr[master]]
            expanded_intent[ca]=self.model.qpos0[self.model.jnt_qposadr[child]]+sum(float(c)*x**i for i,c in enumerate(coefficient))
            if np.all(self.model.actuator_gear[ca]==0):result[ca]=0
        self.preshape_records.append({'time_s':now,'object_ids':ids,'source_baseline':np.asarray(ctrl).copy(),'template_baseline':result.copy(),'expanded_joint_intent':expanded_intent})
        return result

    def apply(self,data,ctrl,t):
        out=np.asarray(ctrl).copy();t=float(t)
        if t+1e-8<self.next_update:
            for s,c in enumerate(self.chains):
                if self.held_ctrl_targets[s]is not None:out[c['aids']]=self.held_ctrl_targets[s]
            return out
        dt=self.control_dt if self.last_time is None else max(t-self.last_time,1e-6);self.last_time=t;self.next_update=t+self.control_dt
        ids=[-1,-1];phases=['idle','idle'];kinds=['none','none'];goals=np.zeros((2,3));actual=np.zeros((2,7));source=np.zeros((2,7));planned=np.zeros((2,7));errors=np.zeros(2);forces=np.zeros(2);source_hand=np.zeros((2,3));grasp_logs=np.zeros((2,8));lift_confirmed=np.zeros(2,bool);digit_force_logs=np.zeros((2,5));required_ready_logs=np.ones(2,bool);target_ready_logs=np.ones(2,bool)
        for side in range(2):
            if self.config.get('absolute_reference_hold',False)and side in self.config.get('active_hands',[0,1]):
                if side not in self.absolute_hold_targets:self.absolute_hold_targets[side]=(self.palm(data,side).copy(),data.xmat[self.hand_bids[side]].reshape(3,3).copy())
                goal,rotation=self.absolute_hold_targets[side];goals[side]=goal;phases[side]='hold';kinds[side]='absolute_initial_actual_pose_no_object_plan'
                target,error=self.solve(data,side,goal,rotation,out);self.last_targets[side]=target;out[self.chains[side]['aids']]=target;errors[side]=error
                continue
            state=self.state[side]
            if state is None:
                eligible=[(k,e)for k,e in enumerate(self.events[side])if k not in self.completed[side]and e['start']-self.lookahead<=t<=e['end']+.25]
                if eligible:
                    k,e=eligible[0];state={'event':k,'oi':e['object_index'],'phase':'acquire','started':t,'end':e['end'],'contact_impulse':0.,'last_contact':-1.,'release_started':None};self.state[side]=state
                else:
                    if self.last_targets[side]is not None:
                        learned=out[self.chains[side]['aids']].copy();previous=self.held_ctrl_targets[side]if self.held_ctrl_targets[side]is not None else self.last_targets[side]
                        next_target=previous+np.clip(learned-previous,-self.maxspeed*dt,self.maxspeed*dt)
                        out[self.chains[side]['aids']]=next_target
                        self.last_targets[side]=None if np.max(abs(learned-next_target))<1e-3 else next_target
                    continue
            oi=state['oi'];ids[side]=int(self.object_ids[oi]);ap=self.actual_pose(data,oi);sp=self.source_object_pose(oi,t);actual[side]=ap;source[side]=sp;source_hand[side]=self.interp(self.ref['robot_hand_target_world'][:,side],t)
            force,digits,_=self.contact(data,side,oi);forces[side]=force;state['contact_impulse']+=force*dt
            if force>.08:state['last_contact']=t
            evidence=self.grasp_evidence(data,side,oi)
            grasp_logs[side]=[evidence[k]for k in ['opposed_thumb_finger','minimum_opposed_normal_dot','normal_force_N','required_normal_force_N','upward_hand_force_N','table_normal_force_N','object_weight_N','contact_digits']]
            digit_force_logs[side]=evidence['digit_normal_force_N'];required_ready_logs[side]=evidence['required_contact_ready'];target_ready_logs[side]=evidence['target_force_ready']
            require_opposed=bool(self.config.get('require_opposed_grasp',False))
            ready=evidence['opposed_thumb_finger']and evidence['normal_force_N']>=evidence['required_normal_force_N']and evidence['required_contact_ready']and evidence['target_force_ready']
            if ready:
                if state.get('grasp_ready_since')is None:state['grasp_ready_since']=t
            else:state['grasp_ready_since']=None
            gate=(state.get('grasp_ready_since')is not None and t-state['grasp_ready_since']+1e-9>=float(self.config.get('grasp_confirm_s',.1)))if require_opposed else(force>.15 and state['contact_impulse']>.02)
            if state['phase']=='acquire' and gate:
                rp=data.xmat[self.hand_bids[side]].reshape(3,3);state.update(phase='grasp'if require_opposed else'transport',grasp_started=t,grip_position=rp.T@(ap[:3]-self.palm(data,side)),grip_rotation=rp.T@self.rot(ap[3:]),source_offset=ap[:3]-sp[:3],planned_position=ap[:3].copy(),planned_rotation=self.rot(ap[3:]),grasp_initial_object=ap.copy(),grasp_initial_palm=self.palm(data,side).copy(),grasp_initial_hand_rotation=rp.copy(),lift_since=None,contact_kind='opposed_grasp_candidate_lift_test'if require_opposed else('multi_digit_grasp'if digits>=2 else'single_contact_push'))
            if state['phase']=='grasp':
                up=-np.asarray(self.model.opt.gravity);up/=max(np.linalg.norm(up),1e-9)
                rise=float(up@(ap[:3]-state['grasp_initial_object'][:3]))
                proof=rise>=float(self.config.get('lift_proof_rise_m',.012))and evidence['table_normal_force_N']<.15 and evidence['upward_hand_force_N']>=.7*evidence['object_weight_N']
                if proof and ready:
                    if state['lift_since']is None:state['lift_since']=t
                    if t-state['lift_since']+1e-9>=.08:
                        rp=data.xmat[self.hand_bids[side]].reshape(3,3);state.update(phase='transport',grasp_started=t,grip_position=rp.T@(ap[:3]-self.palm(data,side)),grip_rotation=rp.T@self.rot(ap[3:]),source_offset=ap[:3]-sp[:3],planned_position=ap[:3].copy(),planned_rotation=self.rot(ap[3:]),contact_kind='opposed_grasp_physically_confirmed_lift',lift_verified=True)
                else:state['lift_since']=None
                if state['phase']=='grasp' and t-state['grasp_started']>float(self.config.get('lift_proof_timeout_s',1.3)):
                    state['phase']='acquire';state['grasp_ready_since']=None;state['contact_impulse']=0.
            if t>state['end']+.35 and state['phase']!='release':state['phase']='release';state['release_started']=t;state['release_goal']=self.palm(data,side).copy()
            if state['phase']=='transport' and t-state['last_contact']>.25:state['phase']='acquire';state['contact_impulse']=0.
            if require_opposed and state['phase']=='transport':
                if evidence['opposed_thumb_finger']:
                    state['opposition_lost_since']=None
                else:
                    if state.get('opposition_lost_since')is None:state['opposition_lost_since']=t
                    if t-state['opposition_lost_since']>.15:state['phase']='acquire';state['lift_verified']=False;state['grasp_ready_since']=None;state['contact_impulse']=0.
            if state['phase']=='acquire':goal,rotation=self.acquire_goal(data,side,oi);planned[side]=ap
            elif state['phase']=='grasp':
                up=-np.asarray(self.model.opt.gravity);up/=max(np.linalg.norm(up),1e-9)
                amount=min(float(self.config.get('lift_proof_target_m',.025)),float(self.config.get('lift_proof_speed_m_s',.04))*(t-state['grasp_started']))
                goal=state['grasp_initial_palm']+up*amount;rotation=state['grasp_initial_hand_rotation'];planned[side]=state['grasp_initial_object'];planned[side,:3]+=up*amount
            elif state['phase']=='transport':
                blend=max(0.,1-(t-state['grasp_started'])/float(self.config.get('source_reconcile_s',1.)))
                wanted=sp[:3]+state['source_offset']*blend;delta=wanted-state['planned_position'];delta*=min(1.,float(self.config.get('object_plan_speed_m_s',.18))*dt/max(np.linalg.norm(delta),1e-9));state['planned_position']+=delta
                target_r=self.rot(sp[3:]);rv=Rotation.from_matrix(target_r@state['planned_rotation'].T).as_rotvec();rv*=min(1.,float(self.config.get('object_plan_angular_speed_rad_s',.8))*dt/max(np.linalg.norm(rv),1e-9));state['planned_rotation']=Rotation.from_rotvec(rv).as_matrix()@state['planned_rotation']
                rotation=state['planned_rotation']@state['grip_rotation'].T;goal=state['planned_position']-rotation@state['grip_position'];qq=Rotation.from_matrix(state['planned_rotation']).as_quat();planned[side]=np.r_[state['planned_position'],qq[[3,0,1,2]]]
            else:
                goal=state['release_goal']+np.array([0.,0.,min(.07,.12*(t-state['release_started']))]);rotation=data.xmat[self.hand_bids[side]].reshape(3,3).copy();planned[side]=ap
                if t-state['release_started']>.65:
                    self.completed[side].add(state['event']);self.state[side]=None;ids[side]=-1
                    if self.held_ctrl_targets[side]is not None:out[self.chains[side]['aids']]=self.held_ctrl_targets[side]
                    continue
            kinds[side]=state.get('contact_kind','none');lift_confirmed[side]=bool(state.get('lift_verified',False)and state['phase']=='transport')
            phases[side]=state['phase'];goals[side]=goal;target,error=self.solve(data,side,goal,rotation,out);self.last_targets[side]=target;out[self.chains[side]['aids']]=target;errors[side]=error
        # A geometric IK angle is not a position-motor target under gravity.
        # Compute g(q) only in disposable MjData with qvel=0, then realize it
        # through the existing arm position motors as q_target += g(q)/kp.
        # This never writes actual qpos/qvel or any generalized/external force.
        virtual=self.gravity_data;virtual.qpos[:]=data.qpos;virtual.qvel[:]=0;virtual.ctrl[:]=0
        active=[s for s in range(2)if phases[s]!='idle']
        for s in active:virtual.qpos[self.chains[s]['qa']]=out[self.chains[s]['aids']]
        mj.mj_forward(self.model,virtual)
        geometric_palm=np.asarray([self.palm(virtual,s)for s in range(2)])
        geometric_targets=np.asarray([out[c['aids']].copy()for c in self.chains])
        gravity_torque=np.asarray([virtual.qfrc_bias[c['va']].copy()for c in self.chains]);gravity_offset=np.zeros((2,7))
        if self.config.get('gravity_compensation',False):
            cap=float(self.config.get('gravity_offset_max_rad',.25))
            for s in active:
                c=self.chains[s];kp=self.model.actuator_gainprm[c['aids'],0]
                if np.any(kp<=0):raise ValueError('Arm gravity feedforward requires positive position-motor gains')
                offset=np.clip(gravity_torque[s]/kp,-cap,cap)
                out[c['aids']]=np.clip(out[c['aids']]+offset,c['lower'],c['upper'])
                gravity_offset[s]=out[c['aids']]-geometric_targets[s]
        for s,c in enumerate(self.chains):
            self.held_ctrl_targets[s]=out[c['aids']].copy()if self.last_targets[s]is not None else None
            virtual.qpos[c['qa']]=out[c['aids']]
        mj.mj_forward(self.model,virtual)
        controlled_by_hand=[];force_targets_by_hand=[];required_digits_by_hand=[]
        for side,state in enumerate(self.state):
            template=self.grasp_template(side,state['oi'])if state is not None else None
            controlled_by_hand.append(list((template or{}).get('controlled_digits',['thumb','index','middle','ring','pinky'])))
            force_targets_by_hand.append(dict(template['digit_force_targets_N'])if template and'digit_force_targets_N'in template else None);required_digits_by_hand.append(list(template['required_contact_digits'])if template and'required_contact_digits'in template else None)
        self.contact_plan={'active_object_ids_by_hand':ids,'phase_by_hand':['idle'if phase=='hold'else phase for phase in phases],'controlled_digits_by_hand':controlled_by_hand,'digit_force_targets_N_by_hand':force_targets_by_hand,'required_contact_digits_by_hand':required_digits_by_hand}
        self.records.append({'time':t,'ids':ids,'phase':phases,'contact_kind':kinds,'goal':goals,'actual':actual,'source':source,'planned':planned,'ik_error':errors,'force':forces,'source_hand':source_hand,'grasp_evidence':grasp_logs,'digit_normal_force_N':digit_force_logs,'required_contact_ready':required_ready_logs,'target_force_ready':target_ready_logs,'lift_confirmed':lift_confirmed,'actual_palm':np.asarray([self.palm(data,s)for s in range(2)]),'geometric_palm':geometric_palm,'applied_target_palm':np.asarray([self.palm(virtual,s)for s in range(2)]),'geometric_joint_targets':geometric_targets,'applied_joint_targets':np.asarray([out[c['aids']]for c in self.chains]),'static_gravity_torque_Nm':gravity_torque,'actuator_gravity_offset_rad':gravity_offset})
        return out

    def history_arrays(self):
        if not self.records:return {}
        arrays={'arm_feedback_'+key:np.asarray([r[key]for r in self.records])for key in self.records[0]}
        if self.preshape_records:arrays.update({'finger_preshape_'+key:np.asarray([r[key]for r in self.preshape_records])for key in self.preshape_records[0]})
        return arrays

    def report(self):
        return {'enabled':True,'override_joint_names':[n for c in self.chains for n in c['names']],
            'learned_leg_waist_neck_outputs_overridden':False,'palm_site':'explicit lh/rh_palm_control_site when present; legacy collision AABB center fallback; never wrist origin',
            'palm_control_site_ids':self.palm_sites,
            'inner_palm_axis_local':[1,0,0],'finger_axis_local':[0,0,1],
            'lookahead_s':self.lookahead,'actual_object_state_feedback':True,'source_targets_preserved':True,
            'arm_gravity_compensation_enabled':bool(self.config.get('gravity_compensation',False)),
            'arm_gravity_compensation_method':'private FK qvel=0; static qfrc_bias/kp added to fourteen position-motor targets only',
            'arm_gravity_offset_max_rad':float(self.config.get('gravity_offset_max_rad',.25)),
            'initial_geometric_seed':self.config.get('initial_geometric_seed','policy_control'),
            'absolute_reference_hold':bool(self.config.get('absolute_reference_hold',False)),
            'absolute_reference_hold_contract':'Optional internal arm-hold mode captures initial actual palm world pose; no object acquire/transport plan. Finger force acquisition must be disabled explicitly in the diagnostic config.',
            'require_opposed_grasp':bool(self.config.get('require_opposed_grasp',False)),
            'duration_comparison_epsilon_s':1e-9,
            'explicit_digit_force_target_readiness_fraction':float(self.config.get('grasp_target_force_fraction',.9)),
            'required_contact_and_target_gates':'When specified by template, all required digits must contact and each actual normal force must reach the explicit target fraction before lift-proof; absent fields preserve legacy readiness.',
            'object_grasp_templates':self.config.get('object_grasp_templates',{}),
            'finger_template_preshape':'optional rate-limited geometric baseline; downstream physical force/table guards retained, no actual state writes',
            'grasp_evidence_columns':['opposed_thumb_finger','minimum_opposed_normal_dot','normal_force_N','required_normal_force_N','upward_hand_force_N','table_normal_force_N','object_weight_N','contact_digits'],
            'lift_proof':'opposed real contacts and normal budget sustained, then actual object rises >=12mm, table force <.15N and hand upward force >=70% weight for >=.08s before raw transport',
            'table_support_geoms':[mj.mj_id2name(self.model,mj.mjtObj.mjOBJ_GEOM,g)for g in self.table_geoms],
            'table_plane_world_abcd':None if self.table_plane is None else self.table_plane.tolist(),
            'table_height_source':'configured source-fitted plane'if self.table_plane is not None or self.table_plane_z is not None else'compiled support collider AABB fallback',
            'observed_candidate_events':[len(x)for x in self.events], 'plan_records':len(self.records),
            'unfiltered_source_event_counts':self.source_event_counts,
            'diagnostic_active_object_ids':self.config.get('active_object_ids'),
            'diagnostic_active_hands':self.config.get('active_hands'),
            'source_timing_retimed':False,'plan_reconcile_and_rate_limits_are_task_adaptations':True,
            'limitations':['IK reach limits can prevent acquisition; contact-driven transport does not guarantee force closure.',
            ('Opposed contact and normal budget are candidates only; lift confirmation uses actual object rise, support unloading and measured upward hand force.'if self.config.get('require_opposed_grasp',False)else'Some transport starts from a single real contact and is labeled a push, not a verified grasp.'),
            '14 learned arm outputs are replaced; original SONIC leg/waist control remains active.']}
