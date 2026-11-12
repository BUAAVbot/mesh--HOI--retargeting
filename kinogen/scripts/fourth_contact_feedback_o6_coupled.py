"""O6 six-command force feedback with true passive mimic transmission.

Optimisation, limits, rate and estimated motor-torque goals are intersected in
six independent variables per hand before expansion to 11 native coordinates.
Ten passive compatibility ctrl slots are always zero. Real model/state/forces
are never written. JSON runtime polynomials/zero offsets are not used.
"""
from pathlib import Path
import copy
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from fourth_contact_feedback import _read_config,PHASES
from fourth_contact_feedback_segmented import SegmentedReferenceContactFeedback
from fourth_convex_separation import ConvexSeparationOracle


def _text(model,name):
    item=mj.mj_name2id(model,mj.mjtObj.mjOBJ_TEXT,name)
    if item<0:raise ValueError('O6 model metadata missing '+name)
    adr=int(model.text_adr[item]);size=int(model.text_size[item])
    return bytes(model.text_data[adr:adr+size]).split(b'\0',1)[0].decode('utf-8')


class O6CoupledReferenceContactFeedback(SegmentedReferenceContactFeedback):
    def __init__(self,model,reference,config=None):
        cfg=_read_config(config);backend=cfg.pop('geometry_backend','convex_sat_o6')
        if backend not in ['convex_sat_o6','convex_sat']:raise ValueError('Wrong O6 geometry backend')
        self.controlled_digits_by_hand=cfg.pop('controlled_digits_by_hand',None)
        self.controlled_digits_by_object=cfg.pop('controlled_digits_by_object',{})
        self.digit_force_targets_by_hand=cfg.pop('digit_force_targets_N_by_hand',None)
        self.digit_force_targets_by_object=cfg.pop('digit_force_targets_N_by_object',{})
        self.required_contact_digits_by_hand=cfg.pop('required_contact_digits_by_hand',None)
        self.load_preload_N=float(cfg.pop('load_preload_normal_force_N',.5))
        self.load_ramp_s=float(cfg.pop('load_force_ramp_s',.3))
        if not np.isfinite(self.load_preload_N)or self.load_preload_N<0 or not np.isfinite(self.load_ramp_s)or self.load_ramp_s<=0:
            raise ValueError('Invalid staged load-force settings')
        self.coupling_fault_limit=float(cfg.pop('mimic_state_fault_error_rad',.03))
        cfg.setdefault('finger_urdf',_text(model,'o6_joint_contract_path'))
        definitions={j.get('name'):j for j in ET.parse(cfg['finger_urdf']).getroot().findall('joint')}
        passive_names=[n for n,j in definitions.items()if j.find('mimic')is not None]
        self.passive_actuator_ids=np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in passive_names],int)
        if len(self.passive_actuator_ids)!=10 or np.any(self.passive_actuator_ids<0):raise ValueError('Expected 10 passive native actuator slots')
        for key in ['actuator_gainprm','actuator_biasprm','actuator_gear']:
            if np.any(getattr(model,key)[self.passive_actuator_ids]!=0):raise ValueError('Passive O6 motor contributes '+key)
        # Reuse strictly validated geometry/guard setup without pretending that
        # passive axes are actual PD motors. Only this private constructor copy
        # gets dummy positive gains, and it is discarded before any apply().
        validation=copy.copy(model)
        validation.actuator_gainprm[self.passive_actuator_ids,0]=1.
        validation.actuator_biasprm[self.passive_actuator_ids,1]=-1.
        for key in ['actuator_gainprm','actuator_biasprm','actuator_gear']:
            if np.any(getattr(model,key)[self.passive_actuator_ids]!=0):raise ValueError('Model copy unexpectedly aliases actual model')
        super().__init__(validation,reference,cfg)
        self.model=model;self._kin=mj.MjData(model);self._guard_kin=mj.MjData(model);self._sat=ConvexSeparationOracle(model)
        active=[]
        for h in self.hands:
            h['kp']=model.actuator_gainprm[h['aids'],0].copy();h['kd']=-model.actuator_biasprm[h['aids'],2].copy()
            independent=[n for n in h['names']if definitions[n].find('mimic')is None]
            if len(independent)!=6:raise ValueError('O6 requires six independent commands per hand')
            root_indices=np.asarray([h['names'].index(n)for n in independent],int)
            K=np.zeros((11,6));b=np.zeros(11);resolved={}
            def resolve(name,stack=()):
                if name in resolved:return resolved[name]
                if name in stack:raise ValueError('Mimic cycle')
                spec=definitions[name].find('mimic')
                if spec is None:
                    row=np.zeros(6);row[independent.index(name)]=1.;answer=(row,0.)
                else:
                    row,offset=resolve(spec.get('joint'),stack+(name,));factor=float(spec.get('multiplier','1'))
                    answer=(row*factor,offset*factor+float(spec.get('offset','0')))
                resolved[name]=answer;return answer
            for i,name in enumerate(h['names']):K[i],b[i]=resolve(name)
            h['coupling_matrix']=K;h['coupling_offset']=b;h['independent_names']=independent;h['root_indices']=root_indices
            h['root_aids']=h['aids'][root_indices];h['root_qa']=h['qa'][root_indices];h['root_va']=h['va'][root_indices]
            if np.any(h['kp'][root_indices]<=0):raise ValueError('Independent O6 motors require actual positive PD gains')
            if np.any(h['kp'][np.setdiff1d(np.arange(11),root_indices)]!=0):raise ValueError('Mimic axes must remain passive')
            h['independent_lower'],h['independent_upper']=self._reduce_bounds(h,h['lower'],h['upper'])
            for i,spec in enumerate(h['joint_specs']):
                spec['passive_mimic_coordinate']=i not in set(root_indices)
                if spec['passive_mimic_coordinate']:spec['feedback_motor_torque_goal_cap_Nm']=0.
            active.extend(h['root_aids'].tolist())
        self.active_finger_actuator_ids=np.asarray(active,int)
        cap=float(self.config['max_motor_torque_Nm'])
        if not np.all(model.actuator_forcelimited[self.active_finger_actuator_ids])or np.max(abs(model.actuator_forcerange[self.active_finger_actuator_ids]))>cap+1e-10:
            raise ValueError('O6 coupled feedback requires the engineering hard-torque-cap model (v3), not a PD goal alone')
        self.reset()

    def reset(self):
        super().reset();self.coupled_records=[];self.coupled_guard_records=[]
        for s in self.states:s['ind_offset']=np.zeros(6);s['ind_previous']=None;s['last_safe_ind']=None
        for s in self.states:s['load_contact_since']=None;s['load_force_object_id']=None
        self.mimic_state_error_max=0.;self.passive_input_max=0.;self.coupled_torque_infeasible=0
        self.source_offset_conflicts=0;self.source_offset_slack_max=0.;self.actual_active_motor_force_max=0.

    @staticmethod
    def _expand(h,u):return h['coupling_matrix']@u+h['coupling_offset']

    @staticmethod
    def _reduce_bounds(h,lower,upper):
        lo=np.full(6,-np.inf);hi=np.full(6,np.inf)
        for i,(row,offset)in enumerate(zip(h['coupling_matrix'],h['coupling_offset'])):
            index=np.flatnonzero(abs(row)>1e-12)
            if len(index)!=1:raise ValueError('Expected each native O6 coordinate to depend on one command')
            j=int(index[0]);a=float(row[j]);x=(float(lower[i])-offset)/a;y=(float(upper[i])-offset)/a
            # Empty native intervals must remain empty. Sorting x/y would
            # incorrectly turn contradictory bounds into a feasible range.
            if a>0:lo[j]=max(lo[j],x);hi[j]=min(hi[j],y)
            else:lo[j]=max(lo[j],y);hi[j]=min(hi[j],x)
        roundoff=(lo>hi)&(lo-hi<1e-10)
        middle=(lo[roundoff]+hi[roundoff])/2
        lo[roundoff]=middle;hi[roundoff]=middle
        return lo,hi

    def _command_bounds(self,h,data,baseline,previous,dt,rate):
        native_base=self._expand(h,baseline);native_previous=self._expand(h,previous)
        speed=np.minimum(h['speed'],rate);maximum=float(self.config['max_offset_from_reference_rad'])
        geometric_lo,geometric_hi=self._reduce_bounds(h,np.maximum(h['lower'],native_previous-speed*dt),np.minimum(h['upper'],native_previous+speed*dt))
        if np.any(geometric_lo>geometric_hi):raise RuntimeError('No hard range/rate feasible coupled O6 command')
        i=h['root_indices'];kp=h['kp'][i];kd=h['kd'][i];eff=h['effort'][i]
        motor_lo=data.qpos[h['root_qa']]+(kd*data.qvel[h['root_va']]-eff)/kp
        motor_hi=data.qpos[h['root_qa']]+(kd*data.qvel[h['root_va']]+eff)/kp
        lo=np.maximum(geometric_lo,motor_lo);hi=np.minimum(geometric_hi,motor_hi)
        motor_feasible=lo<=hi
        # Actual motor torque remains hard-clamped by the physical v3 model,
        # even when a transient state makes the unsaturated PD goal infeasible.
        lo=np.where(motor_feasible,lo,geometric_lo);hi=np.where(motor_feasible,hi,geometric_hi)
        soft_lo,soft_hi=self._reduce_bounds(h,native_base-maximum,native_base+maximum)
        combined_lo=np.maximum(lo,soft_lo);combined_hi=np.minimum(hi,soft_hi);offset_feasible=combined_lo<=combined_hi
        h['_last_offset_conflict']=~offset_feasible
        lo=np.where(offset_feasible,combined_lo,lo);hi=np.where(offset_feasible,combined_hi,hi)
        return lo,hi,bool(np.all(motor_feasible))

    def _controlled_mask(self,side,plan=None,object_id=None):
        digits=['thumb','index','middle','ring','pinky'];hand='left'if side==0 else'right'
        rule=self.controlled_digits_by_hand
        if object_id is None and plan is not None:
            oid=int(plan['active_object_ids_by_hand'][side]);object_id=oid if oid>=1 else None
        if object_id is None and self.states[side]['object_index']>=0:
            object_id=int(self.object_ids[self.states[side]['object_index']])
        if object_id is not None:
            rule=self.controlled_digits_by_object.get(str(object_id),self.controlled_digits_by_object.get(object_id,rule))
        if isinstance(rule,dict):rule=rule.get(hand)
        elif rule is not None and len(rule)==2 and all(isinstance(v,(list,tuple))for v in rule):rule=rule[side]
        if plan is not None and plan.get('controlled_digits_by_hand')is not None:rule=plan['controlled_digits_by_hand'][side]
        if rule is None:rule=digits
        if not isinstance(rule,(list,tuple))or len(set(rule))!=len(rule)or any(d not in digits for d in rule):raise ValueError('Invalid controlled O6 digit mask')
        return np.asarray([d in rule for d in digits],bool)

    def _selected_contact_mu(self,data,side,oi,mask,all_contact_mu):
        if np.all(mask):return all_contact_mu
        names=[d['name']for d,selected in zip(self.hands[side]['digits'],mask)if selected]
        values=[];obj=int(self.object_bodies[oi]);prefix=self.hands[side]['prefix']
        for contact in data.contact[:data.ncon]:
            b1,b2=int(self.model.geom_bodyid[int(contact.geom1)]),int(self.model.geom_bodyid[int(contact.geom2)])
            other=b2 if b1==obj else b1 if b2==obj else-1
            if other>=0 and any(self.body_names[other].startswith(prefix+d)for d in names):values.append(float(contact.friction[0]))
        return min(values)if values else np.nan

    @staticmethod
    def _hand_rule(rule,side):
        if isinstance(rule,dict)and ('left'in rule or'right'in rule):return rule.get('left'if side==0 else'right')
        if isinstance(rule,(list,tuple))and len(rule)==2 and all(v is None or isinstance(v,(dict,list,tuple))for v in rule):return rule[side]
        return rule

    def _explicit_load_targets(self,side,oi,mask,plan):
        """Optional prospective-wrench targets; absent means exact legacy path."""
        rule=self.digit_force_targets_by_hand
        if oi>=0:
            oid=int(self.object_ids[oi]);rule=self.digit_force_targets_by_object.get(str(oid),self.digit_force_targets_by_object.get(oid,rule))
        rule=self._hand_rule(rule,side)
        if plan is not None and plan.get('digit_force_targets_N_by_hand')is not None:rule=plan['digit_force_targets_N_by_hand'][side]
        if rule is None:return None,None
        digits=['thumb','index','middle','ring','pinky'];chosen={d for d,v in zip(digits,mask)if v}
        if not isinstance(rule,dict)or set(rule)!=chosen:raise ValueError('Explicit O6 force targets must name exactly the controlled digits')
        target=np.asarray([float(rule.get(d,0.))for d in digits])
        if not np.isfinite(target).all()or np.any(target<0)or np.any(target>float(self.config['max_force_per_digit_N']))or target.sum()>float(self.config['max_hand_normal_force_N']):
            raise ValueError('Explicit O6 force targets exceed configured engineering budgets')
        required=self._hand_rule(self.required_contact_digits_by_hand,side)
        if plan is not None and plan.get('required_contact_digits_by_hand')is not None:required=plan['required_contact_digits_by_hand'][side]
        if required is None:required=list(rule)
        if not isinstance(required,(list,tuple))or not required or len(set(required))!=len(required)or any(d not in chosen for d in required):raise ValueError('Invalid required load-contact digits')
        return target,np.asarray([d in required for d in digits],bool)

    def _ramp_load_targets(self,side,oi,mask,desired,required,forces,now):
        s=self.states[side];oid=int(self.object_ids[oi])if oi>=0 else-1
        if s['load_force_object_id']!=oid:s['load_contact_since']=None;s['load_force_object_id']=oid
        present=bool(oi>=0 and np.all(forces[required]>float(self.config['contact_detection_force_N'])))
        if present:
            if s['load_contact_since']is None:s['load_contact_since']=now
        else:s['load_contact_since']=None
        elapsed=0. if s['load_contact_since']is None else max(now-s['load_contact_since'],0.)
        fraction=float(np.clip(elapsed/self.load_ramp_s,0,1));blend=fraction*fraction*(3-2*fraction)
        preload=np.where(mask,np.minimum(desired,self.load_preload_N),0.)
        return preload+(desired-preload)*blend,present,fraction

    def apply(self,data,ctrl,t,contact_plan=None):
        baseline=np.asarray(ctrl,float)
        if baseline.shape!=(self.model.nu,)or not np.isfinite(baseline).all():raise ValueError('Expected finite complete native ctrl')
        output=baseline.copy();output[self.passive_actuator_ids]=0.
        self.passive_input_max=max(self.passive_input_max,float(np.max(abs(baseline[self.passive_actuator_ids]))))
        now=float(data.time)
        if self.previous_time is not None and now<self.previous_time-1e-9:raise ValueError('Physics clock reset')
        dt=0. if self.previous_time is None else min(max(now-self.previous_time,0.),float(self.config['max_update_dt_s']))
        self.previous_time=now;self.calls+=1
        frame=int(np.clip(np.searchsorted(self.times,float(t),side='right')-1,0,len(self.times)-1))
        self._kin.qpos[:]=data.qpos;self._kin.qvel[:]=data.qvel;mj.mj_forward(self.model,self._kin)
        forces,total,actual_mu=self._forces(data)
        self.contact_duration+=(total>float(self.config['contact_detection_force_N']))*dt;self.contact_impulse+=total*dt
        self.max_actual_hand_force=max(self.max_actual_hand_force,float(total.max(initial=0)))
        plan_ids=np.asarray(contact_plan['active_object_ids_by_hand'],int)if contact_plan is not None else None
        plan_phases=list(contact_plan['phase_by_hand'])if contact_plan is not None else None
        hand_records=[];guard_records=[]
        for side,h in enumerate(self.hands):
            s=self.states[side];u_base=np.clip(baseline[h['root_aids']],h['independent_lower'],h['independent_upper'])
            if s['ind_previous']is None:s['ind_previous']=np.clip(data.qpos[h['root_qa']],h['independent_lower'],h['independent_upper'])
            previous=s['ind_previous'].copy();actual_u=data.qpos[h['root_qa']]
            mimic_error=float(np.max(abs(data.qpos[h['qa']]-self._expand(h,actual_u))))
            self.mimic_state_error_max=max(self.mimic_state_error_max,mimic_error)
            candidates=self.candidate[frame,side]&(self.confidence[frame,side]>=float(self.config['min_candidate_confidence']))
            if plan_ids is not None:
                candidates&=self.object_ids==plan_ids[side]
                if plan_phases[side]in ['idle','release']:candidates[:]=False
            q=data.qpos[h['qa']];dq=data.qvel[h['va']]
            fault=bool(np.any(q<h['lower']-float(self.config['joint_range_fault_margin_rad']))or np.any(q>h['upper']+float(self.config['joint_range_fault_margin_rad']))or np.any(abs(dq)>np.maximum(h['speed']*float(self.config['joint_velocity_fault_multiplier']),float(self.config['joint_velocity_fault_floor_rad_s'])))or mimic_error>self.coupling_fault_limit)
            if fault:self._phase(side,'fault',now,'Actual O6 range/velocity/mimic-state fault')
            if s['phase']=='idle'and self.config['enabled']:
                possible=[]
                for oi in np.flatnonzero(candidates):
                    candidate_mask=self._controlled_mask(side,contact_plan,int(self.object_ids[oi]))
                    if not np.any(candidate_mask):continue
                    gaps,_=self._geometry(h,int(oi))
                    if np.min(gaps[candidate_mask])<=float(self.config['max_acquire_distance_m']):possible.append((float(self.confidence[frame,side,oi]),int(oi)))
                if possible:
                    s['object_index']=max(possible)[1];s['last_candidate_t']=now;s['contact_since']=None
                    self._phase(side,'acquire',now,'Observed candidate and real collision proximity')
            oi=int(s['object_index']);gaps=np.full(5,np.nan);target_force=float(self.config['force_target_per_digit_N']);required=0.
            mask=self._controlled_mask(side,contact_plan)
            explicit_force,required_mask=self._explicit_load_targets(side,oi,mask,contact_plan)
            force_vector=None;required_present=False;load_ramp_fraction=0.
            controlled_roots=np.asarray([any(name.startswith(h['prefix']+digit['name'])for digit,chosen in zip(h['digits'],mask)if chosen)for name in h['independent_names']],bool)
            s['ind_offset'][~controlled_roots]=0.
            count=0;all_count=0;controlled_force=0.
            if oi>=0:
                controlled_force=float(forces[side,oi,mask].sum())
                all_count=int(np.count_nonzero(forces[side,oi]>float(self.config['contact_detection_force_N'])))
                count=int(np.count_nonzero(forces[side,oi,mask]>float(self.config['contact_detection_force_N'])))
            if oi>=0 and s['phase']!='fault'and np.any(mask):
                if candidates[oi]:s['last_candidate_t']=now
                if plan_ids is not None and s['phase']in ['acquire','grasp']:
                    if plan_ids[side]!=self.object_ids[oi]or plan_phases[side]in ['idle','release']:
                        self._phase(side,'release',now,'Arm plan requests release or object switch')
                    elif plan_phases[side]in ['transport','grasp','hold']and total[side,oi]>float(self.config['contact_detection_force_N']):
                        s['last_candidate_t']=now
                        if not candidates[oi]:self.external_plan_hold_duration[side,oi]+=dt
                if s['phase']in ['acquire','grasp']and now-s['last_candidate_t']>float(self.config['release_grace_s']):self._phase(side,'release',now,'Observed candidate ended')
                selected_total=float(total[side,oi])if np.all(mask)else controlled_force
                self.selected_contact_duration[side,oi]+=float(selected_total>float(self.config['contact_detection_force_N']))*dt
                if self.config['auto_gravity_normal_force']:
                    mu=self._selected_contact_mu(data,side,oi,mask,actual_mu[side,oi])
                    if not np.isfinite(mu):mu=min(float(self.model.geom_friction[g,0])for g in self.object_geoms[oi])
                    mu=max(float(mu),float(self.config['minimum_friction_coefficient']))
                    required=float(self.model.body_mass[self.object_bodies[oi]])*float(np.linalg.norm(self.model.opt.gravity))*float(self.config['gravity_normal_safety_factor'])/mu
                    target_force=min(max(target_force,required/max(count,int(self.config['min_grasp_digits']))),float(self.config['max_force_per_digit_N']),float(self.config['max_hand_normal_force_N'])/max(count,int(self.config['min_grasp_digits'])))
                if explicit_force is not None:
                    force_vector,required_present,load_ramp_fraction=self._ramp_load_targets(side,oi,mask,explicit_force,required_mask,forces[side,oi],now)
                    required=float(explicit_force.sum())
                if s['phase']=='acquire':
                    if (count>=int(self.config['min_grasp_digits']))if explicit_force is None else required_present:
                        if s['contact_since']is None:s['contact_since']=now
                        if now-s['contact_since']>=float(self.config['grasp_confirm_s']):self._phase(side,'grasp',now,'Persistent physical multi-digit contact; not force-closure proof')
                    else:s['contact_since']=None
                if s['phase']=='grasp'and count==0:self._phase(side,'acquire',now,'Actual object contact lost')
                if s['phase']in ['acquire','grasp','release']and self.config['enabled']:
                    gaps,Jnative=self._geometry(h,oi,True);J=(Jnative@h['coupling_matrix'])[mask]
                    if s['phase']=='release':
                        desired=float(self.config['release_clearance_m']);error=np.clip(desired-gaps[mask],0,.008)
                        if np.min(gaps[mask])>=desired and total[side,oi]<=float(self.config['contact_detection_force_N']):
                            s['ind_offset']*=max(0.,1-5*dt)
                            if np.max(abs(s['ind_offset']))<1e-4:
                                self._phase(side,'idle',now,'Clearance restored and force released');s['object_index']=-1;s['ind_offset'][:]=0.
                    else:
                        force_error=forces[side,oi,mask]-target_force if force_vector is None else forces[side,oi,mask]-force_vector[mask]
                        desired=np.clip(gaps[mask]+force_error*float(self.config['force_to_gap_gain_m_per_N']),float(self.config['min_requested_gap_m']),float(self.config['max_backoff_gap_m']))
                        over=forces[side,oi,mask]>float(self.config['max_force_per_digit_N'])
                        if total[side,oi]>float(self.config['max_hand_normal_force_N']):over[:]=True
                        desired[over]=float(self.config['max_backoff_gap_m']);error=np.clip(desired-gaps[mask],-.02,.008)
                    delta=J.T@np.linalg.solve(J@J.T+float(self.config['jacobian_damping_m2'])*np.eye(int(mask.sum())),error)
                    delta[~controlled_roots]=0.
                    s['ind_offset']+=delta
            rate=float(self.config['release_rate_rad_s']if s['phase']in ['idle','release','fault']else self.config['closure_rate_rad_s'])
            lo,hi,torque_feasible=self._command_bounds(h,data,u_base,previous,dt,rate)
            requested=u_base+s['ind_offset']if s['phase']in ['acquire','grasp','release']else u_base
            if s['phase']=='fault':requested=previous
            target=np.clip(requested,lo,hi)
            # Only already-safe/current coupled commands can defer opening;
            # this never fabricates a grip or changes a physical object pose.
            guard_enabled=self.guard_config['enabled']and s['phase']in self.guard_config['guard_phases']
            before,fixed,unknown=self._clearance(data,side,self._expand(h,target));deferred=False;unresolved=False
            clearance=float(self.guard_config['minimum_clearance_m'])
            if guard_enabled and(before<clearance or unknown):
                options=[previous,actual_u]
                if s['last_safe_ind']is not None:options.append(s['last_safe_ind'])
                evaluated=[]
                for option in options:
                    value=np.clip(option,lo,hi);dist,ff,uncertain=self._clearance(data,side,self._expand(h,value))
                    evaluated.append((dist if not uncertain else-1.,value))
                distance,held=max(evaluated,key=lambda v:v[0]);old_target=target.copy();target=held
                if distance>=clearance:
                    low=0.;high=1.
                    for _ in range(int(self.guard_config['bisection_iterations'])):
                        fraction=(low+high)/2;trial=held+fraction*(old_target-held)
                        dist,ff,uncertain=self._clearance(data,side,self._expand(h,trial))
                        if dist>=clearance and not uncertain:low=fraction;target=trial
                        else:high=fraction
                else:unresolved=True
                deferred=bool(np.max(abs(target-old_target))>1e-9)
            after,fixed_after,unknown_after=self._clearance(data,side,self._expand(h,target))
            if after>=clearance and not unknown_after:s['last_safe_ind']=target.copy()
            native_target=self._expand(h,target);output[h['root_aids']]=target
            s['ind_previous']=target.copy();s['ind_offset']=target-u_base;s['previous_command']=native_target.copy();s['offset']=native_target-self._expand(h,u_base)
            i=h['root_indices'];estimated=h['kp'][i]*(target-data.qpos[h['root_qa']])-h['kd'][i]*data.qvel[h['root_va']]
            torque_feasible&=bool(np.all(abs(estimated)<=h['effort'][i]+1e-8))
            self.coupled_torque_infeasible+=int(not torque_feasible)
            native_offset_slack=np.maximum(abs(native_target-self._expand(h,u_base))-float(self.config['max_offset_from_reference_rad']),0.)
            self.source_offset_conflicts+=int(np.any(h['_last_offset_conflict']))
            self.source_offset_slack_max=max(self.source_offset_slack_max,float(native_offset_slack.max()))
            self.actual_active_motor_force_max=max(self.actual_active_motor_force_max,float(np.max(abs(data.actuator_force[h['root_aids']]))))
            hand_records.append({'phase':s['phase'],'selected_object_id':int(self.object_ids[oi])if oi>=0 else-1,
                'independent_command_rad':target.copy(),'native_expanded_target_rad':native_target.copy(),
                'native_actual_mimic_error_rad':mimic_error,'constraint_gap_m':gaps.copy(),
                'target_force_per_digit_N':target_force,'required_total_normal_force_N':required,
                'estimated_independent_motor_torque_Nm':estimated.copy(),'motor_torque_goal_feasible':torque_feasible})
            hand_records[-1]['source_offset_budget_conflict_by_command']=h['_last_offset_conflict'].copy()
            hand_records[-1]['source_offset_budget_native_slack_rad']=native_offset_slack.copy()
            hand_records[-1]['controlled_digit_mask']=mask.copy()
            hand_records[-1]['controlled_independent_command_mask']=controlled_roots.copy()
            hand_records[-1]['controlled_contact_digits_count']=count
            hand_records[-1]['all_contact_digits_count']=all_count
            hand_records[-1]['controlled_digit_normal_force_N']=controlled_force
            hand_records[-1]['all_hand_object_normal_force_N']=float(total[side,oi])if oi>=0 else 0.
            hand_records[-1]['all_physical_digit_normal_force_N']=forces[side,oi].copy()if oi>=0 else np.zeros(5)
            hand_records[-1]['uncontrolled_command_deviation_from_baseline_rad']=np.where(~controlled_roots,target-u_base,0.)
            hand_records[-1]['target_force_per_digit_N_vector']=np.full(5,target_force)if force_vector is None else force_vector.copy()
            hand_records[-1]['explicit_load_force_targets_enabled']=explicit_force is not None
            hand_records[-1]['required_load_contact_digits']=np.zeros(5,bool)if required_mask is None else required_mask.copy()
            hand_records[-1]['required_load_contacts_present']=required_present
            hand_records[-1]['load_force_ramp_fraction']=load_ramp_fraction
            guard_records.append({'output_clearance_m':after,'proposed_clearance_m':before,'release_deferred':deferred,
                                  'no_safe_command_in_current_rate_bounds':unresolved,'arm_lift_required':fixed_after<clearance,
                                  'geometry_uncertain':unknown_after})
        output[self.passive_actuator_ids]=0.
        if not np.array_equal(output[self.preserved_actuator_ids],baseline[self.preserved_actuator_ids]):raise RuntimeError('Body/neck ctrl changed')
        self.max_abs_delta=np.maximum(self.max_abs_delta,abs(output-baseline))
        self.coupled_records.append({'time_s':now,'source_frame':frame,'hands':hand_records,'ctrl_delta':output-baseline})
        self.coupled_guard_records.append(guard_records)
        return output

    def report(self):
        report=super().report();report['schema']='fourth.o6_six_command_contact_feedback.v1'
        report['configuration']['geometry_backend']='convex_sat_o6'
        report['o6_coupling']={'independent_commands_per_hand':6,'actual_active_finger_motor_ids':self.active_finger_actuator_ids.tolist(),
            'passive_zero_ctrl_motor_ids':self.passive_actuator_ids.tolist(),'maximum_ignored_passive_input_ctrl':self.passive_input_max,
            'max_actual_mimic_state_error_rad':self.mimic_state_error_max,'coupling_state_fault_limit_rad':self.coupling_fault_limit,
            'infeasible_estimated_motor_torque_goal_calls':self.coupled_torque_infeasible,
            'hands':[{'hand':h['prefix'],'independent_names':h['independent_names'],'K_native_by_independent':h['coupling_matrix'].tolist(),
                       'offset_rad':h['coupling_offset'].tolist(),'independent_lower_rad':h['independent_lower'].tolist(),
                       'independent_upper_rad':h['independent_upper'].tolist()}for h in self.hands],
            'native_expanded_targets_are_fk_intent_only_passive_ctrl_stays_zero':True,
            'pd_initialiser_validation_copy_is_private_and_discarded_before_apply':True,
            'coupling_model':'URDF linear fallback only; JSON polynomials and extra zero offsets not applied.',
            'limits':'Native range/rate/offset intersections reduced to six commands before expansion; torque goals apply to active motors only.'}
        report['o6_coupling']['limits']='Hard native range/rate and exact target coupling retained; actual actuator torque hard-clamped by v3 model. Soft baseline-offset budget conflicts are delayed and explicitly logged.'
        report['o6_coupling']['source_offset_budget_conflict_calls']=self.source_offset_conflicts
        report['o6_coupling']['maximum_source_offset_budget_slack_rad']=self.source_offset_slack_max
        report['o6_coupling']['maximum_sampled_actual_active_motor_torque_Nm']=self.actual_active_motor_force_max
        report['o6_coupling']['engineering_hard_motor_torque_cap_Nm']=float(self.config['max_motor_torque_Nm'])
        report['segmented_hand_geometry']['effective_independent_hand_commands_total']=12
        report['segmented_hand_geometry']['all_22_native_joint_coordinates_have_collision']=True
        report['segmented_hand_geometry']['all_22_native_finger_joints_geometry_controlled']=False
        report['limitations'].append('Passive coordinates follow physical soft equality transmission; real state residual is logged, never projected during integration.')
        report['calls']=len(self.coupled_records)
        report['schema']='fourth.o6_six_command_contact_feedback.v2'
        report['schema']='fourth.o6_six_command_contact_feedback.v3_controlled_digits'
        report['controlled_digit_feedback']={'default':'All five digits; preserves previous force-feedback semantics.',
            'configuration_by_hand':self.controlled_digits_by_hand,'configuration_by_object':self.controlled_digits_by_object,
            'arm_plan_key':'controlled_digits_by_hand (left list, right list), per active object template',
            'feedback_rows_and_counts':'Only selected digit rows enter J*K solve, contact count and per-digit gravity force budget.',
            'unselected_commands':'No force offset; retain prescribed baseline subject to unchanged native physical guards.',
            'all_physical_contacts_still_recorded':True}
        report['max_abs_actuator_target_delta_rad']=self.max_abs_delta.tolist()
        report['staged_load_force_targets']={'default':'Disabled; legacy scalar targets and phase tests retained.',
            'plan_keys':['digit_force_targets_N_by_hand','required_contact_digits_by_hand'],
            'configuration_by_hand':self.digit_force_targets_by_hand,'configuration_by_object':self.digit_force_targets_by_object,
            'required_configuration_by_hand':self.required_contact_digits_by_hand,
            'preload_normal_force_N':self.load_preload_N,'ramp_duration_s':self.load_ramp_s,
            'activation':'All declared required digit contacts must persist; smoothstep rise resets if any is lost.',
            'scope':'Optional planning targets only; actual contacts, wrench and actuator limits still require physical validation.'}
        report['finger_actuator_ids']=self.active_finger_actuator_ids.tolist()
        report['geometry_controlled_finger_joints']=[h['independent_names']for h in self.hands]
        report['table_clearance_guard']['calls']=len(self.coupled_guard_records)
        report['table_clearance_guard']['release_deferred_calls_by_hand']=[sum(int(r[s]['release_deferred'])for r in self.coupled_guard_records)for s in range(2)]
        report['table_clearance_guard']['unresolved_calls_by_hand']=[sum(int(r[s]['no_safe_command_in_current_rate_bounds'])for r in self.coupled_guard_records)for s in range(2)]
        return report

    def history_arrays(self):
        if not self.coupled_records:return {}
        records=self.coupled_records
        arrays={'time_s':np.asarray([r['time_s']for r in records]),'source_frame':np.asarray([r['source_frame']for r in records]),
                'ctrl_delta':np.asarray([r['ctrl_delta']for r in records])}
        keys=['selected_object_id','independent_command_rad','native_expanded_target_rad','native_actual_mimic_error_rad',
              'constraint_gap_m','target_force_per_digit_N','required_total_normal_force_N',
              'estimated_independent_motor_torque_Nm','motor_torque_goal_feasible']
        keys+=['source_offset_budget_conflict_by_command','source_offset_budget_native_slack_rad']
        keys+=['controlled_digit_mask','controlled_independent_command_mask','controlled_contact_digits_count','all_contact_digits_count',
               'controlled_digit_normal_force_N','all_hand_object_normal_force_N','all_physical_digit_normal_force_N',
               'uncontrolled_command_deviation_from_baseline_rad']
        keys+=['target_force_per_digit_N_vector','explicit_load_force_targets_enabled','required_load_contact_digits','required_load_contacts_present','load_force_ramp_fraction']
        for key in keys:arrays[key]=np.asarray([[h[key]for h in r['hands']]for r in records])
        arrays['phase_codes']=np.asarray([[PHASES[h['phase']]for h in r['hands']]for r in records])
        for key in self.coupled_guard_records[0][0]:arrays['table_guard_'+key]=np.asarray([[h[key]for h in r]for r in self.coupled_guard_records])
        return arrays
