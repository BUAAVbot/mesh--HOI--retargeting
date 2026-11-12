"""Optional release/idle table-clearance postfilter for finger targets only.

The frozen base feedback module remains unchanged. This wrapper evaluates
candidate finger commands in private FK with the current actual arm pose. It
can defer opening while an arm lifts; it cannot repair a colliding palm/wrist
or guarantee clearance after subsequent arm motion.
"""
from __future__ import annotations
import numpy as np
import mujoco as mj
from fourth_contact_feedback import ReferenceContactFeedback,_read_config

GUARD_DEFAULTS={'enabled':True,'minimum_clearance_m':.002,'rate_limit_rad_s':1.,
                'max_update_dt_s':.05,'bisection_iterations':8,
                'guard_phases':['idle','release','fault']}


class ReferenceContactFeedbackWithTableGuard(ReferenceContactFeedback):
    def __init__(self,model,reference,config=None):
        cfg=_read_config(config);guard=dict(cfg.pop('table_clearance_guard',{}))
        unknown=set(guard)-set(GUARD_DEFAULTS)
        if unknown:raise ValueError('Unknown table guard options '+str(sorted(unknown)))
        self.guard_config={**GUARD_DEFAULTS,**guard}
        if self.guard_config['minimum_clearance_m']<0 or self.guard_config['rate_limit_rad_s']<=0:
            raise ValueError('Invalid clearance guard limit')
        super().__init__(model,reference,cfg)
        self._guard_kin=mj.MjData(model)
        self._table_geoms=[g for g in range(model.ngeom)if(mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or'').startswith('table_support')and model.geom_contype[g]>0]
        if not self._table_geoms:raise ValueError('Table guard needs active table_support* collision geoms')
        self._guard_geoms=[];self._fixed_geoms=[]
        for side,hand in enumerate(self.hands):
            arm='left_wrist'if side==0 else'right_wrist'
            geoms=[g for g in range(model.ngeom)if model.geom_contype[g]>0 and self.body_names[int(model.geom_bodyid[g])].startswith((hand['prefix'],arm))]
            self._guard_geoms.append(geoms)
            finger_bodies=set(map(int,model.jnt_bodyid[hand['joints']]))
            fixed=[]
            for g in geoms:
                b=int(model.geom_bodyid[g]);adjustable=False
                while b>0:
                    if b in finger_bodies:adjustable=True;break
                    b=int(model.body_parentid[b])
                if not adjustable:fixed.append(g)
            self._fixed_geoms.append(set(fixed))

    def reset(self):
        super().reset();self._guard_prev_time=None
        self._guard_previous=[None,None];self._guard_last_safe=[None,None]
        self.guard_records=[];self.guard_deferrals=np.zeros(2,int);self.guard_unresolved=np.zeros(2,int)
        self.guard_queries_rejected=0;self.guard_actual_max_delta=np.zeros(self.model.nu)

    def _clearance(self,data,hand_index,targets):
        hand=self.hands[hand_index];kin=self._guard_kin
        kin.qpos[:]=data.qpos;kin.qvel[:]=data.qvel;kin.qpos[hand['qa']]=targets
        mj.mj_kinematics(self.model,kin)
        minimum=1.;fixed_minimum=1.;uncertain=False
        for g in self._guard_geoms[hand_index]:
            values=[]
            for table in self._table_geoms:
                witness=np.zeros(6);distance=float(mj.mj_geomDistance(self.model,kin,g,table,1.,witness))
                valid=np.isfinite(distance)and np.isfinite(witness).all()and abs(np.linalg.norm(witness[:3]-witness[3:])-abs(distance))<1e-4
                for gg,point in [(g,witness[:3]),(table,witness[3:])]:
                    rotation=kin.geom_xmat[gg].reshape(3,3)
                    centre=kin.geom_xpos[gg]+rotation@self.model.geom_aabb[gg,:3]
                    extent=np.abs(rotation)@self.model.geom_aabb[gg,3:]
                    valid=valid and bool(np.all(np.abs(point-centre)<=extent+1e-4))
                if valid:values.append(distance)
                else:self.guard_queries_rejected+=1
            if not values:uncertain=True;continue
            distance=min(values);minimum=min(minimum,distance)
            if g in self._fixed_geoms[hand_index]:fixed_minimum=min(fixed_minimum,distance)
        return minimum,fixed_minimum,uncertain

    def apply(self,data,ctrl,t,contact_plan=None):
        baseline=np.asarray(ctrl,float).copy()
        proposed=super().apply(data,ctrl,t,contact_plan)
        if not self.guard_config['enabled']:return proposed
        output=proposed.copy();now=float(data.time)
        dt=0. if self._guard_prev_time is None else min(max(now-self._guard_prev_time,0.),float(self.guard_config['max_update_dt_s']))
        self._guard_prev_time=now;records=[]
        for side,hand in enumerate(self.hands):
            target=proposed[hand['aids']].copy()
            if self._guard_previous[side]is None:self._guard_previous[side]=baseline[hand['aids']].copy()
            previous=self._guard_previous[side].copy()
            control=hand['geometry_controlled'];rate=np.minimum(hand['speed'],float(self.guard_config['rate_limit_rad_s']))
            phase=self.states[side]['phase'];active=phase in self.guard_config['guard_phases']
            requested=target.copy()
            if active:
                target[control]=np.clip(target[control],previous[control]-rate[control]*dt,previous[control]+rate[control]*dt)
                target[control]=np.clip(target[control],hand['lower'][control],hand['upper'][control])
                output[hand['aids'][control]]=target[control]
            rate_limited=not np.allclose(target[control],requested[control],atol=1e-9)
            before,fixed,unknown=self._clearance(data,side,target)
            clearance=float(self.guard_config['minimum_clearance_m'])
            deferred=False;unresolved=False;motor_feasible=True
            if active and(before<clearance or unknown):
                def constrained(candidate):
                    candidate=np.clip(candidate,previous-rate*dt,previous+rate*dt)
                    candidate=np.clip(candidate,hand['lower'],hand['upper'])
                    lo=np.maximum(np.maximum(hand['lower'],previous-rate*dt),data.qpos[hand['qa']]+(hand['kd']*data.qvel[hand['va']]-hand['effort'])/hand['kp'])
                    hi=np.minimum(np.minimum(hand['upper'],previous+rate*dt),data.qpos[hand['qa']]+(hand['kd']*data.qvel[hand['va']]+hand['effort'])/hand['kp'])
                    ok=bool(np.all(lo[control]<=hi[control]))
                    if ok:candidate=np.clip(candidate,lo,hi)
                    candidate[~control]=target[~control]
                    return candidate,ok
                options=[previous,data.qpos[hand['qa']].copy()]
                if self._guard_last_safe[side]is not None:options.append(self._guard_last_safe[side])
                candidates=[]
                for item in options:
                    value,ok=constrained(item);distance,ff,uncertain=self._clearance(data,side,value)
                    candidates.append((distance if not uncertain else-1.,value,ok,ff,uncertain))
                best=max(candidates,key=lambda x:x[0]);target=best[1];motor_feasible=best[2]
                deferred=not np.allclose(target[control],proposed[hand['aids']][control],atol=1e-9)
                if best[0]>=clearance:
                    # Preserve the largest safe fraction of the proposed
                    # opening, rather than substituting an invented pose.
                    start=target.copy();low=0.;high=1.
                    for _ in range(int(self.guard_config['bisection_iterations'])):
                        fraction=(low+high)/2
                        value,ok=constrained(start+fraction*(proposed[hand['aids']]-start))
                        distance,ff,uncertain=self._clearance(data,side,value)
                        if distance>=clearance and not uncertain:low=fraction;target=value;motor_feasible=ok
                        else:high=fraction
                else:unresolved=True
                ids=hand['aids'][control];output[ids]=target[control]
            after,fixed_after,uncertain_after=self._clearance(data,side,output[hand['aids']])
            estimated=hand['kp']*(output[hand['aids']]-data.qpos[hand['qa']])-hand['kd']*data.qvel[hand['va']]
            motor_feasible=motor_feasible and bool(np.all(np.abs(estimated[control])<=hand['effort'][control]+1e-8))
            if after>=clearance and not uncertain_after:self._guard_last_safe[side]=output[hand['aids']].copy()
            self._guard_previous[side]=output[hand['aids']].copy()
            self.states[side]['previous_command']=output[hand['aids']].copy()
            self.states[side]['offset']=np.clip(output[hand['aids']]-baseline[hand['aids']],
                -float(self.config['max_offset_from_reference_rad']),float(self.config['max_offset_from_reference_rad']))
            self.guard_deferrals[side]+=int(deferred);self.guard_unresolved[side]+=int(unresolved)
            records.append({'phase':phase,'enabled_for_phase':active,'proposed_clearance_m':before,
                'output_clearance_m':after,'fixed_palm_wrist_clearance_m':fixed_after,
                'release_deferred':deferred,'no_clearance_feasible_in_rate_limit':unresolved,
                'rate_limited':rate_limited,'estimated_motor_torque_max_abs_Nm':float(np.max(np.abs(estimated[control]))),
                'arm_lift_required':fixed_after<clearance,'geometry_uncertain':uncertain_after,
                'estimated_motor_torque_guard_feasible':motor_feasible})
        if not np.array_equal(output[self.preserved_actuator_ids],baseline[self.preserved_actuator_ids]):raise RuntimeError('Table guard changed body/neck output')
        self.guard_actual_max_delta=np.maximum(self.guard_actual_max_delta,np.abs(output-baseline))
        if self.records:
            self.records[-1]['ctrl_delta']=output-baseline
            self.records[-1]['finger_offsets_rad']=np.asarray([s['offset'].copy()for s in self.states])
        self.guard_records.append({'time_s':now,'hands':records,'guard_ctrl_delta':output-proposed})
        return output

    def report(self):
        report=super().report();report['max_abs_actuator_target_delta_rad']=self.guard_actual_max_delta.tolist()
        report['table_clearance_guard']={'configuration':self.guard_config,'calls':len(self.guard_records),
            'release_deferred_calls_by_hand':self.guard_deferrals.tolist(),'unresolved_calls_by_hand':self.guard_unresolved.tolist(),
            'geometry_queries_rejected':self.guard_queries_rejected,
            'table_geoms':[self.model.geom(g).name for g in self._table_geoms],
            'limitations':['Private FK uses current actual arms; future arm motion can still cause impact.',
                           'If a palm/wrist or all rate-limited finger choices collide, the guard reports unresolved and requires arm lift.',
                           'Only previous safe targets/current finger angles are used as hold choices; no arbitrary open/closed pose is substituted.',
                           'No contact-force or actuator limit is increased; infeasible estimated motor-torque goals remain labelled.']}
        return report

    def history_arrays(self):
        arrays=super().history_arrays()
        if self.guard_records:
            arrays['table_guard_time_s']=np.asarray([v['time_s']for v in self.guard_records])
            arrays['table_guard_ctrl_delta']=np.asarray([v['guard_ctrl_delta']for v in self.guard_records])
            for key in ['proposed_clearance_m','output_clearance_m','fixed_palm_wrist_clearance_m','release_deferred',
                        'no_clearance_feasible_in_rate_limit','arm_lift_required','geometry_uncertain','estimated_motor_torque_guard_feasible','rate_limited','estimated_motor_torque_max_abs_Nm']:
                arrays['table_guard_'+key]=np.asarray([[h[key]for h in row['hands']]for row in self.guard_records])
        return arrays
