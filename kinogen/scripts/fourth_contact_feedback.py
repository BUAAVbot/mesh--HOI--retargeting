"""Optional force-aware finger targets for observed source contact candidates.

This module returns actuator commands only. Actual robot/object state, applied
forces and model parameters are never written. All geometry queries use a
private MjData copy. The 28 learned body channels and two neck channels are
preserved exactly; only the 22 articulated finger position targets may change.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import mujoco as mj

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_URDF=ROOT/'data/vita_demo_20261026/robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf'
PHASES={'idle':0,'acquire':1,'grasp':2,'release':3,'fault':4}
DEFAULTS={
    'enabled':True,'finger_urdf':str(DEFAULT_URDF),
    'min_candidate_confidence':.30,'max_acquire_distance_m':.10,
    'release_grace_s':.10,'grasp_confirm_s':.04,'min_grasp_digits':2,
    'contact_detection_force_N':.03,'force_target_per_digit_N':.8,
    'max_force_per_digit_N':2.,'max_hand_normal_force_N':8.,
    'auto_gravity_normal_force':False,'gravity_normal_safety_factor':1.3,
    'minimum_friction_coefficient':.10,
    'force_to_gap_gain_m_per_N':.0015,'min_requested_gap_m':-.003,
    'max_backoff_gap_m':.004,'closure_rate_rad_s':.8,'release_rate_rad_s':1.,
    'release_clearance_m':.005,
    'max_offset_from_reference_rad':.45,'max_motor_torque_Nm':.40,
    'jacobian_damping_m2':1e-5,'max_update_dt_s':.05,
    'joint_range_fault_margin_rad':.20,'joint_velocity_fault_multiplier':6.,
    'joint_velocity_fault_floor_rad_s':25.,
}


def _read_config(config):
    if config is None:return {}
    if isinstance(config,(str,Path)):return json.loads(Path(config).read_text(encoding='utf-8'))
    return dict(config)


class ReferenceContactFeedback:
    """Optional post-controller hook: ctrl = feedback.apply(data, ctrl, t).

    ``t`` is reference elapsed seconds; ``data.time`` sets actual feedback dt.
    Call at each controller update, after baseline SONIC/PD produces all 52
    commands. Use report() for provenance/stats and history_arrays() for logs.
    No body/arm channel override, weld, object tracking servo or state replay is
    implemented. Contact force thresholds are feedback goals, not guarantees
    against instantaneous impact-force peaks.
    """
    def __init__(self,model,reference,config=None):
        self.model=model;extra=_read_config(config)
        unknown=set(extra)-set(DEFAULTS)
        if unknown:raise ValueError('Unknown contact feedback options: '+str(sorted(unknown)))
        self.config={**DEFAULTS,**extra};self.config['finger_urdf']=str(self.config['finger_urdf'])
        positive=['max_acquire_distance_m','force_target_per_digit_N','max_force_per_digit_N','max_hand_normal_force_N',
                  'force_to_gap_gain_m_per_N','closure_rate_rad_s','release_rate_rad_s','max_offset_from_reference_rad',
                  'max_motor_torque_Nm','jacobian_damping_m2','max_update_dt_s','release_clearance_m',
                  'gravity_normal_safety_factor','minimum_friction_coefficient']
        if any(float(self.config[k])<=0 for k in positive)or not 0<=float(self.config['min_candidate_confidence'])<=1:
            raise ValueError('Invalid contact feedback gains, limits or confidence threshold')
        if not 1<=int(self.config['min_grasp_digits'])<=5 or float(self.config['min_requested_gap_m'])>0:
            raise ValueError('Invalid grasp digit count or requested contact gap')
        self.reference=dict(np.load(reference,allow_pickle=False))if isinstance(reference,(str,Path))else dict(reference)
        self.times=np.asarray(self.reference['time_s'],float)
        self.times=self.times-self.times[0]
        self.object_ids=np.asarray(self.reference.get('object_ids',[int(self.reference.get('object_id',5))]),int).reshape(-1)
        count=len(self.object_ids);n=len(self.times)
        self.candidate=np.asarray(self.reference.get('contact_candidate_observed',np.zeros((n,2,count),bool)),bool)
        self.confidence=np.asarray(self.reference.get('contact_candidate_confidence',self.candidate.astype(float)),float)
        if self.candidate.shape!=(n,2,count)or self.confidence.shape!=self.candidate.shape:
            raise ValueError('Contact candidates/confidence must be frame x left/right x object')
        if np.any(np.diff(self.times)<=0)or not np.isfinite(self.confidence).all():
            raise ValueError('Invalid reference timing or confidence')
        if model.nu!=52:raise ValueError('Expected complete 52-channel native actuator bank')
        self.object_bodies=np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,f'target_object_{i:02d}')for i in self.object_ids])
        if np.any(self.object_bodies<0):raise ValueError('Missing target dynamic object')
        for body in self.object_bodies:
            joints=np.flatnonzero(model.jnt_bodyid==body)
            if len(joints)!=1 or model.jnt_type[joints[0]]!=mj.mjtJoint.mjJNT_FREE or model.body_mocapid[body]>=0:
                raise ValueError('Contact feedback requires actual free, non-mocap target objects')
        self.object_lookup={int(b):i for i,b in enumerate(self.object_bodies)}
        self.object_geoms=[[g for g in range(model.ngeom)if int(model.geom_bodyid[g])==int(b)and model.geom_contype[g]>0]
                           for b in self.object_bodies]
        if any(not group for group in self.object_geoms):raise ValueError('Object has no active collision geometry')
        urdf=Path(self.config['finger_urdf']);self.urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest()
        definitions={j.get('name'):j for j in ET.parse(urdf).getroot().findall('joint')}
        self.body_names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,b)or'world'for b in range(model.nbody)]
        self.hands=[];all_aids=[]
        for prefix in ['lh_','rh_']:
            joints=[j for j in range(model.njnt)if(mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j)or'').startswith(prefix)]
            if len(joints)!=11:raise ValueError('Expected 11 native joints per hand')
            names=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_JOINT,j)for j in joints]
            aids=np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+name)for name in names],int)
            if np.any(aids<0):raise ValueError('Finger actuator missing')
            qa=model.jnt_qposadr[joints].astype(int);va=model.jnt_dofadr[joints].astype(int)
            if any(model.jnt_type[j]!=mj.mjtJoint.mjJNT_HINGE for j in joints):raise ValueError('Finger joints must be native hinges')
            kp=model.actuator_gainprm[aids,0].copy();kd=-model.actuator_biasprm[aids,2].copy()
            if np.any(kp<=0)or np.any(kd<0)or not np.allclose(model.actuator_biasprm[aids,1],-kp):
                raise ValueError('Contact feedback requires native PD finger position actuators')
            speed=[];effort=[];joint_specs=[]
            for j,name,aid in zip(joints,names,aids):
                source=definitions.get(name)
                if source is None or source.find('limit')is None:raise ValueError('Missing articulated URDF limit for '+name)
                lim=source.find('limit');axis=source.find('axis')
                urdf_axis=np.asarray([float(x)for x in(axis.get('xyz')if axis is not None else'1 0 0').split()])
                urdf_axis/=np.linalg.norm(urdf_axis)
                if not np.allclose(urdf_axis,model.jnt_axis[j],atol=1e-6):raise ValueError('Model/URDF joint axis mismatch '+name)
                velocity=float(lim.get('velocity'));motor_effort=float(lim.get('effort'))
                if velocity<=0 or motor_effort<=0:raise ValueError('Positive URDF velocity/effort limits required')
                speed.append(velocity)
                cap=min(motor_effort,float(self.config['max_motor_torque_Nm']))
                if model.actuator_forcelimited[aid]:cap=min(cap,float(np.max(np.abs(model.actuator_forcerange[aid]))))
                effort.append(cap)
                joint_specs.append({'name':name,'axis_local':urdf_axis.tolist(),'range_rad':model.jnt_range[j].tolist(),
                    'urdf_velocity_rad_s':velocity,'urdf_effort_Nm':motor_effort,'feedback_motor_torque_goal_cap_Nm':cap})
            digits=[]
            for digit in ['thumb','index','middle','ring','pinky']:
                tip=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+digit+'_distal')
                # The recovered hybrid plant places four whole-finger proxies
                # on proximal links, with no active distal geometry. Use these
                # real proxies and do not invent collision at a visual tip.
                tip_geoms=[g for g in range(model.ngeom)if self.body_names[int(model.geom_bodyid[g])].startswith(prefix+digit)and model.geom_contype[g]>0]
                if tip<0 or not tip_geoms:raise ValueError('Missing actual digit collision geometry '+prefix+digit)
                digits.append({'name':digit,'tip_body':tip,'tip_geoms':tip_geoms})
            controlled=[]
            for joint in joints:
                body=int(model.jnt_bodyid[joint]);affected=False
                for digit in digits:
                    for geom in digit['tip_geoms']:
                        cur=int(model.geom_bodyid[geom])
                        while cur>0:
                            if cur==body:affected=True;break
                            cur=int(model.body_parentid[cur])
                controlled.append(affected)
            hand={'prefix':prefix,'names':names,'joints':np.asarray(joints),'qa':qa,'va':va,'aids':aids,
                  'kp':kp,'kd':kd,'speed':np.asarray(speed),'effort':np.asarray(effort),
                  'lower':model.jnt_range[joints,0].copy(),'upper':model.jnt_range[joints,1].copy(),
                  'digits':digits,'joint_specs':joint_specs,'geometry_controlled':np.asarray(controlled,bool)}
            self.hands.append(hand);all_aids.extend(aids.tolist())
        if len(set(all_aids))!=22:raise ValueError('Expected distinct 22 finger actuator channels')
        self.finger_actuator_ids=np.asarray(all_aids,int)
        self.preserved_actuator_ids=np.asarray([i for i in range(model.nu)if i not in set(all_aids)],int)
        self._kin=mj.MjData(model);self._jac=np.zeros((3,model.nv));self._jacr=np.zeros_like(self._jac)
        self.reset()

    def reset(self):
        self.states=[{'phase':'idle','object_index':-1,'last_candidate_t':-np.inf,'contact_since':None,
                      'offset':np.zeros(11),'previous_command':None}for _ in range(2)]
        self.previous_time=None;self.events=[];self.records=[];self.calls=0
        self.contact_duration=np.zeros((2,len(self.object_ids)))
        self.contact_impulse=np.zeros_like(self.contact_duration)
        self.selected_contact_duration=np.zeros_like(self.contact_duration)
        self.max_abs_delta=np.zeros(self.model.nu);self.infeasible_torque_guards=0;self.max_actual_hand_force=0.
        self.external_plan_hold_duration=np.zeros_like(self.contact_duration)
        self.geometry_queries=0;self.geometry_queries_rejected=0

    def _phase(self,hand,phase,now,reason):
        state=self.states[hand]
        if state['phase']==phase:return
        oi=state['object_index']
        self.events.append({'time_s':float(now),'hand':hand,'old':state['phase'],'new':phase,
                            'object_id':int(self.object_ids[oi])if oi>=0 else None,'reason':reason})
        state['phase']=phase

    def _forces(self,data):
        normal=np.zeros((2,len(self.object_ids),5));total=np.zeros((2,len(self.object_ids)));mu=np.full((2,len(self.object_ids)),np.nan)
        for ci in range(data.ncon):
            contact=data.contact[ci]
            b=[int(self.model.geom_bodyid[contact.geom1]),int(self.model.geom_bodyid[contact.geom2])]
            for k,body in enumerate(b):
                if body not in self.object_lookup:continue
                other=self.body_names[b[1-k]];side=0 if other.startswith('lh_')else 1 if other.startswith('rh_')else-1
                if side<0:continue
                wrench=np.zeros(6);mj.mj_contactForce(self.model,data,ci,wrench)
                force=max(float(wrench[0]),0.);total[side,self.object_lookup[body]]+=force
                oi=self.object_lookup[body];value=float(contact.friction[0])
                mu[side,oi]=value if not np.isfinite(mu[side,oi])else min(mu[side,oi],value)
                digit=next((i for i,name in enumerate(['thumb','index','middle','ring','pinky'])if other.startswith(self.hands[side]['prefix']+name)),None)
                if digit is not None:normal[side,self.object_lookup[body],digit]+=force
        return normal,total,mu

    def _geometry(self,hand,object_index,jacobian=False):
        distances=[];rows=[]
        for digit in hand['digits']:
            best=(1.,None,None,None)
            for tip in digit['tip_geoms']:
                for obj in self.object_geoms[object_index]:
                    witness=np.zeros(6)
                    distance=float(mj.mj_geomDistance(self.model,self._kin,tip,obj,1.,witness))
                    self.geometry_queries+=1
                    reliable=np.isfinite(distance)and np.isfinite(witness).all()
                    reliable=reliable and abs(np.linalg.norm(witness[:3]-witness[3:])-abs(distance))<1e-4
                    # Native convex-distance queries occasionally return a
                    # zero/uninitialised witness for separated convex pieces.
                    # Reject it instead of treating an absent witness as touch.
                    for geom,point in [(tip,witness[:3]),(obj,witness[3:])]:
                        rotation=self._kin.geom_xmat[geom].reshape(3,3)
                        centre=self._kin.geom_xpos[geom]+rotation@self.model.geom_aabb[geom,:3]
                        extent=np.abs(rotation)@self.model.geom_aabb[geom,3:]
                        reliable=reliable and bool(np.all(np.abs(point-centre)<=extent+1e-4))
                    if not reliable:
                        self.geometry_queries_rejected+=1;continue
                    if distance<best[0]:best=(distance,witness,tip,obj)
            distance,witness,tip,obj=best;distances.append(distance)
            if jacobian:
                if witness is None:rows.append(np.zeros(11));continue
                difference=witness[:3]-witness[3:]
                if abs(distance)>1e-8 and np.linalg.norm(difference)>1e-8:
                    # Signed witness displacement / signed distance gives the
                    # outward normal, including a penetrating configuration.
                    normal=difference/distance;normal/=max(np.linalg.norm(normal),1e-12)
                else:
                    normal=self._kin.geom_xpos[tip]-self._kin.xpos[self.object_bodies[object_index]]
                    normal/=max(np.linalg.norm(normal),1e-12)
                mj.mj_jac(self.model,self._kin,self._jac,self._jacr,witness[:3],int(self.model.geom_bodyid[tip]))
                rows.append((normal@self._jac)[hand['va']].copy())
        return np.asarray(distances),np.asarray(rows)

    def apply(self,data,ctrl,t,contact_plan=None):
        baseline=np.asarray(ctrl,float)
        if baseline.shape!=(self.model.nu,)or not np.isfinite(baseline).all():raise ValueError('Expected finite complete actuator ctrl packet')
        output=baseline.copy()
        if not self.config['enabled']:return output
        now=float(data.time)
        if self.previous_time is not None and now<self.previous_time-1e-9:raise ValueError('Physics clock reset: explicitly reset contact feedback too')
        dt=0. if self.previous_time is None else min(max(now-self.previous_time,0.),float(self.config['max_update_dt_s']))
        self.previous_time=now;self.calls+=1
        frame=int(np.clip(np.searchsorted(self.times,float(t),side='right')-1,0,len(self.times)-1))
        self._kin.qpos[:]=data.qpos;self._kin.qvel[:]=data.qvel;mj.mj_forward(self.model,self._kin)
        forces,handforces,actual_friction=self._forces(data)
        plan_ids=None;plan_phases=None
        if contact_plan is not None:
            plan_ids=np.asarray(contact_plan['active_object_ids_by_hand'],int)
            plan_phases=[str(v)for v in contact_plan['phase_by_hand']]
            if plan_ids.shape!=(2,)or len(plan_phases)!=2:raise ValueError('Arm contact plan must contain two hands')
        self.contact_duration+=(handforces>float(self.config['contact_detection_force_N']))*dt
        self.contact_impulse+=handforces*dt
        self.max_actual_hand_force=max(self.max_actual_hand_force,float(handforces.max(initial=0)))
        chosen=[];phases=[];selected_conf=[];geometry_logs=[];guard_ok=[];force_targets=[];force_budgets=[]
        for side,hand in enumerate(self.hands):
            state=self.states[side];oldphase=state['phase']
            if state['previous_command']is None:state['previous_command']=baseline[hand['aids']].copy()
            candidates=self.candidate[frame,side]&(self.confidence[frame,side]>=float(self.config['min_candidate_confidence']))
            if plan_ids is not None:
                candidates&=self.object_ids==plan_ids[side]
                if plan_phases[side]in ['idle','release']:candidates[:]=False
            q=data.qpos[hand['qa']];dq=data.qvel[hand['va']]
            if np.any(q<hand['lower']-float(self.config['joint_range_fault_margin_rad']))or np.any(q>hand['upper']+float(self.config['joint_range_fault_margin_rad']))or np.any(np.abs(dq)>np.maximum(hand['speed']*float(self.config['joint_velocity_fault_multiplier']),float(self.config['joint_velocity_fault_floor_rad_s']))):
                self._phase(side,'fault',now,'actual auxiliary joint range/velocity fault')
            if state['phase']=='idle':
                possible=[]
                for oi in np.flatnonzero(candidates):
                    distances,_=self._geometry(hand,int(oi))
                    closest=float(distances.min())
                    if closest<=float(self.config['max_acquire_distance_m']):
                        possible.append((float(self.confidence[frame,side,oi])-.5*max(closest,0),int(oi)))
                if possible:
                    state['object_index']=max(possible)[1];state['last_candidate_t']=now;state['contact_since']=None
                    self._phase(side,'acquire',now,'observed candidate and actual collision proximity')
            oi=state['object_index'];conf=0.;distances=np.full(5,np.nan);feasible=True
            target_force=float(self.config['force_target_per_digit_N']);required_total=0.
            if oi>=0:
                conf=float(self.confidence[frame,side,oi])
                if candidates[oi]:state['last_candidate_t']=now
                if plan_ids is not None and state['phase']in ['acquire','grasp']:
                    if plan_ids[side]!=self.object_ids[oi]or plan_phases[side]in ['idle','release']:
                        self._phase(side,'release',now,'arm controller contact plan requests release/switch')
                    elif plan_phases[side]in ['transport','grasp','hold']and not candidates[oi]and handforces[side,oi]>float(self.config['contact_detection_force_N']):
                        state['last_candidate_t']=now;self.external_plan_hold_duration[side,oi]+=dt
                if state['phase']in ['acquire','grasp']and now-state['last_candidate_t']>float(self.config['release_grace_s']):
                    self._phase(side,'release',now,'observed candidate ended')
                contact_digits=int((forces[side,oi]>float(self.config['contact_detection_force_N'])).sum())
                if self.config['auto_gravity_normal_force']:
                    mu=actual_friction[side,oi]
                    if not np.isfinite(mu):
                        hand_mu=min(float(self.model.geom_friction[g,0])for digit in hand['digits']for g in digit['tip_geoms'])
                        obj_mu=min(float(self.model.geom_friction[g,0])for g in self.object_geoms[oi])
                        mu=min(hand_mu,obj_mu)
                    mu=max(float(mu),float(self.config['minimum_friction_coefficient']))
                    required_total=float(self.model.body_mass[self.object_bodies[oi]])*float(np.linalg.norm(self.model.opt.gravity))*float(self.config['gravity_normal_safety_factor'])/mu
                    active=max(contact_digits,int(self.config['min_grasp_digits']))
                    target_force=max(target_force,required_total/active)
                    target_force=min(target_force,float(self.config['max_force_per_digit_N']),float(self.config['max_hand_normal_force_N'])/active)
                if state['phase']=='acquire':
                    if contact_digits>=int(self.config['min_grasp_digits']):
                        if state['contact_since']is None:state['contact_since']=now
                        if now-state['contact_since']>=float(self.config['grasp_confirm_s']):self._phase(side,'grasp',now,'persistent multi-digit physical contact')
                    else:state['contact_since']=None
                if state['phase']=='grasp'and contact_digits==0:self._phase(side,'acquire',now,'physical contact lost')
                if state['phase']in ['acquire','grasp']:
                    distances,J=self._geometry(hand,oi,True)
                    # Incremental admittance: sufficient force holds the
                    # current gap; insufficient force requests a smaller gap.
                    desired=np.clip(distances+(forces[side,oi]-target_force)*float(self.config['force_to_gap_gain_m_per_N']),float(self.config['min_requested_gap_m']),float(self.config['max_backoff_gap_m']))
                    over=forces[side,oi]>float(self.config['max_force_per_digit_N'])
                    if handforces[side,oi]>float(self.config['max_hand_normal_force_N']):over[:]=True
                    desired[over]=float(self.config['max_backoff_gap_m'])
                    error=np.clip(desired-distances,-.02,.008)
                    delta=J.T@np.linalg.solve(J@J.T+float(self.config['jacobian_damping_m2'])*np.eye(5),error)
                    rate=np.minimum(hand['speed'],float(self.config['closure_rate_rad_s']))
                    state['offset']+=np.clip(delta,-rate*dt,rate*dt)
                    state['offset']=np.clip(state['offset'],-float(self.config['max_offset_from_reference_rad']),float(self.config['max_offset_from_reference_rad']))
                    self.selected_contact_duration[side,oi]+=float(handforces[side,oi]>float(self.config['contact_detection_force_N']))*dt
                elif state['phase']=='release':
                    rate=np.minimum(hand['speed'],float(self.config['release_rate_rad_s']))
                    distances,J=self._geometry(hand,oi,True)
                    # A source finger target is not necessarily an open pose.
                    # Separate by geometry before returning to the baseline.
                    saved=self._kin.qpos[hand['qa']].copy()
                    self._kin.qpos[hand['qa']]=np.clip(baseline[hand['aids']],hand['lower'],hand['upper'])
                    mj.mj_forward(self.model,self._kin)
                    baseline_distances,_=self._geometry(hand,oi)
                    self._kin.qpos[hand['qa']]=saved;mj.mj_forward(self.model,self._kin)
                    clear=float(self.config['release_clearance_m'])
                    if np.min(baseline_distances)>=clear and handforces[side,oi]<=float(self.config['contact_detection_force_N']):
                        state['offset']+=np.clip(-state['offset'],-rate*dt,rate*dt)
                        if np.max(np.abs(state['offset']))<1e-5:
                            self._phase(side,'idle',now,'collision-clear baseline target restored');state['object_index']=-1
                    else:
                        error=np.clip(clear-distances,0,.008)
                        delta=J.T@np.linalg.solve(J@J.T+float(self.config['jacobian_damping_m2'])*np.eye(5),error)
                        state['offset']+=np.clip(delta,-rate*dt,rate*dt)
                        state['offset']=np.clip(state['offset'],-float(self.config['max_offset_from_reference_rad']),float(self.config['max_offset_from_reference_rad']))
            if state['phase']in ['acquire','grasp','release']:
                proposal=baseline[hand['aids']]+state['offset']
                rate=np.minimum(hand['speed'],float(self.config['release_rate_rad_s']if state['phase']=='release'else self.config['closure_rate_rad_s']))
                rate_low=state['previous_command']-rate*dt;rate_high=state['previous_command']+rate*dt
                proposal=np.clip(proposal,rate_low,rate_high)
                low=np.maximum(np.maximum(hand['lower'],rate_low),q+(hand['kd']*dq-hand['effort'])/hand['kp'])
                high=np.minimum(np.minimum(hand['upper'],rate_high),q+(hand['kd']*dq+hand['effort'])/hand['kp'])
                feasible=bool(np.all(low[hand['geometry_controlled']]<=high[hand['geometry_controlled']]))
                if feasible:proposal=np.clip(proposal,low,high)
                else:
                    self.infeasible_torque_guards+=1;proposal=np.clip(proposal,hand['lower'],hand['upper'])
                proposal[~hand['geometry_controlled']]=baseline[hand['aids']][~hand['geometry_controlled']]
                output[hand['aids']]=proposal
            state['previous_command']=output[hand['aids']].copy()
            chosen.append(int(self.object_ids[state['object_index']])if state['object_index']>=0 else-1)
            phases.append(PHASES[state['phase']]);selected_conf.append(conf);geometry_logs.append(distances.tolist());guard_ok.append(feasible)
            force_targets.append(target_force);force_budgets.append(required_total)
        if not np.array_equal(output[self.preserved_actuator_ids],baseline[self.preserved_actuator_ids]):raise RuntimeError('Baseline body/neck control was modified')
        delta=output-baseline;self.max_abs_delta=np.maximum(self.max_abs_delta,np.abs(delta))
        self.records.append({'time_s':now,'source_frame':frame,'source_time_s':float(t),
            'selected_object_ids':chosen,'phase_codes':phases,'candidate_confidence':selected_conf,
            'physical_hand_object_normal_force_N':handforces.copy(),'selected_tip_distance_m':geometry_logs,
            'arm_plan_active_object_ids':plan_ids.copy()if plan_ids is not None else np.asarray([-1,-1]),
            'target_normal_force_per_digit_N':force_targets,'gravity_friction_required_total_normal_force_N':force_budgets,
            'actual_contact_friction_mu':actual_friction.copy(),
            'finger_offsets_rad':np.asarray([s['offset'].copy()for s in self.states]),
            'ctrl_delta':delta.copy(),'torque_guard_feasible':guard_ok})
        return output

    def report(self):
        return {'schema':'fourth.contact_feedback.v1','configuration':self.config,
            'finger_urdf_sha256':self.urdf_sha256,'calls':self.calls,
            'phase_codes':PHASES,'joint_contracts':[h['joint_specs']for h in self.hands],
            'hand_geometry_contract':self.hand_geometry_contract(),
            'finger_actuator_ids':self.finger_actuator_ids.tolist(),'baseline_body_and_neck_preserved':True,
            'geometry_controlled_finger_joints':[[n for n,active in zip(h['names'],h['geometry_controlled'])if active]for h in self.hands],
            'baseline_only_finger_joints_without_active_distal_collision':[[n for n,active in zip(h['names'],h['geometry_controlled'])if not active]for h in self.hands],
            'arm_feedback_enabled':False,'object_ids':self.object_ids.tolist(),
            'actual_contact_duration_per_hand_object_s':self.contact_duration.tolist(),
            'selected_actual_contact_duration_s':self.selected_contact_duration.tolist(),
            'arm_plan_transport_hold_without_current_source_candidate_s':self.external_plan_hold_duration.tolist(),
            'actual_normal_impulse_per_hand_object_Ns':self.contact_impulse.tolist(),
            'max_actual_hand_object_normal_force_N':self.max_actual_hand_force,
            'max_abs_actuator_target_delta_rad':self.max_abs_delta.tolist(),
            'infeasible_estimated_motor_torque_guards':self.infeasible_torque_guards,'events':self.events,
            'geometry_distance_queries':self.geometry_queries,'geometry_distance_queries_rejected':self.geometry_queries_rejected,
            'active_state_writes':0,'applied_external_force_writes':0,'model_parameter_writes':0,
            'limitations':['Source contact candidates/confidence are geometric estimates, not measured touch labels.',
                'Closing direction comes from actual signed collision distance Jacobians and native joint axes; no fixed grasp pose is substituted.',
                'Force thresholds are feedback goals, not guarantees on instantaneous impact forces. Motor torque guard uses the native PD estimate.',
                'Optional gravity normal-force budget assumes opposed contacts support object weight by friction; it is not a general force-closure proof.',
                'Four recovered whole-finger collision proxies per hand are attached to proximal bodies. Their distal DIP visual joints cannot independently affect these contact shapes and retain baseline targets.',
                'Only finger targets are adapted. Arm reach and body stability still depend on the baseline controller.',
                'Acquiring or grasping phase names do not by themselves prove stable object manipulation or full-sequence success.']}

    def history_arrays(self):
        if not self.records:return {}
        return {key:np.asarray([r[key]for r in self.records])for key in self.records[0]}

    def hand_geometry_contract(self):
        """Derive neutral palm sites and native positive-flexion directions."""
        d=mj.MjData(self.model)
        for hand in self.hands:d.qpos[hand['qa']]=hand['lower']
        mj.mj_forward(self.model,d);result=[]
        for hand in self.hands:
            prefix=hand['prefix'];base=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,prefix+'hand_base_link')
            palm=mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_GEOM,'collision_'+prefix+'palm_mesh')
            if palm<0:raise ValueError('Actual palm collision mesh missing')
            R=d.xmat[base].reshape(3,3)
            site=d.geom_xpos[palm]+d.geom_xmat[palm].reshape(3,3)@self.model.geom_aabb[palm,:3]
            digit=next(item for item in hand['digits']if item['name']=='index')
            tip=digit['tip_geoms'][0]
            point=d.geom_xpos[tip]+d.geom_xmat[tip].reshape(3,3)@self.model.geom_aabb[tip,:3]
            index=hand['names'].index(prefix+'index_mcp_pitch');joint=int(hand['joints'][index])
            pivot=d.xpos[int(self.model.jnt_bodyid[joint])]
            length=point-pivot;length/=np.linalg.norm(length)
            jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
            mj.mj_jac(self.model,d,jp,jr,point,digit['tip_body'])
            velocity=jp[:,int(hand['va'][index])]
            velocity-=length*(velocity@length);velocity/=max(np.linalg.norm(velocity),1e-12)
            result.append({'hand':prefix,'neutral_finger_angles':'native joint lower limits',
                'palm_collision_geom':self.model.geom(palm).name,'palm_center_site_in_hand_base_m':(R.T@(site-d.xpos[base])).tolist(),
                'neutral_index_length_direction_in_hand_base':(R.T@length).tolist(),
                'positive_index_mcp_curl_direction_in_hand_base':(R.T@velocity).tolist(),
                'positive_index_mcp_axis_in_own_body':self.model.jnt_axis[joint].tolist(),
                'method':'Actual palm/distal collision AABB centres plus native point-Jacobian derivative; not an assumed +/-X face.'})
        return result
