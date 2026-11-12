"""Two real free-root Vita actors, original SONIC inference and a free shoe.

Only controls and explicitly logged compliant attachment equality events
are changed during the shared physical rollout. Separate
inference-only proxy states mirror actual robot states to preserve the
existing single-robot SONIC observation implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def write_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def named(model, kind, name):
    idx = mujoco.mj_name2id(model, kind, name)
    if idx < 0:
        raise ValueError('Missing '+name)
    return idx


def mapping(model, names, prefix=''):
    roots = [prefix+'vita_demo_free_root', prefix+'vita_ik_free_root']
    root = next((mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in roots
                 if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0), -1)
    if root < 0 or model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Expected named robot free root '+prefix)
    jids = np.asarray([named(model, mujoco.mjtObj.mjOBJ_JOINT, prefix+n) for n in names])
    aids = np.asarray([named(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix+'motor_'+n) for n in names])
    if len(names) != 52 or len(set(jids)) != 52 or len(set(aids)) != 52:
        raise ValueError('Expected full Vita52 mapping')
    if not np.array_equal(model.actuator_trnid[aids, 0], jids):
        raise ValueError('Actuator transmission mismatch')
    return {'root':root, 'rq':int(model.jnt_qposadr[root]), 'rv':int(model.jnt_dofadr[root]),
            'jids':jids, 'qa':model.jnt_qposadr[jids], 'va':model.jnt_dofadr[jids], 'aids':aids,
            'base':named(model, mujoco.mjtObj.mjOBJ_BODY, prefix+'base_link')}


def preserve_passive_hand_channels(model):
    """A coupling follower is never silently restored as an independent PD motor."""
    disabled=[]
    for eq in range(model.neq):
        if model.eq_type[eq]!=mujoco.mjtEq.mjEQ_JOINT:continue
        jid=int(model.eq_obj1id[eq]);name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,jid)or''
        if not any(token in name for token in ('lh_','rh_')):continue
        for aid in np.flatnonzero(model.actuator_trnid[:,0]==jid):
            model.actuator_gainprm[aid]=0.;model.actuator_biasprm[aid]=0.;disabled.append(int(aid))
    return sorted(set(disabled))


class DualSonic:
    def __init__(self, model, proxy_path, reference, config, sonic_weight=1., arm_reference_weight=0., gravity_comp=0.,finger_armature=0.,seated_balance_gain=0.,world_arm_feedback=False,world_leg_feedback=False,root_acceleration_feedback=False,root_feedback_objective='root',preserve_worn_leg=False,support_arm_stabilizer=False):
        from fourth_sonic_controller import Controller
        self.model = model
        self.names = list(map(str, reference['dof_names']))
        self.ref = reference
        self.times = np.asarray(reference['time_s'])
        self.sonic_weight = sonic_weight
        self.arm_reference_weight = arm_reference_weight
        self.gravity_comp = gravity_comp
        self.finger_armature=finger_armature
        self.seated_balance_gain=seated_balance_gain;self.tilt_filter=np.zeros((2,2));self.balance_offsets=[]
        self.parts = []
        self.clipped_targets = 0
        self.raw_policy_targets = []
        self.applied_body_targets = []
        self.raw_learned_actions=[];self.effective_action_observations=[]
        for actor, prefix in enumerate(['p1_', 'p2_']):
            proxy = mujoco.MjModel.from_xml_path(str(proxy_path.resolve()))
            qref = np.asarray(reference.get('qpos_'+prefix[:2], reference['qpos'][:, actor]))
            r = {'time_s':self.times, 'qpos':qref, 'dof_names':reference['dof_names']}
            controller = Controller(proxy, r, config)
            preserve_passive_hand_channels(proxy)
            pm = mapping(proxy, self.names)
            sm = mapping(model, self.names, prefix)
            finger=np.asarray([n.startswith(('lh_','rh_'))for n in self.names])&(proxy.actuator_gainprm[pm['aids'],0]>0)
            proxy.dof_armature[pm['va'][finger]]+=finger_armature
            # Transfer exactly the constructor's effective actuator/rotor model
            # into the shared dynamic plant before reset and freeze.
            for key in ('actuator_gainprm', 'actuator_biasprm', 'actuator_gaintype', 'actuator_biastype',
                        'actuator_dyntype', 'actuator_ctrllimited', 'actuator_ctrlrange',
                        'actuator_forcelimited', 'actuator_forcerange'):
                getattr(model, key)[sm['aids']] = getattr(proxy, key)[pm['aids']]
            for key in ('dof_armature', 'dof_frictionloss', 'dof_damping'):
                getattr(model, key)[sm['va']] = getattr(proxy, key)[pm['va']]
            pd = mujoco.MjData(proxy)
            arm_mask = np.asarray([n.startswith(('left_shoulder','right_shoulder','left_elbow',
                                                'right_elbow','left_wrist','right_wrist')) for n in self.names])
            self.parts.append({'proxy':proxy, 'data':pd, 'controller':controller, 'pm':pm,
                               'sm':sm, 'reference':qref, 'prefix':prefix, 'arm_mask':arm_mask})
        self.world_arm_feedback=None
        self.world_leg_feedback=None
        self.root_acceleration_feedback=None
        if world_arm_feedback:
            from rgb_near_task_feedback import WorldArmFeedback
            self.world_arm_feedback=WorldArmFeedback(model,reference,self.parts,gravity_comp)
        if world_leg_feedback:
            from rgb_near_task_feedback import WorldLegFeedback
            self.world_leg_feedback=WorldLegFeedback(model,reference,self.parts,gravity_comp)
        if root_acceleration_feedback:
            from rgb_near_task_feedback import RootAccelerationFeedback
            self.root_acceleration_feedback=RootAccelerationFeedback(model,reference,self.parts,root_feedback_objective,preserve_worn_leg,support_arm_stabilizer)
        self.passive_abi_channels=preserve_passive_hand_channels(model)

    def sync(self, part, data):
        pd, pm, sm = part['data'], part['pm'], part['sm']
        pd.qpos[pm['rq']:pm['rq']+7] = data.qpos[sm['rq']:sm['rq']+7]
        pd.qvel[pm['rv']:pm['rv']+6] = data.qvel[sm['rv']:sm['rv']+6]
        pd.qpos[pm['qa']] = data.qpos[sm['qa']]
        pd.qvel[pm['va']] = data.qvel[sm['va']]
        pd.time = data.time
        mujoco.mj_forward(part['proxy'], pd)

    def reset(self, data):
        for part in self.parts:
            self.sync(part, data)
            part['controller'].reset(part['data'])

    def action(self, data, elapsed):
        packet = np.zeros(self.model.nu)
        raw, applied,learned_actions,effective_actions = [], [],[],[]
        all_balance=[]
        for actor,part in enumerate(self.parts):
            self.sync(part, data)
            ctrl = part['controller'].action(part['data'], elapsed)
            pm, sm = part['pm'], part['sm']
            target = ctrl[pm['aids']].copy()
            raw.append(target.copy())
            ref = np.asarray([np.interp(elapsed+self.times[0], self.times, part['reference'][:,7+j]) for j in range(52)])
            target = self.sonic_weight*target+(1-self.sonic_weight)*ref
            mask = part['arm_mask']
            target[mask] = (1-self.arm_reference_weight)*target[mask]+self.arm_reference_weight*ref[mask]
            if self.gravity_comp:
                kp = self.model.actuator_gainprm[sm['aids'],0]
                correction = np.clip(self.gravity_comp*data.qfrc_bias[sm['va']]/np.maximum(kp,1e-9),-.2,.2)
                target += correction
            balance=np.zeros(52)
            if self.seated_balance_gain:
                qroot=np.asarray([np.interp(elapsed+self.times[0],self.times,part['reference'][:,j])for j in range(3,7)])
                qroot/=np.linalg.norm(qroot);matrix=np.empty(9);mujoco.mju_quat2Mat(matrix,qroot)
                relative=matrix.reshape(3,3).T@part['data'].xmat[pm['base']].reshape(3,3)
                tilt=np.asarray([np.arctan2(relative[2,1],relative[2,2]),np.arctan2(-relative[2,0],np.hypot(relative[2,1],relative[2,2]))])
                self.tilt_filter[actor]+=(float(self.model.opt.timestep)/(.08+float(self.model.opt.timestep)))*(tilt-self.tilt_filter[actor])
                roll,pitch=np.clip(self.seated_balance_gain*self.tilt_filter[actor],-.6,.6)
                support=set();prefix=part['prefix']
                for ci in range(data.ncon):
                    contact=data.contact[ci];gs=[int(contact.geom1),int(contact.geom2)]
                    gn=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g)or''for g in gs]
                    bn=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(self.model.geom_bodyid[g]))or''for g in gs]
                    if any('ground'in n or'floor'in n for n in gn):
                        for side in ['left','right']:
                            if any(n.startswith(prefix+side+'_ankle')for n in bn):support.add(side)
                        if actor==0 and any(n=='shoe'for n in bn):support.add('left')
                if not support:support={'left','right'}
                for side in support:
                    balance[self.names.index(side+'_hip_pitch_joint')]+=pitch
                    balance[self.names.index(side+'_hip_roll_joint')]+=roll
                balance[self.names.index('waist_pitch_joint')]-=float(np.clip(self.tilt_filter[actor,1],-.6,.6))
                target+=balance
            all_balance.append(balance)
            limits = self.model.jnt_range[sm['jids']]
            bounded = np.clip(target, limits[:,0], limits[:,1])
            self.clipped_targets += int(np.sum(np.abs(target-bounded)>1e-10))
            packet[sm['aids']] = bounded
            applied.append(bounded.copy())
            c=part['controller']
            learned_actions.append(((c.target-c.default)/c.scale).copy())
            effective=((bounded[c.jref]-c.default)/c.scale)
            if self.sonic_weight!=1 or self.arm_reference_weight!=0 or self.gravity_comp!=0 or self.seated_balance_gain!=0:
                c.last=effective.copy()
            effective_actions.append(c.last.copy())
        if self.world_leg_feedback is not None:packet=self.world_leg_feedback.apply(data,packet,elapsed)
        if self.world_arm_feedback is not None:packet=self.world_arm_feedback.apply(data,packet,elapsed)
        if self.root_acceleration_feedback is not None:packet=self.root_acceleration_feedback.apply(data,packet,elapsed)
        if self.world_arm_feedback is not None or self.world_leg_feedback is not None or self.root_acceleration_feedback is not None:
            applied=[];effective_actions=[]
            for part in self.parts:
                target=packet[part['sm']['aids']];c=part['controller'];c.last=((target[c.jref]-c.default)/c.scale).copy()
                applied.append(target.copy());effective_actions.append(c.last.copy())
        inactive=(self.model.actuator_gainprm[:,0]==0)&np.all(self.model.actuator_biasprm==0,axis=1)
        packet[inactive]=0.
        if np.any(inactive):applied=[packet[part['sm']['aids']].copy()for part in self.parts]
        self.raw_policy_targets.append(np.asarray(raw))
        self.applied_body_targets.append(np.asarray(applied))
        self.raw_learned_actions.append(np.asarray(learned_actions));self.effective_action_observations.append(np.asarray(effective_actions))
        self.balance_offsets.append(np.asarray(all_balance))
        return packet

    def report(self):
        replaced=set()
        for feedback in [self.world_leg_feedback,self.world_arm_feedback]:
            if feedback is not None:
                for chain in feedback.chains:replaced.update(map(int,chain['aids']))
        learned=[int(part['sm']['aids'][j])for part in self.parts for j in part['controller'].jref]
        return {'type':'dual original newft15225 SONIC with optional disclosed joint-reference residual',
                'actors':[p['controller'].report() for p in self.parts],
                'sonic_target_weight':self.sonic_weight,
                'final_learned_residual_actuator_names':[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)for a in learned if a not in replaced],
                'final_ik_replaced_actuator_names':[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_ACTUATOR,a)for a in sorted(replaced)],
                'learned_contribution_note':'Real strict SONIC inference is retained; world-leg and world-arm IK replace those target channels after blending. The stated SONIC weight applies directly only to remaining learned channels. Root feedback then adds logged bounded joint corrections.',
                'arm_reference_override_weight':self.arm_reference_weight,
                'gravity_compensation_fraction':self.gravity_comp,
                'assumed_additional_finger_rotor_kg_m2':self.finger_armature,
                'seated_balance_feedback':{'gain':self.seated_balance_gain,'max_joint_offset_rad':.6,'tilt_filter_time_constant_s':.08,'method':'Actual relative root tilt offsets ground-supported hips and opposite waist pitch; actuator targets only.'},
                'world_arm_feedback':self.world_arm_feedback.report()if self.world_arm_feedback is not None else None,
                'world_leg_feedback':self.world_leg_feedback.report()if self.world_leg_feedback is not None else None,
                'root_acceleration_feedback':self.root_acceleration_feedback.report()if self.root_acceleration_feedback is not None else None,
                'joint_limit_clipped_target_values':self.clipped_targets,
                'physical_actuator_channels':int(np.sum(self.model.actuator_gainprm[:,0]!=0)),
                'zero_force_abi_channels':int(np.sum(self.model.actuator_gainprm[:,0]==0)),
                'mimic_follower_zero_force_actuator_ids':self.passive_abi_channels,
                'hybrid_action_history':'When hybrid target corrections are active, previous-action observation is replaced by normalized effective body targets; original learned actions are saved separately.',
                'proxy_states':'Inference-only mirror of actual shared-plant states; no proxy integration/forces applied to shared plant.'}


def contact_snapshot(model, data, step):
    rows = []
    flags = {'seat':[False,False], 'feet':[False,False], 'shoe_hand':[False,False]}
    impulse_force = [0.,0.]
    max_pen = 0.
    for i in range(data.ncon):
        c = data.contact[i]
        gids = [int(c.geom1), int(c.geom2)]
        bids = [int(model.geom_bodyid[g]) for g in gids]
        gn = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or str(g) for g in gids]
        bn = [mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b) or 'world' for b in bids]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model,data,i,wrench)
        frame = np.asarray(c.frame).reshape(3,3)
        normal = max(0.,float(wrench[0]))
        max_pen = max(max_pen,max(0.,-float(c.dist)))
        for actor,prefix in enumerate(['p1_','p2_']):
            robot = [n for n in bn if n.startswith(prefix)]
            if not robot: continue
            if any('bench_seat' in n for n in gn) and any(any(x in n for x in ('base_link','hip_')) for n in robot):
                flags['seat'][actor] = True
            if any('ground' in n or 'floor' in n for n in gn) and any('ankle' in n or 'foot' in n for n in robot):
                flags['feet'][actor] = True
            if any('shoe' in n for n in bn+gn) and any(n.startswith((prefix+'lh_',prefix+'rh_')) for n in robot):
                flags['shoe_hand'][actor] = True
                impulse_force[actor] += normal
        rows.append((step, data.time-model.opt.timestep, gids,bids,np.asarray(c.pos).copy(),
                     frame.T@wrench[:3],frame.T@wrench[3:],normal,float(c.dist)))
    return rows,flags,impulse_force,max_pen


class ShoeAttachments:
    """Explicit compliant wear/grasp hypotheses, never a pose-following force."""
    def __init__(self,model,reference,config=None):
        self.model=model;self.reference=reference;self.config=dict(config or {})
        self.times=np.asarray(reference['time_s']);self.events=[];self.current=''
        self.names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_EQUALITY,i)or str(i)for i in range(model.neq)]
        self.ids={n:i for i,n in enumerate(self.names)if n.startswith('shoe_wear_')or n.startswith('shoe_grasp_')}
        self.shoe=named(model,mujoco.mjtObj.mjOBJ_BODY,'shoe')
        self.phase_names=np.asarray(reference.get('shoe_attachment_name',np.full(len(self.times),'none'))).astype(str)
        if self.phase_names.shape!=(len(self.times),):raise ValueError('shoe_attachment_name must be one string per source frame')
        self.rejected_activations=0;self.last_rejected_name=''
        self.visual_cache={}
        from rgb_near_convex_distance import ShoeHandConvexDistance
        self.convex_distance=ShoeHandConvexDistance(model);self.last_surface_detail={}

    def visual_cloud(self,data,body):
        if body not in self.visual_cache:
            descendants=[]
            for b in range(self.model.nbody):
                cur=b
                while cur>0 and cur!=body:cur=int(self.model.body_parentid[cur])
                if cur==body:descendants.append(b)
            pieces=[]
            for g in range(self.model.ngeom):
                if int(self.model.geom_bodyid[g])not in descendants or self.model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or self.model.geom_group[g]!=1:continue
                mid=int(self.model.geom_dataid[g]);start=int(self.model.mesh_vertadr[mid]);count=int(self.model.mesh_vertnum[mid])
                vertices=self.model.mesh_vert[start:start+count]
                if len(vertices)>5000:vertices=vertices[np.linspace(0,len(vertices)-1,5000).round().astype(int)]
                pieces.append((g,vertices.copy()))
            self.visual_cache[body]=pieces
        return np.concatenate([vertices@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]for g,vertices in self.visual_cache[body]])if self.visual_cache[body]else np.empty((0,3))

    def visual_gap(self,data,body):
        from scipy.spatial import cKDTree
        hand=self.visual_cloud(data,body);shoe=self.visual_cloud(data,self.shoe)
        if not len(hand)or not len(shoe):return np.inf
        return float(np.min(cKDTree(hand).query(shoe,k=1,workers=1)[0]))

    def surface_gap(self,data,body):
        descendants=[]
        for b in range(self.model.nbody):
            cur=b
            while cur>0 and cur!=body:cur=int(self.model.body_parentid[cur])
            if cur==body:descendants.append(b)
        actor_geoms=[g for g in range(self.model.ngeom)if int(self.model.geom_bodyid[g])in descendants and self.model.geom_contype[g]]
        shoe_geoms=[g for g in range(self.model.ngeom)if int(self.model.geom_bodyid[g])==self.shoe and self.model.geom_contype[g]]
        best=(np.inf,None,None)
        best_detail={}
        for g in actor_geoms:
            for s in shoe_geoms:
                witness=np.zeros(6);distance=float(mujoco.mj_geomDistance(self.model,data,g,s,1.,witness))
                if distance==0.:
                    try:
                        detail=self.convex_distance.query(data,g,s,precise=True)
                    except ValueError:
                        continue
                    valid=detail['signed_gap_m']<=0 or detail['precise_distance_success']
                    if valid and detail['signed_gap_m']<best[0]:
                        best=(float(detail['signed_gap_m']),g,s);best_detail=detail
                    continue
                reliable=np.isfinite(distance)and np.isfinite(witness).all()and abs(np.linalg.norm(witness[:3]-witness[3:])-abs(distance))<1e-4
                for geom,point in [(g,witness[:3]),(s,witness[3:])]:
                    r=data.geom_xmat[geom].reshape(3,3);centre=data.geom_xpos[geom]+r@self.model.geom_aabb[geom,:3];extent=np.abs(r)@self.model.geom_aabb[geom,3:]
                    reliable=reliable and bool(np.all(np.abs(point-centre)<=extent+1e-4))
                if reliable and distance<best[0]:
                    best=(distance,g,s);best_detail={'backend':'native_mj_geomDistance_nonzero_witness_checked','signed_gap_m':distance}
        self.last_surface_detail=best_detail
        return best

    def apply(self,data,t,step):
        frame=int(np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-1))
        planned=str(self.phase_names[frame])
        for event in self.config.get('events',[]):
            if float(event['start_s'])<=t<float(event.get('end_s',1e9)):
                planned=str(event['name']);break
        if planned in ('none','free','', '-1'):planned=''
        if planned and planned not in self.ids:raise ValueError('Unknown attachment phase '+planned)
        if planned==self.current:return planned,self.current
        if not planned:
            for idx in self.ids.values():data.eq_active[idx]=False
            self.events.append({'step':step,'time_s':float(t),'event':'release','previous':self.current})
            self.current='';return planned,self.current
        idx=self.ids[planned];b1=int(self.model.eq_obj1id[idx]);b2=int(self.model.eq_obj2id[idx])
        if b2!=self.shoe:raise ValueError('Wear/grasp must act between robot and free shoe')
        distance=float(np.linalg.norm(data.xpos[b2]-data.xpos[b1]))
        maxdist=float(self.config.get('max_grasp_activation_body_distance_m',.25)if'grasp'in planned else self.config.get('max_wear_activation_body_distance_m',.35))
        visual_gap=self.visual_gap(data,b1)if distance<=maxdist else np.inf
        if visual_gap<=.03:
            gap,g1,g2=self.surface_gap(data,b1)
        else:
            gap,g1,g2=np.inf,None,None
            self.last_surface_detail={'backend':'not_queried_visual_or_body_distance_guard_failed'}
        proxy_legal=distance<=maxdist and gap<=float(self.config.get('max_activation_surface_gap_m',.03))and gap>=-.02
        if not proxy_legal or visual_gap>.03:
            # Keep a currently worn shoe until the inferred grasp is reachable.
            self.rejected_activations+=1
            if self.last_rejected_name!=planned:
                self.events.append({'step':step,'time_s':float(t),'event':'activation_rejected_geometric_gap','name':planned,'body_distance_m':distance,'surface_gap_m':float(gap)if np.isfinite(gap)else None,'surface_distance_evidence':self.last_surface_detail,'visual_vertex_gap_upper_bound_m':visual_gap if np.isfinite(visual_gap)else None})
                self.last_rejected_name=planned
            return planned,self.current
        r1=data.xmat[b1].reshape(3,3);pos=r1.T@(data.xpos[b2]-data.xpos[b1])
        qinv=data.xquat[b1].copy();qinv[1:]*=-1;quat=np.empty(4);mujoco.mju_mulQuat(quat,qinv,data.xquat[b2])
        self.model.eq_data[idx,3:10]=np.r_[pos,quat]
        for other in self.ids.values():data.eq_active[other]=False
        data.eq_active[idx]=True
        self.events.append({'step':step,'time_s':float(t),'event':'activate_compliant_attachment','name':planned,
                            'previous':self.current,'body1':mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,b1),
                            'body2':'shoe','activation_body_distance_m':distance,'eq_data':self.model.eq_data[idx].tolist(),
                            'activation_surface_gap_m':float(gap),'activation_surface_geom_ids':[g1,g2],
                            'activation_surface_distance_evidence':self.last_surface_detail,
                            'activation_visual_vertex_gap_upper_bound_m':visual_gap,
                            'visual_distance_method':'Nearest points among compiled visual-mesh vertices; at most5000 perlink; unsigned upper bound on true surface distance, not an exact signed mesh distance.',
                            'interpretation':'Inferred wear/grasp approximation; not evidence of a pure friction grasp.'})
        self.current=planned
        self.last_rejected_name=''
        return planned,self.current


def equality_wrenches(model,data,shoe_vadr):
    """Separate solver equality forces from collision contact wrenches."""
    result=np.zeros((model.neq,6));counts=np.zeros(model.neq,dtype=np.int64);per_eq_generalized=np.zeros((model.neq,model.nv))
    if not model.neq:return result,counts,per_eq_generalized
    selected=np.zeros(data.nefc);generalized=np.zeros(model.nv)
    shoejoint=named(model,mujoco.mjtObj.mjOBJ_JOINT,'shoe_free');body=int(model.jnt_bodyid[shoejoint])
    rotation=data.xmat[body].reshape(3,3)
    for eq in range(model.neq):
        ids=np.flatnonzero((data.efc_type[:data.nefc]==int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))&(data.efc_id[:data.nefc]==eq))
        counts[eq]=len(ids)
        if not len(ids):continue
        selected[:]=0.;selected[ids]=data.efc_force[ids]
        mujoco.mj_mulJacTVec(model,data,generalized,selected)
        per_eq_generalized[eq]=generalized
        result[eq,:3]=generalized[shoe_vadr:shoe_vadr+3]
        result[eq,3:]=rotation@generalized[shoe_vadr+3:shoe_vadr+6]
    return result,counts,per_eq_generalized


def grasp_ablation(model_binary,initial,spec,controls,eq_active,eq_data,actual_qpos,shoe_qadr,output):
    model=mujoco.MjModel.from_binary_path(str(model_binary));data=mujoco.MjData(model)
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_EQUALITY,i)or str(i)for i in range(model.neq)]
    grasp=np.asarray([i for i,n in enumerate(names)if'shoe_grasp_'in n],dtype=int)
    if not len(grasp)or not np.any(np.asarray(eq_active)[:,grasp]):
        return {'performed':False,'reason':'No grasp constraint became active','max_shoe_position_difference_m':0.}
    mujoco.mj_setState(model,data,initial,spec);mujoco.mj_forward(model,data)
    xyz=[data.qpos[shoe_qadr:shoe_qadr+3].copy()]
    for step,u in enumerate(controls):
        model.eq_data[:]=eq_data[step];data.eq_active[:]=eq_active[step];data.eq_active[grasp]=False
        data.ctrl[:]=u;mujoco.mj_step(model,data);xyz.append(data.qpos[shoe_qadr:shoe_qadr+3].copy())
    xyz=np.asarray(xyz);delta=np.linalg.norm(xyz-actual_qpos[:,shoe_qadr:shoe_qadr+3],axis=1)
    path=output/'counterfactual_no_grasp_constraint.npz'
    np.savez_compressed(path,shoe_position_world_m=xyz,actual_minus_no_grasp_position_m=delta,
                        removed_equality_names=np.asarray([names[i]for i in grasp]))
    return {'performed':True,'interpretation':'Same frozen controls and wearing phase; only hand/shoe grasp equalities disabled. Collision contacts remain enabled.',
            'removed_equalities':[names[i]for i in grasp],'same_controls_per_step':True,'max_shoe_position_difference_m':float(np.max(delta)),
            'final_shoe_position_difference_m':float(delta[-1]),'trace':receipt(path)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--proxy-model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seconds',type=float)
    p.add_argument('--hold-last-seconds',type=float,default=0.)
    p.add_argument('--sonic-weight',type=float,default=1.)
    p.add_argument('--arm-reference-weight',type=float,default=0.)
    p.add_argument('--gravity-comp',type=float,default=0.)
    p.add_argument('--finger-armature-kg-m2',type=float,default=0.,help='Explicit added reflected driver rotor, not body mass')
    p.add_argument('--seated-balance-gain',type=float,default=0.)
    p.add_argument('--world-arm-feedback',action='store_true')
    p.add_argument('--world-leg-feedback',action='store_true')
    p.add_argument('--root-acceleration-feedback',action='store_true')
    p.add_argument('--root-feedback-objective',choices=['root','root_feet'],default='root')
    p.add_argument('--preserve-worn-leg',action='store_true')
    p.add_argument('--support-arm-stabilizer',action='store_true')
    p.add_argument('--mode',choices=['sonic','pd'],default='sonic')
    p.add_argument('--attachment-config',type=Path)
    p.add_argument('--max-root-horizontal-error-m',type=float,default=.35)
    p.add_argument('--max-root-height-error-m',type=float,default=.25)
    p.add_argument('--max-root-orientation-error-rad',type=float,default=.6)
    p.add_argument('--max-joint-rms-rad',type=float,default=.4)
    p.add_argument('--max-penetration-m',type=float,default=.02)
    p.add_argument('--max-shoe-position-error-m',type=float,default=.2)
    p.add_argument('--min-seat-support-fraction',type=float,default=.35)
    p.add_argument('--max-grasp-constraint-force-N',type=float,default=50.,help='Shoe hand-grasp only; wearing leg support is reported separately')
    a=p.parse_args()
    if sys.platform != 'linux': p.error('Shared physical rollout runs on SSH server only')
    if a.output.exists(): raise FileExistsError('Use a new output directory: '+str(a.output))
    if not all(0 <= x <= 1 for x in [a.sonic_weight,a.arm_reference_weight,a.gravity_comp]): raise ValueError('Invalid controller weights')
    ref=dict(np.load(a.reference,allow_pickle=False)); names=list(map(str,ref['dof_names']))
    times=np.asarray(ref['time_s'],dtype=float); times=times-times[0];ref['time_s']=times
    source_times=np.asarray(ref.get('source_time_s',times),dtype=float)
    source_times=source_times-source_times[0]
    if source_times.shape!=times.shape or np.any(np.diff(source_times)<=0):raise ValueError('Invalid source/execution timeline mapping')
    source_frame_ids=np.asarray(ref.get('frame_ids',np.arange(len(times))))
    if np.any(np.diff(times)<=0):raise ValueError('Source timeline must increase')
    qref=np.asarray(ref['qpos'])
    if qref.shape != (len(times),2,59):raise ValueError('Need T x 2 x 59 named Vita reference')
    objref=np.asarray(ref['object_target_pose'])
    if objref.shape != (len(times),7) or not np.isfinite(qref).all() or not np.isfinite(objref).all():raise ValueError('Reference nonfinite or shape mismatch')
    model=mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    maps=[mapping(model,names,prefix)for prefix in ['p1_','p2_']]
    if model.nu != 104 or model.opt.disableflags or model.opt.gravity[2] > -9:raise ValueError('Need two dynamic full Vita actors and real gravity/contact')
    oj=named(model,mujoco.mjtObj.mjOBJ_JOINT,'shoe_free');oq=int(model.jnt_qposadr[oj]);ov=int(model.jnt_dofadr[oj]);ob=int(model.jnt_bodyid[oj])
    if model.jnt_type[oj]!=mujoco.mjtJoint.mjJNT_FREE or model.body_mocapid[ob]>=0:raise ValueError('Shoe must be free dynamic rigid body')
    controller=None
    if a.mode=='sonic':
        controller=DualSonic(model,a.proxy_model,ref,{'mode':'newft15225','device':a.device,'cpu_threads':2,'delay_steps':3,'imu_latency_s':.0025,'median_training_friction':True},a.sonic_weight,a.arm_reference_weight,a.gravity_comp,a.finger_armature_kg_m2,a.seated_balance_gain,a.world_arm_feedback,a.world_leg_feedback,a.root_acceleration_feedback,a.root_feedback_objective,a.preserve_worn_leg,a.support_arm_stabilizer)
    else:
        for m in maps:
            finger=np.asarray([n.startswith(('lh_','rh_'))for n in names])&(model.actuator_gainprm[m['aids'],0]>0);model.dof_armature[m['va'][finger]]+=a.finger_armature_kg_m2
    a.output.mkdir(parents=True)
    write_json(a.output/'run_config.json',{k:str(v)if isinstance(v,Path)else v for k,v in vars(a).items()})
    shutil.copyfile(__file__,a.output/'source_rgb_near_physics.py')
    shutil.copyfile(ROOT/'scripts/rgb_near_convex_distance.py',a.output/'source_rgb_near_convex_distance.py')
    if controller is not None:shutil.copyfile(ROOT/'scripts/fourth_sonic_controller.py',a.output/'source_fourth_sonic_controller.py')
    if a.world_arm_feedback or a.world_leg_feedback or a.root_acceleration_feedback:shutil.copyfile(ROOT/'scripts/rgb_near_task_feedback.py',a.output/'source_rgb_near_task_feedback.py')
    model_binary=a.output/'model.mjb';mujoco.mj_saveModel(model,str(model_binary),None)
    mujoco.mj_saveLastXML(str(a.output/'model.xml'),model)
    for i,m in enumerate(maps):
        if controller is not None:mujoco.mj_saveModel(controller.parts[i]['proxy'],str(a.output/f'proxy_p{i+1}.mjb'),None)
    data=mujoco.MjData(model)
    for actor,m in enumerate(maps):
        data.qpos[m['rq']:m['rq']+7]=qref[0,actor,:7]
        data.qpos[m['qa']]=qref[0,actor,7:]
    data.qpos[oq:oq+7]=objref[0];data.qvel[:]=0
    mujoco.mj_forward(model,data)
    attachments=ShoeAttachments(model,ref,json.loads(a.attachment_config.read_text())if a.attachment_config else None)
    if controller is not None:controller.reset(data)
    spec=int(mujoco.mjtState.mjSTATE_INTEGRATION);initial=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,data,initial,spec)
    horizon=float(times[-1]+a.hold_last_seconds)
    if a.seconds is not None:horizon=min(horizon,a.seconds)
    dt=float(model.opt.timestep);steps=int(np.ceil(horizon/dt))
    state=[data.qpos.copy()];vel=[data.qvel.copy()];ts=[float(data.time)]
    controls=[];torques=[];rootxy=[];rootz=[];rootori=[];jrms=[];shoeerr=[];rawshoeerr=[];contactrows=[];seat=[];feet=[];hand=[];handimp=np.zeros(2)
    rawobj=np.asarray(ref.get('raw_source_object_pose_xyz_wxyz',objref))
    penetration=0.;reason='requested_duration';warnings=np.array([w.number for w in data.warning]);wall=time.monotonic()
    eqactive=[];eqdata=[];eqforces=[];eqcounts=[];eqgeneralized=[];qfrcconstraint=[];planned_phases=[];actual_phases=[]
    for step in range(steps):
        elapsed=float(data.time);targettime=min(elapsed,times[-1])
        planned,actual=attachments.apply(data,elapsed,step)
        planned_phases.append(planned);actual_phases.append(actual)
        eqactive.append(data.eq_active.copy());eqdata.append(model.eq_data.copy())
        target=np.asarray([[np.interp(targettime,times,qref[:,actor,j])for j in range(59)]for actor in range(2)])
        if controller is not None:u=controller.action(data,elapsed)
        else:
            u=np.zeros(model.nu)
            for actor,m in enumerate(maps):u[m['aids']]=target[actor,7:]
        data.ctrl[:]=u
        if np.any(data.qfrc_applied)or np.any(data.xfrc_applied):raise ValueError('External force injected')
        controls.append(u.copy());before=float(data.time);mujoco.mj_step(model,data)
        wrench,counts,generalized=equality_wrenches(model,data,ov);eqforces.append(wrench);eqcounts.append(counts);eqgeneralized.append(generalized);qfrcconstraint.append(data.qfrc_constraint.copy())
        state.append(data.qpos.copy());vel.append(data.qvel.copy());ts.append(float(data.time));torques.append(data.actuator_force.copy())
        evaluation_time=min(float(data.time),times[-1])
        evaluation_target=np.asarray([[np.interp(evaluation_time,times,qref[:,actor,j])for j in range(59)]for actor in range(2)])
        xy=[];ze=[];oe=[];je=[]
        for actor,m in enumerate(maps):
            xy.append(float(np.linalg.norm(data.qpos[m['rq']:m['rq']+2]-evaluation_target[actor,:2])))
            ze.append(float(abs(data.qpos[m['rq']+2]-evaluation_target[actor,2])))
            aq=data.qpos[m['rq']+3:m['rq']+7];rq=evaluation_target[actor,3:7];rq=rq/np.linalg.norm(rq)
            oe.append(float(2*np.arccos(np.clip(abs(np.dot(aq/np.linalg.norm(aq),rq)),0,1))))
            je.append(float(np.sqrt(np.mean((data.qpos[m['qa']]-evaluation_target[actor,7:])**2))))
        rootxy.append(xy);rootz.append(ze);rootori.append(oe);jrms.append(je)
        obj=np.asarray([np.interp(evaluation_time,times,objref[:,j])for j in range(3)])
        shoeerr.append(float(np.linalg.norm(data.qpos[oq:oq+3]-obj)))
        raw=np.asarray([np.interp(evaluation_time,times,rawobj[:,j])for j in range(3)])
        rawshoeerr.append(float(np.linalg.norm(data.qpos[oq:oq+3]-raw)))
        rows,flags,forces,depth=contact_snapshot(model,data,step);contactrows.extend(rows);seat.append(flags['seat']);feet.append(flags['feet']);hand.append(flags['shoe_hand']);handimp+=np.asarray(forces)*dt;penetration=max(penetration,depth)
        if not np.isfinite(data.qpos).all()or not np.isfinite(data.qvel).all()or abs(data.time-before-dt)>1e-9 or np.any(np.array([w.number for w in data.warning])>warnings):reason='solver_failure';break
        if any(data.qpos[m['rq']+2]<.15 for m in maps):reason='pelvis_below_safety_height';break
    state=np.asarray(state);vel=np.asarray(vel);controls=np.asarray(controls);torques=np.asarray(torques);ts=np.asarray(ts)
    rootxy=np.asarray(rootxy);rootz=np.asarray(rootz);rootori=np.asarray(rootori);jrms=np.asarray(jrms);seat=np.asarray(seat);feet=np.asarray(feet);hand=np.asarray(hand)
    trace=a.output/'trace.npz'
    np.savez_compressed(trace,time=ts,qpos=state,qvel=vel,ctrl=controls,actuator_force=torques,
        initial_state=initial,state_spec=np.asarray(spec),qfrc_applied=np.zeros((len(controls),model.nv)),xfrc_applied=np.zeros((len(controls),model.nbody,6)),
        joint_names=np.asarray(names),joint_qadr=np.asarray([m['qa']for m in maps]),joint_vadr=np.asarray([m['va']for m in maps]),actuator_ids=np.asarray([m['aids']for m in maps]),
        robot_root_qadr=np.asarray([m['rq']for m in maps]),object_qadr=np.asarray(oq),object_vadr=np.asarray(ov),reference_qpos=qref,reference_time_s=times,reference_object_target_pose=objref,
        reference_execution_time_s=times,reference_source_time_s=source_times,source_time_s=np.interp(ts,times,source_times),source_frame_ids=source_frame_ids,
        root_horizontal_error_m=rootxy,root_height_error_m=rootz,root_orientation_error_rad=rootori,joint_rms_rad=jrms,shoe_reference_error_m=np.asarray(shoeerr),raw_source_shoe_error_m=np.asarray(rawshoeerr),seat_support=seat,foot_support=feet,hand_shoe_contact=hand,
        eq_active=np.asarray(eqactive),eq_data=np.asarray(eqdata),eq_constraint_wrench_world_on_shoe=np.asarray(eqforces),eq_constraint_row_count=np.asarray(eqcounts),
        eq_generalized_force=np.asarray(eqgeneralized),qfrc_constraint=np.asarray(qfrcconstraint),qfrc_equality=np.sum(np.asarray(eqgeneralized),axis=1),
        equality_names=np.asarray(attachments.names),equality_type=model.eq_type.copy(),equality_object1_id=model.eq_obj1id.copy(),equality_object2_id=model.eq_obj2id.copy(),
        equality_object_type=np.asarray(['joint'if typ==int(mujoco.mjtEq.mjEQ_JOINT)else'tendon'if typ==int(mujoco.mjtEq.mjEQ_TENDON)else'body'for typ in model.eq_type]),
        equality_body1_id=np.where(np.isin(model.eq_type,[int(mujoco.mjtEq.mjEQ_WELD),int(mujoco.mjtEq.mjEQ_CONNECT)]),model.eq_obj1id,-1),
        equality_body2_id=np.where(np.isin(model.eq_type,[int(mujoco.mjtEq.mjEQ_WELD),int(mujoco.mjtEq.mjEQ_CONNECT)]),model.eq_obj2id,-1),
        shoe_attachment_planned_name=np.asarray(planned_phases),shoe_attachment_actual_name=np.asarray(actual_phases))
    if controller is not None:np.savez_compressed(a.output/'policy_targets.npz',raw_sonic_target=np.asarray(controller.raw_policy_targets),applied_joint_targets=np.asarray(controller.applied_body_targets),raw_learned_action=np.asarray(controller.raw_learned_actions),effective_action_history=np.asarray(controller.effective_action_observations),seated_balance_offset_rad=np.asarray(controller.balance_offsets))
    if controller is not None and controller.world_arm_feedback is not None:
        records=controller.world_arm_feedback.records
        np.savez_compressed(a.output/'world_arm_feedback.npz',time_s=np.asarray([x['time_s']for x in records]),
                            actor=np.asarray([x['actor']for x in records]),side=np.asarray([x['side']for x in records]),
                            acquiring_shoe=np.asarray([x['acquiring_shoe']for x in records]),
                            target_position_world_m=np.asarray([x['target_position_world_m']for x in records]),
                            solved_position_error_m=np.asarray([x['solved_error_m']for x in records]))
    if controller is not None and controller.world_leg_feedback is not None:
        records=controller.world_leg_feedback.records
        np.savez_compressed(a.output/'world_leg_feedback.npz',time_s=np.asarray([x['time_s']for x in records]),
                            actor=np.asarray([x['actor']for x in records]),side=np.asarray([x['side']for x in records]),
                            target_position_world_m=np.asarray([x['target_position_world_m']for x in records]),
                            solved_position_error_m=np.asarray([x['solved_position_error_m']for x in records]))
    if controller is not None and controller.root_acceleration_feedback is not None:
        records=controller.root_acceleration_feedback.records
        np.savez_compressed(a.output/'root_acceleration_feedback.npz',**{key:np.asarray([x[key]for x in records])for key in records[0]})
    def arraycol(i,shape,dtype=float):return np.asarray([r[i]for r in contactrows],dtype=dtype).reshape(shape)
    contacts=a.output/'contacts.npz'
    np.savez_compressed(contacts,interval_index=arraycol(0,(-1,),np.int64),solver_time_s=arraycol(1,(-1,)),geom_ids=arraycol(2,(-1,2),np.int64),body_ids=arraycol(3,(-1,2),np.int64),position_world_m=arraycol(4,(-1,3)),force_world_on_geom2_N=arraycol(5,(-1,3)),torque_world_on_geom2_Nm=arraycol(6,(-1,3)),normal_force_N=arraycol(7,(-1,)),signed_distance_m=arraycol(8,(-1,)),
        geom_names=np.asarray([mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g)or str(g)for g in range(model.ngeom)]),body_names=np.asarray([mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,b)or 'world'for b in range(model.nbody)]))
    ablation=grasp_ablation(model_binary,initial,spec,controls,np.asarray(eqactive),np.asarray(eqdata),state,oq,a.output)
    graspids=[i for i,n in enumerate(attachments.names)if'shoe_grasp_'in n]
    grasp_duration=float(np.sum(np.any(np.asarray(eqactive)[:,graspids],axis=1))*dt)if graspids else 0.
    grasp_impulse=float(np.sum(np.linalg.norm(np.asarray(eqforces)[:,graspids,:3],axis=-1))*dt)if graspids else 0.
    grasp_peak=float(np.max(np.linalg.norm(np.asarray(eqforces)[:,graspids,:3],axis=-1)))if graspids else 0.
    quality={'full_source_duration':reason=='requested_duration'and float(ts[-1])>=float(times[-1])-dt,
             'finite_no_solver_warning':reason!='solver_failure','both_root_horizontal_tracking':bool(np.max(rootxy)<=a.max_root_horizontal_error_m),
             'both_root_height_tracking':bool(np.max(rootz)<=a.max_root_height_error_m),
             'both_root_orientation_tracking':bool(np.max(rootori)<=a.max_root_orientation_error_rad),
             'both_joint_tracking':bool(np.all(np.median(jrms,axis=0)<=a.max_joint_rms_rad)),
             'penetration':penetration<=a.max_penetration_m,
             'both_seated_support':bool(np.all(np.mean(seat,axis=0)>=a.min_seat_support_fraction)),
             'shoe_source_tracking':bool(max(shoeerr)<=a.max_shoe_position_error_m),
             'hand_shoe_manipulation_coupling':bool((np.max(np.sum(hand,axis=0)*dt)>=.1 and np.max(handimp)>=.2)or(grasp_duration>=.1 and grasp_impulse>=.2)),
             'grasp_constraint_causal_motion':ablation['max_shoe_position_difference_m']>=.005,
             'grasp_constraint_force_plausible':grasp_peak<=a.max_grasp_constraint_force_N}
    report={'schema':'rgb_near_dual_vita_physics.v1','status':'passed_assumed_physics_sequence'if all(quality.values())else'failed_physical_sequence',
        'checks':quality,'simulation_passed':bool(all(quality.values())), 'mode':a.mode,'end_reason':reason,
        'controller':controller.report()if controller is not None else 'Full reference PD diagnostic; no learned policy inference',
        'physics_steps':len(controls),'duration_s':float(ts[-1]),'maximum_physics_time_reached_s':float(np.max(ts)),
        'source_duration_s':float(source_times[-1]),'execution_duration_s':float(times[-1]),'source_time_covered_s':float(np.interp(ts[-1],times,source_times)),
        'explicit_retiming':bool(not np.array_equal(source_times,times)),'reference_time_s_semantics':'Execution timeline; raw source timeline and frame IDs saved separately.',
        'source_frames':len(times),'requested_hold_s':a.hold_last_seconds,
        'physics_dt_s':dt,'mujoco_version':mujoco.__version__,'root_horizontal_error_m_max':np.max(rootxy,axis=0).tolist(),'root_height_error_m_max':np.max(rootz,axis=0).tolist(),
        'root_orientation_error_rad_max':np.max(rootori,axis=0).tolist(),
        'joint_rms_rad_median':np.median(jrms,axis=0).tolist(),'max_contact_penetration_m':penetration,'seat_support_fraction':np.mean(seat,axis=0).tolist(),'foot_support_fraction':np.mean(feet,axis=0).tolist(),
        'hand_shoe_contact_duration_s':(np.sum(hand,axis=0)*dt).tolist(),'hand_shoe_normal_impulse_Ns':handimp.tolist(),'shoe_reference_position_error_m_max':max(shoeerr),'raw_source_shoe_position_error_m_max':max(rawshoeerr),'shoe_displacement_m':float(np.linalg.norm(state[-1,oq:oq+3]-state[0,oq:oq+3])),
        'grasp_constraint_active_duration_s':grasp_duration,'grasp_constraint_force_impulse_Ns':grasp_impulse,'grasp_constraint_force_peak_N':grasp_peak,'counterfactual_no_grasp_constraint':ablation,
        'shoe_final_speed_m_s':float(np.linalg.norm(vel[-1,ov:ov+3])),'actuator_torque_abs_peak_Nm':float(np.max(np.abs(torques))),
        'quality_thresholds':{'root_horizontal_error_m_max':a.max_root_horizontal_error_m,'root_height_error_m_max':a.max_root_height_error_m,'root_orientation_error_rad_max':a.max_root_orientation_error_rad,'joint_rms_median_rad_max':a.max_joint_rms_rad,'max_contact_penetration_m':a.max_penetration_m,'max_shoe_position_error_m':a.max_shoe_position_error_m,'min_each_seat_support_fraction':a.min_seat_support_fraction,'max_hand_grasp_constraint_force_N':a.max_grasp_constraint_force_N},
        'active_qpos_qvel_writes':0,'active_external_force_writes':0,'robot_free_roots':2,'shoe_free_nonmocap':True,'assumed_physical_parameters':True,
        'physical_actuator_channels':int(np.sum(model.actuator_gainprm[:,0]!=0)),
        'zero_force_abi_channels':int(np.sum(model.actuator_gainprm[:,0]==0)),
        'equality_generalized_force_units':'Free-joint translational DOFs N; hinge and rotational DOFs N*m. qfrc_constraint includes contact/limit/equality; qfrc_equality separates equality reactions.',
        'assumed_additional_finger_rotor_kg_m2':a.finger_armature_kg_m2,
        'input':{'model_xml':receipt(a.model),'proxy_xml':receipt(a.proxy_model),'reference':receipt(a.reference),'run_config':receipt(a.output/'run_config.json')},'model':receipt(model_binary),'trace':receipt(trace),'contacts':receipt(contacts),
        'attachments':{'model':'Inferred compliant equality constraints for wearing/grasping, not pure friction grasp',
                       'events':attachments.events,'active_duration_s':{name:float(np.sum(np.asarray(actual_phases)==name)*dt)for name in attachments.ids},
                       'rejected_activation_steps':attachments.rejected_activations,'max_activation_surface_gap_m':.03,
                       'force_units':'Each equality solver wrench on shoe in world frame; torque about shoe free-body origin.',
                       'per_equality_peak_force_N':np.max(np.linalg.norm(np.asarray(eqforces)[...,:3],axis=-1),axis=0).tolist()if model.neq else[],
                       'config':receipt(a.attachment_config)if a.attachment_config else'phase names supplied in reference; default distance guards'},
        'script':receipt(a.output/'source_rgb_near_physics.py'),
        'frozen_controller_sources':[receipt(path)for path in sorted(a.output.glob('source_*.py'))],
        'elapsed_wall_s':time.monotonic()-wall,
        'limitations':['Monocular scale and unobserved bench surfaces are estimated.','Collision proxies and physical parameters are explicit assumptions, not measured.','Source fingers were not fitted; any hand contact targets are disclosed synthetic adaptations.','Native MuJoCo drives transfer original SONIC policy across simulator/hand articulation; not identical Isaac training plant.']}
    write_json(a.output/'report.json',report)
    print(json.dumps({k:report[k]for k in ['status','checks','duration_s','root_horizontal_error_m_max','root_height_error_m_max','seat_support_fraction','max_contact_penetration_m','shoe_reference_position_error_m_max']}),flush=True)
    return 0 if report['simulation_passed']else 3


if __name__=='__main__':raise SystemExit(main())
