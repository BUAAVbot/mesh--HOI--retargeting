"""Shared-world dual Vita PPO: 56 joint-target residuals, no root actuation.

Training episode completion is provisional. Independent native MuJoCo rollout
must pass the unchanged trajectory, penetration and actual sitting checks.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn

POLICY_SENSOR_CONTRACT = 'fresh_private_kinematics_v2: actual integrated qpos/qvel mirrored into isolated sensor scratch; kinematics/com_pos/com_vel refresh policy xpos/xmat/cvel without live solver writes; raw nominal action history before physical bounds; body delay3 physical steps except original empty-queue cold start'


def quat_mul(a,b):
    w=a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdim=True)
    xyz=a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+torch.cross(a[...,1:],b[...,1:],dim=-1)
    return torch.cat([w,xyz],-1)


def rotate_inverse(q,v):
    xyz=-q[...,1:];t=2*torch.cross(xyz,v,dim=-1)
    return v+q[...,:1]*t+torch.cross(xyz,t,dim=-1)


def mapping(model,names,prefix=''):
    import mujoco as mj
    def named(kind,name):
        value=mj.mj_name2id(model,kind,name)
        if value<0:raise ValueError('Missing '+name)
        return value
    roots=[prefix+'vita_demo_free_root',prefix+'vita_ik_free_root']
    root=next((mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n)for n in roots
               if mj.mj_name2id(model,mj.mjtObj.mjOBJ_JOINT,n)>=0),-1)
    if root<0 or model.jnt_type[root]!=mj.mjtJoint.mjJNT_FREE:raise ValueError('Need free root '+prefix)
    jids=np.asarray([named(mj.mjtObj.mjOBJ_JOINT,prefix+n)for n in names])
    aids=np.asarray([named(mj.mjtObj.mjOBJ_ACTUATOR,prefix+'motor_'+n)for n in names])
    if len(names)!=52 or len(set(jids))!=52 or len(set(aids))!=52:raise ValueError('Need52 unique named joint channels')
    if not np.array_equal(model.actuator_trnid[aids,0],jids):raise ValueError('Joint transmission mismatch')
    return {'root':root,'rq':int(model.jnt_qposadr[root]),'rv':int(model.jnt_dofadr[root]),
            'jids':jids,'qa':model.jnt_qposadr[jids],'va':model.jnt_dofadr[jids],'aids':aids,
            'base':named(mj.mjtObj.mjOBJ_BODY,prefix+'base_link')}


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim=56):
        super().__init__()
        def net(out):return nn.Sequential(nn.Linear(obs_dim,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,out))
        self.actor=net(action_dim);self.critic=net(1)
        self.log_std=nn.Parameter(torch.full((action_dim,),-1.6))
        for layer in self.modules():
            if isinstance(layer,nn.Linear):nn.init.orthogonal_(layer.weight,2**.5);nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight,.01);nn.init.orthogonal_(self.critic[-1].weight,1.)
        self.register_buffer('obs_mean',torch.zeros(obs_dim));self.register_buffer('obs_var',torch.ones(obs_dim));self.register_buffer('obs_count',torch.tensor(1e-4))

    def normalize(self,x):return ((x-self.obs_mean)/(self.obs_var+1e-6).sqrt()).clamp(-10.,10.)

    @torch.no_grad()
    def update_stats(self,x):
        var,mean=torch.var_mean(x,dim=0,unbiased=False);count=x.shape[0]
        delta=mean-self.obs_mean;total=self.obs_count+count
        self.obs_var.copy_((self.obs_var*self.obs_count+var*count+delta.square()*self.obs_count*count/total)/total)
        self.obs_mean.add_(delta*count/total);self.obs_count.copy_(total)

    def distribution(self,x):return torch.distributions.Normal(self.actor(x),self.log_std.clamp(-3.5,-.3).exp())


class SingleReference:
    def __init__(self, z, device):
        self.z = z
        self.times = torch.as_tensor(z['time_s'], device=device, dtype=torch.float32)
        self.q = torch.as_tensor(z['qpos'], device=device, dtype=torch.float32)
        self.end = float(z['time_s'][-1])
        self.horizon = self.end+1.

    def indices(self, t):
        t = t.clamp(float(self.times[0]), self.end)
        j = torch.searchsorted(self.times, t.contiguous(), right=True).clamp(1, len(self.times)-1)
        i = j-1
        return i, j, ((t-self.times[i])/(self.times[j]-self.times[i])).clamp(0, 1)


class DualReference:
    def __init__(self, model, path, device):
        import mujoco as mj
        self.device = device
        self.z = dict(np.load(path, allow_pickle=False))
        q = np.asarray(self.z['qpos'], float).copy()
        times = np.asarray(self.z['time_s'], float)
        if q.shape != (len(times), 2, 59) or np.any(np.diff(times) <= 0):
            raise ValueError('Expected qpos[T,2,59] and increasing clock')
        for a in range(2):
            for k in range(1, len(q)):
                if np.dot(q[k, a, 3:7], q[k-1, a, 3:7]) < 0:
                    q[k, a, 3:7] *= -1
        self.names = list(map(str, self.z['dof_names']))
        self.maps = [mapping(model, self.names, prefix) for prefix in ('p1_', 'p2_')]
        self.root_qadr = np.asarray([m['rq'] for m in self.maps])
        self.root_vadr = np.asarray([m['rv'] for m in self.maps])
        self.rootqadr, self.rootvadr = self.root_qadr, self.root_vadr
        self.body_cols = np.asarray([j for j,n in enumerate(self.names) if not n.startswith(('lh_', 'rh_', 'neck_'))])
        if len(self.body_cols) != 28:
            raise ValueError('Need28 independently controlled body joints per actor')
        self.body_names = [prefix+self.names[j] for prefix in ('p1_', 'p2_') for j in self.body_cols]
        self.jids = np.concatenate([m['jids'][self.body_cols] for m in self.maps])
        self.qa = model.jnt_qposadr[self.jids]
        self.va = model.jnt_dofadr[self.jids]
        self.body_aids = np.concatenate([m['aids'][self.body_cols] for m in self.maps])
        self.all_aids = np.asarray([m['aids'] for m in self.maps])
        self.all_qa = np.asarray([m['qa'] for m in self.maps])
        self.feet = np.asarray([mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,prefix+side+'_ankle_roll_link')
                                for prefix in ('p1_', 'p2_') for side in ('left', 'right')])
        if min(self.feet) < 0:
            raise ValueError('Missing canonical feet')
        t = lambda x,dtype=torch.float32:torch.as_tensor(x,device=device,dtype=dtype)
        self.times = t(times)
        self.end = float(times[-1]); self.horizon = self.end+1.
        self.source_times = t(self.z.get('source_time_s',times))
        self.qa_t = t(self.qa,torch.long); self.va_t = t(self.va,torch.long)
        self.body_aids_t = t(self.body_aids,torch.long)
        self.all_aids_t = t(self.all_aids,torch.long); self.all_qa_t = t(self.all_qa,torch.long)
        self.feet_t = t(self.feet,torch.long)
        self.root_qadr_t=t(self.root_qadr,torch.long);self.root_vadr_t=t(self.root_vadr,torch.long)
        self.singles = [SingleReference({'qpos':q[:,a].copy(),'time_s':times.copy(),
                         'source_time_s':np.asarray(self.z.get('source_time_s',times)).copy(),
                         'dof_names':np.asarray(self.names)},device) for a in range(2)]
        flat = np.tile(model.qpos0,(len(times),1))
        for a,m in enumerate(self.maps):
            flat[:,m['rq']:m['rq']+7] = q[:,a,:7]
            flat[:,m['qa']] = q[:,a,7:]
        data=mj.MjData(model);positions=[];vel=[]
        for k in range(len(times)):
            data.qpos[:]=flat[k];mj.mj_fwdPosition(model,data)
            positions.append(data.xpos[self.feet].copy())
            v=np.zeros(model.nv)
            if k<len(times)-1:mj.mj_differentiatePos(model,v,times[k+1]-times[k],flat[k],flat[k+1])
            vel.append(v)
        self.q=t(flat);self.dq=t(np.asarray(vel));self.feet_ref=t(np.asarray(positions))
        self.foot_support=t(np.asarray(self.z.get('foot_support_expected',np.ones((len(times),2,2))),float).reshape(len(times),4))
        self.seat_support=t(np.asarray(self.z.get('seat_support_expected',np.zeros((len(times),2))),float))
        self.software_wrist_command_windows={}
        self.refresh_actuation(model)

    def refresh_actuation(self,model):
        self.passive=np.flatnonzero(np.all(model.actuator_gear==0,axis=1))
        self.passive_t=torch.as_tensor(self.passive,device=self.device,dtype=torch.long)
        self.lower=torch.as_tensor(model.jnt_range[self.jids,0],device=self.device,dtype=torch.float32)
        self.upper=torch.as_tensor(model.jnt_range[self.jids,1],device=self.device,dtype=torch.float32)

    def indices(self,t):return self.singles[0].indices(t)

    def sample(self,t):
        i,j,w=self.indices(t)
        q=self.q[i]+w[:,None]*(self.q[j]-self.q[i])
        for rq in self.root_qadr:
            q[:,rq+3:rq+7]/=q[:,rq+3:rq+7].norm(dim=-1,keepdim=True).clamp_min(1e-8)
        dq=self.dq[i]+w[:,None]*(self.dq[j]-self.dq[i])
        dq=torch.where((t>self.end)[:,None],torch.zeros_like(dq),dq)
        feet=self.feet_ref[i]+w[:,None,None]*(self.feet_ref[j]-self.feet_ref[i])
        return q,dq,feet

    def support_at(self,t):
        i,j,w=self.indices(t)
        return (self.foot_support[i]+w[:,None]*(self.foot_support[j]-self.foot_support[i]),
                self.seat_support[i]+w[:,None]*(self.seat_support[j]-self.seat_support[i]))

    def source_clock(self,t):
        i,j,w=self.indices(t);return self.source_times[i]+w*(self.source_times[j]-self.source_times[i])

    def controls(self,phase,actions,scale=.25):
        target,_,_=self.sample(phase)
        u=torch.zeros((len(phase),104),device=self.device)
        u[:,self.all_aids_t.flatten()]=target[:,self.all_qa_t.flatten()]
        u[:,self.body_aids_t]=(target[:,self.qa_t]+scale*torch.tanh(actions)).clamp(self.lower,self.upper)
        u[:,self.passive_t]=0.
        return u

    def observations(self,q,v,xpos,phase,previous):
        target,dq,feet=self.sample(phase);future,_,_=self.sample(phase+.2)
        obs=[]
        for a,(rq,rv)in enumerate(zip(self.root_qadr,self.root_vadr)):
            actual=q[:,rq+3:rq+7];actual=actual/actual.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            inverse=torch.cat([actual[:,:1],-actual[:,1:]],-1)
            rotation=quat_mul(inverse,target[:,rq+3:rq+7]);rotation=torch.where(rotation[:,:1]<0,-rotation,rotation)
            ids=self.qa_t[a*28:(a+1)*28];vs=self.va_t[a*28:(a+1)*28]
            delta=target[:,rq:rq+3]-q[:,rq:rq+3]
            obs.extend([delta,rotate_inverse(actual,delta),rotation,
                        q[:,ids]-target[:,ids],v[:,vs]*.1,
                        rotate_inverse(actual,v[:,rv:rv+3]-dq[:,rv:rv+3])*.25,
                        (v[:,rv+3:rv+6]-dq[:,rv+3:rv+6])*.25,
                        dq[:,vs]*.1,future[:,ids]-q[:,ids]])
        fgoal,seat=self.support_at(phase)
        r0,r1=self.root_qadr;v0,v1=self.root_vadr
        obs.extend([previous,(feet-xpos[:,self.feet_t]).reshape(-1,12),fgoal,seat,
                    q[:,r1:r1+2]-q[:,r0:r0+2],v[:,v1:v1+2]-v[:,v0:v0+2],
                    (phase/self.horizon)[:,None]])
        return torch.cat(obs,dim=-1).clamp(-10.,10.)


Reference=DualReference


class WarpTracking:
    dt=.02

    @property
    def training_curriculum_contract(self):
        strict=bool(getattr(self,'strict_training',False))
        limits=(.35,.20,.6)if strict else(.65,.30,1.2)
        return {'strict_training':strict,
                'training_failure_limits':{'root_xy_m':limits[0],'root_z_m':limits[1],'root_orientation_rad':limits[2]},
                'dense_root_position_cost':'-2*mean_over_two_actors(root_xy_error_m**2)',
                'dense_root_position_cost_weight':2.,
                'native_final_acceptance_unchanged':{'root_xy_m':.25,'root_z_m':.18,'root_orientation_rad':.65,'penetration_m':.025},
                'scope':'Training curriculum only; training completion is not independent native acceptance.'}

    def __init__(self,model_path,reference_path,n=128,seed=1,action_scale=.25,sonic_base=False,proxy_path=None):
        import mujoco as mj,mujoco_warp as mw,warp as wp
        self.mj,self.mw,self.wp=mj,mw,wp;self.device='cuda:0';self.n=int(n);self.action_scale=float(action_scale)
        torch.manual_seed(seed);wp.init();wp.set_device(self.device)
        self.model=mj.MjModel.from_xml_path(str(Path(model_path).resolve()))
        if (self.model.nq,self.model.nv,self.model.nu)!=(118,116,104):raise ValueError('Need118/116/104 dual plant')
        if abs(self.model.opt.timestep-.005)>1e-9:raise ValueError('Need5ms physics for20ms policy')
        if self.model.nmocap:raise ValueError('Mocap forbidden')
        self.ref=DualReference(self.model,reference_path,self.device)
        if any(int(j)in[m['root']for m in self.ref.maps]for j in self.model.actuator_trnid[:,0]):raise ValueError('Root actuators forbidden')
        self.sonic=None
        if sonic_base:
            if proxy_path is None:raise ValueError('Need proxy_path for frozen SONIC base')
            from two_sitting_batch_sonic import DualBatchSonic
            self.sonic=DualBatchSonic(self.model,self.ref,self.n,self.device,proxy_path)
        self.ref.refresh_actuation(self.model)
        self.cpu_data=mj.MjData(self.model);self.cpu_data.qpos[:]=self.ref.q[0].cpu().numpy();mj.mj_forward(self.model,self.cpu_data)
        self.m=mw.put_model(self.model);self.d=mw.put_data(self.model,self.cpu_data,nworld=self.n,nconmax=256,njmax=1024)
        self.q=wp.to_torch(self.d.qpos);self.v=wp.to_torch(self.d.qvel);self.u=wp.to_torch(self.d.ctrl)
        self._physics_xpos=wp.to_torch(self.d.xpos);self._physics_rotation=wp.to_torch(self.d.xmat);self._physics_cvel=wp.to_torch(self.d.cvel)
        free_before=torch.cuda.mem_get_info()[0]
        self.reset_d=mw.put_data(self.model,self.cpu_data,nworld=self.n,nconmax=256,njmax=1024)
        self.reset_scratch_allocated_bytes=free_before-torch.cuda.mem_get_info()[0]
        self._reset_q=wp.to_torch(self.reset_d.qpos);self._reset_v=wp.to_torch(self.reset_d.qvel);self._reset_u=wp.to_torch(self.reset_d.ctrl)
        self._reset_xpos=wp.to_torch(self.reset_d.xpos);self._reset_rotation=wp.to_torch(self.reset_d.xmat);self._reset_cvel=wp.to_torch(self.reset_d.cvel)
        self.xpos=self._physics_xpos.clone();self.rotation=self._physics_rotation.clone();self.cvel=self._physics_cvel.clone()
        self.policy_sensor_contract=POLICY_SENSOR_CONTRACT
        self.phase=torch.zeros(self.n,device=self.device);self.previous=torch.zeros((self.n,56),device=self.device)
        self.previous_body_target=torch.zeros((self.n,56),device=self.device)
        self.cold_start=torch.ones(self.n,dtype=torch.bool,device=self.device)
        self.from_start=torch.zeros(self.n,dtype=torch.bool,device=self.device);self.steps=torch.zeros(self.n,dtype=torch.long,device=self.device)
        self.start_successes=0;self.starts=0;self.max_from_start_source_s=0.;self.nan_episodes=0
        self._prepare_contact_metadata()
        self.u.copy_(self.ref.controls(self.phase,self.previous,self.action_scale));mw.step(self.m,self.d);wp.synchronize()
        with wp.ScopedCapture()as capture:
            for _ in range(3):mw.step(self.m,self.d)
        self.graph3=capture.graph
        with wp.ScopedCapture()as capture:mw.step(self.m,self.d)
        self.graph1=capture.graph
        self.reset(torch.arange(self.n,device=self.device))

    def _prepare_contact_metadata(self):
        mj=self.mj;model=self.model
        gn=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,g)or''for g in range(model.ngeom)]
        bn=[mj.mj_id2name(model,mj.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[g]))or''for g in range(model.ngeom)]
        t=lambda x,dtype:torch.as_tensor(x,device=self.device,dtype=dtype)
        self.geom_actor=t([0 if b.startswith('p1_')else 1 if b.startswith('p2_')else -1 for b in bn],torch.long)
        self.geom_seat=t(['seat'in g.lower()or'stool'in g.lower()for g in gn],torch.bool)
        self.geom_pelvis=t([any(s in b for s in('base_link','hip_','pelvis','thigh'))for b in bn],torch.bool)
        self.contact_force_available=hasattr(self.d,'efc')and hasattr(self.d.efc,'force')
        self.contact_contract='actual Warp contact geometry, elliptic positive constraint normal force and upward seat reaction'if self.contact_force_available else'actual contact geometry only; no load-bearing claim'

    def contact_summary(self):
        wp=self.wp;n=self.n
        seat=torch.zeros((n,2),device=self.device);cross=torch.zeros(n,device=self.device)
        pen=torch.zeros(n,device=self.device)
        total=int(wp.to_torch(self.d.nacon)[0].item())
        if not total:return seat,cross,pen
        geom=wp.to_torch(self.d.contact.geom)[:total].long()
        world=wp.to_torch(self.d.contact.worldid)[:total].long()
        dist=wp.to_torch(self.d.contact.dist)[:total]
        valid=(world>=0)&(world<n)&(geom[:,0]>=0)&(geom[:,1]>=0)
        geom=geom[valid];world=world[valid];dist=dist[valid]
        pen.scatter_reduce_(0,world,(-dist).clamp_min(0),reduce='amax',include_self=True)
        actors=self.geom_actor[geom]
        bilateral=(actors[:,0]>=0)&(actors[:,1]>=0)&(actors[:,0]!=actors[:,1])
        cross.scatter_add_(0,world,bilateral.float()*((-dist).clamp_min(0)+.005))
        if not self.contact_force_available:return seat,cross,pen
        address=wp.to_torch(self.d.contact.efc_address)[:total][valid].long()
        forces=wp.to_torch(self.d.efc.force)
        good=(address[:,0]>=0)&(address[:,0]<forces.shape[1])
        normal=torch.zeros(len(world),device=self.device)
        # This task's plant uses the elliptic cone. For another cone, do not
        # mistake one pyramid ray for a normal force.
        if int(self.model.opt.cone)!=int(self.mj.mjtCone.mjCONE_ELLIPTIC):
            raise ValueError('Seat force reward requires declared elliptic contact cone')
        normal[good]=forces[world[good],address[good,0]].clamp_min(0)
        frame=wp.to_torch(self.d.contact.frame)[:total][valid].reshape(-1,3,3)
        for side in range(2):
            other=1-side;robot_geom=geom[:,side];scene_geom=geom[:,other]
            up=normal*frame[:,0,2]*(1 if side==1 else -1)
            for actor in range(2):
                mask=(actors[:,side]==actor)&self.geom_seat[scene_geom]&self.geom_pelvis[robot_geom]&(up>.5)&(dist<=.002)
                seat[:,actor].scatter_add_(0,world,torch.where(mask,up,torch.zeros_like(up)))
        return seat,cross,pen

    def refresh_policy_sensors(self,ids=None):
        # A separate scratch batch supplies the same fresh kinematics as the
        # original single-robot inference proxy's mj_forward. These three APIs
        # do not perform collision/constraint solving or touch live warmstarts.
        self._reset_q.copy_(self.q);self._reset_v.copy_(self.v)
        self.mw.kinematics(self.m,self.reset_d)
        self.mw.com_pos(self.m,self.reset_d)
        self.mw.com_vel(self.m,self.reset_d)
        for cached,fresh in((self.xpos,self._reset_xpos),(self.rotation,self._reset_rotation),(self.cvel,self._reset_cvel)):
            if ids is None:cached.copy_(fresh)
            else:cached[ids]=fresh[ids]

    def reset(self,ids,force_start=False):
        if not len(ids):return
        phase=torch.rand(len(ids),device=self.device)*self.ref.end
        start=torch.rand(len(ids),device=self.device)<getattr(self,'start_probability',.30)
        if force_start:start[:]=True
        phase[start]=0.;self.phase[ids]=phase;self.from_start[ids]=start;self.starts+=int(start.sum());self.steps[ids]=0
        mask=torch.zeros(self.n,dtype=torch.bool,device=self.device);mask[ids]=True
        self.mw.reset_data(self.m,self.d,self.wp.from_torch(mask,dtype=self.wp.bool))
        target,velocity,_=self.ref.sample(phase)
        velocity[start]=0.
        self.q[ids]=target;self.v[ids]=velocity
        self.previous[ids]=0.;self.u[ids]=self.ref.controls(phase,self.previous[ids],self.action_scale)
        self.previous_body_target[ids]=self.u[ids][:,self.ref.body_aids_t]
        self.cold_start[ids]=True
        if self.sonic is not None:self.sonic.reset(ids)
        self.mw.reset_data(self.m,self.reset_d,self.wp.from_torch(mask,dtype=self.wp.bool))
        self._reset_q[ids]=target;self._reset_v[ids]=velocity;self._reset_u[ids]=self.u[ids]
        self.mw.forward(self.m,self.reset_d)
        self.xpos[ids]=self._reset_xpos[ids];self.rotation[ids]=self._reset_rotation[ids];self.cvel[ids]=self._reset_cvel[ids]

    def base_controls(self):
        self.refresh_policy_sensors()
        if self.sonic is None:return self.ref.controls(self.phase,torch.zeros_like(self.previous),self.action_scale)
        return self.sonic.targets(self.q,self.v,self.phase,self.rotation,self.cvel).clone()

    def observations(self):
        baseline=self.base_controls()
        obs=self.ref.observations(self.q,self.v,self.xpos,self.phase,self.previous)
        if self.sonic is not None:
            target,_,_=self.ref.sample(self.phase)
            obs=torch.cat([obs,baseline[:,self.ref.body_aids_t]-target[:,self.ref.qa_t]],dim=-1)
        return torch.nan_to_num(obs)

    def step(self,actions):
        packet=self.base_controls()
        nominal=packet.clone()
        nominal[:,self.ref.body_aids_t]+=self.action_scale*torch.tanh(actions)
        body=nominal[:,self.ref.body_aids_t].clamp(self.ref.lower,self.ref.upper)
        packet[:,self.ref.body_aids_t]=body;packet[:,self.ref.passive_t]=0.
        # Three physical steps retain the previous body command, while neck /
        # hands receive the current auxiliary reference. New body target acts
        # on the fourth step, matching a3-step delay at5ms physics.
        delayed_body=torch.where(self.cold_start[:,None],body,self.previous_body_target)
        self.u.copy_(packet);self.u[:,self.ref.body_aids_t]=delayed_body
        self.wp.capture_launch(self.graph3)
        self.u.copy_(packet);self.wp.capture_launch(self.graph1)
        self.previous_body_target.copy_(body)
        self.cold_start[:]=False
        if self.sonic is not None:self.sonic.record_applied(nominal)
        self.refresh_policy_sensors();self.phase+=self.dt;self.steps+=1
        target,velocity,feet=self.ref.sample(self.phase)
        xy=[];z=[];angles=[];vel=[]
        for rq,rv in zip(self.ref.root_qadr,self.ref.root_vadr):
            error=self.q[:,rq:rq+3]-target[:,rq:rq+3]
            xy.append(error[:,:2].norm(dim=-1));z.append(error[:,2].abs())
            angles.append(2*torch.acos((self.q[:,rq+3:rq+7]*target[:,rq+3:rq+7]).sum(-1).abs().clamp(0,1)))
            vel.append((self.v[:,rv:rv+3]-velocity[:,rv:rv+3]).square().mean(-1))
        xy=torch.stack(xy,1);z=torch.stack(z,1);ori=torch.stack(angles,1);vel=torch.stack(vel,1)
        pose=(self.q[:,self.ref.qa_t]-target[:,self.ref.qa_t]).square().reshape(self.n,2,28).mean(-1)
        footerr=(self.xpos[:,self.ref.feet_t]-feet).square().mean((1,2))
        change=(torch.tanh(actions)-torch.tanh(self.previous)).square().mean(-1)
        seat_force,cross,pen=self.contact_summary();_,expected_seat=self.ref.support_at(self.phase)
        seat_match=((seat_force>1.).float()*expected_seat).mean(-1)
        reward=(3.*torch.exp(-xy.square()/.15**2).mean(-1)+torch.exp(-40*z.square()).mean(-1)+
                torch.exp(-5*ori.square()).mean(-1)+torch.exp(-5*pose).mean(-1)+
                torch.exp(-25*footerr)+.5*torch.exp(-2*vel).mean(-1)+seat_match-
                .03*change-.003*actions.square().mean(-1)-5.*cross-2.*(pen-.025).clamp_min(0)-
                2.*xy.square().mean(-1))
        finite=torch.isfinite(self.q).all(-1)&torch.isfinite(self.v).all(-1)
        # Curriculum termination is separate from unchanged native acceptance.
        xy_limit,z_limit,ori_limit=(.35,.20,.6)if getattr(self,'strict_training',False)else(.65,.30,1.2)
        failed=(xy.max(-1).values>xy_limit)|(z.max(-1).values>z_limit)|(ori.max(-1).values>ori_limit)|~finite
        timeout=self.phase>=self.ref.horizon;done=failed|timeout
        reward=torch.nan_to_num(reward,nan=-2.)-2.*failed.float()
        self.previous.copy_(actions)
        if self.from_start.any():self.max_from_start_source_s=max(self.max_from_start_source_s,float(self.ref.source_clock(self.phase[self.from_start]).max()))
        # A training endpoint must at least include both actual upward seat forces.
        end_seated=(seat_force>1.).all(-1)
        self.start_successes+=int((timeout&~failed&self.from_start&end_seated).sum());self.nan_episodes+=int((~finite).sum())
        info={'failed':failed,'timeout':timeout,'root_xy':xy.max(-1).values,'root_angle':ori.max(-1).values,
              'root_xy_per_actor':xy,'root_angle_per_actor':ori,'pose_mse':pose.mean(-1),'feet_mse':footerr,
              'seat_support_force_N':seat_force,'cross_actor_contact_penalty':cross,'max_penetration_m':pen,
              'training_endpoint_seated':end_seated}
        return self.observations(),reward*.1,done,info
