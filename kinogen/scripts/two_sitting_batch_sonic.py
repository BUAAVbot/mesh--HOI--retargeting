"""Two batched frozen SONIC actors for one shared-world dual-Vita plant.

The actor adapter has no internal delay because the shared environment applies
the original three physical steps of body-command delay. Mechanical clipping
does not feed back into the learned actor's previous-action observation.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from two_sitting_ppo_env import rotate_inverse,quat_mul


def dependency_paths():
    for path in ('/data/grail-envs/retarget/lib/python3.10/site-packages','/data/grail-envs/grail/lib/python3.10/site-packages','/data/grail-envs/tracking45/lib/python3.10/site-packages'):
        if path not in sys.path:sys.path.append(path)


class BatchSonic:
    def __init__(self,model,reference,n,device):
        dependency_paths()
        from fourth_sonic_controller import Controller
        self.c=Controller(model,dict(reference.z),{'mode':'newft15225','device':device,'cpu_threads':2,'delay_steps':0,'imu_latency_s':.0025,'hand_object_feedback':False})
        self.ref=reference;self.n=n;self.device=device;self.policy=self.c.policy.eval()
        for param in self.policy.parameters():param.requires_grad_(False)
        def tensor(v,dtype=torch.float32):return torch.as_tensor(v,device=device,dtype=dtype)
        self.qa=tensor(self.c.qa,torch.long);self.va=tensor(self.c.va,torch.long);self.aids=tensor(self.c.aids,torch.long)
        self.default=tensor(self.c.default);self.scale=tensor(self.c.scale);self.qf=tensor(self.c.body_q);self.dqf=tensor(self.c.body_dq)
        self.history={k:torch.zeros((n,10,width),device=device)for k,width in [('base_ang_vel',3),('joint_pos',28),('joint_vel',28),('actions',28),('gravity_dir',3)]}
        self.valid=torch.zeros(n,dtype=torch.bool,device=device);self.phase=torch.full((n,),-1.,device=device)
        self.last=torch.zeros((n,28),device=device);self.previous_imu=torch.zeros((n,3),device=device);self.previous_gravity=torch.zeros((n,3),device=device)
        self.cached=torch.zeros((n,28),device=device);self.tokens={k:v.expand(n,*v.shape[1:]).clone()for k,v in self.c.tokenizer.items()}
        self.raw_network_action=torch.zeros((n,28),device=device)
        self.raw_body_target=torch.zeros((n,28),device=device)
        self.inference_count=0

    def reset(self,ids):
        self.valid[ids]=False;self.phase[ids]=-1.;self.last[ids]=0.
        self.raw_network_action[ids]=0.;self.raw_body_target[ids]=0.

    @torch.no_grad()
    def targets(self,q,v,t,rotation=None,angular_world=None):
        ids=((self.phase!=t)|~self.valid).nonzero().flatten()
        if not len(ids):return self.cached
        actual=q[ids,3:7];actual=actual/actual.norm(dim=-1,keepdim=True).clamp_min(1e-9)
        imu=v[ids,3:6];gravity=rotate_inverse(actual,torch.tensor([0.,0.,-1.],device=self.device).expand(len(ids),3))
        if rotation is not None and angular_world is not None:
            r=rotation[ids];imu=torch.einsum('nij,nj->ni',r.transpose(-1,-2),angular_world[ids]);gravity=-r[:,2,:]
        was=self.valid[ids];observed=torch.where(was[:,None],.875*imu+.125*self.previous_imu[ids],imu);gravobs=torch.where(was[:,None],.875*gravity+.125*self.previous_gravity[ids],gravity)
        self.previous_imu[ids]=imu;self.previous_gravity[ids]=gravity
        current={'base_ang_vel':observed,'joint_pos':q[ids][:,self.qa]-self.default,'joint_vel':v[ids][:,self.va],'actions':self.last[ids],'gravity_dir':gravobs}
        for name,value in current.items():
            shifted=torch.cat([self.history[name][ids,1:],value[:,None]],dim=1)
            self.history[name][ids]=torch.where(was[:,None,None],shifted,value[:,None].expand(-1,10,-1))
        proprio=torch.cat([self.history[name][ids].reshape(len(ids),-1)for name in self.c.terms],dim=-1)
        future_t=(t[ids,None]+torch.arange(10,device=self.device)[None]*.1).clamp_max(self.ref.end);i,j,w=self.ref.indices(future_t.flatten())
        qp=(self.qf[i]+w[:,None]*(self.qf[j]-self.qf[i])).reshape(len(ids),10,28)
        dq=(self.dqf[i]+w[:,None]*(self.dqf[j]-self.dqf[i])).reshape(len(ids),10,28)
        future=torch.cat([qp.flatten(1),dq.flatten(1)],dim=-1).reshape(len(ids),1,10,56)
        fq=(self.ref.q[i,3:7]+w[:,None]*(self.ref.q[j,3:7]-self.ref.q[i,3:7])).reshape(len(ids),10,4);fq/=fq.norm(dim=-1,keepdim=True).clamp_min(1e-9)
        qw,qx,qy,qz=actual.unbind(-1);yaw=torch.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));heading=torch.zeros_like(actual);heading[:,0]=torch.cos(yaw/2);heading[:,3]=-torch.sin(yaw/2)
        if rotation is not None:
            r=rotation[ids];yaw=torch.atan2(r[:,1,0],r[:,0,0]);heading[:,0]=torch.cos(yaw/2);heading[:,3]=-torch.sin(yaw/2)
        rel=quat_mul(heading[:,None].expand(-1,10,-1),fq);rw,rx,ry,rz=rel.unbind(-1)
        orientation=torch.stack([1-2*(ry*ry+rz*rz),2*(rx*ry-rw*rz),2*(rx*ry+rw*rz),1-2*(rx*rx+rz*rz),2*(rx*rz-rw*ry),2*(ry*rz+rw*rx)],dim=-1)[:,None]
        self.tokens['command_multi_future_nonflat'][ids]=future;self.tokens['motion_anchor_ori_heading_mf_nonflat'][ids]=orientation
        tokenizer=torch.cat([self.tokens[name][ids].reshape(len(ids),1,-1)for name in self.c.cfg.env_config.obs.group_obs_names.tokenizer],dim=-1)
        inputs=self.c.TensorDict({'actor_obs':proprio[:,None],'tokenizer':tokenizer},batch_size=[len(ids),1])
        action=self.policy(inputs).reshape(len(ids),28).clamp(-float(self.c.cfg.env_config.action_clip_value),float(self.c.cfg.env_config.action_clip_value))
        self.raw_network_action[ids]=action
        self.raw_body_target[ids]=self.default+self.scale*action
        self.cached[ids]=self.raw_body_target[ids]
        # Match Token15225Controller immediately after inference. A task
        # residual may subsequently replace this with its nominal command,
        # but mechanical joint-limit clipping never enters this history.
        self.last[ids]=action
        self.valid[ids]=True;self.phase[ids]=t[ids];self.inference_count+=len(ids)
        return self.cached

    def record_applied(self,nominal_ctrl):
        """Record pre-limit/pre-delay SONIC target plus any task residual.

        The historical method name is retained for caller compatibility;
        callers must not pass mechanically clipped or delayed controls.
        With zero task residual this is exactly the raw network action.
        """
        self.last.copy_((nominal_ctrl[:,self.aids]-self.default)/self.scale)


class DualBatchSonic:
    """Return complete shared 104-channel controls from two real 28-output actors.

    The constructor mutates the shared model's motor/rotor/hard-limit parameters
    to match the two configured inference proxies. Call before Warp put_model.
    """

    def __init__(self, shared_model, reference, n, device, proxy_path):
        import mujoco as mj
        self.model=shared_model;self.ref=reference;self.n=int(n);self.device=device
        if len(reference.singles)!=2 or shared_model.nu!=104:
            raise ValueError('Dual SONIC requires two references and104 actuator channels')
        if abs(float(shared_model.opt.timestep)-.005)>1e-9:
            raise ValueError('Dual SONIC requires physics_dt=.005')
        self.parts=[];self.synchronized_fields={}
        def tensor(x):return torch.as_tensor(np.asarray(x),device=device,dtype=torch.long)
        def named(model,kind,name):
            value=mj.mj_name2id(model,kind,name)
            if value<0:raise ValueError('Missing named dual/proxy model entry '+name)
            return value
        for actor,prefix in enumerate(('p1_','p2_')):
            proxy=mj.MjModel.from_xml_path(str(Path(proxy_path).resolve()))
            ref=reference.singles[actor]
            batch=BatchSonic(proxy,ref,self.n,device)
            names=list(map(str,ref.z['dof_names']))
            if len(names)!=52:raise ValueError('Need all52 named Vita joints')
            pj=np.array([named(proxy,mj.mjtObj.mjOBJ_JOINT,name)for name in names])
            sj=np.array([named(shared_model,mj.mjtObj.mjOBJ_JOINT,prefix+name)for name in names])
            pa=np.array([named(proxy,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+name)for name in names])
            sa=np.array([named(shared_model,mj.mjtObj.mjOBJ_ACTUATOR,prefix+'motor_'+name)for name in names])
            pfree=next((mj.mj_name2id(proxy,mj.mjtObj.mjOBJ_JOINT,name)for name in ('vita_demo_free_root','vita_ik_free_root')if mj.mj_name2id(proxy,mj.mjtObj.mjOBJ_JOINT,name)>=0),-1)
            sfree=next((mj.mj_name2id(shared_model,mj.mjtObj.mjOBJ_JOINT,prefix+name)for name in ('vita_demo_free_root','vita_ik_free_root')if mj.mj_name2id(shared_model,mj.mjtObj.mjOBJ_JOINT,prefix+name)>=0),-1)
            if pfree<0 or sfree<0:raise ValueError('Both proxy andshared robot require free roots')
            if proxy.jnt_type[pfree]!=mj.mjtJoint.mjJNT_FREE or shared_model.jnt_type[sfree]!=mj.mjtJoint.mjJNT_FREE:
                raise ValueError('Root must be an unactuated freejoint')
            pqa,pva=proxy.jnt_qposadr[pj],proxy.jnt_dofadr[pj]
            sqa,sva=shared_model.jnt_qposadr[sj],shared_model.jnt_dofadr[sj]
            actuator_keys=('actuator_gainprm','actuator_biasprm','actuator_dynprm','actuator_gaintype','actuator_biastype','actuator_dyntype','actuator_ctrllimited','actuator_ctrlrange','actuator_forcelimited','actuator_forcerange','actuator_gear')
            rotor_keys=('dof_armature','dof_frictionloss','dof_damping')
            joint_keys=('jnt_range','jnt_limited')
            for key in actuator_keys:getattr(shared_model,key)[sa]=getattr(proxy,key)[pa]
            for key in rotor_keys:getattr(shared_model,key)[sva]=getattr(proxy,key)[pva]
            for key in joint_keys:getattr(shared_model,key)[sj]=getattr(proxy,key)[pj]
            for key in actuator_keys:
                if not np.array_equal(getattr(shared_model,key)[sa],getattr(proxy,key)[pa]):raise ValueError('Plant motor sync mismatch '+key)
            for key in rotor_keys:
                if not np.array_equal(getattr(shared_model,key)[sva],getattr(proxy,key)[pva]):raise ValueError('Plant rotor sync mismatch '+key)
            body=named(shared_model,mj.mjtObj.mjOBJ_BODY,prefix+'base_link')
            learned=np.array([named(shared_model,mj.mjtObj.mjOBJ_ACTUATOR,prefix+'motor_'+name)for name in batch.c.names])
            self.parts.append(dict(actor=actor,prefix=prefix,proxy=proxy,batch=batch,
                pqa=tensor(pqa),pva=tensor(pva),sqa=tensor(sqa),sva=tensor(sva),pa=tensor(pa),sa=tensor(sa),
                prq=int(proxy.jnt_qposadr[pfree]),prv=int(proxy.jnt_dofadr[pfree]),
                srq=int(shared_model.jnt_qposadr[sfree]),srv=int(shared_model.jnt_dofadr[sfree]),
                base=body,learned_aids=tensor(learned),
                q=torch.zeros((self.n,proxy.nq),device=device),
                v=torch.zeros((self.n,proxy.nv),device=device),
                control=torch.zeros((self.n,proxy.nu),device=device)))
            self.synchronized_fields[prefix]=dict(actuator=list(actuator_keys),rotor=list(rotor_keys),joint=list(joint_keys))
        self.body_aids_t=reference.body_aids_t
        self.aids=self.body_aids_t
        self.all_aids_t=reference.all_aids_t
        self.inactive=torch.as_tensor((shared_model.actuator_gainprm[:,0]==0)&np.all(shared_model.actuator_biasprm==0,axis=1),device=device)
        self.output=torch.zeros((self.n,shared_model.nu),device=device)

    def reset(self,ids):
        for part in self.parts:part['batch'].reset(ids)

    @torch.no_grad()
    def targets(self,q,v,t,rotation=None,cvel=None):
        if q.shape!=(self.n,self.model.nq) or v.shape!=(self.n,self.model.nv) or t.shape!=(self.n,):
            raise ValueError('Shared dual SONIC tensor shapes disagree with plant')
        self.output.zero_()
        for part in self.parts:
            localq,localv=part['q'],part['v']
            prq,prv,srq,srv=part['prq'],part['prv'],part['srq'],part['srv']
            localq[:,prq:prq+7]=q[:,srq:srq+7]
            localv[:,prv:prv+6]=v[:,srv:srv+6]
            localq[:,part['pqa']]=q[:,part['sqa']]
            localv[:,part['pva']]=v[:,part['sva']]
            r=None if rotation is None else rotation[:,part['base']].reshape(self.n,3,3)
            omega=None if cvel is None else cvel[:,part['base'],:3]
            batch=part['batch'];bodytargets=batch.targets(localq,localv,t,r,omega)
            i,j,w=batch.ref.indices(t)
            nominal=batch.ref.q[i]+w[:,None]*(batch.ref.q[j]-batch.ref.q[i])
            controls=part['control'];controls.zero_()
            controls[:,part['pa']]=nominal[:,part['pqa']]
            controls[:,batch.aids]=bodytargets
            self.output[:,part['sa']]=controls[:,part['pa']]
        self.output[:,self.inactive]=0.
        return self.output

    @torch.no_grad()
    def record_applied(self,ctrl):
        """Receive complete nominal targets BEFORE hard clipping and delay."""
        if ctrl.shape!=(self.n,self.model.nu):raise ValueError('Need shared pre-limit/pre-delay nominal targets')
        for part in self.parts:
            local=part['control'];local.zero_()
            local[:,part['pa']]=ctrl[:,part['sa']]
            part['batch'].record_applied(local)

    def report(self):
        return dict(controller='two strictly loaded frozen FSQ SONIC actors + declared external task residual',
            actor_count=2,body_outputs_per_actor=28,shared_plant_actuator_channels=int(self.model.nu),
            original_checkpoint_changed=False,action_delay_physics_steps=3,
            internal_actor_delay_physics_steps=0,required_external_body_delay_physics_steps=3,
            action_delay_disclosure='Actor computes current undelayed targets; environment/evaluation applies3 physical steps of body delay, with original immediate first-command cold start.',
            previous_action_observation='Normalized pre-joint-limit/pre-delay nominal target: frozen SONIC raw target plus task residual. Mechanical clipping is excluded. Zero task residual equals original raw clipped-network action.',
            imu_latency_s=.0025,physics_dt_s=.005,policy_dt_s=.02,
            states_written_by_controller=False,root_forces_applied=False,
            proxy_semantics='Only numerical tensors mirrored for inference; no proxy simulation andno proxy-to-plant state copies.',
            parameter_synchronization_before_warp=self.synchronized_fields,
            actors=[dict(prefix=p['prefix'],batch_inference_count=int(p['batch'].inference_count),
                         original_actor=p['batch'].c.report())for p in self.parts])

