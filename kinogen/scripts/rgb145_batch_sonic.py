"""Batch the frozen original SONIC actor; all 28 body outputs remain active."""
import sys
from pathlib import Path
import numpy as np
import torch
from rgb145_ppo_env import rotate_inverse,quat_mul


def dependency_paths():
    for path in ('/data/grail-envs/retarget/lib/python3.10/site-packages','/data/grail-envs/grail/lib/python3.10/site-packages','/data/grail-envs/tracking45/lib/python3.10/site-packages'):
        if path not in sys.path:sys.path.append(path)


class BatchSonic:
    def __init__(self,model,reference,n,device,heading_mode='training_z_twist',reference_velocity_mode='training_forward_50hz'):
        dependency_paths()
        from rgb145_sonic_controller import Controller
        self.c=Controller(model,dict(reference.z),{'mode':'newft15225','device':device,'cpu_threads':2,'delay_steps':0,'imu_latency_s':.0025,'hand_object_feedback':False,'heading_mode':heading_mode,'reference_velocity_mode':reference_velocity_mode})
        self.heading_mode=self.c.heading_mode
        self.reference_velocity_mode=self.c.reference_velocity_mode
        self.ref=reference;self.n=n;self.device=device;self.policy=self.c.policy.eval()
        for param in self.policy.parameters():param.requires_grad_(False)
        def tensor(v,dtype=torch.float32):return torch.as_tensor(v,device=device,dtype=dtype)
        self.qa=tensor(self.c.qa,torch.long);self.va=tensor(self.c.va,torch.long);self.aids=tensor(self.c.aids,torch.long)
        self.default=tensor(self.c.default);self.scale=tensor(self.c.scale);self.qf=tensor(self.c.body_q);self.dqf=tensor(self.c.body_dq)
        self.history={k:torch.zeros((n,10,width),device=device)for k,width in [('base_ang_vel',3),('joint_pos',28),('joint_vel',28),('actions',28),('gravity_dir',3)]}
        self.valid=torch.zeros(n,dtype=torch.bool,device=device);self.phase=torch.full((n,),-1.,device=device)
        self.last=torch.zeros((n,28),device=device);self.previous_imu=torch.zeros((n,3),device=device);self.previous_gravity=torch.zeros((n,3),device=device)
        self.cached=torch.zeros((n,28),device=device);self.tokens={k:v.expand(n,*v.shape[1:]).clone()for k,v in self.c.tokenizer.items()}
        self.inference_count=0

    def reset(self,ids):
        self.valid[ids]=False;self.phase[ids]=-1.;self.last[ids]=0.

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
        if self.reference_velocity_mode=='legacy_centered':
            dq=(self.dqf[i]+w[:,None]*(self.dqf[j]-self.dqf[i])).reshape(len(ids),10,28)
        else:
            i2,j2,w2=self.ref.indices((future_t+.02).clamp_max(self.ref.end).flatten())
            qp2=(self.qf[i2]+w2[:,None]*(self.qf[j2]-self.qf[i2])).reshape_as(qp)
            dq=(qp2-qp)/.02
            ip,jp,wpast=self.ref.indices((future_t-.02).clamp_min(float(self.ref.times[0])).flatten())
            past=(self.qf[ip]+wpast[:,None]*(self.qf[jp]-self.qf[ip])).reshape_as(qp)
            backward=(qp-past)/.02
            dq=torch.where((future_t>=self.ref.end)[:,:,None],backward,dq)
        future=torch.cat([qp.flatten(1),dq.flatten(1)],dim=-1).reshape(len(ids),1,10,56)
        fq=(self.ref.q[i,3:7]+w[:,None]*(self.ref.q[j,3:7]-self.ref.q[i,3:7])).reshape(len(ids),10,4);fq/=fq.norm(dim=-1,keepdim=True).clamp_min(1e-9)
        qw,qx,qy,qz=actual.unbind(-1)
        if self.heading_mode=='training_z_twist':
            # R10-R01=4*w*z, R00+R11=2*(w*w-z*z): atan2 gives
            # the full Z-twist angle, not its half-angle. This uses the same
            # preintegration rotation sample as native mju_mat2Quat.
            if rotation is not None:
                r=rotation[ids];yaw=torch.atan2(r[:,1,0]-r[:,0,1],r[:,0,0]+r[:,1,1])
            else:yaw=2*torch.atan2(qz,qw)
        elif rotation is not None:
            r=rotation[ids];yaw=torch.atan2(r[:,1,0],r[:,0,0])
        else:yaw=torch.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        heading=torch.zeros_like(actual);heading[:,0]=torch.cos(yaw/2);heading[:,3]=-torch.sin(yaw/2)
        rel=quat_mul(heading[:,None].expand(-1,10,-1),fq);rw,rx,ry,rz=rel.unbind(-1)
        orientation=torch.stack([1-2*(ry*ry+rz*rz),2*(rx*ry-rw*rz),2*(rx*ry+rw*rz),1-2*(rx*rx+rz*rz),2*(rx*rz-rw*ry),2*(ry*rz+rw*rx)],dim=-1)[:,None]
        self.tokens['command_multi_future_nonflat'][ids]=future;self.tokens['motion_anchor_ori_heading_mf_nonflat'][ids]=orientation
        tokenizer=torch.cat([self.tokens[name][ids].reshape(len(ids),1,-1)for name in self.c.cfg.env_config.obs.group_obs_names.tokenizer],dim=-1)
        inputs=self.c.TensorDict({'actor_obs':proprio[:,None],'tokenizer':tokenizer},batch_size=[len(ids),1])
        action=self.policy(inputs).reshape(len(ids),28).clamp(-float(self.c.cfg.env_config.action_clip_value),float(self.c.cfg.env_config.action_clip_value))
        self.cached[ids]=self.default+self.scale*action;self.valid[ids]=True;self.phase[ids]=t[ids];self.inference_count+=len(ids)
        return self.cached

    def record_applied(self,ctrl):self.last.copy_((ctrl[:,self.aids]-self.default)/self.scale)


def main():
    import argparse,json,mujoco as mj
    from rgb145_ppo_env import Reference
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--heading-mode',choices=['training_z_twist','euler_yaw'],default='training_z_twist');p.add_argument('--reference-velocity-mode',choices=['training_forward_50hz','legacy_centered'],default='training_forward_50hz');p.add_argument('--tilted-test',action='store_true');a=p.parse_args()
    dependency_paths();from rgb145_sonic_controller import Controller
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));ref=Reference(m,a.reference,'cuda:0');batch=BatchSonic(m,ref,1,'cuda:0',a.heading_mode,a.reference_velocity_mode)
    native=Controller(m,dict(ref.z),{'mode':'newft15225','device':'cuda:0','cpu_threads':2,'delay_steps':0,'imu_latency_s':.0025,'hand_object_feedback':False,'heading_mode':a.heading_mode,'reference_velocity_mode':a.reference_velocity_mode})
    d=mj.MjData(m);d.qpos[:]=ref.q[0].cpu().numpy();d.qvel[:]=ref.dq[0].cpu().numpy();mj.mj_forward(m,d);native.reset(d);errors=[]
    cases=[]
    if a.tilted_test:
        from scipy.spatial.transform import Rotation
        for yaw,pitch,roll in [(0,0,0),(35,25,20),(-80,-35,30),(160,40,-25),(-175,-45,-35)]:
            for repeat in range(3):cases.append((yaw,pitch,roll,repeat))
    for k in range(len(cases)if a.tilted_test else 20):
        if a.tilted_test:
            yaw,pitch,roll,repeat=cases[k]
            if repeat==0:native.reset(d);batch.reset(torch.tensor([0],device='cuda:0'))
            d.qpos[:]=ref.q[0].cpu().numpy();d.qpos[3:7]=np.roll(Rotation.from_euler('ZYX',[yaw,pitch+repeat*.3,roll-repeat*.2],degrees=True).as_quat(),1)
            d.qvel[:]=0.;mj.mj_forward(m,d)
            t=repeat*.02
        else:t=k*.02
        u=native.action(d,t)
        q=torch.as_tensor(d.qpos.copy(),device='cuda:0',dtype=torch.float32)[None];v=torch.as_tensor(d.qvel.copy(),device='cuda:0',dtype=torch.float32)[None]
        r=torch.as_tensor(d.xmat[native.base].reshape(3,3).copy(),device='cuda:0',dtype=torch.float32)[None];omega=torch.as_tensor(d.cvel[native.base,:3].copy(),device='cuda:0',dtype=torch.float32)[None]
        base=batch.targets(q,v,torch.tensor([t],device='cuda:0'),r,omega);errors.append(float(np.max(abs(base[0].cpu().numpy()-u[native.aids]))))
        packet=torch.zeros((1,m.nu),device='cuda:0');packet[:,batch.aids]=base;batch.record_applied(packet)
        if not a.tilted_test:
            for _ in range(4):d.ctrl[:]=u;mj.mj_step(m,d)
    result={'status':'passed'if max(errors)<2e-4 else'failed','max_body_target_difference_rad':max(errors),'per_update_errors':errors,'updates':len(errors),'all28_body_channels':True,'heading_mode':a.heading_mode,'reference_velocity_mode':a.reference_velocity_mode,'tilted_input_abi_test_only':a.tilted_test,'tilted_cases_yaw_pitch_roll_repeat':cases,'original_policy_sha256':native.report()['policy_sha256']}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if result['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
