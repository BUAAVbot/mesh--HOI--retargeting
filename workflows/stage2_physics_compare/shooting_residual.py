"""Native shooting residual, evaluated in isolated CPU workers for finite differences."""
import mujoco as mj
import numpy as np
from common import classify_robot_static_bodies,configure_position_pd,load_canonical,physics_substeps


def orientation_error(actual,target):
    actual=actual*np.where(np.sum(actual*target,axis=-1,keepdims=True)>=0,1.,-1.)
    xyz=-actual[...,:1]*target[...,1:]+target[...,:1]*actual[...,1:]-np.cross(actual[...,1:],target[...,1:])
    real=np.sum(actual*target,axis=-1);length=np.linalg.norm(xyz,axis=-1)
    return xyz*(2*np.arctan2(length,real)/np.maximum(length,1e-12))[...,None]


class ShootingResidual:
    def __init__(self,model_path,reference_path,initial,knots):
        self.model=mj.MjModel.from_binary_path(str(model_path));configure_position_pd(self.model,mj)
        self.ref=load_canonical(reference_path);self.initial=initial;self.knots=knots
        self.substeps=physics_substeps(float(self.ref['dt'].ravel()[0]),self.model.opt.timestep)
        self.steps=(len(self.ref['qpos_ref'])-1)*self.substeps
        self.fk=mj.MjData(self.model);self.robot,_,_=classify_robot_static_bodies(self.model,mj)
        self.feet=[mj.mj_name2id(self.model,mj.mjtObj.mjOBJ_BODY,side+'_ankle_roll_link') for side in ('left','right')]
        self.jp=np.zeros((3,self.model.nv));self.jr=np.zeros_like(self.jp)
        self.stationary=np.repeat(self.ref['stationary_mask'][:-1],self.substeps,axis=0)

    def __call__(self,item):
        x,trajectory=item;m=self.model;ref=self.ref;fk=self.fk
        states=np.vstack([self.initial[1:],trajectory]);q=states[:,:m.nq];v=states[:,m.nq:]
        sample=q[::self.substeps];velocity=v[::self.substeps]
        pos=(sample[:,:3]-ref['qpos_ref'][:,:3])*np.sqrt([1600.,1600.,3000.])
        ori=orientation_error(sample[:,3:7],ref['qpos_ref'][:,3:7])*np.sqrt(800.)
        joint=(sample[:,7:]-ref['qpos_ref'][:,7:])*np.sqrt(50.)
        speed=(velocity[:,:6]-ref['qvel_ref'][:,:6])*np.array([8.,8.,8.,4.,4.,4.])
        ranges=m.jnt_range[1:]
        limit=np.maximum(np.maximum(ranges[:,0]-q[:,7:],q[:,7:]-ranges[:,1])-.005,0.)*500.
        collision=np.zeros((self.steps,2));slip=np.zeros_like(collision)
        for t in range(self.steps):
            fk.qpos[:]=q[t];mj.mj_kinematics(m,fk);mj.mj_comPos(m,fk);mj.mj_collision(m,fk)
            for c in fk.contact:
                a,b=map(int,m.geom_bodyid[[c.geom1,c.geom2]])
                if a not in self.robot and b not in self.robot:continue
                category=1 if a in self.robot and b in self.robot else 0
                collision[t,category]=max(collision[t,category],-float(c.dist))
            for side,body in enumerate(self.feet):
                if not self.stationary[t,side]:continue
                point=fk.xpos[body]+fk.xmat[body].reshape(3,3)@np.array([0.,0.,-.072])
                mj.mj_jac(m,fk,self.jp,self.jr,point,body)
                slip[t,side]=max(0.,np.linalg.norm((self.jp@v[t])[:2])-.04)*200.
        collision=np.maximum(collision-.0035,0.)*10000.
        delta=x.reshape(self.knots,-1)
        return np.r_[pos.ravel(),ori.ravel(),joint.ravel(),speed.ravel(),
            pos[-1]*10.,ori[-1]*5.,speed[-1]*3.,limit.ravel(),collision.ravel(),slip.ravel(),
            delta.ravel()*.5,np.diff(delta,axis=0).ravel()*2.]


def initialize_worker(model_path,reference_path,initial,knots):
    global _worker
    _worker=ShootingResidual(model_path,reference_path,initial,knots)


def evaluate_worker(item):
    return _worker(item)
