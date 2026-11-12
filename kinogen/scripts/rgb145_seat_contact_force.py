"""Read real MuJoCo-Warp solver forces for the existing hip/base seat-load gate.

This independent helper never advances physics or edits a model/controller.
It reuses the installed public contact_force API, including its actual cone
decoder, and returns preallocated Warp arrays suitable for a planning graph.
"""
import numpy as np
import mujoco as mj
import mujoco_warp as mw
import warp as wp


@wp.kernel
def _eligible_contact_ids(geom:wp.array(dtype=wp.vec2i),worldid:wp.array(dtype=wp.int32),nacon:wp.array(dtype=wp.int32),robot_mask:wp.array(dtype=wp.int32),seat_mask:wp.array(dtype=wp.int32),ids:wp.array(dtype=wp.int32),invalid:wp.array(dtype=wp.int32),nworld:int,ngeom:int):
    i=wp.tid();ids[i]=nacon[0]
    if i==0 and nacon[0]>ids.shape[0]:wp.atomic_max(invalid,0,1)
    if i<nacon[0]:
        world=worldid[i]
        if world<0 or world>=nworld:wp.atomic_max(invalid,0,1)
        else:
            g0=geom[i][0];g1=geom[i][1]
            if g0>=0 and g0<ngeom and g1>=0 and g1<ngeom:
                if (robot_mask[g0]>0 and seat_mask[g1]>0)or(robot_mask[g1]>0 and seat_mask[g0]>0):ids[i]=i


@wp.kernel
def _sum_seat_load(geom:wp.array(dtype=wp.vec2i),worldid:wp.array(dtype=wp.int32),nacon:wp.array(dtype=wp.int32),robot_mask:wp.array(dtype=wp.int32),seat_mask:wp.array(dtype=wp.int32),forces:wp.array(dtype=wp.spatial_vector),net_world:wp.array(dtype=wp.vec3),upward:wp.array(dtype=wp.float32),counts:wp.array(dtype=wp.int32),invalid:wp.array(dtype=wp.int32),nworld:int,ngeom:int):
    i=wp.tid()
    if i<nacon[0]:
        world=worldid[i]
        if world>=0 and world<nworld:
            g0=geom[i][0];g1=geom[i][1];sign=0.0
            if g0>=0 and g0<ngeom and g1>=0 and g1<ngeom:
                if robot_mask[g0]>0 and seat_mask[g1]>0:sign=-1.0
                elif robot_mask[g1]>0 and seat_mask[g0]>0:sign=1.0
            if sign!=0.0:
                # The installed API returns FORCE:TORQUE on geom2. This is
                # deliberately not the usual torque:force spatial convention.
                f=wp.vec3(forces[i][0],forces[i][1],forces[i][2])*sign
                if wp.isfinite(f[0])and wp.isfinite(f[1])and wp.isfinite(f[2]):
                    wp.atomic_add(net_world,world,f);wp.atomic_add(upward,world,wp.max(f[2],0.0));wp.atomic_add(counts,world,1)
                else:wp.atomic_max(invalid,0,1)


class SeatContactForce:
    """Construct once, then call sample() after each real solver step.

net_world_N is the signed total force on eligible robot bodies. upward_load_N
matches rgb145_physics.py exactly: sum(max(robot_contact_Fz,0)), not max of
the summed force. invalid[0] invalidates the extraction on overflow or bad
world ids/nonfinite forces. It must not be interpreted as a zero seat load.
    """
    def __init__(self,native_model,warp_model,warp_data):
        if not callable(getattr(mw,'contact_force',None)):raise RuntimeError('Installed MuJoCo-Warp lacks public contact_force API')
        self.m=warp_model;self.d=warp_data;self.nworld=int(warp_data.nworld);self.ngeom=int(native_model.ngeom);self.device=warp_data.nacon.device
        if len(warp_data.contact.dist.shape)!=1 or warp_data.contact.worldid.shape!=warp_data.contact.dist.shape or tuple(warp_data.nacon.shape)!=(1,):raise ValueError('Unsupported actual contact/world-count ABI')
        if int(native_model.opt.cone)!=int(warp_model.opt.cone):raise ValueError('Native/Warp cone mismatch')
        self.capacity=int(warp_data.contact.dist.shape[0]);robot=np.zeros(self.ngeom,np.int32);seat=np.zeros(self.ngeom,np.int32);self.robot_names=[];self.seat_names=[]
        for g in range(self.ngeom):
            name=mj.mj_id2name(native_model,mj.mjtObj.mjOBJ_GEOM,g)or str(g);body=mj.mj_id2name(native_model,mj.mjtObj.mjOBJ_BODY,int(native_model.geom_bodyid[g]))or'world'
            active=bool(native_model.geom_contype[g]or native_model.geom_conaffinity[g])
            if active and('hip_'in body or body=='base_link'):robot[g]=1;self.robot_names.append(name)
            if active and name.startswith('seat_support_'):seat[g]=1;self.seat_names.append(name)
        self.robot_mask=wp.array(robot,dtype=wp.int32,device=self.device);self.seat_mask=wp.array(seat,dtype=wp.int32,device=self.device)
        self.contact_ids=wp.zeros(self.capacity,dtype=wp.int32,device=self.device);self.force_world_on_geom2=wp.zeros(self.capacity,dtype=wp.spatial_vector,device=self.device)
        self.net_world_N=wp.zeros(self.nworld,dtype=wp.vec3,device=self.device);self.upward_load_N=wp.zeros(self.nworld,dtype=wp.float32,device=self.device);self.eligible_contact_count=wp.zeros(self.nworld,dtype=wp.int32,device=self.device);self.invalid=wp.zeros(1,dtype=wp.int32,device=self.device)
        self.contract={'source':'Installed mujoco_warp.contact_force with to_world_frame=True; actual opt.cone/Contact.dim/efc_address and Data.efc.force','cone':int(native_model.opt.cone),'robot_collision_geoms':self.robot_names,'seat_collision_geoms':self.seat_names,'active_slots':'0 <= slot < Data.nacon[0], capacity checked; inactive output cleared every sample','world_index':'Actual Contact.worldid; invalid ids or overflow invalidate extraction','force_sign':'API force on geom2; negate if eligible robot is geom1','upward_gate':'sum over eligible contacts of max(force_on_robot_world_z,0), matching original physics seat gate','net_world':'Signed resultant world force on eligible robot bodies, without positive clipping','time':'Current solver contact/constraint state; after mj_step these forces correspond to the substep start, not the integrated endpoint','mutations':'Only private helper output buffers; no model/physics/controller/constraint changes'}

    def sample(self):
        self.force_world_on_geom2.zero_();self.net_world_N.zero_();self.upward_load_N.zero_();self.eligible_contact_count.zero_();self.invalid.zero_()
        wp.launch(_eligible_contact_ids,dim=self.capacity,inputs=[self.d.contact.geom,self.d.contact.worldid,self.d.nacon,self.robot_mask,self.seat_mask,self.contact_ids,self.invalid,self.nworld,self.ngeom],device=self.device)
        # Public decoder handles both pyramidal and elliptic cones. An id equal
        # to nacon is intentionally skipped by its outer kernel (< nacon).
        mw.contact_force(self.m,self.d,self.contact_ids,True,self.force_world_on_geom2)
        wp.launch(_sum_seat_load,dim=self.capacity,inputs=[self.d.contact.geom,self.d.contact.worldid,self.d.nacon,self.robot_mask,self.seat_mask,self.force_world_on_geom2,self.net_world_N,self.upward_load_N,self.eligible_contact_count,self.invalid,self.nworld,self.ngeom],device=self.device)
        return {'upward_load_N':self.upward_load_N,'net_world_N':self.net_world_N,'eligible_contact_count':self.eligible_contact_count,'invalid':self.invalid}
