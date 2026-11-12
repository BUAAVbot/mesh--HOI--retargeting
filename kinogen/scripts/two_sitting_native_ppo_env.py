"""Native double-precision MuJoCo worlds with batched Torch policy inference.

Each world owns persistent live and sensor-only MjData. A shared immutable
MjModel is stepped by C calls in worker threads. Only episode resets write
live qpos/qvel; all learned actions use the56 physical body-joint motors.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import mujoco as mj
import numpy as np
import torch

from two_sitting_ppo_env import DualReference, ActorCritic, WarpTracking as _WarpTracking

Reference = DualReference

POLICY_SENSOR_CONTRACT = 'native_parallel_fresh_v1: native CPU mj_step double precision; private mj_kinematics/comPos/comVel for fresh GPU policy observations; raw nominal action history before physical bounds; delay3 physics steps with original immediate cold start'


class NativeTracking(_WarpTracking):
    """Drop-in environment for the existing two_sitting PPO trainer.

    The inherited observation and curriculum definitions deliberately match
    WarpTracking. Physics and actual contact force measurement are native.
    ``threads`` controls persistent CPU workers, not the policy batch size.
    """

    dt = .02

    def __init__(self, model_path, reference_path, n=128, seed=1,
                 action_scale=.25, sonic_base=False, proxy_path=None,
                 threads=16, device=None):
        self.mj = mj
        self.n, self.action_scale = int(n), float(action_scale)
        self.device = str(device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.threads = max(1, min(int(threads), self.n))
        if self.n <= 0:
            raise ValueError('n must be positive')
        torch.manual_seed(seed)
        model_path = Path(model_path).resolve()
        self.model = (mj.MjModel.from_binary_path(str(model_path)) if model_path.suffix == '.mjb'
                      else mj.MjModel.from_xml_path(str(model_path)))
        if (self.model.nq, self.model.nv, self.model.nu) != (118, 116, 104):
            raise ValueError('Expected shared dual-Vita118/116/104 plant')
        if abs(float(self.model.opt.timestep)-.005) > 1e-9:
            raise ValueError('Expected5ms physics/20ms policy')
        if self.model.nmocap:
            raise ValueError('Mocap forbidden')
        if int(self.model.opt.cone) != int(mj.mjtCone.mjCONE_ELLIPTIC):
            raise ValueError('Actual normal-force reward expects elliptic cone')
        if any(self.model.eq_type != mj.mjtEq.mjEQ_JOINT):
            raise ValueError('Only joint coupling equalities are allowed')
        self.ref = DualReference(self.model, reference_path, self.device)
        roots = {m['root'] for m in self.ref.maps}
        if any(int(j) in roots for j in self.model.actuator_trnid[:, 0]):
            raise ValueError('Root actuators forbidden')
        self.sonic = None
        if sonic_base:
            if proxy_path is None:
                raise ValueError('proxy_path required for frozen SONIC')
            from two_sitting_batch_sonic import DualBatchSonic
            self.sonic = DualBatchSonic(self.model, self.ref, self.n, self.device, proxy_path)
        self.ref.refresh_actuation(self.model)
        # All model construction/mutation precedes creation of live worlds.
        self.datas = [mj.MjData(self.model) for _ in range(self.n)]
        self.sensor_datas = [mj.MjData(self.model) for _ in range(self.n)]
        self.data, self.d = self.datas, self.datas
        self.cpu_data = self.datas[0]
        self.executor = ThreadPoolExecutor(max_workers=self.threads,
                                           thread_name_prefix='sitting-mujoco')
        self._closed = False
        self.policy_sensor_contract = POLICY_SENSOR_CONTRACT
        self.contact_force_available = True
        self.contact_contract = 'Actual native elliptic contact normal force with world upward reaction on robot hip/base; contact arrays vectorized per world.'
        self.reset_scratch_allocated_bytes = 0
        self.physics_backend = 'native_mujoco_parallel_cpu'
        self.phase = torch.zeros(self.n, device=self.device)
        self.previous = torch.zeros((self.n, 56), device=self.device)
        self.previous_body_target = torch.zeros((self.n, 56), device=self.device)
        self.cold_start = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.from_start = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.start_successes = 0; self.starts = 0
        self.max_from_start_source_s = 0.; self.nan_episodes = 0
        self.native_policy_steps = 0; self.native_physics_wall_s = 0.
        self.state_write_reset_events = 0
        self._generation = 0; self._host_generation = -1; self._published_generation = -1
        self._host_tensors, self._host_arrays = {}, {}
        shapes = {'q': (self.n, 118), 'v': (self.n, 116),
                  'xpos': (self.n, self.model.nbody, 3),
                  'rotation': (self.n, self.model.nbody, 3, 3),
                  'cvel': (self.n, self.model.nbody, 6)}
        pinned = self.device.startswith('cuda')
        for name, shape in shapes.items():
            host = torch.zeros(shape, dtype=torch.float32, pin_memory=pinned)
            self._host_tensors[name] = host
            self._host_arrays[name] = host.numpy()
            setattr(self, name, torch.zeros(shape, device=self.device))
        self.u = torch.zeros((self.n, 104), device=self.device)
        self._seat_cpu = np.zeros((self.n, 2))
        self._cross_cpu = np.zeros(self.n)
        self._penetration_cpu = np.zeros(self.n)
        self._numeric_fault = np.zeros(self.n, dtype=bool)
        self._prepare_native_contact_metadata()
        self.reset(torch.arange(self.n, device=self.device))

    def _prepare_native_contact_metadata(self):
        model = self.model
        gn = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, g) or ''
              for g in range(model.ngeom)]
        bn = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ''
              for g in range(model.ngeom)]
        self._geom_actor = np.asarray([0 if b.startswith('p1_') else
                                       1 if b.startswith('p2_') else -1 for b in bn])
        self._geom_seat = np.asarray(['seat' in g.lower() or 'stool' in g.lower() for g in gn])
        self._geom_pelvis = np.asarray([any(s in b for s in ('base_link', 'hip_', 'pelvis', 'thigh'))
                                       for b in bn])
        self._warning_ids = [int(getattr(mj.mjtWarning, 'mjWARN_'+name))
                             for name in ('BADQPOS', 'BADQVEL', 'BADQACC', 'BADCTRL')]

    def _cache_one(self, index):
        live, scratch = self.datas[index], self.sensor_datas[index]
        scratch.qpos[:] = live.qpos; scratch.qvel[:] = live.qvel
        scratch.act[:] = live.act; scratch.eq_active[:] = live.eq_active
        scratch.time = live.time
        mj.mj_kinematics(self.model, scratch)
        mj.mj_comPos(self.model, scratch)
        mj.mj_comVel(self.model, scratch)
        self._host_arrays['q'][index] = live.qpos
        self._host_arrays['v'][index] = live.qvel
        self._host_arrays['xpos'][index] = scratch.xpos
        self._host_arrays['rotation'][index] = scratch.xmat.reshape(self.model.nbody, 3, 3)
        self._host_arrays['cvel'][index] = scratch.cvel

    def _contacts_one(self, index):
        d = self.datas[index]
        self._seat_cpu[index] = 0.; self._cross_cpu[index] = 0.; self._penetration_cpu[index] = 0.
        if not d.ncon:
            return
        c = d.contact
        geom = np.asarray(c.geom, dtype=np.int64).reshape(-1, 2)[:d.ncon]
        dist = np.asarray(c.dist)[:d.ncon]
        address = np.asarray(c.efc_address, dtype=np.int64)[:d.ncon]
        frame = np.asarray(c.frame)[:d.ncon].reshape(-1, 3, 3)
        valid = (geom[:, 0] >= 0) & (geom[:, 1] >= 0)
        geom, dist, address, frame = geom[valid], dist[valid], address[valid], frame[valid]
        if not len(dist):
            return
        self._penetration_cpu[index] = max(0., float(np.max(-dist)))
        actors = self._geom_actor[geom]
        crossed = (actors[:, 0] >= 0) & (actors[:, 1] >= 0) & (actors[:, 0] != actors[:, 1])
        self._cross_cpu[index] = float(np.sum(crossed*(np.maximum(-dist, 0.)+.005)))
        normal = np.zeros(len(dist))
        force_valid = (address >= 0) & (address < d.nefc)
        normal[force_valid] = np.maximum(np.asarray(d.efc_force)[address[force_valid]], 0.)
        # Four masks per world; no Python loop over individual contacts.
        for side in range(2):
            other = 1-side
            up = normal*frame[:, 0, 2]*(1 if side == 1 else -1)
            for actor in range(2):
                mask = ((actors[:, side] == actor) & self._geom_seat[geom[:, other]] &
                        self._geom_pelvis[geom[:, side]] & (up > .5) & (dist <= .002))
                self._seat_cpu[index, actor] = np.sum(up[mask]) + self._seat_cpu[index, actor]

    def _advance_one(self, index):
        d = self.datas[index]
        if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
            raise RuntimeError('External force in native training world')
        d.ctrl[:] = self._control_old_cpu[index]
        mj.mj_step(self.model, d, nstep=3)
        d.ctrl[:] = self._control_new_cpu[index]
        mj.mj_step(self.model, d)
        self._numeric_fault[index] = any(d.warning[i].number > 0 for i in self._warning_ids)
        self._contacts_one(index)
        self._cache_one(index)

    def refresh_policy_sensors(self, ids=None):
        # _advance_one and explicit resets normally populate all changed rows.
        if self._host_generation != self._generation:
            selected = range(self.n) if ids is None else list(map(int, ids.detach().cpu().numpy()))
            list(self.executor.map(self._cache_one, selected))
            self._host_generation = self._generation
        if self._published_generation != self._generation:
            for name, host in self._host_tensors.items():
                getattr(self, name).copy_(host, non_blocking=False)
            self._published_generation = self._generation

    def _reset_one(self, packet):
        index, qpos, qvel, control = packet
        d = self.datas[index]
        mj.mj_resetData(self.model, d)
        # These are the only live state assignments in the environment.
        d.qpos[:] = qpos; d.qvel[:] = qvel; d.ctrl[:] = control
        mj.mj_forward(self.model, d)
        self._numeric_fault[index] = False
        self._contacts_one(index)
        self._cache_one(index)

    def reset(self, ids, force_start=False):
        if not len(ids):
            return
        if not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        phase = torch.rand(len(ids), device=self.device)*self.ref.end
        start = torch.rand(len(ids), device=self.device) < getattr(self, 'start_probability', .30)
        if force_start:
            start[:] = True
        phase[start] = 0.
        self.phase[ids] = phase; self.from_start[ids] = start
        self.starts += int(start.sum()); self.steps[ids] = 0
        target, velocity, _ = self.ref.sample(phase)
        velocity[start] = 0.
        self.previous[ids] = 0.
        controls = self.ref.controls(phase, self.previous[ids], self.action_scale)
        self.u[ids] = controls
        self.previous_body_target[ids] = controls[:, self.ref.body_aids_t]
        self.cold_start[ids] = True
        if self.sonic is not None:
            self.sonic.reset(ids)
        cpu_ids = ids.detach().cpu().numpy()
        cpu_q, cpu_v, cpu_u = (v.detach().cpu().numpy() for v in (target, velocity, controls))
        list(self.executor.map(self._reset_one,
             [(int(index), cpu_q[j], cpu_v[j], cpu_u[j]) for j, index in enumerate(cpu_ids)]))
        self.state_write_reset_events += len(cpu_ids)
        self._generation += 1; self._host_generation = self._generation
        self.refresh_policy_sensors()

    def contact_summary(self):
        return tuple(torch.as_tensor(value, device=self.device, dtype=torch.float32)
                     for value in (self._seat_cpu, self._cross_cpu, self._penetration_cpu))

    def step(self, actions):
        packet = self.base_controls()
        nominal = packet.clone()
        nominal[:, self.ref.body_aids_t] += self.action_scale*torch.tanh(actions)
        body = nominal[:, self.ref.body_aids_t].clamp(self.ref.lower, self.ref.upper)
        packet[:, self.ref.body_aids_t] = body; packet[:, self.ref.passive_t] = 0.
        delayed = packet.clone()
        delayed[:, self.ref.body_aids_t] = torch.where(self.cold_start[:, None], body,
                                                     self.previous_body_target)
        self._control_old_cpu = delayed.detach().cpu().numpy()
        self._control_new_cpu = packet.detach().cpu().numpy()
        self.u.copy_(packet)
        began = time.monotonic()
        list(self.executor.map(self._advance_one, range(self.n)))
        self.native_physics_wall_s += time.monotonic()-began
        self.native_policy_steps += 1
        self.previous_body_target.copy_(body); self.cold_start[:] = False
        if self.sonic is not None:
            self.sonic.record_applied(nominal)
        self._generation += 1; self._host_generation = self._generation
        self.refresh_policy_sensors()
        self.phase += self.dt; self.steps += 1
        target, velocity, feet = self.ref.sample(self.phase)
        xy, z, angles, vel = [], [], [], []
        for rq, rv in zip(self.ref.root_qadr, self.ref.root_vadr):
            error = self.q[:, rq:rq+3]-target[:, rq:rq+3]
            xy.append(error[:, :2].norm(dim=-1)); z.append(error[:, 2].abs())
            angles.append(2*torch.acos((self.q[:, rq+3:rq+7]*target[:, rq+3:rq+7]).sum(-1).abs().clamp(0, 1)))
            vel.append((self.v[:, rv:rv+3]-velocity[:, rv:rv+3]).square().mean(-1))
        xy, z, ori, vel = (torch.stack(a, 1) for a in (xy, z, angles, vel))
        pose = (self.q[:, self.ref.qa_t]-target[:, self.ref.qa_t]).square().reshape(self.n, 2, 28).mean(-1)
        footerr = (self.xpos[:, self.ref.feet_t]-feet).square().mean((1, 2))
        change = (torch.tanh(actions)-torch.tanh(self.previous)).square().mean(-1)
        seat_force, cross, pen = self.contact_summary()
        _, expected_seat = self.ref.support_at(self.phase)
        seat_match = ((seat_force > 1.).float()*expected_seat).mean(-1)
        reward = (3.*torch.exp(-xy.square()/.15**2).mean(-1)+torch.exp(-40*z.square()).mean(-1)+
                  torch.exp(-5*ori.square()).mean(-1)+torch.exp(-5*pose).mean(-1)+
                  torch.exp(-25*footerr)+.5*torch.exp(-2*vel).mean(-1)+seat_match-
                  .03*change-.003*actions.square().mean(-1)-5.*cross-
                  2.*(pen-.025).clamp_min(0)-2.*xy.square().mean(-1))
        finite = torch.isfinite(self.q).all(-1) & torch.isfinite(self.v).all(-1)
        xy_limit, z_limit, ori_limit = (.35, .20, .6) if getattr(self, 'strict_training', False) else (.65, .30, 1.2)
        numeric = torch.as_tensor(self._numeric_fault, device=self.device)
        failed = ((xy.max(-1).values > xy_limit) | (z.max(-1).values > z_limit) |
                  (ori.max(-1).values > ori_limit) | ~finite | numeric)
        timeout = self.phase >= self.ref.horizon; done = failed | timeout
        reward = torch.nan_to_num(reward, nan=-2., posinf=-2., neginf=-2.)-2.*failed.float()
        self.previous.copy_(actions)
        if self.from_start.any():
            self.max_from_start_source_s = max(self.max_from_start_source_s,
                float(self.ref.source_clock(self.phase[self.from_start]).max()))
        end_seated = (seat_force > 1.).all(-1)
        self.start_successes += int((timeout & ~failed & self.from_start & end_seated).sum())
        self.nan_episodes += int((~finite | numeric).sum())
        info = {'failed': failed, 'timeout': timeout,
                'root_xy': torch.nan_to_num(xy.max(-1).values, nan=100.),
                'root_angle': torch.nan_to_num(ori.max(-1).values, nan=100.),
                'root_xy_per_actor': xy, 'root_angle_per_actor': ori,
                'pose_mse': pose.mean(-1), 'feet_mse': footerr,
                'seat_support_force_N': seat_force, 'cross_actor_contact_penalty': cross,
                'max_penetration_m': pen, 'training_endpoint_seated': end_seated}
        return self.observations(), reward*.1, done, info

    def backend_report(self):
        return {'backend': self.physics_backend, 'mujoco_version': mj.__version__,
                'worlds': self.n, 'threads': self.threads, 'physics_dt_s': .005,
                'policy_dt_s': self.dt, 'policy_sensor_contract': self.policy_sensor_contract,
                'native_policy_steps': self.native_policy_steps,
                'native_physics_wall_s': self.native_physics_wall_s,
                'state_write_episode_reset_events': self.state_write_reset_events,
                'live_state_writes_outside_resets': 0, 'applied_external_force_writes': 0,
                'contact_contract': self.contact_contract,
                'training_completion_requires_independent_native_acceptance': True}

    def close(self):
        if not self._closed:
            self.executor.shutdown(wait=True)
            self._closed = True

    def __del__(self):
        executor = getattr(self, 'executor', None)
        if executor is not None and not getattr(self, '_closed', True):
            executor.shutdown(wait=False)


# Allows a trainer to change just its import module for the native backend.
WarpTracking = NativeTracking
