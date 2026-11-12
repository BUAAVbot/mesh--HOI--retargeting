#!/usr/bin/env python3
"""CPU audit of actual G1 box-policy evaluation, never a simulator or trainer.

Trace arrays are flattened samples: timestamps[N], episode_id[N], done[N],
reset_events[N], root_pos[N,3], root_quats[N,4], box_pos[N,3], box_quat[N,4],
box_vel[N,3 or 6], hand_contact_force_left_N[N], hand_contact_force_right_N[N],
table_contact_force_N[N], feet_force[N,2 or 2,3], joint_pos/vel/targets[N,J].
seed_id[N] is preferred; otherwise evaluation.episode_seed_by_id is required.
The required live_physics.json schema is checked explicitly below. No absent
guard counter or physical property is interpreted as a safe zero/default.

Video provenance: trace_sha256, checkpoint_sha256, video_sha256,
render_source='actual_physics_trace', source_sample_indices[decoded_frames].
The video can subsample recorded states; it cannot substitute reference poses.
--trace-only reports training feedback, never final delivery verification.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def read_json(path):
    def reject(value):raise ValueError('Non-standard JSON constant: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject)


def record(path):
    p=Path(path).resolve()
    return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}


class Checks:
    def __init__(self):self.rows=[]
    def add(self,name,passed,actual=None,expected=None):
        self.rows.append({'check':name,'passed':bool(passed),'actual':actual,'expected':expected})
        return bool(passed)
    @property
    def passed(self):return all(x['passed']for x in self.rows)


def value(data,path,default=None):
    for key in path.split('.'):
        if not isinstance(data,dict)or key not in data:return default
        data=data[key]
    return data


def physics_checks(data,trace_hash,checkpoint_hash,checks):
    import numpy as np
    expected={'sources.trace_sha256':trace_hash,'sources.checkpoint_sha256':checkpoint_hash,
              'stage.box.dynamic':True,'stage.box.kinematic':False,'stage.box.gravity_enabled':True,
              'stage.robot.free_base':True,'stage.robot.root_kinematic':False,
              'stage.object_attachment_joint_count':0,
              'contacts.hand_box_filtered':True,'contacts.table_box_filtered':True,
              'guards.active_state_write_calls':0,'guards.active_external_wrench_calls':0,
              'guards.started_before_active':True,
              'evaluation.sample_phase':'post_physics_pre_reset'}
    for key,wanted in expected.items():
        actual=value(data,key);checks.add(key,actual is not None and actual==wanted,actual,wanted)
    gravity=value(data,'stage.gravity_m_s2')
    checks.add('stage.gravity_m_s2',isinstance(gravity,list)and len(gravity)==3 and np.allclose(gravity,[0,0,-9.81],atol=.05),gravity,[0,0,-9.81])
    mass=value(data,'stage.box.mass_kg')
    checks.add('stage.box.mass_kg',isinstance(mass,(int,float))and abs(mass-1.)<=1e-6,mass,1.)
    apis=value(data,'guards.instrumented_apis')
    checks.add('guard_api_coverage_declared',isinstance(apis,list)and len(apis)>0,apis,'nonempty actual wrapped API names')
    raw=value(data,'guards.per_api')
    valid_raw=isinstance(raw,dict)and bool(raw)and all(isinstance(x,dict)and all(isinstance(x.get(k),int)and not isinstance(x.get(k),bool)and x[k]>=0 for k in ('active_calls','reset_calls'))for x in raw.values())
    checks.add('raw_instrumented_guard_counters',valid_raw,raw,'actual per-API active/reset counters')
    if valid_raw:
        state=sum(x['active_calls']for k,x in raw.items()if not any(s in k.lower()for s in ('external','wrench')))
        wrench=sum(x['active_calls']for k,x in raw.items()if any(s in k.lower()for s in ('external','wrench')))
        checks.add('independent_active_state_counter_sum',state==0 and state==value(data,'guards.active_state_write_calls'),state,0)
        checks.add('independent_active_wrench_counter_sum',wrench==0 and wrench==value(data,'guards.active_external_wrench_calls'),wrench,0)
        checks.add('guards_include_robot_box_and_joint_writes',any(k.startswith('robot.write_root')for k in raw)and any(k.startswith('box.write_root')for k in raw)and 'robot.write_joint_state_to_sim'in raw,list(raw),'robot/box root writes and robot joint-state writes')
        checks.add('guard_reset_path_actually_exercised',sum(x['reset_calls']for x in raw.values())>0,sum(x['reset_calls']for x in raw.values()),'>0 actual initialization calls')
    for key in ('active_box_pose_write_calls','active_root_pose_write_calls','active_joint_state_write_calls'):
        actual=value(data,'guards.'+key)
        if actual is not None:checks.add('guards.'+key,actual==0,actual,0)
    assisted=value(data,'stage.artificial_grasp_assist')
    checks.add('no_artificial_grasp_assist',assisted is False,assisted,False)
    for key in ('training.optimizer_steps','training.environment_steps','evaluation.policy_inference_calls'):
        actual=value(data,key);checks.add(key,isinstance(actual,int)and not isinstance(actual,bool)and actual>0,actual,'positive actual count')
    initial=value(data,'training.initial_actor_sha256');final=value(data,'training.final_actor_sha256');evaluated=value(data,'evaluation.actor_sha256')
    hashes=all(isinstance(x,str)and len(x)==64 and all(c in '0123456789abcdef'for c in x)for x in (initial,final,evaluated))
    checks.add('actor_parameters_updated_and_evaluated',hashes and initial!=final and evaluated==final,{'initial':initial,'trained':final,'evaluated':evaluated},'different trained parameters, exact evaluated actor hash')
    return checks.passed


def quaternion_geometry(quats,order):
    import numpy as np
    q=np.asarray(quats,dtype=np.float64)
    if order=='wxyz':q=q[:,[1,2,3,0]]
    if order not in ('xyzw','wxyz'):raise ValueError('Explicit quaternion order must be xyzw or wxyz')
    norm=np.linalg.norm(q,axis=1)
    if np.any(norm<=0)or not np.allclose(norm,1,atol=1e-3):raise ValueError('Non-unit/invalid quaternion')
    q=q/norm[:,None];x,y,z,w=q.T
    # World-Z row of each rotation; its last entry is local up dot world up.
    zrow=np.column_stack((2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))
    return zrow,np.degrees(np.arccos(np.clip(zrow[:,2],-1,1)))


def trace_arrays(path):
    import numpy as np
    with np.load(path,allow_pickle=False)as z:d={k:z[k]for k in z.files}
    aliases={'root_pos':('root_pos','root_position'),
             'root_quats':('root_quats','root_quat','root_quaternion'),
             'box_quat':('box_quat','box_quats','box_quaternion'),
             'joint_targets':('joint_targets','joint_target'),
             'source_checkpoint_sha256':('source_checkpoint_sha256','checkpoint_sha256')}
    for target,names in aliases.items():
        if target not in d:
            for name in names:
                if name in d:d[target]=d[name];break
    return d


def audit_episodes(d,physics,args,checks):
    import numpy as np
    names=('timestamps','episode_id','done','reset_events','root_pos','root_quats','box_pos','box_quat','box_vel',
           'hand_contact_force_left_N','hand_contact_force_right_N','table_contact_force_N','feet_force','joint_pos','joint_vel','joint_targets')
    for name in names:
        if name not in d:raise ValueError('Required trace array missing: '+name)
    n=len(d['timestamps'])
    if n<2:raise ValueError('Trace has fewer than two samples')
    for name in names:
        if d[name].shape[0]!=n:raise ValueError('Inconsistent first dimension: '+name)
        if not np.isfinite(d[name]).all():raise ValueError('Nonfinite actual state/force: '+name)
    for name in ('timestamps','episode_id','done','reset_events','hand_contact_force_left_N','hand_contact_force_right_N','table_contact_force_N'):
        if d[name].shape!=(n,):raise ValueError('Expected scalar per sample: '+name)
    for name,dim in (('root_pos',3),('root_quats',4),('box_pos',3),('box_quat',4)):
        if d[name].shape!=(n,dim):raise ValueError('Wrong actual state shape: '+name)
    if d['box_vel'].ndim!=2 or d['box_vel'].shape[1]not in (3,6):raise ValueError('box_vel must contain linear3 or spatial6')
    if d['joint_pos'].ndim!=2 or d['joint_pos'].shape!=d['joint_vel'].shape or d['joint_pos'].shape!=d['joint_targets'].shape:raise ValueError('Joint actual/velocity/target dimensions differ')
    for name in ('done','reset_events'):
        if not np.isin(d[name],[0,1]).all():raise ValueError('Nonboolean trace array: '+name)
    if not np.array_equal(d['episode_id'],d['episode_id'].astype(int)):raise ValueError('Noninteger episode IDs')
    for name in ('hand_contact_force_left_N','hand_contact_force_right_N','table_contact_force_N'):
        if np.any(d[name]<0):raise ValueError('Force magnitudes must be nonnegative: '+name)
    size=value(physics,'stage.box.size_xyz_m');table_z=value(physics,'stage.table.top_z_m')
    if not isinstance(size,list)or len(size)!=3 or not np.isfinite(size).all()or min(size)<=0:raise ValueError('Missing/invalid actual box dimensions')
    if not isinstance(table_z,(int,float))or not np.isfinite(table_z):raise ValueError('Missing actual table top Z')
    order=value(physics,'trace_layout.quaternion_order')
    box_zrow,_=quaternion_geometry(d['box_quat'],order);_,tilt=quaternion_geometry(d['root_quats'],order)
    bottom=d['box_pos'][:,2]-np.abs(box_zrow)@(np.asarray(size)/2)
    feet=np.asarray(d['feet_force'])
    if feet.ndim==3 and feet.shape[1:]==(2,3):feet_total=np.linalg.norm(feet,axis=2).sum(axis=1)
    elif feet.ndim==2 and feet.shape[1]==2:
        if np.any(feet<0):raise ValueError('feet_force scalar magnitudes must be nonnegative')
        feet_total=feet.sum(axis=1)
    else:raise ValueError('feet_force must be [N,2] magnitudes or [N,2,3] actual vectors')
    if 'source_checkpoint_sha256'in d:checks.add('trace_checkpoint_binding',str(d['source_checkpoint_sha256'])==sha(args.checkpoint),str(d['source_checkpoint_sha256']),sha(args.checkpoint))
    else:checks.add('trace_checkpoint_binding',False,None,'NPZ scalar source_checkpoint_sha256')
    ids=np.unique(d['episode_id']).astype(int)
    seeds=value(physics,'evaluation.seeds')
    checks.add('evaluation_seed_count',isinstance(seeds,list)and len(seeds)==args.episodes and len(set(seeds))==args.episodes,seeds,args.episodes)
    checks.add('actual_episode_count',len(ids)==args.episodes,int(len(ids)),args.episodes)
    declared_seed_map=value(physics,'evaluation.episode_seed_by_id',{})
    per=[];used_seeds=[]
    for eid in ids:
        idx=np.flatnonzero(d['episode_id']==eid);t=d['timestamps'][idx];errors=[]
        if len(idx)<2 or not np.all(np.diff(t)>0):errors.append('nonmonotonic_or_short_episode')
        if t[0]>.1 or t[0]<-1e-8:errors.append('missing_actual_episode_beginning')
        if not bool(d['reset_events'][idx[0]])or int(d['reset_events'][idx].sum())!=1:errors.append('reset_inside_active_episode_or_missing_initial_reset')
        if not bool(d['done'][idx[-1]])or np.any(d['done'][idx[:-1]]):errors.append('trace_extends_past_done_or_missing_actual_terminal')
        if 'seed_id'in d:
            ss=np.unique(d['seed_id'][idx]);seed=int(ss[0])if len(ss)==1 else None
        else:seed=declared_seed_map.get(str(eid))
        if seed is None or not isinstance(seeds,list)or seed not in seeds:errors.append('missing_or_undeclared_episode_seed')
        used_seeds.append(seed)
        initial=idx[0]
        pregrasp=bool(d['hand_contact_force_left_N'][initial]>args.hand_force or d['hand_contact_force_right_N'][initial]>args.hand_force)
        initial_linear_speed=float(np.linalg.norm(d['box_vel'][initial,:3]))
        initial_table_rest=bool(abs(bottom[initial]-table_z)<=.005 and d['table_contact_force_N'][initial]>args.table_force and initial_linear_speed<=.1)
        if pregrasp:errors.append('pregrasp_at_initial_state')
        if not initial_table_rest:errors.append('box_not_initially_supported_on_table')
        upright=(d['root_pos'][idx,2]>=args.min_root_height)&(tilt[idx]<=args.max_root_tilt_deg)
        fall=bool(not upright.all())
        condition=(d['hand_contact_force_left_N'][idx]>args.hand_force)&(d['hand_contact_force_right_N'][idx]>args.hand_force)&(bottom[idx]>=table_z+args.lift_height)&(d['table_contact_force_N'][idx]<args.table_force)&upright&(feet_total[idx]>=args.min_feet_force)
        start=len(idx)
        while start>0 and condition[start-1]:start-=1
        duration=float(t[-1]-t[start])if start<len(idx)else 0.
        contiguous=True;dt=None;max_gap=None
        if len(t)>2:
            dt=float(np.median(np.diff(t)))
            max_gap=float(np.max(np.diff(t)));contiguous=bool(max_gap<=max(dt*1.5,dt+1e-6))
            if not contiguous:errors.append('missing_physics_samples_or_time_gap')
        if 'termination_reason'in d:
            reason=str(d['termination_reason'][idx[-1]])
            if any(word in reason.lower()for word in ('fall','nan','invalid','diverg')):fall=True
        else:reason=None
        success=bool(not errors and not fall and duration+1e-8>=args.hold_seconds)
        per.append({'episode_id':int(eid),'seed':seed,'samples':len(idx),'duration_s':float(t[-1]-t[0]),'median_recorded_step_s':dt,'max_recorded_step_gap_s':max_gap,'protocol_errors':errors,'initial_pregrasp':pregrasp,'initial_table_rest':initial_table_rest,'initial_box_linear_speed_m_s':initial_linear_speed,'initial_box_bottom_minus_table_m':float(bottom[initial]-table_z),'fell_or_lost_upright':fall,'terminal_required_hold_s':duration,'success':success,'max_box_bottom_clearance_m':float((bottom[idx]-table_z).max()),'terminal_box_bottom_clearance_m':float(bottom[idx[-1]]-table_z),'max_root_tilt_deg':float(tilt[idx].max()),'min_root_height_m':float(d['root_pos'][idx,2].min()),'terminal_reason':reason})
    checks.add('one_distinct_seed_per_actual_episode',len(used_seeds)==len(set(used_seeds))and None not in used_seeds,used_seeds,'unique evaluated seeds')
    checks.add('episode_protocols',all(not x['protocol_errors']for x in per),[{'episode_id':x['episode_id'],'errors':x['protocol_errors']}for x in per if x['protocol_errors']],[])
    return per,{'samples':n,'joint_state_count':int(d['joint_pos'].shape[1]),'quaternion_order':order,'box_size_xyz_m':size,'table_top_z_m':table_z,'lift_definition':'Oriented cuboid minimum worldZ >= actual table topZ + required lift; no COM-only rotation shortcut.'}


def video_checks(args,d,trace_hash,checkpoint_hash,checks):
    import numpy as np
    if args.video is None or args.video_provenance is None:
        checks.add('video_and_actual_trace_provenance_present',False,None,'--video and --video-provenance');return None
    p=read_json(args.video_provenance)
    for key,wanted in (('trace_sha256',trace_hash),('checkpoint_sha256',checkpoint_hash),('video_sha256',sha(args.video)),('render_source','actual_physics_trace')):
        checks.add('video.'+key,p.get(key)==wanted,p.get(key),wanted)
    indices=p.get('source_sample_indices')
    valid=isinstance(indices,list)and len(indices)>1 and all(isinstance(x,int)and 0<=x<len(d['timestamps'])for x in indices)
    checks.add('video_source_sample_indices',valid,len(indices)if isinstance(indices,list)else None,'one valid actual trace index per frame')
    if valid:
        ids=d['episode_id'][indices]
        checks.add('video_uses_one_actual_episode',len(np.unique(ids))==1,list(map(int,np.unique(ids))),1)
        checks.add('video_recorded_state_order',bool(np.all(np.diff(indices)>0)),None,'strictly increasing actual sample rows, no substituted states')
    ffprobe=shutil.which('ffprobe')
    if ffprobe is None:checks.add('video_full_cpu_decode',False,None,'ffprobe available');return None
    proc=subprocess.run([ffprobe,'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames,width,height,avg_frame_rate,duration','-of','json',str(args.video)],capture_output=True,text=True,timeout=180)
    if proc.returncode or proc.stderr.strip():checks.add('video_full_cpu_decode',False,proc.stderr[-2000:],'complete decode without errors');return None
    info=json.loads(proc.stdout)['streams'][0];frames=int(info['nb_read_frames'])
    checks.add('video_full_cpu_decode',frames>1,frames,'>1 decoded frame')
    checks.add('video_frame_source_count',valid and frames==len(indices),frames,len(indices)if valid else None)
    return {'video':record(args.video),'provenance':record(args.video_provenance),'decoded_stream':info,'selection_policy':p.get('episode_selection_policy'),'physics_claim':'Video only visualizes the bound actual simulated trajectory; it is not an independent dynamics test.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('trace','physics','checkpoint'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--video',type=Path);ap.add_argument('--video-provenance',type=Path)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--trace-only',action='store_true')
    ap.add_argument('--episodes',type=int,default=32);ap.add_argument('--required-successes',type=int,default=32)
    ap.add_argument('--hand-force',type=float,default=.5);ap.add_argument('--table-force',type=float,default=.2)
    ap.add_argument('--lift-height',type=float,default=.10);ap.add_argument('--hold-seconds',type=float,default=2.)
    ap.add_argument('--min-root-height',type=float,default=.45);ap.add_argument('--max-root-tilt-deg',type=float,default=35.)
    ap.add_argument('--min-feet-force',type=float,default=5.)
    args=ap.parse_args()
    if any(not math.isfinite(getattr(args,k))for k in ('hand_force','table_force','lift_height','hold_seconds','min_root_height','max_root_tilt_deg','min_feet_force')):ap.error('Acceptance thresholds must be finite')
    if args.output.resolve()in {p.resolve()for p in (args.trace,args.physics,args.checkpoint,args.video,args.video_provenance)if p is not None}:ap.error('Output cannot overwrite a source artifact')
    c=Checks();result={'schema':'kinogen.g1_box_policy.cpu_audit.v1','checked_at_utc':datetime.now(timezone.utc).isoformat(),'verification_script':record(Path(__file__)),'gpu_used':False,'simulation_or_training_launched':False,'thresholds':{k:getattr(args,k)for k in ('episodes','required_successes','hand_force','table_force','lift_height','hold_seconds','min_root_height','max_root_tilt_deg','min_feet_force')},'physical_hardware_validated':False,'final_delivery_verified':False}
    try:
        if not 1<=args.required_successes<=args.episodes:raise ValueError('required successes outside episode count')
        if not args.trace_only and args.episodes!=32:raise ValueError('Final delivery requires exactly32 unique evaluation seeds; use --trace-only for smaller smoke evaluations')
        bounds={'hand_force':(.5,float('inf')),'table_force':(0,.2),'lift_height':(.10,float('inf')),'hold_seconds':(2.,float('inf')),'min_root_height':(.45,float('inf')),'max_root_tilt_deg':(0,35.),'min_feet_force':(5.,float('inf'))}
        for name,(lo,hi)in bounds.items():
            actual=getattr(args,name)
            if not math.isfinite(actual)or not lo<=actual<=hi:raise ValueError('Cannot loosen original acceptance threshold: '+name)
        th,ch=sha(args.trace),sha(args.checkpoint);physics=read_json(args.physics);d=trace_arrays(args.trace)
        result['sources']=[record(p)for p in (args.trace,args.physics,args.checkpoint)]
        physics_checks(physics,th,ch,c)
        per,meta=audit_episodes(d,physics,args,c);result.update(episodes=per,trace_metadata=meta)
        count=sum(x['success']for x in per);result['successful_episodes']=count;result['success_rate']=count/len(per)if per else 0.
        learned=lambda row:row['check'].startswith('training.')or row['check']in ('evaluation.policy_inference_calls','actor_parameters_updated_and_evaluated')
        result['training_provenance_verified']=all(x['passed']for x in c.rows if learned(x))
        result['actual_physics_trace_verified']=all(x['passed']for x in c.rows if not learned(x))
        protocol=c.passed;result['trace_protocol_verified']=protocol
        if not args.trace_only:result['video_audit']=video_checks(args,d,th,ch,c)
        result['trace_training_goal_passed']=bool(protocol and count>=args.required_successes)
        result['final_delivery_verified']=bool(not args.trace_only and c.passed and count>=args.required_successes)
        result['status']='verified_policy_evaluation'if result['final_delivery_verified']else('trace_feedback_only'if args.trace_only and protocol else'policy_goal_failed_or_evidence_not_verified')
    except Exception as exc:
        c.add('audit_input_or_evidence_read',False,str(exc),'complete real artifacts');result['status']='not_verified';result['trace_protocol_verified']=False
    result['checks']=c.rows;result['limitations']=['The JSON live-stage and API-guard evidence must originate from instrumented actual simulation, not declarations synthesized after evaluation.','Sampled/contact-force/task success is simulation evidence, not hardware, friction-model accuracy or exact continuous collision proof.','State playback video alone never proves policy execution; checkpoint, active-write guards, real contacts and actual trace are independently required.']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'output':record(args.output),'status':result['status'],'trace_protocol_verified':result.get('trace_protocol_verified'),'successful_episodes':result.get('successful_episodes'),'success_rate':result.get('success_rate'),'final_delivery_verified':result['final_delivery_verified'],'failed_checks':[x for x in c.rows if not x['passed']]},indent=2))
    if not result.get('trace_protocol_verified'):return 2
    return 0 if(result.get('trace_training_goal_passed')if args.trace_only else result['final_delivery_verified'])else 3


if __name__=='__main__':raise SystemExit(main())
