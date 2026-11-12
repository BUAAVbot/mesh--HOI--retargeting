"""Compare batched SONIC against original controllers on one native plant.

Only one shared physical state is integrated, using the original controllers.
Four separate comparison runs isolate fresh/stale FK and raw/bounded history.
No controller source or training process is modified.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
from collections import deque


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','proxy-model','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--updates',type=int,default=20)
    p.add_argument('--cases',nargs='+',default=['fresh_raw','fresh_applied','stale_raw','stale_applied'])
    p.add_argument('--cold-start',choices=['legacy_reference','immediate_learned'],default='legacy_reference')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np,mujoco as mj,torch
    from two_sitting_ppo_env import Reference,mapping
    from two_sitting_batch_sonic import DualBatchSonic,dependency_paths
    dependency_paths()
    from fourth_sonic_controller import Controller
    torch.set_num_threads(2);a.output.mkdir(parents=True)
    report={'schema':'two_sitting_batch_contract_audit.v1','method':'One original-native-controlled physical plant per comparison case; batch sees exactly the same qpos/qvel without integrating a second state.','updates':a.updates,'physics_dt_s':.005,'policy_dt_s':.02,'cases':{},'inputs':{key:{'path':str(getattr(a,key)),'sha256':hashlib.sha256(getattr(a,key).read_bytes()).hexdigest()}for key in ('model','proxy_model','reference')}}
    snapshots={}

    def run_case(case):
        fresh=case.startswith('fresh_');raw_history=case.endswith('_raw')
        model=mj.MjModel.from_xml_path(str(a.model.resolve()));ref=Reference(model,a.reference,a.device)
        dual=DualBatchSonic(model,ref,1,a.device,a.proxy_model);d=mj.MjData(model)
        native=[];captures={};handles=[]
        def capture(name):
            def hook(module,args):
                td=args[0];captures[name]={'actor_obs':td['actor_obs'].detach().cpu().numpy().copy(),'tokenizer':td['tokenizer'].detach().cpu().numpy().copy()}
            return hook
        for actor in range(2):
            pm=mj.MjModel.from_xml_path(str(a.proxy_model.resolve()))
            c=Controller(pm,dict(ref.singles[actor].z),{'mode':'newft15225','device':a.device,'cpu_threads':2,'delay_steps':3,'imu_latency_s':.0025,'hand_object_feedback':False})
            pd=mj.MjData(pm);mp=mapping(pm,ref.names);sm=ref.maps[actor]
            c.reset(pd);native.append({'model':pm,'data':pd,'controller':c,'pm':mp,'sm':sm})
            handles.append(c.policy.register_forward_pre_hook(capture('native'+str(actor))))
            handles.append(dual.parts[actor]['batch'].policy.register_forward_pre_hook(capture('batch'+str(actor))))
        d.qpos[:]=ref.q[0].detach().cpu().numpy();d.qvel[:]=0.;mj.mj_forward(model,d)
        old=ref.controls(torch.zeros(1,device=a.device),torch.zeros((1,56),device=a.device),0.)[0].detach().cpu().numpy()
        latest=old.copy();rows=[];delayrows=[];native_records=[];batch_records=[]
        def sync(part):
            pm,sm,pd=part['pm'],part['sm'],part['data']
            pd.qpos[pm['rq']:pm['rq']+7]=d.qpos[sm['rq']:sm['rq']+7]
            pd.qvel[pm['rv']:pm['rv']+6]=d.qvel[sm['rv']:sm['rv']+6]
            pd.qpos[pm['qa']]=d.qpos[sm['qa']];pd.qvel[pm['va']]=d.qvel[sm['va']];pd.time=d.time
            mj.mj_forward(part['model'],pd)
        for step in range(a.updates*4):
            elapsed=float(d.time);u=np.zeros(model.nu)
            for part in native:
                sync(part);packet=part['controller'].action(part['data'],elapsed)
                u[part['sm']['aids']]=packet[part['pm']['aids']]
            # This is the same named-joint target clamp as original Motors.
            for sm in ref.maps:
                limited=model.jnt_limited[sm['jids']].astype(bool);ids=sm['aids'][limited];limits=model.jnt_range[sm['jids'][limited]]
                u[ids]=np.clip(u[ids],limits[:,0],limits[:,1])
            if step%4==0:
                tq=torch.as_tensor(d.qpos.copy(),device=a.device,dtype=torch.float32)[None]
                tv=torch.as_tensor(d.qvel.copy(),device=a.device,dtype=torch.float32)[None]
                rot=d.xmat.copy().reshape(model.nbody,3,3);cv=d.cvel.copy()
                if fresh:
                    for actor,part in enumerate(native):
                        rot[ref.maps[actor]['base']]=part['data'].xmat[part['controller'].base].reshape(3,3)
                        cv[ref.maps[actor]['base']]=part['data'].cvel[part['controller'].base]
                beforeq=d.qpos.copy();beforev=d.qvel.copy()
                out=dual.targets(tq,tv,torch.tensor([elapsed],device=a.device),torch.as_tensor(rot,device=a.device,dtype=torch.float32)[None],torch.as_tensor(cv,device=a.device,dtype=torch.float32)[None]).clone()
                if not np.array_equal(beforeq,d.qpos)or not np.array_equal(beforev,d.qvel):raise ValueError('Batch modified the plant')
                latest=out[0].detach().cpu().numpy()
                for sm in ref.maps:
                    limited=model.jnt_limited[sm['jids']].astype(bool);ids=sm['aids'][limited];limits=model.jnt_range[sm['jids'][limited]]
                    latest[ids]=np.clip(latest[ids],limits[:,0],limits[:,1])
                for actor,part in enumerate(native):
                    nc,bc=part['controller'],dual.parts[actor]['batch']
                    nt=captures['native'+str(actor)];bt=captures['batch'+str(actor)]
                    nobs=nt['actor_obs'].reshape(-1);bobs=bt['actor_obs'].reshape(-1)
                    termdiff={};start=0
                    for key,width in [('base_ang_vel',3),('joint_pos',28),('joint_vel',28),('actions',28),('gravity_dir',3)]:
                        termdiff[key]=float(np.max(abs(nobs[start:start+10*width]-bobs[start:start+10*width])));start+=10*width
                    td={key:float(np.max(abs(nc.tokenizer[key].detach().cpu().numpy()-bc.tokens[key].detach().cpu().numpy())))for key in nc.cfg.env_config.obs.group_obs_names.tokenizer}
                    raw=bc.cached[0].detach().cpu().numpy()
                    network_action=(raw-bc.c.default)/bc.c.scale
                    row={'update':step//4,'actor':actor+1,'time_s':elapsed,'actor_obs_maxdiff':float(np.max(abs(nobs-bobs))),'actor_obs_term_maxdiff':termdiff,'tokenizer_flat_maxdiff':float(np.max(abs(nt['tokenizer']-bt['tokenizer']))),'tokenizer_field_maxdiff':td,'raw_body_target_maxdiff_rad':float(np.max(abs(nc.target-raw))),'raw_network_action_maxdiff':float(np.max(abs(nc.last-network_action))),'native_new_action_vs_batch_previous_history_maxdiff':float(np.max(abs(nc.last-bc.last[0].detach().cpu().numpy())))}
                    rows.append(row);native_records.append(nt);batch_records.append(bt)
                # Raw action history follows the original controller; bounded
                # feedback follows current task-residual BatchSonic behavior.
                if raw_history:
                    for part in dual.parts:
                        batch=part['batch'];batch.last.copy_((batch.cached-batch.default)/batch.scale)
            envu=latest.copy()
            if step%4<3 and not(a.cold_start=='immediate_learned'and step<4):envu[ref.body_aids]=old[ref.body_aids]
            delayrows.append({'step':step,'time_s':elapsed,'body_applied_difference_rad':float(np.max(abs(envu[ref.body_aids]-u[ref.body_aids])))})
            d.ctrl[:]=u;mj.mj_step(model,d)
            if step%4==3:
                old=latest.copy()
                if not raw_history:dual.record_applied(torch.as_tensor(latest.copy(),device=a.device,dtype=torch.float32)[None])
        for handle in handles:handle.remove()
        # Full input tensors allow exact term-level independent examination.
        np.savez_compressed(a.output/(case+'_inputs.npz'),native_actor_obs=np.stack([x['actor_obs']for x in native_records]),batch_actor_obs=np.stack([x['actor_obs']for x in batch_records]),native_tokenizer=np.stack([x['tokenizer']for x in native_records]),batch_tokenizer=np.stack([x['tokenizer']for x in batch_records]))
        result={'fresh_proxy_forward_sensor_input':fresh,'previous_action_history':'raw network normalized action'if raw_history else'bounded applied motor target normalized action','max_actor_obs_diff':max(x['actor_obs_maxdiff']for x in rows),'max_tokenizer_flat_diff':max(x['tokenizer_flat_maxdiff']for x in rows),'max_raw_body_target_diff_rad':max(x['raw_body_target_maxdiff_rad']for x in rows),'max_raw_network_action_diff':max(x['raw_network_action_maxdiff']for x in rows),'max_delayed_applied_body_target_diff_rad':max(x['body_applied_difference_rad']for x in delayrows),'initial_update':rows[:2],'updates_detail':rows,'delay_detail':delayrows,'one_physical_plant_only':True,'plant_control':'original native Controller delay3 plus original motor target clamp','batch_cold_start_contract':a.cold_start,'batch_external_delay_comparison':'3 prior target steps plus1 current target step; first cycle uses either legacy reference target or corrected immediate learned target, as declared by cold_start.'}
        print(json.dumps({'case':case,**{k:v for k,v in result.items()if k.startswith('max_')}}),flush=True)
        return result

    for case in a.cases:
        if case not in ('fresh_raw','fresh_applied','stale_raw','stale_applied'):raise ValueError(case)
        report['cases'][case]=run_case(case);gc.collect();torch.cuda.empty_cache()
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(a.output/'report.json')}),flush=True)


if __name__=='__main__':main()
