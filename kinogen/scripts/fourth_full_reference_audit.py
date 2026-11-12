"""Kinematic whole-scene audit of the full reference; no integration claims."""
from pathlib import Path
import argparse,json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ik-self-collision',action='store_true')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    z=np.load(a.reference,allow_pickle=False);m=mj.MjModel.from_xml_path(str(a.model));d=mj.MjData(m)
    if a.ik_self_collision:
        for gid in range(m.ngeom):
            bn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[gid]))or'world'
            if bn!='world'and not bn.startswith('target_object_')and m.geom_contype[gid]>0:m.geom_conaffinity[gid]|=int(m.geom_contype[gid])
    names=[str(n)for n in z['dof_names']];jids=np.asarray([mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)for n in names]);qa=m.jnt_qposadr[jids]
    oq=[int(m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,f'target_object_{i:02d}_free')])for i in range(1,7)]
    floor=float(z['physics_floor_z_m']);rows=[]
    feet=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,s+'_ankle_roll_link')for s in ['left','right']]
    ancestors=[]
    for bid in range(m.nbody):
        path={};cur=bid;depth=0
        while cur>0:path[cur]=depth;cur=int(m.body_parentid[cur]);depth+=1
        path[0]=depth;ancestors.append(path)
    def structural(b1,b2):
        common=set(ancestors[b1]).intersection(ancestors[b2])
        return min(ancestors[b1][b]+ancestors[b2][b]for b in common)<=3
    for frame,q in enumerate(z['qpos']):
        d.qpos[:7]=q[:7];d.qpos[qa]=q[7:]
        for oi,adr in enumerate(oq):d.qpos[adr:adr+7]=z['object_target_pose'][frame,oi]
        mj.mj_forward(m,d)
        max_depth=0.;actionable=0.;hip_table=0.;hand_table=0.;pairs=[];allpairs=[];bykind={k:0. for k in ['object_object','robot_self','robot_object','robot_scene','object_scene','native_filtered_self']}
        for ci in range(d.ncon):
            c=d.contact[ci];g=[int(c.geom1),int(c.geom2)];bn=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[x]))or'world'for x in g];gn=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,x)or''for x in g]
            depth=max(0.,-float(c.dist));max_depth=max(max_depth,depth)
            if depth>.001:
                obj=[n.startswith('target_object_')for n in bn]
                robot=[n!='world'and not o for n,o in zip(bn,obj)]
                kind=('object_object'if all(obj)else'robot_self'if all(robot)else'robot_object'if any(obj)and any(robot)else'robot_scene'if any(robot)else'object_scene')
                if all(robot)and(structural(int(m.geom_bodyid[g[0]]),int(m.geom_bodyid[g[1]]))or not any(n.startswith(('lh_','rh_','left_wrist','right_wrist'))for n in bn)):kind='native_filtered_self'
                if kind!='native_filtered_self':actionable=max(actionable,depth)
                imputed=any(not z['object_observed_mask'][frame,int(n[-2:])-1]for n,o in zip(bn,obj)if o)
                imputed=imputed or(any(robot)and not z['human_observed_mask'][frame])
                bykind[kind]=max(bykind[kind],depth)
                allpairs.append({'body_names':bn,'geom_names':gn,'depth_m':depth,'kind':kind,'reference_contains_imputation':bool(imputed)})
            if 'table_support'in gn:
                if any(n.startswith(('base_link','left_hip','right_hip','waist_'))for n in bn):hip_table=max(hip_table,depth)
                if any(n.startswith(('lh_','rh_','left_wrist','right_wrist'))for n in bn):hand_table=max(hand_table,depth)
                if depth>.02:pairs.append({'body_names':bn,'geom_names':gn,'depth_m':depth})
        lateral=np.asarray([-2*q[3]*q[6],q[3]*q[3]-q[6]*q[6],0.])
        width=float((d.xpos[feet[0]]-d.xpos[feet[1]])@lateral)
        rows.append({'frame':frame,'max_penetration_m':max_depth,'max_actionable_penetration_m':actionable,'max_depth_by_kind_m':bykind,'hip_table_penetration_m':hip_table,'hand_table_penetration_m':hand_table,'signed_foot_width_m':width,'bad_table_pairs':pairs,'top_pairs':sorted(allpairs,key=lambda r:-r['depth_m'])[:5]})
    v=np.diff(z['qpos'][:,7:],axis=0)/np.diff(z['time_s'])[:,None]
    report={'schema':'fourth_full_reference_kinematic_audit.v1','reference':str(a.reference),'model':str(a.model),'frames':len(rows),
            'virtual_self_collision_enabled':a.ik_self_collision,
            'max_penetration_m':max(r['max_penetration_m']for r in rows),'max_hip_table_penetration_m':max(r['hip_table_penetration_m']for r in rows),
            'max_hand_table_penetration_m':max(r['hand_table_penetration_m']for r in rows),'signed_foot_width_min_m':min(r['signed_foot_width_m']for r in rows),
            'max_actionable_penetration_m':max(r['max_actionable_penetration_m']for r in rows),'native_self_filter_note':'Original SONIC plant excludes robot-self. Source structural/nearby proxies and native body/body pairs are separately labelled native_filtered_self, not reported as motion repair failure.',
            'initial':rows[0],'frames_hip_table_gt1mm':[r['frame']for r in rows if r['hip_table_penetration_m']>.001],
            'frames_hand_table_gt20mm':[r['frame']for r in rows if r['hand_table_penetration_m']>.02],
            'max_joint_speed_rad_s':float(np.max(np.abs(v))),'joint_limit_violation_max_rad':float(np.max(np.maximum(m.jnt_range[jids,0]-z['qpos'][:,7:],z['qpos'][:,7:]-m.jnt_range[jids,1]))),
            'dynamics_run':False,'rows':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    kinds=list(rows[0]['max_depth_by_kind_m'])
    np.savez_compressed(a.output.with_suffix('.npz'),frame_ids=np.arange(len(rows)),
                        max_penetration_m=np.asarray([r['max_penetration_m']for r in rows]),
                        max_actionable_penetration_m=np.asarray([r['max_actionable_penetration_m']for r in rows]),
                        hip_table_penetration_m=np.asarray([r['hip_table_penetration_m']for r in rows]),
                        hand_table_penetration_m=np.asarray([r['hand_table_penetration_m']for r in rows]),
                        signed_foot_width_m=np.asarray([r['signed_foot_width_m']for r in rows]),
                        penetration_kind_names=np.asarray(kinds),penetration_by_kind_m=np.asarray([[r['max_depth_by_kind_m'][k]for k in kinds]for r in rows]))
    print(json.dumps({k:v for k,v in report.items()if k!='rows'}))


if __name__=='__main__':main()
