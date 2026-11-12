"""Inspect the declared reference geometry; this never integrates a state."""
from pathlib import Path
import argparse
import hashlib
import json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--reference',default='outputs/rgb_near_pipeline/reference_v5_wearclearance.npz')
    p.add_argument('--model',default='outputs/rgb_near_pipeline/assets/vita_bench_two52_wear_v2.xml')
    p.add_argument('--output',default='outputs/rgb_near_pipeline/reference_v5_geometry_audit.json')
    a=p.parse_args()
    import numpy as np
    import mujoco as mj
    ref=a.root/a.reference;mp=a.root/a.model;z=np.load(ref,allow_pickle=False)
    m=mj.MjModel.from_xml_path(str(mp));d=mj.MjData(m)
    names=[str(x)for x in z['dof_names']]
    maps=[]
    for pre in('p1_','p2_'):
        rootj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+'vita_demo_free_root')
        ids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,pre+n)for n in names]
        if rootj<0 or min(ids)<0:raise ValueError('Reference joint schema missing from '+pre)
        maps.append((int(m.jnt_qposadr[rootj]),np.array([m.jnt_qposadr[j]for j in ids]),ids))
    sj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'shoe_free');sq=int(m.jnt_qposadr[sj])
    def gn(g):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,int(g))or str(g)
    def bn(b):return mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(b))or'world'
    right=[g for g in range(m.ngeom)if bn(m.geom_bodyid[g]).startswith('p1_rh_')and m.geom_contype[g]]
    visual=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'shoe_visual')
    shoe=[g for g in range(m.ngeom)if gn(g).startswith('shoe_')and m.geom_contype[g]]
    def closest(targets):
        rows=[]
        for h in right:
            for obj in targets:
                rows.append((float(mj.mj_geomDistance(m,d,h,obj,2.,np.zeros(6))),gn(h),gn(obj)))
        return min(rows)
    max_pen=0.;worst=None;window=[];bounds=0;rows=[];max_hand_self=0.;max_shoe_hand=0.;max_shoe_hand_row=None;max_mimic=0.
    mimic=[eq for eq in range(m.neq)if m.eq_type[eq]==mj.mjtEq.mjEQ_JOINT and m.eq_active0[eq]]
    handkeys=('p1_lh_','p1_rh_','p2_lh_','p2_rh_')
    for f in range(len(z['time_s'])):
        for actor,(rq,qa,jids)in enumerate(maps):
            q=z['qpos'][f,actor];d.qpos[rq:rq+7]=q[:7];d.qpos[qa]=q[7:]
            bounds+=int(np.sum((q[7:]<m.jnt_range[jids,0]-1e-7)|(q[7:]>m.jnt_range[jids,1]+1e-7)))
        d.qpos[sq:sq+7]=z['object_target_pose'][f]
        mj.mj_forward(m,d)
        for eq in mimic:
            j1,j2=int(m.eq_obj1id[eq]),int(m.eq_obj2id[eq])
            if j1>=0 and j2>=0:
                x=d.qpos[m.jnt_qposadr[j2]];expected=sum(m.eq_data[eq,k]*x**k for k in range(5))
                max_mimic=max(max_mimic,abs(float(d.qpos[m.jnt_qposadr[j1]]-expected)))
        pen=0.;pair=None
        for c in d.contact:
            if -float(c.dist)>pen:pen=-float(c.dist);pair=[gn(c.geom1),gn(c.geom2)]
            b1,b2=bn(m.geom_bodyid[c.geom1]),bn(m.geom_bodyid[c.geom2])
            if any(b1.startswith(k)and b2.startswith(k)for k in handkeys):max_hand_self=max(max_hand_self,-float(c.dist))
            if (c.geom1 in shoe and b2.startswith(handkeys))or(c.geom2 in shoe and b1.startswith(handkeys)):
                if -float(c.dist)>max_shoe_hand:max_shoe_hand=-float(c.dist);max_shoe_hand_row={'frame':f,'time_s':float(z['time_s'][f]),'pair':[gn(c.geom1),gn(c.geom2)],'penetration_m':max_shoe_hand}
        rows.append({'frame':f,'time_s':float(z['time_s'][f]),'max_model_contact_penetration_m':pen,'pair':pair})
        if pen>max_pen:max_pen=pen;worst=rows[-1]
        if 230<=f<=275:
            # Critical: use the foot-bound pose even after the nominal release,
            # so a hand-defined object target cannot manufacture a grasp gap.
            d.qpos[sq:sq+7]=z['shoe_wear_only_reference_pose'][f]
            mj.mj_forward(m,d)
            pr=closest(shoe);vr=closest([visual])
            window.append({'frame':f,'time_s':float(z['time_s'][f]),
                           'right_hand_to_foot_bound_shoe_proxy_signed_m':pr[0],'proxy_pair':pr[1:],
                           'right_hand_to_foot_bound_shoe_visual_hull_signed_m':vr[0],'visual_pair':vr[1:],
                           'geometric_handover_candidate':bool(-.02<=pr[0]<=.03)})
    report={'schema':'rgb_near_reference_geometry_audit.v1','reference_sha256':sha(ref),'model_sha256':sha(mp),
            'frames':len(z['time_s']),'actors':2,'joint_limit_violations':bounds,
            'max_model_contact_penetration_m':max_pen,'worst_collision':worst,
            'handover_foot_bound_shoe_geometry':window,
            'handover_candidate_frames':[r['frame']for r in window if r['geometric_handover_candidate']],
            'checks_do_not_prove_dynamics':True,'source_hands_valid':False,
            'collision_scope':'Exactly the collision pairs enabled by the model identified by SHA; this may include the explicit within-hand self-collision variant. No disabled body pairs are certified.',
            'max_within_hand_self_penetration_m':max_hand_self,
            'max_shoe_hand_contact_penetration_m':max_shoe_hand,
            'worst_shoe_hand_contact':max_shoe_hand_row,
            'active_joint_mimic_equalities':len(mimic),'max_reference_mimic_residual_rad':max_mimic,
            'max_physical_hand_joint_reference_velocity_rad_s':float(np.max(np.abs(np.diff(z['qpos'][:,:,7+np.array([k for k,nm in enumerate(names)if nm.startswith(('lh_','rh_'))])],axis=0)/np.diff(z['time_s'])[:,None,None])))if len(z['time_s'])>1 else 0.,
            'per_frame_collisions':rows}
    out=a.root/a.output;out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items()if k not in('per_frame_collisions','handover_foot_bound_shoe_geometry')}),flush=True)


if __name__=='__main__':main()
