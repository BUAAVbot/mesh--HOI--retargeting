"""Read-only quasistatic wrench feasibility on frozen actual contact geometry.

An optimistic box friction cone is intentional: infeasibility under it is a
stronger warning than failure under a conservative friction pyramid. This is
not a physical rollout, and a feasible solution does not prove stable grasp.
"""
from pathlib import Path
import argparse, json
import numpy as np
import mujoco as mj
from scipy.optimize import linprog

p=argparse.ArgumentParser();p.add_argument('--trial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--times',type=float,nargs='+',default=[4.20,4.28,4.30,4.36,4.44]);p.add_argument('--normal-cap',type=float,default=3.0);a=p.parse_args()
r=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(r['model']['path']);z=dict(np.load(a.trial/'physical_trace.npz'));h=dict(np.load(a.trial/'sonic_contact_feedback.npz'));bid=int(z['object_body_ids'][0]);d=mj.MjData(m);rows=[]
for target in a.times:
    k=int(np.argmin(abs(z['time']-target)));d.qpos[:]=z['qpos'][k];d.qvel[:]=z['qvel'][k];d.ctrl[:]=z['ctrl'][min(k,len(z['ctrl'])-1)];mj.mj_forward(m,d)
    contacts=[]
    for ci in range(d.ncon):
        c=d.contact[ci];g1,g2=map(int,c.geom);b1,b2=map(int,m.geom_bodyid[[g1,g2]])
        if bid not in (b1,b2):continue
        obj2=b2==bid;hg=g1 if obj2 else g2;hb=b1 if obj2 else b2;name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,hg)or'';bn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,hb)or''
        if not bn.startswith('rh_'):continue
        digit=next((x for x in ['thumb','index']if x in name),None)
        if digit is None:continue
        frame=np.asarray(c.frame).reshape(3,3).copy();frame*=1 if obj2 else-1
        # First row is normal force direction ON OBJECT; remaining rows span its tangent plane.
        basis=frame.T;lever=np.asarray(c.pos)-d.xipos[bid];W=np.zeros((6,6));W[:3,:3]=basis
        W[3:,:3]=np.column_stack([np.cross(lever,basis[:,i])for i in range(3)]);W[3:,3:]=basis
        contacts.append({'digit':digit,'hand_geom':name,'position_m':np.asarray(c.pos).tolist(),'normal_on_object':frame[0].tolist(),'signed_distance_m':float(c.dist),'friction':np.asarray(c.friction).tolist(),'wrench_matrix':W})
    n=len(contacts);nv=n*6+1;obj=np.zeros(nv);obj[-1]=-1;Aeq=np.zeros((6,nv));Aub=[];bub=[];bounds=[]
    for i,c in enumerate(contacts):
        Aeq[:,i*6:i*6+6]=c['wrench_matrix'];bounds.extend([(0,None)]+[(None,None)]*5)
        # optimistic independent tangential/torsional/rolling bounds
        f=c['friction'];mu=[f[0],f[1],f[2],f[3],f[4]]
        for j,coefficient in enumerate(mu,start=1):
            for sign in (-1,1):
                row=np.zeros(nv);row[i*6+j]=sign;row[i*6]=-coefficient;Aub.append(row);bub.append(0.)
    for digit in ['thumb','index']:
        row=np.zeros(nv)
        for i,c in enumerate(contacts):
            if c['digit']==digit:row[i*6]=1
        Aub.append(row);bub.append(a.normal_cap)
    # scale weight through lambda; maximizing lambda quantifies necessary force closure.
    Aeq[:3,-1]=m.body_mass[bid]*m.opt.gravity;bounds.append((0,None))
    result=linprog(obj,A_ub=np.asarray(Aub),b_ub=np.asarray(bub),A_eq=Aeq,b_eq=np.zeros(6),bounds=bounds,method='highs')
    for c in contacts:c.pop('wrench_matrix')
    row={'time_s':float(z['time'][k]),'object_COM_world_m':d.xipos[bid].tolist(),'object_body_origin_world_m':d.xpos[bid].tolist(),'contact_count':n,'contacts':contacts,'normal_force_cap_per_digit_N':a.normal_cap,'weight_N':float(m.body_mass[bid]*np.linalg.norm(m.opt.gravity)),'LP_status':int(result.status),'LP_message':result.message,'maximum_sustainable_weight_fraction_under_optimistic_box_cone':float(result.x[-1])if result.success else None,'contact_normal_solution_N':result.x[:-1].reshape(n,6)[:,0].tolist()if result.success else None}
    ah=int(np.argmin(abs(h['arm_feedback_time']-z['time'][k])));row['arm_goal']=h['arm_feedback_goal'][ah,1].tolist();row['actual_palm']=h['arm_feedback_actual_palm'][ah,1].tolist();row['geometric_palm']=h['arm_feedback_geometric_palm'][ah,1].tolist();row['arm_phase']=str(h['arm_feedback_phase'][ah,1]);rows.append(row)
out={'scope':'Read-only private mj_forward on recorded actual states; no state changes or physical rollout. Optimistic independent friction bounds, no finger motor limits; failure is diagnostic, feasibility is not proof.','trial':str(a.trial),'samples':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'scope':out['scope'],'samples':[{k:v for k,v in row.items()if k!='contacts'}for row in rows]},indent=2),flush=True)
