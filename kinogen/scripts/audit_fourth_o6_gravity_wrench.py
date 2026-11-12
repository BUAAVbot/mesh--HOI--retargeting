"""Gravity + eccentric-moment feasibility at observed thumb/index contacts.

Exact MuJoCo pyramidal friction rays are decoded by the running library. For
elliptic cones, outer/inner linear bounds are labelled separately. No absent
contact is invented, and infeasibility describes only observed snapshots.
"""
from pathlib import Path
import argparse,itertools,json
import numpy as np
import mujoco as mj
from scipy.optimize import linprog
from fourth_contact_wrench import two_point_gravity_axis_certificate

R=Path(__file__).resolve().parents[1]
def rays(contact,outer=False):
    dim=int(contact.dim);count=dim-1
    if dim==1:return[np.asarray([1.,0,0,0,0,0])]
    result=[]
    if outer:
        for signs in itertools.product([-1.,1.],repeat=count):
            force=np.zeros(6);force[0]=1.;force[1:dim]=contact.friction[:count]*np.asarray(signs);result.append(force)
    else:
        for i in range(2*count):
            edge=np.zeros(2*dim);edge[i]=1.;native=np.zeros(dim+1)
            # The Python binding infers dim from the mu vector length; C only
            # consumes its first dim-1 entries; the wrapper also requires two
            # dummy pyramid entries. Reconstruct actual efc forces to verify.
            mj.mju_decodePyramid(native,edge,np.r_[np.asarray(contact.friction[:count],float),0.])
            force=np.zeros(6);force[:dim]=native[:dim];result.append(force)
    return result

def solve(columns,normal_rows,gravity_wrench,cap):
    A=np.asarray(columns).T;N=np.asarray(normal_rows).T;scale=np.asarray([1,1,1,10,10,10.])
    fixed=linprog(N.sum(0),A_ub=N,b_ub=np.full(2,cap),A_eq=scale[:,None]*A,b_eq=-scale*gravity_wrench,bounds=(0,None),method='highs')
    unrestricted=linprog(N.sum(0),A_eq=scale[:,None]*A,b_eq=-scale*gravity_wrench,bounds=(0,None),method='highs')
    force_only=linprog(N.sum(0),A_ub=N,b_ub=np.full(2,cap),A_eq=A[:3],b_eq=-gravity_wrench[:3],bounds=(0,None),method='highs')
    # Minimise necessary maximum normal force per digit, without assuming cap.
    augmented=np.column_stack([N,-np.ones(2)]);target=np.r_[np.zeros(A.shape[1]),1.]
    minimax=linprog(target,A_ub=augmented,b_ub=np.zeros(2),A_eq=np.column_stack([scale[:,None]*A,np.zeros(6)]),b_eq=-scale*gravity_wrench,bounds=(0,None),method='highs')
    force_minimax=linprog(target,A_ub=augmented,b_ub=np.zeros(2),A_eq=np.column_stack([A[:3],np.zeros(3)]),b_eq=-gravity_wrench[:3],bounds=(0,None),method='highs')
    result={'cap_N_per_digit':cap,'cap_feasible':bool(fixed.success),'cap_solver_status':int(fixed.status),
            'force_only_without_moment_feasible_at_cap':bool(force_only.success),
            'force_only_minimum_maximum_digit_normal_N':float(force_minimax.x[-1])if force_minimax.success else None,
            'no_normal_budget_full_wrench_feasible':bool(unrestricted.success),
            'no_normal_budget_solver_status':int(unrestricted.status),
            'minimum_possible_maximum_digit_normal_N':float(minimax.x[-1])if minimax.success else None,
            'unbounded_normal_budget_wrench_feasible':bool(minimax.success)}
    if fixed.success:
        result.update(normal_per_digit_N=(N@fixed.x).tolist(),total_normal_N=float(N.sum(0)@fixed.x),
                      force_moment_equilibrium_max_residual=float(np.max(abs(A@fixed.x+gravity_wrench))))
    return result

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--trial',type=Path,default=R/'outputs/fourth_full_sequence/physics_sonic15225_o6_obj1_seedfix_v2_6s')
p.add_argument('--output',type=Path,default=R/'reports/fourth_o6_observed_thumb_index_gravity_wrench.json')
p.add_argument('--normal-cap-N',type=float,default=2.7);a=p.parse_args()
r=json.loads((a.trial/'report.json').read_text());m=mj.MjModel.from_binary_path(r['model']['path']);z=dict(np.load(a.trial/'physical_trace.npz',allow_pickle=False));c=dict(np.load(a.trial/'physical_contacts.npz',allow_pickle=False))
body_names=list(map(str,c['body_names']));obj=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'target_object_01');force=np.zeros((len(z['ctrl']),5));digits=['thumb','index','middle','ring','pinky']
for index,bs in enumerate(c['body_ids']):
    if obj not in bs:continue
    other=int(bs[1]if bs[0]==obj else bs[0]);digit=next((i for i,name in enumerate(digits)if body_names[other].startswith('rh_'+name)),None)
    if digit is not None:force[int(c['interval_index'][index]),digit]+=float(c['normal_force_N'][index])
both=(force[:,0]>.03)&(force[:,1]>.03);paired=np.flatnonzero(both)
pair_stats={'thumb_index_simultaneous_duration_s':float(len(paired)*m.opt.timestep),
            'maximum_thumb_index_total_normal_N':float(np.max(force[:,:2].sum(1))),
            'all_actual_digit_pair_durations_s':{digits[i]+'+'+digits[j]:float(np.count_nonzero((force[:,i]>.03)&(force[:,j]>.03))*m.opt.timestep)for i in range(5)for j in range(i+1,5)}}
if not len(paired):
    report={'schema':'fourth.observed_gravity_wrench.v1','status':'No simultaneous actual thumb/index contacts; no wrench LP used to reject candidate.',**pair_stats}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(0)
chosen=set(paired[np.linspace(0,len(paired)-1,min(10,len(paired))).round().astype(int)].tolist())
chosen.add(int(np.argmax(np.where(both,force[:,:2].sum(1),-1.))))
d=mj.MjData(m);mj.mj_setState(m,d,z['initial_state'],int(np.asarray(z['state_spec']).reshape(-1)[0]));mj.mj_forward(m,d)
snapshots=[];maximum_replay_error=0.
for step in range(max(chosen)+1):
    d.ctrl[:]=z['ctrl'][step];mj.mj_step(m,d)
    maximum_replay_error=max(maximum_replay_error,float(np.max(abs(d.qpos-z['qpos'][step+1]))))
    if step not in chosen:continue
    contacts=[];normals=[[],[]];columns=[];normal_rows=[];outer_columns=[];outer_rows=[];table_columns=[];actual_edges=[];actual_selected_wrench=np.zeros(6)
    for ci,contact in enumerate(d.contact[:d.ncon]):
        gs=[int(contact.geom1),int(contact.geom2)];bs=m.geom_bodyid[gs]
        if obj not in bs:continue
        other=int(bs[1]if bs[0]==obj else bs[0]);name=body_names[other];digit=0 if name.startswith('rh_thumb')else 1 if name.startswith('rh_index')else-1
        is_table=any((m.geom(g).name or'').startswith('table_support')for g in gs)
        if digit<0 and not is_table:continue
        actual=np.zeros(6);mj.mj_contactForce(m,d,ci,actual)
        if actual[0]<=.03:continue
        sign=1. if int(bs[1])==obj else-1.;rotation=np.asarray(contact.frame).reshape(3,3).T
        normal=sign*rotation[:,0]
        if digit>=0:normals[digit].append(normal)
        point=np.asarray(contact.pos);arm=point-d.xipos[obj]
        if is_table:
            for local in rays(contact):
                F=sign*(rotation@local[:3]);T=sign*(rotation@local[3:])+np.cross(arm,F);table_columns.append(np.r_[F,T])
            continue
        contacts.append({'digit':digits[digit],'geoms':[m.geom(g).name for g in gs],'point_world_m':point.tolist(),
                         'force_on_object_normal_direction':normal.tolist(),'actual_normal_N':float(actual[0]),'friction':contact.friction.tolist(),'contact_dim':int(contact.dim)})
        actual_force=sign*(rotation@actual[:3]);actual_selected_wrench+=np.r_[actual_force,sign*(rotation@actual[3:])+np.cross(arm,actual_force)]
        if m.opt.cone==mj.mjtCone.mjCONE_PYRAMIDAL:
            edge_count=1 if int(contact.dim)==1 else 2*(int(contact.dim)-1)
            actual_edges.extend(d.efc_force[int(contact.efc_address):int(contact.efc_address)+edge_count].tolist())
        for outer in [False,True]:
            for local in rays(contact,outer):
                F=sign*(rotation@local[:3]);T=sign*(rotation@local[3:])+np.cross(arm,F);nrow=np.zeros(2);nrow[digit]=float(local[0])
                (outer_columns if outer else columns).append(np.r_[F,T]);(outer_rows if outer else normal_rows).append(nrow)
    if not all(normals):continue
    external=np.r_[float(m.body_mass[obj])*m.opt.gravity,np.zeros(3)]
    exact=bool(m.opt.cone==mj.mjtCone.mjCONE_PYRAMIDAL);inner=solve(columns,normal_rows,external,a.normal_cap_N)
    outer=None if exact else solve(outer_columns,outer_rows,external,a.normal_cap_N)
    support_control=solve(columns+table_columns,normal_rows+[np.zeros(2)for _ in table_columns],external,a.normal_cap_N)if table_columns else None
    reconstruction=float(np.max(abs(np.asarray(columns).T@np.asarray(actual_edges)-actual_selected_wrench)))if exact else None
    if reconstruction is not None and reconstruction>1e-7:raise ValueError(('Friction ray / object sign reconstruction mismatch',reconstruction))
    centre=np.mean([item['point_world_m']for item in contacts],axis=0)
    torque=np.cross(d.xipos[obj]-centre,external[:3]);dot=min(float(x@y)for x in normals[0]for y in normals[1])
    certificate=two_point_gravity_axis_certificate([item['point_world_m']for item in contacts],d.xipos[obj],float(m.body_mass[obj]),m.opt.gravity)if all(item['contact_dim']==3 for item in contacts)else None
    snapshots.append({'time_s':float(d.time),'minimum_real_opposed_normal_dot':dot,'object_com_world_m':d.xipos[obj].tolist(),
                      'gravity_torque_about_contact_cluster_Nm':torque.tolist(),'gravity_torque_magnitude_Nm':float(np.linalg.norm(torque)),
                      'contact_cone':'Exact physical MuJoCo pyramidal cone'if exact else'Pyramidal inner bound and independent-component outer bound of elliptic cone',
                      'contacts':contacts,'wrench_feasibility':inner,'elliptic_outer_bound':outer,
                      'actual_recorded_wrench_from_constraint_edges_reconstruction_max_error':reconstruction,
                      'two_point_unresisted_gravity_axis_certificate':certificate,
                      'positive_control_with_actual_table_support':support_control})
if maximum_replay_error>1e-7:raise ValueError('Frozen real-state replay mismatch')
report={'schema':'fourth.observed_thumb_index_gravity_wrench.v1','trial':str(a.trial),'frozen_replay_qpos_max_error':maximum_replay_error,
        'mass_kg':float(m.body_mass[obj]),'weight_N':float(np.linalg.norm(m.body_mass[obj]*m.opt.gravity)),
        'model_cone_type':int(m.opt.cone),**pair_stats,'observed_snapshots':snapshots,
        'cap_feasible_snapshot_count':sum(int(item['wrench_feasibility']['cap_feasible'])for item in snapshots),
        'scope':'Actual observed contact locations/friction/dimensions; gravity and eccentric moment balance about object COM; no table support or other fingers credited.',
        'limitations':['Snapshot feasibility is not a dynamic stability proof or a statement about every possible future grasp.',
                       'Only simultaneous actual thumb/index contacts were analysed; missing contacts do not refute the pregrasp candidate.',
                       '2.7 N is a tested normal-force budget per digit, not an increased actuator limit.'],
        'actual_model_or_controller_modified':False,'new_grasp_or_lift_success_claimed':False}
a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({**pair_stats,'snapshots':[{'time_s':item['time_s'],'normal_dot':item['minimum_real_opposed_normal_dot'],'eccentric_gravity_moment_Nm':item['gravity_torque_magnitude_Nm'],**item['wrench_feasibility'],
    'table_support_positive_control_feasible':None if item['positive_control_with_actual_table_support']is None else item['positive_control_with_actual_table_support']['cap_feasible']}for item in snapshots],'frozen_replay_qpos_max_error':maximum_replay_error}),flush=True)
