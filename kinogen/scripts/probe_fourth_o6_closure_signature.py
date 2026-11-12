"""Read-only hand-local closure signature from the original O6 geometry."""
from pathlib import Path
import argparse,json
import numpy as np
import mujoco as mj
from scipy.optimize import brentq

p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);hand=m.body('rh_hand_base_link').id
masters=[m.joint('rh_'+name).id for name in ['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']]
qa=m.jnt_qposadr[masters];couplings=[(int(m.eq_obj1id[e]),int(m.eq_obj2id[e]),float(m.eq_data[e,1]))for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT]
digits=['thumb','index','middle','ring','pinky'];geoms={digit:[g for g in range(m.ngeom)if (m.geom(g).name or'').startswith('collision_rh_'+digit)]for digit in digits}
localverts={}
for digit,gs in geoms.items():
    for g in gs:
        mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);localverts[g]=m.mesh_vert[start:start+m.mesh_vertnum[mesh]].copy()
def cloud(q):
    d.qpos[:]=m.qpos0;d.qpos[qa]=q
    for child,parent,ratio in couplings:d.qpos[m.jnt_qposadr[child]]=ratio*d.qpos[m.jnt_qposadr[parent]]
    mj.mj_kinematics(m,d);R=d.xmat[hand].reshape(3,3);origin=d.xpos[hand];allv={};tip={}
    for digit,gs in geoms.items():
        parts=[(g,(localverts[g]@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]-origin)@R)for g in gs]
        allv[digit]=np.concatenate([v for _,v in parts]);dist=[v for g,v in parts if'distal'in m.geom(g).name];tip[digit]=np.concatenate(dist).mean(0)
    return allv,tip
rows=[];best=[]
for yaw in [.1,.35,.6,.85,1.1,1.24]:
    sweep=[]
    for s in np.linspace(0,1,9):
        q=np.array([yaw,.05+s*.52,*([.1+s*1.35]*4)]);v,tip=cloud(q);bank=np.mean([tip[x]for x in digits[1:]],axis=0);axis=bank-tip['thumb'];axis/=np.linalg.norm(axis);thumb=v['thumb']@axis;thumbidx=int(np.argmax(thumb));thumbpoint=v['thumb'][thumbidx];gaps={};points={}
        for digit in digits[1:]:
            projected=v[digit]@axis;pi=int(np.argmin(projected));gaps[digit]=float(projected[pi]-thumb.max());points[digit]=v[digit][pi].tolist()
        row={'closure_fraction':float(s),'six_command_rad':q.tolist(),'tip_centers_hand_base_m':{k:x.tolist()for k,x in tip.items()},'thumb_to_four_axis_hand_base':axis.tolist(),'thumb_support_point_hand_base_m':thumbpoint.tolist(),'four_support_points_hand_base_m':points,'projected_clearance_m':gaps,'tip_bank_span_m':float(np.max(np.linalg.norm(np.array([tip[x]for x in digits[1:]])-bank,axis=1))*2)};sweep.append(row)
    centers0=sweep[0]['tip_centers_hand_base_m'];centers1=sweep[-1]['tip_centers_hand_base_m'];movement={digit:(np.array(centers1[digit])-centers0[digit]).tolist()for digit in digits};rows.append({'thumb_yaw_rad':yaw,'open_to_closed_tip_displacement_hand_base_m':movement,'samples':sweep})
    for i in range(1,len(sweep)-1):
        row=sweep[i];gap=np.array(list(row['projected_clearance_m'].values()));axis=np.array(row['thumb_to_four_axis_hand_base']);vel=np.mean([np.array(sweep[i+1]['tip_centers_hand_base_m'][x])-sweep[i-1]['tip_centers_hand_base_m'][x]for x in digits[1:]],axis=0)-(np.array(sweep[i+1]['tip_centers_hand_base_m']['thumb'])-sweep[i-1]['tip_centers_hand_base_m']['thumb']);closing=-float(vel@axis)
        if gap.min()>0 and gap.max()<.09:best.append({'thumb_yaw_rad':yaw,**row,'relative_normal_closing_travel_per_quarter_sweep_m':closing,'gap_spread_m':float(gap.max()-gap.min()),'score':float(abs(gap.mean()-.014)+gap.std()+max(0,-closing)*5)})
best.sort(key=lambda x:x['score'])
aligned=[]
anchor=next(x for x in best if x['thumb_yaw_rad']==1.24 and x['closure_fraction']==.5);axis=np.asarray(anchor['thumb_to_four_axis_hand_base'])
for desired_gap in [.008,.012,.016,.020]:
    q=np.asarray(anchor['six_command_rad']);v,tip=cloud(q);thumbsupport=float(np.max(v['thumb']@axis))
    roots={}
    for di,digit in enumerate(digits[1:]):
        def residual(value):
            probe=q.copy();probe[di+2]=value;vv,_=cloud(probe);return float(np.min(vv[digit]@axis)-thumbsupport-desired_gap)
        grid=np.linspace(.05,1.45,30);intervals=[(lo,hi)for lo,hi in zip(grid[:-1],grid[1:])if residual(lo)*residual(hi)<=0]
        options=[brentq(residual,lo,hi,xtol=1e-12)for lo,hi in intervals]
        if options:q[di+2]=min(options,key=lambda r:abs(r-.775));roots[digit]=float(q[di+2])
    vv,tt=cloud(q);support={digit:vv[digit][int(np.argmin(vv[digit]@axis))]for digit in digits[1:]};gaps={digit:float(np.min(vv[digit]@axis)-thumbsupport)for digit in digits[1:]};derivatives={}
    for di,digit in enumerate(digits[1:]):
        probe=q.copy();probe[di+2]+=.001;pv,_=cloud(probe);derivatives[digit]=float((np.min(pv[digit]@axis)-thumbsupport-gaps[digit])/.001)
    aligned.append({'desired_directional_gap_m':desired_gap,'axis_hand_base':axis.tolist(),'six_command_rad':q.tolist(),'all_four_roots_found':len(roots)==4,'four_directional_gaps_m':gaps,'gap_derivative_m_per_MCP_rad':derivatives,'thumb_support_point_hand_base_m':anchor['thumb_support_point_hand_base_m'],'four_support_points_hand_base_m':{k:x.tolist()for k,x in support.items()},'tip_centers_hand_base_m':{k:x.tolist()for k,x in tt.items()}})
out={'schema':'fourth.o6_hand_local_closure_signature.v1','scope':'Original hand mesh forward kinematics only; support-plane gaps along thumb-to-four center axis are directional clearances, not exact signed Euclidean distances. Independent MCP adjustment and arm IK still required; no physical contact/lift claim.','hand_base_body':m.body(hand).name,'coordinates':'meters in original right hand_base, +z generally extends along open fingers; sign and axis from actual sampled mesh','sweeps':rows,'candidate_common_closing_regions':best[:12],'independent_MCP_aligned_directional_gaps':aligned};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'output':str(a.output),'hand':m.body(hand).name,'aligned':aligned}),flush=True)

# Free thumb pitch to open wider while keeping the common closing normal fixed.
wide=[]
for desired_gap in [.010,.012,.016,.020]:
    options=[]
    for pitch in np.linspace(.02,.55,28):
        q=np.array([1.24,pitch,.775,.775,.775,.775]);v,tip=cloud(q);th=float(np.max(v['thumb']@axis));roots=[]
        for di,digit in enumerate(digits[1:]):
            def residual(value):
                probe=q.copy();probe[di+2]=value;vv,_=cloud(probe);return float(np.min(vv[digit]@axis)-th-desired_gap)
            grid=np.linspace(.05,1.45,30);intervals=[(lo,hi)for lo,hi in zip(grid[:-1],grid[1:])if residual(lo)*residual(hi)<=0];possible=[brentq(residual,lo,hi,xtol=1e-12)for lo,hi in intervals]
            if possible:q[di+2]=min(possible,key=lambda r:abs(r-.775));roots.append(q[di+2])
        if len(roots)==4:
            vv,tt=cloud(q);tp=vv['thumb'][int(np.argmax(vv['thumb']@axis))];bp={digit:vv[digit][int(np.argmin(vv[digit]@axis))]for digit in digits[1:]};options.append({'six_command_rad':q.tolist(),'thumb_support_point_hand_base_m':tp.tolist(),'four_support_points_hand_base_m':{k:x.tolist()for k,x in bp.items()},'four_directional_gaps_m':{digit:float(np.min(vv[digit]@axis)-th)for digit in digits[1:]},'cost':float(np.linalg.norm(q-np.asarray(anchor['six_command_rad'])))})
    options.sort(key=lambda x:x['cost']);wide.append({'desired_directional_gap_m':desired_gap,'axis_hand_base':axis.tolist(),'feasible':bool(options),'best':options[0]if options else None})
out['thumb_pitch_free_common_gaps']=wide;a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'wide':wide}),flush=True)


