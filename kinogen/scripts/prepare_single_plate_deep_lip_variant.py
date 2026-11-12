"""Isolated plate1 inward broad lip plus unchanged deep under-support grasp."""
from pathlib import Path
import copy,hashlib,json,math,xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from scipy.optimize import root_scalar,linprog
from fourth_arm_object_feedback import ArmObjectFeedback
from fourth_convex_separation import ConvexSeparationOracle
from fourth_contact_wrench import contact_columns
R=Path(__file__).resolve().parents[1];base=R/'outputs/fourth_full_sequence/assets/vita_six_objects_newft_o6_calibrated_mimic_v3.xml';dest=base.with_name('vita_six_objects_o6_plate1_deep_lip_in10_up4_v1.xml');setup=json.loads((R/'reports/single_plate_explicit_edge_reset_benchmark_open_v4.json').read_text());openold=json.loads((R/'reports/single_plate_open_preshape_v1.json').read_text());xml=ET.parse(base).getroot();bm=mj.MjModel.from_xml_path(str(base));asset=xml.find('asset');body=xml.find("worldbody/body[@name='target_object_01']");wall=body.find("geom[@name='target_object_01_wall_00']");mesh=asset.find("mesh[@name='%s']"%wall.get('mesh'));v=np.fromstring(mesh.get('vertex'),sep=' ').reshape(-1,3);radius=float(np.linalg.norm(v[:,:2],axis=1).max());top=float(v[:,2].max());inner=radius-.010;low=top-.001;high=top+.004;color=body.find("geom[@name='target_object_01_visual_geom']").get('rgba');added=[]
for k in range(20):
 theta=2*math.pi*k/20;vertices=[[r*math.cos(t),r*math.sin(t),z]for z in(low,high)for r in(inner,radius)for t in(theta-math.pi/20,theta+math.pi/20)];name='target_object_01_deep_support_lip_%02d'%k;ET.SubElement(asset,'mesh',name=name+'_mesh',vertex=' '.join('%.12g'%x for x in np.asarray(vertices).ravel()));ET.SubElement(body,'geom',name=name,type='mesh',mesh=name+'_mesh',rgba=color,group='1',density='0',contype='1',conaffinity='1',friction=wall.get('friction'));added.append(name)
if dest.exists():raise FileExistsError(dest)
dest.write_text(ET.tostring(xml,encoding='unicode'));m=mj.MjModel.from_xml_path(str(dest));d=mj.MjData(m);ref=dict(np.load(R/'outputs/fourth_full_sequence/physics_single_plate_groundaware_stand_v1/source_reference.npz'));h=ArmObjectFeedback(m,ref);o=ConvexSeparationOracle(m);preserve=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_range','jnt_axis','dof_armature','dof_damping','eq_data','eq_solref','eq_solimp','actuator_gainprm','actuator_biasprm','actuator_gear','actuator_forcerange'];assert all(np.array_equal(getattr(bm,k),getattr(m,k))for k in preserve);oldxml=ET.parse(base).getroot();unchanged_bodies=[b.get('name')for b in oldxml.find('worldbody').findall('body')if b.get('name')!='target_object_01'];assert all(ET.tostring(oldxml.find("worldbody/body[@name='%s']"%n))==ET.tostring(xml.find("worldbody/body[@name='%s']"%n))for n in unchanged_bodies);bid=m.body('target_object_01').id;manifest={'schema':'single_plate.deep_under_support_inward_lip.v1','scope':'Engineering plate1 broad top lip added for the deep-under-support grasp. Not original reconstructed tableware. Original hand and robot remain unchanged.','base_model':str(base),'base_model_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'model':str(dest),'model_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'added_geoms':added,'changed_object_ids':[1],'original_outer_radius_m':radius,'lip_inner_radius_m':inner,'lip_radial_inward_width_m':.010,'original_top_z_m':top,'lip_low_z_m':low,'lip_top_z_m':high,'top_rise_m':.004,'central_opening_inscribed_radius_m':inner*math.cos(math.pi/20),'original_lowest_surface_unchanged':True,'visual_and_collision_same_added_mesh':True,'retained_plate_mass_kg':float(m.body_mass[bid]),'retained_plate_inertia':m.body_inertia[bid].tolist(),'other_five_objects_unchanged':True,'robot_xml_subtrees_unchanged':True,'preserved_exactly':preserve};dest.with_suffix('.report.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'phase':'model_built','manifest':manifest}),flush=True)
seed=np.asarray(setup['best']['planning_grasp_qpos']).copy();q=seed.copy();digits=['thumb','index','middle','ring','pinky'];names=[m.joint(j).name or''for j in range(m.njnt)];masters=[names.index('rh_'+n)for n in['thumb_cmc_yaw','thumb_cmc_pitch','index_mcp_pitch','middle_mcp_pitch','ring_mcp_pitch','pinky_mcp_pitch']];ya,pa=m.jnt_qposadr[masters[:2]];relations=[(int(m.eq_obj1id[e]),int(m.eq_obj2id[e]),float(m.eq_data[e,1]))for e in range(m.neq)if m.eq_type[e]==mj.mjtEq.mjEQ_JOINT];P=np.zeros((m.nv,6))
for k,j in enumerate(masters):P[m.jnt_dofadr[j],k]=1.
for ch,pr,ratio in relations:
 if pr in masters:P[m.jnt_dofadr[ch],masters.index(pr)]=ratio
hg={s:[g for g in h.hand_geoms[1]if(m.body(int(m.geom_bodyid[g])).name or'').startswith('rh_'+s)]for s in digits};og=h.object_geoms[0];original_margin=m.geom_margin.copy()
def update():
 for ch,pr,ratio in relations:q[m.jnt_qposadr[ch]]=ratio*q[m.jnt_qposadr[pr]]
 d.qpos[:]=q;d.qvel[:]=0.;mj.mj_kinematics(m,d);mj.mj_comPos(m,d)
def bounds(g):
 rr=d.geom_xmat[g].reshape(3,3);cc=d.geom_xpos[g]+rr@m.geom_aabb[g,:3];ee=abs(rr)@m.geom_aabb[g,3:];return cc-ee,cc+ee
def gap(gs,ts):
 bb={g:bounds(g)for g in gs+ts};best=float('inf')
 for g in gs:
  gl,gu=bb[g]
  for t in ts:
   tl,tu=bb[t];lb=float(np.linalg.norm(np.maximum(np.maximum(gl-tu,tl-gu),0)))
   if lb>min(.015,best):value=lb
   else:value=float(o.query(d,g,t)[0])
   best=min(best,value)
 return best
def pitch_for_gap(yaw,target,centre):
 q[ya]=yaw;lo=max(m.jnt_range[masters[1],0]+1e-5,centre-.12);hi=min(m.jnt_range[masters[1],1]-1e-5,centre+.12)
 def f(p):q[pa]=p;update();return gap(hg['thumb'],og)-target
 fl,fh=f(lo),f(hi)
 if fl*fh>0:return None
 fit=root_scalar(f,bracket=(lo,hi),xtol=1e-8,maxiter=20);q[pa]=fit.root;update();return float(fit.root)
rows=[]
for dy in[0.,-.04,.04,-.08,.08]:
 yaw=float(seed[ya]+dy);pitch=pitch_for_gap(yaw,.001,float(seed[pa]))
 if pitch is None:continue
 m.geom_margin[h.hand_geoms[1]]=.0015;mj.mj_forward(m,d);items=contact_columns(m,d,bid,tuple('rh_'+s for s in digits),P,max_gap_m=.003);m.geom_margin[:]=original_margin;A=np.asarray(items['columns']).T;N=np.asarray(items['normal_rows']).T;M=items['motor_torque_matrix'];bias=P.T@d.qfrc_bias;K=A.shape[1];scale=np.array([1,1,1,10,10,10]);gravity=np.r_[m.body_mass[bid]*m.opt.gravity,np.zeros(3)];top_patches=[p for p in items['patches']if p['digit']==0 and p['normal_on_object'][2]<-.8];U=np.r_[np.c_[N,-np.ones(5)],np.c_[M,np.zeros(6)],np.c_[-M,np.zeros(6)],np.c_[-N,np.zeros(5)]];B=np.r_[np.zeros(5),bias+.35,.35-bias,-.1*np.ones(5)];fit=linprog(np.r_[1e-5*N.sum(0),1.],A_ub=U,b_ub=B,A_eq=np.c_[scale[:,None]*A,np.zeros(6)],b_eq=-scale*gravity,bounds=(0,None),method='highs');mj.mj_forward(m,d);gaps={v:gap(hg[v],og)for v in digits};tg=gap(h.hand_geoms[1],h.table_geoms);lp={'feasible':bool(fit.success),'motor_budget_Nm':.35}
 if fit.success:
  weights=fit.x[:-1];lp.update({'normal_per_digit_N':(N@weights).tolist(),'motor_torque_Nm':(bias-M@weights).tolist(),'motor_torque_scaled105_Nm':(bias-M@(1.05*weights)).tolist(),'equilibrium_max_error':float(np.max(abs(A@weights+gravity)))})
 row={'thumb_yaw_rad':yaw,'thumb_pitch_rad':pitch,'top_patches':top_patches,'digit_gaps_m':gaps,'hand_table_gap_m':tg,'lp':lp,'qpos':q.tolist()};rows.append(row);print(json.dumps({k:v for k,v in row.items()if k!='qpos'}),flush=True)
good=[r for r in rows if r['top_patches']and r['lp']['feasible']and min(r['digit_gaps_m'].values())>=.0004 and r['hand_table_gap_m']>=.002]
if not good:
 out={'scope':'No feasible isolated wide-lip deep-support candidate in bounded thumb-only search; no physical success claimed.','model':str(dest),'rows':rows};(R/'reports/single_plate_deep_lip_candidate_v1.json').write_text(json.dumps(out,indent=2)+'\n');raise RuntimeError('No bounded thumb-only candidate; see report')
best=min(good,key=lambda r:(max(abs(np.asarray(r['lp']['motor_torque_scaled105_Nm']))),abs(r['thumb_yaw_rad']-seed[ya])));q=np.asarray(best['qpos']).copy();closed=q.copy();template=copy.deepcopy(setup['best']['template']);template['finger_joint_targets']={n:float(q[m.jnt_qposadr[j]])for j,n in enumerate(names)if n.startswith('rh_')};template['required_contact_digits']=digits;template['digit_force_targets_N']={v:float(x)for v,x in zip(digits,best['lp']['normal_per_digit_N'])};closedtargets=copy.deepcopy(template['finger_joint_targets']);openpitch=pitch_for_gap(float(q[ya]),.0045,float(q[pa]));assert openpitch is not None
for n,v in openold['open_finger_joint_targets'].items():
 if not n.startswith('rh_thumb'):q[m.jnt_qposadr[names.index(n)]]=v
update();openq=q.copy();opentargets={n:float(q[m.jnt_qposadr[j]])for j,n in enumerate(names)if n.startswith('rh_')};opengaps={v:gap(hg[v],og)for v in digits};opentable=gap(h.hand_geoms[1],h.table_geoms);assert min(opengaps.values())>=.004 and opentable>=.002;newsetup=copy.deepcopy(setup);newsetup['model']=str(dest);initial=np.asarray(setup['best']['initial_qpos']).copy()
for j,n in enumerate(names):
 if n in opentargets:initial[m.jnt_qposadr[j]]=opentargets[n]
newsetup['best'].update({'initial_qpos':initial.tolist(),'planning_grasp_qpos':closed.tolist(),'qpos':closed.tolist(),'template':template,'open_finger_joint_targets':opentargets,'open_master_targets':{names[j]:float(openq[m.jnt_qposadr[j]])for j in masters}});newsetup['deep_lip_engineering_variant']={'manifest':str(dest.with_suffix('.report.json')),'model_sha256':manifest['model_sha256'],'object_mesh_scope':'Only plate1 broad hollow top lip; retained original mass/inertia, all other objects and hands unchanged.','robot_body_root_arms_unchanged_from_source_setup':True,'closed_four_fingers_exactly_unchanged':all(closedtargets[n]==setup['best']['template']['finger_joint_targets'][n]for n in closedtargets if not n.startswith('rh_thumb')),'new_closed_thumb_top_patches':best['top_patches'],'closed_static_lp':best['lp'],'open_gaps_m':opengaps,'open_hand_table_gap_m':opentable,'physical_rollout_or_lift_success':False};outpath=R/'reports/single_plate_deep_lip_open_setup_v1.json';newsetup['open_preshape_provenance']['rule']='Original table-external/upward pregrasp arms/root/body/objects retained. Original thumb recalibrated for explicitly engineered plate1 broad top lip; four-bank closed and open targets unchanged. Open during approach; actuator-only close after arrival.';newsetup['open_preshape_provenance']['superseded_by']=newsetup['deep_lip_engineering_variant'];outpath.write_text(json.dumps(newsetup,indent=2)+'\n');candidate={'scope':'Private broad lip + deep under-support planning only; no physical success.','model':str(dest),'best':{'qpos':closed.tolist(),'template':template},'wide_lip_manifest':manifest,'best_static':best,'open_setup':str(outpath),'rows':rows};(R/'reports/single_plate_deep_lip_candidate_v1.json').write_text(json.dumps(candidate,indent=2)+'\n');print(json.dumps({'phase':'PASS','setup':str(outpath),'candidate':str(R/'reports/single_plate_deep_lip_candidate_v1.json'),'static':newsetup['deep_lip_engineering_variant']}),flush=True)
