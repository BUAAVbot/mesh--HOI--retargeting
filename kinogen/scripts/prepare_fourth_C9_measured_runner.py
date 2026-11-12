from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'scripts/run_fourth_o6_C8_locked_actual_load.py').read_text()
text=text.replace('from fourth_contact_wrench import contact_columns,wrench_lp',
                  'from fourth_contact_wrench_balanced_c9 import contact_columns,wrench_lp')
text=text.replace("'fourth_o6_load_hysteresis_c7.py']:", "'fourth_o6_load_hysteresis_c7.py','fourth_contact_wrench_balanced_c9.py']:")
text=text.replace("longest_proof=0.;stop=", "longest_proof=0.;measured_support_checks=0;stop=")
point="for localstep in range(int(np.ceil(a.seconds/m.opt.timestep))):"
helper="""table_plane=np.array([.010271048830894367,.02292500590934548,.999684425036206,1.0058665495441206]);table_plane/=np.linalg.norm(table_plane[:3])
object_shapes=[]
for og in arm.object_geoms[0]:
 if m.geom_type[og]==mj.mjtGeom.mjGEOM_MESH:
  mid=int(m.geom_dataid[og]);adr=int(m.mesh_vertadr[mid]);num=int(m.mesh_vertnum[mid]);object_shapes.append((og,m.mesh_vert[adr:adr+num].copy()))
 else:object_shapes.append((og,None))
def lowest_object_clearance():
 values=[];normal=table_plane[:3]
 for og,vertices in object_shapes:
  G=d.geom_xmat[og].reshape(3,3);center=d.geom_xpos[og]
  if vertices is not None:values.append(float(np.min((vertices@G.T+center)@normal+table_plane[3])))
  else:
   projection=abs(float(G[:,2]@normal));size=m.geom_size[og]
   if m.geom_type[og]==mj.mjtGeom.mjGEOM_CYLINDER:values.append(float(center@normal+table_plane[3]-size[1]*projection-size[0]*np.sqrt(max(0,1-projection*projection))))
   else:raise ValueError('Unsupported dynamic object collider in clearance proof')
 return min(values)
"""
assert point in text
text=text.replace(point,helper+point)
old="ready=lp_ok and all_contact and opposition<=-.25 and all(Fdigit[v]>=.9*targets[v]for v in required)and sum(Fdigit.values())>=1.3*m.body_mass[obj]*9.81"
new="""measured_ok=False
 if all_contact and sum(Fdigit.values())>=1.3*m.body_mass[obj]*9.81:
  measured_items=contact_columns(m,d,obj,tuple('rh_'+v for v in digits),P,max_gap_m=0.)
  kin.qpos[:]=d.qpos;kin.qvel[:]=0;mj.mj_forward(m,kin)
  measured_lp=wrench_lp(measured_items['columns'],measured_items['normal_rows'],np.r_[m.body_mass[obj]*m.opt.gravity,np.zeros(3)],normal_cap_N=.95*np.asarray([Fdigit[v]for v in digits]),motor_torque_matrix=measured_items['motor_torque_matrix'],motor_gravity_bias_Nm=P.T@kin.qfrc_bias,motor_cap_Nm=.4)
  measured_ok=bool(measured_lp['feasible']);measured_support_checks+=int(measured_ok)
 ready=lp_ok and all_contact and opposition<=-.25 and measured_ok"""
assert old in text
text=text.replace(old,new)
text=text.replace("proof=ready and rise>=.03 and tableforce<.15 and upforce>=.7*m.body_mass[obj]*9.81", "clearance=lowest_object_clearance();proof=ready and clearance>=.03 and tableforce<.15 and upforce>=.7*m.body_mass[obj]*9.81")
text=text.replace("*Fdigit.values(),lp_ok]);", "*Fdigit.values(),lp_ok,measured_ok,clearance]);")
text=text.replace("'pinky_N','actual_contact_wrench_feasible']))", "'pinky_N','actual_contact_wrench_feasible','measured_normal_budget_wrench_feasible','object_min_clearance_from_table_plane_m']))")
text=text.replace("'actual_LP_feasible_checks':sum", "'measured_normal_budget_feasible_steps':measured_support_checks,'prelift_readiness':'Actual current contact LP feasible with each digit normal capped at95pct of measured normal force; raw required contact and1.3mg total retained','proof_height':'Lowest actual object collision geometry30mm above fitted tabletop plane, not object origin','actual_LP_feasible_checks':sum")
text=text.replace("'palm_error_m':palmerror}),", "'palm_error_m':palmerror,'measured_budget_LP':measured_ok,'min_clearance_m':clearance}),")
target=ROOT/'scripts/run_fourth_o6_C9_measured_support.py'
target.write_text(text)
print(target)
