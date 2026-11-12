"""Independent thin-plate measured reaction compensation at1ms integration."""
from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'scripts/source_microstep_benchmark_v7.py').read_text()
s=s.replace('from fourth_single_plate_benchmark_groundaware import prepare',
    'from fourth_single_plate_benchmark_microstep_v1 import prepare\nfrom fourth_o6_microstep_clock_v1 import CONTROL_DT,step_packet,packet_mimic_peak,write_micro_trace,numerical_report')
s=s.replace('int(np.ceil(a.seconds/m.opt.timestep))','int(np.ceil(a.seconds/CONTROL_DT))').replace('mj.mj_step(m,d)','step_packet(m,d)')
s=s.replace('maxmimic=max(maxmimic,float(mimic))','mimic=max(float(mimic),packet_mimic_peak(m,d));maxmimic=max(maxmimic,float(mimic))')
start=s.index("np.savez_compressed(a.output/'physical_trace.npz'")
end=s.index("np.savez_compressed(a.output/'physical_contacts.npz'",start)
s=s[:start]+"write_micro_trace(a.output/'physical_trace.npz',m,d)\n"+s[end:]
s=s.replace("step=np.asarray([c[0]for c in contacts])","step=np.asarray([(c[0]+1)*5-1 for c in contacts])")
metadata="numerics=numerical_report(m,d);numerics['source_sonic_nominal_actuator_delay_s']=numerics.pop('nominal_actuator_delay_s');numerics['source_sonic_policy_dt_s']=numerics.pop('policy_dt_s');numerics['explicit_body_control_delay_s']=0.;numerics['learned_policy_used']=False\n"
metadata+="report.update(numerical_integration=numerics,contact_recording='Packet end contact samples mapped to exact microstep indices; every1ms q/v/ctrl/torque recorded',body_control_detail='Ground-aware PD with actual hand/object reaction compensation; source SONIC actor and delay queue unused')\n"
s=s.replace("(a.output/'report.json').write_text",metadata+"(a.output/'report.json').write_text")
s=s.replace("'fourth_contact_wrench_balanced_c9.py']:","'fourth_contact_wrench_balanced_c9.py','fourth_o6_microstep_clock_v1.py','fourth_single_plate_benchmark_microstep_v1.py','fourth_ground_reaction_pd_controller.py','fourth_o6_load_hysteresis_cached_c10.py']:")
out=root/'scripts/run_fourth_single_plate_microstep_v7_reaction.py'
if out.exists():raise FileExistsError(out)
compile(s,str(out),'exec');out.write_text(s)
print(out)
