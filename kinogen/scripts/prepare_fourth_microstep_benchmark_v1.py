"""Declare a microstep/actual-segment/latched-allocation benchmark variant."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
helper=(R/'scripts/fourth_single_plate_benchmark_groundaware.py').read_text()
helper=helper.replace('from fourth_sonic_controller import Controller',
 'from fourth_o6_microstep_clock_v1 import Controller,CONTROL_DT,step_packet,packet_mimic_peak,start_micro_recording,write_micro_trace')
helper=helper.replace('range(round(2./m.opt.timestep))','range(round(2./CONTROL_DT))')
helper=helper.replace('mj.mj_step(m, d)','step_packet(m, d)').replace('*m.opt.timestep','*CONTROL_DT')
helper=helper.replace("state0, spec)\n    sequences", "state0, spec)\n    start_micro_recording(m,d,spec,state0)\n    sequences")
needle="        upright = float(d.xmat[sonic.base].reshape(3, 3)[2, 2])"
helper=helper.replace(needle,"        maxmimic=max(maxmimic,packet_mimic_peak(m,d))\n"+needle)
start=helper.index("    np.savez_compressed(output/'settle_trace.npz'")
end=helper.index("    np.savez_compressed(output/'source_reference.npz'",start)
helper=helper[:start]+"    write_micro_trace(output/'settle_trace.npz',m,d)\n"+helper[end:]
hp=R/'scripts/fourth_single_plate_benchmark_microstep_v1.py';assert not hp.exists();compile(helper,str(hp),'exec');hp.write_text(helper)
s=(R/'scripts/run_fourth_single_plate_benchmark_v3.py').read_text()
s=s.replace('from fourth_o6_load_hysteresis_cached_c10 import','from fourth_o6_actual_segment_control_c11 import')
s=s.replace('from fourth_single_plate_benchmark_groundaware import prepare',
 'from fourth_single_plate_benchmark_microstep_v1 import prepare\nfrom fourth_o6_microstep_clock_v1 import CONTROL_DT,step_packet,packet_mimic_peak,write_micro_trace,numerical_report')
s=s.replace('int(np.ceil(a.seconds/m.opt.timestep))','int(np.ceil(a.seconds/CONTROL_DT))').replace('mj.mj_step(m,d)','step_packet(m,d)')
s=s.replace("verified=False;locked_palm=None;", "verified=False;locked_palm=None;locked_force_targets=None;")
s=s.replace("    targets=desired", "    if locked_force_targets is None:locked_force_targets=desired.copy()\n    targets=locked_force_targets.copy()")
s=s.replace("if now-lost_since>.15:locked_palm=None", "if now-lost_since>.15:locked_palm=None;locked_force_targets=None")
s=s.replace("maxmimic=max(maxmimic,float(mimic))", "mimic=max(float(mimic),packet_mimic_peak(m,d));maxmimic=max(maxmimic,float(mimic))")
start=s.index("np.savez_compressed(a.output/'physical_trace.npz'")
end=s.index("np.savez_compressed(a.output/'physical_contacts.npz'",start)
s=s[:start]+"write_micro_trace(a.output/'physical_trace.npz',m,d)\n"+s[end:]
s=s.replace("step=np.asarray([c[0]for c in contacts])", "step=np.asarray([(c[0]+1)*5-1 for c in contacts])")
s=s.replace("(a.output/'report.json').write_text", "report.update(numerical_integration=numerical_report(m,d),force_allocation_policy='First margin-certified allocation held while loaded topology remains recent; reset only after150ms loss',contact_recording='Packet end samples mapped to their exact microstep indices')\n(a.output/'report.json').write_text")
s=s.replace("'fourth_contact_wrench_balanced_c9.py']:","'fourth_contact_wrench_balanced_c9.py','fourth_o6_microstep_clock_v1.py','fourth_single_plate_benchmark_microstep_v1.py','fourth_ground_reaction_pd_controller.py','fourth_o6_load_hysteresis_cached_c10.py','fourth_o6_actual_segment_control_c11.py']:")
sp=R/'scripts/run_fourth_single_plate_microstep_v1.py';assert not sp.exists();compile(s,str(sp),'exec');sp.write_text(s)
print(hp);print(sp)
