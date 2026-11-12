from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'scripts/run_fourth_o6_C9_measured_support.py').read_text()
text=text.replace('from fourth_o6_load_hysteresis_c7 import O6CoupledReferenceContactFeedback',
                  'from fourth_o6_load_hysteresis_cached_c10 import O6CoupledReferenceContactFeedback')
text=text.replace("'fourth_contact_wrench_balanced_c9.py']:", "'fourth_contact_wrench_balanced_c9.py','fourth_o6_load_hysteresis_cached_c10.py']:")
text=text.replace(" gaps,_=finger._geometry(finger.hands[1],0)", " if localstep%4==0 or localstep==0:\n  finger._kin.qpos[:]=d.qpos;finger._kin.qvel[:]=d.qvel;mj.mj_forward(m,finger._kin);gaps,_=finger._geometry(finger.hands[1],0)")
text=text.replace("'load_contact_grace_s':.1", "'load_contact_grace_s':.1,'geometry_query_hz':50,'SAT_cache':'Exact private qpos bytes and hand/object key; no model/state mutation'")
target=ROOT/'scripts/run_fourth_o6_C10_cached_support.py'
target.write_text(text)
print(target)
