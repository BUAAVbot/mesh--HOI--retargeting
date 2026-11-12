from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'scripts/run_fourth_o6_C10_cached_support.py').read_text()
text=text.replace('from fourth_o6_load_hysteresis_cached_c10 import O6CoupledReferenceContactFeedback',
                  'from fourth_o6_actual_segment_control_c11 import O6CoupledReferenceContactFeedback')
text=text.replace("'fourth_o6_load_hysteresis_cached_c10.py']:", "'fourth_o6_load_hysteresis_cached_c10.py','fourth_o6_actual_segment_control_c11.py']:")
text=text.replace("'geometry_query_hz':50", "'geometry_query_hz':50,'physical_hand_collision_changed':False,'control_geometry':'Original all finger segments; actual contact carrying link point/normal Jacobian'")
target=ROOT/'scripts/run_fourth_o6_C11_actual_segment.py'
target.write_text(text)
print(target)
