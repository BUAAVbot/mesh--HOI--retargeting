from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
text=(ROOT/'scripts/run_fourth_single_plate_benchmark_v4.py').read_text()
text=text.replace("required=list(template.get('required_contact_digits',digits))", "required=['thumb','index','middle','ring']")
old="targets=desired;allocation_latched=True;allocation_since=now;load_started=now\n     allocations.append({'time_s':now,'normal_targets_N':targets.copy(),'unscaled_planning_motor_budget_Nm':.35,'scaled_motor_max_Nm':lp['scaled_load_max_motor_Nm']})"
new="""active_load_digits=[v for v,x in zip(digits,lp['normal_per_digit_N'])if float(x)>.1]
     if 'thumb'in active_load_digits and sum(v!='thumb'for v in active_load_digits)>=2:
      targets=desired;required=active_load_digits;allocation_latched=True;allocation_since=now;load_started=now
      allocations.append({'time_s':now,'normal_targets_N':targets.copy(),'required_load_digits':required.copy(),'unscaled_planning_motor_budget_Nm':.35,'scaled_motor_max_Nm':lp['scaled_load_max_motor_Nm']})"""
assert old in text
text=text.replace(old,new)
text=text.replace("report.update(load_allocation_policy=", "report.update(required_load_digit_policy='At each frozen actual LP allocation, normal>0.1N digits, requiring thumb plus at least2 bank fingers; all5 fingers remain controlled as a bank',final_required_load_digits=required,load_allocation_policy=")
target=ROOT/'scripts/run_fourth_single_plate_benchmark_v4b.py'
target.write_text(text)
print(target)
