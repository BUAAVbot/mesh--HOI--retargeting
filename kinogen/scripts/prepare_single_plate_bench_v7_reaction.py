"""Add measured contact-reaction feedforward; keep v5b motion and all gains."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
s = (ROOT/'scripts/run_fourth_single_plate_benchmark_v5b.py').read_text()
def once(old,new):
    global s
    assert s.count(old)==1, old
    s=s.replace(old,new)
once('from fourth_contact_wrench_balanced_c9 import contact_columns,wrench_lp', 'from fourth_contact_wrench_balanced_c9 import contact_columns,wrench_lp\nfrom fourth_ground_reaction_pd_loaded_v7 import GroundReactionPD')
once("m=g['m'];d=g['d'];sonic=g['sonic'];t0=float(d.time);qerr=verr=None", "m=g['m'];d=g['d'];sonic=g['sonic'];t0=float(d.time);qerr=verr=None\nif a.body_control!='pd_groundaware':raise ValueError('Reaction FF comparison requires ground-aware PD benchmark')\nloaded_ground=GroundReactionPD(m,g['q0'],sonic.aids,sonic.qa,sonic.va);sonic.action=loaded_ground.action;sonic.ground_controller=loaded_ground;reaction=loaded_ground.reaction\nfor v7_name in ['fourth_hand_reaction_ff_v7.py','fourth_ground_reaction_pd_loaded_v7.py']:\n shutil.copy2(Path(__file__).with_name(v7_name),a.output/('source_'+v7_name))")
once("np.clip(kin.qfrc_bias[va]/m.actuator_gainprm[aa,0],-.25,.25)", "np.clip((kin.qfrc_bias[va]-reaction.filtered[va])/m.actuator_gainprm[aa,0],-.25,.25)")
once("ctrl[waids]=whold+kin.qfrc_bias[wva]/m.actuator_gainprm[waids,0];ctrl[laids]=lhold+kin.qfrc_bias[lva]/m.actuator_gainprm[laids,0]", "ctrl[waids]=whold+(kin.qfrc_bias[wva]-reaction.filtered[wva])/m.actuator_gainprm[waids,0];ctrl[laids]=lhold+(kin.qfrc_bias[lva]-reaction.filtered[lva])/m.actuator_gainprm[laids,0]")
once("report.update(finger_motion=", "(a.output/'hand_reaction_feedforward.json').write_text(json.dumps(reaction.report(),indent=2)+'\\n');(a.output/'ground_reaction_loaded_control.json').write_text(json.dumps(loaded_ground.report(),indent=2)+'\\n')\nreport.update(contact_reaction_feedforward='100ms lowpass measured hand/object contact Jacobian reaction; bias-minus-reaction on RH7, waist2 and LH7 through native actuators only; corresponding root6 reaction included in ground-reaction inverse statics; finger force control unchanged',finger_motion=")
target=ROOT/'scripts/run_fourth_single_plate_benchmark_v7_reaction.py'
target.write_text(s)
print(target)
