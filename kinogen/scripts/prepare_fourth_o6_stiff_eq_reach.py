"""Prepare a fresh-model continuous prefix; candidate states remain private."""
from pathlib import Path
import json
import numpy as np

R=Path(__file__).resolve().parents[1]
push=R/'outputs/fourth_full_sequence/physics_o6_stiff_eq_raw_push_v1'
source=R/'scripts/run_fourth_o6_preslide_deepgrasp_v4.py'
out=R/'scripts/run_fourth_o6_stiff_eq_reach_v1.py'
if out.exists():raise FileExistsError(out)
text=source.read_text()
old="R/'outputs/fourth_full_sequence/physics_o6_contact_preslide_v3/source_plan.json'"
assert text.count(old)==1
text=text.replace(old,"R/'reports/fourth_o6_stiff_eq_push_plan_v1.json'")
old="R/'outputs/fourth_full_sequence/physics_o6_contact_preslide_v4_extended40/physical_trace.npz'"
assert text.count(old)==1
text=text.replace(old,"R/'outputs/fourth_full_sequence/physics_o6_stiff_eq_raw_push_v1/physical_trace.npz'")
text=text.replace("'scope':'Continuous raw-scene push then deep under-rim grasp; diagnostic object1 only, original thin dishes and original hands.'",
                  "'scope':'Explicit stiff-transmission variant from raw reset; original thin dishes and all original hand collision geometry; diagnostic continuous reach only.'")
compile(text,str(out),'exec')
out.write_text(text)
c=json.loads((R/'reports/fourth_o6_C_fourfinger_bank_frozen_v4.json').read_text())
actual=np.load(push/'physical_trace.npz')['qpos'][-1]
q=np.asarray(c['best']['qpos']);q[:7]=actual[:7];q[59:]=actual[59:]
c['best']['qpos']=q.tolist()
c['model']=str(push/'effective_model.mjb')
c['scope']='Private robot/reference seed mapped to the new stiff-equality raw-push scene. No actual state is initialized at this planned grasp.'
(R/'reports/fourth_o6_stiff_eq_reach_candidate_v1.json').write_text(json.dumps(c,indent=2)+'\n')
print(str(out))
