"""Bind the C7 actual-contact loader to the fresh stiff-equality prefix."""
from pathlib import Path
R=Path(__file__).resolve().parents[1]
src=R/'scripts/run_fourth_o6_C7_latched_actual_load.py'
dst=R/'scripts/run_fourth_o6_stiff_eq_C7_v1.py'
if dst.exists():raise FileExistsError(dst)
s=src.read_text()
s=s.replace('run_fourth_o6_preslide_deepgrasp_v4.py','run_fourth_o6_stiff_eq_reach_v1.py')
s=s.replace("R/'reports/fourth_o6_C_fourfinger_bank_frozen_v4.json'","R/'reports/fourth_o6_stiff_eq_reach_candidate_v1.json'")
s=s.replace("R/'outputs/fourth_full_sequence/physics_o6_C_continuous_undergrasp_v4/physical_trace.npz'",
            "R/'outputs/fourth_full_sequence/physics_o6_stiff_eq_reach_v1/physical_trace.npz'")
s=s.replace('Continuous original raw scene C4 prefix then five-finger tactile alignment and actual-contact wrench-gated lift; diagnostic single object.',
            'Explicit stiff-transmission variant from original raw reset, unchanged hands and thin dishes; actual-contact LP-gated lift.')
compile(s,str(dst),'exec');dst.write_text(s)
print(dst)
