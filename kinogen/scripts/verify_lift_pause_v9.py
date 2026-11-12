"""Pure mathematical fixture: no catch-up after a pause, bounded resume speed."""
from pathlib import Path
import json
from fourth_lift_clock_v9 import IncrementalLift

clock=IncrementalLift();clock.start(0.)
for i in range(1,201):clock.advance(i*.005,True)
assert abs(clock.height-.012)<1e-12
held=clock.height
for i in range(201,2201):assert clock.advance(i*.005,False)==held
resumed=clock.advance(11.005,True)
assert abs(resumed-held-.012*.005)<1e-12
for i in range(2202,20001):clock.advance(i*.005,True)
assert clock.height==.08
report={'success':True,'one_second_lift_m':held,'ten_second_pause_drift_m':0.,
        'first_resume_increment_m':resumed-held,'maximum_m':clock.height,
        'resume_speed_m_s':.012,'paused_time_not_accumulated':True}
path=Path(__file__).resolve().parents[1]/'reports/lift_pause_v9_fixture.json'
path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
