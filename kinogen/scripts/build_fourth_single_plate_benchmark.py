"""Produce a separate controlled-reset runner from the reviewed C9 snapshot."""
from pathlib import Path

folder = Path(__file__).resolve().parent
source = (folder/'bench_reference_C9_snapshot.py').read_text(encoding='utf-8')
def replace(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError('Source snapshot changed: '+old[:75])
    source = source.replace(old, new)

replace('a=p.parse_args()', "p.add_argument('--setup',type=Path,required=True);p.add_argument('--setup-model',type=Path);a=p.parse_args()")
start = source.index('argv=sys.argv.copy();')
end = source.index('oldtimes=sonic.times.copy();')
source = source[:start]+'''from fourth_single_plate_benchmark_setup import prepare
g=prepare(a.setup,a.output,a.setup_model)
m=g['m'];d=g['d'];sonic=g['sonic'];t0=float(d.time);qerr=verr=None
print(json.dumps({'phase':'controlled_reset_settled','time_s':t0}),flush=True)
'''+source[end:]
replace("Path(__file__).name,'run_fourth_o6_preslide_deepgrasp_v4.py','run_fourth_o6_preslide_extended_v4.py'",
        "Path(__file__).name,'fourth_single_plate_benchmark_setup.py'")
replace(" if lift_started is None:\n  phase='tactile_align'if not lp_ok else'load'",
        """ if t<2.:
  phase='descend_outside';out=np.array([.3521665876,.9359373348,-.02508]);out/=np.linalg.norm(out)
  alpha=np.clip(t/2.,0,1);alpha=alpha*alpha*(3-2*alpha)
  goal=pstart+alpha*(want+out*.08-pstart);rotation=wantR
 elif t<6.:
  phase='insert_under';out=np.array([.3521665876,.9359373348,-.02508]);out/=np.linalg.norm(out)
  alpha=np.clip((t-2.)/4.,0,1);alpha=alpha*alpha*(3-2*alpha)
  goal=want+out*.08*(1-alpha);rotation=wantR
 elif lift_started is None:
  phase='tactile_align'if not lp_ok else'load'""")
replace(" if localstep%4==0 or held is None:ctrl=finger.apply(d,ctrl,now,plan);held=ctrl[finger.active_finger_actuator_ids].copy()",
        """ if t<6.:
  plan['active_object_ids_by_hand']=[-1,-1];plan['phase_by_hand']=['idle','idle']
 if localstep%4==0 or held is None:ctrl=finger.apply(d,ctrl,now,plan);held=ctrl[finger.active_finger_actuator_ids].copy()""")
replace(" gaps,_=finger._geometry(finger.hands[1],0)",
        """ if localstep%4==0:
  finger._kin.qpos[:]=d.qpos;finger._kin.qvel[:]=d.qvel;mj.mj_kinematics(m,finger._kin);mj.mj_comPos(m,finger._kin)
  cached_actual_gaps,_=finger._geometry(finger.hands[1],0)
 gaps=cached_actual_gaps""")
replace(" if now-lastlp>=.019999999:", " if t>=6. and now-lastlp>=.019999999:")
replace("'scope':'Continuous original raw scene C4 prefix then five-finger tactile alignment and actual-contact wrench-gated lift; diagnostic single object.'",
        "'scope':'Controlled table-edge reset benchmark: plate1 initial position explicitly shifted; free SONIC robot and original hand geometry; not source raw-scene reproduction.'")
replace("'prefix_q_error':qerr,'prefix_v_error':verr,", "'source_raw_prefix_not_run':True,'setup_report':'settle_report.json',")
target = folder/'run_fourth_single_plate_benchmark.py'
if target.exists():
    raise FileExistsError('Preserve existing runner; choose a new version')
target.write_text(source, encoding='utf-8')
print(target)
