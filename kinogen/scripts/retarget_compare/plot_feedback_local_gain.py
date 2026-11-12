"""Static numerical diagnostic from saved virtual probes; no policy execution."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/data/KinoGen/outputs/sonic_expert_20261028')
def main():
    out=ROOT/'audit/feedback_local_gain_checkpoint50';d=json.loads((out/'report.json').read_text());pairs=d['plus_minus_decomposition']
    fig,axes=plt.subplots(2,2,figsize=(12,9));records=[]
    for phase,axis in zip(d['nominal_phases_s'],axes.flat):
        selected=[p for p in pairs if p['phase_s']==phase and p['probe']=='p1_root_x'];x=np.asarray([p['epsilon']for p in selected])
        odd=np.asarray([p['gated_feedback_rad']['odd_l2']for p in selected]);even=np.asarray([p['gated_feedback_rad']['even_l2']for p in selected]);centered=np.asarray([p['ungated_policy_offset_rad']['odd_l2']for p in selected])
        for y,label,color in [(odd,'Current gated odd','#2478a0'),(even,'Current gated even','#bd7951'),(centered,'Ungated policy odd (centering candidate)','#3a956b')]:axis.loglog(x,y,'o-',label=label,color=color)
        axis.set(title=f'Actual CEM nominal t={phase:g}s; P1 root X probe',xlabel='Virtual state perturbation epsilon (m)',ylabel='56-motor response L2 (rad)');axis.grid(alpha=.25,which='both');axis.legend(fontsize=8)
        records.append({'phase_s':phase,'gated_odd_log_slope_last_decade':float(np.log10(odd[-1]/odd[-2])),
            'gated_even_log_slope_last_decade':float(np.log10(even[-1]/even[-2])),
            'ungated_policy_odd_log_slope_last_decade':float(np.log10(centered[-1]/centered[-2])),
            'even_over_odd_at1e3':float(even[-1]/odd[-1])})
    fig.suptitle('CPU virtual probes only: norm gate suppresses directional local response\nNo rollout, no training; finite-precision evidence is not a stability proof',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(out/'feedback_local_gain.png',dpi=160);plt.close(fig)
    x=[p for p in pairs if p['epsilon']==1e-3 and p['probe']not in ['p1_root_vx','p1_left_knee_v']]
    slopes=[p['gated_feedback_rad']['odd_observed_log_slope']for p in x]
    analysis={'source_report_sha256':hashlib.sha256((out/'report.json').read_bytes()).hexdigest(),'rootX_phase_summaries':records,
        'position_probe_last_decade_slope_in1p9_to2p1_count':sum(1.9<s<2.1 for s in slopes),'position_probe_count':len(slopes),
        'nominal_feedback_gates':[p['feedback_gain']for p in d['rows']if p['probe']=='nominal'],
        'counterfactual_candidate_not_implemented':'h(obs(actual q,v, same phase/history))-h(obs(nominal q,v,same phase/history)); remove norm multiplier and retain independently tested safety bounds. This centers nominal output to zero and can retain first-order directional response; stability and task robustness require paired physical rollout.',
        'scope':'All histories fixed to zero, only four phases and six one-dimensional virtual directions. ReLU/clipping/float32 and nonzero off-nominal operating points invalidate a blanket smooth expansion; no closed-loop stability proof.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'mathematical_scope_and_summary.json').write_text(json.dumps(analysis,indent=2)+'\n');print(json.dumps(analysis,indent=2))
if __name__=='__main__':main()
