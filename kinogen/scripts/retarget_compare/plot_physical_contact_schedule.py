"""Plot the diagnosed early sitting labels against actual historical seat load."""
from __future__ import annotations
import json
from pathlib import Path


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('/data/KinoGen/outputs/sonic_expert_20261028')
    data=np.load(root/'references/phc_time_scale2_refined_contacts.npz',allow_pickle=False)
    report=json.loads((root/'physical_contact_schedule_diagnostic.json').read_text())
    time=data['source_time_s'];dt=float(np.median(np.diff(time)))
    coarse=data['coarse_seat_support_expected'];physical=data['seat_support_expected']
    force=data['physical_refined_seat_force_mean_N']/np.asarray(report['robot_weight_N'])[None]
    fig,axes=plt.subplots(2,2,figsize=(13,6.5),sharex=True,gridspec_kw={'width_ratios':[1.25,1]})
    def segments(mask):
        edges=np.diff(np.r_[False,mask,False].astype(int))
        return [(float(time[s]),float(time[min(e,len(time)-1)]-time[s]+(dt if e==len(time)else 0.)))
                for s,e in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]
    for actor in range(2):
        ax=axes[actor,0]
        ax.broken_barh(segments(coarse[:,actor]),(.65,.25),facecolors='#df9851')
        ax.broken_barh(segments(physical[:,actor]),(.15,.25),facecolors='#418a78')
        ax.set_yticks([.275,.775],['Actual refined','Coarse kinematic'])
        ax.set_ylim(0,1.1);ax.set_title(f'Person {actor+1}: expected sitting intervals')
        ax.grid(axis='x',alpha=.22)
        load=axes[actor,1]
        load.plot(time,force[:,actor],color='#315e83',lw=1.5,label='Actual upward seat load / body weight')
        load.axhline(.1,color='#a74640',ls='--',lw=1,label='10% body-weight threshold')
        load.set_ylabel('Body-weight fraction');load.set_ylim(0,max(1.2,float(force[:,actor].max())*1.1))
        load.set_title(f'Person {actor+1}: historical physical seat load');load.grid(alpha=.22)
        if actor==0:load.legend(fontsize=8,loc='upper left')
    for ax in axes[-1]:ax.set_xlabel('Original source clock (s); execution time is 2x in this controlled variant')
    for ax in axes.flat:ax.set_xlim(0,float(time[-1]));ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Refining the contact schedule from actual physics\nTrajectory and timing arrays unchanged; historical validated data, not new expert success',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.9])
    output=root/'report';output.mkdir(exist_ok=True)
    fig.savefig(output/'physical_contact_schedule.png',dpi=180)
    fig.savefig(output/'physical_contact_schedule.svg')
    print(output/'physical_contact_schedule.png')


if __name__=='__main__':main()
