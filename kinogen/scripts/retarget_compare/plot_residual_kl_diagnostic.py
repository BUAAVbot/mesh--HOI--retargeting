"""Plot measured optimizer safeguards; not a physical success metric."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/data/KinoGen/outputs/sonic_expert_20261028')
run=ROOT/'expert_phc_capacity1_kl_positive_pool'
rows=[json.loads(line)for line in(run/'minibatch_kl.jsonl').read_text().splitlines()]
metrics=[json.loads(line)for line in(run/'metrics.jsonl').read_text().splitlines()]
steps=[row for row in rows if row.get('gradient_step')]
x=np.arange(1,len(steps)+1)
kl=np.asarray([row['full_rollout_kl_after_proposed_step']for row in steps])
accepted=np.asarray([row['accepted']for row in steps],bool)
fig,axes=plt.subplots(3,1,figsize=(10,9),layout='constrained')
axes[0].scatter(x[accepted],kl[accepted],s=4,color='#26728c',label='Accepted optimizer step')
axes[0].scatter(x[~accepted],kl[~accepted],s=35,color='#c94040',marker='x',label='Model + Adam rollback')
axes[0].axhline(.01,color='#4b7345',ls='--',label='Desired KL = 0.01')
axes[0].axhline(.02,color='#c94040',ls=':',label='Own rollback threshold = 0.02')
axes[0].set(yscale='log',ylabel='Exact 56-D Normal KL',xlabel='Proposed minibatch optimizer step')
axes[0].legend(fontsize=8,ncol=2)
axes[1].plot([r['iteration']for r in metrics],[r['learning_rate']for r in metrics],color='#724b88')
axes[1].set(yscale='log',ylabel='Adaptive learning rate',xlabel='PPO rollout iteration')
axes[2].plot(x,[r['actor_motor_mean_delta_rms_rad']for r in steps],color='#bb8038',lw=.6)
axes[2].set(yscale='log',ylabel='Motor mean delta RMS [rad]',xlabel='Proposed step: delta relative to its rollout policy')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Frozen-SONIC residual expert: measured update scale',fontsize=15)
fig.text(.01,-.025,
    'Effective initial sigma = 0.028 rad; residual range = 1.0 rad. KL is summed across 56 action dimensions.\n'
    'Optimizer stability evidence only: this plot does not establish physical task success.',fontsize=9)
out=ROOT/'report';out.mkdir(exist_ok=True)
fig.savefig(out/'residual_kl_update_diagnostic.png',dpi=180,bbox_inches='tight')
fig.savefig(out/'residual_kl_update_diagnostic.svg',bbox_inches='tight')
summary={'through_iteration':metrics[-1]['iteration'],'proposed_optimizer_steps':len(steps),
    'rollbacks':int((~accepted).sum()),'accepted_steps':int(accepted.sum()),
    'max_accepted_kl':float(kl[accepted].max()),'first_proposed_step':steps[0],
    'final_learning_rate':metrics[-1]['learning_rate'],'interpretation':'Optimizer diagnostic, not functional physics validation'}
(out/'residual_kl_update_diagnostic.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
