"""Visual evidence for the declared source-visible table completion hypothesis."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parents[1];A=R/'outputs/fourth_full_sequence/assets'
E=R/'outputs/fourth_full_sequence/table_surface_audit'
r=json.loads((A/'vita_six_objects_newft_source_table_v1.report.json').read_text())
e=np.load(E/'table_surface_evidence.npz',allow_pickle=False)
completion=np.load(A/'vita_six_objects_newft_source_table_v1.evidence.npz',allow_pickle=False)
ob=np.load(R/'data/fourth_sim_handoff/objects/poses_sim.npz',allow_pickle=False)
human=np.load(R/'data/fourth_sim_handoff/human/root_poses_sim.npz',allow_pickle=False)['position_m'][:838]
ref=np.load(R/'outputs/fourth_full_sequence/reference_v6.npz',allow_pickle=False)
candidate=ref['contact_candidate_observed'].any(axis=1)
fig,axes=plt.subplots(1,2,figsize=(16,8),layout='constrained')
polygon=np.asarray(r['polygon_xy']);patch=e['patch_vertices'];visible=e['hull_xy']
oldlow=np.asarray([-1.0066061807,1.3451216030]);oldhigh=np.asarray([-.1370155323,2.2665904951])
box=np.array([oldlow,[oldhigh[0],oldlow[1]],oldhigh,[oldlow[0],oldhigh[1]],oldlow])
for ax in axes:
    ax.scatter(patch[::2,0],patch[::2,1],s=1.5,c='#3690c0',alpha=.6,label='measured connected tabletop points')
    ax.fill(polygon[:,0],polygon[:,1],color='#b8865b',alpha=.13)
    ax.plot(*np.vstack([polygon,polygon[0]]).T,color='#8c510a',lw=2,label='NEW convex completion hypothesis')
    ax.plot(*np.vstack([visible,visible[0]]).T,color='#0570b0',lw=1,label='visible tabletop hull (not a physical edge)')
    ax.plot(*box.T,'--',color='#bd0026',lw=1.7,label='OLD object-swept box')
    for oi,ring in enumerate(completion['initial_support_ring_points']):
        ax.plot(*np.vstack([ring[:,:2],ring[0,:2]]).T,lw=.9,color='#555555',alpha=.8,label='initial base support + 1.5cm prior'if oi==0 else None)
        ax.scatter(ob['position_m'][0,oi,0],ob['position_m'][0,oi,1],marker='x',c='black',s=35)
        ax.text(ob['position_m'][0,oi,0],ob['position_m'][0,oi,1],str(oi+1))
        held=np.asarray(ob['valid'][:,oi],bool)&candidate[:,oi]
        if np.any(held):ax.scatter(ob['position_m'][held,oi,0],ob['position_m'][held,oi,1],s=2,c='#88419d',alpha=.25,label='near-hand source paths EXCLUDED from fitting'if oi==0 else None)
    ax.plot(human[:,0],human[:,1],color='#222222',lw=1,alpha=.7,label='raw observed human pelvis path')
    ax.scatter([.08],[1.8],marker='*',s=120,c='red',label='current stationary robot root')
    ax.set(xlabel='world X (m)',ylabel='world Y (m)');ax.set_aspect('equal');ax.grid(alpha=.2)
axes[0].set(xlim=(-1.2,.45),ylim=(1.2,2.4),title='Near-table evidence and old/new support proxies')
axes[1].set(xlim=(-1.7,1.65),ylim=(1.,2.9),title='Full observed human path and tabletop context')
axes[0].legend(fontsize=7,loc='upper left')
fig.suptitle('Source-visible minimal convex support completion\nTrue occluded table edge is NOT uniquely measured; old/new proxies are explicit hypotheses.',fontsize=12)
out=R/'previews/fourth_table_completion_comparison.png';fig.savefig(out,dpi=170);plt.close(fig)
print(str(out))
