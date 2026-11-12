"""Check whether a bounded learned residual can represent the declared poses."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import mujoco as mj,numpy as np,torch
    from rgb145_ppo_env import Reference
    from rgb145_batch_sonic import BatchSonic
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));r=Reference(m,a.reference,'cuda:0');n=len(r.q);b=BatchSonic(m,r,n,'cuda:0');d=mj.MjData(m);rot=[];omega=[]
    for q,v in zip(r.q.cpu().numpy(),r.dq.cpu().numpy()):
        d.qpos[:]=q;d.qvel[:]=v;mj.mj_forward(m,d);rot.append(d.xmat[b.c.base].reshape(3,3).copy());omega.append(d.cvel[b.c.base,:3].copy())
    target=b.targets(r.q,r.dq,r.times,torch.tensor(np.array(rot),device='cuda:0',dtype=torch.float32),torch.tensor(np.array(omega),device='cuda:0',dtype=torch.float32))
    reference=r.q[:,b.qa];gap=(reference-target).cpu().numpy();legs=np.array(['hip'in n or'knee'in n or'ankle'in n for n in b.c.names]);i,j=np.unravel_index(np.argmax(abs(gap[:,legs])),gap[:,legs].shape);col=np.flatnonzero(legs)[j]
    rows=[]
    for scale in (.45,.8,1.2,1.6):rows.append({'residual_limit_rad':scale,'leg_samples_outside_range_fraction':float(np.mean(abs(gap[:,legs])>scale)),'any_leg_outside_frames':int(np.sum(np.any(abs(gap[:,legs])>scale,axis=1)))})
    out={'status':'representability_diagnostic_not_policy_success','all350_poses':True,'max_leg_nominal_gap_rad':float(abs(gap[:,legs]).max()),'worst_source_frame':int(i),'worst_joint':b.c.names[col],'source_time_s':float(r.source_times[i]),'nominal_joint_rad':float(reference[i,col]),'base_joint_target_rad':float(target[i,col]),'bounds':rows}
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)


if __name__=='__main__':main()
