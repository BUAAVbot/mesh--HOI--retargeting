"""Isolated server GPU-physics capability check; not a task success test."""
import argparse,json,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worlds',type=int,default=32);a=p.parse_args()
    import numpy as np,mujoco as mj,mujoco_warp as mw,warp as wp,torch
    wp.init();wp.set_device('cuda:0');torch.set_num_threads(2)
    m=mj.MjModel.from_xml_path(str(a.model.resolve()));d=mj.MjData(m);ref=np.load(a.reference,allow_pickle=False)
    d.qpos[:]=ref['qpos'][0];mj.mj_forward(m,d)
    start=time.monotonic();wm=mw.put_model(m);wd=mw.put_data(m,d,nworld=a.worlds,nconmax=128,njmax=512)
    q=wp.to_torch(wd.qpos);v=wp.to_torch(wd.qvel);ctrl=wp.to_torch(wd.ctrl)
    names=list(map(str,ref['dof_names']));aids=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_ACTUATOR,'motor_'+n)for n in names]
    ctrl[:,aids]=torch.as_tensor(ref['qpos'][0,7:],device='cuda:0',dtype=ctrl.dtype)
    passive=np.all(m.actuator_gear==0,axis=1);ctrl[:,passive]=0.
    mw.step(wm,wd);wp.synchronize()
    compiled=time.monotonic()-start
    with wp.ScopedCapture()as capture:
        for _ in range(4):mw.step(wm,wd)
    start=time.monotonic()
    for _ in range(50):wp.capture_launch(capture.graph)
    wp.synchronize();elapsed=time.monotonic()-start
    result={'status':'gpu_physics_probe','mujoco_version':mj.__version__,'worlds':a.worlds,'compile_s':compiled,'timed_steps_per_world':200,'timed_wall_s':elapsed,'aggregate_steps_per_s':200*a.worlds/elapsed,'finite':bool(torch.isfinite(q).all()and torch.isfinite(v).all()),'root_height_range_m':[float(q[:,2].min()),float(q[:,2].max())],'nq':m.nq,'nv':m.nv,'nu':m.nu,'neq':m.neq}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
