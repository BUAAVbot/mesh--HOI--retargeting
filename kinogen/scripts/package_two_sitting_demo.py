"""Package the verified functional demo, its data and explicit limitations."""
from pathlib import Path
import argparse,hashlib,json,zipfile,importlib.metadata


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/data/KinoGen'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=a.root;task=r/'outputs/two_sitting_pipeline'
    if a.output.exists():raise FileExistsError(a.output)
    chosen=task/'physics_selected450_candidate156';delivery=task/'delivery_final_candidate156'
    report=json.loads((chosen/'report.json').read_text())
    if not report['functional_task_success']or not report['candidate_policy_state_consistency']['passed']:
        raise ValueError('Functional demo and complete-state consistency must be verified')
    paths={
        'videos/two_vita_functional.mp4':r/'previews/two_sitting_final_functional.mp4',
        'videos/contactsheet.jpg':r/'previews/two_sitting_final_functional_contactsheet.jpg',
        'reports/final_status.json':task/'final_20261027/final_status.json',
        'reports/FEISHU_FINAL.txt':task/'final_20261027/FEISHU_FINAL.txt',
        'physics/report.json':chosen/'report.json','physics/trace.npz':chosen/'trace.npz',
        'physics/initial_state.npz':chosen/'initial_state.npz',
        'physics/selected_candidate.npz':task/'search_native450_settle_v2/candidates/candidate_0156.npz',
        'reference/reference.npz':task/'reference_v3_deploy_slow.npz',
        'policy/residual_checkpoint_450.pt':task/'training_native_v1/checkpoint_000450.pt',
        'policy/training_manifest.json':task/'training_native_v1/manifest.json',
        'policy/training_receipt.json':task/'training_native_v1/training_receipt.json',
        'provenance/source_archive_audit.json':r/'reports/two_sitting_source_audit.json',
        'provenance/search_summary.json':task/'search_native450_settle_v2/summary.json',
        'assets/support_geometry.json':task/'assets_deploy_v3/support_geometry.json',
        'assets/assets_report.json':task/'assets_deploy_v3/assets_report.json'}
    for folder,prefix in ((delivery,'hoi'),(r/'previews/two_sitting_final_ego','ego')):
        for f in folder.iterdir():
            if f.is_file():paths[prefix+'/'+f.name]=f
    for name in ('two_sitting_replay_candidate.py','two_sitting_eval_ppo.py','two_sitting_physics.py',
                 'two_sitting_ppo_env.py','two_sitting_native_ppo_env.py','two_sitting_batch_sonic.py',
                 'two_sitting_search_native.py','two_sitting_train_ppo.py','two_sitting_assets.py',
                 'two_sitting_retarget.py','two_sitting_deliver.py','two_sitting_final_audit.py',
                 'rgb_near_physics.py','fourth_sonic_controller.py'):
        paths['code/'+name]=r/'scripts'/name
    if any(not f.exists()for f in paths.values()):raise FileNotFoundError([str(f)for f in paths.values()if not f.exists()])
    model=chosen/'model.mjb'
    manifest={'scope':'verified functional physical demo; strict source tracking failed',
        'functional_task_pass':True,'source_tracking_pass':False,'candidate_id':156,
        'selected_checkpoint_iteration':450,'sampling':'5 confirmed functional candidates selected from this batch of256 attempts; not a robustness/generalization estimate',
        'frozen_model_external':{'path':str(model),'sha256':sha(model),'bytes':model.stat().st_size,'mujoco_version':'3.5.0'},
        'files':{n:{'bytes':f.stat().st_size,'sha256':sha(f)}for n,f in paths.items()}}
    readme='''双 Vita 起身、交叉换位、落座：功能性物理演示

实际 MuJoCo 动力学执行9.56秒/1912个5毫秒步。两个浮动根机器人，未使用根状态强写、外力托举或mocap。
两机末端凳面承重约418/537N，独立控制重放和候选完整状态对照误差均为0。
原视频全程根轨迹最大误差约0.51/0.48米，超过0.25米精度门槛；source_tracking_pass保持false。
本批256个固定权重采样中筛得5条并完成独立功能验收；其余251条未全部做完整5毫秒独立验收。这不是泛化或可靠成功率。

videos/ 为实际仿真，ego/ 为未标定的合成头部相机，hoi/ 含运动、接触图、接触力、关节力矩及物性元数据。
physics/ 为可核对控制与完整状态，policy/ 包含选用450次迭代检查点。training_receipt描述训练最终到750，演示实际用450。
场景碰撞体和材料参数采用重建拟合与工程假设，力/力矩属于仿真结果。

精确复现：冻结模型体积较大，保留在manifest.json的frozen_model_external.path；本演示数据包未重复收录该模型。
在既有服务器yz-ai-4090-01.bce上，使用/root/vita02a-mjlab/.venv310/bin/python（MuJoCo3.5.0）：
python code/two_sitting_replay_candidate.py --model <冻结model.mjb> --candidate physics/selected_candidate.npz --reference reference/reference.npz --output <新的回放目录>
运行前将python替换为上述既有环境；冻结MJB需要匹配MuJoCo版本。单纯控制重放无需再次运行SONIC策略。
重新生成随机策略回合需原工程中的原始SONIC权重及近源依赖；本包的残差检查点不替代基础权重。
'''
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.output,'x',zipfile.ZIP_DEFLATED,compresslevel=6)as z:
        for name,f in paths.items():z.write(f,name)
        z.writestr('README.zh-CN.txt',readme)
        z.writestr('manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    with zipfile.ZipFile(a.output)as z:
        if z.testzip():raise ValueError('ZIP CRC failure')
        for name,item in manifest['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=item['sha256']:raise ValueError(name)
    result={'archive':str(a.output),'bytes':a.output.stat().st_size,'sha256':sha(a.output),
            'verified_payload_files':len(paths),'frozen_model_included':False,
            'functional_task_pass':True,'source_tracking_pass':False}
    a.output.with_suffix('.verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
