"""Rebuild O6 from frozen wear_v2 in a fresh directory and record all hashes."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,xml.etree.ElementTree as ET

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rebuild-dir',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    if a.rebuild_dir.exists()or a.receipt.exists():raise FileExistsError('Rebuild directory and receipt must be new')
    base=Path('outputs/rgb_near_pipeline/assets');wear=base/'vita_bench_two52_wear_v2.xml'
    urdf=Path('data/vita_demo_20261026/robot/vita02a/urdf/vita02a_linkerhand_meshcoll_relative.urdf');hr=Path('data/linkerhand_o6_06')
    commands=[];logs=[]
    def run(script,args):
        command=[sys.executable,'scripts/'+script,*[str(x)for x in args]];commands.append(command)
        r=subprocess.run(command,check=True,text=True,capture_output=True);logs.append({'command':command,'stdout':r.stdout,'stderr':r.stderr})
    newhand=a.rebuild_dir/'vita_bench_two52_hand_v3.xml'
    run('rgb_near_hand_collision_variant.py',['--source',wear,'--urdf',urdf,'--output',newhand])
    for suffix,selfopt in [('',False),('_self',True)]:
        scene=a.rebuild_dir/('vita_bench_two52_O6_v4'+suffix+'.xml');proxy=a.rebuild_dir/('vita_single_proxy_O6_v4'+suffix+'.xml')
        args=['--source',newhand,'--hand-root',hr,'--output',scene]
        if selfopt:args.append('--hand-self-collision')
        run('rgb_near_o6_hand_variant.py',args)
        run('rgb_near_extract_robot_proxy.py',['--source',scene,'--output',proxy])
    files={};missing=[]
    def add(path,role):
        path=Path(path).resolve();key=str(path)
        if not path.is_file():missing.append({'path':key,'role':role});return
        if key not in files:files[key]={'path':key,'bytes':path.stat().st_size,'sha256':sha(path),'roles':[]}
        if role not in files[key]['roles']:files[key]['roles'].append(role)
    scripts=['rgb_near_o6_build_receipt.py','rgb_near_hand_collision_variant.py','rgb_near_o6_hand_variant.py','rgb_near_extract_robot_proxy.py','rgb_near_hand_collision_audit.py','rgb_near_o6_interface_audit.py']
    for name in scripts:add(Path('scripts')/name,'script')
    add(wear,'frozen_input_scene');add(urdf,'original_perlink_URDF')
    for path in hr.rglob('*'):
        if path.is_file():add(path,'06_extracted_input')
    # Actual original per-link URDF meshes are required for the first stage.
    for mesh in ET.parse(urdf).findall('.//mesh'):
        add(urdf.parent/mesh.get('filename'),'original_perlink_URDF_mesh')
    def model_dependencies(path):
        root=ET.parse(path).getroot();compiler=root.find('compiler')
        for node in root.findall('asset/*'):
            if not node.get('file'):continue
            f=Path(node.get('file'))
            if not f.is_absolute():
                folder=compiler.get('meshdir','')if node.tag=='mesh'else compiler.get('texturedir','')
                f=path.parent/folder/f
            add(f,'compiled_model_'+node.tag+'_dependency')
    model_dependencies(wear)
    comparisons=[]
    for path in sorted(a.rebuild_dir.iterdir()):
        if path.is_file():add(path,'rebuilt_output')
        if path.suffix=='.xml':
            model_dependencies(path);frozen=base/path.name;add(frozen,'frozen_output_comparison')
            comparisons.append({'file':path.name,'rebuilt_sha256':sha(path),'frozen_sha256':sha(frozen),'identical':sha(path)==sha(frozen)})
    for suffix in ['.xml','.mjb','.json']:
        for stem in ['vita_bench_two52_hand_v3','vita_bench_two52_O6_v4','vita_bench_two52_O6_v4_self','vita_single_proxy_O6_v4','vita_single_proxy_O6_v4_self']:
            add(base/(stem+suffix),'frozen_output')
    evidence=Path('outputs/rgb_near_pipeline/O6_hand_collision_audit')
    for path in evidence.iterdir():
        if path.is_file():add(path,'static_or_ABI_audit_evidence')
    add(Path('outputs/rgb_near_pipeline/reference_v11_fixed_grip.npz'),'audit_reference_not_final_dynamic_acceptance')
    if missing:raise FileNotFoundError(json.dumps(missing))
    if not all(r['identical']for r in comparisons):raise ValueError('Rebuild XML differs from frozen outputs')
    import mujoco as mj
    report={'schema':'rgb_near.O6_build_receipt.v1','working_directory':str(Path.cwd()),'python_executable':sys.executable,'mujoco_version':mj.__version__,
            'source_zip':{'local_path':'D:/workspace/SyntheticRobotDemonstrations/06.zip','sha256':'9c5ec039afc3e510ec101370b269f8e6ebac61e8402f0e326c032baa5fb911c1','extracted_payload_files':28,'archive_hash_verified_by_parent':True},
            'rebuild_directory':str(a.rebuild_dir.resolve()),'commands_executed':commands,'command_results':logs,'dependencies_checked':True,'missing_dependencies':missing,
            'xml_reproducibility_comparisons':comparisons,'all_five_rebuilt_xml_identical_to_frozen':True,'files':list(files.values()),
            'scope':'Asset builds, compiler/static collision snapshots and interface forward checks only. Full398-frame contact dynamics remain subject to separate acceptance.'}
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'receipt':str(a.receipt),'receipt_sha256':sha(a.receipt),'dependency_and_output_files_hashed':len(files),'missing_dependencies':0,'all_five_xml_identical':True}))
if __name__=='__main__':main()
