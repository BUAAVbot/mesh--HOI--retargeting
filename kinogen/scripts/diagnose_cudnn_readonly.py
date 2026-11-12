"""Run bounded, read-only CUDA library diagnostics through the existing SSH alias."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REMOTE = r'''
import glob, importlib.metadata as md, json, os, subprocess, sys
child = r"""
import json, os, sys, traceback
result = {'mode': sys.argv[1], 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'ld_library_path': os.environ.get('LD_LIBRARY_PATH')}
try:
    import torch
    if sys.argv[1].endswith('_utils'):
        sys.path.insert(0,'/data/KinoGen/dependencies/foundationpose')
        sys.path.insert(0,'/data/KinoGen/.build/foundationpose-mycpp')
        import Utils
        result['utils_imported']=True
        result['warp_available']=Utils.wp is not None
    result.update(torch=torch.__version__, cuda=torch.version.cuda, torch_file=torch.__file__, cudnn_version=torch.backends.cudnn.version())
    torch.cuda.set_device(0)
    result['gpu'] = torch.cuda.get_device_name(0)
    mode = sys.argv[1]
    if mode == 'matmul':
        x = torch.randn(32,32,device='cuda'); y=x@x
    else:
        torch.backends.cudnn.enabled = mode.startswith('conv_cudnn')
        x=torch.randn(1,3,16,16,device='cuda'); w=torch.randn(4,3,3,3,device='cuda'); y=torch.nn.functional.conv2d(x,w)
    torch.cuda.synchronize()
    result.update(ok=True, result_shape=list(y.shape), result_finite=bool(torch.isfinite(y).all()))
except Exception as e:
    result.update(ok=False, error=repr(e), traceback=traceback.format_exc())
result['loaded_cuda_libraries'] = sorted(set(line.split()[-1] for line in open('/proc/self/maps') if any(name in line for name in ('libcudnn','libcublas','libcuda.','libcudart','libnvrtc'))))
print(json.dumps(result))
"""
result = {'diagnostic': 'read_only_tiny_cuda_tests', 'python': sys.executable, 'packages': [], 'tests': []}
for dist in md.distributions():
    name=dist.metadata.get('Name','')
    if name.lower().startswith('nvidia-cudnn') or name.lower().startswith('nvidia-cublas') or name.lower() == 'torch':
        result['packages'].append({'name':name,'version':dist.version,'location':str(dist.locate_file(''))})
for variant, mode in [('baseline','matmul'),('baseline','conv_no_cudnn'),('baseline','conv_cudnn'),('cuda12_first','conv_cudnn'),('cuda12_preload','conv_cudnn')]:
    env=os.environ.copy()
    if variant != 'baseline':
        env['LD_LIBRARY_PATH']='/usr/local/cuda-12.4/lib64:'+env.get('LD_LIBRARY_PATH','')
    if variant == 'cuda12_preload':
        env['LD_PRELOAD']='/usr/local/cuda-12.4/lib64/libcudnn.so.9:/usr/local/cuda-12.4/lib64/libcudnn_graph.so.9'
    proc = subprocess.run([sys.executable, '-c', child, mode], env=env,capture_output=True,text=True,timeout=60)
    try: parsed=json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception: parsed={'stdout':proc.stdout,'stderr':proc.stderr,'exit_code':proc.returncode}
    parsed['stderr']=proc.stderr
    parsed['variant']=variant
    result['tests'].append(parsed)
patterns = [
 '/data/grail-envs/grail/lib/libcudnn*.so*',
 '/data/grail-envs/grail/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn*.so*',
 '/usr/local/cuda-12.4/lib64/libcudnn*.so*',
 '/usr/lib/x86_64-linux-gnu/libcudnn*.so*',
 '/usr/local/lib/python*/dist-packages/nvidia/cudnn/lib/libcudnn*.so*',
 '/root/.cache/uv/archive-v0/*/nvidia/cudnn/lib/libcudnn*.so*',
 '/data/KinoGen/.venvs/*/lib/python*/site-packages/nvidia/cudnn/lib/libcudnn*.so*',
]
result['bounded_cudnn_library_search']={p:glob.glob(p)[:40] for p in patterns}
print(json.dumps(result,indent=2))
'''
if '--utils-only' in sys.argv:
    REMOTE = REMOTE.replace("[('baseline','matmul'),('baseline','conv_no_cudnn'),('baseline','conv_cudnn'),('cuda12_first','conv_cudnn'),('cuda12_preload','conv_cudnn')]", "[('cuda12_preload','conv_cudnn_utils')]")
shell = "export KINOGEN_CUDA_DEVICES=6\nsource /data/KinoGen/scripts/activate-server.sh gpu\nexec /data/KinoGen/.venvs/gpu/bin/python - <<'KINOGEN_DIAG'\n" + REMOTE + "\nKINOGEN_DIAG\n"
proc = subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "yz-ai-4090-01.bce", "bash -s"], input=shell.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
print(proc.stdout.decode(errors="replace"))
if proc.stderr:
    print(proc.stderr.decode(errors="replace"))
if proc.returncode == 0:
    report=json.loads(proc.stdout)
    output=ROOT / ('reports/cudnn-foundationpose-utils-diagnostic.json' if '--utils-only' in sys.argv else 'reports/cudnn-readonly-diagnostic.json')
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
raise SystemExit(proc.returncode)
