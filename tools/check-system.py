"""Read-only runtime diagnosis. Never reads or prints accounts or credentials."""
import csv, ctypes, importlib.util, os, pathlib, platform, shutil, subprocess, sys
root=pathlib.Path(__file__).resolve().parents[1]
errors=[]
def check(label,ok,detail):
    print(('PASS' if ok else 'FAIL')+'  '+label+': '+str(detail))
    if not ok:errors.append(label)
check('Architecture',platform.machine() in ('x86_64','AMD64'),platform.machine())
check('Python',sys.version_info>=(3,11),platform.python_version())
check('Rich',importlib.util.find_spec('rich') is not None,'install with bash setup-debian.sh runtime if missing')
for name in ('libgmp.so.10','libcrypto.so.3','libcuda.so.1'):
    library='/usr/lib/wsl/lib/libcuda.so.1' if name=='libcuda.so.1' and pathlib.Path('/usr/lib/wsl/lib/libcuda.so.1').is_file() else name
    try:ctypes.CDLL(library);check(name,True,'available')
    except OSError:check(name,False,'missing from loader paths; see docs/TROUBLESHOOTING.md')
for name in ('riecoin-network-backend','riecoin-r346','data-generator'):
    path=root/'cli/riecoin-gpu/bin'/name
    check(name,path.is_file() and os.access(path,os.X_OK),'executable present' if path.exists() else 'missing; rebuild/extract the complete archive')
smi='/usr/lib/wsl/lib/nvidia-smi' if pathlib.Path('/usr/lib/wsl/lib/nvidia-smi').exists() else shutil.which('nvidia-smi')
if smi:
    try:
        env=os.environ.copy()
        if pathlib.Path('/usr/lib/wsl/lib').is_dir():env['LD_LIBRARY_PATH']='/usr/lib/wsl/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
        result=subprocess.run([smi,'--query-gpu=index,name,driver_version,compute_cap,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=10,env=env)
        check('NVIDIA driver',result.returncode==0,result.stdout.strip() if result.returncode==0 else 'query failed')
        if result.returncode==0:
            rows=list(csv.reader(result.stdout.splitlines()))
            suitable=any(len(row)>=5 and float(row[3].strip())>=8.6 for row in rows)
            check('GPU capability',suitable,'at least one capability8.6+ device required; choose it with --gpu')
    except (OSError,subprocess.TimeoutExpired):check('NVIDIA driver',False,'query unavailable')
else:check('NVIDIA driver',False,'nvidia-smi unavailable; install a suitable driver (host Windows driver in WSL)')
print('Requirements: NVIDIA compute capability >=8.6; CUDA12.8-capable driver recommended; >=3GB free data storage.')
print('This diagnostic does not replace a GPU/pool test. It changes no system settings.')
sys.exit(1 if errors else 0)
