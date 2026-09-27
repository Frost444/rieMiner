"""Standalone Riecoin CLI. No UI, wallet secret, GPU identity or host path embedded."""
import argparse, atexit, contextlib, csv, ctypes, hashlib, json, os, pathlib, re, signal, subprocess, sys, time, uuid, shutil

ROOT = pathlib.Path(__file__).resolve().parent
ENGINE = os.environ.get('RIECOIN_CLI_ENGINE', 'gpu')
NAME = 'riecoin-' + ENGINE
OFFSETS = '114023297140211,114189340938131,114355384736051'

def parser():
    p = argparse.ArgumentParser(prog=NAME, add_help=False,
        description='Riecoin ' + ('GPU / R346 (native per-job preparation)' if ENGINE == 'gpu' else 'CPU / Bronze R167') + ' - standalone pool miner; no UI required.')
    p.add_argument('-help', '--help', '-h', action='help', help='show this help and exit')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--version', action='store_true', help='show packaged engine versions')
    mode.add_argument('--init-config', metavar='FILE', help='create a disabled pool config and account template; never overwrite')
    mode.add_argument('--check', action='store_true', help='verify package, config and account binding; no mining')
    mode.add_argument('--mine', action='store_true', help='mine in foreground; Ctrl+C requests a graceful drain')
    mode.add_argument('--status', action='store_true', help='show the owned backend status')
    mode.add_argument('--stop', action='store_true', help='request graceful stop of the backend in --state-dir')
    if ENGINE == 'gpu':
        mode.add_argument('--prepare-data', type=pathlib.Path, metavar='DIRECTORY', help='generate and SHA-256-verify deterministic tables locally; no GPU or pool needed')
        mode.add_argument('--list-devices', action='store_true', help='list NVIDIA GPUs detected by the installed driver')
        p.add_argument('--gpu', metavar='INDEX_OR_UUID', help='GPU selection; automatic only when exactly one GPU is visible')
        p.add_argument('--data-dir', type=pathlib.Path, default=ROOT/'prepared/data/riecoin', help='directory containing generated base/ and wide/ admission tables')
        p.add_argument('--basis', type=pathlib.Path, default=ROOT/'prepared/backend/riecoin-prime-inverse-cache-v1/154529a28f6b8356b2c6a20635a29d1f878b836bf63388a74f615c5d36e42ca7.rcpi', help='generated prime/inverse basis file')
    else:
        p.add_argument('--threads', type=int, default=min(os.cpu_count() or 1, 256), help='CPU worker threads (default: detected logical CPUs, max 256)')
    p.add_argument('--config', type=pathlib.Path, metavar='FILE', help='pool JSON created by --init-config and completed by you')
    p.add_argument('--state-dir', type=pathlib.Path, default=pathlib.Path.home()/'.horizon-riecoin'/ENGINE, help='private state directory; use a distinct directory per running instance')
    p.add_argument('--duration', type=int, default=0, metavar='SECONDS', help='request drain after this duration; 0 = continuous; startup is included')
    p.add_argument('--session-seconds', type=int, default=3600, metavar='SECONDS', help='maximum GPU work duration, >=30; default 3600; new network work can end it earlier')
    p.add_argument('--json', action='store_true', help='print filtered machine-readable events instead of the readable mining console')
    p.add_argument('--console', choices=('auto','table','lines'), default='auto', help='interactive dashboard or append-only text; auto uses table on a terminal')
    p.add_argument('--no-color', action='store_true', help='disable terminal dashboard colors')
    return p

def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with open(tmp, 'x', encoding='utf-8') as f:
        json.dump(value, f, indent=2); f.flush(); os.fsync(f.fileno())
    try: os.chmod(tmp, 0o600)
    except OSError: pass
    os.replace(tmp, p)

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest().upper()

def identity(pid):
    if os.name != 'nt':
        p=pathlib.Path('/proc')/str(pid)
        # /proc exposes different path prefixes through chroot/bind mounts.
        # Bind identity to executable bytes and process birth, not that spelling.
        return ['sha256:'+digest(p/'exe'), (p/'stat').read_text().rsplit(')',1)[1].split()[19]]
    from ctypes import wintypes as w
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];k.OpenProcess.restype=w.HANDLE
    k.CloseHandle.argtypes=[w.HANDLE]
    k.QueryFullProcessImageNameW.argtypes=[w.HANDLE,w.DWORD,w.LPWSTR,ctypes.POINTER(w.DWORD)]
    k.GetProcessTimes.argtypes=[w.HANDLE]+[ctypes.POINTER(w.FILETIME)]*4
    h=k.OpenProcess(0x1000,False,pid)
    if not h: raise OSError('process unavailable')
    try:
        size=w.DWORD(32768); buf=ctypes.create_unicode_buffer(size.value)
        times=[w.FILETIME() for _ in range(4)]
        if not k.QueryFullProcessImageNameW(h,0,buf,ctypes.byref(size)) or not k.GetProcessTimes(h,*[ctypes.byref(t) for t in times]): raise OSError('process identity unavailable')
        return [buf.value.casefold(),str((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)]
    finally: k.CloseHandle(h)

def owner(state):
    p=state/'owner.json'
    if not p.exists(): return None
    v=json.loads(p.read_text(encoding='utf-8'))
    try:
        actual=identity(v['pid'])
        if actual==v['identity']: return v
        if os.name!='nt' and actual[1]==v['identity'][1] and not v['identity'][0].startswith('sha256:'):
            # Read-only compatibility with an old path-based owner receipt.
            if actual[0]=='sha256:'+digest(pathlib.Path(v['identity'][0])): return v
            raise ValueError('Running legacy owner cannot be authenticated; use Ctrl+C in its console.')
        return None
    except (OSError, ProcessLookupError):
        if os.name!='nt' and (pathlib.Path('/proc')/str(v['pid'])).exists():
            raise ValueError('Owned process exists but its identity is inaccessible; refusing a second start.')
        return None

def stop(v):
    save(pathlib.Path(v['live_state'])/'stop-request.json',dict(schema='riecoin-graceful-stop-v1',action='drain',backend_pid=v['pid'],control_token=v['token'],request_id=uuid.uuid4().hex))

def devices():
    smi='nvidia-smi.exe' if os.name=='nt' else 'nvidia-smi'
    if os.name=='nt' and not shutil.which(smi):
        for parent in (pathlib.Path(os.environ.get('SystemRoot','C:/Windows'))/'System32',pathlib.Path(os.environ.get('ProgramFiles','C:/Program Files'))/'NVIDIA Corporation'/'NVSMI'):
            if (parent/smi).exists(): smi=str(parent/smi);break
    if os.name!='nt' and pathlib.Path('/usr/lib/wsl/lib/nvidia-smi').exists(): smi='/usr/lib/wsl/lib/nvidia-smi'
    text=subprocess.check_output([smi,'--query-gpu=index,uuid,pci.bus_id,name,compute_cap','--format=csv,noheader,nounits'],text=True)
    rows=[]
    for row in csv.reader(text.splitlines()):
        index,uid,pci,name,cc=map(str.strip,row)
        rows.append(dict(index=index,uuid=uid,pci=pci,name=name,compute=float(cc)))
    return rows

def lock_state(state):
    state.mkdir(parents=True,exist_ok=True,mode=0o700)
    f=open(state/'instance.lock','a+b');f.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            if f.read(1)==b'': f.write(b'0');f.flush()
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        f.close();raise ValueError('this state directory is already owned by a running CLI')
    return f

def main():
    a=parser().parse_args(['--help'] if sys.argv[1:]==['/help'] else None);state=a.state_dir.expanduser().resolve()
    if a.version:
        print(NAME+(' R346 RC6 candidate' if ENGINE=='gpu' else ' R167 portable Alpha')); return 0
    if a.init_config:
        p=pathlib.Path(a.init_config).resolve(); account=p.with_suffix('.account.conf')
        if p.exists() or account.exists(): raise ValueError('config/account already exists; nothing overwritten')
        p.parent.mkdir(parents=True,exist_ok=True)
        account.write_text('mode=pool\nhost=YOUR_POOL_HOST\nport=YOUR_POOL_PORT\npayout_address=YOUR_RIECOIN_ADDRESS\nusername=YOUR_POOL_ACCOUNT_OR_WORKER\npassword=YOUR_POOL_PASSWORD\n',encoding='utf-8')
        os.chmod(account,0o600)
        save(p,dict(schema='riecoin-pool-config-v1',network='mainnet',host='YOUR_POOL_HOST',port=0,payout_address='YOUR_RIECOIN_ADDRESS',credential_source=account.name,gpu_uuid='',gpu_pci='',network_mining_enabled=True,observability_gate='ready-full-horizon-network-runtime'))
        print('Edit both files, set a valid port:',p,account);return 0
    if a.status or a.stop:
        v=owner(state)
        if a.stop and v: stop(v)
        print(json.dumps(dict(running=bool(v),stop_requested=bool(a.stop and v),state_dir=str(state))));return 0
    if ENGINE=='gpu' and a.list_devices:
        print(json.dumps(devices(),indent=2));return 0
    backend=ROOT/"bin/riecoin-network-backend";stage=ROOT/"bin/riecoin-r346"
    if ENGINE=='gpu' and a.prepare_data:
        from prepare_data import prepare
        prepare(a.prepare_data)
        return 0
    if a.duration<0 or a.session_seconds<30: raise ValueError('duration >=0 and session-seconds >=30 required')
    if not a.config: raise ValueError('--config FILE is required')
    state_lock=lock_state(state)  # Held until this foreground controller exits.
    if owner(state): raise ValueError('an owned miner already uses this state directory')
    original=a.config.expanduser().resolve();cfg=json.loads(original.read_text(encoding='utf-8-sig'))
    cfg['credential_source']=str((original.parent/cfg['credential_source']).resolve())
    env=os.environ.copy()
    for key in list(env):
        if key.startswith('ASTRA_') or key in ('CUDA_VISIBLE_DEVICES','HORIZON_CPU_THREADS','HORIZON_CPU_CACHE','HORIZON_RIECOIN_BASIS','HORIZON_RIECOIN_DATA'): env.pop(key)
    if ENGINE=='gpu':
        rows=devices();selected=[d for d in rows if a.gpu in (d['index'],d['uuid'])] if a.gpu else rows
        if len(selected)!=1: raise ValueError('select exactly one GPU with --gpu; see --list-devices')
        d=selected[0]
        if d['compute']<8.6: raise ValueError('R346 requires NVIDIA compute capability >=8.6')
        cfg.update(gpu_uuid=d['uuid'],gpu_pci=d['pci'])
        env.update(CUDA_VISIBLE_DEVICES=d['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',HORIZON_RIECOIN_DATA=str(a.data_dir.resolve()),HORIZON_RIECOIN_BASIS=str(a.basis.resolve()),ASTRA_CUDA_WAIT_POLICY='block',ASTRA_A95_PLANES='32',ASTRA_A93_FRONTIER_GROUPS='8',ASTRA_R178_PHASES='1')
    else:
        if not 1<=a.threads<=256: raise ValueError('threads must be 1..256')
        cfg.update(gpu_uuid='CPU-HOST',gpu_pci='CPU')
        env.update(HORIZON_CPU_THREADS=str(a.threads),HORIZON_CPU_CACHE=str(state/'cache'))
    run=state/'runs'/(time.strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]);run.mkdir(parents=True)
    config=run/'pool.json';save(config,cfg)
    if not cfg.get('network_mining_enabled'): raise ValueError('network mining is disabled in your config')
    if ENGINE=='gpu':
        from prepare_data import verify_mining_data
        with contextlib.redirect_stdout(sys.stderr if a.json else sys.stdout):
            verify_mining_data(a.data_dir.resolve(), a.basis.resolve())
    token=uuid.uuid4().hex;live=run/'live-state'
    args=[str(backend),'--live','--config',str(config),'--stage-exe',str(stage),'--live-state',str(live),'--control-token',token,'--positions',str(524288 if ENGINE=='gpu' else 1048576),'--primorial-number','114','--primorial-offset','114023297140211','--device','0','--difficulty-offset','0','--factor-origin','0']
    if ENGINE=='gpu': args+=['--production-session-seconds',str(a.session_seconds),'--production-max-batches','15728640','--rejected-audit-per-stage','8']
    else: args+=['--primorial-offsets',OFFSETS]
    from console import Console
    console = Console(a.json)
    with contextlib.redirect_stdout(sys.stderr if a.json else sys.stdout):
        print('Horizon RC6 | Riecoin GPU | Silver R346', flush=True)
        print('---------------------------------------------------------------', flush=True)
        print('Pool       : {}:{}'.format(cfg.get('host',''),cfg.get('port')), flush=True)
        print('GPU        : {} | CUDA capability {}'.format(d['name'],d['compute']), flush=True)
        print('Session    : {} s per stage | exact replay before submission'.format(a.session_seconds), flush=True)
        print('Preparation: verified local tables; Linux per-job atlas fallback', flush=True)
        print('Statistics : Speed = measured candidates/s; Average = rolling 60 seconds of work', flush=True)
        print('             PRP/s = Q0 tests / GPU execution time; r = current-work Q0/Q1', flush=True)
        print('             No rate is pool-credited hashrate; preparation is not hidden in a yield claim', flush=True)
        print('             Sh = accepted/sent; R/S/E = rejected/stale/errors', flush=True)
        print('             r* and Q7 appear when measured; -- means unavailable', flush=True)
        print('Control    : Ctrl+C requests drain; private logs in --state-dir', flush=True)
        print('---------------------------------------------------------------', flush=True)
    started=time.monotonic();draining=False
    console.open_dashboard(dict(gpu=d['name'],compute=d['compute'],pool='{}:{}'.format(cfg.get('host',''),cfg.get('port'))),a.console,a.no_color)
    console_parent=os.getppid() if sys.stdin.isatty() else None
    with open(run/'backend.jsonl','wb') as output,open(run/'backend.stderr.log','wb') as errors:
        child=subprocess.Popen(args,env=env,cwd=ROOT,stdout=output,stderr=errors,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0,start_new_session=os.name!='nt')
        v=dict(pid=child.pid,identity=identity(child.pid),token=token,live_state=str(live),log=str(run/'backend.jsonl'));save(state/'owner.json',v)
        def cleanup_child():
            if child.poll() is None:
                try: stop(v)
                except OSError: pass  # Native backend also watches parent lifetime.
            console.close()
        atexit.register(cleanup_child)
        console.emit('Connecting to the pool; waiting for verified work...')
        def request_stop(*_):
            nonlocal draining
            if not draining: stop(v);draining=True;console.emit('Drain requested; waiting for verified work and acknowledgements.')
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        if hasattr(signal, 'SIGHUP'): signal.signal(signal.SIGHUP,request_stop)
        telemetry=open(run/'backend.jsonl',encoding='utf-8',errors='replace')
        while child.poll() is None:
            console.tick()
            # The journal contains structured metrics, never credential content.
            while True:
                position=telemetry.tell();line=telemetry.readline()
                if not line: break
                if not line.endswith('\n'):
                    telemetry.seek(position);break
                try: event=json.loads(line)
                except ValueError: continue
                if isinstance(event,dict):
                    console.event(event)
            if a.duration and time.monotonic()-started>=a.duration: request_stop()
            if console_parent is not None and os.getppid()!=console_parent: request_stop()
            if console.closed: request_stop()
            if console.stop_requested: request_stop()
            time.sleep(0.25)
        for line in telemetry:
            try: event=json.loads(line)
            except ValueError: continue
            if isinstance(event,dict): console.event(event)
        telemetry.close()
        console.close()
        atexit.unregister(cleanup_child)
        print('Backend exited:',child.returncode,'; detailed logs retained in the private state directory.',flush=True,file=sys.stderr if a.json else sys.stdout)
        return child.returncode

if __name__=='__main__':
    try: sys.exit(main())
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:
        print(NAME+': '+str(e),file=sys.stderr);sys.exit(2)
