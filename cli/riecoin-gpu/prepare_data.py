"""Reproduce and verify the public deterministic tables; never download them."""
import hashlib, json, pathlib, re, subprocess, shutil, uuid

ROOT = pathlib.Path(__file__).resolve().parent

def entries():
    return json.loads((ROOT/'riecoin-data-manifest.json').read_text())['files']

def valid(path, entry):
    if not path.is_file() or path.stat().st_size != entry['raw_bytes']:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest().upper() == entry['raw_sha256'].upper()

def verify_mining_data(data_dir, basis):
    print('Verifying deterministic admission data (size and SHA-256)...', flush=True)
    for entry in entries():
        relative = pathlib.PurePosixPath(entry['path'])
        path = basis if relative.suffix == '.rcpi' else data_dir / pathlib.Path(*relative.parts[2:])
        if not valid(path, entry):
            raise ValueError('Missing or invalid admission data: '+str(path)+'; use --prepare-data or supply valid --data-dir/--basis')

def prepare(destination):
    destination = pathlib.Path(destination).expanduser().resolve()
    destination.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (destination/'.prepare.lock').open('a+b') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError: raise ValueError('Another preparation owns this directory; let it finish first.')
        _prepare_locked(destination)


def _prepare_locked(destination):
    print('First preparation creates about 1.7 GB locally; it is not bundled or downloaded.', flush=True)
    for entry in entries():
        relative = pathlib.PurePosixPath(entry['path'])
        target = destination / pathlib.Path(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if valid(target, entry):
            print('Verified:', relative, flush=True)
            continue
        if target.exists():
            raise ValueError('Existing data fails verification; move it aside before retrying: '+str(target))
        temporary = target.with_name(target.name+'.preparing')
        if temporary.exists():
            if valid(temporary,entry):
                temporary.replace(target)
                print('Recovered verified table:',relative,flush=True)
                continue
            retained=temporary.with_name(temporary.name+'.interrupted-'+uuid.uuid4().hex[:8])
            temporary.rename(retained)
            print('Preserved incomplete table as:',retained,flush=True)
        if shutil.disk_usage(destination).free < entry['raw_bytes']+64*1024*1024:
            raise ValueError('Insufficient free space for the next verified table; no existing data was deleted.')
        if relative.suffix == '.rcpi':
            arguments = ['basis', '1000000000', '114', str(temporary)]
        else:
            match = re.fullmatch(r'(delta4b-)?band-(\d+)-(\d+)\.u32', relative.name)
            if not match: raise ValueError('Unknown manifest table shape')
            arguments = ['band', match[2], match[3], '4000000000' if match[1] else '0', str(temporary)]
        subprocess.run([str(ROOT/'bin/data-generator'), *arguments], check=True)
        if not valid(temporary, entry):
            raise ValueError('Generated file failed its published SHA-256: '+str(temporary))
        temporary.replace(target)
        print('Generated and verified:', relative, flush=True)
