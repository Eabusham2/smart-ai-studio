"""Fail publication unless every native-OS package matches its tested receipt."""
from pathlib import Path
import argparse
import hashlib
import json


def verify(root, commit):
    root=Path(root)
    for target in ('Linux-x86_64','Windows-x86_64','macOS-arm64'):
        record=json.loads((root/f'SmartAI-{target}.receipt.json').read_text(encoding='utf-8'))
        if record.get('schema')!=1 or record.get('target')!=target or record.get('commit')!=commit or record.get('gui_smoke')!='passed':
            raise RuntimeError('Package receipt does not describe the tested commit: '+target)
        files=record.get('files')
        if not isinstance(files,list) or len(files)<2:raise RuntimeError('Package files missing')
        names=set()
        for item in files:
            name=item['name']
            if Path(name).name!=name or name in names:raise RuntimeError('Unsafe or duplicate package name')
            names.add(name);path=root/name
            if path.stat().st_size!=item['size'] or item['size']<=0:raise RuntimeError('Package size mismatch: '+name)
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
            if h.hexdigest()!=item['sha256']:raise RuntimeError('Package hash mismatch: '+name)
        if not any('portable.zip' in n for n in names) or not any('Setup.zip' in n for n in names):
            raise RuntimeError('Both portable and setup packages are required: '+target)
    print('All portable/setup package receipts verified for',commit)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--commit',required=True)
    a=p.parse_args();verify(a.root,a.commit)
