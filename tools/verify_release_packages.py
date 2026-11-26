"""Verify the small, canonical download set before publishing a release."""
from pathlib import Path
import argparse
import hashlib
import json
import re


TARGETS = ('Linux-x86_64', 'Windows-x86_64', 'macOS-arm64')
INSTALLER = 'Install-SmartAI-macOS-Full.command'
LINUX_DOWNLOADER = 'Download-SmartAI-Linux.sh'
MAX_RELEASE_ASSETS = 19
HASH = re.compile(r'^[a-f0-9]{64}$')


def file_record(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(block)
    return {'name': path.name, 'size': path.stat().st_size, 'sha256': h.hexdigest()}


def verify(root, commit, *, include_installer=False):
    root = Path(root)
    expected = []
    for target in TARGETS:
        receipt_path = root / f'SmartAI-{target}.receipt.json'
        record = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (record.get('schema') != 1 or record.get('target') != target
                or record.get('commit') != commit or record.get('gui_smoke') != 'passed'):
            raise RuntimeError('Package receipt does not describe the tested commit: ' + target)
        if record.get('setup_smoke') != 'passed':
            raise RuntimeError('Setup first launch was not verified: ' + target)
        if record.get('bonsai_native_smoke') != 'passed':
            raise RuntimeError('Bonsai MTP runtime launch was not verified: ' + target)
        native_status = 'not_applicable' if target.startswith('Windows') else 'passed'
        if record.get('bundled_native_verified') != native_status:
            raise RuntimeError('Bundled native runtime was not verified: ' + target)
        files = record.get('files')
        if not isinstance(files, list) or len(files) < 2:
            raise RuntimeError('Package files missing: ' + target)
        portable = f'SmartAI-{target}-portable.zip'
        setup = f'SmartAI-{target}-Setup.zip'
        allowed = {portable, setup}
        if target.startswith('Linux'):
            allowed.update({portable + '.parts.json', LINUX_DOWNLOADER})
        names = set()
        rows = {}
        for item in files:
            name = item.get('name')
            if (not isinstance(name, str) or Path(name).name != name or name in names
                    or '\\' in name or not HASH.fullmatch(str(item.get('sha256', '')))):
                raise RuntimeError('Unsafe, duplicate or unverified package name')
            is_part = target.startswith('Linux') and re.fullmatch(re.escape(portable) + r'\.part\d{3}', name)
            if name not in allowed and not is_part:
                raise RuntimeError('Unexpected release file: ' + name)
            names.add(name)
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise RuntimeError('Package file missing or linked: ' + name)
            actual = file_record(path)
            if (not isinstance(item.get('size'), int) or isinstance(item['size'], bool)
                    or not 0 < item['size'] < 2 * 1024 ** 3 or actual['size'] != item['size']):
                raise RuntimeError('Package size mismatch: ' + name)
            if actual['sha256'] != item['sha256']:
                raise RuntimeError('Package hash mismatch: ' + name)
            rows[name] = item
            expected.append(actual)
        if setup not in names:
            raise RuntimeError('Both portable and setup packages are required: ' + target)
        if portable in names:
            if names != {portable, setup}:
                raise RuntimeError('Unsplit package has redundant support files: ' + target)
        else:
            if not target.startswith('Linux') or portable + '.parts.json' not in names or LINUX_DOWNLOADER not in names:
                raise RuntimeError('Split portable package or automatic downloader is missing: ' + target)
            manifest = json.loads((root / (portable + '.parts.json')).read_text(encoding='utf-8'))
            parts = manifest.get('parts')
            if (manifest.get('archive') != portable or not HASH.fullmatch(str(manifest.get('sha256', '')))
                    or not isinstance(parts, list) or not parts):
                raise RuntimeError('Invalid split archive manifest: ' + target)
            part_names = []
            for index, part in enumerate(parts):
                name = portable + f'.part{index:03d}'
                if not isinstance(part, dict) or part.get('name') != name or rows.get(name) != part:
                    raise RuntimeError('Incomplete or inconsistent split archive: ' + target)
                part_names.append(name)
            if names != {setup, portable + '.parts.json', LINUX_DOWNLOADER, *part_names}:
                raise RuntimeError('Split archive contains redundant files: ' + target)
        expected.append(file_record(receipt_path))
    if include_installer:
        path = root / INSTALLER
        script = path.read_text(encoding='utf-8')
        portable_name = 'SmartAI-macOS-arm64-portable.zip'
        portable = next(row for row in expected if row['name'] == portable_name)
        if (f'COMMIT="{commit}"' not in script or f'ARCHIVE_NAME="{portable_name}"' not in script
                or f'ARCHIVE_SHA256="{portable["sha256"]}"' not in script):
            raise RuntimeError('Mac installer does not pin the verified app archive and commit')
        expected.append(file_record(path))
    names = [item['name'] for item in expected]
    if len(names) != len(set(names)) or len(names) > MAX_RELEASE_ASSETS:
        raise RuntimeError('The release has duplicate or too many assets')
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != set(names):
        raise RuntimeError('Unexpected files in release directory: ' + ', '.join(sorted(actual_names - set(names))))
    print('All portable/setup package receipts verified for', commit)
    return sorted(expected, key=lambda item: item['name'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--commit', required=True)
    parser.add_argument('--include-installer', action='store_true')
    parser.add_argument('--manifest')
    args = parser.parse_args()
    assets = verify(args.root, args.commit, include_installer=args.include_installer)
    if args.manifest:
        Path(args.manifest).write_text(json.dumps({'schema': 1, 'commit': args.commit, 'assets': assets}, indent=2) + '\n', encoding='utf-8')
