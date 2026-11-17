#!/usr/bin/env python3
"""Create the separate user-space macOS portable release option.

The full macOS ZIP already embeds tested dependencies plus both Bonsai-2 MLX
snapshots. GitHub caps each release asset below 2 GiB, so the no-admin kit
reuses its authenticated multipart files instead of uploading 18 GiB twice.
After assembly, the full app runs locally/offline from ~/Applications without
any package-manager, Python, root, or installer privileges.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
import zipfile


ARCHIVE = 'SmartAI-macOS-arm64.zip'
MANIFEST = 'SmartAI-macOS-arm64.bundle.json'
ASSEMBLER = 'Reassemble-SmartAI-macOS-arm64.command'
KIT = 'SmartAI-macOS-arm64-NoAdmin-Portable-Kit.zip'
LAUNCHER = 'Install-and-Launch-SmartAI-NoAdmin.command'
README = 'README-NoAdmin-Portable.txt'
REPOS = {
    'prism-ml/Ternary-Bonsai-2-27B-mlx-2bit',
    'dealignai/Bonsai-2-27B-CRACK-Ternary-JANG',
}


def verify_manifest(data):
    """Reject incomplete, wrong-model or path-injection bundle metadata."""
    if not isinstance(data, dict) or data.get('archive_name') != ARCHIVE:
        raise ValueError('Portable kit requires the model-bundled macOS ZIP')
    models = data.get('models')
    if not isinstance(models, list) or {x.get('repo_id') for x in models if isinstance(x, dict)} != REPOS or len(models) != 2:
        raise ValueError('The macOS bundle must contain both original Bonsai 2 model repositories')
    for item in models:
        if not re.fullmatch(r'[0-9a-f]{40}', str(item.get('revision', ''))):
            raise ValueError('Model release revision must be immutable')
        if type(item.get('files')) is not int or item['files'] < 2:
            raise ValueError('A model is missing its published files')
        if type(item.get('expected_bytes')) is not int or item['expected_bytes'] <= 0:
            raise ValueError('A model has no published model-weight size')
    parts = data.get('parts')
    if not isinstance(parts, list) or not parts:
        raise ValueError('The archive has no multipart download receipts')
    if type(data.get('archive_size')) is not int or data['archive_size'] <= 0:
        raise ValueError('Archive size missing')
    if not re.fullmatch(r'[a-f0-9]{64}', str(data.get('archive_sha256', ''))):
        raise ValueError('Archive SHA256 missing')
    seen = set()
    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            raise ValueError('Invalid part metadata')
        name = part.get('name')
        if name != f'{ARCHIVE}.part{idx:03d}' or name in seen:
            raise ValueError('Multipart parts must be complete, unique, contiguous and ordered')
        seen.add(name)
        if type(part.get('size')) is not int or not 0 < part['size'] < 2**31:
            raise ValueError('Part exceeds GitHub release limit')
        if not re.fullmatch(r'[a-f0-9]{64}', str(part.get('sha256', ''))):
            raise ValueError('Missing model part hash')
    if sum(part['size'] for part in parts) != data['archive_size']:
        raise ValueError('Part sizes do not match the complete archive')
    return data


LAUNCHER_TEMPLATE = r'''#!/bin/bash
set -euo pipefail

# No administrator rights, Homebrew, Xcode or system Python are used here.
# The verified archive contains an embedded Python environment and BOTH models.
if [ "$(/usr/bin/uname -s)" != "Darwin" ] || [ "$(/usr/bin/uname -m)" != "arm64" ]; then
  echo "This package requires an Apple Silicon Mac running macOS." >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="$HERE/__MANIFEST__"
ASSEMBLER="$HERE/__ASSEMBLER__"
CACHE="${SMARTAI_PORTABLE_CACHE:-$HOME/Library/Caches/SmartAI/NoAdmin-__SHA__}"
INSTALL_ROOT="${SMARTAI_PORTABLE_DEST:-$HOME/Applications/SmartAI-Portable}"
APP_DIR="$INSTALL_ROOT/__SHA__"
APP="$APP_DIR/SmartAI.app"
CODE="$APP/Contents/Resources/app"

check_app() {
  [ -x "$CODE/runtime/python/bin/python3" ] && \
  [ -f "$CODE/runtime/python/.smartai-deps-ready" ] && \
  [ -f "$CODE/preloaded_models/prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/config.json" ] && \
  [ -f "$CODE/preloaded_models/dealignai--Bonsai-2-27B-CRACK-Ternary-JANG/config.json" ] && \
  /usr/bin/find "$CODE/preloaded_models/prism-ml--Ternary-Bonsai-2-27B-mlx-2bit" -name '*.safetensors' -type f -print -quit | /usr/bin/grep -q . && \
  /usr/bin/find "$CODE/preloaded_models/dealignai--Bonsai-2-27B-CRACK-Ternary-JANG" -name '*.safetensors' -type f -print -quit | /usr/bin/grep -q .
}

if check_app; then
  echo "Verified installed Smart AI Studio, including both preloaded Bonsai models: $APP"
  if [ "${SMARTAI_PORTABLE_INSTALL_ONLY:-0}" != "1" ]; then /usr/bin/open "$APP"; fi
  exit 0
fi

if [ -e "$APP_DIR" ]; then
  echo "Existing portable install is incomplete; move it aside before reinstalling: $APP_DIR" >&2
  exit 1
fi

/bin/mkdir -p "$CACHE" "$INSTALL_ROOT"
LIST="$CACHE/.smartai-manifest-parts-$$"
trap '/bin/rm -f "$LIST"' EXIT
# The OS-bundled JXA runtime parses JSON. No Python, jq or third-party tools.
/usr/bin/osascript -l JavaScript - "$MANIFEST" <<'JXA' > "$LIST"
ObjC.import('Foundation');
function run(args) {
  const bytes = $.NSData.dataWithContentsOfFile(args[0]);
  if (!bytes) throw Error('Missing portable manifest');
  const text = $.NSString.alloc.initWithDataEncoding(bytes, $.NSUTF8StringEncoding).js;
  const manifest = JSON.parse(text);
  if (manifest.archive_name !== '__ARCHIVE__' || !Array.isArray(manifest.parts) || !manifest.parts.length) throw Error('Invalid archive manifest');
  if (manifest.models.length !== 2) throw Error('Both Bonsai models must be included');
  const expected = new Set(['prism-ml/Ternary-Bonsai-2-27B-mlx-2bit', 'dealignai/Bonsai-2-27B-CRACK-Ternary-JANG']);
  if (manifest.models.some(m => !expected.delete(m.repo_id)) || expected.size) throw Error('Unexpected Bonsai models');
  return manifest.parts.map((p, i) => {
    const name = '__ARCHIVE__.part' + String(i).padStart(3,'0');
    if (p.name !== name || !/^[a-f0-9]{64}$/.test(p.sha256) || !Number.isSafeInteger(p.size) || p.size <= 0 || p.size >= 2147483648) throw Error('Invalid model part metadata');
    return p.sha256 + ' ' + p.size + ' ' + name;
  }).join('\n');
}
JXA

while IFS=' ' read -r hash size name; do
  [ -n "$name" ] || continue
  part="$CACHE/$name"
  valid=0
  if [ -f "$part" ] && [ "$(/usr/bin/stat -f%z "$part")" = "$size" ]; then
    if /bin/echo "$hash  $part" | /usr/bin/shasum -a 256 -c - >/dev/null 2>&1; then valid=1; fi
  fi
  if [ "$valid" != "1" ]; then
    if [ "${SMARTAI_PORTABLE_OFFLINE:-0}" = "1" ]; then
      echo "Offline mode: missing or corrupt model part $name. Download the release parts first." >&2
      exit 1
    fi
    echo "Downloading verified model-bundle part $name ($size bytes)..."
    temp="$part.download"
    /bin/rm -f "$temp"
    /usr/bin/curl --fail --location --silent --show-error --retry 5 \
      --connect-timeout 30 --output "$temp" \
      "https://github.com/__REPO__/releases/download/__TAG__/$name"
    if [ "$(/usr/bin/stat -f%z "$temp")" != "$size" ] || \
       ! /bin/echo "$hash  $temp" | /usr/bin/shasum -a 256 -c - >/dev/null 2>&1; then
      echo "The downloaded model part failed integrity verification: $name" >&2
      exit 1
    fi
    /bin/mv "$temp" "$part"
  fi
done < "$LIST"

# Model files are already IN this verified archive: no model downloads on launch.
/bin/cp "$MANIFEST" "$CACHE/__MANIFEST__"
/bin/cp "$ASSEMBLER" "$CACHE/__ASSEMBLER__"
(
  cd "$CACHE"
  /bin/bash "./__ASSEMBLER__"
)

STAGING="$INSTALL_ROOT/.SmartAI-portable-__SHA__-$$"
/bin/mkdir -p "$STAGING"
cleanup() {
  /bin/rm -rf "$STAGING"
}
trap 'cleanup; /bin/rm -f "$LIST"' EXIT
/usr/bin/ditto -x -k "$CACHE/__ARCHIVE__" "$STAGING"
if [ ! -x "$STAGING/SmartAI.app/Contents/Resources/app/runtime/python/bin/python3" ] || \
   [ ! -f "$STAGING/SmartAI.app/Contents/Resources/app/runtime/python/.smartai-deps-ready" ] || \
   [ ! -f "$STAGING/SmartAI.app/Contents/Resources/app/preloaded_models/prism-ml--Ternary-Bonsai-2-27B-mlx-2bit/config.json" ] || \
   [ ! -f "$STAGING/SmartAI.app/Contents/Resources/app/preloaded_models/dealignai--Bonsai-2-27B-CRACK-Ternary-JANG/config.json" ]; then
  echo "Extracted app lacks the runtime or a Bonsai model. No installation was performed." >&2
  exit 1
fi
/usr/bin/find "$STAGING/SmartAI.app/Contents/Resources/app/preloaded_models/prism-ml--Ternary-Bonsai-2-27B-mlx-2bit" -name '*.safetensors' -type f -print -quit | /usr/bin/grep -q .
/usr/bin/find "$STAGING/SmartAI.app/Contents/Resources/app/preloaded_models/dealignai--Bonsai-2-27B-CRACK-Ternary-JANG" -name '*.safetensors' -type f -print -quit | /usr/bin/grep -q .
/bin/mv "$STAGING" "$APP_DIR"
trap '/bin/rm -f "$LIST"' EXIT
if ! check_app; then
  echo "The portable app failed final model/runtime verification." >&2
  exit 1
fi

echo "Portable Smart AI Studio installed with BOTH Bonsai models at $APP"
echo "No administrator permission or dependency installation was used."
if [ "${SMARTAI_PORTABLE_INSTALL_ONLY:-0}" != "1" ]; then
  /usr/bin/open "$APP"
fi
'''


def make_readme(repo, tag, commit, size, count):
    return f'''SMART AI STUDIO — NO-ADMIN PORTABLE MAC (APPLE SILICON)

This is an additional portable release option. It does NOT modify your system
Python, /Applications, Homebrew, Xcode or macOS settings. It never asks for
administrator credentials. The extracted app runs with its already-bundled
Python runtime, installed dependencies and BOTH original Bonsai 2 models:

1. Bonsai 2 27B Ternary Multimodal (PrismML, Qwen3.8-based)
2. Bonsai 2 27B CRACK Ternary JANG (uncensored/abliterated, Qwen3.8-based)

RELEASE: {repo} / {tag} / {commit}

HOW TO USE
1. Unzip this small Kit into Downloads (or another user-owned folder).
2. Double-click Install-and-Launch-SmartAI-NoAdmin.command, or open Terminal and
   run: /bin/bash "<kit folder>/Install-and-Launch-SmartAI-NoAdmin.command"
3. The helper downloads {count} SHA256-verified model/archive parts from this
   GitHub release (the combined ZIP is {size / 1e9:.1f} GB). It reassembles and
   verifies their exact contents before extracting SmartAI.app.
4. The full app is installed to ~/Applications/SmartAI-Portable/{commit[:12]}/SmartAI.app
   (or set SMARTAI_PORTABLE_DEST to a user-owned directory first). It launches
   with zero admin permissions. The 2 models are already present on disk and
   do NOT need to be downloaded by the app or its MLX runtime.
5. Later launches: open that SmartAI.app normally, including while offline.

GitHub caps individual release assets at 2 GiB. The large model-bearing ZIP is
therefore split across the same parts used by the standard Mac bundle; this
small Kit is a separate one-click ASSEMBLY OPTION, not an additional 18-GB copy.
You can instead manually download the shared parts and run the included helper
with SMARTAI_PORTABLE_OFFLINE=1. Allow substantial free disk space to download,
reassemble and expand the archive (recommend at least 55 GB available).

This app is not notarized. If macOS asks you to approve opening it, review the
publisher/source and use Finder's ordinary Open/Privacy & Security controls.
Do NOT disable Gatekeeper globally and do NOT use elevated privileges.

No machine-learning inference is executed during kit assembly. Actual model
inference and speed depend on your Mac's available unified memory and Metal.
'''


def build(manifest_file, assembler_file, output_dir, *, repo, tag, commit):
    data = verify_manifest(json.loads(Path(manifest_file).read_text('utf-8')))
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
        raise ValueError('Invalid source repository')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', tag):
        raise ValueError('Invalid release tag')
    if not re.fullmatch(r'[a-f0-9]{40}', commit):
        raise ValueError('Missing exact tested commit SHA')
    assembler = Path(assembler_file).read_bytes()
    if not assembler.startswith(b'#!/bin/bash') or b'/usr/bin/shasum' not in assembler:
        raise ValueError('Missing verified native reassembler')
    text = LAUNCHER_TEMPLATE
    for key, value in {
        'MANIFEST': MANIFEST, 'ASSEMBLER': ASSEMBLER, 'ARCHIVE': ARCHIVE,
        'REPO': repo, 'TAG': tag, 'SHA': commit[:12],
    }.items():
        text = text.replace(f'__{key}__', value)
    if '__' in text:  # Fail closed for unrendered template values.
        raise ValueError('Unexpanded launcher placeholder')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / KIT
    content = {
        LAUNCHER: (text.encode('utf-8'), 0o755),
        MANIFEST: (Path(manifest_file).read_bytes(), 0o644),
        ASSEMBLER: (assembler, 0o755),
        README: (make_readme(repo, tag, commit, data['archive_size'], len(data['parts'])).encode('utf-8'), 0o644),
    }
    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name, (blob, mode) in content.items():
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, blob)
    verify(dest, expected=data, repo=repo, tag=tag, commit=commit)
    # A small independently downloadable authenticity receipt for the kit.
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_name(dest.name + '.sha256').write_text(
        f'{digest}  {dest.name}\n', encoding='ascii')
    return dest


def verify(path, *, expected=None, repo=None, tag=None, commit=None):
    with zipfile.ZipFile(path) as zf:
        if set(zf.namelist()) != {LAUNCHER, MANIFEST, ASSEMBLER, README} or zf.testzip():
            raise ValueError('No-admin portable kit missing a required or intact file')
        manifest = verify_manifest(json.loads(zf.read(MANIFEST)))
        if expected is not None and manifest != expected:
            raise ValueError('Kit model archive manifest differs from tested release bundle')
        script = zf.read(LAUNCHER).decode('utf-8')
        if not script.startswith('#!/bin/bash') or '\nset -euo pipefail\n' not in script:
            raise ValueError('Unverified portable launcher')
        if any(token in script for token in ('sudo ', '/usr/bin/sudo', 'brew install', 'pip install', 'curl | sh')):
            raise ValueError('Portable launcher attempts privileged installation')
        for needle in ('SMARTAI_PORTABLE_OFFLINE', 'SMARTAI_PORTABLE_INSTALL_ONLY',
                       'runtime/python/.smartai-deps-ready', 'prism-ml--Ternary-Bonsai-2-27B-mlx-2bit',
                       'dealignai--Bonsai-2-27B-CRACK-Ternary-JANG', '/usr/bin/shasum -a 256', '/usr/bin/ditto -x -k'):
            if needle not in script:
                raise ValueError('No-admin launcher lacks required verification: '+needle)
        if repo and f'https://github.com/{repo}/releases/download/{tag}/' not in script:
            raise ValueError('Incorrect release download URL')
        if commit and f'SmartAI-Portable/{commit[:12]}' not in script and f'NoAdmin-{commit[:12]}' not in script:
            raise ValueError('Portable install location lacks version isolation')
        for executable in (LAUNCHER, ASSEMBLER):
            if not ((zf.getinfo(executable).external_attr >> 16) & 0o111):
                raise ValueError('Portable command file lacks executable permissions')
        if any(name.startswith('/') or '..' in Path(name).parts for name in zf.namelist()):
            raise ValueError('Portable kit contains unsafe paths')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='dist/'+MANIFEST)
    parser.add_argument('--assembler', default='dist/'+ASSEMBLER)
    parser.add_argument('--output-dir', default='dist')
    parser.add_argument('--repo', default='Eabusham2/smart-ai-studio')
    parser.add_argument('--tag', default='latest')
    parser.add_argument('--commit', required=True)
    args = parser.parse_args()
    path = build(args.manifest, args.assembler, args.output_dir, repo=args.repo, tag=args.tag, commit=args.commit)
    print('VERIFIED_NO_ADMIN_PORTABLE_KIT', path, path.stat().st_size)


if __name__ == '__main__':
    main()
