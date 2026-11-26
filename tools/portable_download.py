"""Generate one self-contained Linux portable downloader for split ZIPs."""
from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
from urllib.parse import quote, urlsplit


_SCRIPT = r'''#!/bin/sh
# Download the complete portable app. No system Python is required.
set -eu
ARCHIVE=@@ARCHIVE@@
ARCHIVE_SHA256=@@HASH@@
DOWNLOAD_BASE=@@BASE@@
DOWNLOAD_ONLY=0
OUTPUT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fail() { printf '%s\n' "SmartAI: $*" >&2; exit 1; }
while [ "$#" -gt 0 ]; do
    case "$1" in
        --download-only) DOWNLOAD_ONLY=1; shift ;;
        --output-dir)
            [ "$#" -ge 2 ] || fail '--output-dir needs a directory'
            OUTPUT_DIR=$2; shift 2 ;;
        --help)
            printf '%s\n' 'Usage: sh Download-SmartAI-Linux.sh [--output-dir DIRECTORY] [--download-only]'
            exit 0 ;;
        *) fail "Unknown argument: $1" ;;
    esac
done
if command -v sha256sum >/dev/null 2>&1; then
    checksum() { sha256sum < "$1"; }
elif command -v shasum >/dev/null 2>&1; then
    checksum() { shasum -a 256 < "$1"; }
else
    fail 'Install sha256sum or shasum, then run this downloader again.'
fi
verify() {
    actual=$(checksum "$1") || return 1
    [ "${actual%% *}" = "$2" ]
}
if [ "$DOWNLOAD_ONLY" -eq 0 ]; then
    command -v unzip >/dev/null 2>&1 || fail 'Install unzip, or use --download-only.'
fi
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR=$(CDPATH= cd -- "$OUTPUT_DIR" && pwd)
APP_DIR="$OUTPUT_DIR/SmartAI-Linux"
if [ "$DOWNLOAD_ONLY" -eq 0 ] && { [ -e "$APP_DIR" ] || [ -L "$APP_DIR" ]; }; then
    fail 'SmartAI-Linux already exists here. Use the in-app updater or a different --output-dir.'
fi
WORK=$(mktemp -d "$OUTPUT_DIR/.smartai-download.XXXXXX")
trap 'rm -rf -- "$WORK"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
PACKAGE="$OUTPUT_DIR/$ARCHIVE"
fetch_part() {
    name=$1; size=$2; expected=$3
    printf '%s\n' "Downloading $name..."
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 2 --connect-timeout 30 @@CURL_PROTOCOL@@ \
            --output "$WORK/$name" "$DOWNLOAD_BASE/$name"
    elif command -v wget >/dev/null 2>&1; then
        wget @@WGET_PROTOCOL@@ --output-document="$WORK/$name" -- "$DOWNLOAD_BASE/$name"
    else
        fail 'Install curl or wget, then run this downloader again.'
    fi
    actual_size=$(wc -c < "$WORK/$name" | tr -d '[:space:]')
    [ "$actual_size" = "$size" ] || fail "Wrong size for $name; nothing was installed."
    verify "$WORK/$name" "$expected" || fail "Checksum mismatch for $name; nothing was installed."
    cat "$WORK/$name" >> "$WORK/$ARCHIVE"
    rm -- "$WORK/$name"
}
if [ -e "$PACKAGE" ] || [ -L "$PACKAGE" ]; then
    [ -f "$PACKAGE" ] && [ ! -L "$PACKAGE" ] || fail 'Existing archive is not a regular file.'
    verify "$PACKAGE" "$ARCHIVE_SHA256" || fail 'Existing ZIP has a different checksum. Move it or choose another --output-dir.'
    printf '%s\n' 'Reusing the verified portable ZIP.'
else
    : > "$WORK/$ARCHIVE"
@@PARTS@@
    verify "$WORK/$ARCHIVE" "$ARCHIVE_SHA256" || fail 'Complete ZIP checksum mismatch; nothing was installed.'
    mv -n -- "$WORK/$ARCHIVE" "$PACKAGE"
    [ ! -e "$WORK/$ARCHIVE" ] || fail 'An archive appeared at the destination; nothing was overwritten.'
fi
printf '%s\n' "Verified portable ZIP: $PACKAGE"
if [ "$DOWNLOAD_ONLY" -eq 1 ]; then
    exit 0
fi
# The archive is hash-pinned above. Validate its root before writing any members.
unzip -Z -1 "$PACKAGE" > "$WORK/entries"
while IFS= read -r member; do
    case "$member" in
        SmartAI-Linux/*) ;;
        *) fail 'Unexpected path in portable ZIP.' ;;
    esac
    case "/$member/" in
        *'/../'*|*'/./'*|*'\'*) fail 'Unsafe path in portable ZIP.' ;;
    esac
done < "$WORK/entries"
unzip -q "$PACKAGE" -d "$WORK/extracted"
LAUNCHER=SmartAI-Linux/SmartAI.AppDir/AppRun
[ -f "$WORK/extracted/$LAUNCHER" ] || fail 'The portable ZIP is missing its launcher.'
chmod +x "$WORK/extracted/$LAUNCHER"
mv -T -n -- "$WORK/extracted/SmartAI-Linux" "$APP_DIR"
[ ! -e "$WORK/extracted/SmartAI-Linux" ] || fail 'The destination appeared during extraction; nothing was overwritten.'
printf '%s\n' "Portable app ready: $APP_DIR" "Launch with: \"$OUTPUT_DIR/$LAUNCHER\""
'''


def write_linux_downloader(manifest_path, output_dir, *, repository, tag,
                           download_base=None):
    """Embed verified archive metadata so users need only this download helper.

    ``download_base`` supports a local HTTP fixture or HTTPS mirror. Normal
    release generation always uses the repository's GitHub release URL.
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    archive = manifest.get('archive', '')
    if not re.fullmatch(r'SmartAI-Linux-[A-Za-z0-9_]+-portable\.zip', archive):
        raise ValueError('Unsafe or unsupported portable archive name')
    full_hash = manifest.get('sha256', '')
    if not re.fullmatch(r'[0-9a-f]{64}', full_hash):
        raise ValueError('Invalid portable archive checksum')
    parts = manifest.get('parts')
    if not isinstance(parts, list) or not 1 <= len(parts) <= 1000:
        raise ValueError('Invalid portable archive parts')
    commands = []
    for index, row in enumerate(parts):
        if (not isinstance(row, dict)
                or row.get('name') != f'{archive}.part{index:03d}'
                or type(row.get('size')) is not int or row['size'] <= 0
                or not re.fullmatch(r'[0-9a-f]{64}', row.get('sha256', ''))):
            raise ValueError('Invalid or unsafe portable archive part')
        commands.append('    fetch_part ' + ' '.join(shlex.quote(str(row[key]))
                                                    for key in ('name', 'size', 'sha256')))
    if (not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repository)
            or any(part in ('.', '..') for part in repository.split('/'))
            or not isinstance(tag, str) or not tag or any(ord(c) < 32 for c in tag)):
        raise ValueError('Invalid release location')
    base = download_base or f'https://github.com/{repository}/releases/download/{quote(tag, safe="")}'
    parsed = urlsplit(base)
    local_http = parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')
    if (parsed.scheme != 'https' and not local_http) or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Downloads require HTTPS or a loopback HTTP fixture')
    replacements = {
        '@@ARCHIVE@@': shlex.quote(archive),
        '@@HASH@@': shlex.quote(full_hash),
        '@@BASE@@': shlex.quote(base.rstrip('/')),
        '@@PARTS@@': '\n'.join(commands),
        '@@CURL_PROTOCOL@@': '' if local_http else "--proto '=https' --proto-redir '=https'",
        '@@WGET_PROTOCOL@@': '' if local_http else '--https-only',
    }
    script = _SCRIPT
    for marker, value in replacements.items():
        script = script.replace(marker, value)
    destination = Path(output_dir) / 'Download-SmartAI-Linux.sh'
    destination.write_text(script, encoding='utf-8')
    destination.chmod(0o755)
    return destination
