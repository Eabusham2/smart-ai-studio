#!/usr/bin/env python3
"""Publish a standalone, commit-pinned full Mac installer using the verified kit."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

from tools.build_macos_noadmin_portable import KIT, LAUNCHER, verify


INSTALLER = "Install-SmartAI-macOS-Full.command"

TEMPLATE = r'''#!/bin/bash
set -euo pipefail

# Full Apple Silicon install: bundled runtime + BOTH Bonsai 2 models.
# This file pins the exact kit digest; a changing rolling release fails closed.
COMMIT="__COMMIT__"
KIT_SHA256="__KIT_SHA256__"
KIT_NAME="__KIT__"
URL="https://github.com/__REPO__/releases/download/__TAG__"
TARGET="${SMARTAI_FULL_TARGET:-/Applications/SmartAI.app}"
CACHE_ROOT="${SMARTAI_FULL_CACHE:-$HOME/Library/Caches/SmartAI/Full-__SHORT_SHA__}"
DOWNLOADS="$CACHE_ROOT/model-downloads-__SHORT_SHA__"
STAMP="Contents/Resources/app/.smartai-full-install-commit"
WORK=""
BACKUP=""
INCOMING=""
DEST_LOCK=""
USE_SUDO=0
PUBLISHED=0

fail() { echo "Installation stopped: $*" >&2; exit 1; }
if [ "$(/usr/bin/uname -s)" != Darwin ] || [ "$(/usr/bin/uname -m)" != arm64 ]; then
  fail "This installer requires an Apple Silicon Mac."
fi
case "$TARGET" in
  /*/SmartAI.app) ;;
  *) fail "SMARTAI_FULL_TARGET must be an absolute path ending in /SmartAI.app." ;;
esac
[ ! -L "$TARGET" ] || fail "The target is a symbolic link; choose an ordinary app path."
PARENT="$(/usr/bin/dirname "$TARGET")"

check_app() {
  local app="$1" code="$1/Contents/Resources/app" model
  [ -f "$app/Contents/Info.plist" ] && [ -x "$app/Contents/MacOS/applet" ] &&
  [ -f "$code/app_gui.py" ] && [ -f "$code/launch_studio.sh" ] &&
  [ -x "$code/runtime/python/bin/python3" ] &&
  [ -f "$code/runtime/python/.smartai-deps-ready" ] || return 1
  for model in prism-ml--Ternary-Bonsai-2-27B-mlx-2bit dealignai--Bonsai-2-27B-CRACK-Ternary-JANG; do
    [ -f "$code/preloaded_models/$model/config.json" ] || return 1
    /usr/bin/find "$code/preloaded_models/$model" -name '*.safetensors' -type f -size +0c -print -quit |
      /usr/bin/grep -q . || return 1
  done
}

app_running() {
  /bin/ps ax -o command= | /usr/bin/grep -F -- "$TARGET/Contents/Resources/app/" |
    /usr/bin/grep -v -E '(^|[[:space:]/])grep[[:space:]]' >/dev/null
}

if check_app "$TARGET" && [ -f "$TARGET/$STAMP" ] && [ "$(/bin/cat "$TARGET/$STAMP")" = "$COMMIT" ]; then
  echo "This release and both Bonsai 2 models are already installed: $TARGET"
  if app_running; then
    echo "SmartAI is already running; its active chat and learning work is preserved."
  elif [ "${SMARTAI_FULL_INSTALL_ONLY:-0}" != 1 ]; then
    /usr/bin/open "$TARGET"
  fi
  exit 0
fi

/bin/mkdir -p "$CACHE_ROOT"
[ ! -L "$DOWNLOADS" ] || fail "The managed download directory must not be a symbolic link."
/bin/mkdir "$CACHE_ROOT/install.lock" 2>/dev/null ||
  fail "Another installer may be running. If it has exited, remove the empty lock: $CACHE_ROOT/install.lock"

install_command() {
  if [ "$USE_SUDO" = 1 ]; then /usr/bin/sudo "$@"; else "$@"; fi
}
cleanup() {
  local status=$?
  set +e
  trap - EXIT
  # Only our fresh staging folder and locks are removed on failure. Validated
  # downloads remain for retry, and an existing app backup is never deleted.
  if [ "$PUBLISHED" = 0 ] && [ -n "$BACKUP" ] && [ -d "$BACKUP" ] && [ ! -e "$TARGET" ]; then
    install_command /bin/mv "$BACKUP" "$TARGET"
  fi
  if [ -n "$WORK" ]; then /bin/rm -rf "$WORK"; fi
  if [ -n "$INCOMING" ] && [ -e "$INCOMING" ]; then install_command /bin/rm -rf "$INCOMING"; fi
  if [ -n "$DEST_LOCK" ]; then install_command /bin/rmdir "$DEST_LOCK" 2>/dev/null || true; fi
  /bin/rmdir "$CACHE_ROOT/install.lock" 2>/dev/null || true
  if [ "$status" != 0 ] && [ "$PUBLISHED" = 0 ]; then
    echo "Download cache retained for retry: $DOWNLOADS" >&2
    if [ -n "$BACKUP" ] && [ -d "$BACKUP" ]; then echo "Previous app retained at: $BACKUP" >&2; fi
  fi
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
WORK="$(/usr/bin/mktemp -d "$CACHE_ROOT/work.XXXXXX")"

# Refuse to delete or reuse a caller's unrelated directory. This marker belongs
# only to the exact release's dedicated part cache, never to CACHE_ROOT itself.
if [ -e "$DOWNLOADS" ]; then
  [ -f "$DOWNLOADS/.full-installer-owner" ] &&
  [ "$(/bin/cat "$DOWNLOADS/.full-installer-owner")" = "$COMMIT" ] ||
    fail "Download directory has no matching ownership marker: $DOWNLOADS"
else
  /bin/mkdir "$DOWNLOADS"
  printf '%s\n' "$COMMIT" > "$DOWNLOADS/.full-installer-owner"
fi

KIT_PATH="$DOWNLOADS/$KIT_NAME"
verify_kit() { printf '%s  %s\n' "$KIT_SHA256" "$KIT_PATH" | /usr/bin/shasum -a 256 -c - >/dev/null 2>&1; }
if ! verify_kit; then
  [ "${SMARTAI_FULL_OFFLINE:-0}" != 1 ] || fail "Offline mode: the pinned installer kit is missing or corrupt."
  echo "Downloading the verified full installer for $COMMIT..."
  /usr/bin/curl --fail --location --show-error --retry 5 --connect-timeout 30 \
    --proto '=https' --tlsv1.2 --output "$KIT_PATH.download" "$URL/$KIT_NAME"
  printf '%s  %s\n' "$KIT_SHA256" "$KIT_PATH.download" | /usr/bin/shasum -a 256 -c - ||
    fail "Kit checksum mismatch. The rolling release may have changed; download its current full installer."
  /bin/mv "$KIT_PATH.download" "$KIT_PATH"
fi
/usr/bin/ditto -x -k "$KIT_PATH" "$WORK/kit"
echo "Downloading, checking and assembling the app with both Bonsai 2 models..."
SMARTAI_PORTABLE_CACHE="$DOWNLOADS" \
SMARTAI_PORTABLE_DEST="$WORK/staging" \
SMARTAI_PORTABLE_OFFLINE="${SMARTAI_FULL_OFFLINE:-0}" \
SMARTAI_PORTABLE_INSTALL_ONLY=1 \
  /bin/bash "$WORK/kit/__LAUNCHER__"
SOURCE="$WORK/staging/__SHORT_SHA__/SmartAI.app"
check_app "$SOURCE" || fail "The assembled app is missing its launcher, embedded runtime or a model."
printf '%s\n' "$COMMIT" > "$SOURCE/$STAMP"

# All downloads and verification finish before requesting permission to write
# /Applications. No dependency installation or global security changes occur.
if [ ! -d "$PARENT" ]; then /bin/mkdir -p "$PARENT" || fail "Cannot create application directory: $PARENT"; fi
if [ ! -w "$PARENT" ]; then
  USE_SUDO=1
  echo "macOS needs your administrator password to install into $PARENT."
  /usr/bin/sudo -v
fi
install_command /bin/mkdir "$PARENT/.SmartAI-install.lock" 2>/dev/null ||
  fail "Another installation is using $PARENT/.SmartAI-install.lock."
DEST_LOCK="$PARENT/.SmartAI-install.lock"
[ ! -L "$TARGET" ] || fail "The application target became a symbolic link."
# Do not replace an app while its model or learning worker is running.
if app_running; then
  fail "Quit SmartAI before replacing it, then rerun this installer; completed downloads will be reused."
fi
INCOMING="$PARENT/.SmartAI-incoming-__SHORT_SHA__-$$.app"
[ ! -e "$INCOMING" ] || fail "Staging target already exists: $INCOMING"
install_command /bin/mv "$SOURCE" "$INCOMING"
check_app "$INCOMING" || fail "The app did not survive the move to the Applications volume."
if [ -e "$TARGET" ]; then
  [ -d "$TARGET" ] || fail "The target exists and is not an app directory."
  BACKUP="$PARENT/SmartAI.backup-$(/bin/date +%Y%m%d-%H%M%S)-$$.app"
  install_command /bin/mv "$TARGET" "$BACKUP"
fi
if ! install_command /bin/mv "$INCOMING" "$TARGET"; then
  if [ -n "$BACKUP" ] && [ ! -e "$TARGET" ]; then install_command /bin/mv "$BACKUP" "$TARGET"; fi
  fail "Final installation move failed; the previous app was retained."
fi
INCOMING=""
if ! check_app "$TARGET"; then
  REJECTED="$PARENT/SmartAI.incomplete-__SHORT_SHA__-$$.app"
  install_command /bin/mv "$TARGET" "$REJECTED"
  if [ -n "$BACKUP" ]; then install_command /bin/mv "$BACKUP" "$TARGET"; fi
  fail "Final app verification failed. The rejected copy is at $REJECTED."
fi
PUBLISHED=1

echo "Removing the verified archive, split parts and temporary installer files..."
# DOWNLOADS is an owned, dedicated subdirectory; never delete CACHE_ROOT or
# SmartAI_Data, ~/.smartai, old app backups, user models or chat/learning data.
[ "$(/bin/cat "$DOWNLOADS/.full-installer-owner")" = "$COMMIT" ]
/bin/rm -rf "$DOWNLOADS"
echo "Installed: $TARGET"
echo "Both Bonsai 2 model snapshots and the full runtime are installed."
if [ -n "$BACKUP" ]; then echo "Previous app backup: $BACKUP"; fi
if [ "${SMARTAI_FULL_INSTALL_ONLY:-0}" != 1 ]; then /usr/bin/open "$TARGET"; fi
'''


def build(kit_path, output_dir, *, repo, tag, commit):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid source repository")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("Invalid release tag")
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Missing exact tested commit SHA")
    kit_path = Path(kit_path)
    verify(kit_path, repo=repo, tag=tag, commit=commit)
    values = {"COMMIT": commit, "SHORT_SHA": commit[:12], "REPO": repo,
              "TAG": tag, "KIT": KIT, "LAUNCHER": LAUNCHER,
              "KIT_SHA256": hashlib.sha256(kit_path.read_bytes()).hexdigest()}
    script = TEMPLATE
    for key, value in values.items():
        script = script.replace(f"__{key}__", value)
    if re.search(r"__[A-Z_]+__", script):
        raise ValueError("Unexpanded installer placeholder")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / INSTALLER
    dest.write_text(script, encoding="utf-8", newline="\n")
    dest.chmod(0o755)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    dest.with_name(INSTALLER + ".sha256").write_text(f"{digest}  {INSTALLER}\n", encoding="ascii")
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", default="dist/" + KIT)
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--repo", default="Eabusham2/smart-ai-studio")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    path = build(args.kit, args.output_dir, repo=args.repo, tag=args.tag, commit=args.commit)
    print("VERIFIED_FULL_MAC_INSTALLER", path, path.stat().st_size)


if __name__ == "__main__":
    main()
