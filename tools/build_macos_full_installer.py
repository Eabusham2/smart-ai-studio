#!/usr/bin/env python3
"""Package an executable Mac installer that pins the canonical portable ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import zipfile

INSTALLER = "Install-SmartAI-macOS-Full.command"
INSTALLER_ZIP = "Install-SmartAI-macOS-Full.zip"
ARCHIVE = "SmartAI-macOS-arm64-portable.zip"
RECEIPT = "SmartAI-macOS-arm64.receipt.json"

TEMPLATE = r'''#!/bin/bash
set -euo pipefail

# Full Apple Silicon install with the bundled app runtime and dependencies.
# This file pins the app archive digest; a changing rolling release fails closed.
COMMIT="__COMMIT__"
ARCHIVE_SHA256="__ARCHIVE_SHA256__"
ARCHIVE_NAME="__ARCHIVE__"
URL="https://github.com/__REPO__/releases/download/__TAG__"
SYSTEM_APP="/Applications/SmartAI.app"
TARGET="${SMARTAI_FULL_TARGET-}"
CACHE_ROOT="${SMARTAI_FULL_CACHE:-$HOME/Library/Caches/SmartAI/Full-__SHORT_SHA__}"
DOWNLOADS="$CACHE_ROOT/app-downloads-__SHORT_SHA__"
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
# Keep the existing app parent: it also determines where SmartAI finds saved
# chats, models and learning data. Only fresh installs use the no-admin default.
if [ "${SMARTAI_FULL_TARGET+x}" != x ]; then
  CANDIDATES=()
  CANDIDATE_COUNT=0
  for candidate in "$HOME/Applications/SmartAI.app" "$SYSTEM_APP" \
      "$HOME"/Applications/SmartAI-Portable/*/SmartAI.app; do
    if [ -e "$candidate" ] || [ -L "$candidate" ]; then
      CANDIDATES[$CANDIDATE_COUNT]="$candidate"
      CANDIDATE_COUNT=$((CANDIDATE_COUNT + 1))
    fi
  done
  case "$CANDIDATE_COUNT" in
    0) TARGET="$HOME/Applications/SmartAI.app" ;;
    1) TARGET="${CANDIDATES[0]}" ;;
    *)
      CHOICE=""
      if [ -x /usr/bin/osascript ]; then
        CHOICE=$(/usr/bin/osascript - "${CANDIDATES[@]}" <<'SMARTAI_CHOOSE_APP'
on run appPaths
  set chosenApp to choose from list appPaths with title "Update Smart AI Studio" with prompt "Choose the existing installation to update. Its saved-data location will stay the same." OK button name "Update This Copy" cancel button name "Cancel" multiple selections allowed false empty selection allowed false
  if chosenApp is false then return ""
  return item 1 of chosenApp
end run
SMARTAI_CHOOSE_APP
        ) || CHOICE=""
      fi
      for candidate in "${CANDIDATES[@]}"; do
        if [ "$CHOICE" = "$candidate" ]; then TARGET="$candidate"; break; fi
      done
      if [ -z "$TARGET" ]; then
        echo 'Multiple installations were found. No app or saved data was changed.' >&2
        echo 'Rerun the installer and select a copy, or run the command for its exact location:' >&2
        for candidate in "${CANDIDATES[@]}"; do
          printf '  SMARTAI_FULL_TARGET=%q /bin/bash %q\n' "$candidate" "$0" >&2
        done
        fail 'No recognized installation was selected.'
      fi
      ;;
  esac
fi
case "$TARGET" in
  /*/SmartAI.app) ;;
  *) fail "SMARTAI_FULL_TARGET must be an absolute path ending in /SmartAI.app." ;;
esac
[ ! -L "$TARGET" ] || fail "The target is a symbolic link; choose an ordinary app path."
PARENT="$(/usr/bin/dirname "$TARGET")"

check_app() {
  local app="$1" code="$1/Contents/Resources/app"
  [ -f "$app/Contents/Info.plist" ] && [ -x "$app/Contents/MacOS/applet" ] &&
  [ -f "$code/app_gui.py" ] && [ -f "$code/launch_studio.sh" ] &&
  [ -x "$code/runtime/python/bin/python3" ] &&
  [ -f "$code/runtime/python/.smartai-deps-ready" ]
}

app_running() {
  /bin/ps ax -o command= | /usr/bin/grep -F -- "$TARGET/Contents/Resources/app/" |
    /usr/bin/grep -v -E '(^|[[:space:]/])grep[[:space:]]' >/dev/null
}

if check_app "$TARGET" && [ -f "$TARGET/$STAMP" ] && [ "$(/bin/cat "$TARGET/$STAMP")" = "$COMMIT" ]; then
  echo "This release and its runtime are already installed: $TARGET"
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
# only to the exact release's download cache, never to CACHE_ROOT itself.
if [ -e "$DOWNLOADS" ]; then
  [ -f "$DOWNLOADS/.full-installer-owner" ] &&
  [ "$(/bin/cat "$DOWNLOADS/.full-installer-owner")" = "$COMMIT" ] ||
    fail "Download directory has no matching ownership marker: $DOWNLOADS"
else
  /bin/mkdir "$DOWNLOADS"
  printf '%s\n' "$COMMIT" > "$DOWNLOADS/.full-installer-owner"
fi

ARCHIVE_PATH="$DOWNLOADS/$ARCHIVE_NAME"
verify_archive() { printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE_PATH" | /usr/bin/shasum -a 256 -c - >/dev/null 2>&1; }
if ! verify_archive; then
  [ "${SMARTAI_FULL_OFFLINE:-0}" != 1 ] || fail "Offline mode: the pinned app archive is missing or corrupt."
  echo "Downloading Smart AI Studio and its complete runtime..."
  /usr/bin/curl --fail --location --show-error --retry 5 --connect-timeout 30 \
    --proto '=https' --tlsv1.2 --output "$ARCHIVE_PATH.download" "$URL/$ARCHIVE_NAME"
  printf '%s  %s\n' "$ARCHIVE_SHA256" "$ARCHIVE_PATH.download" | /usr/bin/shasum -a 256 -c - ||
    fail "App checksum mismatch. The rolling release may have changed; download its current full installer."
  /bin/mv "$ARCHIVE_PATH.download" "$ARCHIVE_PATH"
fi
echo "Extracting the verified app and embedded runtime..."
/usr/bin/ditto -x -k "$ARCHIVE_PATH" "$WORK/staging"
SOURCE="$WORK/staging/SmartAI.app"
check_app "$SOURCE" || fail "The assembled app is missing its launcher or required bundle files."
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

echo "Removing the verified download and temporary installer files..."
# DOWNLOADS is an owned, dedicated subdirectory; never delete CACHE_ROOT or
# SmartAI_Data, ~/.smartai, old app backups, user models or chat/learning data.
[ "$(/bin/cat "$DOWNLOADS/.full-installer-owner")" = "$COMMIT" ]
/bin/rm -rf "$DOWNLOADS"
echo "Installed: $TARGET"
echo "The app and full runtime are installed. Download your chosen model inside Smart AI Studio."
if [ -n "$BACKUP" ]; then echo "Previous app backup: $BACKUP"; fi
if [ "${SMARTAI_FULL_INSTALL_ONLY:-0}" != 1 ]; then /usr/bin/open "$TARGET"; fi
'''


def write_installer_zip(script, output_dir):
    """Keep the execute bit through a browser download and Mac extraction."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / INSTALLER_ZIP
    entry = zipfile.ZipInfo(INSTALLER)
    entry.create_system = 3  # Unix permissions, even when built on Windows.
    entry.external_attr = (stat.S_IFREG | 0o755) << 16
    entry.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(dest, "w") as archive:
        archive.writestr(entry, script.encode("utf-8"))
    return dest


def build(receipt_path, output_dir, *, repo, tag, commit):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError("Invalid source repository")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("Invalid release tag")
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Missing exact tested commit SHA")
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("schema") != 1 or receipt.get("target") != "macOS-arm64"
            or receipt.get("commit") != commit or receipt.get("gui_smoke") != "passed"
            or receipt.get("setup_smoke") != "passed"):
        raise ValueError("Mac package receipt does not identify the verified release")
    rows = receipt.get("files")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Invalid Mac package receipt")
    archives = [row for row in rows if row.get("name") == ARCHIVE]
    if len(archives) != 1:
        raise ValueError("The Mac installer requires one complete portable ZIP")
    archive = archives[0]
    digest = archive.get("sha256", "")
    size = archive.get("size")
    if (not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest)
            or not isinstance(size, int) or isinstance(size, bool) or not 0 < size < 2**31):
        raise ValueError("Invalid portable archive size or SHA256")
    path = receipt_path.parent / ARCHIVE
    actual = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            actual.update(chunk)
    if path.stat().st_size != size or actual.hexdigest() != digest:
        raise ValueError("Portable archive does not match its verified receipt")
    values = {"COMMIT": commit, "SHORT_SHA": commit[:12], "REPO": repo,
              "TAG": tag, "ARCHIVE": ARCHIVE, "ARCHIVE_SHA256": digest}
    script = TEMPLATE
    for key, value in values.items():
        script = script.replace(f"__{key}__", value)
    if re.search(r"__[A-Z_]+__", script):
        raise ValueError("Unexpanded installer placeholder")
    return write_installer_zip(script, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default="dist/" + RECEIPT)
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--repo", default="Eabusham2/smart-ai-studio")
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    path = build(args.receipt, args.output_dir, repo=args.repo, tag=args.tag, commit=args.commit)
    print("VERIFIED_FULL_MAC_INSTALLER", path, path.stat().st_size)


if __name__ == "__main__":
    main()
