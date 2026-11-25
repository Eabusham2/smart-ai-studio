"""Verified GitHub release updates for the existing packaged desktop app.

All methods are blocking and are called by the UI's background worker.  No
third-party packages, model downloads, or changes to the running app are needed
to check or stage an update.  Publication uses a rolling ``latest`` tag, so the
release receipt and Git history identify a build, not the tag's display text.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import tempfile
from typing import Callable
import urllib.error
import urllib.parse
import urllib.request
import zipfile


REPOSITORY = "Eabusham2/smart-ai-studio"
API_ROOT = "https://api.github.com/repos/" + REPOSITORY
BUILD_FILE = ".smartai-build.json"
_STAGING_RECORD = ".smartai-download.json"
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES = 16 * 1024**3
MAX_EXPANDED_BYTES = 48 * 1024**3
MAX_ARCHIVE_FILES = 250_000
_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
Progress = Callable[[str, int, int], None]


class UpdateError(RuntimeError):
    """An actionable update error which leaves the running app unchanged."""


@dataclass(frozen=True)
class Download:
    name: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ReleaseUpdate:
    tag: str
    commit: str
    html_url: str
    target: str
    package_kind: str
    archive_name: str
    archive_sha256: str
    downloads: tuple[Download, ...]

    @property
    def total_bytes(self) -> int:
        return sum(part.size for part in self.downloads)


@dataclass(frozen=True)
class UpdateCheck:
    status: str
    message: str
    release: ReleaseUpdate | None = None


@dataclass(frozen=True)
class PreparedUpdate:
    release: ReleaseUpdate
    staging_dir: Path
    staged_root: Path


def _target_name() -> str:
    machine = platform.machine().lower()
    architecture = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    system = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}.get(platform.system())
    if not system:
        raise UpdateError("In-app updates are not available on this operating system.")
    return system + "-" + architecture


def _read_build(path: Path) -> dict:
    try:
        if path.stat().st_size > MAX_METADATA_BYTES:
            return {}
        result = json.loads(path.read_text(encoding="utf-8"))
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError):
        return {}


def _layout(code: Path, target: str) -> tuple[Path, str]:
    relative = ("Contents/Resources/app" if target.startswith("macOS-") else
                "SmartAI.AppDir/usr/bin" if target.startswith("Linux-") else ".")
    if relative == ".":
        return code, relative
    suffix = PurePosixPath(relative).parts
    if tuple(code.parts[-len(suffix):]) != suffix:
        raise UpdateError("The installed app layout does not match its build information. Reinstall a complete release package.")
    return code.parents[len(suffix) - 1], relative


def _asset_name(value) -> str:
    if (not isinstance(value, str) or not value or value in {".", ".."}
            or any(character in value for character in "/\\:\x00\r\n")):
        raise UpdateError("The release contains an invalid asset name.")
    return value


def _asset_url(value: str, tag: str, name: str) -> str:
    parts = urllib.parse.urlsplit(value)
    expected = f"/{REPOSITORY}/releases/download/{tag}/{name}"
    if (parts.scheme != "https" or parts.netloc.lower() != "github.com"
            or parts.query or parts.fragment
            or urllib.parse.unquote(parts.path).lower() != expected.lower()):
        raise UpdateError("The release download is not an asset of this GitHub repository.")
    return value


def _asset_digest(asset: dict) -> str | None:
    digest = asset.get("digest")
    if digest is None:
        return None  # Older GitHub assets have no digest; receipts still hash packages.
    if not isinstance(digest, str) or not digest.startswith("sha256:") or not _SHA256.fullmatch(digest[7:]):
        raise UpdateError("The release asset has an invalid SHA-256 digest.")
    return digest[7:].lower()


def _verified_download(asset: dict, row: dict, tag: str) -> Download:
    name = _asset_name(row.get("name"))
    expected_hash = row.get("sha256", "")
    size = row.get("size")
    if (not isinstance(expected_hash, str) or not _SHA256.fullmatch(expected_hash)
            or not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_ARCHIVE_BYTES
            or asset.get("name") != name or asset.get("size") != size):
        raise UpdateError("Release assets and their verification receipt do not agree. Try again after publication finishes.")
    digest = _asset_digest(asset)
    if digest and digest != expected_hash.lower():
        raise UpdateError("Release asset checksums disagree. Try again after publication finishes.")
    return Download(name, _asset_url(asset.get("browser_download_url", ""), tag, name), size, expected_hash.lower())


class AppUpdater:
    def __init__(self, app_root: Path | str | None = None):
        self.app_root = Path(app_root or Path(__file__).resolve().parents[1]).resolve()
        self.build = _read_build(self.app_root / BUILD_FILE)
        self.install_root: Path | None = None
        self.code_relative = "."
        self._install_error = ""
        if self.build:
            try:
                self.install_root, self.code_relative = _layout(self.app_root, str(self.build.get("target", "")))
            except UpdateError as exc:
                self._install_error = str(exc)

    @property
    def current_label(self) -> str:
        commit = str(self.build.get("commit", ""))
        tag = str(self.build.get("tag", ""))
        return f"{tag or 'Installed build'} ({commit[:12]})" if _SHA.fullmatch(commit) else "Source / unversioned build"

    def _open(self, url: str):
        request = urllib.request.Request(url, headers={
            "User-Agent": "SmartAI-Studio-Updater", "Accept": "application/vnd.github+json" if url.startswith(API_ROOT + "/") else "application/octet-stream",
            "X-GitHub-Api-Version": "2022-11-28", "Accept-Encoding": "identity",
        })
        try:
            return urllib.request.urlopen(request, timeout=30)
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise UpdateError("GitHub is limiting update checks. Try again later.") from exc
            if exc.code == 404:
                raise UpdateError("The release or update file is not available yet. Try again later.") from exc
            raise UpdateError(f"GitHub could not complete the update request (HTTP {exc.code}).") from exc
        except (OSError, urllib.error.URLError) as exc:
            raise UpdateError("Could not reach GitHub. Check your connection and try again.") from exc

    def _json(self, url: str, *, asset: dict | None = None) -> dict:
        with self._open(url) as response:
            raw = response.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise UpdateError("Update metadata exceeds the supported size.")
        if asset:
            if len(raw) != asset.get("size"):
                raise UpdateError("The release metadata download was incomplete.")
            digest = _asset_digest(asset)
            if digest and hashlib.sha256(raw).hexdigest() != digest:
                raise UpdateError("Release metadata failed checksum verification.")
        try:
            result = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise UpdateError("GitHub returned invalid update metadata.") from exc
        if not isinstance(result, dict):
            raise UpdateError("GitHub returned invalid update metadata.")
        return result

    def _tag_commit(self, tag: str) -> str:
        ref = self._json(API_ROOT + "/git/ref/tags/" + urllib.parse.quote(tag, safe=""))
        obj = ref.get("object", {})
        for _ in range(5):
            if not isinstance(obj, dict) or not _SHA.fullmatch(str(obj.get("sha", ""))):
                break
            if obj.get("type") == "commit":
                return obj["sha"].lower()
            if obj.get("type") != "tag":
                break
            obj = self._json(API_ROOT + "/git/tags/" + obj["sha"]).get("object", {})
        raise UpdateError("The release tag does not resolve to a valid source commit.")

    def check(self) -> UpdateCheck:
        if not self.build:
            return UpdateCheck("unsupported", "In-app installation is available in packaged releases. This source checkout can continue to be updated with Git.")
        if self._install_error:
            raise UpdateError(self._install_error)
        target = self.build.get("target")
        kind = self.build.get("package_kind")
        current = str(self.build.get("commit", ""))
        if (self.build.get("schema") != 1 or self.build.get("repository", "").lower() != REPOSITORY.lower()
                or not _SHA.fullmatch(current) or kind not in {"portable", "setup"}):
            raise UpdateError("The installed build information is incomplete. Reinstall a complete release package.")
        if target != _target_name():
            return UpdateCheck("unsupported", "This app package does not match the current operating system and architecture.")
        release = self._json(API_ROOT + "/releases/latest")
        if release.get("draft") or release.get("prerelease"):
            return UpdateCheck("current", "No newer stable release is available.")
        tag = release.get("tag_name")
        if not isinstance(tag, str) or not tag or len(tag) > 200:
            raise UpdateError("The release has no valid tag.")
        commit = self._tag_commit(tag)
        if commit == current.lower():
            return UpdateCheck("current", "You are running the latest release.")
        # A rolling release can lag a locally installed build. Never downgrade
        # on the basis of a different SHA or a release's original publish date.
        comparison = self._json(API_ROOT + "/compare/" + current + "..." + commit)
        relation = comparison.get("status")
        if relation in {"behind", "identical"}:
            return UpdateCheck("ahead", "Your installed build is already at or ahead of the published release.")
        if relation != "ahead":
            return UpdateCheck("unsupported", "This build and the release have different histories. Automatic replacement is unavailable.")
        assets = release.get("assets")
        if not isinstance(assets, list):
            raise UpdateError("The release has no downloadable app packages.")
        by_name = {}
        for asset in assets:
            if not isinstance(asset, dict):
                raise UpdateError("The release has invalid asset metadata.")
            name = _asset_name(asset.get("name"))
            if name in by_name:
                raise UpdateError("The release has duplicate asset names.")
            by_name[name] = asset
        receipt_name = f"SmartAI-{target}.receipt.json"
        receipt_asset = by_name.get(receipt_name)
        if not receipt_asset:
            raise UpdateError("The verified package for this platform is not published yet. Try again later.")
        receipt = self._json(_asset_url(receipt_asset.get("browser_download_url", ""), tag, receipt_name), asset=receipt_asset)
        if receipt.get("schema") != 1 or receipt.get("target") != target or receipt.get("commit") != commit:
            raise UpdateError("Release publication is still in progress: the package and tag identify different builds. Try again later.")
        rows = receipt.get("files")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise UpdateError("The release package receipt is invalid.")
        recorded = {}
        for row in rows:
            name = _asset_name(row.get("name"))
            if name in recorded:
                raise UpdateError("The release receipt contains duplicate filenames.")
            recorded[name] = row
        suffix = "portable" if kind == "portable" else "Setup"
        archive_name = f"SmartAI-{target}-{suffix}.zip"
        if archive_name in recorded and archive_name in by_name:
            part = _verified_download(by_name[archive_name], recorded[archive_name], tag)
            downloads, archive_hash = (part,), part.sha256
        else:
            manifest_name = archive_name + ".parts.json"
            if manifest_name not in recorded or manifest_name not in by_name:
                raise UpdateError("The verified update package is not complete yet. Try again later.")
            manifest_part = _verified_download(by_name[manifest_name], recorded[manifest_name], tag)
            # The receipt also authenticates metadata from older assets that
            # predate GitHub's own digest field.
            manifest_bytes = self._metadata_bytes(manifest_part)
            try:
                manifest = json.loads(manifest_bytes)
            except (ValueError, UnicodeError) as exc:
                raise UpdateError("The split package manifest is invalid.") from exc
            if (not isinstance(manifest, dict) or manifest.get("archive") != archive_name
                    or not _SHA256.fullmatch(str(manifest.get("sha256", "")))):
                raise UpdateError("The split package manifest is invalid.")
            parts = manifest.get("parts")
            if not isinstance(parts, list) or not 1 <= len(parts) <= 100:
                raise UpdateError("The split package manifest has an invalid part count.")
            downloads_list = []
            for index, row in enumerate(parts):
                name = archive_name + f".part{index:03d}"
                if (not isinstance(row, dict) or row.get("name") != name or name not in by_name
                        or recorded.get(name) != row):
                    raise UpdateError("The release's split package is incomplete or inconsistent.")
                downloads_list.append(_verified_download(by_name[name], row, tag))
            downloads, archive_hash = tuple(downloads_list), manifest["sha256"].lower()
        update = ReleaseUpdate(tag, commit, "https://github.com/" + REPOSITORY + "/releases/tag/" + urllib.parse.quote(tag, safe=""),
                               target, kind, archive_name, archive_hash, downloads)
        if update.total_bytes > MAX_ARCHIVE_BYTES:
            raise UpdateError("The update package exceeds the supported size.")
        return UpdateCheck("available", f"Update available: {tag} ({commit[:12]}).", update)

    def _metadata_bytes(self, download: Download) -> bytes:
        if download.size > MAX_METADATA_BYTES:
            raise UpdateError("The release metadata exceeds the supported size.")
        with self._open(download.url) as response:
            raw = response.read(download.size + 1)
        if len(raw) != download.size or hashlib.sha256(raw).hexdigest() != download.sha256:
            raise UpdateError("The release metadata failed checksum verification.")
        return raw

    def prepare(self, release: ReleaseUpdate, progress: Progress | None = None) -> PreparedUpdate:
        progress = progress or (lambda message, done, total: None)
        if self.install_root is None or not self.build:
            raise UpdateError("A complete packaged installation is required for in-app updates.")
        if release.target != self.build.get("target") or release.package_kind != self.build.get("package_kind"):
            raise UpdateError("The update does not match this installation.")
        if (self.install_root / ".git").exists() or (self.app_root / ".git").exists():
            raise UpdateError("Use Git to update a source checkout; in-app installation will not replace it.")
        # Refresh the selected release before a potentially large download.
        # The receipt may have changed since the user opened the dialog.
        fresh = self.check()
        if fresh.status != "available" or fresh.release != release:
            raise UpdateError("The published update changed. Check for updates again before downloading.")
        parent = self.install_root.parent
        required = release.total_bytes + 128 * 1024**2
        if shutil.disk_usage(parent).free < required:
            raise UpdateError("There is not enough free space to download this update.")
        try:
            staging = Path(tempfile.mkdtemp(prefix=".smartai-update-", dir=parent))
        except OSError as exc:
            raise UpdateError("The app's folder is not writable. Move the complete app to a folder you own, then try again.") from exc
        try:
            (staging / _STAGING_RECORD).write_text(json.dumps({
                "repository": REPOSITORY, "commit": release.commit,
            }), encoding="utf-8")
            archive = staging / release.archive_name
            full_hash = hashlib.sha256()
            completed = 0
            with archive.open("xb") as output:
                for part in release.downloads:
                    digest, received = hashlib.sha256(), 0
                    with self._open(part.url) as response:
                        while True:
                            block = response.read(min(1024 * 1024, part.size - received + 1))
                            if not block:
                                break
                            received += len(block)
                            if received > part.size:
                                raise UpdateError("An update download exceeded its verified size.")
                            output.write(block)
                            digest.update(block)
                            full_hash.update(block)
                            progress("Downloading update", completed + received, release.total_bytes)
                    if received != part.size or digest.hexdigest() != part.sha256:
                        raise UpdateError("The update download was incomplete or failed SHA-256 verification. Your installed app is unchanged.")
                    completed += received
            if full_hash.hexdigest() != release.archive_sha256:
                raise UpdateError("The assembled update failed SHA-256 verification.")
            progress("Verifying and unpacking update", completed, release.total_bytes)
            staged_root = extract_update(archive, staging / "unpacked", release.target)
            new_code = staged_root / self.code_relative
            identity = _read_build(new_code / BUILD_FILE)
            for key, expected in (("repository", REPOSITORY), ("commit", release.commit), ("target", release.target), ("package_kind", release.package_kind)):
                if identity.get(key) != expected:
                    raise UpdateError("The downloaded app's build identity does not match the verified release.")
            if identity.get("schema") != 1:
                raise UpdateError("The downloaded app has unsupported build information.")
            if not (new_code / "bootstrap_runtime.py").is_file():
                raise UpdateError("The downloaded app has no supported launcher.")
            runtime = new_code / ("runtime/python/python.exe" if release.target.startswith("Windows-") else "runtime/python/bin/python3")
            if not runtime.is_file():
                raise UpdateError("The downloaded app is missing its Python runtime.")
            archive.unlink()
            progress("Update ready to install", completed, release.total_bytes)
            return PreparedUpdate(release, staging, staged_root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def discard(self, prepared: PreparedUpdate) -> None:
        """Remove an abandoned download, never a live app or helper transaction."""
        if self.install_root is None:
            return
        staging = prepared.staging_dir
        if (staging.is_symlink() or staging.resolve().parent != self.install_root.parent.resolve()
                or not staging.name.startswith(".smartai-update-")
                or prepared.staged_root.parent != staging / "unpacked"
                or _read_build(staging / _STAGING_RECORD) != {
                    "repository": REPOSITORY, "commit": prepared.release.commit,
                }):
            raise UpdateError("The abandoned update folder could not be identified safely.")
        shutil.rmtree(staging, ignore_errors=True)

    def install(self, prepared: PreparedUpdate):
        from core.update_installer import schedule_install
        if self.install_root is None or not prepared.staged_root.is_relative_to(prepared.staging_dir):
            raise UpdateError("The staged update is no longer available.")
        try:
            parent_pid = int(os.environ.get("SMARTAI_BOOTSTRAP_PID", "0")) or None
        except ValueError:
            parent_pid = None
        return schedule_install(self.install_root, prepared.staged_root, self.code_relative,
                                gui_pid=os.getpid(), bootstrap_pid=parent_pid)


def extract_update(archive: Path, destination: Path, target: str) -> Path:
    """Extract regular package files only, checking all paths before writing."""
    root_name = "SmartAI.app" if target.startswith("macOS-") else "SmartAI-Windows" if target.startswith("Windows-") else "SmartAI-Linux"
    destination = Path(destination)
    if destination.exists():
        raise UpdateError("The update extraction folder already exists.")
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise UpdateError("The update contains too many files.")
        expanded = sum(member.file_size for member in members)
        if expanded > MAX_EXPANDED_BYTES:
            raise UpdateError("The unpacked update exceeds the supported size.")
        if shutil.disk_usage(destination.parent).free < expanded + 128 * 1024**2:
            raise UpdateError("There is not enough free space to unpack this update.")
        paths = set()
        for member in members:
            # ZipInfo.filename has already normalized Windows separators and
            # truncated NUL bytes; validate the original archived spelling.
            name = member.orig_filename
            relative = PurePosixPath(name)
            mode = member.external_attr >> 16
            if (not name or "\\" in name or ":" in name or "\x00" in name
                    or relative.is_absolute() or ".." in relative.parts
                    or not relative.parts or relative.parts[0] != root_name
                    or stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}):
                raise UpdateError("The update archive contains an unsafe file path or link.")
            key = str(relative) if target.startswith("Linux-") else str(relative).casefold()
            if key in paths:
                raise UpdateError("The update archive contains duplicate file paths.")
            paths.add(key)
        destination.mkdir()
        for member in members:
            path = destination.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as incoming, path.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            mode = member.external_attr >> 16
            if mode & 0o777:
                path.chmod(mode & 0o777)
    return destination / root_name
