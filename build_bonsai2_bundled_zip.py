#!/usr/bin/env python3
"""Build the macOS release ZIP with both Bonsai-2 MLX snapshots without needing >14 GB free disk.

The logical archive is streamed into sub-2-GiB parts. In CI, each completed part can
be uploaded to the target GitHub Release immediately and deleted locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from huggingface_hub import HfApi, hf_hub_url


MODEL_REPOS = (
    "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit",
    "dealignai/Bonsai-2-27B-CRACK-Ternary-JANG",
)
DEFAULT_PART_BYTES = 1_900_000_000  # safely below GitHub's 2 GiB per-release-asset cap


class MultipartWriter:
    def __init__(
        self,
        out_dir: Path,
        archive_name: str,
        part_limit: int,
        upload_tag: Optional[str] = None,
    ):
        self.out_dir = out_dir
        self.archive_name = archive_name
        self.part_limit = int(part_limit)
        self.upload_tag = upload_tag
        self.total = 0
        self.part_index = 0
        self.part_size = 0
        self.fp = None
        self.part_path: Optional[Path] = None
        self.part_hash = None
        self.archive_hash = hashlib.sha256()
        self.parts: List[Dict[str, object]] = []
        self.closed = False
        self._open_part()

    def writable(self):
        return True

    def seekable(self):
        return False

    def tell(self):
        return self.total

    def seek(self, *_args, **_kwargs):
        raise OSError("multipart release stream is intentionally non-seekable")

    def flush(self):
        if self.fp:
            self.fp.flush()

    def _open_part(self):
        self.part_path = self.out_dir / f"{self.archive_name}.part{self.part_index:03d}"
        self.fp = open(self.part_path, "wb")
        self.part_hash = hashlib.sha256()
        self.part_size = 0

    def _finish_part(self):
        if not self.fp or not self.part_path or self.part_size <= 0:
            return
        self.fp.flush()
        self.fp.close()
        record = {
            "name": self.part_path.name,
            "size": self.part_size,
            "sha256": self.part_hash.hexdigest(),
        }
        self.parts.append(record)
        print(
            f"[bundle] completed {record['name']} "
            f"({self.part_size / (1024 ** 3):.2f} GiB)"
        )

        if self.upload_tag:
            subprocess.run(
                ["gh", "release", "upload", self.upload_tag, str(self.part_path), "--clobber"],
                check=True,
            )
            self.part_path.unlink(missing_ok=True)
            print(f"[bundle] uploaded {record['name']} and reclaimed local disk")

        self.fp = None
        self.part_path = None

    def write(self, data):
        if self.closed:
            raise ValueError("write to closed multipart stream")
        view = memoryview(data)
        original = len(view)
        while view:
            room = self.part_limit - self.part_size
            if room <= 0:
                self._finish_part()
                self.part_index += 1
                self._open_part()
                room = self.part_limit

            piece = view[:room]
            self.fp.write(piece)
            self.part_hash.update(piece)
            self.archive_hash.update(piece)
            n = len(piece)
            self.part_size += n
            self.total += n
            view = view[n:]
        return original

    def close(self):
        if self.closed:
            return
        self._finish_part()
        self.closed = True


def _zip_local_tree(zf: zipfile.ZipFile, root: Path):
    parent = root.parent
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        arcname = str(path.relative_to(parent))
        info = zipfile.ZipInfo.from_file(path, arcname)
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(path, "rb") as src, zf.open(info, "w", force_zip64=True) as dst:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)


def _repo_files(api: HfApi, repo_id: str):
    info = api.model_info(repo_id, files_metadata=True)
    files = []
    for sibling in info.siblings or []:
        name = getattr(sibling, "rfilename", None)
        if not name:
            continue
        files.append(
            {
                "name": name,
                "size": int(getattr(sibling, "size", 0) or 0),
            }
        )
    return str(info.sha or "main"), files


def _request(url: str):
    headers = {"User-Agent": "SmartAI-Studio-CI-Bundler/1.0"}
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _stream_remote_file(
    zf: zipfile.ZipFile,
    repo_id: str,
    revision: str,
    filename: str,
    expected_size: int,
):
    safe_repo = repo_id.replace("/", "--")
    arcname = (
        f"SmartAI.app/Contents/Resources/app/preloaded_models/"
        f"{safe_repo}/{filename}"
    )
    zi = zipfile.ZipInfo(arcname, date_time=time.localtime()[:6])
    zi.compress_type = zipfile.ZIP_STORED
    zi.external_attr = (stat.S_IFREG | 0o644) << 16
    if expected_size > 0:
        zi.file_size = expected_size

    url = hf_hub_url(repo_id=repo_id, filename=filename, revision=revision)
    print(f"[bundle] {repo_id}: {filename}")
    written = 0
    with urllib.request.urlopen(_request(url), timeout=300) as response:
        with zf.open(zi, "w", force_zip64=True) as dst:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
                written += len(chunk)

    if expected_size and written != expected_size:
        raise RuntimeError(
            f"{repo_id}/{filename}: expected {expected_size} bytes, got {written}"
        )


def _write_reassembler(out_dir: Path, archive_name: str, manifest_name: str) -> Path:
    path = out_dir / "Reassemble-SmartAI-macOS-arm64.command"
    text = f"""#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path(".")
manifest = json.loads((root / "{manifest_name}").read_text())
out = root / manifest["archive_name"]
whole = hashlib.sha256()
with out.open("wb") as dst:
    for part in manifest["parts"]:
        p = root / part["name"]
        if not p.is_file():
            raise SystemExit(f"Missing release part: {{p.name}}")
        h = hashlib.sha256()
        with p.open("rb") as src:
            while True:
                chunk = src.read(8 * 1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
                whole.update(chunk)
                dst.write(chunk)
        if h.hexdigest() != part["sha256"]:
            raise SystemExit(f"SHA256 mismatch: {{p.name}}")
if whole.hexdigest() != manifest["archive_sha256"]:
    raise SystemExit("Whole ZIP SHA256 mismatch")
print(f"Reassembled and verified: {{out}}")
PY
unzip -t "{archive_name}" >/dev/null
echo "ZIP integrity verified. Extract {archive_name} normally."
"""
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-bundle", default="dist/SmartAI.app")
    parser.add_argument("--output-dir", default="dist")
    parser.add_argument("--archive-name", default="SmartAI-macOS-arm64.zip")
    parser.add_argument("--part-bytes", type=int, default=DEFAULT_PART_BYTES)
    parser.add_argument("--upload-tag", default=None)
    args = parser.parse_args()

    app_bundle = Path(args.app_bundle).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not app_bundle.is_dir():
        raise SystemExit(f"Missing macOS app bundle: {app_bundle}")

    api = HfApi()
    repo_manifest = []
    repo_file_sets = []
    for repo_id in MODEL_REPOS:
        revision, files = _repo_files(api, repo_id)
        repo_file_sets.append((repo_id, revision, files))
        repo_manifest.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "files": len(files),
                "expected_bytes": sum(int(f["size"]) for f in files),
            }
        )

    writer = MultipartWriter(
        out_dir=out_dir,
        archive_name=args.archive_name,
        part_limit=args.part_bytes,
        upload_tag=args.upload_tag,
    )
    try:
        with zipfile.ZipFile(
            writer,
            mode="w",
            compression=zipfile.ZIP_STORED,
            allowZip64=True,
        ) as zf:
            print("[bundle] adding SmartAI.app")
            _zip_local_tree(zf, app_bundle)

            bundled_manifest = {
                "format": 1,
                "models": repo_manifest,
                "layout": "preloaded_models/<owner--repo>/",
            }
            zf.writestr(
                "SmartAI.app/Contents/Resources/app/preloaded_models/manifest.json",
                json.dumps(bundled_manifest, indent=2, sort_keys=True),
            )

            for repo_id, revision, files in repo_file_sets:
                for item in files:
                    _stream_remote_file(
                        zf,
                        repo_id=repo_id,
                        revision=revision,
                        filename=str(item["name"]),
                        expected_size=int(item["size"]),
                    )
    finally:
        writer.close()

    manifest_name = "SmartAI-macOS-arm64.bundle.json"
    external_manifest = {
        "format": 1,
        "archive_name": args.archive_name,
        "archive_size": writer.total,
        "archive_sha256": writer.archive_hash.hexdigest(),
        "part_limit": args.part_bytes,
        "parts": writer.parts,
        "models": repo_manifest,
    }
    manifest_path = out_dir / manifest_name
    manifest_path.write_text(
        json.dumps(external_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    reassembler = _write_reassembler(out_dir, args.archive_name, manifest_name)

    if args.upload_tag:
        subprocess.run(
            [
                "gh",
                "release",
                "upload",
                args.upload_tag,
                str(manifest_path),
                str(reassembler),
                "--clobber",
            ],
            check=True,
        )

    print(
        f"[bundle] complete: {writer.total / (1024 ** 3):.2f} GiB logical ZIP, "
        f"{len(writer.parts)} release parts"
    )


if __name__ == "__main__":
    main()
