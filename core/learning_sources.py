"""Source ingestion for /learn.

Keeps file/folder parsing separate from the existing parameter-update path.
Text extraction is deterministic; media files are only classified here and are not
pretended to be understood by a text-only controller.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

TEXT_SUFFIXES = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".ini", ".cfg", ".toml",
    ".yaml", ".yml", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".c", ".cc", ".cpp", ".h",
    ".hpp", ".rs", ".go", ".swift", ".kt", ".kts", ".sh", ".zsh", ".ps1",
    ".sql", ".css", ".scss",
}
STRUCTURED_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".parquet"}
DOCUMENT_SUFFIXES = {".pdf", ".docx"}
MEDIA_SUFFIXES = {
    "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"},
    "video": {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"},
    "audio": {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus"},
}
SUPPORTED_TEXT_SUFFIXES = TEXT_SUFFIXES | STRUCTURED_SUFFIXES | DOCUMENT_SUFFIXES


def media_kind(path: Path) -> str | None:
    suffix = path.suffix.lower()
    for kind, suffixes in MEDIA_SUFFIXES.items():
        if suffix in suffixes:
            return kind
    return None


def _iter_source_files(source: Path, *, exclude=()) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    excluded = {Path(path).resolve() for path in exclude}
    skip = {".git", ".venv", "__pycache__", "node_modules", ".pytest_cache"}
    for directory, folders, files in os.walk(source, followlinks=False):
        folders[:] = sorted(name for name in folders if name not in skip and Path(directory) / name not in excluded
                            and not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if not path.is_symlink() and path.is_file():
                yield path


def learning_media_kinds(source):
    """Report detected modalities in a mixed learning folder without retaining a path list."""
    root = Path(source).expanduser().resolve(strict=True)
    detected = set()
    for file in _iter_source_files(root):
        kind = media_kind(file)
        if kind:
            detected.add(kind)
        if len(detected) == len(MEDIA_SUFFIXES):
            break
    return [kind for kind in MEDIA_SUFFIXES if kind in detected]


def inspect_learning_source(source: str | os.PathLike[str]) -> Dict[str, Any]:
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Learning source does not exist: {path}")
    text_files: List[str] = []
    media_files: Dict[str, List[str]] = {"image": [], "video": [], "audio": []}
    ignored: List[str] = []
    for file in _iter_source_files(path):
        suffix = file.suffix.lower()
        kind = media_kind(file)
        if kind:
            media_files[kind].append(str(file))
        elif suffix in SUPPORTED_TEXT_SUFFIXES:
            text_files.append(str(file))
        else:
            ignored.append(str(file))
    return {
        "path": str(path),
        "is_folder": path.is_dir(),
        "text_files": text_files,
        "media_files": media_files,
        "media_kinds": [kind for kind, files in media_files.items() if files],
        "ignored_files": ignored,
    }


def _clip(text: str, max_chars: int | None) -> str:
    return text if max_chars is None or len(text) <= max_chars else text[:max_chars] + "\n[...truncated...]"


def _extract_json(path: Path) -> str:
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _extract_jsonl(path: Path) -> str:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"raw": line})
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)


def _extract_delimited(path: Path, delimiter: str) -> str:
    out: List[str] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            out.append(delimiter.join(str(cell) for cell in row))
    return "\n".join(out)


def _extract_parquet(path: Path) -> str:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet /learn requires pyarrow") from exc
    table = pq.read_table(path)
    rows = table.to_pylist()
    return "\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows)


def _extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError("PDF /learn requires pypdf") from exc
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _extract_docx(path: Path) -> str:
    try:
        from docx import Document
    except Exception as exc:
        raise RuntimeError("DOCX /learn requires python-docx") from exc
    document = Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def extract_text_file(path: Path, max_chars: int | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        text = _extract_json(path)
    elif suffix == ".jsonl":
        text = _extract_jsonl(path)
    elif suffix == ".csv":
        text = _extract_delimited(path, ",")
    elif suffix == ".tsv":
        text = _extract_delimited(path, "\t")
    elif suffix == ".parquet":
        text = _extract_parquet(path)
    elif suffix == ".pdf":
        text = _extract_pdf(path)
    elif suffix == ".docx":
        text = _extract_docx(path)
    elif suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        return ""
    return _clip(text, max_chars)


def extract_learning_text(
    source: str | os.PathLike[str],
    *,
    max_total_chars: int | None = None,
    max_chars_per_file: int | None = None,
) -> Dict[str, Any]:
    inspection = inspect_learning_source(source)
    chunks: List[str] = []
    total = 0
    errors: List[str] = []
    for raw in inspection["text_files"]:
        path = Path(raw)
        try:
            text = extract_text_file(path, max_chars=max_chars_per_file)
        except Exception as exc:
            errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if not text.strip():
            continue
        if max_total_chars is not None:
            remaining = max_total_chars - total
            if remaining <= 0:
                break
            text = text[:remaining]
        chunks.append(f"===== SOURCE FILE: {path.name} =====\n{text}")
        total += len(text)
    return {
        **inspection,
        "text": "\n\n".join(chunks),
        "text_chars": total,
        "extraction_errors": errors,
    }


def iter_learning_files(source, *, exclude=()):
    """Walk a stable ordered directory one level at a time. Skip dependency and VCS trees."""
    root = Path(source).expanduser().resolve(strict=True)
    if root.is_file():
        if root.suffix.lower() not in SUPPORTED_TEXT_SUFFIXES:
            raise ValueError("Unsupported text-learning file: " + root.name)
        yield root
        return
    for path in _iter_source_files(root, exclude=exclude):
        if path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES:
            yield path


def source_snapshot(source, cancel_event=None, *, exclude=()):
    """Identify the precise file set, order and versions without reading payloads into RAM."""
    import hashlib
    from core.learning_recall import check_cancel

    signature = hashlib.sha256()
    count = 0
    for path in iter_learning_files(source, exclude=exclude):
        check_cancel(cancel_event)
        st = path.stat()
        signature.update(json.dumps([str(path), st.st_dev, st.st_ino, st.st_size,
                                    st.st_mtime_ns, st.st_ctime_ns]).encode())
        count += 1
    if not count:
        raise ValueError("Source contains no supported text documents")
    return signature.hexdigest(), count


def _document_fragments(path):
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        with path.open("rb") as handle:
            for page in PdfReader(handle).pages:
                yield page.extract_text() or ""
                yield "\n"
    elif path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
            yield json.dumps(batch.to_pylist()[0], ensure_ascii=False, default=str) + "\n"
    else:
        import xml.sax
        import zipfile
        from collections import deque

        class DocumentText(xml.sax.handler.ContentHandler):
            def __init__(self):
                super().__init__()
                self.in_text = False
                self.pieces = deque()

            def startElement(self, name, attrs):
                if name == "w:t":
                    self.in_text = True
                elif name in ("w:tab", "w:br"):
                    self.pieces.append("\t" if name == "w:tab" else "\n")

            def characters(self, value):
                if self.in_text:
                    self.pieces.append(value)

            def endElement(self, name):
                if name == "w:t":
                    self.in_text = False
                elif name == "w:p":
                    self.pieces.append("\n")

        parser = xml.sax.make_parser()
        handler = DocumentText()
        parser.setContentHandler(handler)
        with zipfile.ZipFile(path) as archive, archive.open("word/document.xml") as stream:
            while block := stream.read(8192):
                parser.feed(block)
                while handler.pieces:
                    yield handler.pieces.popleft()
            parser.close()
            yield from handler.pieces


def iter_file_positions(path, chunk_chars, *, first=0, seek=None):
    """Contiguous, complete Unicode chunks. Structured text retains its original syntax."""
    if path.suffix.lower() not in (".pdf", ".parquet", ".docx"):
        with path.open("r", encoding="utf-8", errors="strict", newline="") as stream:
            if seek is not None:
                stream.seek(seek)
            while chunk := stream.read(chunk_chars):
                yield chunk, stream.tell()
        return
    pending = ""
    index = 0
    for text in _document_fragments(path):
        cursor = 0
        while cursor < len(text):
            take = min(len(text) - cursor, chunk_chars - len(pending))
            pending += text[cursor:cursor + take]
            cursor += take
            if len(pending) == chunk_chars:
                if index >= first:
                    yield pending, None
                index += 1
                pending = ""
    if pending and index >= first:
        yield pending, None


def iter_file_chunks(path, chunk_chars):
    for chunk, _ in iter_file_positions(path, chunk_chars):
        yield chunk
