"""Bounded, quality-preserving execution tuning and portable profile storage.

This module does not load models, change prompts, or mutate a running engine.
Adapters supply isolated measurements of one fixed synthetic workload and apply
the returned settings only after restoring their pre-calibration runtime state.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Mapping, Optional


_KNOB_LIMITS = {"n_threads": 1024, "n_threads_batch": 1024, "prefill_step_size": 8192}
_SCHEMA = 1
_MAX_CACHE_BYTES = 1024 * 1024
_MAX_PROFILES = 64
_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _settings(value: Mapping[str, Any]) -> Dict[str, int]:
    if not isinstance(value, Mapping) or len(value) > len(_KNOB_LIMITS):
        raise ValueError("Execution settings must be a bounded mapping")
    result = {}
    for name, number in value.items():
        if name not in _KNOB_LIMITS or type(number) is not int or not 1 <= number <= _KNOB_LIMITS[name]:
            raise ValueError(f"Unsupported execution setting: {name}")
        result[name] = number
    return result


@dataclass(frozen=True)
class SpeedSample:
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    output_digest: str

    def validate(self) -> None:
        if type(self.prompt_tokens) is not int or self.prompt_tokens < 1:
            raise ValueError("A speed sample needs real prompt token counts")
        if type(self.generated_tokens) is not int or self.generated_tokens < 2:
            raise ValueError("At least two generated tokens are required to time decoding")
        for value in (self.prefill_seconds, self.decode_seconds):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Sample timings must be finite and positive")
        if not isinstance(self.output_digest, str) or not 1 <= len(self.output_digest) <= 256:
            raise ValueError("Samples must identify the complete generated output")

    def rates(self) -> Dict[str, float]:
        self.validate()
        rates = {
            "prefill_tokens_per_second": self.prompt_tokens / self.prefill_seconds,
            # Adapters time first-token latency separately from later decoding.
            "decode_tokens_per_second": (self.generated_tokens - 1) / self.decode_seconds,
        }
        if not all(math.isfinite(x) and x > 0 for x in rates.values()):
            raise ValueError("Sample rates must be finite and positive")
        return rates


@dataclass
class SpeedResult:
    status: str
    settings: Dict[str, int]
    message: str
    gain_pct: float = 0.0
    baseline: Dict[str, Any] = field(default_factory=dict)
    selected: Dict[str, Any] = field(default_factory=dict)
    trials: List[Dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SpeedTuningStopped(Exception):
    """Adapters raise this at token boundaries to stop calibration safely."""

    def __init__(self, status: str, message: str):
        if status not in {"cancelled", "timeout", "memory_pressure"}:
            raise ValueError("Unknown calibration stop status")
        self.status = status
        super().__init__(message)


def _geometric(values: List[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _drift(a: Dict[str, float], b: Dict[str, float]) -> float:
    # Symmetric: switching the order of two runs cannot hide a slowdown.
    return 100.0 * max(max(a[key], b[key]) / min(a[key], b[key]) - 1.0 for key in a)


def tune_speed(
    baseline: Dict[str, int],
    candidates: List[Dict[str, int]],
    measure: Callable[[Dict[str, int]], SpeedSample],
    *,
    cancelled: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[str], None]] = None,
    budget_seconds: float = 90.0,
    rounds: int = 2,
    clock: Callable[[], float] = time.monotonic,
    memory_ok: Optional[Callable[[], bool]] = None,
) -> SpeedResult:
    """Compare execution settings using repeated interleaved baseline controls.

    Every candidate must improve the geometric mean of prefill/decode throughput
    in *every* round, beat a 3% / twice-noise / twice-drift margin, and avoid a
    >2% regression in either metric. Output digests and token counts must match.
    Exceptions, cancellation, pressure, and expired budgets discard all partial
    winners. The callback must also enforce its own token-boundary deadline;
    Python cannot interrupt a native inference call safely from this function.
    """
    initial = dict(baseline) if isinstance(baseline, Mapping) else {}
    trials: List[Dict[str, Any]] = []
    baseline_rates: Dict[str, float] = {}

    def retained(status: str, message: str) -> SpeedResult:
        return SpeedResult(status, dict(initial), message, baseline=dict(baseline_rates),
                           selected=dict(baseline_rates), trials=list(trials))

    try:
        initial = _settings(baseline)
        if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (float, int)) or not math.isfinite(budget_seconds) or budget_seconds <= 0:
            raise ValueError("Calibration budget must be finite and positive")
        if type(rounds) is not int or not 1 <= rounds <= 5:
            raise ValueError("Calibration needs between one and five rounds")
        if not isinstance(candidates, (list, tuple)) or len(candidates) > 16:
            raise ValueError("At most 16 execution candidates are allowed")
        unique = []
        for candidate in candidates:
            validated = _settings(candidate)
            if set(validated) != set(initial):
                raise ValueError("Candidates must tune the same execution settings as baseline")
            if validated != initial and validated not in unique:
                unique.append(validated)
        if not unique:
            return retained("unavailable", "Baseline retained; no alternative execution settings are available.")

        started = clock()
        deadline = started + float(budget_seconds)
        if not math.isfinite(started) or not math.isfinite(deadline):
            raise ValueError("Calibration clock must be finite")

        def check() -> None:
            if cancelled is not None and cancelled():
                raise SpeedTuningStopped("cancelled", "Calibration cancelled; baseline settings retained.")
            now = clock()
            if not math.isfinite(now):
                raise ValueError("Calibration clock must remain finite")
            if now >= deadline:
                raise SpeedTuningStopped("timeout", "Calibration budget expired; baseline settings retained.")
            if memory_ok is not None and not memory_ok():
                raise SpeedTuningStopped("memory_pressure", "Memory headroom fell during calibration; baseline settings retained.")

        def notify(message: str) -> None:
            if progress is not None:
                try:
                    progress(message)
                except Exception:
                    # A UI teardown must not change the outcome of measurements.
                    pass

        def run(settings: Dict[str, int], label: str) -> SpeedSample:
            check()
            notify(label)
            sample = measure(dict(settings))
            check()
            if not isinstance(sample, SpeedSample):
                raise ValueError("Measurement adapter did not return a SpeedSample")
            sample.validate()
            sample.rates()
            return sample

        # Compile lazy kernels and touch model pages before estimating A/A noise.
        # This warmup still obeys the overall budget and validates real output,
        # but its cold-start latency is never treated as a speed measurement.
        run(initial, "Warming native kernels (not scored)")
        first = run(initial, "Measuring baseline (1/2)")
        second = run(initial, "Checking baseline repeatability (2/2)")

        def same_output(sample: SpeedSample) -> bool:
            return (sample.output_digest == first.output_digest
                    and sample.prompt_tokens == first.prompt_tokens
                    and sample.generated_tokens == first.generated_tokens)

        first_rates, second_rates = first.rates(), second.rates()
        baseline_rates = {key: _geometric([first_rates[key], second_rates[key]]) for key in first_rates}
        if not same_output(second):
            return retained("error", "Baseline retained; deterministic baseline output was not repeatable.")
        noise = _drift(first_rates, second_rates)
        observations: List[List[Dict[str, Any]]] = [[] for _ in unique]

        for round_index in range(rounds):
            # Rotate order so the same candidate does not always run on a colder device.
            order = list(range(len(unique)))
            offset = round_index % len(order)
            order = order[offset:] + order[:offset]
            for index in order:
                candidate = unique[index]
                label = f"Candidate {index + 1}/{len(unique)}, round {round_index + 1}/{rounds}"
                before = run(initial, label + ": baseline before")
                sample = run(candidate, label + ": measure")
                after = run(initial, label + ": baseline after")
                if not same_output(before) or not same_output(after):
                    return retained("error", "Baseline retained; control output changed during calibration.")
                left, right, candidate_rates = before.rates(), after.rates(), sample.rates()
                reference = {key: _geometric([left[key], right[key]]) for key in left}
                gains = {key: candidate_rates[key] / reference[key] - 1.0 for key in reference}
                gain = 100.0 * (_geometric([1.0 + value for value in gains.values()]) - 1.0)
                drift = _drift(left, right)
                threshold = max(3.0, 2.0 * noise, 2.0 * drift)
                if not all(math.isfinite(value) for value in (gain, noise, drift, threshold)):
                    raise ValueError("Calibration ratios exceeded finite numeric bounds")
                identical = same_output(sample)
                regression_ok = min(gains.values()) >= -0.02
                accepted = identical and regression_ok and gain > threshold
                reason = ("output changed" if not identical else "one metric regressed by more than 2%"
                          if not regression_ok else "gain did not exceed measured noise and drift"
                          if not accepted else "repeatable improvement")
                trial = {
                    "settings": dict(candidate), "round": round_index + 1,
                    "gain_pct": gain, "threshold_pct": threshold, "noise_pct": noise,
                    "drift_pct": drift, "accepted": accepted, "reason": reason,
                    "rates": candidate_rates,
                }
                trials.append(trial)
                observations[index].append(trial)

        check()
        eligible = [index for index, rows in enumerate(observations)
                    if len(rows) == rounds and all(row["accepted"] for row in rows)]
        if not eligible:
            return retained("baseline", "Baseline retained; no candidate showed a repeatable, quality-preserving speed improvement.")
        winner = max(eligible, key=lambda index: min(row["gain_pct"] for row in observations[index]))
        rows = observations[winner]
        selected = {key: _geometric([row["rates"][key] for row in rows]) for key in baseline_rates}
        gain_pct = 100.0 * (_geometric([1.0 + row["gain_pct"] / 100.0 for row in rows]) - 1.0)
        return SpeedResult(
            "improved", dict(unique[winner]),
            "Selected the best repeatably faster settings among the measured candidates.",
            gain_pct=gain_pct, baseline=baseline_rates, selected=selected, trials=trials,
        )
    except SpeedTuningStopped as exc:
        return retained(exc.status, str(exc))
    except TimeoutError:
        return retained("timeout", "Calibration measurement timed out; baseline settings retained.")
    except (KeyboardInterrupt, GeneratorExit):
        return retained("cancelled", "Calibration interrupted; baseline settings retained.")
    except Exception as exc:
        return retained("error", f"Calibration failed ({type(exc).__name__}: {str(exc)[:300]}); baseline settings retained.")


def physical_cpu_count() -> int:
    """Return physical cores without loading an inference framework."""
    try:
        import psutil
        count = psutil.cpu_count(logical=False)
        if count:
            return max(1, min(1024, int(count)))
    except Exception:
        pass
    return max(1, min(1024, (os.cpu_count() or 2) // 2))


def _cpu_identity() -> str:
    """Read a bounded native CPU description without invoking shell commands."""
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:
                description = handle.read(16384)
            parts = {}
            for line in description.splitlines():
                key, separator, value = line.partition(":")
                if separator and key.strip().lower() in {"model name", "hardware", "cpu implementer", "cpu part"}:
                    parts.setdefault(key.strip().lower(), value.strip())
            if parts:
                return json.dumps(parts, sort_keys=True)
        except OSError:
            pass
    elif platform.system() == "Darwin":
        try:
            import ctypes
            libc = ctypes.CDLL(None)
            buffer = ctypes.create_string_buffer(1024)
            length = ctypes.c_size_t(len(buffer))
            if libc.sysctlbyname(b"machdep.cpu.brand_string", buffer, ctypes.byref(length), None, 0) == 0:
                return buffer.value.decode("utf-8", errors="replace")
        except Exception:
            pass
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")


def _hardware_identity() -> Dict[str, Any]:
    """Bounded read-only device discovery; never imports torch/MLX or allocates GPU memory."""
    identity: Dict[str, Any] = {"gpus": []}
    try:
        import psutil
        identity["ram_bytes"] = int(psutil.virtual_memory().total)
    except Exception:
        pass
    if platform.system() == "Linux":
        for card in sorted(Path("/sys/class/drm").glob("card[0-9]*"))[:32]:
            if not card.name[4:].isdigit():
                continue
            device = card / "device"
            if not device.is_dir():
                continue
            properties: Dict[str, Any] = {"pci_path": str(device.resolve())}
            for field_name in ("vendor", "device", "revision", "subsystem_vendor", "subsystem_device", "mem_info_vram_total"):
                try:
                    with (device / field_name).open("r", encoding="utf-8") as handle:
                        properties[field_name] = handle.read(256).strip()
                except OSError:
                    pass
            identity["gpus"].append(properties)
    elif platform.system() == "Windows":
        try:
            import ctypes
            from ctypes import wintypes
            class DisplayDevice(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("DeviceName", wintypes.WCHAR * 32),
                            ("DeviceString", wintypes.WCHAR * 128), ("StateFlags", wintypes.DWORD),
                            ("DeviceID", wintypes.WCHAR * 128), ("DeviceKey", wintypes.WCHAR * 128)]
            for index in range(32):
                display = DisplayDevice()
                display.cb = ctypes.sizeof(display)
                if not ctypes.windll.user32.EnumDisplayDevicesW(None, index, ctypes.byref(display), 0):
                    break
                identity["gpus"].append({"name": display.DeviceString, "device_id": display.DeviceID})
        except Exception:
            pass
    # NVIDIA's utility reports stable GPU UUIDs and driver version even when its
    # native inference library is loaded outside Python package metadata.
    utility = shutil.which("nvidia-smi")
    if utility:
        try:
            result = subprocess.run(
                [utility, "--query-gpu=uuid,name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.5, check=False,
            )
            if result.returncode == 0 and len(result.stdout) <= 16384:
                identity["nvidia"] = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass
    return identity


def _artifact_identity(value: Optional[str]) -> Any:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        return {"identifier": str(value), "local": False}
    resolved = path.resolve()

    def stamp(item: Path) -> Dict[str, Any]:
        stat = item.stat()
        data: Dict[str, Any] = {"path": str(item.relative_to(path) if path.is_dir() else item.name),
                                "resolved": str(item.resolve()), "bytes": stat.st_size,
                                "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
        # Small configs can change without a size change; weights are never read.
        if item.suffix.lower() in {".json", ".py"} and stat.st_size <= 1024 * 1024:
            data["sha256"] = hashlib.sha256(item.read_bytes()).hexdigest()
        return data

    if resolved.is_file():
        return {"root": str(resolved), "file": stamp(path)}
    files = []
    visited = 0
    for root, directories, names in os.walk(path, followlinks=False):
        directories[:] = sorted(name for name in directories if name not in {".git", "__pycache__"})
        for name in sorted(names):
            visited += 1
            if visited > 8192:
                # Refuse an incomplete fingerprint rather than reusing a stale profile.
                raise ValueError("Model directory is too large to fingerprint safely")
            item = Path(root) / name
            if item.suffix.lower() in {".gguf", ".safetensors", ".bin", ".npz", ".pt", ".pth", ".json", ".model", ".tiktoken", ".py"}:
                files.append(stamp(item))
    return {"root": str(resolved), "files": sorted(files, key=lambda item: item["path"])}


def _stable_hardware(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        volatile = ("available", "free", "used", "usage", "utilization", "temperature", "rss", "pressure")
        return {str(key): _stable_hardware(item) for key, item in value.items()
                if not any(part in str(key).lower() for part in volatile)}
    if isinstance(value, (list, tuple)):
        return [_stable_hardware(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("Hardware identity must contain finite JSON values")


def speed_fingerprint(
    backend: str,
    model_path: str,
    *,
    context: int,
    adapter_path: Optional[str] = None,
    projector_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    hardware: Any = None,
) -> str:
    """Identify stable runtime/model state; live free memory is deliberately excluded."""
    versions = {}
    loaded_modules = {}
    for package, module_name in (("mlx", "mlx.core"), ("mlx-lm", "mlx_lm"),
                                 ("mlx-vlm", "mlx_vlm"), ("jang", "jang_tools"),
                                 ("turboquant-mlx-full", "turboquant_mlx"),
                                 ("llama-cpp-python", "llama_cpp"), ("torch", "torch")):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            module = sys.modules.get(module_name)
            versions[package] = str(getattr(module, "__version__", "unknown"))
            module_path = getattr(module, "__file__", None)
            if module_path and Path(module_path).is_file():
                stat = Path(module_path).stat()
                loaded_modules[package] = {"path": str(Path(module_path).resolve()),
                                          "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                          "ctime_ns": stat.st_ctime_ns}
    if type(context) is not int or context < 1:
        raise ValueError("Context must be a positive integer")
    resolved_model = model_path
    if not Path(model_path).expanduser().exists():
        # MLX commonly retains the HF identifier after loading its local snapshot.
        # Resolve it offline so a new snapshot cannot reuse the old model's profile.
        try:
            from huggingface_hub import snapshot_download
            resolved_model = snapshot_download(repo_id=model_path, local_files_only=True)
        except Exception:
            pass
    model_identity = _artifact_identity(resolved_model)
    if isinstance(model_identity, dict) and model_identity.get("local") is False:
        # Unknown model bytes must never match a saved profile on the next call.
        model_identity["unresolved_nonce"] = time.monotonic_ns()
    identity = {
        "schema": _SCHEMA, "backend": str(backend), "context": context,
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine(), "processor": _cpu_identity(),
                     "physical_cores": physical_cpu_count(), "logical_cores": os.cpu_count(),
                     "python": platform.python_version()},
        "visible_accelerators": {name: os.environ.get(name) for name in
                                 ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")},
        "versions": versions, "unpackaged_modules": loaded_modules,
        "detected_hardware": _hardware_identity(),
        "hardware": _stable_hardware(hardware),
        "model": model_identity,
        "adapter": _artifact_identity(adapter_path),
        "projector": _artifact_identity(projector_path),
        "extra": extra or {},
    }
    raw = json.dumps(identity, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finite_tree(value: Any, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if value is None or type(value) in (bool, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, str):
        return len(value) <= 2048
    if isinstance(value, list):
        return len(value) <= 80 and all(_finite_tree(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return len(value) <= 32 and all(isinstance(key, str) and len(key) <= 256 and _finite_tree(item, depth + 1)
                                      for key, item in value.items())
    return False


def _validated_record(value: Any, fingerprint: str) -> Optional[Dict[str, Any]]:
    try:
        if not isinstance(value, dict) or not _finite_tree(value):
            return None
        if value.get("status") not in {"improved", "baseline"} or value.get("fingerprint") != fingerprint:
            return None
        settings = _settings(value["settings"])
        gain = value.get("gain_pct", 0.0)
        if isinstance(gain, bool) or not isinstance(gain, (float, int)) or not 0 <= gain <= 10000:
            return None
        if not isinstance(value.get("message"), str):
            return None
        if not isinstance(value.get("baseline"), dict) or not isinstance(value.get("selected"), dict) or not isinstance(value.get("trials"), list):
            return None
        # Validate nested candidate settings too; cache data cannot inject quality knobs.
        for trial in value["trials"]:
            if not isinstance(trial, dict):
                return None
            _settings(trial.get("settings", {}))
        return SpeedResult(value["status"], settings, value["message"], float(gain),
                           value["baseline"], value["selected"], value["trials"], fingerprint).to_dict()
    except (ValueError, TypeError, KeyError):
        return None


class SpeedProfileStore:
    """Small atomic profile cache with thread and process serialization."""

    def __init__(self, path: Optional[str] = None):
        if path is None:
            from config.paths import get_portable_data_dir
            path = os.path.join(get_portable_data_dir(), "speed_profiles.json")
        self.path = Path(path)
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(str(self.path.resolve()), threading.RLock())

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(str(self.path) + ".lock", "a+b") as handle:
                if os.name == "nt":
                    import msvcrt
                    if handle.tell() == 0:
                        handle.write(b"\0")
                        handle.flush()
                    def acquire():
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    def release():
                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    def acquire():
                        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    def release():
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                deadline = time.monotonic() + 3.0
                while True:
                    try:
                        acquire()
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Speed profile cache is busy")
                        time.sleep(0.02)
                try:
                    yield
                finally:
                    release()

    def _read(self) -> Dict[str, Any]:
        if not self.path.is_file() or self.path.stat().st_size > _MAX_CACHE_BYTES:
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict) or data.get("schema") != _SCHEMA:
                return {}
            profiles = data.get("profiles")
            if not isinstance(profiles, dict) or len(profiles) > _MAX_PROFILES:
                return {}
            return {key: record for key, value in profiles.items()
                    if isinstance(key, str) and 1 <= len(key) <= 256
                    and (record := _validated_record(value, key)) is not None}
        except (OSError, ValueError, TypeError, RecursionError):
            return {}

    def get(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        if not isinstance(fingerprint, str) or not 1 <= len(fingerprint) <= 256:
            return None
        try:
            with self._locked():
                return self._read().get(fingerprint)
        except (OSError, TimeoutError):
            return None

    def put(self, fingerprint: str, result: SpeedResult) -> bool:
        if not isinstance(fingerprint, str) or not 1 <= len(fingerprint) <= 256 or not isinstance(result, SpeedResult):
            return False
        value = result.to_dict()
        value["fingerprint"] = fingerprint
        record = _validated_record(value, fingerprint)
        if record is None:
            return False
        temporary = None
        try:
            with self._locked():
                profiles = self._read()
                profiles.pop(fingerprint, None)
                profiles[fingerprint] = record
                while len(profiles) > _MAX_PROFILES:
                    profiles.pop(next(iter(profiles)))
                raw = json.dumps({"schema": _SCHEMA, "profiles": profiles}, allow_nan=False,
                                 ensure_ascii=True, separators=(",", ":"))
                if len(raw.encode("utf-8")) > _MAX_CACHE_BYTES:
                    return False
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                                 prefix=self.path.name + ".", suffix=".tmp", delete=False) as handle:
                    temporary = handle.name
                    handle.write(raw)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                temporary = None
            return True
        except (OSError, ValueError, TimeoutError):
            return False
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
