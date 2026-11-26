"""Check selected weight rows from both pinned publisher MLX packs.

Run from the checkout with python tools/verify_publisher_mlx.py --download-samples
and optionally --device cuda or --device metal. The check fetches selected
HTTP byte ranges and the two verified publisher Python files in a temporary
directory. It compares the publisher's projection to an independent NumPy
calculation. Full model inference requires the complete original snapshot.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.bonsai_models import BONSAI_MLX_REPO, BONSAI_MLX_REVISION, CRACK_MLX_REPO, CRACK_MLX_REVISION
from core.repository_runtime_imports import repository_runtime_imports
from mlx_import_compat import load_mlx_lm


RUNTIME_FILES = {
    "codec.py": "7f7fd67637a7d9363eab8ebf7db59771830b690330ffef772379a0169ddb9608",
    "runtime.py": "98180a782214c655cb49f4f96e6291828c943b39bc916ba57ebc032a88777690",
}
SAMPLES = ("model.layers.63.self_attn.q_proj", "model.layers.63.mlp.down_proj")


def read_range(url, start, length):
    end = start + length - 1
    # Cache keys on some HTTP proxies do not include the Range request header.
    request = Request(f"{url}?studio_range={start}-{end}", headers={"Range": f"bytes={start}-{end}"})
    with urlopen(request, timeout=30) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise ValueError("The model host did not return the requested byte range")
        data = response.read(length + 1)
    if len(data) != length:
        raise ValueError("Incomplete safetensors range")
    return data


def read_arrays(url, base_offset, header, module_path):
    import numpy as np

    arrays = {}
    for name in ("weight", "scales", "biases", "signs"):
        desc = header[f"language_model.{module_path}.{name}"]
        dtype = np.dtype({"F16": "<f2", "F32": "<f4", "I8": "i1", "U32": "<u4"}[desc["dtype"]])
        shape = list(desc["shape"])
        if name != "signs":
            shape[0] = min(8, shape[0])
        start, end = desc["data_offsets"]
        size = int(np.prod(shape)) * dtype.itemsize
        if not 0 < size <= min(end - start, 1024 * 1024):
            raise ValueError(f"Unexpected range length for {name}")
        data = read_range(url, base_offset + start, size)
        arrays[name] = np.frombuffer(data, dtype=dtype).reshape(shape)
    return arrays


def project_reference(x, arrays, block):
    import numpy as np

    width = x.shape[-1]
    rotated = (x.astype("float32") * arrays["signs"]).reshape(-1, block)
    stride = 1
    while stride < block:
        pair = rotated.reshape(-1, 2, stride)
        rotated = np.stack((pair[:, 0] + pair[:, 1], pair[:, 0] - pair[:, 1]), axis=1).reshape(-1, block)
        stride *= 2
    # The pack specifies FP16 activations after the normalized FP32 transform.
    rotated = (rotated * block**-.5).reshape(x.shape).astype("float16").astype("float32")
    codes = (arrays["weight"][:, :, None] >> (2 * np.arange(16, dtype="uint32"))) & 3
    weights = codes.reshape(-1, width).astype("float32") * np.repeat(arrays["scales"], 128, axis=1)
    weights += np.repeat(arrays["biases"], 128, axis=1)
    return rotated @ weights.T


def verify(repo, revision):
    import mlx.core as mx
    import numpy as np

    root = f"https://huggingface.co/{repo}/resolve/{revision}/"
    with urlopen(root + "config.json", timeout=30) as response:
        config = json.load(response)
    url = root + "model.safetensors"
    header_size, = struct.unpack("<Q", read_range(url, 0, 8))
    if not 0 < header_size <= 8 * 1024 * 1024:
        raise ValueError("Unexpected safetensors header length")
    header = json.loads(read_range(url, 8, header_size))
    with TemporaryDirectory() as directory:
        for name, expected in RUNTIME_FILES.items():
            with urlopen(root + "runtime/" + name, timeout=30) as response:
                source = response.read(1024 * 1024)
            if hashlib.sha256(source).hexdigest() != expected:
                raise ValueError(f"Publisher source hash mismatch for {name}")
            (Path(directory) / name).write_bytes(source)
        with repository_runtime_imports(directory):
            from runtime import Packed
            for index, path in enumerate(SAMPLES):
                record = next(item for item in config["modules"] if item["path"] == path)
                arrays = read_arrays(url, header_size + 8, header, path)
                weights = tuple(mx.array(arrays[key]) for key in ("weight", "scales", "biases"))
                projection = Packed(weights, record["block"], mx.array(arrays["signs"]), False, mx.float16)
                width = len(arrays["signs"])
                x = np.random.default_rng((991, 777)[index]).standard_normal((1, width)).astype("float16")
                actual = np.asarray(projection(mx.array(x))).astype("float32")
                expected = project_reference(x, arrays, record["block"])
                np.testing.assert_allclose(actual, expected, rtol=.004, atol=.004)
                print(f"{repo}@{revision} {path}: {actual.size} rows x {width}, "
                      f"max absolute error {np.max(np.abs(actual - expected)):.8f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-samples", action="store_true", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "metal"), default="cpu")
    args = parser.parse_args()
    import mlx.core as mx
    if args.device == "cuda" and not mx.cuda.is_available():
        parser.error("MLX CUDA device unavailable")
    if args.device == "metal" and not mx.metal.is_available():
        parser.error("MLX Metal device unavailable")
    mx.set_default_device(mx.cpu if args.device == "cpu" else mx.gpu)
    load_mlx_lm()
    for repo, revision in ((BONSAI_MLX_REPO, BONSAI_MLX_REVISION), (CRACK_MLX_REPO, CRACK_MLX_REVISION)):
        verify(repo, revision)


if __name__ == "__main__":
    main()
