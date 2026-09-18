"""Measured media evidence for a text controller, not simulated vision/hearing."""
from __future__ import annotations

import gc
import hashlib
import math
from pathlib import Path


def _digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def review_artifact(record):
    path = Path(record["path"])
    if not path.is_file():
        raise ValueError("Artifact file is missing")
    kind, prompt = record["kind"], record["prompt"]
    evidence = {"sha256": _digest(path), "bytes": path.stat().st_size,
                "kind": kind, "quality_verified": False,
                "limitation": "Alignment is a proxy, not proof of fidelity, beauty, correctness, or learning improvement."}
    try:
        if kind in ("image", "video"):
            from PIL import Image
            images = []
            if kind == "image":
                with Image.open(path) as image:
                    evidence["dimensions"] = list(image.size)
                    image.thumbnail((1024,1024))
                    images.append(image.convert("RGB").copy())
                evidence["review_scope"] = "one generated image"
            else:
                import imageio.v2 as imageio
                reader = imageio.get_reader(str(path))
                try:
                    meta = reader.get_meta_data()
                    count = meta.get("nframes", 0)
                    if not isinstance(count,(int,float)) or not math.isfinite(count):
                        count = float(meta.get("duration",0) or 0) * float(meta.get("fps",0) or 0)
                    count = max(1,int(count))
                    indices = sorted(set(int(i * (count - 1) / 7) for i in range(8)))
                    for index in indices:
                        image = Image.fromarray(reader.get_data(index))
                        image.thumbnail((1024,1024)); images.append(image.convert("RGB"))
                    evidence["review_scope"] = "sampled video frames only; not audio or temporal-motion quality"
                    evidence["sampled_frame_indices"] = indices
                finally:
                    reader.close()
            import torch
            from transformers import CLIPModel, CLIPProcessor
            repo = "openai/clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(repo)
            model = CLIPModel.from_pretrained(repo).eval()
            try:
                token_count = len(processor.tokenizer(prompt)["input_ids"])
                evidence["prompt_truncated_for_reviewer"] = token_count > 77
                inputs = processor(text=[prompt], images=images, return_tensors="pt", padding=True, truncation=True, max_length=77)
                with torch.inference_mode():
                    output = model(**inputs)
                    scores = (output.image_embeds @ output.text_embeds.T).flatten().cpu().tolist()
                evidence.update({"reviewer":repo,"alignment_cosine":sum(scores)/len(scores),"frame_alignment_cosines":scores,"perception_available":True})
            finally:
                del model
        elif kind == "audio":
            import numpy as np
            import soundfile as sf
            from scipy.signal import resample_poly
            import torch
            from transformers import ClapModel, ClapProcessor
            meta = sf.info(str(path))
            with sf.SoundFile(str(path)) as handle:
                samples = handle.read(frames=min(meta.frames, meta.samplerate * 10),dtype="float32",always_2d=True).mean(axis=1)
            if samples.size == 0:
                raise ValueError("Audio contains no decoded samples")
            evidence.update({"duration_seconds":meta.duration,"sample_rate":meta.samplerate,
                             "review_scope":"first 10 seconds at most; mono analysis, original audio unchanged",
                             "rms":float(np.sqrt(np.mean(samples ** 2))),
                             "clipped_sample_fraction":float(np.mean(np.abs(samples)>=0.999))})
            if meta.samplerate != 48000:
                divisor = math.gcd(meta.samplerate,48000)
                samples = resample_poly(samples,48000//divisor,meta.samplerate//divisor)
            repo = "laion/clap-htsat-unfused"
            processor = ClapProcessor.from_pretrained(repo)
            model = ClapModel.from_pretrained(repo).eval()
            try:
                inputs = processor(text=[prompt], audios=[samples],sampling_rate=48000,return_tensors="pt",padding=True)
                with torch.inference_mode():
                    output = model(**inputs)
                    score = float((output.audio_embeds @ output.text_embeds.T).item())
                evidence.update({"reviewer":repo,"alignment_cosine":score,"perception_available":True})
            finally:
                del model
        else:
            raise ValueError("Unsupported review modality")
    except Exception as exc:
        evidence.update({"perception_available":False,"review_error":f"{type(exc).__name__}: {exc}"})
        evidence.pop("alignment_cosine",None)
    finally:
        gc.collect()
    return evidence
