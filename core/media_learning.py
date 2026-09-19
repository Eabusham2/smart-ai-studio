"""Separate, capability-checked media adapter updates. Never text-tokenize media.

Native SDXL is supported. Other backends must register their own differentiable
loss and persistence adapter; generation support alone is not training support.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import threading
import uuid
import html
import mimetypes
import re
import urllib.parse
import urllib.request

_FACTORIES = {}
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def register_media_training_backend(name, factory, matcher=None):
    """Register a trusted trainer plus an optional capability matcher.

    Matchers make media learning architecture/runtime driven rather than tied to
    specific catalog IDs. Imported models receive the same resolution path.
    """
    if not isinstance(name, str) or not name or not callable(factory):
        raise ValueError("A named callable training adapter is required")
    if matcher is not None and not callable(matcher):
        raise ValueError("Media trainer matcher must be callable")
    if matcher is not None:
        factory.matches = matcher
    _FACTORIES[name] = factory


def _key(repo):
    return hashlib.sha256(str(repo).encode()).hexdigest()[:24]


def _atomic_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data,sort_keys=True),encoding="utf-8")
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)


def _persisted_lora_delta_proven(directory: Path) -> bool:
    """Prove an external LoRA artifact contains a learned nonzero update factor.

    Standard LoRA initializes one factor (usually B/up) at zero. A successful
    optimization must move that output/update factor away from zero. If the
    artifact format does not expose a recognizable LoRA update tensor, fail
    closed rather than claiming a parameter update that cannot be verified.
    """
    files = list(Path(directory).rglob("*.safetensors"))
    if not files:
        return False

    output_markers = (
        "lora_b",
        "lora.b",
        "lora_up",
        "lora.up",
        "lora_out",
        "lora.out",
        "adapter_b",
        "adapter.b",
    )
    try:
        import torch
        from safetensors import safe_open
    except Exception:
        return False

    saw_output_factor = False
    for file in files:
        try:
            with safe_open(str(file), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    lowered = str(key).lower()
                    if not any(marker in lowered for marker in output_markers):
                        continue
                    saw_output_factor = True
                    tensor = handle.get_tensor(key)
                    if tensor.numel() <= 0:
                        continue
                    if not bool(torch.isfinite(tensor).all().item()):
                        raise RuntimeError(
                            f"Persisted media LoRA contains non-finite tensor: {key}"
                        )
                    if float(tensor.detach().abs().max().item()) > 0.0:
                        return True
        except RuntimeError:
            raise
        except Exception:
            continue

    if saw_output_factor:
        return False
    return False


class MediaLearningService:
    def __init__(self, directory):
        self.directory = Path(directory)
        # Registers optional, model-specific image/video/audio trainers. Importing
        # here avoids changing the existing text/RSI training stack.
        try:
            from core import media_training_backends  # noqa: F401
        except Exception:
            pass

    @staticmethod
    def _backend(info):
        configured = str(info.get("training_backend", "") or "").strip()
        if configured:
            return configured

        # Resolve by trainer-declared capability, never by catalog slot/model ID.
        # Prefer the first *available* compatible trainer, but keep the first match
        # as a useful unsupported reason if no installed runtime can train it.
        first_match = ""
        for name, factory in list(_FACTORIES.items()):
            matcher = getattr(factory, "matches", None)
            if not callable(matcher):
                continue
            try:
                if not bool(matcher(info)):
                    continue
            except Exception:
                continue
            if not first_match:
                first_match = name
            available = True
            checker = getattr(factory, "available", None)
            if callable(checker):
                try:
                    available = bool(checker(info))
                except Exception:
                    available = False
            if available:
                return name
        return first_match

    def capabilities(self, info):
        backend = self._backend(info)
        factory = _FACTORIES.get(backend)
        available = callable(factory)
        if callable(factory) and hasattr(factory, "available"):
            try:
                available = bool(factory.available(info))
            except Exception:
                available = False
        return {
            "supported": available,
            "backend": backend or None,
            "reason": None if available else (
                "No installed model-specific media trainer is available for this backend; "
                "generation remains available but is not a weight update."
            ),
        }

    @staticmethod
    def _allowed_extensions(kind):
        return {
            "image": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"},
            "video": {".mp4", ".mov", ".mkv", ".webm", ".avi"},
            "audio": {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"},
        }.get(str(kind or "").lower(), set())

    @staticmethod
    def _caption_from_name(path_or_url, fallback=None):
        if fallback:
            return str(fallback).strip()
        name = Path(urllib.parse.urlparse(str(path_or_url)).path).stem
        text = re.sub(r"[_\-]+", " ", name).strip()
        return text or "media sample"

    def _download_media_url(self, url, folder, kind, caption=None, index=0):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(str(url), headers={"User-Agent": "SmartAI-MediaLearn/1.0"})
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            ctype = str(resp.headers.get_content_type() or "").lower()
            detected = "image" if ctype.startswith("image/") else "audio" if ctype.startswith("audio/") else "video" if ctype.startswith("video/") else ""
            if detected and detected != kind:
                raise ValueError(f"Expected {kind} media but URL returned {detected}")
            suffix = Path(urllib.parse.urlparse(str(url)).path).suffix.lower()
            if suffix not in self._allowed_extensions(kind):
                suffix = mimetypes.guess_extension(ctype) or next(iter(self._allowed_extensions(kind)), ".bin")
            path = folder / f"{index:04d}{suffix}"
            total = 0
            with open(path, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > 128 * 1024 * 1024:
                        raise ValueError("Individual training media is limited to 128 MiB")
                    out.write(chunk)
        return {"path": str(path.resolve()), "caption": self._caption_from_name(url, caption)}

    def _extract_page_media(self, url, kind, limit=8):
        req = urllib.request.Request(str(url), headers={"User-Agent": "SmartAI-MediaLearn/1.0"})
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            ctype = str(resp.headers.get_content_type() or "").lower()
            if ctype.startswith(("image/", "audio/", "video/")):
                return [(str(url), None)]
            raw = resp.read(2 * 1024 * 1024).decode("utf-8", errors="ignore")
        candidates = []
        if kind == "image":
            pattern = r'<img\b[^>]*?src=["\']([^"\']+)["\'][^>]*?(?:alt=["\']([^"\']*)["\'])?'
        elif kind == "video":
            pattern = r'<(?:video|source)\b[^>]*?src=["\']([^"\']+)["\'][^>]*?(?:title=["\']([^"\']*)["\'])?'
        else:
            pattern = r'<(?:audio|source)\b[^>]*?src=["\']([^"\']+)["\'][^>]*?(?:title=["\']([^"\']*)["\'])?'
        for match in re.finditer(pattern, raw, flags=re.I):
            media_url = urllib.parse.urljoin(str(url), html.unescape(match.group(1)))
            label = html.unescape(match.group(2) or "").strip() or None
            if media_url.startswith(("http://", "https://")):
                candidates.append((media_url, label))
            if len(candidates) >= limit:
                break
        if len(candidates) < limit:
            exts = "|".join(re.escape(x.lstrip(".")) for x in self._allowed_extensions(kind))
            if exts:
                for href in re.findall(r'href=["\']([^"\']+\.(?:' + exts + r')(?:\?[^"\']*)?)["\']', raw, flags=re.I):
                    media_url = urllib.parse.urljoin(str(url), html.unescape(href))
                    if media_url.startswith(("http://", "https://")) and all(media_url != u for u,_ in candidates):
                        candidates.append((media_url, None))
                    if len(candidates) >= limit:
                        break
        return candidates[:limit]

    def _search_pages(self, query, limit=4):
        url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(str(query))
        req = urllib.request.Request(url, headers={"User-Agent": "SmartAI-MediaLearn/1.0"})
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            raw = resp.read(2 * 1024 * 1024).decode("utf-8", errors="ignore")
        results = []
        for href in re.findall(r'class=["\']result__a["\'][^>]*href=["\']([^"\']+)["\']', raw, flags=re.I):
            parsed = urllib.parse.urlparse(html.unescape(href))
            target = urllib.parse.parse_qs(parsed.query).get("uddg", [href])[0]
            target = urllib.parse.unquote(target)
            if target.startswith(("http://", "https://")) and target not in results:
                results.append(target)
            if len(results) >= limit:
                break
        return results

    def search_media_candidates(self, query, kinds, max_items=6):
        """Return remote media URLs for supported controller modalities without downloading them."""
        kinds = [str(k).lower() for k in kinds if str(k).lower() in ("image", "video", "audio")]
        if not kinds:
            return []
        results = []
        for page in self._search_pages(query, limit=4):
            for kind in kinds:
                try:
                    found = self._extract_page_media(page, kind, max_items - len(results))
                except Exception:
                    continue
                for media_url, label in found:
                    if any(row["url"] == media_url for row in results):
                        continue
                    results.append({"kind": kind, "url": media_url, "label": label})
                    if len(results) >= max_items:
                        return results
        return results

    def gather_samples(self, source, workspace, kind, caption=None, max_items=8):
        """Gather normal files/folders/URLs/search results into a local training folder."""
        kind = str(kind or "").lower()
        allowed = self._allowed_extensions(kind)
        if not allowed:
            raise ValueError("Training kind must be image, video, or audio")
        workspace = Path(workspace).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        source_text = str(source or "").strip()
        if not source_text:
            raise ValueError("A media learning source is required")

        gather_root = workspace / "media_learning_sources" / uuid.uuid4().hex
        gather_root.mkdir(parents=True, exist_ok=False)
        samples = []

        def add_local(path, label=None):
            path = Path(path).resolve()
            if not path.is_file() or path.suffix.lower() not in allowed:
                return
            dst = gather_root / f"{len(samples):04d}{path.suffix.lower()}"
            import shutil
            shutil.copy2(path, dst)
            samples.append({"path": str(dst.resolve()), "caption": self._caption_from_name(path, label or caption)})

        local = Path(os.path.expanduser(source_text))
        if not local.is_absolute():
            local = workspace / local

        if local.exists():
            if local.is_file() and local.suffix.lower() in (".jsonl", ".json", ".csv", ".tsv", ".parquet"):
                base = local.parent
                rows = []
                suffix = local.suffix.lower()
                if suffix == ".jsonl":
                    for line in local.read_text(encoding="utf-8", errors="replace").splitlines():
                        if line.strip():
                            rows.append(json.loads(line))
                elif suffix == ".json":
                    loaded = json.loads(local.read_text(encoding="utf-8", errors="replace"))
                    rows = loaded if isinstance(loaded, list) else loaded.get("samples", []) if isinstance(loaded, dict) else []
                elif suffix in (".csv", ".tsv"):
                    import csv
                    with open(local, "r", encoding="utf-8", errors="replace", newline="") as handle:
                        rows = list(csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ","))
                else:
                    try:
                        import pyarrow.parquet as pq
                    except Exception as exc:
                        raise RuntimeError("Parquet media learning requires pyarrow") from exc
                    rows = pq.read_table(local).slice(0, max_items).to_pylist()

                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    raw_path = row.get("path") or row.get("file") or row.get("media") or row.get("url")
                    if not raw_path:
                        continue
                    label = row.get("caption") or row.get("text") or row.get("label")
                    raw_path = str(raw_path)
                    if raw_path.startswith(("http://", "https://")):
                        try:
                            samples.append(self._download_media_url(raw_path, gather_root, kind, label or caption, len(samples)))
                        except Exception:
                            continue
                    else:
                        media = Path(raw_path)
                        if not media.is_absolute():
                            media = base / media
                        add_local(media, label)
                    if len(samples) >= max_items:
                        break
            elif local.is_file():
                add_local(local)
            else:
                for path in sorted(local.rglob("*")):
                    add_local(path)
                    if len(samples) >= max_items:
                        break
        elif source_text.startswith(("http://", "https://")):
            for media_url, label in self._extract_page_media(source_text, kind, max_items):
                try:
                    samples.append(self._download_media_url(media_url, gather_root, kind, label or caption, len(samples)))
                except Exception:
                    continue
                if len(samples) >= max_items:
                    break
        else:
            for page in self._search_pages(source_text, limit=4):
                try:
                    found = self._extract_page_media(page, kind, max_items - len(samples))
                except Exception:
                    continue
                for media_url, label in found:
                    try:
                        samples.append(self._download_media_url(media_url, gather_root, kind, label or caption or source_text, len(samples)))
                    except Exception:
                        continue
                    if len(samples) >= max_items:
                        break
                if len(samples) >= max_items:
                    break

        if not samples:
            import shutil
            shutil.rmtree(gather_root, ignore_errors=True)
            raise ValueError("No compatible media could be gathered from that source")
        return {"folder": gather_root, "samples": samples[:max_items]}

    def read_samples(self, source, workspace, kind, caption=None):
        gathered = self.gather_samples(source, workspace, kind=kind, caption=caption, max_items=8)
        return gathered["samples"]

    def learn(self, info, source, workspace, media_engine, audio_engine, cancel_event=None, caption=None):
        caps = self.capabilities(info)
        if not caps["supported"]:
            return {"status":"unsupported","weights_updated":False,**caps}
        samples = self.read_samples(source, workspace, kind=info.get("model_type"), caption=caption)
        repo = str(info.get("repo_id", ""))
        key = _key(repo)
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key,threading.RLock())
        with lock:
            session = None
            try:
                factory = _FACTORIES[caps["backend"]]
                session = factory(info,media_engine,audio_engine)
                if callable(session.get("external_train")):
                    return self._external_update(session, samples, repo, cancel_event)
                return self._update(session,samples,repo,cancel_event)
            finally:
                close = session.get("close") if session else None
                if callable(close): close()

    def _external_update(self, session, samples, repo, cancel_event):
        """Run a trusted upstream trainer and publish only a verified adapter/checkpoint."""
        import shutil

        trainer = session["external_train"]
        directory = self.directory / _key(repo) / uuid.uuid4().hex
        directory.mkdir(parents=True, exist_ok=False)
        try:
            result = trainer(samples, directory, cancel_event)
            files = [
                p for p in directory.rglob("*")
                if p.is_file() and p.stat().st_size > 0
                and p.suffix in (".safetensors", ".bin", ".pt", ".ckpt", ".zip")
            ]
            if not files:
                raise RuntimeError("Trainer completed without a persisted adapter/checkpoint")
            verifier = session.get("verify_saved")
            if callable(verifier) and not verifier(directory):
                raise RuntimeError("Saved media adapter/checkpoint failed verification")
            if not _persisted_lora_delta_proven(directory):
                raise RuntimeError(
                    "External media trainer produced an artifact, but a real nonzero "
                    "LoRA update factor could not be proven; refusing weights_updated=True"
                )
            _atomic_json(
                self.directory / _key(repo) / "current.json",
                {
                    "repo_id": repo,
                    "adapter_path": str(directory.resolve()),
                    "trainer_backend": session.get("backend"),
                },
            )
            payload = dict(result or {})
            payload.update({
                "status": "success",
                "weights_updated": True,
                "adapter_path": str(directory),
                "training_examples": len(samples),
                "trainer_backend": session.get("backend"),
            })
            return payload
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise

    def _update(self, session, samples, repo, cancel_event):
        import torch
        from consolidation.ewc_loss import EWCLossCalculator
        module, loss_fn, save = session["module"], session["loss"], session["save"]
        named = [(n,p) for n,p in module.named_parameters() if p.requires_grad]
        if not named or not callable(loss_fn) or not callable(save):
            raise RuntimeError("Trainer must expose real trainable parameters, loss, and adapter persistence")
        count = sum(p.numel() for _,p in named)
        if count > 20_000_000:
            raise RuntimeError("Interactive media learning is adapter-only, not a full base-model rewrite")
        before = {n:p.detach().clone() for n,p in named}
        was_training = module.training
        optimizer = torch.optim.AdamW([p for _,p in named],lr=1e-4,weight_decay=0.0)
        ewc = EWCLossCalculator(lambda_ewc=400.0,device=str(named[0][1].device))
        losses = []
        directory = None
        try:
            module.train()
            for sample in samples:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("Media training cancelled")
                optimizer.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    loss = loss_fn(sample)
                    fisher = session.get("fisher", {})
                    anchor = session.get("anchor", {})
                    if fisher and anchor:
                        loss = ewc.combine_losses(loss,ewc.calculate_penalty(module.named_parameters(),fisher,anchor))
                    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
                        raise RuntimeError("Media loss is not a finite scalar")
                    loss.backward()
                if not any(p.grad is not None for _,p in named):
                    raise RuntimeError("Media loss produced no parameter gradients")
                torch.nn.utils.clip_grad_norm_([p for _,p in named],1.0,error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach().item()))
            delta_sq = 0.0
            for name,param in named:
                if not bool(torch.isfinite(param).all().item()):
                    raise RuntimeError("Media update produced nonfinite weights")
                delta_sq += float(((param.detach().float()-before[name].float()) ** 2).sum().item())
            if not math.isfinite(delta_sq) or delta_sq <= 0:
                raise RuntimeError("No measurable parameter change; not reporting learning")
            directory = self.directory / _key(repo) / uuid.uuid4().hex
            directory.mkdir(parents=True,exist_ok=False)
            save(directory)
            files = [p for p in directory.rglob("*") if p.is_file() and p.stat().st_size > 0 and p.suffix in (".safetensors",".bin",".pt")]
            if not files:
                raise RuntimeError("Trainer did not persist adapter tensors")
            verifier = session.get("verify_saved")
            if not callable(verifier) or not verifier(directory):
                raise RuntimeError("Saved media tensors were not verified against the updated parameters")
            # The pointer changes only after a valid update and adapter save. Existing
            # checkpoints survive errors; the next media generation loads this adapter.
            _atomic_json(self.directory / _key(repo) / "current.json", {"repo_id":repo,"adapter_path":str(directory.resolve())})
            return {"status":"success","weights_updated":True,"parameter_delta_l2":math.sqrt(delta_sq),
                    "trained_parameters":count,"training_examples":len(samples),"losses":losses,
                    "adapter_path":str(directory),"quality_improvement_verified":False}
        except BaseException:
            with torch.no_grad():
                for name,param in named:
                    param.copy_(before[name])
            if directory is not None:
                import shutil
                shutil.rmtree(directory, ignore_errors=True)
            raise
        finally:
            optimizer.zero_grad(set_to_none=True)
            module.train(was_training)


def get_saved_media_adapter(repo_id):
    from config.paths import get_portable_data_dir
    root = Path(get_portable_data_dir()) / "media_adapters" / _key(repo_id)
    pointer = root / "current.json"
    if not pointer.is_file():
        return None
    saved = json.loads(pointer.read_text())
    directory = Path(saved["adapter_path"]).resolve()
    if saved.get("repo_id") != repo_id or not directory.is_relative_to(root.resolve()):
        raise RuntimeError("Media adapter identity/path mismatch")
    if not directory.is_dir():
        raise RuntimeError("Saved media adapter directory is missing")
    return directory


def apply_saved_media_adapter(pipeline, repo_id):
    directory = get_saved_media_adapter(repo_id)
    if directory is None:
        return
    loader = getattr(pipeline,"load_lora_weights",None)
    if not callable(loader):
        raise RuntimeError("This pipeline cannot reload the learned media adapter")
    loader(str(directory))


def _sdxl_session(info,media_engine,_audio_engine):
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from diffusers import DDPMScheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    pipe = media_engine._load_pipeline(info["repo_id"])
    try:
        if pipe.__class__.__name__ != "StableDiffusionXLPipeline":
            raise RuntimeError("The SDXL trainer requires an actual StableDiffusionXLPipeline")
        # Inference CPU-offload hooks must not evict parameters between the forward
        # and backward pass. Training gets a separate, resident model session.
        device = pipe._execution_device
        if callable(getattr(pipe,"remove_all_hooks",None)):
            pipe.remove_all_hooks()
        pipe.to(device)
        unet = pipe.unet
        unet.requires_grad_(False)
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder_2.requires_grad_(False)
        if not getattr(unet,"peft_config",None):
            unet.add_adapter(LoraConfig(r=4,lora_alpha=4,init_lora_weights="gaussian",target_modules=["to_q","to_k","to_v","to_out.0"]))
        for name,param in unet.named_parameters():
            if "lora_" in name:
                param.requires_grad_(True)
                param.data = param.data.float()
        if hasattr(unet,"enable_gradient_checkpointing"):
            unet.enable_gradient_checkpointing()
        scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        device = pipe._execution_device
        def loss(sample):
            with Image.open(sample["path"]) as image:
                image = ImageOps.fit(image.convert("RGB"),(256,256))
                pixels = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).unsqueeze(0).float()/127.5-1
            with torch.no_grad():
                pipe.vae.to(dtype=torch.float32)
                latents = pipe.vae.encode(pixels.to(device)).latent_dist.sample() * pipe.vae.config.scaling_factor
                embeds,_,pooled,_ = pipe.encode_prompt(prompt=sample["caption"],device=device,do_classifier_free_guidance=False)
                latents = latents.to(dtype=embeds.dtype)
                noise = torch.randn_like(latents)
                timestep = torch.randint(0,scheduler.config.num_train_timesteps,(1,),device=device,dtype=torch.long)
                noisy = scheduler.add_noise(latents,noise,timestep)
                time_ids = pipe._get_add_time_ids((256,256),(0,0),(256,256),dtype=embeds.dtype,text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim).to(device)
            output = unet(noisy,timestep,encoder_hidden_states=embeds,added_cond_kwargs={"text_embeds":pooled,"time_ids":time_ids}).sample
            prediction = scheduler.config.prediction_type
            if prediction == "epsilon": target = noise
            elif prediction == "v_prediction": target = scheduler.get_velocity(latents,noise,timestep)
            else: raise RuntimeError("Unsupported SDXL diffusion training target")
            return torch.nn.functional.mse_loss(output.float(),target.float())
        def save(directory):
            state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            pipe.save_lora_weights(str(directory),unet_lora_layers=state,safe_serialization=True)
        def verify_saved(directory):
            from safetensors.torch import load_file
            files = list(directory.glob("*.safetensors"))
            if len(files) != 1: return False
            saved = load_file(str(files[0]))
            expected = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
            return all("unet."+name in saved and torch.equal(saved["unet."+name],value.detach().cpu()) for name,value in expected.items())
        return {"module":unet,"loss":loss,"save":save,"verify_saved":verify_saved,"close":media_engine.unload_model}
    except Exception:
        media_engine.unload_model()
        raise


def _sdxl_matches(info):
    repo = str(info.get("repo_id", "") or "").lower()
    name = str(info.get("name", "") or "").lower()
    precision = str(info.get("precision", "") or "").lower()
    kind = str(info.get("model_type", "") or "").lower()
    blob = " ".join((repo, name, precision))
    return kind == "image" and any(
        marker in blob for marker in ("sdxl", "stable-diffusion-xl", "realvisxl")
    )


register_media_training_backend("diffusers_sdxl", _sdxl_session, matcher=_sdxl_matches)
