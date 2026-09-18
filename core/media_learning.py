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

_FACTORIES = {}
_LOCKS = {}
_LOCKS_GUARD = threading.Lock()


def register_media_training_backend(name, factory):
    """Trusted Python extension: returns module/loss/save hooks, not model-generated code."""
    if not isinstance(name,str) or not name or not callable(factory):
        raise ValueError("A named callable training adapter is required")
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
        configured = str(info.get("training_backend", ""))
        if configured:
            return configured
        if info.get("repo_id") == "SG161222/RealVisXL_V5.0" and info.get("model_type") == "image":
            return "diffusers_sdxl"
        return ""

    def capabilities(self, info):
        backend = self._backend(info)
        factory = _FACTORIES.get(backend)
        available = backend == "diffusers_sdxl" or callable(factory)
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

    def read_samples(self, manifest, workspace):
        workspace = Path(workspace).resolve()
        path = (workspace / manifest).resolve()
        if not path.is_relative_to(workspace) or path.suffix != ".jsonl" or not path.is_file():
            raise ValueError("Dataset must be an existing JSONL file inside the workspace")
        if path.stat().st_size > 1024 * 1024:
            raise ValueError("This interactive learning path accepts at most a 1 MiB manifest")
        samples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            media = (path.parent / str(row.get("path", ""))).resolve()
            caption = row.get("caption")
            if not media.is_relative_to(workspace) or not media.is_file() or not isinstance(caption,str) or not caption.strip():
                raise ValueError("Each example needs a workspace media path and a nonempty caption")
            samples.append({"path":str(media),"caption":caption})
        if not 1 <= len(samples) <= 8:
            raise ValueError("Use one to eight real media examples for an interactive adapter update")
        return samples

    def learn(self, info, manifest, workspace, media_engine, audio_engine, cancel_event=None):
        caps = self.capabilities(info)
        if not caps["supported"]:
            return {"status":"unsupported","weights_updated":False,**caps}
        samples = self.read_samples(manifest,workspace)
        repo = str(info.get("repo_id", ""))
        key = _key(repo)
        with _LOCKS_GUARD:
            lock = _LOCKS.setdefault(key,threading.RLock())
        with lock:
            session = None
            try:
                factory = _sdxl_session if caps["backend"] == "diffusers_sdxl" else _FACTORIES[caps["backend"]]
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
        except Exception:
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
        except Exception:
            with torch.no_grad():
                for name,param in named:
                    param.copy_(before[name])
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
    if not directory.is_dir():
        raise RuntimeError("Saved media adapter directory is missing")
    return directory


def apply_saved_media_adapter(pipeline, repo_id):
    directory = get_saved_media_adapter(repo_id)
    if directory is None:
        return
    if saved.get("repo_id") != repo_id or not directory.is_relative_to(root.resolve()):
        raise RuntimeError("Media adapter identity/path mismatch")
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
