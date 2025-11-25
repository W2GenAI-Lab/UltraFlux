from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ultraflux.autoencoder_kl import AutoencoderKL
from ultraflux.pipeline_flux import FluxPipeline
from ultraflux.transformer_flux_visionyarn import FluxTransformer2DModel


MODEL_ID = os.getenv("ULTRAFLUX_MODEL_ID", "Owen777/UltraFlux-v1")
TRANSFORMER_SUBFOLDER = os.getenv("ULTRAFLUX_TRANSFORMER_SUBFOLDER", "transformer")
VAE_SUBFOLDER = os.getenv("ULTRAFLUX_VAE_SUBFOLDER", "vae")
DEVICE = os.getenv("ULTRAFLUX_DEVICE", "cuda")
RESULTS_DIR = Path(os.getenv("ULTRAFLUX_RESULTS_DIR", "results"))
MAX_SEQ_LEN = int(os.getenv("ULTRAFLUX_MAX_SEQUENCE_LENGTH", "512"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_pipeline: Optional[FluxPipeline] = None
_pipeline_lock = threading.Lock()
_inference_lock = threading.Lock()


def _load_pipeline() -> FluxPipeline:
    """
    Lazily initialize the UltraFlux pipeline.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        local_vae = AutoencoderKL.from_pretrained(
            MODEL_ID,
            subfolder=VAE_SUBFOLDER,
            torch_dtype=torch.bfloat16,
        )
        transformer = FluxTransformer2DModel.from_pretrained(
            MODEL_ID,
            subfolder=TRANSFORMER_SUBFOLDER,
            torch_dtype=torch.bfloat16,
        )
        pipeline = FluxPipeline.from_pretrained(
            MODEL_ID,
            vae=local_vae,
            torch_dtype=torch.bfloat16,
            transformer=transformer,
        )
        pipeline.scheduler.config.use_dynamic_shifting = False
        pipeline.scheduler.config.time_shift = 4
        pipeline = pipeline.to(DEVICE)
        pipeline.set_progress_bar_config(disable=True)
        _pipeline = pipeline
        return _pipeline


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1600)
    height: int = Field(4096, ge=256, le=4096)
    width: int = Field(4096, ge=256, le=4096)
    num_inference_steps: int = Field(50, ge=10, le=200)
    guidance_scale: float = Field(4.0, ge=0.0, le=20.0)
    seed: Optional[int] = Field(default=None, description="Seed for deterministic generation")


class GenerateResponse(BaseModel):
    prompt: str
    seed: int
    image_path: str


app = FastAPI(
    title="UltraFlux Inference Service",
    version="1.0.0",
    description="HTTP service for running UltraFlux inference on user-provided prompts.",
)


@app.on_event("startup")
def _startup_event() -> None:
    _load_pipeline()


@app.get("/healthz")
def healthcheck() -> dict:
    pipeline_ready = _pipeline is not None
    return {"status": "ok" if pipeline_ready else "initializing", "device": DEVICE}


@app.post("/generate", response_model=GenerateResponse)
def generate_image(request: GenerateRequest) -> GenerateResponse:
    pipeline = _load_pipeline()

    generator = torch.Generator("cpu")
    seed = request.seed if request.seed is not None else torch.randint(0, 2**31 - 1, (1,)).item()
    generator = generator.manual_seed(seed)

    # Serialization: guard against concurrent GPU access in this simple service.
    with _inference_lock:
        try:
            output = pipeline(
                request.prompt,
                height=request.height,
                width=request.width,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_inference_steps,
                max_sequence_length=MAX_SEQ_LEN,
                generator=generator,
            )
        except torch.cuda.OutOfMemoryError as exc:
            raise HTTPException(status_code=503, detail="CUDA out of memory during inference") from exc
        except Exception as exc:  # pylint: disable=broad-except
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    image = output.images[0]
    timestamp = int(time.time() * 1000)
    out_path = RESULTS_DIR / f"ultra_flux_{timestamp}.jpeg"
    image.save(out_path)

    return GenerateResponse(prompt=request.prompt, seed=seed, image_path=str(out_path.resolve()))


@app.get("/")
def root() -> dict:
    return {
        "message": "UltraFlux inference service is running. POST to /generate with a prompt to create images.",
        "model_id": MODEL_ID,
        "device": DEVICE,
    }


