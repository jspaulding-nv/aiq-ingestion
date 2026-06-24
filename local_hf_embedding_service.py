#!/usr/bin/env python3
"""OpenAI-compatible local embedding service for Nemotron Embed VL.

This intentionally uses the Hugging Face/PyTorch path instead of NIM, ONNX,
TensorRT, vLLM, or flash-attn so it can run on B300 with SDPA.
"""

from __future__ import annotations

import base64
import io
import os
import threading
import time
from typing import Any, Dict, List, Optional
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoModel


MODEL_NAME = os.environ.get("HF_EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2")
MODEL_REVISION = os.environ.get("HF_EMBED_REVISION", "0c6f636ed4c022e427277c4c336054d6cdffaa87")
DEVICE = os.environ.get("HF_EMBED_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
ATTN_IMPLEMENTATION = os.environ.get("HF_EMBED_ATTN", "sdpa")
DTYPE_NAME = os.environ.get("HF_EMBED_DTYPE", "bfloat16")
P_MAX_LENGTH = int(os.environ.get("HF_EMBED_P_MAX_LENGTH", "2048"))
MAX_INPUT_TILES = int(os.environ.get("HF_EMBED_MAX_INPUT_TILES", "6"))
USE_THUMBNAIL = os.environ.get("HF_EMBED_USE_THUMBNAIL", "true").lower() in {"1", "true", "yes"}
ALLOW_FILE_IMAGES = os.environ.get("HF_EMBED_ALLOW_FILE_IMAGES", "false").lower() in {"1", "true", "yes"}


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class EmbeddingRequest(BaseModel):
    input: Optional[Any] = None
    messages: Optional[List[Dict[str, Any]]] = None
    model: Optional[str] = None
    input_type: Optional[str] = None
    modality: Optional[str] = None


class EmbeddingItem:
    def __init__(self, text: Optional[str] = None, image: Optional[Image.Image] = None) -> None:
        self.text = text
        self.image = image

    @property
    def has_text(self) -> bool:
        return self.text is not None and self.text != ""

    @property
    def has_image(self) -> bool:
        return self.image is not None


class ModelState:
    def __init__(self) -> None:
        self.model: Any = None
        self.loaded_at: Optional[float] = None
        self.lock = threading.Lock()


state = ModelState()
app = FastAPI(title="Local HF Nemotron Embed VL Service", version="1.0")


def load_model() -> None:
    if state.model is not None:
        return

    dtype = DTYPES.get(DTYPE_NAME)
    if dtype is None:
        raise RuntimeError(f"Unsupported HF_EMBED_DTYPE={DTYPE_NAME!r}; use one of {sorted(DTYPES)}")

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,
        "attn_implementation": ATTN_IMPLEMENTATION,
    }
    if MODEL_REVISION:
        kwargs["revision"] = MODEL_REVISION

    model = AutoModel.from_pretrained(MODEL_NAME, **kwargs).to(DEVICE).eval()
    model.processor.p_max_length = P_MAX_LENGTH
    model.processor.max_input_tiles = MAX_INPUT_TILES
    model.processor.use_thumbnail = USE_THUMBNAIL

    state.model = model
    state.loaded_at = time.time()


@app.on_event("startup")
def startup() -> None:
    load_model()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if state.model is not None else "loading",
        "model": MODEL_NAME,
        "revision": MODEL_REVISION,
        "device": DEVICE,
        "dtype": DTYPE_NAME,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "loaded_at": state.loaded_at,
    }


@app.get("/v1/health/ready")
def ready() -> Dict[str, Any]:
    if state.model is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    return health()


@app.get("/v1/models")
def models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/embeddings")
def embeddings(request: EmbeddingRequest) -> Dict[str, Any]:
    if state.model is None:
        raise HTTPException(status_code=503, detail="model is not loaded")

    try:
        items = parse_items(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid embedding request: {exc}") from exc

    if not items:
        raise HTTPException(status_code=400, detail="request must include input or messages")

    input_type = (request.input_type or "passage").lower()
    is_query = input_type in {"query", "queries"}

    try:
        with state.lock:
            with torch.inference_mode():
                vectors = encode_items(items, is_query=is_query)
                if DEVICE.startswith("cuda"):
                    torch.cuda.synchronize()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"embedding failed: {exc}") from exc

    vectors = vectors.detach().float().cpu()
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": index, "embedding": vector.tolist()}
            for index, vector in enumerate(vectors)
        ],
        "model": request.model or MODEL_NAME,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def encode_items(items: List[EmbeddingItem], *, is_query: bool) -> torch.Tensor:
    if is_query and any(item.has_image for item in items):
        raise ValueError("image query embeddings are not supported by this model wrapper")

    if all(item.has_text and not item.has_image for item in items):
        texts = [item.text or "" for item in items]
        if is_query:
            return state.model.encode_queries(texts)
        return state.model.encode_documents(texts=texts)

    if all(item.has_image and not item.has_text for item in items):
        return state.model.encode_documents(images=[item.image for item in items])

    if all(item.has_image and item.has_text for item in items):
        return state.model.encode_documents(
            images=[item.image for item in items],
            texts=[item.text or "" for item in items],
        )

    encoded = []
    for item in items:
        encoded.append(encode_items([item], is_query=is_query))
    return torch.cat(encoded, dim=0)


def parse_items(request: EmbeddingRequest) -> List[EmbeddingItem]:
    if request.messages is not None:
        return [parse_message(message) for message in request.messages]

    value = request.input
    if value is None:
        return []
    if isinstance(value, str):
        return [EmbeddingItem(text=value)]
    if isinstance(value, list):
        return [parse_input_item(item) for item in value]
    return [parse_input_item(value)]


def parse_input_item(item: Any) -> EmbeddingItem:
    if isinstance(item, str):
        return EmbeddingItem(text=item)

    if isinstance(item, dict):
        text = item.get("text")
        image_value = item.get("image")
        if image_value is None and isinstance(item.get("image_url"), dict):
            image_value = item["image_url"].get("url")
        image = load_image(image_value) if image_value is not None else None
        return EmbeddingItem(text=text, image=image)

    if isinstance(item, list):
        text_parts: List[str] = []
        image: Optional[Image.Image] = None
        for part in item:
            parsed = parse_input_item(part)
            if parsed.has_text:
                text_parts.append(parsed.text or "")
            if parsed.has_image:
                image = parsed.image
        return EmbeddingItem(text="".join(text_parts) or None, image=image)

    raise ValueError(f"Unsupported input item type: {type(item).__name__}")


def parse_message(message: Dict[str, Any]) -> EmbeddingItem:
    content = message.get("content")
    if isinstance(content, str):
        return EmbeddingItem(text=content)
    if isinstance(content, list):
        text_parts: List[str] = []
        image: Optional[Image.Image] = None
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text_parts.append(str(part.get("text", "")))
            elif part_type in {"image", "image_url"}:
                image_value = part.get("image")
                if image_value is None and isinstance(part.get("image_url"), dict):
                    image_value = part["image_url"].get("url")
                elif image_value is None:
                    image_value = part.get("image_url")
                image = load_image(image_value)
        return EmbeddingItem(text="".join(text_parts) or None, image=image)
    raise ValueError("message content must be a string or a content-part list")


def load_image(value: Any) -> Image.Image:
    if not isinstance(value, str):
        raise ValueError("image must be a URL, data URL, or allowed local path string")

    if value.startswith("data:image/"):
        _, encoded = value.split(",", 1)
        raw = base64.b64decode(encoded)
    elif value.startswith(("http://", "https://")):
        with urllib.request.urlopen(value, timeout=30) as response:
            raw = response.read()
    elif ALLOW_FILE_IMAGES:
        with open(value, "rb") as handle:
            raw = handle.read()
    else:
        raise ValueError("local image paths are disabled; set HF_EMBED_ALLOW_FILE_IMAGES=true to enable")

    return Image.open(io.BytesIO(raw)).convert("RGB")
