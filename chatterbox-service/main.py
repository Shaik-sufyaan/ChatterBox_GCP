import asyncio
import audioop
import base64
import json
import os
import re
import tempfile

import numpy as np
import scipy.signal
import torch
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from google.cloud import storage
from pydantic import BaseModel
from chatterbox.tts import ChatterboxTurboTTS

app = FastAPI()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTurboTTS.from_pretrained(device=device)

gcs_client = storage.Client()

_API_KEY = os.environ.get("CHATTERBOX_API_KEY", "")

# In-memory cache: gcs_url → local temp file path
# Avoids re-downloading the same voice reference on every call
_voice_ref_cache: dict[str, str] = {}


class SynthRequest(BaseModel):
    text: str
    voice_ref_url: str | None = None  # GCS URL: gs://bucket/voice-refs/agent-id.wav
    speed: float = 1.0


def _check_key(x_api_key: str = Header(default="")):
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _download_voice_ref(gcs_url: str) -> str:
    """Download voice reference from GCS to a temp file, return local path.
    Caches result in memory so repeated calls for the same URL skip the download.
    """
    if gcs_url in _voice_ref_cache:
        cached = _voice_ref_cache[gcs_url]
        if os.path.exists(cached):
            return cached

    parts = gcs_url.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    blob.download_to_filename(tmp.name)
    _voice_ref_cache[gcs_url] = tmp.name
    return tmp.name


def _pcm_to_mulaw_b64(wav_tensor: torch.Tensor, sample_rate: int) -> str:
    """Convert a torchaudio tensor to base64 mulaw 8kHz."""
    pcm = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
    pcm_8k = scipy.signal.resample_poly(pcm, 8000, sample_rate)
    pcm_i16 = (np.clip(pcm_8k, -1.0, 1.0) * 32767).astype(np.int16)
    mulaw = audioop.lin2ulaw(pcm_i16.tobytes(), 2)
    return base64.b64encode(mulaw).decode()


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, keeping each chunk non-empty."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/synthesize")
async def synthesize(
    req: SynthRequest,
    request: Request,
    x_api_key: str = Header(default=""),
):
    _check_key(x_api_key)
    sentences = _split_sentences(req.text)

    async def generate():
        try:
            voice_ref_path = None
            if req.voice_ref_url:
                voice_ref_path = await asyncio.to_thread(_download_voice_ref, req.voice_ref_url)

            for sentence in sentences:
                if await request.is_disconnected():
                    break
                try:
                    wav = await asyncio.to_thread(
                        model.generate,
                        sentence,
                        voice_ref_path,
                    )
                    b64 = _pcm_to_mulaw_b64(wav, model.sr)
                    yield json.dumps({"audio": b64}) + "\n"
                except Exception as exc:
                    yield json.dumps({"error": str(exc)}) + "\n"
                    break

        except Exception as exc:
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
