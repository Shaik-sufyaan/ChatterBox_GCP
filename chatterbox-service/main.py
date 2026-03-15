import base64
import audioop
import json
import io
import os
import re
import tempfile

import numpy as np
import scipy.signal
import torch
import torchaudio as ta
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from chatterbox.tts import ChatterboxTTS
from google.cloud import storage

app = FastAPI()

# Load model once at startup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTTS.from_pretrained(device=device)

gcs_client = storage.Client()
BUCKET = os.environ.get("GCS_BUCKET_NAME", "sayops-recordings")

# In-memory cache: agentId → local temp file path
# Avoids re-downloading the same voice reference on every call
_voice_ref_cache: dict[str, str] = {}


class SynthRequest(BaseModel):
    text: str
    voice_ref_url: str | None = None  # GCS URL: gs://bucket/voice-refs/agent-id.wav
    speed: float = 1.0


def download_voice_ref(gcs_url: str) -> str:
    """Download voice reference from GCS to a temp file, return local path.
    Caches result in memory so repeated calls for the same URL skip the download.
    """
    if gcs_url in _voice_ref_cache:
        cached = _voice_ref_cache[gcs_url]
        if os.path.exists(cached):
            return cached

    # gcs_url format: gs://bucket/path/to/file.wav
    parts = gcs_url.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    blob.download_to_filename(tmp.name)
    _voice_ref_cache[gcs_url] = tmp.name
    return tmp.name


def pcm_to_mulaw_b64(wav_tensor: torch.Tensor, sample_rate: int) -> str:
    """Convert a torchaudio tensor to base64 mulaw 8kHz."""
    pcm = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
    # Resample to 8kHz
    ratio = sample_rate // 8000
    pcm_8k = scipy.signal.resample_poly(pcm, 1, ratio)
    pcm_i16 = (np.clip(pcm_8k, -1.0, 1.0) * 32767).astype(np.int16)
    mulaw = audioop.lin2ulaw(pcm_i16.tobytes(), 2)
    return base64.b64encode(mulaw).decode()


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries for chunked streaming."""
    parts = re.split(r'(?<=[^A-Z][.!?])\s+', text)
    return [p for p in parts if p.strip()]


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    async def generate():
        voice_ref_path = None
        try:
            if req.voice_ref_url:
                voice_ref_path = download_voice_ref(req.voice_ref_url)

            sentences = split_sentences(req.text)
            for sentence in sentences:
                if not sentence.strip():
                    continue
                wav = model.generate(
                    text=sentence,
                    audio_prompt_path=voice_ref_path,  # None = default voice
                )
                b64 = pcm_to_mulaw_b64(wav, model.sr)
                yield json.dumps({"audio": b64}) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
        # Note: voice_ref_path is kept alive in the cache — do NOT delete it here

    return StreamingResponse(generate(), media_type="application/x-ndjson")
