# Chatterbox GCP Deployment & Integration Plan

## Context

Chatterbox is the **future** voice cloning layer for SpeakOps — used when a business owner wants their AI agent to speak in a custom cloned voice (e.g., their own voice, a brand voice, or a specific persona). Kokoro handles the platform default voices. Chatterbox handles voice-specific agents.

- **When to use Chatterbox**: Customer provides a 10-second voice reference clip → agent speaks in that voice
- **When to use Kokoro**: No reference voice provided → use platform default (`af_heart`, etc.)
- **License**: MIT (Resemble AI)
- **Model**: Chatterbox-Turbo (350M params, lowest latency of the Chatterbox family)

---

## Part 1: Chatterbox vs Kokoro — Which to Use When

| Scenario | Service |
|---|---|
| New agent, no custom voice | Kokoro (`af_heart` default) |
| Agent with uploaded voice sample | Chatterbox (voice cloning) |
| Platform default, fast response needed | Kokoro |
| Brand/persona voice required | Chatterbox |

The `tts.service.ts` factory function selects the backend based on whether the agent has a `tts_voice_ref_url` set (GCS URL to the reference WAV).

---

## Part 2: Audio Format Conversion

**Chatterbox output**: 24kHz PCM float32 (same as Kokoro)
**Twilio requires**: mulaw 8kHz base64

Identical conversion pipeline to the Kokoro service:

```
Chatterbox 24kHz PCM (float32 tensor)
  → scipy.signal.resample_poly(audio, 1, 3)   # 24kHz → 8kHz
  → (clip * 32767).astype(int16)               # float32 → int16
  → audioop.lin2ulaw(bytes, 2)                  # int16 → mulaw
  → base64.b64encode()                          # → base64
  → stream as ndjson: {"audio": "<base64>"}
```

---

## Part 3: GPU Requirements

Chatterbox requires a CUDA GPU. **Cloud Run does not support GPUs** — deploy on **GKE with GPU node pool**.

| GPU | VRAM | RTF (Turbo) | GCP type | Cost/month |
|---|---|---|---|---|
| T4 | 16GB | ~0.3-0.4 | `nvidia-tesla-t4` | ~$180 always-on |
| L4 | 24GB | ~0.2 | `nvidia-l4` | ~$280 always-on |
| A100 (40GB) | 40GB | ~0.1 | `nvidia-tesla-a100` | ~$2,100 |

**Recommendation**: Start with T4. At RTF ~0.35, a single T4 handles ~3 concurrent real-time calls. Add nodes as needed.

---

## Part 4: Chatterbox Service Files

### `chatterbox-service/requirements.txt`
```
fastapi>=0.115.0
uvicorn>=0.30.0
chatterbox-tts>=0.1.0
torch>=2.2.0
torchaudio>=2.2.0
scipy>=1.13.0
numpy>=1.26.0
google-cloud-storage>=2.14.0
```

### `chatterbox-service/main.py`
```python
import base64, audioop, json, io, os, tempfile
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

class SynthRequest(BaseModel):
    text: str
    voice_ref_url: str | None = None  # GCS URL: gs://bucket/voice-refs/agent-id.wav
    speed: float = 1.0

def download_voice_ref(gcs_url: str) -> str:
    """Download voice reference from GCS to a temp file, return local path."""
    # gcs_url format: gs://bucket/path/to/file.wav
    parts = gcs_url.replace("gs://", "").split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    blob.download_to_filename(tmp.name)
    return tmp.name

def pcm_to_mulaw_b64(wav_tensor: torch.Tensor, sample_rate: int) -> str:
    """Convert a torchaudio tensor to base64 mulaw 8kHz."""
    pcm = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
    # Resample to 8kHz
    pcm_8k = scipy.signal.resample_poly(pcm, 1, sample_rate // 8000)
    pcm_i16 = (np.clip(pcm_8k, -1.0, 1.0) * 32767).astype(np.int16)
    mulaw = audioop.lin2ulaw(pcm_i16.tobytes(), 2)
    return base64.b64encode(mulaw).decode()

@app.get("/health")
def health():
    return {"status": "ok", "device": device}

@app.post("/synthesize")
async def synthesize(req: SynthRequest):
    async def generate():
        voice_ref_path = None
        try:
            # Download voice reference from GCS if provided
            if req.voice_ref_url:
                voice_ref_path = download_voice_ref(req.voice_ref_url)

            # Chatterbox generates full audio (not streaming natively)
            # Split into sentences for streaming chunks
            sentences = split_sentences(req.text)
            for sentence in sentences:
                if not sentence.strip():
                    continue
                wav = model.generate(
                    text=sentence,
                    audio_prompt_path=voice_ref_path,  # None = use default voice
                )
                b64 = pcm_to_mulaw_b64(wav, model.sr)
                yield json.dumps({"audio": b64}) + "\n"

        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
        finally:
            if voice_ref_path and os.path.exists(voice_ref_path):
                os.unlink(voice_ref_path)

    return StreamingResponse(generate(), media_type="application/x-ndjson")

def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries for chunked streaming."""
    import re
    # Split on .!? but not on abbreviations (Mr., Dr., etc.)
    parts = re.split(r'(?<=[^A-Z][.!?])\s+', text)
    return [p for p in parts if p.strip()]
```

### `chatterbox-service/Dockerfile`
```dockerfile
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Chatterbox model at build time
RUN python -c "from chatterbox.tts import ChatterboxTTS; ChatterboxTTS.from_pretrained(device='cpu')" || true

COPY main.py .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

---

## Part 5: GKE Deployment (NOT Cloud Run)

Cloud Run does not support GPU workloads. Use GKE with a GPU node pool.

### Step 1 — Create GKE cluster with GPU node pool

```bash
# Create cluster (if not exists)
gcloud container clusters create speakops-voice \
  --region us-central1 \
  --num-nodes 1 \
  --machine-type n1-standard-4 \
  --project $PROJECT_ID

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster speakops-voice \
  --region us-central1 \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 1 \
  --min-nodes 1 \
  --max-nodes 3 \
  --enable-autoscaling \
  --project $PROJECT_ID

# Install NVIDIA GPU drivers on the cluster
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Step 2 — Build and push Docker image

```bash
# Build and push to Artifact Registry
gcloud builds submit --tag us-central1-docker.pkg.dev/$PROJECT_ID/evently-repo/chatterbox-service:latest \
  chatterbox-service/

# Or build locally:
docker build -t us-central1-docker.pkg.dev/$PROJECT_ID/evently-repo/chatterbox-service:latest chatterbox-service/
docker push us-central1-docker.pkg.dev/$PROJECT_ID/evently-repo/chatterbox-service:latest
```

### Step 3 — Kubernetes deployment manifest

File: `chatterbox-service/k8s-deployment.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: chatterbox-service
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: chatterbox-service
  template:
    metadata:
      labels:
        app: chatterbox-service
    spec:
      nodeSelector:
        cloud.google.com/gke-accelerator: nvidia-tesla-t4
      containers:
        - name: chatterbox
          image: us-central1-docker.pkg.dev/PROJECT_ID/evently-repo/chatterbox-service:latest
          ports:
            - containerPort: 8080
          resources:
            limits:
              nvidia.com/gpu: 1
              memory: "8Gi"
              cpu: "4"
            requests:
              memory: "6Gi"
              cpu: "2"
          env:
            - name: GCS_BUCKET_NAME
              value: "sayops-recordings"
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 60
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: chatterbox-service
  namespace: default
spec:
  selector:
    app: chatterbox-service
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
```

```bash
# Deploy
kubectl apply -f chatterbox-service/k8s-deployment.yaml

# Get internal cluster IP
kubectl get svc chatterbox-service
```

The backend calls Chatterbox via the internal Kubernetes ClusterIP service URL.

---

## Part 6: Voice Reference Storage

Voice reference WAV files are stored in GCS at a predictable path:

```
gs://sayops-recordings/voice-refs/{agentId}.wav
```

**Upload flow** (new API endpoint needed):
1. Org member uploads a 10s WAV via `POST /api/agents/:agentId/voice-ref`
2. Backend validates: WAV format, 7-20s duration, 16kHz+ sample rate
3. Saved to `gs://sayops-recordings/voice-refs/{agentId}.wav`
4. Agent record updated: `tts_voice_ref_url = "gs://sayops-recordings/voice-refs/{agentId}.wav"`

When `tts.service.ts` creates a TTS stream, it reads `agent.tts_voice_ref_url` and passes it to the Chatterbox service.

---

## Part 7: `tts.service.ts` — Routing Kokoro vs Chatterbox

Add a `backendUrl` parameter to `createTTSStream()` — `voice.service.ts` passes the right URL based on whether the agent has a voice reference:

```typescript
// In voice.service.ts initAudioPipeline():
const ttsUrl = session.phoneUser?.tts_voice_ref_url
  ? process.env.CHATTERBOX_SERVICE_URL!    // has custom voice
  : process.env.KOKORO_SERVICE_URL!        // platform default

session.tts = createTTSStream(
  (audioBase64) => { if (session.sendAudio) session.sendAudio(audioBase64) },
  () => { session.isSpeaking = false },
  { voiceId: session.phoneUser?.tts_voice_id, voiceRefUrl: session.phoneUser?.tts_voice_ref_url },
  ttsUrl,
  ttsTraceCtx
)
```

The `POST /synthesize` body is the same shape for both Kokoro and Chatterbox — Chatterbox just adds `voice_ref_url`. Kokoro ignores unknown fields.

---

## Part 8: New DB Column & Agent Field

Add to `agents` table via migration:
```sql
ALTER TABLE agents
  ADD COLUMN tts_voice_ref_url TEXT,     -- GCS URL to reference WAV
  ADD COLUMN tts_voice_id      TEXT;     -- Kokoro voice ID (overrides platform default)
```

Migration file: `migrations/042_agents_tts_voice.sql`

---

## Part 9: New Environment Variables

| Variable | Value | Notes |
|---|---|---|
| `CHATTERBOX_SERVICE_URL` | GKE internal or external URL | e.g., `http://chatterbox-service.default.svc.cluster.local` |

Add to `cloudbuild.yaml` `--set-env-vars`:
```
--set-env-vars=CHATTERBOX_SERVICE_URL=$_CHATTERBOX_SERVICE_URL
```

Add substitution:
```yaml
_CHATTERBOX_SERVICE_URL: 'http://chatterbox-service.default.svc.cluster.local'
```

Note: If backend (Cloud Run) and Chatterbox (GKE) are in different VPCs, use a private load balancer or expose via internal IP. Simplest: add a `LoadBalancer` service with internal annotation.

---

## Part 10: Cost

| Component | Config | Monthly cost |
|---|---|---|
| GKE n1-standard-4 (1 node) | Always-on | ~$120 |
| T4 GPU (1 unit) | Always-on | ~$180 |
| Node pool autoscale (2nd node) | Only under load | +~$300 burst |
| **Total baseline** | 1 node + 1 T4 | **~$300/month** |

Shared across all businesses using custom voices. At 5 businesses with voice cloning → **$60/business/month** for GPU TTS (vs ~$45/business on ElevenLabs alone).

---

## Part 11: Latency Profile

| Step | Chatterbox Turbo / T4 | Notes |
|---|---|---|
| GCS voice ref download | 20-50ms | Cached after first call per pod |
| Model inference (sentence) | 150-300ms | RTF ~0.35 on T4 |
| mulaw conversion | ~5ms | |
| First audio chunk | ~200-380ms | |
| **Total (STT + LLM + TTS)** | ~600-800ms | Slightly higher than Kokoro |

Optimization: cache downloaded voice reference WAV in pod memory (Python dict keyed by `agentId`) — eliminates GCS download on every call.

---

## Part 12: Verification Steps

1. `GET http://chatterbox-service.../health` → `{"status":"ok","device":"cuda"}`
2. `POST /synthesize {"text":"Hello","voice_ref_url":null}` → ndjson `{"audio":"..."}` using default voice
3. `POST /synthesize {"text":"Hello","voice_ref_url":"gs://sayops-recordings/voice-refs/test.wav"}` → cloned voice audio
4. Decode mulaw and play: `sox -t ul -r 8000 -c 1 audio.mulaw -d`
5. Voice call with custom-voice agent: speak → hear cloned voice within ~700ms
6. Barge-in: speak while agent talks → stops immediately
7. `kubectl get pods` → chatterbox pod Running, GPU allocated

---

## Part 13: Timeline & Phases

| Phase | Work | Duration |
|---|---|---|
| 1. GKE cluster + GPU node pool | gcloud commands, NVIDIA driver install | 1 day |
| 2. Chatterbox service | `main.py`, `Dockerfile`, smoke test on T4 | 3 days |
| 3. Voice ref upload API | `POST /api/agents/:id/voice-ref`, GCS save, DB migration | 2 days |
| 4. `tts.service.ts` routing | Kokoro vs Chatterbox URL selection by agent config | 1 day |
| 5. End-to-end testing | 20+ calls with cloned voice, latency profiling | 3 days |
| **Total** | | **~2 weeks** |

**This is Phase 2** — implement after the Kokoro/Deepgram base system is working (Phase 1).
