# ChatterBox TTS Service

Voice cloning microservice using [Chatterbox-Turbo](https://github.com/resemble-ai/chatterbox) (350M params, MIT license). Runs on GKE with T4 GPU. Handles agents that have a custom voice reference — Kokoro handles platform default voices.

---

## When to Use Chatterbox vs Kokoro

| Scenario | Service |
|---|---|
| No custom voice uploaded | **Kokoro** (`af_heart` default, GCE MIG) |
| Agent has uploaded voice sample | **Chatterbox** (voice cloning, GKE T4) |

Routing happens in `tts.service.ts` based on whether `agent.tts_voice_ref_url` is set.

---

## Architecture

```
Twilio call
  → Deepgram STT
  → LLM (Gemini / OpenAI)
  → tts.service.ts
      agent.tts_voice_ref_url set?
        YES → POST http://chatterbox-service.default.svc.cluster.local/synthesize
        NO  → POST http://136.110.174.100/synthesize  (Kokoro)
  → ndjson stream: {"audio": "<base64 mulaw 8kHz>"}
  → Twilio plays audio
```

**Audio pipeline (inside Chatterbox service):**
```
Chatterbox 24kHz float32 tensor
  → scipy.signal.resample_poly(audio, 8000, 24000)  # 24kHz → 8kHz
  → clip([-1,1]) * 32767 → int16
  → audioop.lin2ulaw(bytes, 2)                       # int16 → mulaw
  → base64.b64encode()
  → ndjson: {"audio": "<base64>"}
```

---

## Infrastructure

| Component | Value |
|---|---|
| Platform | GKE (not Cloud Run — GPU required) |
| Machine | n1-standard-4 + nvidia-tesla-t4 |
| VRAM | 16GB (T4) |
| Region | us-central1 |
| Cluster | speakops-voice |
| Min pods | 1 (always warm — GPU cold start = ~60s) |
| Max pods | 3 |
| Cost | ~$300/month (1 always-on T4 node) |

---

## Latency Profile

| Step | Time |
|---|---|
| GCS voice ref download (first call) | 20–50ms (cached after) |
| Model inference per sentence (T4) | 150–300ms (RTF ~0.35) |
| mulaw conversion | ~5ms |
| **First audio chunk to caller** | ~200–380ms |
| **Total (STT + LLM + TTS)** | ~600–800ms |

---

## API

### `GET /health`
```json
{"status": "ok", "device": "cuda"}
```

### `POST /synthesize`

**Headers:**
```
Content-Type: application/json
x-api-key: <CHATTERBOX_API_KEY>
```

**Request:**
```json
{
  "text": "Hello, how can I help you today?",
  "voice_ref_url": "gs://sayops-recordings/voice-refs/agent-id.wav",
  "speed": 1.0
}
```
`voice_ref_url` is optional. Omit for Chatterbox's built-in default voice.

**Response:** `application/x-ndjson` stream, one object per sentence:
```
{"audio": "<base64-encoded mulaw 8kHz>"}
{"audio": "<base64-encoded mulaw 8kHz>"}
...
```

---

## Voice Reference Storage

Voice reference WAV files live in GCS at:
```
gs://sayops-recordings/voice-refs/{agentId}.wav
```

Requirements:
- WAV format, mono
- 10–30 seconds, single speaker
- 16kHz+ sample rate, no background noise

---

## Initial Deployment (run once)

```bash
export PROJECT_ID=evently-486001
export CHATTERBOX_API_KEY=your-secret-key
chmod +x gke-deploy/deploy.sh
./gke-deploy/deploy.sh
```

The script creates the GKE cluster, GPU node pool, installs NVIDIA drivers, builds the Docker image, and deploys.

**Tear down:**
```bash
./gke-deploy/teardown.sh
```

---

## Continuous Deployment (automatic)

Every push to `main` triggers Cloud Build (`cloudbuild.yaml`):
1. Builds + pushes Docker image tagged with `$COMMIT_SHA`
2. `kubectl set image` rolling update — zero downtime

Set `_CHATTERBOX_API_KEY` in the Cloud Build trigger substitution variables.

---

## Connecting Backend (Cloud Run) → Chatterbox (GKE)

Cloud Run and GKE are in separate VPC networks by default. Options:

| Option | Latency | Setup effort |
|---|---|---|
| Internal LoadBalancer (recommended) | ~2ms | Medium |
| VPC peering | ~1ms | High |
| External IP (simplest) | ~5ms | None |

**Simplest — expose via internal LoadBalancer:**
```bash
kubectl patch svc chatterbox-service \
  -p '{"spec":{"type":"LoadBalancer"},"metadata":{"annotations":{"cloud.google.com/load-balancer-type":"Internal"}}}'
# Get the internal IP:
kubectl get svc chatterbox-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

Then set in Cloud Run backend:
```
CHATTERBOX_SERVICE_URL=http://<internal-lb-ip>
```

---

## Backend Integration

In `tts.service.ts`, route based on agent config:

```typescript
const ttsUrl = agent.tts_voice_ref_url
  ? process.env.CHATTERBOX_SERVICE_URL!
  : process.env.KOKORO_SERVICE_URL!

const tts = createTTSStream(
  (audioBase64) => sendToTwilio(audioBase64),
  () => console.log('TTS done'),
  {
    voiceRefUrl: agent.tts_voice_ref_url ?? undefined,
  },
  ttsUrl,
)
tts.synthesize('Hello, how can I help you today?')
```

---

## Environment Variables

| Variable | Example | Notes |
|---|---|---|
| `CHATTERBOX_SERVICE_URL` | `http://10.x.x.x` | GKE internal LB IP |
| `CHATTERBOX_API_KEY` | `your-secret-key` | Must match `_CHATTERBOX_API_KEY` in Cloud Build trigger |
| `GCS_BUCKET_NAME` | `sayops-recordings` | Bucket containing voice-refs/ |

---

## Verification Checklist

- [ ] `GET /health` → `{"status":"ok","device":"cuda"}`
- [ ] `POST /synthesize` (no voice ref) → ndjson `{"audio":"..."}` using default voice
- [ ] `POST /synthesize` (with `voice_ref_url`) → cloned voice audio
- [ ] Request without `x-api-key` → 401
- [ ] `kubectl get pods` → Running, GPU allocated (`nvidia.com/gpu: 1`)
- [ ] End-to-end call with custom-voice agent → hear cloned voice within ~700ms
- [ ] Scale test: 4+ concurrent calls → HPA scales to 2 pods
