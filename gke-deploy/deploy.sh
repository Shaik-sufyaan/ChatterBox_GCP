#!/usr/bin/env bash
# Chatterbox TTS — GKE GPU deployment (run once)
# Creates: GKE cluster, T4 GPU node pool, Artifact Registry repo, K8s deployment + HPA
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ID="${PROJECT_ID:-evently-486001}"
REGION="us-central1"
CLUSTER_NAME="speakops-voice"
NODE_POOL_NAME="gpu-pool"
IMAGE_REPO="us-central1-docker.pkg.dev/${PROJECT_ID}/evently-repo"
IMAGE="${IMAGE_REPO}/chatterbox-service:latest"
CHATTERBOX_API_KEY="${CHATTERBOX_API_KEY:-}"

if [[ -z "$CHATTERBOX_API_KEY" ]]; then
  echo "ERROR: Set CHATTERBOX_API_KEY env var before running this script."
  exit 1
fi

# ── 1. Enable required APIs ───────────────────────────────────────────────────
echo "▶ Enabling APIs..."
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project="${PROJECT_ID}"

# ── 2. Artifact Registry repo ─────────────────────────────────────────────────
echo "▶ Creating Artifact Registry repo..."
gcloud artifacts repositories create evently-repo \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT_ID}" 2>/dev/null || echo "  (already exists)"

# ── 3. GKE cluster ────────────────────────────────────────────────────────────
echo "▶ Creating GKE cluster (if not exists)..."
gcloud container clusters describe "${CLUSTER_NAME}" \
  --region="${REGION}" --project="${PROJECT_ID}" &>/dev/null || \
gcloud container clusters create "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --num-nodes=1 \
  --machine-type=e2-standard-2 \
  --no-enable-autoupgrade \
  --project="${PROJECT_ID}"

# ── 4. GPU node pool ──────────────────────────────────────────────────────────
# n1-standard-4 + T4: 4 vCPU, 15GB RAM, 16GB VRAM
# min=1 always-on (cold start on GPU pod = ~60s — unacceptable for live calls)
echo "▶ Creating GPU node pool (if not exists)..."
gcloud container node-pools describe "${NODE_POOL_NAME}" \
  --cluster="${CLUSTER_NAME}" --region="${REGION}" --project="${PROJECT_ID}" &>/dev/null || \
gcloud container node-pools create "${NODE_POOL_NAME}" \
  --cluster="${CLUSTER_NAME}" \
  --region="${REGION}" \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=1 \
  --min-nodes=1 \
  --max-nodes=3 \
  --enable-autoscaling \
  --no-enable-autoupgrade \
  --project="${PROJECT_ID}"

# ── 5. NVIDIA GPU drivers ─────────────────────────────────────────────────────
echo "▶ Installing NVIDIA GPU drivers on cluster..."
gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --region="${REGION}" --project="${PROJECT_ID}"
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# ── 6. Build & push image ─────────────────────────────────────────────────────
echo "▶ Building and pushing Docker image..."
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
gcloud builds submit \
  --tag "${IMAGE}" \
  ./chatterbox-service/ \
  --project="${PROJECT_ID}"

# ── 7. API key secret ─────────────────────────────────────────────────────────
echo "▶ Creating API key secret in cluster..."
kubectl create secret generic chatterbox-api-key \
  --from-literal=key="${CHATTERBOX_API_KEY}" \
  --dry-run=client -o yaml | kubectl apply -f -

# ── 8. Deploy ─────────────────────────────────────────────────────────────────
echo "▶ Deploying to GKE..."
# Replace PROJECT_ID placeholder in manifest
sed "s/PROJECT_ID/${PROJECT_ID}/g" chatterbox-service/k8s-deployment.yaml | kubectl apply -f -
kubectl apply -f chatterbox-service/k8s-hpa.yaml

# Wait for rollout
kubectl rollout status deployment/chatterbox-service --timeout=300s

# ── Done ──────────────────────────────────────────────────────────────────────
echo "▶ Waiting for internal load balancer IP..."
LB_IP=""
for _ in {1..60}; do
  LB_IP=$(kubectl get svc chatterbox-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  if [[ -n "${LB_IP}" ]]; then
    break
  fi
  sleep 10
done

echo ""
echo "✅ Deployment complete."
echo ""
echo "  Internal service URL : http://chatterbox-service.default.svc.cluster.local/synthesize"
if [[ -n "${LB_IP}" ]]; then
  echo "  Internal LB IP       : http://${LB_IP}/synthesize"
  echo "  Health check         : http://${LB_IP}/health"
else
  echo "  Internal LB IP       : pending"
  echo "  Check with           : kubectl get svc chatterbox-service"
fi
echo ""
echo "Set in your backend (Cloud Run env vars):"
if [[ -n "${LB_IP}" ]]; then
  echo "  CHATTERBOX_SERVICE_URL=http://${LB_IP}"
else
  echo "  CHATTERBOX_SERVICE_URL=http://<internal-lb-ip>"
fi
echo ""
echo "NOTE: Cloud Run cannot reach a GKE ClusterIP directly."
echo "      This service is exposed through an internal GKE load balancer."
