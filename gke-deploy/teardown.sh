#!/usr/bin/env bash
# Chatterbox TTS — tear down all GKE resources
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-evently-486001}"
REGION="us-central1"
CLUSTER_NAME="speakops-voice"

gcloud container clusters get-credentials "${CLUSTER_NAME}" \
  --region="${REGION}" --project="${PROJECT_ID}"

echo "▶ Deleting K8s resources..."
kubectl delete -f chatterbox-service/k8s-hpa.yaml --ignore-not-found
kubectl delete -f chatterbox-service/k8s-deployment.yaml --ignore-not-found
kubectl delete secret chatterbox-api-key --ignore-not-found

echo "▶ Deleting GKE cluster (this deletes ALL node pools including GPU pool)..."
gcloud container clusters delete "${CLUSTER_NAME}" \
  --region="${REGION}" \
  --project="${PROJECT_ID}" \
  --quiet

echo "✅ Teardown complete."
