#!/usr/bin/env bash
set -euo pipefail

# Interactive helper to build images and deploy to Kubernetes
# Options:
# - Use kind (load local images)
# - Or push to a registry and update manifests on-the-fly (temporary overlay)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT"

have() { command -v "$1" >/dev/null 2>&1; }

if ! have docker; then
  echo "Docker is required to build images. Please install Docker first." >&2
  exit 1
fi

if ! have kubectl; then
  echo "kubectl is required to apply manifests. Please install kubectl first." >&2
  exit 1
fi

use_kind="n"
if have kind; then
  read -r -p "Use kind (auto-load local images) [y/N]? " use_kind || true
fi

TRAIN_IMG_DEFAULT="mlops-demo-train:latest"
SERVE_IMG_DEFAULT="mlops-demo-serve:latest"

if [[ "${use_kind,,}" == "y" ]]; then
  REGISTRY=""
  TRAIN_IMG="$TRAIN_IMG_DEFAULT"
  SERVE_IMG="$SERVE_IMG_DEFAULT"
else
  read -r -p "Enter registry prefix (e.g., ghcr.io/your-org) or leave empty to use local tags only: " REGISTRY || true
  if [ -n "${REGISTRY:-}" ]; then
    TRAIN_IMG="$REGISTRY/mlops-demo-train:latest"
    SERVE_IMG="$REGISTRY/mlops-demo-serve:latest"
  else
    TRAIN_IMG="$TRAIN_IMG_DEFAULT"
    SERVE_IMG="$SERVE_IMG_DEFAULT"
  fi
fi

echo "\n==> Building images"
docker build -t "$TRAIN_IMG" -f Dockerfile.train .
docker build -t "$SERVE_IMG" -f Dockerfile.serve .

if [[ "${use_kind,,}" == "y" ]]; then
  echo "\n==> Loading images into kind"
  kind load docker-image "$TRAIN_IMG"
  kind load docker-image "$SERVE_IMG"
else
  if [ -n "${REGISTRY:-}" ]; then
    echo "\n==> Pushing images"
    docker push "$TRAIN_IMG"
    docker push "$SERVE_IMG"
  else
    echo "\nNo registry provided; ensure your cluster can pull local images (Docker Desktop usually works)."
  fi
fi

# Apply manifests, optionally override image fields at apply-time
K8S_DIR="k8s"
TRAIN_JOB="$K8S_DIR/train-job.yaml"
SERVE_DEPLOY="$K8S_DIR/serve-deployment.yaml"
SERVE_SVC="$K8S_DIR/serve-service.yaml"
PVC="$K8S_DIR/pvc.yaml"

apply_with_image() {
  local file="$1" image="$2" path="$3"
  # Use kubectl set image after apply to avoid editing files
  kubectl apply -f "$file"
  kubectl set image -f "$file" "$path"="$image" --local -o yaml | kubectl apply -f -
}

echo "\n==> Applying PVC"
kubectl apply -f "$PVC"

echo "\n==> Applying training Job"
apply_with_image "$TRAIN_JOB" "$TRAIN_IMG" "job/mlops-demo-train=trainer=$TRAIN_IMG" || true

echo "\nYou can monitor logs with: kubectl logs -f job/mlops-demo-train"

read -r -p $'Press Enter after the Job completes to deploy serving...\n' _ || true

echo "\n==> Applying serving Deployment and Service"
apply_with_image "$SERVE_DEPLOY" "$SERVE_IMG" "deployment/mlops-demo-serve=server=$SERVE_IMG" || true
kubectl apply -f "$SERVE_SVC"

echo "\n==> Done. Test from inside the cluster (example):"
echo "kubectl run tester --rm -it --image=curlimages/curl --restart=Never -- sh -c \"curl -s http://mlops-demo-serve/predict -X POST -H 'Content-Type: application/json' -d '{\\"features\\":[0.1,0.2,0.3,0.0,0.4,0.1,0.2,0.3,0.0,0.4,0.1,0.2,0.3,0.0,0.4,0.1,0.2,0.3,0.0,0.4]}'\""
