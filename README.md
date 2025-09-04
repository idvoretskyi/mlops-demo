# MLOps Demo (PyTorch + Kubernetes)

A tiny end-to-end MLOps tutorial that trains a PyTorch model, containerizes it, and serves predictions on Kubernetes with minimal setup.

[![CI](https://github.com/your-org/mlops-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/mlops-demo/actions/workflows/ci.yml)

A minimal, beginner-friendly MLOps demo showing:
- Training a tiny PyTorch model and saving an artifact
- Serving the model via a FastAPI endpoint
- Containerizing both steps with Docker
- Running training as a Kubernetes Job and serving as a Deployment with a shared PVC

No external datasets needed: we generate synthetic data for binary classification.

If you prefer a real sample dataset, you can use the built-in option to train on the scikit-learn Breast Cancer dataset by setting an environment variable (see "Datasets" section below).

## Repo Layout

```
.
├── src/
│   ├── train.py          # trains a model, saves to /model/model.pt
│   └── serve.py          # FastAPI inference service reading /model/model.pt
├── requirements.txt      # Python deps
├── Dockerfile.train      # builds training container
├── Dockerfile.serve      # builds serving container
├── k8s/
│   ├── pvc.yaml          # persistent storage for model
│   ├── train-job.yaml    # Kubernetes Job to run training once
│   ├── serve-deployment.yaml  # Deployment to run inference service
│   └── serve-service.yaml     # Service to expose the inference app in-cluster
├── scripts/
│   ├── install_tools_mac_linux.sh  # interactive tools checker/installer
│   ├── setup_python.sh             # create venv and install deps
│   └── k8s_quickstart.sh           # build images and deploy to k8s interactively
└── main.py               # original placeholder
```

## Local quickstart (optional)

### Interactive scripts (beginner-friendly)
- Check/install common tools (macOS/Linux):
  ./scripts/install_tools_mac_linux.sh
- Set up Python virtual environment and dependencies:
  ./scripts/setup_python.sh
- Build images and deploy to Kubernetes interactively (kind or registry):
  ./scripts/k8s_quickstart.sh

1) Setup environment

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Train locally

```
python src/train.py  # writes model to ./model if MODEL_DIR is set; default is /model
```

To save locally, set a writable directory:

```
export MODEL_DIR=./model
python src/train.py
ls ./model  # should contain model.pt
```

3) Serve locally

```
export MODEL_DIR=./model
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

Test:

```
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": [0.1, 0.2, 0.3, 0.0, 0.4, 0.1, 0.2, 0.3, 0.0, 0.4, 0.1, 0.2, 0.3, 0.0, 0.4, 0.1, 0.2, 0.3, 0.0, 0.4]}'
```

## Docker

Build images:

```
# Training image
docker build -t mlops-demo-train -f Dockerfile.train .

# Serving image
docker build -t mlops-demo-serve -f Dockerfile.serve .
```

Run training with a mounted local folder for artifact:

```
mkdir -p ./model
docker run --rm -e EPOCHS=3 -v $(pwd)/model:/model mlops-demo-train
```

Run serving using the produced artifact:

```
docker run --rm -p 8000:8000 -v $(pwd)/model:/model mlops-demo-serve
```

## Kubernetes

Prereqs: a Kubernetes cluster and kubectl context configured. Push images to a registry you can pull from and update image fields in k8s/*.yaml (they are placeholders `ghcr.io/your-org/...`).

1) Create PVC for model storage

```
kubectl apply -f k8s/pvc.yaml
```

2) Run training as a Job (writes /model/model.pt to the PVC)

Edit `k8s/train-job.yaml` to point to your registry image, then:

```
kubectl apply -f k8s/train-job.yaml
kubectl logs job/mlops-demo-train
kubectl get pods -l job-name=mlops-demo-train
```

3) Deploy serving Deployment and Service (mounts the same PVC)

Edit `k8s/serve-deployment.yaml` image to your registry, then:

```
kubectl apply -f k8s/serve-deployment.yaml
kubectl apply -f k8s/serve-service.yaml
```

4) Test from inside the cluster

```
kubectl run tester --rm -it --image=curlimages/curl --restart=Never -- \
  sh -c "curl -s http://mlops-demo-serve/predict -X POST -H 'Content-Type: application/json' \
  -d '{\"features\":[0.1,0.2,0.3,0.0,0.4,0.1,0.2,0.3,0.0,0.4,0.1,0.2,0.3,0.0,0.4,0.1,0.2,0.3,0.0,0.4]}'"
```

## What this demonstrates (MLOps concepts)
- Reproducible environments via requirements and Docker images
- Separation of concerns between training and serving
- Artifact management via shared persistent volume
- Configuration via environment variables (hyperparameters, paths)
- Kubernetes primitives to orchestrate training (Job) and serving (Deployment/Service)

## Datasets: where to get one and how to use it
- Default: synthetic dataset is generated automatically; you don’t need to download anything.
- Sample real dataset (built-in option): set DATA_SOURCE=sklearn_breast_cancer to train on the scikit-learn Breast Cancer dataset.

Examples:

Local Python:
```
export MODEL_DIR=./model
export DATA_SOURCE=sklearn_breast_cancer
python src/train.py
```

Docker:
```
mkdir -p ./model
# Train on sklearn dataset
docker run --rm -e DATA_SOURCE=sklearn_breast_cancer -v $(pwd)/model:/model mlops-demo-train
```

Kubernetes (Job): add this environment variable to k8s/train-job.yaml under the trainer container:
```
- name: DATA_SOURCE
  value: "sklearn_breast_cancer"
```

If you want to bring your own dataset:
- Replace the dataset loader in src/train.py with your data loading logic (e.g., reading CSV/Parquet, image folders, etc.).
- Ensure tensors are shaped as: features X tensor of shape [N, D] (float32), labels y tensor of shape [N] (long/int64), and update in_features/classes accordingly.

A few good public dataset sources:
- Scikit-learn toy datasets: https://scikit-learn.org/stable/datasets/toy_dataset.html
- UCI Machine Learning Repository: https://archive.ics.uci.edu/
- Kaggle Datasets (requires account): https://www.kaggle.com/datasets

## Hardware and platform requirements
- CPU: No GPU required. The demo trains and serves on CPU by default. Optional CUDA GPU will be used automatically if available in your environment (not required).
- RAM: 2 GB free RAM is sufficient for the default settings (more is always better).
- Disk: ~1 GB free for Docker images and ~10–50 MB for the model and Python deps (more if you build multi-arch images).
- Network: Internet access is required to build images (pip install) and to pull base images unless you pre-cache or mirror them.
- Runtime: Training with default settings typically completes in seconds to a couple of minutes on a modern CPU.

## Running on Apple Silicon (M1/M2/M3/M4)
Yes—you can run this locally on an M1/M2/M3/M4 Mac, including a local Kubernetes cluster.

Notes:
- Container architecture: On Apple Silicon, Docker builds arm64 (linux/arm64) images by default. The provided Dockerfiles work on arm64.
- PyTorch wheels: The specified torch version has prebuilt wheels for Linux aarch64, so pip install inside the container should work on Apple Silicon (M1/M2/M3/M4).
- Multi-arch (optional): If you plan to run your images on amd64 clusters too, build multi-arch images using buildx, e.g.:
  docker buildx build --platform linux/amd64,linux/arm64 -t <registry>/mlops-demo-train:latest -f Dockerfile.train --push .
  docker buildx build --platform linux/amd64,linux/arm64 -t <registry>/mlops-demo-serve:latest -f Dockerfile.serve --push .
- Local only on M1: If you only run locally on your Mac, standard docker build is fine (arm64 only).

## Local Kubernetes on macOS (M1/M2/M3/M4)
You have a few options:
- Docker Desktop Kubernetes: Easiest path. It includes a default StorageClass so the provided PVC usually binds automatically.
- kind (Kubernetes in Docker): Lightweight. Note that kind does not ship a default dynamic storage provisioner—your PVC may remain Pending. Install a storage provisioner (e.g., Rancher local-path-provisioner) or switch the manifests to hostPath for demos.
  - To use local images with kind: build locally and then run:
    kind load docker-image mlops-demo-train:latest
    kind load docker-image mlops-demo-serve:latest
- Colima + k8s: Also works; ensure your Docker/nerdctl context matches the Colima VM and that a StorageClass is available.

Image pulling tips:
- Update k8s/*.yaml image fields to your own registry if you push images.
- For purely local clusters: with Docker Desktop, local images are usually visible directly. With kind, use kind load docker-image as shown above.
- imagePullPolicy is set to IfNotPresent in the manifests, which works well for local iterations.

Persistent volume tips:
- If your PVC stays Pending, verify you have a default StorageClass. Docker Desktop provides one named docker-desktop. For kind, install a provisioner like local-path-provisioner and set it as default, or temporarily replace the PVC with a hostPath volume in the manifests for demo purposes.

## Notes for learners
- Replace synthetic data with your dataset and data loaders
- Add experiment tracking (e.g., MLflow or Weights & Biases)
- Add CI to build and push images on commits
- Secure the service (auth/ingress) for production scenarios
