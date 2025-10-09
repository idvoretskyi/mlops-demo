import os
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from src.models import SimpleNet, CheckpointMeta, save_checkpoint  # when run as a script
except ImportError:  # pragma: no cover
    from models import SimpleNet, CheckpointMeta, save_checkpoint


def get_synthetic_dataset(n_samples: int = 10_000, n_features: int = 20):
    torch.manual_seed(42)
    X = torch.randn(n_samples, n_features)
    # Create a simple linear boundary with noise
    weights = torch.randn(n_features)
    logits = X @ weights + 0.5 * torch.randn(n_samples)
    y = (logits > 0).long()
    return TensorDataset(X, y)


def train(
    epochs: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    model_dir: str = "/model",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_source = os.getenv("DATA_SOURCE", "synthetic").lower()
    if data_source == "sklearn_breast_cancer":
        try:
            from sklearn.datasets import load_breast_cancer
        except Exception as e:
            raise RuntimeError(
                "scikit-learn is required for DATA_SOURCE=sklearn_breast_cancer. "
                "Add it to requirements and rebuild."
            ) from e
        ds_raw = load_breast_cancer()
        X = torch.tensor(ds_raw.data, dtype=torch.float32)
        y = torch.tensor(ds_raw.target, dtype=torch.long)
        n_features = X.shape[1]
        ds = TensorDataset(X, y)
        num_classes = len(torch.unique(y))
        print(
            "using_dataset=sklearn_breast_cancer "
            f"n_samples={len(ds)} n_features={n_features} "
            f"classes={num_classes}"
        )
    else:
        n_features = 20
        ds = get_synthetic_dataset(n_samples=10_000, n_features=n_features)
        num_classes = 2
        print(
            "using_dataset=synthetic "
            f"n_samples=10000 n_features={n_features} "
            f"classes={num_classes}"
        )

    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = SimpleNet(in_features=n_features, out_features=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        avg_loss = total_loss / total
        acc = correct / total
        print(f"epoch={epoch} loss={avg_loss:.4f} acc={acc:.4f}")

    duration = time.time() - start
    print(f"training_done duration_sec={duration:.2f}")

    # Save model
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(model_dir, "model.pt")
    meta = CheckpointMeta(
        in_features=n_features,
        classes=num_classes,
        data_source=os.getenv("DATA_SOURCE", "synthetic"),
    )
    save_checkpoint(model, model_path, meta)
    print(f"model_saved path={model_path}")


if __name__ == "__main__":
    # Read hyperparams from env (K8s-friendly)
    epochs = int(os.getenv("EPOCHS", "5"))
    batch = int(os.getenv("BATCH_SIZE", "128"))
    lr = float(os.getenv("LR", "0.001"))
    model_dir = os.getenv("MODEL_DIR", "/model")
    train(epochs=epochs, batch_size=batch, lr=lr, model_dir=model_dir)
