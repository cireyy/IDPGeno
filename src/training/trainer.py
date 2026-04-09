import os
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.metrics import compute_binary_classification_metrics


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probs = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(
                idp=batch["idp"],
                geno_ids=batch["geno_ids"],
                pos_values=batch["pos_values"],
                func_values=batch["func_values"],
                snp_mask=batch["snp_mask"],
            )

            logits = outputs["logits"]
            labels = batch["label"]

            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels_np)

    mean_loss = total_loss / max(total_samples, 1)

    y_true = np.concatenate(all_labels, axis=0)
    y_prob = np.concatenate(all_probs, axis=0)

    metrics = compute_binary_classification_metrics(y_true=y_true, y_prob=y_prob)

    return mean_loss, metrics


def build_loss_fn(train_labels: np.ndarray, device: torch.device) -> nn.Module:
    train_labels = np.asarray(train_labels).astype(int)
    num_pos = int((train_labels == 1).sum())
    num_neg = int((train_labels == 0).sum())

    if num_pos == 0:
        pos_weight = 1.0
    else:
        pos_weight = num_neg / max(num_pos, 1)

    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)


def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        },
        path,
    )