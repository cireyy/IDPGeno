import argparse
import copy
import os
import random
from typing import Any, Dict, List
import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from data.dataset import IDPGenoProcessedDataset
from models.idpgeno_model import IDPGenoModel
from training.metrics import format_metrics
from training.trainer import run_one_epoch, build_loss_fn, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train IDPGeno")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/configs.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_dataset(config: Dict[str, Any]) -> IDPGenoProcessedDataset:
    data_cfg = config["data"]

    dataset = IDPGenoProcessedDataset(
        samples_csv=data_cfg["samples_csv"],
        snp_tokens_csv=data_cfg["snp_tokens_csv"],
        participant_col=data_cfg.get("participant_col", "participant_id"),
        label_col=data_cfg.get("label_col", "label"),
        gene_col=data_cfg.get("gene_col", "gene_id"),
        token_index_col=data_cfg.get("token_index_col", "token_index_within_gene"),
        geno_col=data_cfg.get("geno_col", "geno_code"),
        pos_col=data_cfg.get("pos_col", "relative_pos"),
        func_cols=data_cfg.get("func_cols", ["maf", "within_gene_weight_a"]),
        mask_col=data_cfg.get("mask_col", "is_valid"),
        normalize_idp=data_cfg.get("normalize_idp", True),
    )
    return dataset


def build_model(feature_info: Dict[str, Any], config: Dict[str, Any]) -> IDPGenoModel:
    model_cfg = config["model"]

    token_cfg = model_cfg["token_embedding"]
    gene_cfg = model_cfg["gene_encoder"]
    idp_cfg = model_cfg["idp_encoder"]
    backbone_cfg = model_cfg["backbone"]
    classifier_cfg = model_cfg["classifier"]

    model = IDPGenoModel(
        num_idps=feature_info["num_idps"],
        num_genes=feature_info["num_genes"],
        max_snps_per_gene=feature_info["max_snps_per_gene"],
        func_dim=feature_info["func_dim"],
        genotype_vocab_size=model_cfg.get("genotype_vocab_size", 3),
        token_embed_dim=token_cfg["token_embed_dim"],
        gene_embed_dim=gene_cfg["gene_embed_dim"],
        idp_hidden_dim=idp_cfg["hidden_dim"],
        backbone_num_layers=backbone_cfg["num_layers"],
        backbone_num_heads=backbone_cfg["num_heads"],
        backbone_ff_hidden_dim=backbone_cfg["ff_hidden_dim"],
        backbone_num_experts=backbone_cfg["num_experts"],
        classifier_hidden_dim=classifier_cfg["hidden_dim"],
        dropout=token_cfg.get("dropout", 0.1),
    )
    return model


def build_optimizer(model: IDPGenoModel, config: Dict[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = config["training"]["optimizer"]
    opt_type = opt_cfg.get("type", "adam").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)

    if opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer type: {opt_type}")


def subset_labels(dataset: IDPGenoProcessedDataset, indices: np.ndarray) -> np.ndarray:
    return dataset.labels[indices]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    project_cfg = config["project"]
    train_cfg = config["training"]
    ckpt_cfg = train_cfg["checkpoint"]
    eval_cfg = config["evaluation"]

    set_seed(project_cfg.get("seed", 42))
    device = get_device(train_cfg.get("device", "auto"))
    print(f"Using device: {device}")
    print(f"Config: {args.config}")

    dataset = build_dataset(config)
    feature_info = dataset.get_feature_info()
    labels = dataset.labels.astype(int)

    num_folds = train_cfg["num_folds"]
    num_epochs = train_cfg["num_epochs"]
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg.get("num_workers", 0)
    save_dir = ckpt_cfg["dir"]
    os.makedirs(save_dir, exist_ok=True)

    skf = StratifiedKFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=project_cfg.get("seed", 42),
    )

    fold_results: List[Dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        print("\n" + "=" * 100)
        print(f"Fold {fold_idx}/{num_folds}")
        print("=" * 100)

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        model = build_model(feature_info, config).to(device)
        criterion = build_loss_fn(subset_labels(dataset, train_idx), device=device)
        optimizer = build_optimizer(model, config)

        best_auc = -1.0
        best_epoch = -1
        best_metrics = None
        best_state = None

        for epoch in range(1, num_epochs + 1):
            train_loss, train_metrics = run_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )

            val_loss, val_metrics = run_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                optimizer=None,
            )

            print(
                f"[Fold {fold_idx}][Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} | {format_metrics(train_metrics)} || "
                f"val_loss={val_loss:.4f} | {format_metrics(val_metrics)}"
            )

            val_auc = val_metrics.get("auc", float("nan"))
            if not np.isnan(val_auc) and val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                best_metrics = copy.deepcopy(val_metrics)
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)

        ckpt_path = os.path.join(save_dir, f"fold_{fold_idx}_best.pt")
        save_checkpoint(
            model=model,
            path=ckpt_path,
            epoch=best_epoch,
            metrics=best_metrics if best_metrics is not None else {},
        )

        print(f"\nBest fold {fold_idx} epoch: {best_epoch}")
        if best_metrics is not None:
            print(f"Best fold {fold_idx} metrics: {format_metrics(best_metrics)}")
            fold_results.append(best_metrics)

    print("\n" + "=" * 100)
    print("Cross-validation summary")
    print("=" * 100)

    if len(fold_results) > 0:
        metric_names = eval_cfg.get("metrics", ["auc", "acc", "precision", "recall", "f1"])
        for metric_name in metric_names:
            values = np.array([fr[metric_name] for fr in fold_results], dtype=float)
            print(f"{metric_name}: {np.nanmean(values):.4f} ± {np.nanstd(values):.4f}")


if __name__ == "__main__":
    main()