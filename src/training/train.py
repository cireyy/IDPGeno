import argparse
import copy
import os
import random
from typing import Any, Dict, List
import numpy as np
import torch
import yaml
import json
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from src.data.dataset import IDPGenoProcessedDataset
from src.models.idpgeno_model import IDPGenoModel
from src.training.metrics import format_metrics, compute_binary_classification_metrics
from src.training.trainer import run_one_epoch, build_loss_fn, save_checkpoint, predict_probabilities

from dataclasses import dataclass
from typing import Iterator
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class FoldSplit:
    fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def make_splits(labels, num_folds=5, validation_fraction=0.2, seed=42) -> Iterator[FoldSplit]:
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be a one-dimensional binary array.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    counts = np.bincount(labels.astype(int), minlength=2)
    if num_folds < 2 or counts.min() < num_folds:
        raise ValueError("Each class needs at least num_folds participants.")
    outer = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for fold, (development, test) in enumerate(outer.split(np.zeros(len(labels)), labels), 1):
        try:
            train, validation = train_test_split(
                development, test_size=validation_fraction,
                stratify=labels[development], random_state=seed + fold,
            )
        except ValueError as exc:
            raise ValueError(f"Fold {fold}: insufficient cases/controls for the inner split.") from exc
        for name, indices in (("training", train), ("validation", validation), ("test", test)):
            if len(np.unique(labels[indices])) != 2:
                raise ValueError(f"Fold {fold} {name} subset must contain both classes.")
        yield FoldSplit(fold, train, validation, test)


def class_distribution(labels, indices, fold, subset, positive_weight=None):
    selected = np.asarray(labels)[indices].astype(int)
    cases = int((selected == 1).sum())
    controls = int((selected == 0).sum())
    if cases == 0 or controls == 0:
        raise ValueError("Both classes are required.")
    return {
        "fold": int(fold),
        "subset": subset,
        "n_participants": cases + controls,
        "cases": cases,
        "controls": controls,
        "case_to_control_counts": f"{cases}:{controls}",
        "controls_per_case": controls / cases,
        "case_prevalence": cases / (cases + controls),
        "positive_loss_weight": positive_weight if subset == "train" else None,
        "resampled": False,
    }


def probability_logits(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_sigmoid_calibration(labels, probabilities):
    if len(np.unique(labels)) != 2:
        raise ValueError("Calibration requires both classes in the validation subset.")
    
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(probability_logits(probabilities), labels)
    return {
        "method": "sigmoid",
        "coefficient": float(model.coef_[0, 0]),
        "intercept": float(model.intercept_[0]),
        "fit_subset": "inner_validation",
    }


def apply_calibration(probabilities, state):
    if state["method"] == "none":
        return np.asarray(probabilities, dtype=float)
    logits = probability_logits(probabilities).ravel()
    z = np.clip(state["coefficient"] * logits + state["intercept"], -700, 700)
    return 1 / (1 + np.exp(-z))



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
        normalize_idp=False,  
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



def write_json(path, value):
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_cfg = config["training"]
    eval_cfg = config.get("evaluation", {})
    seed = int(config["project"].get("seed", 42))
    set_seed(seed)
    if train_cfg.get("torch_num_threads"):
        torch.set_num_threads(int(train_cfg["torch_num_threads"]))
    device = get_device(train_cfg.get("device", "auto"))
    print(f"Using device: {device}")

    dataset = build_dataset(config)
    info = dataset.get_feature_info()
    labels = dataset.labels.astype(int)
    split_cfg = train_cfg.get("split", {})
    validation_fraction = float(split_cfg.get("validation_fraction", 0.2))
    splits = list(make_splits(labels, int(train_cfg["num_folds"]), validation_fraction, seed))
    threshold = float(eval_cfg.get("threshold", 0.5))
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must be between 0 and 1.")
    calibration_method = eval_cfg.get("calibration", "sigmoid")
    if calibration_method not in ("none", "sigmoid"):
        raise ValueError("evaluation.calibration must be 'none' or 'sigmoid'.")
    use_weights = bool(train_cfg.get("loss", {}).get("use_pos_weight", True))
    output = Path(config["project"]["output_dir"])
    checkpoint_dir = Path(train_cfg["checkpoint"]["dir"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    write_json(output / "run_metadata.json", {
        "evaluation_design": "outer_stratified_cv_with_inner_validation",
        "validation_fraction_of_outer_development": validation_fraction,
        "resampling": "none",
        "positive_weight_source": "training_subset_only" if use_weights else "unweighted",
        "normalisation_source": "training_subset_only",
        "checkpoint_selection_source": "inner_validation_auc",
        "calibration_source": "inner_validation" if calibration_method != "none" else "none",
        "classification_threshold": threshold,
        "num_participants": len(dataset),
        "num_genes": info["num_genes"],
        "num_idps": info["num_idps"],
        "config": config,
    })

    import pandas as pd
    composition = []
    fold_results = []
    selected_models = []
    oof_raw = np.full(len(labels), np.nan)
    oof_calibrated = np.full(len(labels), np.nan)

    for split in splits:
        fold_seed = seed + split.fold
        set_seed(fold_seed)
        if config["data"].get("normalize_idp", True):
            dataset.fit_idp_normalisation(split.train)

        loaders = {}
        for name, indices in (("train", split.train), ("validation", split.validation), ("test", split.test)):
            generator = torch.Generator().manual_seed(fold_seed)
            loaders[name] = DataLoader(
                Subset(dataset, indices.tolist()),
                batch_size=int(train_cfg["batch_size"]),
                shuffle=(name == "train"),
                num_workers=int(train_cfg.get("num_workers", 0)),
                generator=generator,
            )

        train_labels = labels[split.train]
        criterion = build_loss_fn(train_labels, device, use_pos_weight=use_weights)
        weight = float(criterion.pos_weight.item()) if criterion.pos_weight is not None else 1.0
        for name, indices in (("train", split.train), ("validation", split.validation), ("test", split.test)):
            composition.append(class_distribution(labels, indices, split.fold, name, weight))
        pd.DataFrame(composition).to_csv(output / "fold_class_distribution.csv", index=False)
        if train_cfg.get("save_split_indices", False):
            np.savez_compressed(
                output / f"fold_{split.fold}_indices.npz",
                train=split.train, validation=split.validation, test=split.test,
            )

        model = build_model(info, config).to(device)
        optimiser = build_optimizer(model, config)
        scheduler_cfg = train_cfg.get("scheduler", {})
        scheduler = None
        if scheduler_cfg.get("use", False):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, mode="max",
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 3)),
                min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
            )
        early_cfg = train_cfg.get("early_stopping", {})
        best_auc = -float("inf")
        best_epoch = None
        best_state = None
        best_validation_metrics = None
        stale_epochs = 0
        history = []

        for epoch in range(1, int(train_cfg["num_epochs"]) + 1):
            train_loss, train_metrics = run_one_epoch(
                model, loaders["train"], criterion, device,
                optimizer=optimiser, threshold=threshold,
            )
            validation_loss, validation_metrics = run_one_epoch(
                model, loaders["validation"], criterion, device,
                threshold=threshold,
            )
            validation_auc = validation_metrics["auc"]
            row = {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss}
            row.update({f"train_{key}": val for key, val in train_metrics.items()})
            row.update({f"validation_{key}": val for key, val in validation_metrics.items()})
            history.append(row)
            print(
                f"[Fold {split.fold}][Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} validation_loss={validation_loss:.4f} "
                f"validation {format_metrics(validation_metrics)}"
            )
            if not np.isfinite(validation_auc):
                raise ValueError("Validation AUROC is non-finite; check labels and predictions.")
            if validation_auc > best_auc:
                best_auc = validation_auc
                best_epoch = epoch
                best_validation_metrics = copy.deepcopy(validation_metrics)
                best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
            if scheduler is not None:
                scheduler.step(validation_auc)
            if early_cfg.get("use", False) and stale_epochs >= int(early_cfg.get("patience", 10)):
                break

        if best_state is None:
            raise ValueError("No valid checkpoint was selected; num_epochs must be positive.")
        model.load_state_dict(best_state)
        pd.DataFrame(history).to_csv(output / f"fold_{split.fold}_training_history.csv", index=False)

        
        calibration_state = {"method": "none", "fit_subset": None}
        if calibration_method == "sigmoid":
            validation_labels, validation_probabilities = predict_probabilities(
                model, loaders["validation"], device,
            )
            calibration_state = fit_sigmoid_calibration(validation_labels, validation_probabilities)

        statistics = dataset.get_idp_stats()
        normalisation = None if statistics is None else {
            "mean": statistics[0].tolist(), "std": statistics[1].tolist(),
            "fit_subset": "train",
        }
        checkpoint = checkpoint_dir / f"fold_{split.fold}_best.pt"
        save_checkpoint(
            model, str(checkpoint), best_epoch, best_validation_metrics,
            extra={
                "metric_subset": "inner_validation",
                "normalisation": normalisation,
                "calibration": calibration_state,
                "gene_ids": info["gene_ids"],
                "idp_columns": info["idp_cols"],
                "feature_info": info,
                "config": config,
                "positive_loss_weight": weight,
                "classification_threshold": threshold,
            },
        )

        
        test_labels, raw_probabilities = predict_probabilities(model, loaders["test"], device)
        calibrated_probabilities = apply_calibration(raw_probabilities, calibration_state)
        raw_metrics = compute_binary_classification_metrics(test_labels, raw_probabilities, threshold)
        calibrated_metrics = compute_binary_classification_metrics(
            test_labels, calibrated_probabilities, threshold,
        )
        test_row = {"fold": split.fold}
        test_row.update({f"raw_{key}": val for key, val in raw_metrics.items()})
        test_row.update({f"calibrated_{key}": val for key, val in calibrated_metrics.items()})
        fold_results.append(test_row)
        oof_raw[split.test] = raw_probabilities
        oof_calibrated[split.test] = calibrated_probabilities
        selected_models.append({
            "fold": split.fold, "selected_epoch": best_epoch,
            "validation_auc": best_auc, "positive_loss_weight": weight,
            "checkpoint": str(checkpoint), "calibration": calibration_state,
        })
        pd.DataFrame(fold_results).to_csv(output / "fold_test_metrics.csv", index=False)
        write_json(output / "selected_models.json", selected_models)
        print(f"[Fold {split.fold}] final test (raw): {format_metrics(raw_metrics)}")
        if calibration_method != "none":
            print(f"[Fold {split.fold}] final test (calibrated): {format_metrics(calibrated_metrics)}")

    if not np.isfinite(oof_raw).all() or not np.isfinite(oof_calibrated).all():
        raise RuntimeError("Every participant must receive exactly one outer-test prediction.")
    result_frame = pd.DataFrame(fold_results)
    summaries = {}
    for column in result_frame.columns:
        if column != "fold":
            values = result_frame[column].to_numpy(dtype=float)
            summaries[column] = {
                "mean": float(np.mean(values)),
                "sd_across_folds": float(np.std(values, ddof=1)),
            }
    write_json(output / "cross_validation_summary.json", {
        "fold_metrics": summaries,
        "pooled_out_of_fold_raw": compute_binary_classification_metrics(labels, oof_raw, threshold),
        "pooled_out_of_fold_calibrated": compute_binary_classification_metrics(labels, oof_calibrated, threshold),
        "calibration_method": calibration_method,
        "classification_threshold": threshold,
    })
    if config.get("logging", {}).get("save_fold_predictions", False):
        pd.DataFrame({
            "participant_id": dataset.participant_ids, "label": labels,
            "raw_probability": oof_raw, "calibrated_probability": oof_calibrated,
        }).to_csv(output / "out_of_fold_predictions.csv", index=False)
    print(f"Saved class distributions, training histories and final test metrics to {output}")


if __name__ == "__main__":
    main()
