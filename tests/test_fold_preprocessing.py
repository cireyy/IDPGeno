from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from src.data.dataset import IDPGenoProcessedDataset, build_fold_datasets, validate_idp_stats
from src.training import train


@pytest.fixture
def data_config(tmp_path):
    # Last two rows are deliberately unlike the training distribution.
    pd.DataFrame({"participant_id": range(10), "label": [0, 1] * 5,
                  "idp_a": [0, 2, 4, 6, 8, 10, 12, 14, 1000, 2000],
                  "idp_b": [5] * 8 + [9, 10]}).to_csv(tmp_path / "samples.csv", index=False)
    pd.DataFrame({"participant_id": range(10), "gene_id": ["g1"] * 10,
                  "token_index_within_gene": [1] * 10, "geno_code": [0, 1] * 5,
                  "relative_pos": [0.] * 10, "maf": [.2] * 10,
                  "within_gene_weight_a": [1.] * 10, "is_valid": [1] * 10
                  }).to_csv(tmp_path / "tokens.csv", index=False)
    return {"data": {"samples_csv": str(tmp_path / "samples.csv"),
                     "snp_tokens_csv": str(tmp_path / "tokens.csv"), "normalize_idp": True}}


def test_train_only_statistics_and_batched_validation(data_config):
    raw = train.build_dataset(data_config)
    before = raw.idp_array.copy()
    tr, va = build_fold_datasets(raw, np.arange(8), np.arange(8, 10))
    np.testing.assert_allclose(tr.idp_mean, [7, 5])
    np.testing.assert_allclose(tr.idp_std, [np.sqrt(21), 1])
    np.testing.assert_array_equal(tr.idp_mean, va.idp_mean)
    batch = next(iter(DataLoader(va, batch_size=2)))
    np.testing.assert_allclose(batch["idp"], (before[8:] - tr.idp_mean) / tr.idp_std)
    np.testing.assert_array_equal(batch["label"], raw.labels[8:])
    np.testing.assert_array_equal(batch["geno_ids"], raw.geno_tensor[8:])
    np.testing.assert_array_equal(raw.idp_array, before)
    assert not raw.normalize_idp


def test_validation_perturbation_cannot_change_training(data_config):
    raw = train.build_dataset(data_config)
    tr, va = build_fold_datasets(raw, np.arange(8), np.arange(8, 10))
    expected = torch.stack([tr[i]["idp"] for i in range(len(tr))])
    raw.idp_array[8:] += 100000
    other, changed_val = build_fold_datasets(raw, np.arange(8), np.arange(8, 10))
    np.testing.assert_array_equal(other.idp_mean, tr.idp_mean)
    np.testing.assert_array_equal(other.idp_std, tr.idp_std)
    torch.testing.assert_close(torch.stack([other[i]["idp"] for i in range(len(other))]), expected)
    # A later fold must not overwrite an earlier fold's fitted state.
    saved = tr.idp_mean.copy()
    build_fold_datasets(raw, np.arange(2, 10), np.arange(2))
    np.testing.assert_array_equal(tr.idp_mean, saved)


def test_disabled_normalization(data_config):
    raw = train.build_dataset(data_config)
    tr, va = build_fold_datasets(raw, np.arange(8), np.arange(8, 10), False)
    assert tr.idp_mean is None and va.idp_std is None
    np.testing.assert_array_equal(next(iter(DataLoader(va, batch_size=2)))["idp"], raw.idp_array[8:])
    assert tr.preprocessing_state()["normalize_idp"] is False


@pytest.mark.parametrize("mean,std", [(None, None), ([0, 0], None), (None, [1, 1]),
                                        ([0], [1]), ([np.nan, 0], [1, 1]),
                                        ([0, 0], [-1, 1]), ([0, 0], [np.inf, 1])])
def test_invalid_statistics_fail(mean, std):
    with pytest.raises(ValueError):
        validate_idp_stats(mean, std, 2)


def test_dataset_never_implicitly_fits_and_external_stats_work(data_config):
    kwargs = dict(data_config["data"])
    with pytest.raises(ValueError, match="training rows"):
        IDPGenoProcessedDataset(**kwargs)
    raw = train.build_dataset(data_config)
    external = IDPGenoProcessedDataset(**kwargs, idp_mean=np.array([7., 5.]), idp_std=np.array([2., 0.]))
    np.testing.assert_allclose(external.idp_array, (raw.idp_array - [7, 5]) / [2, 1])
    with pytest.raises(ValueError, match="raw"):
        build_fold_datasets(external, np.arange(8), np.arange(8, 10))


@pytest.mark.parametrize("tr,va", [([0, 1], [1, 2]), ([], [1]), ([0, 0], [1]),
                                   ([-1], [1]), ([10], [1]), ([0.5], [1])])
def test_invalid_folds_fail(data_config, tr, va):
    with pytest.raises(ValueError):
        build_fold_datasets(train.build_dataset(data_config), tr, va)


@pytest.mark.parametrize("bad", ["duplicate", "nan", "inf"])
def test_invalid_sample_input_fails(data_config, bad):
    path = data_config["data"]["samples_csv"]
    frame = pd.read_csv(path)
    if bad == "duplicate":
        frame.loc[1, "participant_id"] = frame.loc[0, "participant_id"]
    else:
        frame["idp_a"] = frame["idp_a"].astype(float)
        frame.loc[0, "idp_a"] = np.nan if bad == "nan" else np.inf
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        train.build_dataset(data_config)


def test_main_five_folds_checkpoint_round_trip(data_config, tmp_path, monkeypatch):
    config = train.load_config("configs/configs.yaml")
    config["data"] = data_config["data"]
    config["training"].update(num_epochs=1, batch_size=4, device="cpu", num_workers=0)
    config["training"]["checkpoint"]["dir"] = str(tmp_path / "checkpoints")
    # Run the real model, optimizer, loaders, training entry point and checkpoint writer.
    torch.set_num_threads(1)
    monkeypatch.setattr(train, "parse_args", lambda: SimpleNamespace(config="fixture"))
    monkeypatch.setattr(train, "load_config", lambda _: config)
    train.main()
    raw = train.build_dataset(config)
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(
            np.zeros(len(raw)), raw.labels), 1):
        checkpoint = torch.load(tmp_path / "checkpoints" / f"fold_{fold}_best.pt", weights_only=True)
        state = checkpoint["preprocessing"]
        np.testing.assert_allclose(state["idp_mean"], raw.idp_array[tr].mean(0, dtype=np.float64), rtol=1e-6)
        expected_std = raw.idp_array[tr].std(0, dtype=np.float64)
        np.testing.assert_allclose(state["idp_std"], np.where(expected_std < 1e-8, 1, expected_std), rtol=1e-6)
        assert state["idp_cols"] == raw.idp_cols
        restored = IDPGenoProcessedDataset(**config["data"],
                                          idp_mean=state["idp_mean"].numpy(), idp_std=state["idp_std"].numpy())
        _, val_view = build_fold_datasets(raw, tr, va)
        torch.testing.assert_close(restored[int(va[0])]["idp"], val_view[0]["idp"])
        model = train.build_model(checkpoint["feature_info"], config)
        model.load_state_dict(checkpoint["model_state_dict"])


def test_repository_simulated_data_five_fold_scaling():
    # Deliberately use the checked-in simulated CSVs, not the synthetic fixture above.
    config = train.load_config("configs/configs.yaml")
    raw = train.build_dataset(config)
    before = raw.idp_array.copy()
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(before, raw.labels):
        training, validation = build_fold_datasets(raw, tr, va)
        np.testing.assert_allclose(training.idp_mean, before[tr].mean(0, dtype=np.float64), rtol=1e-6)
        batch = next(iter(DataLoader(validation, batch_size=len(va))))
        np.testing.assert_allclose(batch["idp"], (before[va] - training.idp_mean) / training.idp_std)
    np.testing.assert_array_equal(raw.idp_array, before)
