from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class IDPGenoProcessedDataset(Dataset):
    def __init__(
        self,
        samples_csv: str,
        snp_tokens_csv: str,
        participant_col: str = "participant_id",
        label_col: str = "label",
        gene_col: str = "gene_id",
        token_index_col: str = "token_index_within_gene",
        geno_col: str = "geno_code",
        pos_col: str = "relative_pos",
        func_cols: Optional[List[str]] = None,
        mask_col: str = "is_valid",
        normalize_idp: bool = True,
        idp_mean: Optional[np.ndarray] = None,
        idp_std: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        self.samples_csv = samples_csv
        self.snp_tokens_csv = snp_tokens_csv

        self.participant_col = participant_col
        self.label_col = label_col
        self.gene_col = gene_col
        self.token_index_col = token_index_col
        self.geno_col = geno_col
        self.pos_col = pos_col
        self.mask_col = mask_col

        if func_cols is None:
            func_cols = ["maf", "within_gene_weight_a"]
        self.func_cols = func_cols

        # Load tables
        self.samples_df = pd.read_csv(samples_csv)
        self.tokens_df = pd.read_csv(snp_tokens_csv)

        # Basic checks
        self._check_required_columns()

        # Detect IDP columns from sample table
        self.idp_cols = self._detect_idp_columns()

        # Build participant order from sample table
        self.participant_ids = self.samples_df[self.participant_col].astype(str).tolist()
        self.num_samples = len(self.participant_ids)

        # Labels
        self.labels = self.samples_df[self.label_col].to_numpy(dtype=np.float32)

        # IDP matrix
        self.idp_array = self.samples_df[self.idp_cols].to_numpy(dtype=np.float32)

        # Normalize IDPs
        self.normalize_idp = normalize_idp
        if self.normalize_idp:
            if idp_mean is None or idp_std is None:
                self.idp_mean = self.idp_array.mean(axis=0)
                self.idp_std = self.idp_array.std(axis=0)
            else:
                self.idp_mean = idp_mean.astype(np.float32)
                self.idp_std = idp_std.astype(np.float32)

            self.idp_std = np.where(self.idp_std < 1e-8, 1.0, self.idp_std)
            self.idp_array = (self.idp_array - self.idp_mean) / self.idp_std
        else:
            self.idp_mean = None
            self.idp_std = None

        # Build gene order
        self.gene_ids = sorted(self.tokens_df[self.gene_col].astype(str).unique().tolist())
        self.num_genes = len(self.gene_ids)
        self.gene_to_idx = {g: i for i, g in enumerate(self.gene_ids)}

        # Build max token count from token_index
        self.max_snps_per_gene = int(self.tokens_df[self.token_index_col].max())

        # Function feature dimension
        self.func_dim = len(self.func_cols)

        # Build tensors
        (
            self.geno_tensor,
            self.pos_tensor,
            self.func_tensor,
            self.snp_mask,
        ) = self._build_token_tensors()

        print("=" * 80)
        print("IDPGenoProcessedDataset Summary")
        print("=" * 80)
        print(f"Samples CSV: {self.samples_csv}")
        print(f"Tokens CSV: {self.snp_tokens_csv}")
        print(f"Number of samples: {self.num_samples}")
        print(f"Number of IDPs: {len(self.idp_cols)}")
        print(f"Number of genes: {self.num_genes}")
        print(f"Max SNPs per gene: {self.max_snps_per_gene}")
        print(f"Function feature dim: {self.func_dim}")
        print(f"First 5 genes: {self.gene_ids[:5]}")
        print("=" * 80)

    def _check_required_columns(self) -> None:
        sample_required = {self.participant_col, self.label_col}
        token_required = {
            self.participant_col,
            self.gene_col,
            self.token_index_col,
            self.geno_col,
            self.pos_col,
            self.mask_col,
        }.union(set(self.func_cols))

        missing_sample = sample_required - set(self.samples_df.columns)
        missing_token = token_required - set(self.tokens_df.columns)

        if missing_sample:
            raise ValueError(f"Missing required columns in sample table: {missing_sample}")
        if missing_token:
            raise ValueError(f"Missing required columns in token table: {missing_token}")

    def _detect_idp_columns(self) -> List[str]:
        excluded = {self.participant_col, self.label_col}
        idp_cols = [c for c in self.samples_df.columns if c not in excluded]
        if len(idp_cols) == 0:
            raise ValueError("No IDP columns detected in sample table.")
        return idp_cols

    def _build_token_tensors(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build:
            geno_tensor: [N, G, M]
            pos_tensor:  [N, G, M]
            func_tensor: [N, G, M, F]
            snp_mask:    [N, G, M]
        """
        N = self.num_samples
        G = self.num_genes
        M = self.max_snps_per_gene
        F = self.func_dim

        geno_tensor = np.zeros((N, G, M), dtype=np.int64)
        pos_tensor = np.zeros((N, G, M), dtype=np.float32)
        func_tensor = np.zeros((N, G, M, F), dtype=np.float32)
        snp_mask = np.zeros((N, G, M), dtype=np.float32)

        # Make participant -> index lookup
        participant_to_idx = {
            str(pid): idx for idx, pid in enumerate(self.participant_ids)
        }

        # Iterate through token rows
        for row in self.tokens_df.itertuples(index=False):
            pid = str(getattr(row, self.participant_col))
            gene_id = str(getattr(row, self.gene_col))
            token_idx_1based = int(getattr(row, self.token_index_col))

            if pid not in participant_to_idx:
                continue
            if gene_id not in self.gene_to_idx:
                continue

            n_idx = participant_to_idx[pid]
            g_idx = self.gene_to_idx[gene_id]
            t_idx = token_idx_1based - 1  # convert to 0-based

            if t_idx < 0 or t_idx >= M:
                continue

            geno_tensor[n_idx, g_idx, t_idx] = int(getattr(row, self.geno_col))
            pos_tensor[n_idx, g_idx, t_idx] = float(getattr(row, self.pos_col))
            snp_mask[n_idx, g_idx, t_idx] = float(getattr(row, self.mask_col))

            for f_idx, func_col in enumerate(self.func_cols):
                func_tensor[n_idx, g_idx, t_idx, f_idx] = float(getattr(row, func_col))

        return geno_tensor, pos_tensor, func_tensor, snp_mask

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "idp": torch.tensor(self.idp_array[idx], dtype=torch.float32),
            "geno_ids": torch.tensor(self.geno_tensor[idx], dtype=torch.long),
            "pos_values": torch.tensor(self.pos_tensor[idx], dtype=torch.float32),
            "func_values": torch.tensor(self.func_tensor[idx], dtype=torch.float32),
            "snp_mask": torch.tensor(self.snp_mask[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

    def get_idp_stats(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.idp_mean is None or self.idp_std is None:
            return None
        return self.idp_mean, self.idp_std

    def get_feature_info(self) -> Dict[str, object]:
        return {
            "num_idps": len(self.idp_cols),
            "idp_cols": self.idp_cols,
            "num_genes": self.num_genes,
            "gene_ids": self.gene_ids,
            "max_snps_per_gene": self.max_snps_per_gene,
            "func_dim": self.func_dim,
            "func_cols": self.func_cols,
        }