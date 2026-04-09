from typing import Tuple
import torch
import torch.nn as nn


class GeneWiseFiLM(nn.Module):
    def __init__(
        self,
        idp_hidden_dim: int,
        num_genes: int,
        gene_embed_dim: int,
        use_residual: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.idp_hidden_dim = idp_hidden_dim
        self.num_genes = num_genes
        self.gene_embed_dim = gene_embed_dim
        self.use_residual = use_residual

        self.gamma_generator = nn.Sequential(
            nn.Linear(idp_hidden_dim, idp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(idp_hidden_dim, num_genes * gene_embed_dim),
        )

        self.beta_generator = nn.Sequential(
            nn.Linear(idp_hidden_dim, idp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(idp_hidden_dim, num_genes * gene_embed_dim),
        )

    def forward(
        self,
        gene_emb: torch.Tensor,
        idp_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if gene_emb.dim() != 3:
            raise ValueError(
                f"Expected gene_emb shape [B, G, D], but got {tuple(gene_emb.shape)}"
            )
        if idp_feat.dim() != 2:
            raise ValueError(
                f"Expected idp_feat shape [B, H], but got {tuple(idp_feat.shape)}"
            )

        batch_size, num_genes, gene_dim = gene_emb.shape

        if num_genes != self.num_genes:
            raise ValueError(
                f"num_genes mismatch: got {num_genes}, expected {self.num_genes}"
            )
        if gene_dim != self.gene_embed_dim:
            raise ValueError(
                f"gene_embed_dim mismatch: got {gene_dim}, expected {self.gene_embed_dim}"
            )

        gamma = self.gamma_generator(idp_feat).view(
            batch_size, self.num_genes, self.gene_embed_dim
        )
        beta = self.beta_generator(idp_feat).view(
            batch_size, self.num_genes, self.gene_embed_dim
        )

        if self.use_residual:
            modulated_gene_emb = (1.0 + gamma) * gene_emb + beta
        else:
            modulated_gene_emb = gamma * gene_emb + beta

        return modulated_gene_emb, gamma, beta