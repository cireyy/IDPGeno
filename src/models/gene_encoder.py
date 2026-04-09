from typing import Tuple

import torch
import torch.nn as nn

from models.snp_embedding import SNPTokenEmbedding


class MaskedSNPAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        self.score = nn.Linear(embed_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        snp_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Expected x shape [B, G, M, D], but got {tuple(x.shape)}")
        if snp_mask.dim() != 3:
            raise ValueError(
                f"Expected snp_mask shape [B, G, M], but got {tuple(snp_mask.shape)}"
            )

        # [B, G, M, 1]
        attn_logits = self.score(x)

        # Invalid positions get very negative logits before softmax
        mask = snp_mask.unsqueeze(-1)  # [B, G, M, 1]
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)

        # [B, G, M, 1]
        attn_weights = torch.softmax(attn_logits, dim=2)
        attn_weights = attn_weights * mask

        # Re-normalize in case some positions are masked
        denom = attn_weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
        attn_weights = attn_weights / denom

        # [B, G, D]
        pooled = torch.sum(attn_weights * x, dim=2)

        # [B, G, M]
        attn_weights = attn_weights.squeeze(-1)

        return pooled, attn_weights


class GeneEncoder(nn.Module):
    def __init__(
        self,
        genotype_vocab_size: int = 3,
        token_embed_dim: int = 64,
        gene_embed_dim: int = 64,
        func_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.token_embed_dim = token_embed_dim
        self.gene_embed_dim = gene_embed_dim
        self.func_dim = func_dim

        self.snp_embedding = SNPTokenEmbedding(
            genotype_vocab_size=genotype_vocab_size,
            token_embed_dim=token_embed_dim,
            func_dim=func_dim,
            dropout=dropout,
        )

        self.snp_pool = MaskedSNPAttentionPooling(embed_dim=token_embed_dim)

        self.gene_projector = nn.Sequential(
            nn.Linear(token_embed_dim, gene_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        geno_ids: torch.Tensor,
        pos_values: torch.Tensor,
        func_values: torch.Tensor,
        snp_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token_emb = self.snp_embedding(
            geno_ids=geno_ids,
            pos_values=pos_values,
            func_values=func_values,
        )

        pooled, attn_weights = self.snp_pool(
            x=token_emb,
            snp_mask=snp_mask,
        )

        gene_emb = self.gene_projector(pooled)

        return gene_emb, attn_weights, token_emb