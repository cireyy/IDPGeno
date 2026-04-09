from typing import Tuple
import torch
import torch.nn as nn


class SNPTokenEmbedding(nn.Module):
    def __init__(
        self,
        genotype_vocab_size: int = 3,
        token_embed_dim: int = 64,
        func_dim: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.genotype_vocab_size = genotype_vocab_size
        self.token_embed_dim = token_embed_dim
        self.func_dim = func_dim

        # For genotype codes 0 / 1 / 2
        self.geno_embedding = nn.Embedding(
            num_embeddings=genotype_vocab_size,
            embedding_dim=token_embed_dim,
        )

        # Project scalar relative position to token dim
        self.pos_projection = nn.Sequential(
            nn.Linear(1, token_embed_dim),
            nn.ReLU(),
            nn.Linear(token_embed_dim, token_embed_dim),
        )

        # Project functional features (e.g. maf, annotation weights)
        self.func_projection = nn.Sequential(
            nn.Linear(func_dim, token_embed_dim),
            nn.ReLU(),
            nn.Linear(token_embed_dim, token_embed_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(token_embed_dim)

    def forward(
        self,
        geno_ids: torch.Tensor,
        pos_values: torch.Tensor,
        func_values: torch.Tensor,
    ) -> torch.Tensor:

        if geno_ids.dim() != 3:
            raise ValueError(
                f"Expected geno_ids shape [B, G, M], but got {tuple(geno_ids.shape)}"
            )
        if pos_values.dim() != 3:
            raise ValueError(
                f"Expected pos_values shape [B, G, M], but got {tuple(pos_values.shape)}"
            )
        if func_values.dim() != 4:
            raise ValueError(
                f"Expected func_values shape [B, G, M, F], but got {tuple(func_values.shape)}"
            )

        # [B, G, M] -> [B, G, M, D]
        geno_emb = self.geno_embedding(geno_ids)

        # [B, G, M] -> [B, G, M, 1] -> [B, G, M, D]
        pos_emb = self.pos_projection(pos_values.unsqueeze(-1))

        # [B, G, M, F] -> [B, G, M, D]
        func_emb = self.func_projection(func_values)

        token_emb = geno_emb + pos_emb + func_emb
        token_emb = self.layer_norm(token_emb)
        token_emb = self.dropout(token_emb)

        return token_emb