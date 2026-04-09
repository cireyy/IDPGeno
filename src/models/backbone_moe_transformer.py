from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, max_len: int, embed_dim: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, G, D]
        Returns:
            x + pos: [B, G, D]
        """
        seq_len = x.size(1)
        return x + self.pos_embed[:, :seq_len, :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, G, D = x.shape

        qkv = self.qkv(x)  # [B, G, 3D]
        qkv = qkv.view(B, G, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, G, Hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, G, G]
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        context = torch.matmul(attn_probs, v)  # [B, H, G, Hd]
        context = context.transpose(1, 2).contiguous().view(B, G, D)

        out = self.out_proj(context)
        return out


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_hidden_dim: int,
        num_experts: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.ff_hidden_dim = ff_hidden_dim
        self.num_experts = num_experts

        self.gate = nn.Linear(embed_dim, num_experts)

        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, ff_hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ff_hidden_dim, embed_dim),
                )
                for _ in range(num_experts)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_logits = self.gate(x)                     # [B, G, E]
        gate_probs = torch.softmax(gate_logits, dim=-1)

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))           # each [B, G, D]

        expert_stack = torch.stack(expert_outputs, dim=-2)  # [B, G, E, D]
        out = torch.sum(gate_probs.unsqueeze(-1) * expert_stack, dim=-2)  # [B, G, D]
        out = self.dropout(out)

        return out, gate_probs


class MoETransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        ff_hidden_dim: int = 128,
        num_experts: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.moe_ffn = MoEFeedForward(
            embed_dim=embed_dim,
            ff_hidden_dim=ff_hidden_dim,
            num_experts=num_experts,
            dropout=dropout,
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_out)

        moe_out, gate_probs = self.moe_ffn(self.norm2(x))
        x = x + self.dropout2(moe_out)

        return x, gate_probs


class MoETransformerBackbone(nn.Module):
    def __init__(
        self,
        num_genes: int,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        ff_hidden_dim: int = 128,
        num_experts: int = 4,
        dropout: float = 0.1,
        pooling: str = "mean",
    ) -> None:
        super().__init__()

        self.num_genes = num_genes
        self.embed_dim = embed_dim
        self.pooling = pooling

        self.pos_encoder = PositionalEncoding(
            max_len=num_genes,
            embed_dim=embed_dim,
        )

        self.blocks = nn.ModuleList(
            [
                MoETransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_hidden_dim=ff_hidden_dim,
                    num_experts=num_experts,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, G, D], but got {tuple(x.shape)}")

        if x.size(1) > self.num_genes:
            raise ValueError(
                f"Input gene length {x.size(1)} exceeds configured num_genes {self.num_genes}"
            )

        x = self.pos_encoder(x)

        last_gate_probs = None
        for block in self.blocks:
            x, last_gate_probs = block(x)

        sequence_out = self.final_norm(x)

        if self.pooling == "mean":
            pooled_out = sequence_out.mean(dim=1)
        elif self.pooling == "cls":
            pooled_out = sequence_out[:, 0, :]
        else:
            raise ValueError(f"Unsupported pooling type: {self.pooling}")

        return sequence_out, pooled_out, last_gate_probs