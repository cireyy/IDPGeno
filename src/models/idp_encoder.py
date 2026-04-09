import torch
import torch.nn as nn


class IDPEncoder(nn.Module):
    def __init__(
        self,
        num_idps: int = 14,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_idps = num_idps
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(num_idps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, idp: torch.Tensor) -> torch.Tensor:
        if idp.dim() != 2:
            raise ValueError(
                f"Expected idp shape [B, num_idps], but got {tuple(idp.shape)}"
            )

        return self.mlp(idp)