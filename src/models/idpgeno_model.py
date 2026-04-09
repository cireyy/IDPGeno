# models/idpgeno_model.py

from typing import Dict

import torch
import torch.nn as nn

from models.gene_encoder import GeneEncoder
from models.idp_encoder import IDPEncoder
from models.film import GeneWiseFiLM
from models.backbone_moe_transformer import MoETransformerBackbone


class IDPGenoModel(nn.Module):
    """
    Full IDPGeno model:
    1. SNP token embedding + masked attention pooling -> gene embeddings
    2. IDP encoding
    3. Gene-wise FiLM modulation
    4. MoE-Transformer over gene sequence
    5. Classification head

    Inputs:
        idp:         [B, num_idps]
        geno_ids:    [B, G, M]
        pos_values:  [B, G, M]
        func_values: [B, G, M, F]
        snp_mask:    [B, G, M]

    Outputs:
        dict containing:
            logits:              [B]
            probabilities:       [B]
            gene_emb:            [B, G, D]
            modulated_gene_emb:  [B, G, D]
            sequence_out:        [B, G, D]
            pooled_out:          [B, D]
            attn_weights:        [B, G, M]
            gamma:               [B, G, D]
            beta:                [B, G, D]
            gate_probs:          [B, G, E]
    """

    def __init__(
        self,
        num_idps: int,
        num_genes: int,
        max_snps_per_gene: int,
        func_dim: int,
        genotype_vocab_size: int = 3,
        token_embed_dim: int = 64,
        gene_embed_dim: int = 64,
        idp_hidden_dim: int = 64,
        backbone_num_layers: int = 2,
        backbone_num_heads: int = 4,
        backbone_ff_hidden_dim: int = 128,
        backbone_num_experts: int = 4,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.num_idps = num_idps
        self.num_genes = num_genes
        self.max_snps_per_gene = max_snps_per_gene
        self.func_dim = func_dim

        self.gene_encoder = GeneEncoder(
            genotype_vocab_size=genotype_vocab_size,
            token_embed_dim=token_embed_dim,
            gene_embed_dim=gene_embed_dim,
            func_dim=func_dim,
            dropout=dropout,
        )

        self.idp_encoder = IDPEncoder(
            num_idps=num_idps,
            hidden_dim=idp_hidden_dim,
            dropout=dropout,
        )

        self.film = GeneWiseFiLM(
            idp_hidden_dim=idp_hidden_dim,
            num_genes=num_genes,
            gene_embed_dim=gene_embed_dim,
            use_residual=True,
            dropout=dropout,
        )

        self.backbone = MoETransformerBackbone(
            num_genes=num_genes,
            embed_dim=gene_embed_dim,
            num_layers=backbone_num_layers,
            num_heads=backbone_num_heads,
            ff_hidden_dim=backbone_ff_hidden_dim,
            num_experts=backbone_num_experts,
            dropout=dropout,
            pooling="mean",
        )

        self.classifier = nn.Sequential(
            nn.Linear(gene_embed_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def forward(
        self,
        idp: torch.Tensor,
        geno_ids: torch.Tensor,
        pos_values: torch.Tensor,
        func_values: torch.Tensor,
        snp_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        gene_emb, attn_weights, token_emb = self.gene_encoder(
            geno_ids=geno_ids,
            pos_values=pos_values,
            func_values=func_values,
            snp_mask=snp_mask,
        )

        idp_feat = self.idp_encoder(idp)

        modulated_gene_emb, gamma, beta = self.film(
            gene_emb=gene_emb,
            idp_feat=idp_feat,
        )

        sequence_out, pooled_out, gate_probs = self.backbone(modulated_gene_emb)

        logits = self.classifier(pooled_out).squeeze(-1)
        probabilities = torch.sigmoid(logits)

        return {
            "logits": logits,
            "probabilities": probabilities,
            "token_emb": token_emb,
            "gene_emb": gene_emb,
            "modulated_gene_emb": modulated_gene_emb,
            "sequence_out": sequence_out,
            "pooled_out": pooled_out,
            "attn_weights": attn_weights,
            "gamma": gamma,
            "beta": beta,
            "gate_probs": gate_probs,
        }