"""Tabular and graph-guided feature backbones."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..data import VarSpec
from ..graph import _build_random_dag_mask
from .embeddings import OrdinalAwareEmbedding, StandardEmbedding


class ResMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(dim)
        self.lin1 = nn.Linear(dim, dim * 2)
        self.bn2 = nn.BatchNorm1d(dim * 2)
        self.lin2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn1(x))
        h = self.act(self.bn2(self.lin1(h)))
        h = self.drop(self.lin2(h))
        return x + h


class DeepResMLPBackbone(nn.Module):
    """Residual tabular backbone with a causal feature branch."""

    def __init__(
        self,
        var_specs: List["VarSpec"],
        emb_dim: int = 128,
        n_blocks: int = 6,
        dropout: float = 0.1,
        causal_focus_idx: Optional[Sequence[int]] = None,
        variant: str = "full",
        n_engineered: int = 0,
    ):
        super().__init__()
        self.variant = str(variant)
        self.emb_dim = int(emb_dim)
        self.n_vars = len(var_specs)
        self.n_engineered = int(n_engineered)
        use_std_emb = self.variant == "std_emb"
        if use_std_emb:
            self.emb: nn.Module = StandardEmbedding(var_specs, emb_dim)
        else:
            self.emb = OrdinalAwareEmbedding(var_specs, emb_dim)
        in_dim = self.n_vars * emb_dim + self.n_engineered
        hidden_dim = emb_dim * 3
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        n_blk = int(n_blocks)
        self.blocks = nn.ModuleList([ResMLPBlock(hidden_dim, dropout) for _ in range(n_blk)])
        focus_idx = []
        if causal_focus_idx and variant not in ("no_causal_branch",):
            focus_idx = sorted(set((int(i) for i in causal_focus_idx if 0 <= int(i) < self.n_vars)))
        self.focus_idx_list = focus_idx
        self.has_causal_branch = len(focus_idx) >= 3 and variant not in ("no_causal_branch",)
        if self.has_causal_branch:
            self.register_buffer("focus_idx_t", torch.tensor(focus_idx, dtype=torch.long))
            causal_in_dim = len(focus_idx) * emb_dim
            causal_hidden = min(emb_dim * 2, 256)
            self.causal_branch = nn.Sequential(
                nn.Linear(causal_in_dim, causal_hidden),
                nn.BatchNorm1d(causal_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(causal_hidden, emb_dim),
                nn.GELU(),
            )
            total_out = hidden_dim + emb_dim
        else:
            self.register_buffer("focus_idx_t", None)
            self.causal_branch = None
            total_out = hidden_dim
        self.output_proj = nn.Sequential(
            nn.BatchNorm1d(total_out), nn.Linear(total_out, emb_dim), nn.GELU()
        )
        self.out_norm = nn.LayerNorm(emb_dim)

    def forward(self, X: torch.Tensor, X_eng: Optional[torch.Tensor] = None) -> torch.Tensor:
        E = self.emb(X)
        B = E.shape[0]
        flat = E.reshape(B, -1)
        if self.n_engineered > 0:
            if X_eng is not None:
                flat = torch.cat([flat, X_eng.to(dtype=flat.dtype, device=flat.device)], dim=-1)
            else:
                pad = torch.zeros(B, self.n_engineered, dtype=flat.dtype, device=flat.device)
                flat = torch.cat([flat, pad], dim=-1)
        h = self.input_proj(flat)
        for blk in self.blocks:
            h = blk(h)
        if (
            self.has_causal_branch
            and self.focus_idx_t is not None
            and (self.causal_branch is not None)
        ):
            Ef = E.index_select(1, self.focus_idx_t).reshape(B, -1)
            h_causal = self.causal_branch(Ef)
            h = torch.cat([h, h_causal], dim=-1)
        out = self.output_proj(h)
        return self.out_norm(out)


class CausalMaskedSelfAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, dropout: float):
        super().__init__()
        if emb_dim % n_heads != 0:
            raise ValueError(f"emb_dim({emb_dim}) must be divisible by n_heads({n_heads})")
        self.emb_dim = int(emb_dim)
        self.n_heads = int(n_heads)
        self.dk = self.emb_dim // self.n_heads
        self.qkv = nn.Linear(self.emb_dim, self.emb_dim * 3)
        self.out = nn.Linear(self.emb_dim, self.emb_dim)
        self.drop = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, M, D = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, M, self.n_heads, self.dk).transpose(1, 2)
        k = k.view(B, M, self.n_heads, self.dk).transpose(1, 2)
        v = v.view(B, M, self.n_heads, self.dk).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)
        if attn_bias is not None:
            bias = attn_bias.to(scores.device, dtype=scores.dtype)
            scores = scores + bias.view(1, 1, M, M)
        if attn_mask is not None:
            mask = attn_mask.to(scores.device)
            if mask.dtype != torch.bool:
                mask = mask > 0
            scores = scores.masked_fill(~mask.view(1, 1, M, M), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)
        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(B, M, D)
        y = self.out(y)
        y = self.drop(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = CausalMaskedSelfAttention(emb_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask, attn_bias=attn_bias)
        x = x + self.ff(self.ln2(x))
        return x


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))


def _subset_graph_matrix(
    causal_attn_mask: Optional[np.ndarray],
    causal_attn_bias: Optional[np.ndarray],
    focus_idx: Sequence[int],
    variant: str = "full",
) -> Optional[np.ndarray]:
    if focus_idx is None or len(focus_idx) == 0:
        return None
    idx = np.array(list(focus_idx), dtype=int)
    K = len(idx)
    if variant == "no_mask":
        base = np.ones((K, K), dtype=np.float32)
        return base / np.clip(base.sum(axis=1, keepdims=True), 1.0, None)
    if variant == "random_mask":
        if causal_attn_mask is not None:
            sub = causal_attn_mask[np.ix_(idx, idx)].astype(np.bool_)
            rand_mask = _build_random_dag_mask(sub, seed=999)
            W = rand_mask.astype(np.float32)
        else:
            rng = np.random.RandomState(999)
            W = (rng.rand(K, K) < min(0.2, 3.0 / max(K, 1))).astype(np.float32)
    elif causal_attn_mask is not None:
        W = causal_attn_mask[np.ix_(idx, idx)].astype(np.float32)
    elif causal_attn_bias is not None:
        B = causal_attn_bias[np.ix_(idx, idx)].astype(np.float32)
        W = (B > -0.35).astype(np.float32)
    else:
        W = np.zeros((K, K), dtype=np.float32)
    if causal_attn_bias is not None and variant in ("full", "std_emb"):
        B = causal_attn_bias[np.ix_(idx, idx)].astype(np.float32)
        W = W + 0.35 * (1.0 / (1.0 + np.exp(-np.clip(B, -4.0, 4.0))))
    np.fill_diagonal(W, 1.0)
    row_sum = W.sum(axis=1, keepdims=True)
    row_sum[row_sum <= 0] = 1.0
    return W / row_sum


class FTTransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, ffn_factor: int = 4, dropout: float = 0.15):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(emb_dim)
        d_ff = emb_dim * ffn_factor
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, emb_dim),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        x_norm = self.ln1(x)
        attn_mask = None
        if attn_bias is not None:
            H = self.attn.num_heads
            ab = attn_bias.to(x.device, dtype=x.dtype)
            if ab.dim() == 2:
                ab = ab.unsqueeze(0)
            if ab.dim() == 3:
                ab = ab.unsqueeze(1)
            attn_mask = ab.expand(B, H, T, T).reshape(B * H, T, T)
        out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + self.drop(out)
        x = x + self.ffn(self.ln2(x))
        return x


class FTTransformerBackbone(nn.Module):
    def __init__(
        self,
        var_specs: List["VarSpec"],
        emb_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_factor: int = 2,
        dropout: float = 0.15,
        causal_attn_bias: Optional[np.ndarray] = None,
        causal_focus_idx: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.n_vars = len(var_specs)
        self.has_causal_branch = False
        self.tokenizer = OrdinalAwareEmbedding(var_specs, emb_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList(
            [FTTransformerBlock(emb_dim, n_heads, ffn_factor, dropout) for _ in range(n_layers)]
        )
        self.pool_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim), nn.GELU(), nn.LayerNorm(emb_dim)
        )
        self.final_norm = nn.LayerNorm(emb_dim)
        if causal_attn_bias is not None:
            M = self.n_vars
            full_bias = np.zeros((M + 1, M + 1), dtype=np.float32)
            full_bias[1:, 1:] = causal_attn_bias.astype(np.float32)
            self.register_buffer("attn_bias", torch.from_numpy(full_bias))
        else:
            self.register_buffer("attn_bias", None)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        E = self.tokenizer(X)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, E], dim=1)
        bias = self.attn_bias if self.attn_bias is not None else None
        for blk in self.blocks:
            x = blk(x, attn_bias=bias)
        x = self.final_norm(x)
        h_cls = x[:, 0, :]
        h_mean = x[:, 1:, :].mean(dim=1)
        return self.pool_proj(torch.cat([h_cls, h_mean], dim=-1))


class CausalAttentionBackbone(nn.Module):
    """Combine a tabular representation with a causal graph branch."""

    def __init__(
        self,
        var_specs: List[VarSpec],
        emb_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        causal_attn_mask: Optional[np.ndarray],
        causal_attn_bias: Optional[np.ndarray] = None,
        causal_focus_idx: Optional[Sequence[int]] = None,
        variant: str = "full",
        causal_residual_dropout: float = config.DEFAULT_CAUSAL_RESIDUAL_DROPOUT,
    ):
        super().__init__()
        self.variant = str(variant)
        use_std_emb = self.variant == "std_emb"
        self.emb: nn.Module = (
            StandardEmbedding(var_specs, emb_dim)
            if use_std_emb
            else OrdinalAwareEmbedding(var_specs, emb_dim)
        )
        self.emb_dim = int(emb_dim)
        self.n_vars = int(len(var_specs))
        in_dim = self.n_vars * self.emb_dim
        hidden = max(self.emb_dim * 4, 192)
        self.main_in = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.main_blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden, dropout) for _ in range(max(1, int(n_layers) + 1))]
        )
        self.main_out = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, self.emb_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        focus_idx = list(
            sorted(set((int(i) for i in causal_focus_idx or [] if 0 <= int(i) < self.n_vars)))
        )
        self.focus_idx_list = focus_idx
        self.has_causal_branch = len(focus_idx) >= 2 and self.variant != "mlp_only"
        if self.has_causal_branch:
            self.register_buffer("focus_idx", torch.tensor(focus_idx, dtype=torch.long))
            W = _subset_graph_matrix(
                causal_attn_mask, causal_attn_bias, focus_idx, variant=self.variant
            )
            if W is None:
                W = np.eye(len(focus_idx), dtype=np.float32)
            self.register_buffer("causal_W", torch.from_numpy(W.astype(np.float32)))
            self.causal_node = nn.Sequential(
                nn.Linear(self.emb_dim * 3, self.emb_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.emb_dim * 2, self.emb_dim),
                nn.GELU(),
            )
            self.causal_proj = nn.Sequential(
                nn.LayerNorm(self.emb_dim),
                nn.Dropout(float(causal_residual_dropout)),
                nn.Linear(self.emb_dim, self.emb_dim),
            )
            self.gate = nn.Sequential(
                nn.Linear(self.emb_dim * 2, self.emb_dim), nn.GELU(), nn.Linear(self.emb_dim, 1)
            )
        else:
            self.register_buffer("focus_idx", None)
            self.register_buffer("causal_W", None)
            self.causal_node = None
            self.causal_proj = None
            self.gate = None
        self.out_norm = nn.LayerNorm(self.emb_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        E = self.emb(X)
        B = E.shape[0]
        flat = E.reshape(B, -1)
        h = self.main_in(flat)
        for blk in self.main_blocks:
            h = blk(h)
        h_main = self.main_out(h)
        if not self.has_causal_branch or self.focus_idx is None or self.causal_W is None:
            return self.out_norm(h_main)
        Ef = E.index_select(1, self.focus_idx)
        agg = torch.einsum("ij,bjd->bid", self.causal_W, Ef)
        pair = torch.cat([Ef, agg, Ef * agg], dim=-1)
        node_repr = self.causal_node(pair)
        h_causal = node_repr.mean(dim=1)
        h_causal = self.causal_proj(h_causal)
        alpha = torch.sigmoid(self.gate(torch.cat([h_main, h_causal], dim=-1)))
        return self.out_norm(h_main + alpha * h_causal)


class MLPOnlyBackbone(nn.Module):
    def __init__(
        self, var_specs: List[VarSpec], emb_dim: int, n_layers: int = 3, dropout: float = 0.15
    ):
        super().__init__()
        self.emb = OrdinalAwareEmbedding(var_specs, emb_dim)
        n_vars = len(var_specs)
        in_dim = emb_dim * n_vars
        hidden = emb_dim * 2
        layers: List[nn.Module] = []
        for i in range(n_layers - 1):
            layers += [
                nn.Linear(in_dim if i == 0 else hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(hidden if n_layers > 1 else in_dim, emb_dim))
        self.mlp = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        E = self.emb(X)
        B = E.shape[0]
        flat = E.reshape(B, -1)
        out = self.mlp(flat)
        return self.ln(out)


class AblationBackbone(nn.Module):
    def __init__(
        self,
        var_specs: List[VarSpec],
        emb_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
        causal_attn_mask: Optional[np.ndarray],
        causal_attn_bias: Optional[np.ndarray] = None,
        variant: str = "full",
        prior_mode: str = config.DEFAULT_ATTN_PRIOR_MODE,
        prior_strength: float = config.DEFAULT_ATTN_PRIOR_STRENGTH,
        nonancestor_penalty: float = config.DEFAULT_ATTN_NONANCESTOR_PENALTY,
    ):
        super().__init__()
        self.variant = variant
        if variant == "std_emb":
            self.emb: nn.Module = StandardEmbedding(var_specs, emb_dim)
        else:
            self.emb = OrdinalAwareEmbedding(var_specs, emb_dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(emb_dim)
        self.pool_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim), nn.GELU(), nn.LayerNorm(emb_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        attn_mask_np: Optional[np.ndarray] = None
        attn_bias_np: Optional[np.ndarray] = None
        mode = str(prior_mode).lower().strip()
        if variant in ("full", "std_emb"):
            attn_mask_np = causal_attn_mask
            attn_bias_np = causal_attn_bias
        elif variant == "no_mask":
            attn_mask_np = None
            attn_bias_np = None
        elif variant == "random_mask":
            if causal_attn_mask is not None:
                rand_anc = _build_random_dag_mask(causal_attn_mask, seed=999)
                if mode == "hard":
                    attn_mask_np = rand_anc
                    attn_bias_np = None
                elif mode == "hybrid":
                    attn_mask_np = rand_anc
                    attn_bias_np = np.zeros_like(rand_anc, dtype=np.float32)
                else:
                    attn_mask_np = np.ones_like(rand_anc, dtype=np.bool_)
                    attn_bias_np = np.where(rand_anc, 0.0, float(nonancestor_penalty)).astype(
                        np.float32
                    )
            else:
                attn_mask_np = None
                attn_bias_np = None
        if attn_mask_np is None:
            self.register_buffer("attn_mask", None)
        else:
            M = attn_mask_np.shape[0]
            full_mask = np.ones((M + 1, M + 1), dtype=np.bool_)
            full_mask[1:, 1:] = attn_mask_np.astype(np.bool_)
            full_mask[0, :] = True
            full_mask[:, 0] = True
            self.register_buffer("attn_mask", torch.from_numpy(full_mask))
        if attn_bias_np is None:
            self.register_buffer("attn_bias", None)
        else:
            M = attn_bias_np.shape[0]
            full_bias = np.zeros((M + 1, M + 1), dtype=np.float32)
            full_bias[1:, 1:] = attn_bias_np.astype(np.float32)
            self.register_buffer("attn_bias", torch.from_numpy(full_bias))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        x = self.emb(X)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x, attn_mask=self.attn_mask, attn_bias=self.attn_bias)
        x = self.ln(x)
        h_cls = x[:, 0, :]
        h_mean = x[:, 1:, :].mean(dim=1)
        return self.pool_proj(torch.cat([h_cls, h_mean], dim=-1))
