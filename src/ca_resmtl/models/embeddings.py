"""Categorical and ordinal variable embeddings."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import VarSpec


class OrdinalAwareEmbedding(nn.Module):
    """Represent ordered categories and missingness separately."""

    def __init__(self, var_specs: List[VarSpec], emb_dim: int):
        super().__init__()
        self.var_specs = var_specs
        self.emb_dim = emb_dim
        self.base = nn.ParameterList()
        self.direction = nn.ParameterList()
        self.increments = nn.ParameterList()
        self.offsets = nn.ParameterList()
        for spec in var_specs:
            K = len(spec.values)
            self.base.append(nn.Parameter(torch.zeros(emb_dim)))
            self.direction.append(nn.Parameter(torch.randn(emb_dim) * 0.02))
            inc = torch.zeros(max(1, K - 1))
            self.increments.append(nn.Parameter(inc))
            off = torch.zeros((max(1, K), emb_dim))
            self.offsets.append(nn.Parameter(off))
        self.reset_parameters()

    def reset_parameters(self):
        for b in self.base:
            nn.init.normal_(b, mean=0.0, std=0.02)
        for d in self.direction:
            nn.init.normal_(d, mean=0.0, std=0.02)
        for inc in self.increments:
            nn.init.constant_(inc, 0.0)
        for off in self.offsets:
            nn.init.normal_(off, mean=0.0, std=0.01)

    def _build_table(self, j: int, spec: VarSpec, device: torch.device) -> torch.Tensor:
        K = len(spec.values)
        base = self.base[j].to(device)
        off = self.offsets[j].to(device)
        if off.shape[0] < K:
            pad = torch.zeros((K - off.shape[0], self.emb_dim), device=device, dtype=off.dtype)
            off = torch.cat([off, pad], dim=0)
        table = torch.zeros((K, self.emb_dim), device=device, dtype=base.dtype)
        table[0] = base + off[0]
        if not bool(getattr(spec, "is_ordinal", False)):
            for k in range(1, K):
                table[k] = base + off[k]
            return table
        inc = self.increments[j].to(device)
        if inc.numel() < K - 1:
            pad = torch.zeros((K - 1 - inc.numel(),), device=device, dtype=inc.dtype)
            inc = torch.cat([inc, pad], dim=0)
        deltas = F.softplus(inc[: max(1, K - 1)])
        alphas = torch.cumsum(deltas, dim=0)
        direction = self.direction[j].to(device)
        direction = direction / (direction.norm() + 1e-09)
        for k in range(1, K):
            alpha = alphas[k - 1] if K - 1 > 0 else 0.0
            table[k] = base + alpha * direction + off[k]
        return table

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        device = X.device
        B, M = X.shape
        if torch.is_floating_point(X):
            X = torch.nan_to_num(X, nan=0.0).round().long()
        else:
            X = X.long()
        out = torch.zeros((B, M, self.emb_dim), device=device)
        for j, spec in enumerate(self.var_specs):
            K = len(spec.values)
            xj = X[:, j].clamp(0, K - 1)
            table = self._build_table(j, spec, device=device)
            out[:, j, :] = table[xj]
        return out

    @torch.no_grad()
    def precompute_tables(self, device: Optional[torch.device] = None) -> List[torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        cache_dev = str(device)
        if (
            not self.training
            and hasattr(self, "_cached_tables")
            and (getattr(self, "_cached_tables") is not None)
            and (getattr(self, "_cached_device", None) == cache_dev)
        ):
            return getattr(self, "_cached_tables")
        tables: List[torch.Tensor] = []
        for j, spec in enumerate(self.var_specs):
            tables.append(self._build_table(j, spec, device=device))
        setattr(self, "_cached_tables", tables)
        setattr(self, "_cached_device", cache_dev)
        return tables

    @torch.no_grad()
    def forward_cached(
        self, X: torch.Tensor, tables: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        device = X.device
        if tables is None:
            tables = self.precompute_tables(device=device)
        B, M = X.shape
        out = torch.zeros((B, M, self.emb_dim), device=device, dtype=tables[0].dtype)
        for j, spec in enumerate(self.var_specs):
            K = len(spec.values)
            xj = X[:, j].clamp(0, K - 1).long()
            out[:, j, :] = tables[j].index_select(0, xj)
        return out

    @torch.no_grad()
    def embed_column_cached(
        self, j: int, xj: torch.Tensor, tables: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        if tables is None:
            tables = self.precompute_tables(device=xj.device)
        K = int(tables[j].shape[0])
        idx = xj.clamp(0, K - 1).long()
        return tables[j].index_select(0, idx)


class StandardEmbedding(nn.Module):
    def __init__(self, var_specs: List[VarSpec], emb_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.tables = nn.ModuleList()
        for spec in var_specs:
            K = max(2, len(spec.values))
            self.tables.append(nn.Embedding(K, emb_dim, padding_idx=0))
        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.tables:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)
            if emb.padding_idx is not None:
                with torch.no_grad():
                    emb.weight[emb.padding_idx].fill_(0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, M = X.shape
        device = X.device
        if torch.is_floating_point(X):
            X = X.round().long()
        else:
            X = X.long()
        out = torch.zeros(B, M, self.emb_dim, device=device, dtype=torch.float32)
        for j in range(M):
            K = self.tables[j].num_embeddings
            xj = X[:, j].clamp(0, K - 1)
            out[:, j, :] = self.tables[j](xj)
        return out

    @torch.no_grad()
    def precompute_tables(self, device=None):
        if device is None:
            device = next(self.parameters()).device
        return [self.tables[j].weight.to(device) for j in range(len(self.tables))]

    @torch.no_grad()
    def forward_cached(self, X: torch.Tensor, tables=None) -> torch.Tensor:
        return self.forward(X)
