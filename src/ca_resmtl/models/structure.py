"""Conditional distributions and directed structure learning."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config
from ..data import VarSpec
from ..graph import acyclicity_hutchinson_expm, gumbel_sigmoid
from ..utils import finite_check
from .embeddings import OrdinalAwareEmbedding


class OrderedProbitHead(nn.Module):
    def __init__(self, n_cats_real: int):
        super().__init__()
        self.n_cats_real = max(1, n_cats_real)
        if self.n_cats_real <= 1:
            self.raw = nn.Parameter(torch.zeros(1))
        else:
            self.raw = nn.Parameter(torch.zeros(self.n_cats_real - 1))

    def thresholds(self) -> torch.Tensor:
        inc = F.softplus(self.raw)
        return torch.cumsum(inc, dim=0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        K = self.n_cats_real
        if K <= 1:
            return torch.ones((B, 1), device=z.device)
        th = self.thresholds().to(z.device)

        def Phi(x):
            return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

        bounds = torch.cat(
            [
                torch.tensor([-1000000.0], device=z.device),
                th,
                torch.tensor([1000000.0], device=z.device),
            ],
            dim=0,
        )
        probs = []
        for k in range(1, K + 1):
            upper = bounds[k] - z
            lower = bounds[k - 1] - z
            pk = Phi(upper) - Phi(lower)
            probs.append(pk)
        return torch.stack(probs, dim=-1).clamp_min(1e-12)


class DOCDModel(nn.Module):
    """Learn conditional distributions and a directed X-to-Y structure."""

    def __init__(
        self,
        var_specs: List[VarSpec],
        emb_dim: int,
        allowed_mask: np.ndarray,
        y_index: int,
        y_nll_weight: float = config.DEFAULT_Y_NLL_WEIGHT,
        init_logit: float = config.DEFAULT_INIT_LOGIT,
        init_logit_y_in: float = config.DEFAULT_INIT_LOGIT_Y_IN,
        y_in_prior: Optional[np.ndarray] = None,
        y_in_prior_lambda: float = config.DEFAULT_MI_PRIOR_LAMBDA,
        mlp_hidden: int = 192,
        mlp_depth: int = 3,
    ):
        super().__init__()
        self.var_specs = var_specs
        self.M = len(var_specs)
        self.emb_dim = emb_dim
        self.y_index = int(y_index)
        self.y_nll_weight = float(y_nll_weight)
        self.y_in_prior_lambda = float(y_in_prior_lambda)
        if y_in_prior is None:
            prior = np.zeros((self.M,), dtype=np.float32)
        else:
            prior = np.asarray(y_in_prior, dtype=np.float32).reshape(-1)
            if prior.shape[0] != self.M:
                raise ValueError(f"y_in_prior length mismatch: {prior.shape[0]} vs M={self.M}")
        if self.y_index < prior.shape[0]:
            prior[self.y_index] = 0.0
        self.register_buffer("y_in_prior", torch.from_numpy(prior))
        self.embed = OrdinalAwareEmbedding(var_specs, emb_dim=emb_dim)
        self.S = nn.Parameter(torch.zeros((self.M, self.M)))
        with torch.no_grad():
            self.S.fill_(float(init_logit))
            self.S.fill_diagonal_(-14.0)
            self.S[:, self.y_index].fill_(float(init_logit_y_in))
            self.S[self.y_index, self.y_index] = -14.0
        self.register_buffer("allowed_mask", torch.from_numpy(allowed_mask.astype(np.float32)))
        self.f_list = nn.ModuleList()
        for _ in range(self.M):
            layers = []
            in_dim = emb_dim
            for _d in range(mlp_depth - 1):
                layers.append(nn.Linear(in_dim, mlp_hidden))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(0.1))
                in_dim = mlp_hidden
            layers.append(nn.Linear(in_dim, 1))
            self.f_list.append(nn.Sequential(*layers))
        self.probit_heads = nn.ModuleList(
            [OrderedProbitHead(len(spec.values) - 1) for spec in var_specs]
        )
        self.z_scale = nn.Parameter(torch.ones(self.M))
        self.z_bias = nn.Parameter(torch.zeros(self.M))

    def masked_logits(self) -> torch.Tensor:
        S = self.S - torch.diag_embed(torch.diag(self.S))
        huge_neg = torch.tensor(-16.0, device=S.device, dtype=S.dtype)
        S = torch.where(self.allowed_mask.to(S.device) > 0, S, huge_neg)
        S = S - torch.diag_embed(torch.diag(S))
        return S

    def adj_prob(self) -> torch.Tensor:
        S = self.masked_logits()
        A = torch.sigmoid(S) * self.allowed_mask.to(S.device)
        A = A * (1.0 - torch.eye(self.M, device=A.device))
        return A

    def sample_adj(self, tau: float, hard: bool = False) -> torch.Tensor:
        S = self.masked_logits()
        A = gumbel_sigmoid(S, tau=tau, hard=hard)
        A = A * self.allowed_mask.to(A.device)
        A = A * (1.0 - torch.eye(self.M, device=A.device))
        return A

    def forward(self, X: torch.Tensor, tau: float, mix_sample: float) -> Dict[str, torch.Tensor]:
        B, M = X.shape
        assert M == self.M
        A_prob = self.adj_prob()
        A_samp = self.sample_adj(tau=tau, hard=False)
        A_used = (1.0 - mix_sample) * A_prob + mix_sample * A_samp
        finite_check("A_prob", A_prob)
        finite_check("A_used", A_used)
        E = self.embed(X)
        finite_check("E", E)
        P = torch.einsum("bid,ij->bjd", E, A_used)
        finite_check("P", P)
        z_list = []
        for j in range(self.M):
            zj = self.f_list[j](P[:, j, :]).squeeze(-1)
            z_list.append(zj)
        z = torch.stack(z_list, dim=1)
        z = z * self.z_scale + self.z_bias
        finite_check("z", z)
        nlls = []
        for j in range(self.M):
            probs = self.probit_heads[j](z[:, j])
            xj = X[:, j].clamp_min(0)
            mask = (xj > 0).float()
            idx = (xj - 1).clamp_min(0)
            idx = torch.minimum(idx, torch.tensor(probs.shape[1] - 1, device=idx.device))
            pj = probs.gather(1, idx.view(-1, 1)).squeeze(1).clamp_min(1e-12)
            nll = (-torch.log(pj) * mask).mean()
            nlls.append(nll)
        nll_per_var = torch.stack(nlls, dim=0)
        y_idx = self.y_index
        mask_x = torch.ones((self.M,), device=X.device, dtype=torch.bool)
        mask_x[y_idx] = False
        nll_x = nll_per_var[mask_x].mean()
        nll_y = nll_per_var[y_idx]
        prior_term = torch.zeros((), device=X.device)
        if self.y_in_prior_lambda > 0.0:
            w = self.y_in_prior.to(X.device)
            denom = w.sum().clamp_min(1e-06)
            if float(denom.item()) > 0.0:
                prior_term = -self.y_in_prior_lambda * (A_prob[:, y_idx] * w).sum() / denom
        nll = nll_x + self.y_nll_weight * nll_y + prior_term
        h_raw = acyclicity_hutchinson_expm(
            A_prob,
            n_trace=config.DEFAULT_HUTCH_SAMPLES,
            series_k=config.DEFAULT_HUTCH_SERIES_K,
            scale_by_dim=True,
            clamp_nonneg=False,
        )
        h = h_raw.clamp_min(0.0)
        finite_check("nll", nll)
        finite_check("h_raw", h_raw)
        finite_check("h", h)
        return {
            "nll": nll,
            "nll_x": nll_x,
            "nll_y": nll_y,
            "prior_term": prior_term,
            "nll_per_var": nll_per_var,
            "A_prob": A_prob,
            "A_used": A_used,
            "h": h,
            "h_raw": h_raw,
        }

    @torch.no_grad()
    def adjacency_prob_numpy(self) -> np.ndarray:
        A = self.adj_prob().detach().cpu().numpy()
        np.fill_diagonal(A, 0.0)
        return A
