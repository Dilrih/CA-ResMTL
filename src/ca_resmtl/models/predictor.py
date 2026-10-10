"""Risk prediction and auxiliary task heads."""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .. import config
from ..losses import UncertaintyWeightedMTL
from .backbones import CausalAttentionBackbone, DeepResMLPBackbone


class CAResMTL(nn.Module):
    """Predict risk through score, probability, and auxiliary heads."""

    def __init__(
        self,
        backbone: DeepResMLPBackbone,
        emb_dim: int,
        dropout: float,
        n_aux_tasks: int = 0,
        logit_scale_init: float = 1.0,
        gate_max_delta: float = 0.35,
        aux_backbone_grad_scale: float = 0.35,
    ):
        super().__init__()
        self.backbone = backbone
        self.n_aux_tasks = int(n_aux_tasks)
        self.gate_max_delta = float(np.clip(gate_max_delta, 0.05, 0.45))
        self.aux_backbone_grad_scale = float(np.clip(aux_backbone_grad_scale, 0.05, 1.0))
        self.score_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(emb_dim, 1),
        )
        self.score_shortcut = nn.Linear(emb_dim, 1)
        self.score_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7)))
        self.prob_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.35),
            nn.Linear(emb_dim, 1),
        )
        self.prob_shortcut = nn.Linear(emb_dim, 1)
        self.prob_blend = nn.Parameter(torch.tensor(math.log(0.4 / 0.6)))
        self.prob_temperature_log = nn.Parameter(torch.zeros(1))
        gate_hidden = max(16, emb_dim // 2)
        self.fusion_gate = nn.Sequential(
            nn.Linear(emb_dim, gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(gate_hidden, 1),
        )
        if self.n_aux_tasks > 0:
            self.aux_heads = nn.ModuleList([nn.Linear(emb_dim, 1) for _ in range(self.n_aux_tasks)])
            self.uncertainty_mtl = UncertaintyWeightedMTL(
                n_tasks=1 + self.n_aux_tasks, init_log_sigma=0.0
            )
        else:
            self.aux_heads = None
            self.uncertainty_mtl = None
        init_scale = float(np.clip(logit_scale_init, 0.7, 2.0))
        self.logit_scale_log = nn.Parameter(torch.tensor(math.log(init_scale)))
        with torch.no_grad():
            self.score_head[-1].bias.data.fill_(0.0)
            self.score_shortcut.bias.data.fill_(0.0)
            self.prob_head[-1].bias.data.fill_(0.0)
            self.prob_shortcut.bias.data.fill_(0.0)
            if self.aux_heads:
                for h in self.aux_heads:
                    h.bias.data.fill_(0.0)

    def _apply_logit_scale(self, raw: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.logit_scale_log.clamp(math.log(0.7), math.log(2.5)))
        return raw * scale

    def _aux_feature_mix(self, h: torch.Tensor) -> torch.Tensor:
        g = float(self.aux_backbone_grad_scale)
        if g >= 0.999:
            return h
        return h * g + h.detach() * (1.0 - g)

    def forward_features(
        self, X: torch.Tensor, X_eng: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.backbone(X, X_eng=X_eng)

    def score_from_features(self, h: torch.Tensor) -> torch.Tensor:
        blend = torch.sigmoid(self.score_blend)
        main = self.score_head(h).squeeze(-1)
        skip = self.score_shortcut(h).squeeze(-1)
        return self._apply_logit_scale((1.0 - blend) * main + blend * skip)

    def prob_from_features(self, h: torch.Tensor) -> torch.Tensor:
        blend = torch.sigmoid(self.prob_blend)
        main = self.prob_head(h).squeeze(-1)
        skip = self.prob_shortcut(h).squeeze(-1)
        raw = (1.0 - blend) * main + blend * skip
        temp = torch.exp(self.prob_temperature_log.clamp(math.log(0.7), math.log(1.8)))
        return raw / temp

    def aux_logits_from_features(self, h: torch.Tensor) -> List[torch.Tensor]:
        if self.aux_heads is None:
            return []
        h_aux = self._aux_feature_mix(h)
        return [head(h_aux).squeeze(-1) for head in self.aux_heads]

    def fused_logit_from_branches(
        self,
        score_logit: torch.Tensor,
        prob_logit: Optional[torch.Tensor] = None,
        h: Optional[torch.Tensor] = None,
        sample_stats: Optional[torch.Tensor] = None,
        return_alpha: bool = False,
    ):
        if prob_logit is None:
            prob_logit = score_logit
        if h is None:
            alpha = torch.full_like(score_logit, 0.5)
        else:
            alpha_raw = torch.sigmoid(self.fusion_gate(h).squeeze(-1))
            delta = float(self.gate_max_delta)
            alpha = 0.5 + (alpha_raw - 0.5) * (2.0 * delta)
        fused = (1.0 - alpha) * score_logit + alpha * prob_logit
        if return_alpha:
            return (fused, alpha)
        return fused

    def forward_all(
        self,
        X: torch.Tensor,
        X_eng: Optional[torch.Tensor] = None,
        detach_prob_features: bool = False,
        return_details: bool = False,
    ):
        h = self.forward_features(X, X_eng=X_eng)
        score_logit = self.score_from_features(h)
        prob_h = h.detach() if detach_prob_features else h
        prob_logit = self.prob_from_features(prob_h)
        fused_logit, alpha = self.fused_logit_from_branches(
            score_logit, prob_logit, h=h, return_alpha=True
        )
        if return_details:
            return (score_logit, prob_logit, fused_logit, h, alpha)
        return (score_logit, prob_logit, fused_logit)

    def forward_heads(
        self,
        X: torch.Tensor,
        X_eng: Optional[torch.Tensor] = None,
        detach_prob_features: bool = False,
    ):
        score_logit, prob_logit, _ = self.forward_all(
            X, X_eng=X_eng, detach_prob_features=detach_prob_features, return_details=False
        )
        return (score_logit, prob_logit)

    def forward_score(self, X: torch.Tensor, X_eng: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_all(X, X_eng=X_eng, detach_prob_features=False, return_details=False)[0]

    def forward(self, X: torch.Tensor, X_eng: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_score(X, X_eng=X_eng)

    def forward_multitask(self, X: torch.Tensor, X_eng: Optional[torch.Tensor] = None):
        h = self.forward_features(X, X_eng=X_eng)
        main_logit = self.score_from_features(h)
        aux_logits = self.aux_logits_from_features(h)
        return (main_logit, aux_logits)


class DualHeadPredictor(nn.Module):
    """Fuse risk-score and probability branches."""

    def __init__(
        self,
        backbone: CausalAttentionBackbone,
        emb_dim: int,
        dropout: float,
        num_domains: int = 0,
        domain_adv: bool = False,
        logit_scale_init: float = config.DEFAULT_LOGIT_SCALE_INIT,
    ):
        super().__init__()
        self.backbone = backbone
        self.score_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(emb_dim, 1),
        )
        self.score_shortcut = nn.Linear(emb_dim, 1)
        self.score_blend = nn.Parameter(torch.tensor(math.log(0.3 / 0.7), dtype=torch.float32))
        with torch.no_grad():
            self.score_head[-1].bias.data.fill_(0.0)
            self.score_shortcut.bias.data.fill_(0.0)
        init_scale = float(np.clip(logit_scale_init, 0.7, 2.0))
        self.logit_scale_log = nn.Parameter(torch.tensor(math.log(init_scale), dtype=torch.float32))
        self.domain_adv = bool(domain_adv) and int(num_domains) > 1
        self.num_domains = int(num_domains)
        if self.domain_adv:
            self.domain_head = nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(emb_dim, int(num_domains)),
            )
        else:
            self.domain_head = None

    def _apply_logit_scale(self, raw_logit: torch.Tensor) -> torch.Tensor:
        scale = torch.exp(self.logit_scale_log.clamp(math.log(0.7), math.log(2.5)))
        return raw_logit * scale

    def score_from_features(self, h: torch.Tensor) -> torch.Tensor:
        blend = torch.sigmoid(self.score_blend)
        main = self.score_head(h).squeeze(-1)
        skip = self.score_shortcut(h).squeeze(-1)
        raw = (1.0 - blend) * main + blend * skip
        return self._apply_logit_scale(raw)

    def forward_features(
        self, X: torch.Tensor, X_eng: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.backbone(X)

    def forward_score(self, X: torch.Tensor, X_eng: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.forward_features(X)
        return self.score_from_features(h)

    def forward_all(
        self,
        X: torch.Tensor,
        X_eng: Optional[torch.Tensor] = None,
        detach_prob_features: bool = False,
        return_details: bool = False,
    ):
        h = self.forward_features(X)
        score_logit = self.score_from_features(h)
        if return_details:
            alpha = torch.full_like(score_logit, 0.0)
            return (score_logit, score_logit, score_logit, h, alpha)
        return (score_logit, score_logit, score_logit)

    def forward_heads(
        self,
        X: torch.Tensor,
        X_eng: Optional[torch.Tensor] = None,
        detach_prob_features: bool = False,
    ):
        score_logit = self.forward_score(X)
        return (score_logit, score_logit)

    def fused_logit_from_branches(
        self,
        score_logit: torch.Tensor,
        prob_logit: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        return_alpha: bool = False,
    ):
        if return_alpha:
            return (score_logit, torch.zeros_like(score_logit))
        return score_logit

    def forward(self, X: torch.Tensor, X_eng: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_score(X)

    def forward_domain(self, X: torch.Tensor, grl_lambda: float = 1.0):
        h = self.forward_features(X)
        score_logit = self.score_from_features(h)
        if self.domain_head is None:
            return (score_logit, score_logit, None)
        dom_logits = self.domain_head(grad_reverse(h, float(grl_lambda)))
        return (score_logit, score_logit, dom_logits)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (-ctx.lambd * grad_output, None)


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, float(lambd))
