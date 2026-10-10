"""Prediction objectives and auxiliary regularizers."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.embeddings import OrdinalAwareEmbedding


class FocalLossWithLogits(nn.Module):
    """Focal loss with optional label smoothing."""

    def __init__(
        self,
        alpha: float = 0.55,
        gamma: float = 1.0,
        label_smoothing: float = 0.02,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = self.label_smoothing
        y_smooth = targets * (1.0 - eps) + 0.5 * eps
        bce = F.binary_cross_entropy_with_logits(logits, y_smooth, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t).pow(self.gamma)
        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class HybridScreeningLoss(nn.Module):
    """Combine weighted BCE, sampled ranking, and prevalence regularization."""

    def __init__(
        self,
        pos_weight: float = 1.0,
        rank_weight: float = 0.1,
        margin_weight: float = 0.015,
        target_margin: float = 0.25,
        label_smoothing: float = 0.01,
        max_rank_pairs: int = 256,
        train_prevalence: float = 0.12,
        prior_weight: float = 0.08,
        overpredict_weight: float = 0.1,
        overpredict_tol: float = 0.045,
    ):
        super().__init__()
        self.pos_weight = float(max(pos_weight, 1.0))
        self.rank_weight = float(max(rank_weight, 0.0))
        self.margin_weight = float(max(margin_weight, 0.0))
        self.target_margin = float(max(target_margin, 0.0))
        self.label_smoothing = float(max(label_smoothing, 0.0))
        self.max_rank_pairs = int(max(16, max_rank_pairs))
        self.train_prevalence = float(min(max(train_prevalence, 0.0001), 1.0 - 0.0001))
        self.prior_weight = float(max(prior_weight, 0.0))
        self.overpredict_weight = float(max(overpredict_weight, 0.0))
        self.overpredict_tol = float(max(overpredict_tol, 0.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = self.label_smoothing
        y_smooth = targets * (1.0 - eps) + 0.5 * eps
        pos_w = torch.as_tensor(self.pos_weight, dtype=logits.dtype, device=logits.device)
        loss_bce = F.binary_cross_entropy_with_logits(
            logits, y_smooth, pos_weight=pos_w, reduction="mean"
        )
        pos = logits[targets > 0.5]
        neg = logits[targets <= 0.5]
        if pos.numel() > 0 and neg.numel() > 0 and (self.rank_weight > 0.0):
            n_pairs = int(min(self.max_rank_pairs, pos.numel(), neg.numel()))
            pos_s = pos[
                torch.randint(low=0, high=pos.numel(), size=(n_pairs,), device=logits.device)
            ]
            neg_s = neg[
                torch.randint(low=0, high=neg.numel(), size=(n_pairs,), device=logits.device)
            ]
            loss_rank = F.softplus(-(pos_s - neg_s)).mean()
        else:
            loss_rank = logits.new_zeros(())
        if self.margin_weight > 0.0 and self.target_margin > 0.0:
            signed_y = targets * 2.0 - 1.0
            loss_margin = F.relu(self.target_margin - signed_y * logits).mean()
        else:
            loss_margin = logits.new_zeros(())
        probs = torch.sigmoid(logits)
        mean_prob = probs.mean()
        target_prev = torch.as_tensor(
            self.train_prevalence, dtype=logits.dtype, device=logits.device
        )
        loss_prior = (mean_prob - target_prev).pow(2)
        allowed_upper = target_prev + self.overpredict_tol
        loss_over = F.relu(mean_prob - allowed_upper).pow(2)
        return (
            loss_bce
            + self.rank_weight * loss_rank
            + self.margin_weight * loss_margin
            + self.prior_weight * loss_prior
            + self.overpredict_weight * loss_over
        )


class AsymmetricLoss(nn.Module):
    """Apply separate focusing exponents to positive and negative labels."""

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-08,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_neg = (probs + self.clip).clamp(max=1.0)
        xs_pos = probs
        xs_neg = 1.0 - probs_neg
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0.0 or self.gamma_pos > 0.0:
            pt0 = xs_neg * (1.0 - targets)
            pt1 = xs_pos * targets
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            one_sided_w = torch.pow(1.0 - pt, one_sided_gamma)
            loss = loss * one_sided_w
        loss = -loss
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class ListNetRankLoss(nn.Module):
    """Compute a smooth listwise ranking loss."""

    def __init__(self, pos_weight: float = 8.0, temperature: float = 1.0, eps: float = 1e-08):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.shape[0]
        if n < 2:
            return logits.new_zeros(())
        relevance = targets * float(self.pos_weight - 1.0) + 1.0
        pred_dist = torch.softmax(logits / float(self.temperature), dim=0)
        tgt_dist = torch.softmax(relevance, dim=0)
        kl = (tgt_dist * torch.log(tgt_dist / pred_dist.clamp(min=self.eps))).sum()
        return kl


class AUCMLoss(nn.Module):
    """AUC margin loss with bounded moving-average updates."""

    def __init__(
        self,
        margin: float = 1.0,
        p_hat: float = 0.12,
        ema_decay: float = 0.95,
        lam_max: float = 5.0,
    ):
        super().__init__()
        self.margin = float(margin)
        self.p_hat = float(p_hat)
        self.ema = float(ema_decay)
        self.lam_max = float(lam_max)
        self.register_buffer("a_var", torch.tensor(0.0))
        self.register_buffer("b_var", torch.tensor(1.0))
        self.register_buffer("lam", torch.tensor(0.1))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() < 2 or neg_mask.sum() < 2:
            return logits.new_zeros(())
        f_pos = probs[pos_mask]
        f_neg = probs[neg_mask]
        a = self.a_var.detach()
        b = self.b_var.detach()
        lam = self.lam.detach().clamp(min=0.0, max=self.lam_max)
        loss_pos = (f_pos - a).pow(2).mean()
        loss_neg = (f_neg - b).pow(2).mean()
        mean_pos = f_pos.mean()
        mean_neg = f_neg.mean()
        coupling_inner = (self.margin + mean_pos - mean_neg).clamp(-2.0, 2.0)
        aucm_loss = loss_pos + loss_neg + 2.0 * lam * coupling_inner - lam.pow(2)
        with torch.no_grad():
            a_grad = 2.0 * (a.item() - mean_pos.item())
            b_grad = 2.0 * (b.item() - mean_neg.item())
            lam_grad = float(self.margin) + mean_pos.item() - mean_neg.item()
            a_new = a.item() - (1.0 - self.ema) * a_grad
            b_new = b.item() - (1.0 - self.ema) * b_grad
            lam_new = float(np.clip(lam.item() + (1.0 - self.ema) * lam_grad, 0.0, self.lam_max))
            self.a_var.copy_(torch.tensor(float(np.clip(a_new, -2.0, 2.0))))
            self.b_var.copy_(torch.tensor(float(np.clip(b_new, -2.0, 2.0))))
            self.lam.copy_(torch.tensor(lam_new))
        return aucm_loss.clamp(-10.0, 10.0)


class UncertaintyWeightedMTL(nn.Module):
    """Learn relative weights for the main and auxiliary tasks."""

    def __init__(self, n_tasks: int = 3, init_log_sigma: float = 0.0):
        super().__init__()
        self.log_vars = nn.Parameter(torch.full((n_tasks,), float(init_log_sigma)))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        total = losses[0].new_zeros(())
        for i, loss in enumerate(losses):
            if i >= len(self.log_vars):
                total = total + loss
                continue
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + 0.5 * self.log_vars[i]
        return total

    def get_weights(self) -> List[float]:
        return [float(torch.exp(-lv).item()) for lv in self.log_vars]


def pcgrad_project(
    grads_main: List[Optional[torch.Tensor]], grads_aux: List[Optional[torch.Tensor]]
) -> List[Optional[torch.Tensor]]:
    projected = []
    for g_m, g_a in zip(grads_main, grads_aux):
        if g_m is None or g_a is None:
            projected.append(g_a)
            continue
        g_m_flat = g_m.view(-1)
        g_a_flat = g_a.view(-1)
        dot = torch.dot(g_m_flat, g_a_flat)
        if dot < 0.0:
            g_m_norm_sq = torch.dot(g_m_flat, g_m_flat).clamp(min=1e-12)
            g_a_proj = g_a - dot / g_m_norm_sq * g_m
        else:
            g_a_proj = g_a
        projected.append(g_a_proj)
    return projected


def coral_loss(h_src: torch.Tensor, h_tgt: torch.Tensor, eps: float = 1e-05) -> torch.Tensor:
    """Match covariance matrices across domains."""
    d = h_src.shape[1]

    def _cov(h: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        h_c = h - h.mean(dim=0, keepdim=True)
        return h_c.T @ h_c / max(n - 1, 1)

    C_s = _cov(h_src)
    C_t = _cov(h_tgt)
    diff = C_s - C_t
    return diff.pow(2).sum() / (4.0 * d * d + eps)


def ordinal_monotonicity_loss(
    emb_module: "OrdinalAwareEmbedding", device: torch.device, sample_vars: int = 32
) -> torch.Tensor:
    """Penalize nonmonotone projections of ordinal category embeddings."""
    total = torch.zeros((), device=device)
    n_ord = 0
    specs = emb_module.var_specs
    n_vars = min(len(specs), sample_vars)
    indices = [i for i, s in enumerate(specs) if getattr(s, "is_ordinal", False)]
    if not indices:
        return total
    if len(indices) > n_vars:
        sel = np.random.choice(indices, size=n_vars, replace=False).tolist()
    else:
        sel = indices
    for j in sel:
        spec = specs[j]
        K = len(spec.values)
        if K < 3:
            continue
        table = emb_module._build_table(j, spec, device=device)
        direction = emb_module.direction[j].to(device)
        direction_n = direction / (direction.norm() + 1e-09)
        projections = table @ direction_n
        diffs = projections[1:] - projections[:-1]
        penalty = F.relu(-diffs).pow(2).mean()
        total = total + penalty
        n_ord += 1
    return total / max(n_ord, 1)


class SigmoidAPLoss(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
        n_pos_sample: int = 128,
        n_neg_sample: int = 512,
        eps: float = 1e-06,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.n_pos_sample = int(n_pos_sample)
        self.n_neg_sample = int(n_neg_sample)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_mask = targets > 0.5
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        if n_pos == 0 or n_neg == 0:
            return logits.new_zeros(())
        pos_logits = logits[pos_mask]
        neg_logits = logits[neg_mask]
        n_p = min(n_pos, self.n_pos_sample)
        n_n = min(n_neg, self.n_neg_sample)
        if n_p < n_pos:
            idx = torch.randperm(n_pos, device=logits.device)[:n_p]
            pos_logits = pos_logits[idx]
        if n_n < n_neg:
            idx = torch.randperm(n_neg, device=logits.device)[:n_n]
            neg_logits = neg_logits[idx]
        pos_expand = pos_logits.unsqueeze(1)
        neg_expand = neg_logits.unsqueeze(0)
        diff = (pos_expand - neg_expand) / self.temperature
        frac_neg_below = torch.sigmoid(diff).mean(dim=1)
        sorted_idx = torch.argsort(pos_logits, descending=True)
        frac_neg_below_sorted = frac_neg_below[sorted_idx]
        k = torch.arange(1, n_p + 1, device=logits.device, dtype=logits.dtype)
        n_neg_f = float(n_neg)
        est_rank = k + n_neg_f * (1.0 - frac_neg_below_sorted).clamp(0.0, 1.0)
        soft_precision = k / est_rank.clamp(min=self.eps)
        ap_approx = soft_precision.mean()
        return 1.0 - ap_approx
