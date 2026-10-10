import numpy as np
import pytest
import torch

from ca_resmtl.calibration import BetaCalibrator
from ca_resmtl.data import VarSpec
from ca_resmtl.graph import build_allowed_mask_with_priors, transitive_closure
from ca_resmtl.losses import AsymmetricLoss
from ca_resmtl.models.backbones import (
    CausalAttentionBackbone,
    DeepResMLPBackbone,
    FTTransformerBackbone,
)
from ca_resmtl.models.predictor import CAResMTL, DualHeadPredictor
from ca_resmtl.models.structure import DOCDModel


def specifications():
    return [
        VarSpec(f"{i + 1}、item", ["<phi>", "1", "2", "3"], True, True, (i + 1, 0))
        for i in range(4)
    ]


def test_structure_blocks_outgoing_outcome_edges():
    specs = specifications()
    mask = build_allowed_mask_with_priors(specs, y_index=3, use_question_order_prior=True)
    assert not mask[3].any()
    assert not np.diag(mask).any()
    model = DOCDModel(specs, emb_dim=8, allowed_mask=mask, y_index=3)
    probabilities = model.adj_prob().detach().numpy()
    assert not probabilities[3].any()


def test_reachability_follows_directed_paths():
    adjacency = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    reach = transitive_closure(adjacency)
    assert reach[0, 2] and not reach[2, 0]
    assert np.diag(reach).all()


@pytest.mark.parametrize("kind", ["causal_attention", "ft_transformer", "deep_resmlp"])
def test_predictor_branches_have_finite_gradients(kind):
    specs = specifications()
    X = torch.randint(0, 4, (16, len(specs)))
    if kind == "deep_resmlp":
        backbone = DeepResMLPBackbone(
            specs, emb_dim=16, n_blocks=2, dropout=0.0, causal_focus_idx=[0, 1], n_engineered=0
        )
        model = CAResMTL(backbone, emb_dim=16, dropout=0.0, n_aux_tasks=1)
    elif kind == "ft_transformer":
        backbone = FTTransformerBackbone(specs, emb_dim=16, n_layers=1, n_heads=4, dropout=0.0)
        model = DualHeadPredictor(backbone, emb_dim=16, dropout=0.0)
    else:
        backbone = CausalAttentionBackbone(
            specs,
            emb_dim=16,
            n_layers=1,
            n_heads=4,
            dropout=0.0,
            causal_attn_mask=np.zeros((4, 4), dtype=np.float32),
        )
        model = DualHeadPredictor(backbone, emb_dim=16, dropout=0.0)
    score, probability, fused, _, alpha = model.forward_all(X, return_details=True)
    assert score.shape == probability.shape == fused.shape == alpha.shape == (16,)
    assert torch.isfinite(fused).all() and ((alpha >= 0) & (alpha <= 1)).all()
    AsymmetricLoss()(fused, torch.arange(16).remainder(2).float()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_beta_identity_and_fitted_map_preserve_order():
    probabilities = np.linspace(0.01, 0.99, 200)
    calibrator = BetaCalibrator()
    np.testing.assert_allclose(calibrator.calibrate(probabilities), probabilities, atol=1e-6)
    labels = (np.random.default_rng(42).random(200) < probabilities).astype(int)
    calibrator.fit(probabilities, labels, verbose=False)
    calibrated = calibrator.calibrate(probabilities)
    assert np.isfinite(calibrated).all()
    assert np.all(np.diff(calibrated) >= 0)
