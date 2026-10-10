import json

import numpy as np
import pandas as pd
import pytest
import torch

from ca_resmtl import optional
from ca_resmtl.data import VarSpec
from ca_resmtl.demo import demo_arguments, generate_synthetic_data
from ca_resmtl.features import CausalFeatureEngineer
from ca_resmtl.graph import build_allowed_mask_with_priors
from ca_resmtl.pipeline import run_experiment
from ca_resmtl.training import bootstrap_structure_y_parents


def test_feature_selection_does_not_use_holdout_labels():
    rng = np.random.default_rng(42)
    X = rng.integers(0, 5, (80, 5))
    y = rng.integers(0, 2, 80)
    names = [f"{i + 10}、item" for i in range(5)]
    train = np.arange(50)
    first = CausalFeatureEngineer(names, names, n_interaction_pairs=3)
    second = CausalFeatureEngineer(names, names, n_interaction_pairs=3)
    changed_y = y.copy()
    changed_y[50:] = 1 - changed_y[50:]
    changed_X = X.copy()
    changed_X[50:] = 4
    a = first.fit_transform(X, y, train)
    b = second.fit_transform(changed_X, changed_y, train)
    assert first.interaction_pairs == second.interaction_pairs
    np.testing.assert_allclose(a[train], b[train])
    np.testing.assert_allclose(first.transform(X[:1]), a[:1])


def test_bootstrap_stability_uses_multiple_resamples():
    variables = [
        VarSpec(f"{i + 1}、item", ["<phi>", "1", "2"], True, False, (i + 1, 0)) for i in range(4)
    ]
    matrix = np.random.default_rng(42).integers(1, 3, (24, 4))
    result = bootstrap_structure_y_parents(
        X_xy=matrix,
        var_specs_xy=variables,
        names_xy=[item.name for item in variables],
        y_index=3,
        allowed_mask=build_allowed_mask_with_priors(variables, y_index=3),
        idx_train=np.arange(24),
        device=torch.device("cpu"),
        n_bootstrap=2,
        edge_threshold=0.01,
        lr_docd=0.001,
        lr_struct_mult=4,
        inner_steps=2,
        outer_loops=1,
        min_outer_loops=1,
        l1_lambda=0.0006,
        l1_y_in_mult=0.2,
        dag_tol=1e-7,
        rho_max=1e6,
        batch_size=16,
        emb_dim=8,
        y_nll_weight=25,
        init_logit=-3.6,
        init_logit_y_in=-2.7,
        y_in_prior=None,
        y_in_prior_lambda=0,
        topn=2,
    )
    assert len(result) == 3
    assert result["freq_rate_topn"].between(0, 1).all()
    assert np.isfinite(result["mean_prob_topn"]).all()


@pytest.mark.integration
def test_optional_analyses_work_without_optional_tree_libraries(tmp_path, monkeypatch):
    monkeypatch.setattr(optional, "_HAS_CATBOOST", False)
    monkeypatch.setattr(optional, "_HAS_LIGHTGBM", False)
    pattern = generate_synthetic_data(tmp_path / "inputs", rows_per_domain=32)
    output = tmp_path / "results"
    args = demo_arguments(pattern, output)
    args.run_ablation = True
    args.run_extended_baselines = True
    args.run_calibration = True
    args.run_hybrid_stack = True
    args.ate_robust = True
    args.ate_adaptive = True
    args.ate_min_base = 8
    args.ate_step_base = 8
    args.run_subgroup_ate = True
    args.subgroup_ate_base_samples = 8
    args.crosstask_ate_path = str(output / "ate_ranking.csv")
    run_experiment(args)
    for name in (
        "ablation_summary.csv",
        "baseline_extended_summary.csv",
        "calibration_curve_ca_resmtl.csv",
        "hybrid_stacking_test.csv",
        "ate_ranking_robust.csv",
        "ate_threshold_sensitivity.csv",
    ):
        assert (output / name).is_file(), name
    ablation = pd.read_csv(output / "ablation_summary.csv")
    assert len(ablation) == 5
    assert np.isfinite(ablation["test_auc"]).all()
    metrics = json.loads((output / "pred_metrics.json").read_text(encoding="utf-8"))
    assert "hybrid_stack" in metrics and "calibration_ca_resmtl" in metrics
