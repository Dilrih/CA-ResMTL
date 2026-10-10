import io
import json
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import pytest

from ca_resmtl.cli import parse_args
from ca_resmtl.demo import demo_arguments, generate_synthetic_data
from ca_resmtl.pipeline import run_experiment


@pytest.mark.integration
@pytest.mark.parametrize("backbone", ["deep_resmlp", "causal_attention", "ft_transformer"])
def test_full_pipeline_and_resumption(tmp_path, backbone):
    pattern = generate_synthetic_data(tmp_path / "inputs", rows_per_domain=32)
    output = tmp_path / "results"
    args = demo_arguments(pattern, output)
    args.backbone = backbone
    with io.TextIOWrapper(io.BytesIO(), encoding="gbk") as stream, redirect_stdout(stream):
        run_experiment(args)
    for name in (
        "docd_final.pth",
        "causal_adj_full.csv",
        "resmtl_best_model.pth",
        "pred_metrics.json",
        "temperature_calibration_meta.json",
        "ate_ranking.csv",
        "auxiliary_targets.npz",
        "prov_ids.npy",
    ):
        assert (output / name).is_file(), name
    metrics = json.loads((output / "pred_metrics.json").read_text(encoding="utf-8"))
    assert metrics["backbone"] == backbone
    assert metrics["n_aux_tasks"] == (2 if backbone == "deep_resmlp" else 0)
    assert "calibrated" in metrics and np.isfinite(metrics["test"]["auc"])
    assert np.isfinite(pd.read_csv(output / "ate_ranking.csv")["ate_range"]).all()
    original = (output / "run_args.json").read_bytes()
    original_matrix = np.load(output / "X_xy.npy").copy()
    run_experiment(
        parse_args(
            [
                "--stage2_only",
                "--output_dir",
                str(output),
                "--skip_stage3",
                "--epochs_pred",
                "1",
                "--device",
                "cpu",
                "--num_threads",
                "2",
            ]
        )
    )
    assert (output / "run_args.json").read_bytes() == original
    resumed = json.loads((output / "stage2_run_args.json").read_text(encoding="utf-8"))
    assert resumed["emb_dim"] == 16 and resumed["epochs_pred"] == 1
    resumed_metrics = json.loads((output / "pred_metrics.json").read_text(encoding="utf-8"))
    assert resumed_metrics["n_aux_tasks"] == (2 if backbone == "deep_resmlp" else 0)
    run_experiment(
        parse_args(
            ["--ate_only", "--output_dir", str(output), "--device", "cpu", "--num_threads", "2"]
        )
    )
    assert (output / "run_args.json").read_bytes() == original
    np.testing.assert_array_equal(np.load(output / "X_xy.npy"), original_matrix)
    with pytest.raises(FileExistsError):
        run_experiment(demo_arguments(pattern, output))
