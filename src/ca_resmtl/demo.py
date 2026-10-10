"""A small reproducible example using entirely artificial responses."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .cli import parse_args


def generate_synthetic_data(directory, seed=42, rows_per_domain=64):
    """Create six fictitious domains without reading any study data."""
    if rows_per_domain < 32:
        raise ValueError("Use at least 32 rows per domain for the synthetic example.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for domain in range(6):
        path = directory / f"synthetic_domain_{domain + 1:02d}.xlsx"
        if path.exists():
            raise FileExistsError(f"Synthetic workbook already exists: {path.name}")
        n = rows_per_domain
        sleep = rng.integers(1, 6, n)
        support = rng.integers(1, 6, n)
        stress = rng.integers(1, 6, n)
        exercise = rng.integers(1, 5, n)
        logits = -1.8 + 0.30 * stress + 0.20 * sleep - 0.22 * support
        outcome = (rng.random(n) < 1 / (1 + np.exp(-logits))).astype(float)
        outcome[:8] = 0
        outcome[8:16] = 1
        outcome[-1] = np.nan
        frame = pd.DataFrame(
            {
                "1、性别": rng.integers(1, 3, n),
                "2、学段": rng.integers(1, 4, n),
                "6、户籍类型": rng.integers(1, 3, n),
                "8、留守状态": rng.integers(1, 3, n),
                "10、睡眠状况": sleep,
                "11、体育活动": exercise,
                "12、家庭支持": support,
                "13、学业压力": stress,
                "14、生活安排": rng.integers(1, 5, n),
                "15、同伴支持": rng.integers(1, 5, n),
                "16、学习环境": rng.integers(1, 4, n),
                "17、日常活动": rng.integers(1, 4, n),
                "35、很认真地考虑过自杀": outcome,
                "39、多久自我伤害一次": 1 + (rng.random(n) < 0.16).astype(int),
            }
        )
        frame.loc[frame.index[::13], "10、睡眠状况"] = -3
        for item in range(8):
            frame[f"62、抑郁条目（{item + 1}）"] = rng.integers(1, 5, n)
        frame.to_excel(path, index=False)
    return str(directory / "*.xlsx")


def demo_arguments(data_glob, output_dir):
    """Configure a short CPU run for the synthetic example."""
    return parse_args(
        [
            "--data_glob",
            str(data_glob),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--num_threads",
            "2",
            "--seed",
            "42",
            "--emb_dim",
            "16",
            "--heads",
            "4",
            "--layers",
            "1",
            "--resmlp_blocks",
            "2",
            "--batch_size",
            "64",
            "--epochs_pred",
            "3",
            "--docd_inner",
            "3",
            "--docd_outer",
            "2",
            "--min_outer_loops",
            "1",
            "--bootstrap_struct",
            "0",
            "--ate_bootstrap_reps",
            "2",
            "--ate_bootstrap_topk",
            "2",
            "--ate_base_samples",
            "16",
            "--ate_mc_samples",
            "2",
            "--ate_chunk",
            "16",
            "--ate_topk",
            "3",
            "--y_parent_min_k",
            "4",
            "--y_parent_max_k",
            "12",
            "--graph_target_degree",
            "2",
            "--causal_topk",
            "8",
            "--n_interaction_pairs",
            "6",
            "--disable_auto_flip_y",
            "--y_positive_value",
            "1",
        ]
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/demo")
    args = parser.parse_args(argv)
    directory = Path(args.output_dir)
    if directory.exists() and any(directory.iterdir()):
        parser.error("Choose an empty output directory for the demo.")
    from .pipeline import run_experiment

    pattern = generate_synthetic_data(directory / "synthetic_data")
    run_experiment(demo_arguments(pattern, directory))


if __name__ == "__main__":
    main()
