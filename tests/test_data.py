import numpy as np
import pandas as pd

from ca_resmtl import config
from ca_resmtl.auxiliary import build_auxiliary_targets
from ca_resmtl.data import (
    build_splits,
    build_target_y,
    clean_dataframe,
    encode_dataframe,
    indices_by_provinces,
    leakage_filter_columns,
)


def test_skip_codes_become_missing_categories():
    raw = pd.DataFrame({"10、睡眠": [1, 2, -3, "-99", " "], "__province__": ["a"] * 5})
    encoded = encode_dataframe(clean_dataframe(raw), max_unique=40, hash_buckets=64)
    np.testing.assert_array_equal(encoded.X[:, 0], [1, 2, 0, 0, 0])
    assert encoded.var_specs[0].values[0] == "<phi>"
    assert len(encoded.var_specs[0].values) == 3


def test_missing_primary_labels_are_not_negative_controls():
    raw = pd.DataFrame({"35、自杀意念": [1, 2, np.nan, "是", "否"]})
    y, _, meta = build_target_y(
        raw, config.TARGET_SUICIDE, ["自杀意念"], y_positive_value=1, disable_auto_flip_y=True
    )
    np.testing.assert_array_equal(y, [1, 0, -1, 1, 0])
    assert meta["valid_mask"] == [True, True, False, True, True]


def test_outcome_family_and_metadata_do_not_enter_features():
    raw = pd.DataFrame(
        {
            "35、自杀意念": [0, 1],
            "35、相关结果": [1, 1],
            "39、自我伤害": [1, 2],
            "来自IP": ["a", "b"],
            "10、睡眠": [1, 2],
            "__province__": ["a", "b"],
        }
    )
    features, _ = leakage_filter_columns(raw, config.TARGET_SUICIDE, "35、自杀意念", None)
    assert "10、睡眠" in features
    assert not any(name.startswith(("35、", "39、")) or name == "来自IP" for name in features)


def test_domain_splits_are_disjoint_and_deterministic():
    domains = np.repeat([f"domain_{i}" for i in range(6)], 12)
    split = build_splits(domains, seed=42, val_provinces=2, test_provinces=2)
    assert split == build_splits(domains, seed=42, val_provinces=2, test_provinces=2)
    groups = [set(indices_by_provinces(domains, split[part])) for part in ("train", "val", "test")]
    assert (
        groups[0].isdisjoint(groups[1])
        and groups[0].isdisjoint(groups[2])
        and groups[1].isdisjoint(groups[2])
    )
    assert len(set.union(*groups)) == len(domains)


def test_missing_auxiliary_labels_are_masked():
    raw = pd.DataFrame({"39、自我伤害": [1, 2, np.nan, 1]})
    target = build_auxiliary_targets(raw, config.TARGET_SUICIDE)[0]
    np.testing.assert_array_equal(target[[0, 1, 3]], [0, 1, 0])
    assert np.isnan(target[2])
