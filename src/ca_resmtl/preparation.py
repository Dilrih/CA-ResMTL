"""Build local data artifacts and training-only structure priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from . import config
from .artifacts import save_run_manifest
from .auxiliary import build_auxiliary_targets
from .data import (
    apply_schema_preprocessing,
    build_splits,
    build_target_y,
    clean_dataframe,
    drop_group_leakage_location_columns,
    drop_sparse_or_constant_columns,
    encode_dataframe,
    indices_by_provinces,
    leakage_filter_columns,
    load_all_excels,
    load_variable_type_overrides,
    split_location_column,
)
from .graph import build_allowed_mask_with_priors, build_xy_for_docd, mutual_info_screening


def prepare_experiment(args, output_dir):
    OUTPUT_DIR = Path(output_dir)
    raw = load_all_excels(args.data_glob)
    raw = clean_dataframe(raw)
    loc_col = None
    for c in raw.columns:
        if "目前居住" in str(c):
            loc_col = c
            break
    if loc_col is not None:
        raw = split_location_column(raw, loc_col)
    y, y_name, y_meta = build_target_y(
        raw,
        args.target_mode,
        args.label_keywords,
        y_positive_value=args.y_positive_value,
        disable_auto_flip_y=args.disable_auto_flip_y,
    )
    y_label_col = y_meta.get("label_col", None)
    dep_cols = y_meta.get("dep_cols", None)
    valid_mask = np.asarray(y_meta.get("valid_mask", [True] * len(raw)), dtype=bool)
    if valid_mask.shape[0] != len(raw):
        raise RuntimeError("valid_mask 与原始样本数不一致，无法安全过滤缺失标签样本。")
    n_missing_label = int((~valid_mask).sum())
    if n_missing_label > 0:
        print(
            f"[LabelFilter] 检测到 {n_missing_label} 个标签缺失样本，训练/评估前将直接剔除，避免被错误编码为阴性。",
            flush=True,
        )
        raw = raw.loc[valid_mask].reset_index(drop=True)
        y = y[valid_mask].astype(np.int64)
        y_meta["label_filter_applied"] = True
    else:
        y = y.astype(np.int64)
        y_meta["label_filter_applied"] = False
    y_meta.pop("valid_mask", None)
    x_raw, leakage_report = leakage_filter_columns(
        raw=raw,
        target_mode=args.target_mode,
        y_label_col=y_label_col,
        dep_cols=dep_cols,
        aggressive_drop_post_outcome=not args.no_aggressive_post_outcome_drop,
        drop_symptom_items_for_suicide=not args.keep_symptom_items_for_suicide,
        strict_construct_filter=bool(getattr(args, "strict_construct_filter", False)),
        drop_state_proxy_items=bool(getattr(args, "drop_state_proxy_items", False)),
    )
    if bool(getattr(args, "strict_construct_filter", False)):
        print(
            f"[Info] 严格构念重叠过滤已启用，额外移除了 {len(leakage_report['dropped'].get('construct_overlap', []))} 个情绪自评类变量。",
            flush=True,
        )
    schema_overrides = load_variable_type_overrides(getattr(args, "var_schema_json", ""))
    x_raw, schema_prep_report = apply_schema_preprocessing(x_raw, schema_overrides)
    if not bool(getattr(args, "keep_location_question", False)):
        x_raw, location_drop_report = drop_group_leakage_location_columns(x_raw)
    else:
        location_drop_report = {"dropped": []}
    x_raw, sparse_report = drop_sparse_or_constant_columns(
        x_raw,
        min_nonmissing_per_feature=int(getattr(args, "min_nonmissing_per_feature", 15)),
        max_missing_rate_per_feature=float(getattr(args, "max_missing_rate_per_feature", 0.985)),
    )
    preprocessing_report = {
        "leakage_report": leakage_report,
        "schema_preprocessing": schema_prep_report,
        "location_drop": location_drop_report,
        "sparse_filter": sparse_report,
    }
    (OUTPUT_DIR / "leakage_drop_report.json").write_text(
        json.dumps(leakage_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "preprocessing_report.json").write_text(
        json.dumps(preprocessing_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if len(x_raw.columns.difference(["__province__", "__source_file__"])) == 0:
        raise ValueError("No usable features remain after preprocessing.")
    if len(np.unique(y)) != 2:
        raise ValueError("The outcome must contain both classes.")
    enc_x = encode_dataframe(
        x_raw,
        max_unique=args.max_unique,
        hash_buckets=args.hash_buckets,
        schema_overrides=schema_overrides,
    )
    schema = {
        "target_mode": args.target_mode,
        "y_name": y_name,
        "y_meta": y_meta,
        "n_samples": int(enc_x.X.shape[0]),
        "n_x_vars": int(enc_x.X.shape[1]),
        "provinces": sorted(list(set(enc_x.provinces.tolist()))),
        "x_vars": [
            {
                "name": vs.name,
                "n_cats": len(vs.values),
                "is_numeric": vs.is_numeric,
                "is_ordinal": vs.is_ordinal,
                "q_key": list(vs.q_key),
            }
            for vs in enc_x.var_specs
        ],
    }
    (OUTPUT_DIR / "data_schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    splits = build_splits(
        enc_x.provinces,
        seed=args.seed,
        train_k=int(getattr(args, "train_k", -1)),
        val_provinces=int(getattr(args, "val_provinces", 1)),
        test_provinces=int(getattr(args, "test_provinces", 1)),
    )
    (OUTPUT_DIR / "splits.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    idx_train = indices_by_provinces(enc_x.provinces, splits["train"])
    idx_val = indices_by_provinces(enc_x.provinces, splits["val"])
    idx_test = indices_by_provinces(enc_x.provinces, splits["test"])
    for part, index in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        if len(index) == 0 or len(np.unique(y[index])) != 2:
            raise ValueError(f"{part} must contain both outcome classes; review the domain split.")
    print("Split provinces:", splits)
    print(f"Train/Val/Test sizes: {len(idx_train)}/{len(idx_val)}/{len(idx_test)}")
    print(
        f"Positive rate: train={y[idx_train].mean():.4f} val={y[idx_val].mean():.4f} test={y[idx_test].mean():.4f}"
    )
    save_run_manifest(
        OUTPUT_DIR,
        args,
        split_info={
            "splits": splits,
            "n_train": int(len(idx_train)),
            "n_val": int(len(idx_val)),
            "n_test": int(len(idx_test)),
            "train_positive_rate": float(y[idx_train].mean()),
            "val_positive_rate": float(y[idx_val].mean()),
            "test_positive_rate": float(y[idx_test].mean()),
        },
        extra={"y_meta": y_meta, "schema_rule_count": int(len(schema_overrides))},
    )
    X_xy, var_specs_xy, names_xy, y_index = build_xy_for_docd(enc_x, y=y, y_name=y_name)
    np.save(OUTPUT_DIR / "X_xy.npy", X_xy.astype(np.int64))
    idx_pool = np.concatenate([idx_train, idx_val], axis=0)
    np.save(OUTPUT_DIR / "idx_pool.npy", idx_pool.astype(np.int64))
    domains = sorted(set(enc_x.provinces.tolist()))
    domain_index = {name: index for index, name in enumerate(domains)}
    np.save(
        OUTPUT_DIR / "prov_ids.npy",
        np.array([domain_index[name] for name in enc_x.provinces], dtype=np.int64),
    )
    auxiliary = build_auxiliary_targets(raw, args.target_mode) if args.use_multitask else None
    if auxiliary is not None:
        np.savez(
            OUTPUT_DIR / "auxiliary_targets.npz",
            **{f"task_{index}": values for index, values in enumerate(auxiliary)},
        )
    np.save(OUTPUT_DIR / "idx_train.npy", idx_train.astype(np.int64))
    np.save(OUTPUT_DIR / "idx_val.npy", idx_val.astype(np.int64))
    np.save(OUTPUT_DIR / "idx_test.npy", idx_test.astype(np.int64))

    def _vs_to_dict(vs):
        return {
            "name": vs.name,
            "values": list(vs.values),
            "is_numeric": bool(vs.is_numeric),
            "is_ordinal": bool(vs.is_ordinal),
            "q_key": list(vs.q_key),
        }

    (OUTPUT_DIR / "var_specs_xy.json").write_text(
        json.dumps([_vs_to_dict(v) for v in var_specs_xy], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    allowed_mask = build_allowed_mask_with_priors(
        var_specs_xy, y_index=y_index, use_question_order_prior=True
    )
    mi_df = None
    mi_map: Dict[str, float] = {}
    if not args.disable_mi_screening:
        mi_df = mutual_info_screening(
            enc_x.X, y, enc_x.col_names, idx_train, topk=args.y_cand_topk, min_mi=args.y_cand_minmi
        )
        mi_df.to_csv(OUTPUT_DIR / "y_candidate_screening.csv", index=False, encoding="utf-8-sig")
        selected_vars = set(mi_df[mi_df["selected"]]["var"].tolist())
        for i, name in enumerate(names_xy[:-1]):
            if name not in selected_vars:
                allowed_mask[i, y_index] = 0.0
        if allowed_mask[:, y_index].sum() < 10:
            top30 = mi_df.head(30)["var"].tolist()
            for i, name in enumerate(names_xy[:-1]):
                if name in top30:
                    allowed_mask[i, y_index] = 1.0
        for _, r in mi_df.iterrows():
            mi_map[str(r["var"])] = float(r["mi"])

    def _is_symptom_like(nm: str) -> bool:
        nm = str(nm)
        if any((nm.startswith(pfx) for pfx in config.SYMPTOM_ITEM_PREFIXES)):
            return True
        if any((k in nm for k in config.POST_OUTCOME_KEYWORDS)):
            return True
        return False

    y_in_prior = np.zeros((len(names_xy),), dtype=np.float32)
    for i, nm in enumerate(names_xy):
        if i == y_index:
            continue
        if allowed_mask[i, y_index] <= 0.0:
            continue
        w = float(mi_map.get(str(nm), 0.0))
        w = max(0.0, w)
        if _is_symptom_like(nm):
            w *= 0.35
        y_in_prior[i] = float(w)
    y_in_prior = np.log1p(y_in_prior).astype(np.float32)
    s = float(y_in_prior.sum())
    if s > 0:
        y_in_prior = (y_in_prior / s).astype(np.float32)
    else:
        y_in_prior = None
    list(range(X_xy.shape[1]))
    return (
        raw,
        enc_x,
        X_xy,
        var_specs_xy,
        names_xy,
        y_index,
        idx_train,
        idx_val,
        idx_test,
        allowed_mask,
        y_in_prior,
    )
