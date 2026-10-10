"""Reference classifiers and optional stacking."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import config, optional
from .calibration import BetaCalibrator
from .metrics import choose_best_f1_threshold, compute_prob_metrics


def baseline_hgbdt(
    X: np.ndarray, y: np.ndarray, idx_train: np.ndarray, idx_val: np.ndarray, idx_test: np.ndarray
) -> Dict[str, Any]:
    clf = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=350, random_state=0
    )
    clf.fit(X[idx_train], y[idx_train])
    p_val = clf.predict_proba(X[idx_val])[:, 1]
    p_test = clf.predict_proba(X[idx_test])[:, 1]
    thr_opt, _ = choose_best_f1_threshold(
        y[idx_val], p_val, sen_floor=float(config.DEFAULT_MIN_SENSITIVITY_THR)
    )
    if _ <= 0.0:
        thr_opt, _ = choose_best_f1_threshold(y[idx_val], p_val, sen_floor=0.0)
    return {
        "val": compute_prob_metrics(y[idx_val], p_val, thr=thr_opt),
        "test": compute_prob_metrics(y[idx_test], p_test, thr=thr_opt),
        "val_thr05": compute_prob_metrics(y[idx_val], p_val, thr=0.5),
        "test_thr05": compute_prob_metrics(y[idx_test], p_test, thr=0.5),
        "threshold_selected_on_val": float(thr_opt),
    }


def _eval_sklearn_clf(
    clf,
    X_te: np.ndarray,
    y_te: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    p_te = clf.predict_proba(X_te)[:, 1]
    out = {"test": compute_prob_metrics(y_te, p_te, thr=0.5)}
    if X_val is not None and y_val is not None:
        p_val = clf.predict_proba(X_val)[:, 1]
        thr_opt, _ = choose_best_f1_threshold(
            y_val, p_val, sen_floor=float(config.DEFAULT_MIN_SENSITIVITY_THR)
        )
        if _ <= 0.0:
            thr_opt, _ = choose_best_f1_threshold(y_val, p_val, sen_floor=0.0)
        out["val"] = compute_prob_metrics(y_val, p_val, thr=thr_opt)
        out["val_thr05"] = compute_prob_metrics(y_val, p_val, thr=0.5)
        out["val_opt"] = compute_prob_metrics(y_val, p_val, thr=thr_opt)
        out["test_opt"] = compute_prob_metrics(y_te, p_te, thr=thr_opt)
        out["threshold_selected_on_val"] = float(thr_opt)
    return out


def _fit_lgbm_safely(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    if not optional._HAS_LIGHTGBM:
        return None
    clf = optional.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_samples=20,
        class_weight="balanced",
        random_state=42,
    )
    try:
        clf.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[optional.lgb.early_stopping(60, verbose=False)],
        )
    except Exception:
        clf.fit(X_tr, y_tr)
    return clf


def baseline_extended(
    X: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    var_names: List[str],
    output_dir: Optional[Path] = None,
    prov_ids: Optional[np.ndarray] = None,
    focus_idx: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    X_tr, y_tr = (X[idx_train].astype(float), y[idx_train])
    X_val_arr, y_val_arr = (X[idx_val].astype(float), y[idx_val])
    X_te, y_te = (X[idx_test].astype(float), y[idx_test])
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val_arr)
    X_te_s = scaler.transform(X_te)
    results: Dict[str, Any] = {}
    print("[Baseline-Extended] Training Logistic Regression (L2)...", flush=True)
    lr = LogisticRegression(
        C=1.0, max_iter=2000, class_weight="balanced", solver="saga", random_state=42, n_jobs=-1
    )
    lr.fit(X_tr_s, y_tr)
    lr_res = _eval_sklearn_clf(lr, X_te_s, y_te, X_val_s, y_val_arr)
    coef = np.abs(lr.coef_[0])
    top_idx = np.argsort(-coef)[:30]
    lr_res["top_features"] = [
        {
            "rank": i + 1,
            "var": var_names[j] if j < len(var_names) else f"feat_{j}",
            "coef_abs": float(coef[j]),
        }
        for i, j in enumerate(top_idx)
    ]
    lr_res["val_probs"] = lr.predict_proba(X_val_s)[:, 1].tolist()
    lr_res["test_probs"] = lr.predict_proba(X_te_s)[:, 1].tolist()
    results["logistic_regression"] = lr_res
    print("[Baseline-Extended] Training LASSO-LR (L1, CV)...", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 12),
            cv=5,
            penalty="l1",
            solver="saga",
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )
        lasso.fit(X_tr_s, y_tr)
    lasso_res = _eval_sklearn_clf(lasso, X_te_s, y_te, X_val_s, y_val_arr)
    coef_l = np.abs(lasso.coef_[0])
    top_l = np.argsort(-coef_l)[:30]
    lasso_res["n_nonzero_features"] = int((coef_l > 1e-06).sum())
    lasso_res["best_C"] = float(lasso.C_[0])
    lasso_res["top_features"] = [
        {
            "rank": i + 1,
            "var": var_names[j] if j < len(var_names) else f"feat_{j}",
            "coef_abs": float(coef_l[j]),
        }
        for i, j in enumerate(top_l)
        if coef_l[j] > 1e-06
    ]
    lasso_res["val_probs"] = lasso.predict_proba(X_val_s)[:, 1].tolist()
    lasso_res["test_probs"] = lasso.predict_proba(X_te_s)[:, 1].tolist()
    results["lasso_lr"] = lasso_res
    print("[Baseline-Extended] Training Random Forest (n=300)...", flush=True)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X[idx_train], y_tr)
    rf_res = _eval_sklearn_clf(rf, X[idx_test], y_te, X[idx_val], y_val_arr)
    imp = rf.feature_importances_
    top_rf = np.argsort(-imp)[:30]
    rf_res["top_features"] = [
        {
            "rank": i + 1,
            "var": var_names[j] if j < len(var_names) else f"feat_{j}",
            "importance": float(imp[j]),
        }
        for i, j in enumerate(top_rf)
    ]
    rf_res["val_probs"] = rf.predict_proba(X[idx_val])[:, 1].tolist()
    rf_res["test_probs"] = rf.predict_proba(X[idx_test])[:, 1].tolist()
    results["random_forest"] = rf_res
    print("[Baseline-Extended] Training HistGBDT...", flush=True)
    hgbdt = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=350, random_state=0
    )
    hgbdt.fit(X_tr, y_tr)
    hgbdt_res = _eval_sklearn_clf(hgbdt, X_te, y_te, X_val_arr, y_val_arr)
    hgbdt_res["val_probs"] = hgbdt.predict_proba(X_val_arr)[:, 1].tolist()
    hgbdt_res["test_probs"] = hgbdt.predict_proba(X_te)[:, 1].tolist()
    results["histgbdt"] = hgbdt_res
    if optional._HAS_CATBOOST:
        try:
            print("[Baseline-Extended] Training CatBoost...", flush=True)
            cat = optional.CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.04,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                verbose=False,
                auto_class_weights="Balanced",
                allow_writing_files=False,
            )
            cat.fit(X_tr, y_tr, eval_set=(X_val_arr, y_val_arr), use_best_model=True, verbose=False)
            cat_res = _eval_sklearn_clf(cat, X_te, y_te, X_val_arr, y_val_arr)
            try:
                cat_imp = cat.get_feature_importance()
                top_cat = np.argsort(-cat_imp)[:30]
                cat_res["top_features"] = [
                    {
                        "rank": i + 1,
                        "var": var_names[j] if j < len(var_names) else f"feat_{j}",
                        "importance": float(cat_imp[j]),
                    }
                    for i, j in enumerate(top_cat)
                ]
            except Exception:
                cat_res["top_features"] = []
            cat_res["val_probs"] = cat.predict_proba(X_val_arr)[:, 1].tolist()
            cat_res["test_probs"] = cat.predict_proba(X_te)[:, 1].tolist()
            results["catboost"] = cat_res
        except Exception as e:
            print(f"[Baseline-Extended] CatBoost skipped: {e}", flush=True)
    if optional._HAS_LIGHTGBM:
        try:
            print("[Baseline-Extended] Training LightGBM...", flush=True)
            lgbm = _fit_lgbm_safely(X_tr, y_tr, X_val_arr, y_val_arr)
            if lgbm is not None:
                lgbm_res = _eval_sklearn_clf(lgbm, X_te, y_te, X_val_arr, y_val_arr)
                try:
                    lgb_imp = lgbm.feature_importances_
                    top_lgb = np.argsort(-lgb_imp)[:30]
                    lgbm_res["top_features"] = [
                        {
                            "rank": i + 1,
                            "var": var_names[j] if j < len(var_names) else f"feat_{j}",
                            "importance": float(lgb_imp[j]),
                        }
                        for i, j in enumerate(top_lgb)
                    ]
                except Exception:
                    lgbm_res["top_features"] = []
                lgbm_res["val_probs"] = lgbm.predict_proba(X_val_arr)[:, 1].tolist()
                lgbm_res["test_probs"] = lgbm.predict_proba(X_te)[:, 1].tolist()
                results["lightgbm"] = lgbm_res
        except Exception as e:
            print(f"[Baseline-Extended] LightGBM skipped: {e}", flush=True)
    summary_rows = []
    for name, res in results.items():
        t = res["test"]
        t_opt = res.get("test_opt", t)
        row = {
            "model": name,
            "test_auc": round(float(t["auc"]), 4),
            "test_ap": round(float(t["ap"]), 4),
            "test_f1": round(float(t["f1"]), 4),
            "test_f1_opt": round(float(t_opt.get("f1", float("nan"))), 4),
            "test_acc": round(float(t["acc"]), 4),
            "test_sensitivity": round(float(t.get("sensitivity", float("nan"))), 4),
            "test_specificity": round(float(t.get("specificity", float("nan"))), 4),
            "threshold_selected_on_val": round(float(res.get("threshold_selected_on_val", 0.5)), 4),
        }
        cm = t["cm"]
        row["test_fn"] = int(cm[1][0])
        row["test_fp"] = int(cm[0][1])
        summary_rows.append(row)
    df_sum = pd.DataFrame(summary_rows)
    if output_dir:
        df_sum.to_csv(
            output_dir / "baseline_extended_summary.csv", index=False, encoding="utf-8-sig"
        )
        for mname, mres in results.items():
            feats = mres.get("top_features", [])
            if feats:
                pd.DataFrame(feats).to_csv(
                    output_dir / f"baseline_{mname}_top_features.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
        ser_results = {}
        for k, v in results.items():
            sv = {
                kk: vv
                for kk, vv in v.items()
                if kk not in {"top_features", "val_probs", "test_probs"}
            }
            sv["n_top_features"] = len(v.get("top_features", []))
            ser_results[k] = sv
        (output_dir / "baseline_extended_full.json").write_text(
            json.dumps(ser_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("\n[Baseline-Extended] Summary:")
    print(df_sum.to_string(index=False))
    results["_summary"] = df_sum
    return results


def run_hybrid_stacking(
    X: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    predictor_probs_val: Dict[str, np.ndarray],
    predictor_probs_test: Dict[str, np.ndarray],
    output_dir: Optional[Path] = None,
    predictor_probs_train: Optional[Dict[str, np.ndarray]] = None,
    prov_ids: Optional[np.ndarray] = None,
    X_eng: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit a stacker with tree OOF features and prefitted neural predictions."""
    X_tr_raw = X[idx_train].astype(float)
    y_tr = y[idx_train]
    X_val_raw = X[idx_val].astype(float)
    y_val = y[idx_val]
    X_te_raw = X[idx_test].astype(float)
    y_te = y[idx_test]
    if X_eng is not None:
        X_tr_tree = np.hstack([X_tr_raw, X_eng[idx_train].astype(float)])
        X_val_tree = np.hstack([X_val_raw, X_eng[idx_val].astype(float)])
        X_te_tree = np.hstack([X_te_raw, X_eng[idx_test].astype(float)])
        print(
            f"[Stacking] Using augmented features: {X_tr_raw.shape[1]} + {X_eng.shape[1]} = {X_tr_tree.shape[1]}",
            flush=True,
        )
    else:
        X_tr_tree, X_val_tree, X_te_tree = (X_tr_raw, X_val_raw, X_te_raw)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_tree)
    X_val_s = scaler.transform(X_val_tree)
    X_te_s = scaler.transform(X_te_tree)
    base_feats_val: Dict[str, np.ndarray] = {"resmtl_score": predictor_probs_val["score"]}
    base_feats_test: Dict[str, np.ndarray] = {"resmtl_score": predictor_probs_test["score"]}
    if "prob" in predictor_probs_val and "prob" in predictor_probs_test:
        base_feats_val["resmtl_prob"] = predictor_probs_val["prob"]
        base_feats_test["resmtl_prob"] = predictor_probs_test["prob"]
    if "fused" in predictor_probs_val and "fused" in predictor_probs_test:
        base_feats_val["resmtl_fused"] = predictor_probs_val["fused"]
        base_feats_test["resmtl_fused"] = predictor_probs_test["fused"]
    lr_m = LogisticRegression(
        C=1.0, max_iter=2000, class_weight=None, solver="saga", random_state=42, n_jobs=-1
    )
    lr_m.fit(X_tr_s, y_tr)
    base_feats_val["lr"] = lr_m.predict_proba(X_val_s)[:, 1]
    base_feats_test["lr"] = lr_m.predict_proba(X_te_s)[:, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lasso = LogisticRegressionCV(
            Cs=np.logspace(-3, 1, 12),
            cv=5,
            penalty="l1",
            solver="saga",
            class_weight=None,
            max_iter=2000,
            random_state=42,
            n_jobs=-1,
        )
        lasso.fit(X_tr_s, y_tr)
    base_feats_val["lasso"] = lasso.predict_proba(X_val_s)[:, 1]
    base_feats_test["lasso"] = lasso.predict_proba(X_te_s)[:, 1]
    rf = RandomForestClassifier(
        n_estimators=360,
        max_depth=None,
        class_weight="balanced_subsample",
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tr_tree, y_tr)
    base_feats_val["rf"] = rf.predict_proba(X_val_tree)[:, 1]
    base_feats_test["rf"] = rf.predict_proba(X_te_tree)[:, 1]
    hgbdt = HistGradientBoostingClassifier(
        max_depth=6, learning_rate=0.08, max_iter=380, random_state=0
    )
    hgbdt.fit(X_tr_tree, y_tr)
    base_feats_val["hgbdt"] = hgbdt.predict_proba(X_val_tree)[:, 1]
    base_feats_test["hgbdt"] = hgbdt.predict_proba(X_te_tree)[:, 1]
    if optional._HAS_CATBOOST:
        try:
            cat = optional.CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.04,
                loss_function="Logloss",
                eval_metric="AUC",
                random_seed=42,
                verbose=False,
                auto_class_weights="Balanced",
                allow_writing_files=False,
            )
            cat.fit(
                X_tr_tree, y_tr, eval_set=(X_val_tree, y_val), use_best_model=True, verbose=False
            )
            base_feats_val["catboost"] = cat.predict_proba(X_val_tree)[:, 1]
            base_feats_test["catboost"] = cat.predict_proba(X_te_tree)[:, 1]
        except Exception:
            pass
    if optional._HAS_LIGHTGBM:
        try:
            lgbm = _fit_lgbm_safely(X_tr_tree, y_tr, X_val_tree, y_val)
            if lgbm is not None:
                base_feats_val["lightgbm"] = lgbm.predict_proba(X_val_tree)[:, 1]
                base_feats_test["lightgbm"] = lgbm.predict_proba(X_te_tree)[:, 1]
        except Exception:
            pass
    model_names = [
        m for m in config.DEFAULT_STACK_USE_MODELS if m in base_feats_val and m in base_feats_test
    ]
    meta_X_val = np.column_stack([base_feats_val[m] for m in model_names])
    meta_X_test = np.column_stack([base_feats_test[m] for m in model_names])
    meta_scaler = StandardScaler()
    meta_X_val_s = meta_scaler.fit_transform(meta_X_val)
    meta_X_test_s = meta_scaler.transform(meta_X_test)
    meta = LogisticRegression(
        C=float(config.DEFAULT_STACK_META_C),
        max_iter=4000,
        class_weight=None,
        solver="lbfgs",
        random_state=42,
    )
    meta.fit(meta_X_val_s, y_val)
    p_val_raw = meta.predict_proba(meta_X_val_s)[:, 1]
    p_test_raw = meta.predict_proba(meta_X_test_s)[:, 1]
    p_val, p_test = (p_val_raw, p_test_raw)
    try:
        beta_meta = BetaCalibrator()
        beta_meta.fit(p_val_raw, y_val.astype(float), verbose=True)
        p_val_beta = beta_meta.calibrate(p_val_raw)
        p_test_beta = beta_meta.calibrate(p_test_raw)
        if brier_score_loss(y_val, p_val_beta) <= brier_score_loss(y_val, p_val):
            p_val, p_test = (p_val_beta, p_test_beta)
    except Exception:
        pass
    iso_meta = IsotonicRegression(out_of_bounds="clip")
    try:
        iso_meta.fit(p_val, y_val.astype(float))
        p_val_iso = np.clip(iso_meta.predict(p_val), 1e-06, 1 - 1e-06)
        p_test_iso = np.clip(iso_meta.predict(p_test), 1e-06, 1 - 1e-06)
        brier_cur = float(brier_score_loss(y_val, p_val))
        brier_iso = float(brier_score_loss(y_val, p_val_iso))
        auc_cur = float(roc_auc_score(y_val, p_val)) if len(np.unique(y_val)) > 1 else 0.5
        auc_iso = float(roc_auc_score(y_val, p_val_iso)) if len(np.unique(y_val)) > 1 else 0.5
        if brier_iso < brier_cur and auc_cur - auc_iso <= 0.003:
            p_val, p_test = (p_val_iso, p_test_iso)
            print(
                f"[Stacking] Isotonic better: Brier {brier_cur:.4f} -> {brier_iso:.4f}", flush=True
            )
    except Exception:
        pass
    thr_opt, _best = choose_best_f1_threshold(y_val, p_val, sen_floor=0.5)
    if _best <= 0.0:
        thr_opt, _best = choose_best_f1_threshold(y_val, p_val, sen_floor=0.0)
    res = {
        "models": model_names,
        "val": compute_prob_metrics(y_val, p_val, thr=0.5),
        "test": compute_prob_metrics(y_te, p_test, thr=0.5),
        "val_opt": compute_prob_metrics(y_val, p_val, thr=thr_opt),
        "test_opt": compute_prob_metrics(y_te, p_test, thr=thr_opt),
        "threshold_selected_on_val": float(thr_opt),
        "meta_coefficients": {
            name: float(coef) for name, coef in zip(model_names, meta.coef_[0].tolist())
        },
    }
    if output_dir:
        pd.DataFrame(
            {
                "y_true": y_te,
                **{f"prob_{m}": base_feats_test[m] for m in model_names},
                "prob_stack_raw": p_test_raw,
                "prob_stack": p_test,
            }
        ).to_csv(output_dir / "hybrid_stacking_test.csv", index=False, encoding="utf-8-sig")
        (output_dir / "hybrid_stacking_meta.json").write_text(
            json.dumps(
                {
                    "models": model_names,
                    "threshold_selected_on_val": float(thr_opt),
                    "meta_coefficients": res["meta_coefficients"],
                    "calibration_protocol": "class_weight=None + affine+isotonic post-calibration",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return res


def _get_hgbdt_test_probs(
    X: np.ndarray, y: np.ndarray, idx_train: np.ndarray, idx_test: np.ndarray
) -> Optional[np.ndarray]:
    try:
        clf = HistGradientBoostingClassifier(
            max_depth=6, learning_rate=0.08, max_iter=350, random_state=0
        )
        clf.fit(X[idx_train], y[idx_train])
        return clf.predict_proba(X[idx_test])[:, 1]
    except Exception as e:
        print(f"[Calibration] HistGBDT重跑失败: {e}，跳过HistGBDT校准。", flush=True)
        return None
