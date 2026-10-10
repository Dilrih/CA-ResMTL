"""Discrimination, threshold, and rank-comparison metrics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def f2_score_at_sensitivity_floor(
    y_true: np.ndarray, probs: np.ndarray, sen_floor: float = 0.6
) -> Tuple[float, float]:
    best_thr, best_f2 = (0.5, -1.0)
    for thr in np.linspace(0.03, 0.8, 200):
        pred = (probs >= thr).astype(np.int64)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        sen = float(tp) / max(tp + fn, 1)
        if sen < sen_floor:
            continue
        prec = float(tp) / max(tp + fp, 1)
        denom = 4.0 * prec + sen
        f2 = 5.0 * prec * sen / max(denom, 1e-08)
        if f2 > best_f2:
            best_f2 = f2
            best_thr = float(thr)
    if best_f2 < 0:
        for thr in np.linspace(0.03, 0.8, 200):
            pred = (probs >= thr).astype(np.int64)
            if pred.sum() == 0:
                continue
            cm = confusion_matrix(y_true, pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sen = float(tp) / max(tp + fn, 1)
            prec = float(tp) / max(tp + fp, 1)
            denom = 4.0 * prec + sen
            f2 = 5.0 * prec * sen / max(denom, 1e-08)
            if f2 > best_f2:
                best_f2 = f2
                best_thr = float(thr)
    return (best_thr, max(best_f2, 0.0))


def province_loo_cv_score(
    model_factory,
    train_fn,
    X: np.ndarray,
    y: np.ndarray,
    prov_ids: np.ndarray,
    train_prov_ids: np.ndarray,
    device,
    n_folds: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[float, float, List[float]]:
    unique_prov = list(np.unique(train_prov_ids))
    n_folds_eff = (
        len(unique_prov) if n_folds is None or n_folds <= 0 else min(n_folds, len(unique_prov))
    )
    if n_folds_eff < len(unique_prov):
        rng = np.random.RandomState(42)
        unique_prov = sorted(rng.choice(unique_prov, size=n_folds_eff, replace=False).tolist())
    per_fold_aucs: List[float] = []
    for fold_prov in unique_prov:
        val_mask = prov_ids == int(fold_prov)
        train_mask = np.isin(prov_ids, [p for p in train_prov_ids if p != int(fold_prov)])
        idx_tr_fold = np.where(train_mask)[0]
        idx_val_fold = np.where(val_mask)[0]
        if len(idx_tr_fold) == 0 or len(idx_val_fold) == 0:
            continue
        if len(np.unique(y[idx_val_fold])) < 2:
            continue
        m = model_factory()
        _, fold_auc = train_fn(m, idx_tr_fold, idx_val_fold)
        per_fold_aucs.append(float(fold_auc))
        if verbose:
            print(f"[ProvinceCV] Fold {fold_prov}: val_AUC={fold_auc:.4f}", flush=True)
    if not per_fold_aucs:
        return (0.5, 0.0, [])
    mean_auc = float(np.mean(per_fold_aucs))
    std_auc = float(np.std(per_fold_aucs))
    if verbose:
        print(
            f"[ProvinceCV] Mean AUC={mean_auc:.4f} ± {std_auc:.4f} over {len(per_fold_aucs)} folds",
            flush=True,
        )
    return (mean_auc, std_auc, per_fold_aucs)


def delong_auc_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> Dict[str, Any]:
    from scipy import stats as scipy_stats

    y = y_true.astype(np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {
            "auc_a": float("nan"),
            "auc_b": float("nan"),
            "auc_diff": 0.0,
            "z_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
        }
    pos_mask = y == 1
    neg_mask = y == 0

    def _structural_components(preds: np.ndarray):
        pos_p = preds[pos_mask]
        neg_p = preds[neg_mask]
        v10 = np.zeros(n_pos)
        for i, p in enumerate(pos_p):
            v10[i] = float(np.mean(neg_p < p)) + 0.5 * float(np.mean(neg_p == p))
        v01 = np.zeros(n_neg)
        for i, p in enumerate(neg_p):
            v01[i] = float(np.mean(pos_p > p)) + 0.5 * float(np.mean(pos_p == p))
        theta = v10.mean()
        return (theta, v10, v01)

    theta_a, v10_a, v01_a = _structural_components(preds_a)
    theta_b, v10_b, v01_b = _structural_components(preds_b)
    var_a = np.var(v10_a, ddof=1) / n_pos + np.var(v01_a, ddof=1) / n_neg
    var_b = np.var(v10_b, ddof=1) / n_pos + np.var(v01_b, ddof=1) / n_neg
    cov_10 = np.cov(v10_a, v10_b, ddof=1)[0, 1] if n_pos > 1 else 0.0
    cov_01 = np.cov(v01_a, v01_b, ddof=1)[0, 1] if n_neg > 1 else 0.0
    cov_ab = cov_10 / n_pos + cov_01 / n_neg
    var_diff = float(var_a + var_b - 2.0 * cov_ab)
    if var_diff <= 1e-12:
        return {
            "auc_a": float(theta_a),
            "auc_b": float(theta_b),
            "auc_diff": float(theta_a - theta_b),
            "z_stat": 0.0,
            "p_value": 1.0,
            "significant": False,
        }
    z = (theta_a - theta_b) / math.sqrt(var_diff)
    p = float(2.0 * (1.0 - scipy_stats.norm.cdf(abs(z))))
    return {
        "auc_a": float(theta_a),
        "auc_b": float(theta_b),
        "auc_diff": float(theta_a - theta_b),
        "z_stat": float(z),
        "p_value": float(p),
        "significant": bool(p < 0.05),
        "interpretation": f"AUC_A={'%.4f' % theta_a} vs AUC_B={'%.4f' % theta_b}, ΔAUC={'%.4f' % (theta_a - theta_b)}, z={'%.3f' % z}, p={'%.4f' % p} ({('significant' if p < 0.05 else 'not significant')} at α=0.05)",
    }


def compute_binary_metrics_from_logits(
    y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y = np.asarray(y_true).reshape(-1)
    lg = np.asarray(logits)
    if lg.ndim > 1:
        lg = lg.reshape(-1)
    lg_clip = np.clip(lg, -30.0, 30.0)
    p = 1.0 / (1.0 + np.exp(-lg_clip))
    pred = (p >= float(threshold)).astype(int)
    out: Dict[str, float] = {}
    try:
        if len(np.unique(y)) < 2:
            auc = float("nan")
            ap = float("nan")
        else:
            auc = float(roc_auc_score(y, p))
            ap = float(average_precision_score(y, p))
    except Exception:
        auc = float("nan")
        ap = float("nan")
    try:
        acc = float(accuracy_score(y, pred))
    except Exception:
        acc = float("nan")
    try:
        f1 = float(f1_score(y, pred))
    except Exception:
        f1 = float("nan")
    try:
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel().tolist()
    except Exception:
        tn = int(np.sum((y == 0) & (pred == 0)))
        fp = int(np.sum((y == 0) & (pred == 1)))
        fn = int(np.sum((y == 1) & (pred == 0)))
        tp = int(np.sum((y == 1) & (pred == 1)))
    out.update(
        {
            "auc": auc,
            "ap": ap,
            "acc": acc,
            "f1": f1,
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
        }
    )
    return out


def choose_best_f1_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    thr_grid: Optional[np.ndarray] = None,
    sen_floor: float = 0.0,
) -> Tuple[float, float]:
    if thr_grid is None:
        thr_grid = np.linspace(0.03, 0.9, 176)
    best_thr, best_f1 = (0.5, -1.0)
    for thr in thr_grid:
        pred = (probs >= float(thr)).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sen = float(tp) / max(tp + fn, 1)
        if sen < float(sen_floor):
            continue
        f1v = float(f1_score(y_true, pred))
        if f1v > best_f1:
            best_f1 = f1v
            best_thr = float(thr)
    if best_f1 < 0:
        return (0.5, 0.0)
    return (best_thr, best_f1)


def compute_prob_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float = 0.5) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    probs = np.clip(np.asarray(probs, dtype=float), 1e-07, 1 - 1e-07)
    pred = (probs >= float(thr)).astype(int)
    auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")
    ap = (
        float(average_precision_score(y_true, probs))
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    brier = float(brier_score_loss(y_true, probs))
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    sens = float(tp) / max(tp + fn, 1)
    spec = float(tn) / max(tn + fp, 1)
    ece = 0.0
    n = len(y_true)
    for lo in np.linspace(0, 1, 11)[:-1]:
        hi = lo + 0.1
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() / max(n, 1) * abs(float(y_true[mask].mean()) - float(probs[mask].mean()))
    return {
        "auc": auc,
        "ap": ap,
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)) if 0 < pred.sum() < len(pred) else 0.0,
        "brier": brier,
        "ece": float(ece),
        "sensitivity": sens,
        "specificity": spec,
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "mean_prob": float(np.mean(probs)),
        "prevalence": float(np.mean(y_true)),
        "cm": cm.tolist(),
    }


def compute_crosstask_spearman(
    ate_df1: pd.DataFrame,
    ate_df2: pd.DataFrame,
    task1_name: str = "Task1",
    task2_name: str = "Task2",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    names1 = set(ate_df1["name"].astype(str).tolist())
    names2 = set(ate_df2["name"].astype(str).tolist())
    common = sorted(names1 & names2)
    if len(common) < 3:
        print(f"[Cross-task] 公共变量仅 {len(common)} 个，无法计算有意义的相关。", flush=True)
        return {
            "n_common": len(common),
            "rank_spearman_r": float("nan"),
            "rank_spearman_p": float("nan"),
            "ate_spearman_r": float("nan"),
            "ate_spearman_p": float("nan"),
        }
    rank1 = ate_df1.set_index("name")["rank"].astype(float)
    rank2 = ate_df2.set_index("name")["rank"].astype(float)
    ate1 = ate_df1.set_index("name")["ate_range"].astype(float)
    ate2 = ate_df2.set_index("name")["ate_range"].astype(float)
    r1 = np.array([float(rank1[n]) for n in common])
    r2 = np.array([float(rank2[n]) for n in common])
    a1 = np.array([float(ate1.get(n, 0.0)) for n in common])
    a2 = np.array([float(ate2.get(n, 0.0)) for n in common])
    sp_rank_r, sp_rank_p = spearmanr(r1, r2)
    kt_r, kt_p = kendalltau(r1, r2)
    sp_ate_r, sp_ate_p = spearmanr(a1, a2)
    result: Dict[str, Any] = {
        "task1": task1_name,
        "task2": task2_name,
        "n_common_variables": len(common),
        "rank_spearman_r": round(float(sp_rank_r), 4),
        "rank_spearman_p": round(float(sp_rank_p), 4),
        "rank_kendall_tau": round(float(kt_r), 4),
        "rank_kendall_p": round(float(kt_p), 4),
        "ate_range_spearman_r": round(float(sp_ate_r), 4),
        "ate_range_spearman_p": round(float(sp_ate_p), 4),
        "interpretation": "SIGNIFICANT cross-task structure similarity (p<0.05)"
        if float(sp_rank_p) < 0.05
        else "No significant cross-task rank correlation",
    }
    rows_joint = []
    for n in common:
        rows_joint.append(
            {
                "variable": n,
                f"rank_{task1_name}": float(rank1.get(n, float("nan"))),
                f"rank_{task2_name}": float(rank2.get(n, float("nan"))),
                f"ate_{task1_name}": float(ate1.get(n, float("nan"))),
                f"ate_{task2_name}": float(ate2.get(n, float("nan"))),
            }
        )
    df_joint = pd.DataFrame(rows_joint).sort_values(f"rank_{task1_name}", ascending=True)
    if output_dir:
        df_joint.to_csv(
            output_dir / "crosstask_ate_comparison.csv", index=False, encoding="utf-8-sig"
        )
        (output_dir / "crosstask_spearman.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[Cross-task] 保存至 {output_dir}/crosstask_spearman.json", flush=True)
    print(f"\n[Cross-task Spearman]  {task1_name} vs {task2_name}")
    print(f"  公共变量数={len(common)}")
    print(f"  Rank Spearman  r={sp_rank_r:.3f}  p={sp_rank_p:.4f}")
    print(f"  ATE Spearman   r={sp_ate_r:.3f}   p={sp_ate_p:.4f}")
    print(f"  Kendall tau    τ={kt_r:.3f}        p={kt_p:.4f}")
    print(f"  >> {result['interpretation']}")
    return {**result, "joint_table": df_joint}


def _kendall_tau_a(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n <= 1:
        return float("nan")
    conc = 0
    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s1 = a[i] - a[j]
            s2 = b[i] - b[j]
            prod = s1 * s2
            if prod > 0:
                conc += 1
            elif prod < 0:
                disc += 1
    denom = n * (n - 1) / 2.0
    return float((conc - disc) / denom) if denom > 0 else float("nan")


def summarize_threshold_sensitivity(
    per_thr_ranks: List[pd.DataFrame], topk: int = 10
) -> pd.DataFrame:
    if not per_thr_ranks:
        return pd.DataFrame()
    df = pd.concat(per_thr_ranks, axis=0, ignore_index=True)
    if "threshold" not in df.columns or "rank_at_thr" not in df.columns:
        return pd.DataFrame()
    thrs = sorted(df["threshold"].unique().tolist())
    rows = []
    for i in range(len(thrs)):
        for j in range(i + 1, len(thrs)):
            t1, t2 = (float(thrs[i]), float(thrs[j]))
            d1 = df[df["threshold"] == t1].copy()
            d2 = df[df["threshold"] == t2].copy()
            top1 = d1.sort_values("rank_at_thr").head(int(topk))["cause"].tolist()
            top2 = d2.sort_values("rank_at_thr").head(int(topk))["cause"].tolist()
            set1, set2 = (set(top1), set(top2))
            jac = float(len(set1 & set2) / max(1, len(set1 | set2)))
            inter = sorted(list(set1 & set2))
            sp = float("nan")
            kt = float("nan")
            if len(inter) >= 3:
                r1 = d1.set_index("cause").loc[inter]["rank_at_thr"].astype(float).to_numpy()
                r2 = d2.set_index("cause").loc[inter]["rank_at_thr"].astype(float).to_numpy()
                if np.std(r1) > 1e-12 and np.std(r2) > 1e-12:
                    sp = float(np.corrcoef(r1, r2)[0, 1])
                kt = _kendall_tau_a(r1.tolist(), r2.tolist())
            rows.append(
                {
                    "thr_a": t1,
                    "thr_b": t2,
                    "topk": int(topk),
                    "jaccard_topk": jac,
                    "n_intersection": int(len(inter)),
                    "spearman_rank_intersection": sp,
                    "kendall_tau_intersection": kt,
                }
            )
    return pd.DataFrame(rows)
