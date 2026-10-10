"""Validation-only probability calibration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .data import split_val_for_nested_calibration
from .metrics import choose_best_f1_threshold, compute_prob_metrics
from .models.predictor import DualHeadPredictor


class BetaCalibrator:
    """Fit a monotone beta calibration map."""

    def __init__(self):
        self.a: float = 1.0
        self.b: float = 1.0
        self.c: float = 0.0
        self._fitted: bool = False

    def fit(
        self, probs: np.ndarray, labels: np.ndarray, verbose: bool = True
    ) -> Tuple[float, float, float]:
        from scipy.optimize import minimize
        from scipy.special import expit as sp_expit

        p = np.clip(probs.astype(np.float64), 1e-07, 1 - 1e-07)
        y = labels.astype(np.float64)
        auc_before = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else 0.5

        def nll(params):
            a_, b_, c_ = params
            logit_cal = a_ * np.log(p) - b_ * np.log(1 - p) + c_
            p_cal = sp_expit(logit_cal)
            p_cal = np.clip(p_cal, 1e-12, 1 - 1e-12)
            return -np.mean(y * np.log(p_cal) + (1 - y) * np.log(1 - p_cal))

        x0 = np.array([1.0, 1.0, 0.0])
        try:
            res = minimize(
                nll,
                x0,
                method="L-BFGS-B",
                bounds=[(0.0, None), (0.0, None), (None, None)],
                options={"maxiter": 5000, "ftol": 1e-08},
            )
            a, b, c = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
        except Exception:
            a, b, c = (1.0, 1.0, float(x0[2]))
        if not np.all(np.isfinite([a, b, c])) or a < 0 or b < 0:
            a, b, c = (1.0, 1.0, float(x0[2]))
        self.a, self.b, self.c = (float(a), float(b), float(c))
        self._fitted = True
        p_cal = self.calibrate(probs)
        try:
            auc_after = float(roc_auc_score(y, p_cal))
        except Exception:
            auc_after = auc_before
        if auc_before - auc_after > 0.003:
            if verbose:
                print(
                    f"[BetaCalibrator] Warning: AUC drop {auc_before:.4f}->{auc_after:.4f}, reverting to identity",
                    flush=True,
                )
            self.a, self.b, self.c = (1.0, 1.0, 0.0)
            p_cal = self.calibrate(probs)
            try:
                auc_after = float(roc_auc_score(y, p_cal))
            except Exception:
                auc_after = auc_before
        if verbose:
            ece_before = self._ece(y, p)
            ece_after = self._ece(y, p_cal)
            print(
                f"[BetaCalibrator] a={self.a:.4f} b={self.b:.4f} c={self.c:+.4f}  AUC: {auc_before:.4f}->{auc_after:.4f}  ECE: {ece_before:.4f}->{ece_after:.4f}",
                flush=True,
            )
        return (self.a, self.b, self.c)

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        from scipy.special import expit as sp_expit

        p = np.clip(probs.astype(np.float64), 1e-07, 1 - 1e-07)
        # Positive a and b keep this map nondecreasing in the original risk score.
        logit_cal = self.a * np.log(p) - self.b * np.log(1 - p) + self.c
        return sp_expit(logit_cal).astype(np.float32)

    @staticmethod
    def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
        ece = 0.0
        n = len(y)
        for lo in np.linspace(0, 1, n_bins + 1)[:-1]:
            hi = lo + 1.0 / n_bins
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            ece += mask.sum() / max(n, 1) * abs(float(y[mask].mean()) - float(p[mask].mean()))
        return float(ece)


class AffineCalibrator(nn.Module):
    """Calibrate logits with a positive temperature and a bias."""

    def __init__(self):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))

    @property
    def temperature_value(self) -> float:
        return float(torch.exp(self.log_temperature).item())

    @property
    def bias_value(self) -> float:
        return float(self.bias.item())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_temperature)
        return logits / T + self.bias

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.05,
        max_iter: int = 300,
        verbose: bool = True,
        auc_drop_tolerance: float = 0.002,
    ) -> Tuple[float, float]:
        logits_np = logits.astype(np.float32).reshape(-1, 1)
        labels_np = labels.astype(np.float32).reshape(-1, 1)
        logits_t = torch.from_numpy(logits_np)
        labels_t = torch.from_numpy(labels_np)
        nll_crit = nn.BCEWithLogitsLoss()
        prev = float(np.clip(labels_np.mean(), 0.0001, 1.0 - 0.0001))
        mean_logit = float(logits_np.mean())
        init_bias = math.log(prev / (1.0 - prev)) - mean_logit
        with torch.no_grad():
            self.log_temperature.fill_(0.0)
            self.bias.fill_(float(np.clip(init_bias, -6.0, 6.0)))
        probs_before = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        try:
            auc_before = float(roc_auc_score(labels, probs_before))
        except Exception:
            auc_before = 0.5
        optimizer = torch.optim.LBFGS(
            [self.log_temperature, self.bias],
            lr=lr,
            max_iter=max_iter,
            line_search_fn="strong_wolfe",
        )

        def _closure():
            optimizer.zero_grad()
            scaled = self.forward(logits_t)
            loss = nll_crit(scaled, labels_t)
            loss.backward()
            return loss

        optimizer.step(_closure)
        T_learned = self.temperature_value
        b_learned = self.bias_value
        if not 0.05 <= T_learned <= 20.0:
            if verbose:
                print(
                    f"[AffineCalibrator] Warning: T={T_learned:.4f} 超出合理范围，回退到 T=1.0",
                    flush=True,
                )
            with torch.no_grad():
                self.log_temperature.fill_(0.0)
            T_learned = 1.0
        probs_after = self.calibrate(logits)
        try:
            auc_after = float(roc_auc_score(labels, probs_after))
        except Exception:
            auc_after = auc_before
        auc_drop = auc_before - auc_after
        if auc_drop > float(auc_drop_tolerance):
            if verbose:
                print(
                    f"[AffineCalibrator] Warning: AUC 下降 {auc_drop:.4f}，回退到恒等校准",
                    flush=True,
                )
            with torch.no_grad():
                self.log_temperature.fill_(0.0)
                self.bias.fill_(0.0)
            T_learned = 1.0
            b_learned = 0.0
        elif verbose:
            print(
                f"[AffineCalibrator] T={T_learned:.4f}, b={b_learned:+.4f}  val AUC: {auc_before:.4f} -> {auc_after:.4f} (Δ={auc_after - auc_before:+.4f})",
                flush=True,
            )
        return (T_learned, b_learned)

    @torch.no_grad()
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        logits_t = torch.from_numpy(logits.astype(np.float32))
        scaled = self.forward(logits_t)
        return torch.sigmoid(scaled).numpy().reshape(-1)


def apply_temperature_calibration(
    model: "DualHeadPredictor",
    X: np.ndarray,
    y: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    prov_ids: Optional[np.ndarray] = None,
    focus_idx: Optional[Sequence[int]] = None,
    seed: int = 42,
    X_eng: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, float]:
    """Select calibration and decision threshold within the validation set."""
    model.eval()

    def _extract(idx: np.ndarray):
        score_logits_list, prob_logits_list, fused_logits_list, labels_list = ([], [], [], [])
        for s in range(0, len(idx), batch_size):
            b = idx[s : s + batch_size]
            with torch.no_grad():
                xb = torch.from_numpy(X[b]).long().to(device)
                eb = None
                if X_eng is not None:
                    eb = torch.from_numpy(X_eng[b].astype(np.float32)).to(device)
                slogit, plogit, flogit, _h, _alpha = model.forward_all(
                    xb, X_eng=eb, detach_prob_features=False, return_details=True
                )
            score_logits_list.append(slogit.detach().cpu().numpy())
            prob_logits_list.append(plogit.detach().cpu().numpy())
            fused_logits_list.append(flogit.detach().cpu().numpy())
            labels_list.append(y[b])
        return (
            np.concatenate(score_logits_list),
            np.concatenate(prob_logits_list),
            np.concatenate(fused_logits_list),
            np.concatenate(labels_list),
        )

    idx_cal_fit, idx_cal_select = split_val_for_nested_calibration(
        idx_val=idx_val, y=y, prov_ids=prov_ids, seed=seed
    )
    fit_slogits, fit_plogits, fit_flogits, fit_labels = _extract(idx_cal_fit)
    sel_slogits, sel_plogits, sel_flogits, sel_labels = _extract(idx_cal_select)
    test_slogits, test_plogits, test_flogits, test_labels = _extract(idx_test)

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    score_probs_sel = _sigmoid(sel_slogits)
    score_probs_test = _sigmoid(test_slogits)
    prob_head_sel = _sigmoid(sel_plogits)
    prob_head_test = _sigmoid(test_plogits)
    fused_probs_sel = _sigmoid(sel_flogits)
    fused_probs_test = _sigmoid(test_flogits)
    cal_score = AffineCalibrator()
    T_score, bias_score = cal_score.fit(fit_slogits, fit_labels, verbose=True)
    aff_score_sel = cal_score.calibrate(sel_slogits)
    aff_score_test = cal_score.calibrate(test_slogits)
    cal_fused = AffineCalibrator()
    T_fused, bias_fused = cal_fused.fit(fit_flogits, fit_labels, verbose=True)
    aff_fused_sel = cal_fused.calibrate(sel_flogits)
    aff_fused_test = cal_fused.calibrate(test_flogits)
    iso_ok = False
    iso_fused_sel = fused_probs_sel
    iso_fused_test = fused_probs_test
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(_sigmoid(fit_flogits), fit_labels.astype(float))
        iso_fused_sel = np.clip(iso.predict(fused_probs_sel), 1e-06, 1 - 1e-06)
        iso_fused_test = np.clip(iso.predict(fused_probs_test), 1e-06, 1 - 1e-06)
        iso_ok = True
    except Exception:
        pass
    candidates = {
        "fused_uncalibrated": (fused_probs_sel, fused_probs_test),
        "fused_affine_platt": (aff_fused_sel, aff_fused_test),
        "prob_head_raw": (prob_head_sel, prob_head_test),
        "score_affine_platt": (aff_score_sel, aff_score_test),
        "score_uncalibrated": (score_probs_sel, score_probs_test),
    }
    score_probs_fit = _sigmoid(fit_slogits)
    try:
        beta_cal = BetaCalibrator()
        beta_cal.fit(score_probs_fit, fit_labels, verbose=True)
        beta_score_sel = beta_cal.calibrate(score_probs_sel)
        beta_score_test = beta_cal.calibrate(score_probs_test)
        candidates["score_beta_cal"] = (beta_score_sel, beta_score_test)
        print("[BetaCalibrator] Added score_beta_cal as candidate", flush=True)
    except Exception as _be:
        print(f"[BetaCalibrator] Skipped: {_be}", flush=True)
    if iso_ok:
        candidates["fused_isotonic"] = (iso_fused_sel, iso_fused_test)

    def _select_score(
        y_true: np.ndarray, probs: np.ndarray, thr_opt: float
    ) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        m05 = compute_prob_metrics(y_true, probs, thr=0.5)
        mopt = compute_prob_metrics(y_true, probs, thr=thr_opt)
        prev_gap = abs(float(np.mean(probs)) - float(np.mean(y_true)))
        score = (
            0.38 * float(m05["auc"])
            + 0.18 * float(m05["ap"])
            + 0.22 * float(mopt["f1"])
            + 0.1 * float(mopt["sensitivity"])
            + 0.04 * float(m05["specificity"])
            - 0.04 * float(m05["brier"])
            - 0.02 * float(m05["ece"])
            - 0.02 * float(min(prev_gap, 0.3))
        )
        return (float(score), m05, mopt)

    val_scores = {}
    candidate_meta = {}
    for name, (vp, tp) in candidates.items():
        thr, f1v = choose_best_f1_threshold(sel_labels, vp, sen_floor=0.5)
        if f1v <= 0.0:
            thr, f1v = choose_best_f1_threshold(sel_labels, vp, sen_floor=0.0)
        sc, m05, mopt = _select_score(sel_labels, vp, thr)
        val_scores[name] = (sc, thr, f1v)
        candidate_meta[name] = {
            "selection_thr05": m05,
            "selection_opt": mopt,
            "thr": thr,
            "selection_f1": f1v,
            "score": sc,
        }
        print(
            f"[OptimalThreshold] {name}：选择集最优阈值={thr:.3f} (F1={f1v:.4f}, score={sc:.4f})",
            flush=True,
        )
    best_source = max(val_scores.items(), key=lambda kv: kv[1][0])[0]
    best_thr = float(val_scores[best_source][1])
    best_probs_test = candidates[best_source][1]
    m_score_05 = compute_prob_metrics(test_labels, score_probs_test, thr=0.5)
    m_prob_05 = compute_prob_metrics(test_labels, prob_head_test, thr=0.5)
    m_fused_05 = compute_prob_metrics(test_labels, fused_probs_test, thr=0.5)
    m_best_opt = compute_prob_metrics(test_labels, best_probs_test, thr=best_thr)
    print(
        f"[Calibration] score原始(thr=0.5): AUC={m_score_05['auc']:.4f} ECE={m_score_05['ece']:.4f} F1={m_score_05['f1']:.4f} Sen={m_score_05['sensitivity']:.4f}",
        flush=True,
    )
    print(
        f"[Calibration] prob_head(thr=0.5): AUC={m_prob_05['auc']:.4f} ECE={m_prob_05['ece']:.4f} F1={m_prob_05['f1']:.4f} Sen={m_prob_05['sensitivity']:.4f}",
        flush=True,
    )
    print(
        f"[Calibration] fused原始(thr=0.5): AUC={m_fused_05['auc']:.4f} ECE={m_fused_05['ece']:.4f} F1={m_fused_05['f1']:.4f} Sen={m_fused_05['sensitivity']:.4f}",
        flush=True,
    )
    print(
        f"[Calibration] 最终选择: src={best_source} AUC={m_best_opt['auc']:.4f} ECE={m_best_opt['ece']:.4f} F1={m_best_opt['f1']:.4f} Sen={m_best_opt['sensitivity']:.4f} (thr={best_thr:.3f})",
        flush=True,
    )
    calib_df = pd.DataFrame(
        {
            "y_true": test_labels,
            "prob_score_uncalibrated": score_probs_test,
            "prob_score_affine": aff_score_test,
            "prob_fused_uncalibrated": fused_probs_test,
            "prob_fused_affine": aff_fused_test,
            "prob_fused_isotonic": iso_fused_test if iso_ok else fused_probs_test,
            "prob_prob_head": prob_head_test,
            "prob_best": best_probs_test,
        }
    )
    calib_df.to_csv(
        output_dir / "temperature_calibration_test.csv", index=False, encoding="utf-8-sig"
    )
    meta = {
        "calibration_method": "nested_posthoc_calibration",
        "temperature_score": float(T_score),
        "bias_score": float(bias_score),
        "temperature_fused": float(T_fused),
        "bias_fused": float(bias_fused),
        "prob_source": best_source,
        "optimal_threshold": float(best_thr),
        "idx_cal_fit_size": int(len(idx_cal_fit)),
        "idx_cal_select_size": int(len(idx_cal_select)),
        "val_candidates": candidate_meta,
        "score_uncalibrated_thr05": m_score_05,
        "prob_head_thr05": m_prob_05,
        "fused_uncalibrated_thr05": m_fused_05,
        "best_optimal_thr": m_best_opt,
        "focus_idx_count": int(len(focus_idx)) if focus_idx is not None else 0,
        "has_prov_ids": bool(prov_ids is not None),
    }
    (output_dir / "temperature_calibration_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (best_probs_test, float(T_fused if "fused" in best_source else T_score), float(best_thr))


def compute_calibration_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "model",
    n_bins: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(y_prob, dtype=np.float64), 1e-07, 1 - 1e-07)
    brier = float(brier_score_loss(y, p))
    frac_pos, mean_pred_val = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
    bins_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = len(y)
    ece = 0.0
    calib_rows = []
    for i in range(n_bins):
        lo, hi = (float(bins_edges[i]), float(bins_edges[i + 1]))
        mask = (p >= lo) & (p < hi)
        n_bin = int(mask.sum())
        if n_bin == 0:
            continue
        acc_bin = float(y[mask].mean())
        conf_bin = float(p[mask].mean())
        ece += n_bin / n_total * abs(acc_bin - conf_bin)
        calib_rows.append(
            {
                "bin_idx": i + 1,
                "bin_lo": round(lo, 3),
                "bin_hi": round(hi, 3),
                "n_samples": n_bin,
                "mean_predicted_prob": round(conf_bin, 4),
                "fraction_positive": round(acc_bin, 4),
                "calibration_gap": round(acc_bin - conf_bin, 4),
            }
        )
    df_calib = pd.DataFrame(calib_rows)
    result = {
        "model": model_name,
        "brier_score": round(brier, 5),
        "ece": round(ece, 5),
        "n_bins": n_bins,
        "n_samples": int(n_total),
        "positive_rate": round(float(y.mean()), 4),
    }
    if output_dir:
        df_calib.to_csv(
            output_dir / f"calibration_curve_{model_name}.csv", index=False, encoding="utf-8-sig"
        )
        (output_dir / f"calibration_metrics_{model_name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[Calibration] {model_name}: Brier={brier:.4f}  ECE={ece:.4f}", flush=True)
    return {**result, "calibration_curve": df_calib}
