"""Structure and prediction optimization routines."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

from . import config
from .data import VarSpec
from .graph import acyclicity_hutchinson_expm
from .losses import (
    AsymmetricLoss,
    AUCMLoss,
    ListNetRankLoss,
    UncertaintyWeightedMTL,
    coral_loss,
    ordinal_monotonicity_loss,
)
from .metrics import f2_score_at_sensitivity_floor
from .models.embeddings import OrdinalAwareEmbedding
from .models.predictor import DualHeadPredictor
from .models.structure import DOCDModel


def train_docd(
    docd: DOCDModel,
    X_xy: np.ndarray,
    idx_train: np.ndarray,
    device,
    inner_steps: int,
    outer_loops: int,
    min_outer_loops: int,
    lr_docd: float,
    lr_struct_mult: float,
    l1_lambda: float,
    l1_y_in_mult: float,
    dag_tol: float,
    rho_max: float,
    batch_size: int,
    warmup_outer: int = 2,
    stop_patience: int = config.DEFAULT_STOP_PATIENCE,
    hutch_eval_samples: int = config.DEFAULT_HUTCH_EVAL_SAMPLES,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit the structure model with an augmented Lagrangian objective."""
    docd.to(device)
    params = [
        {"params": [docd.S], "lr": lr_docd * lr_struct_mult, "weight_decay": 0.0},
        {
            "params": [p for n, p in docd.named_parameters() if n != "S"],
            "lr": lr_docd,
            "weight_decay": 0.0001,
        },
    ]
    opt = torch.optim.AdamW(params)
    rho = 1.0
    alpha = 0.0
    logs: Dict[str, Any] = {"outer": []}
    tau0, tau1 = (config.DEFAULT_GUMBEL_TAU_START, config.DEFAULT_GUMBEL_TAU_END)
    below_cnt = 0
    outer_pbar = tqdm(range(1, int(outer_loops) + 1), desc="DOCD Outer", disable=not verbose)
    for out_it in outer_pbar:
        tau = float(tau0 + (tau1 - tau0) * (out_it - 1) / max(1, int(outer_loops) - 1))
        mix = float(max(0.0, min(0.65, 0.65 * (tau / tau0))))
        if out_it <= int(warmup_outer):
            l1_eff = 0.0
        else:
            frac = (out_it - int(warmup_outer)) / max(1, int(outer_loops) - int(warmup_outer))
            l1_eff = float(l1_lambda) * float(0.25 + 0.75 * frac)
        inner_pbar = tqdm(
            range(1, int(inner_steps) + 1),
            desc=f"DOCD Inner (Loop {out_it})",
            leave=False,
            disable=not verbose,
        )
        for step in inner_pbar:
            b = np.random.choice(
                idx_train, size=min(int(batch_size), len(idx_train)), replace=False
            )
            xb = torch.from_numpy(X_xy[b]).long().to(device)
            out = docd(xb, tau=tau, mix_sample=mix)
            nll = out["nll"]
            out["nll_x"]
            out["nll_y"]
            out["prior_term"]
            A_prob = out["A_prob"]
            h = out["h"]
            out["h_raw"]
            y_idx = docd.y_index
            identity = torch.eye(A_prob.shape[0], device=A_prob.device)
            mask_nodiag = 1.0 - identity
            mask_y_in = torch.zeros_like(A_prob)
            mask_y_in[:, y_idx] = 1.0
            mask_y_in[y_idx, y_idx] = 0.0
            mask_non_y = mask_nodiag.clone()
            mask_non_y[:, y_idx] = 0.0
            denom_y = mask_y_in.sum().clamp_min(1.0)
            denom_non = mask_non_y.sum().clamp_min(1.0)
            l1_y_in = (A_prob.abs() * mask_y_in).sum() / denom_y
            l1_non_y = (A_prob.abs() * mask_non_y).sum() / denom_non
            l1_term = float(l1_eff) * l1_non_y + float(l1_eff) * float(l1_y_in_mult) * l1_y_in
            loss = nll + l1_term + alpha * h + 0.5 * rho * h * h
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered in DOCD training.")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(docd.parameters(), 5.0)
            opt.step()
            with torch.no_grad():
                docd.S.fill_diagonal_(-14.0)
            inner_pbar.set_postfix(nll=f"{nll.item():.2f}", h=f"{h.item():.2e}", rho=f"{rho:.0e}")
        with torch.no_grad():
            A_prob_eval = docd.adj_prob()
            h_eval_raw = acyclicity_hutchinson_expm(
                A_prob_eval,
                n_trace=int(hutch_eval_samples),
                series_k=config.DEFAULT_HUTCH_SERIES_K,
                scale_by_dim=True,
                clamp_nonneg=False,
            )
            h_eval = float(h_eval_raw.clamp_min(0.0).cpu().item())
            b = np.random.choice(idx_train, size=min(4096, len(idx_train)), replace=False)
            xb = torch.from_numpy(X_xy[b]).long().to(device)
            out_eval = docd(xb, tau=tau, mix_sample=0.0)
            nll_eval = float(out_eval["nll"].cpu().item())
            Amax = float(A_prob_eval.max().detach().cpu().item())
        logs["outer"].append(
            {
                "iter": int(out_it),
                "tau": float(tau),
                "mix": float(mix),
                "h_eval": float(h_eval),
                "h_eval_raw": float(h_eval_raw.cpu().item()),
                "nll_eval": float(nll_eval),
                "rho": float(rho),
                "alpha": float(alpha),
                "Amax": float(Amax),
                "l1_eff": float(l1_eff),
            }
        )
        if verbose:
            tqdm.write(
                f"[DOCD] outer={out_it:02d} h={h_eval:.3e} rho={rho:.1e} nll={nll_eval:.3f} Amax={Amax:.3f}"
            )
        if out_it >= int(min_outer_loops):
            if h_eval <= float(dag_tol):
                below_cnt += 1
            else:
                below_cnt = 0
            if below_cnt >= int(stop_patience) or rho >= float(rho_max):
                if verbose:
                    tqdm.write(
                        f"[DOCD] STOP: outer={out_it} h_eval={h_eval:.3e} below_cnt={below_cnt}"
                    )
                break
        alpha = alpha + rho * h_eval
        if h_eval > float(dag_tol) * 10.0 and rho < float(rho_max):
            rho = min(rho * 10.0, float(rho_max))
    logs["A_prob"] = docd.adjacency_prob_numpy()
    return logs


def bootstrap_structure_y_parents(
    X_xy: np.ndarray,
    var_specs_xy: List[VarSpec],
    names_xy: List[str],
    y_index: int,
    allowed_mask: np.ndarray,
    idx_train: np.ndarray,
    device,
    n_bootstrap: int,
    edge_threshold: float,
    lr_docd: float,
    lr_struct_mult: float,
    inner_steps: int,
    outer_loops: int,
    min_outer_loops: int,
    l1_lambda: float,
    l1_y_in_mult: float,
    dag_tol: float,
    rho_max: float,
    batch_size: int,
    emb_dim: int,
    y_nll_weight: float,
    init_logit: float,
    init_logit_y_in: float,
    y_in_prior: Optional[np.ndarray],
    y_in_prior_lambda: float,
    topn: int = config.DEFAULT_STAB_TOPN,
    save_yin_path: Optional[Path] = None,
    save_yin_summary_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Estimate the stability of incoming Y edges over bootstrap samples."""
    counts_topn: Dict[str, int] = {}
    probs_topn: Dict[str, float] = {}
    counts_thr: Dict[str, int] = {}
    probs_thr: Dict[str, float] = {}
    y_in_prob_mat: Optional[np.ndarray] = None
    if save_yin_path is not None:
        M = int(X_xy.shape[1])
        y_in_prob_mat = np.zeros((int(n_bootstrap), int(M)), dtype=np.float32)
    for b in tqdm(range(1, int(n_bootstrap) + 1), desc="Stability Bootstrap"):
        boot_idx = np.random.choice(idx_train, size=len(idx_train), replace=True)
        docd = DOCDModel(
            var_specs=var_specs_xy,
            emb_dim=int(emb_dim),
            allowed_mask=allowed_mask,
            y_index=int(y_index),
            y_nll_weight=float(y_nll_weight),
            init_logit=float(init_logit),
            init_logit_y_in=float(init_logit_y_in),
            y_in_prior=y_in_prior,
            y_in_prior_lambda=float(y_in_prior_lambda),
            mlp_hidden=192,
            mlp_depth=3,
        )
        logs = train_docd(
            docd=docd,
            X_xy=X_xy,
            idx_train=boot_idx,
            device=device,
            inner_steps=int(inner_steps),
            outer_loops=int(outer_loops),
            min_outer_loops=int(min_outer_loops),
            lr_docd=float(lr_docd),
            lr_struct_mult=float(lr_struct_mult),
            l1_lambda=float(l1_lambda),
            l1_y_in_mult=float(l1_y_in_mult),
            dag_tol=float(dag_tol),
            rho_max=float(rho_max),
            batch_size=int(batch_size),
            verbose=False,
        )
        A_prob = logs["A_prob"]
        y_in = [(i, float(A_prob[i, y_index])) for i in range(A_prob.shape[0]) if i != y_index]
        y_in.sort(key=lambda x: -x[1])
        if y_in_prob_mat is not None:
            y_in_prob_mat[b - 1, :] = A_prob[:, y_index].astype(np.float32)
        for i, w in y_in[: max(1, int(topn))]:
            key = names_xy[i]
            counts_topn[key] = counts_topn.get(key, 0) + 1
            probs_topn[key] = probs_topn.get(key, 0.0) + float(w)
        for i, w in y_in:
            if w >= float(edge_threshold):
                key = names_xy[i]
                counts_thr[key] = counts_thr.get(key, 0) + 1
                probs_thr[key] = probs_thr.get(key, 0.0) + float(w)
    if y_in_prob_mat is not None and save_yin_path is not None:
        try:
            np.save(str(save_yin_path), y_in_prob_mat)
        except Exception as e:
            print(f"[Warn] Failed to save y_in_prob_mat to {save_yin_path}: {e}", flush=True)
        try:
            summ_rows = []
            for i, name in enumerate(names_xy):
                if i == int(y_index):
                    continue
                v = y_in_prob_mat[:, i]
                summ_rows.append(
                    {
                        "cause": name,
                        "mean_prob": float(np.mean(v)),
                        "std_prob": float(np.std(v)),
                        "q025": float(np.quantile(v, 0.025)),
                        "q975": float(np.quantile(v, 0.975)),
                        "min": float(np.min(v)),
                        "max": float(np.max(v)),
                    }
                )
            df_yin = (
                pd.DataFrame(summ_rows)
                .sort_values("mean_prob", ascending=False)
                .reset_index(drop=True)
            )
            if save_yin_summary_path is not None:
                df_yin.to_csv(save_yin_summary_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"[Warn] Failed to summarize y_in_prob_mat: {e}", flush=True)
    keys = sorted(set(list(counts_topn.keys()) + list(counts_thr.keys())))
    rows = []
    for k in keys:
        ct = int(counts_topn.get(k, 0))
        cthr = int(counts_thr.get(k, 0))
        rows.append(
            {
                "cause": k,
                "freq_topn": ct,
                "freq_rate_topn": ct / float(max(1, int(n_bootstrap))),
                "mean_prob_topn": probs_topn.get(k, 0.0) / float(max(1, ct)) if ct > 0 else 0.0,
                "freq_threshold": cthr,
                "freq_rate_threshold": cthr / float(max(1, int(n_bootstrap))),
                "mean_prob_threshold": probs_thr.get(k, 0.0) / float(max(1, cthr))
                if cthr > 0
                else 0.0,
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["freq_rate_topn", "mean_prob_topn"], ascending=[False, False]
        ).reset_index(drop=True)
    return df


def make_dataloader(X: np.ndarray, y: np.ndarray, idx: np.ndarray, batch_size: int, shuffle: bool):
    ids = idx.copy()
    if shuffle:
        np.random.shuffle(ids)
    for s in range(0, len(ids), int(batch_size)):
        b = ids[s : s + int(batch_size)]
        yield (X[b], y[b])


@torch.no_grad()
def eval_predictor(
    model: DualHeadPredictor,
    X: np.ndarray,
    y: np.ndarray,
    idx: np.ndarray,
    device,
    batch_size: int,
    source: str = config.DEFAULT_EVAL_SOURCE,
    X_eng: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate a predictor and retain branch-specific probability metrics."""
    model.eval()
    score_probs, prob_probs, fused_probs, alpha_vals, ys = ([], [], [], [], [])
    for s in range(0, len(idx), int(batch_size)):
        b = idx[s : s + int(batch_size)]
        xb = X[b]
        yb = y[b]
        xb_t = torch.from_numpy(xb).long().to(device)
        eb_t = None
        if X_eng is not None:
            eb_t = torch.from_numpy(X_eng[b].astype(np.float32)).to(device)
        slogit, plogit, flogit, _h, alpha = model.forward_all(
            xb_t, X_eng=eb_t, detach_prob_features=False, return_details=True
        )
        score_probs.append(torch.sigmoid(slogit).detach().cpu().numpy())
        prob_probs.append(torch.sigmoid(plogit).detach().cpu().numpy())
        fused_probs.append(torch.sigmoid(flogit).detach().cpu().numpy())
        alpha_vals.append(alpha.detach().cpu().numpy())
        ys.append(yb)
    score_probs = np.concatenate(score_probs, axis=0)
    prob_probs = np.concatenate(prob_probs, axis=0)
    fused_probs = np.concatenate(fused_probs, axis=0)
    alpha_vals = np.concatenate(alpha_vals, axis=0)
    ys = np.concatenate(ys, axis=0)
    src_key = str(source).lower().strip()
    if src_key == "score":
        probs = score_probs
    elif src_key == "prob":
        probs = prob_probs
    else:
        probs = fused_probs
        src_key = "fused"
    auc = roc_auc_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
    ap = average_precision_score(ys, probs) if len(np.unique(ys)) > 1 else float("nan")
    best_f1_bal, best_thr_bal = (-1.0, 0.5)
    for _t in np.linspace(0.03, 0.8, 300):
        _pred = (probs >= _t).astype(np.int64)
        if _pred.sum() == 0 or _pred.sum() == len(_pred):
            continue
        _cm = confusion_matrix(ys, _pred, labels=[0, 1])
        _tn, _fp, _fn, _tp = _cm.ravel()
        _sen = float(_tp) / max(_tp + _fn, 1)
        if _sen < float(config.DEFAULT_MIN_SENSITIVITY_THR):
            continue
        _f1 = 2.0 * _tp / max(2 * _tp + _fp + _fn, 1)
        if _f1 > best_f1_bal:
            best_f1_bal = _f1
            best_thr_bal = float(_t)
    if best_f1_bal < 0:
        for _t in np.linspace(0.03, 0.8, 300):
            _pred = (probs >= _t).astype(np.int64)
            if _pred.sum() == 0 or _pred.sum() == len(_pred):
                continue
            _f1 = float(f1_score(ys, _pred))
            if _f1 > best_f1_bal:
                best_f1_bal = _f1
                best_thr_bal = float(_t)
    best_f1_scr, best_thr_scr = (-1.0, 0.5)
    for _t in np.linspace(0.03, 0.8, 300):
        _pred = (probs >= _t).astype(np.int64)
        if _pred.sum() == 0:
            continue
        _cm = confusion_matrix(ys, _pred, labels=[0, 1])
        _tn, _fp, _fn, _tp = _cm.ravel()
        _sen = float(_tp) / max(_tp + _fn, 1)
        if _sen < 0.75:
            continue
        _f1 = 2.0 * _tp / max(2 * _tp + _fp + _fn, 1)
        if _f1 > best_f1_scr:
            best_f1_scr = _f1
            best_thr_scr = float(_t)
    if best_f1_scr < 0:
        best_thr_scr = best_thr_bal
    pred = (probs >= best_thr_bal).astype(np.int64)
    acc = accuracy_score(ys, pred)
    f1 = f1_score(ys, pred) if len(np.unique(pred)) > 1 else 0.0
    cm = confusion_matrix(ys, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    sensitivity = float(tp) / max(float(tp + fn), 1.0)
    specificity = float(tn) / max(float(tn + fp), 1.0)
    pred_05 = (probs >= 0.5).astype(np.int64)
    cm_05 = confusion_matrix(ys, pred_05, labels=[0, 1])
    tn_05, fp_05, fn_05, tp_05 = cm_05.ravel().tolist()
    pred_scr = (probs >= best_thr_scr).astype(np.int64)
    cm_scr = confusion_matrix(ys, pred_scr, labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm_scr.ravel().tolist()

    def _ece(y_true: np.ndarray, probs_in: np.ndarray, n_bins: int = 10) -> float:
        ece = 0.0
        n = len(y_true)
        for lo in np.linspace(0, 1, n_bins + 1)[:-1]:
            hi = lo + 1.0 / n_bins
            mask = (probs_in >= lo) & (probs_in < hi)
            if mask.sum() == 0:
                continue
            ece += (
                mask.sum()
                / max(n, 1)
                * abs(float(y_true[mask].mean()) - float(probs_in[mask].mean()))
            )
        return float(ece)

    return {
        "source": src_key,
        "auc": float(auc),
        "ap": float(ap),
        "threshold": float(best_thr_bal),
        "acc": float(acc),
        "f1": float(f1),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "f1_thr05": float(f1_score(ys, pred_05)) if len(np.unique(pred_05)) > 1 else 0.0,
        "sensitivity_thr05": float(tp_05) / max(tp_05 + fn_05, 1),
        "specificity_thr05": float(tn_05) / max(tn_05 + fp_05, 1),
        "tp_thr05": int(tp_05),
        "fn_thr05": int(fn_05),
        "fp_thr05": int(fp_05),
        "threshold_screening": float(best_thr_scr),
        "sensitivity_screening": float(tp_s) / max(tp_s + fn_s, 1),
        "specificity_screening": float(tn_s) / max(tn_s + fp_s, 1),
        "f1_screening": float(f1_score(ys, pred_scr)) if len(np.unique(pred_scr)) > 1 else 0.0,
        "tp_screening": int(tp_s),
        "fn_screening": int(fn_s),
        "fp_screening": int(fp_s),
        "mean_prob": float(np.mean(probs)),
        "score_mean_prob": float(np.mean(score_probs)),
        "prob_head_mean_prob": float(np.mean(prob_probs)),
        "fused_mean_prob": float(np.mean(fused_probs)),
        "fusion_alpha_mean": float(np.mean(alpha_vals)),
        "score_brier": float(brier_score_loss(ys, score_probs)),
        "prob_head_brier": float(brier_score_loss(ys, prob_probs)),
        "fused_brier": float(brier_score_loss(ys, fused_probs)),
        "score_ece": float(_ece(ys, score_probs)),
        "prob_head_ece": float(_ece(ys, prob_probs)),
        "fused_ece": float(_ece(ys, fused_probs)),
        "cm": cm.tolist(),
    }


def train_predictor(
    model: DualHeadPredictor,
    X: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 512,
    weight_decay: float = 0.0001,
    early_stop_patience: int = 10,
    pos_weight: Optional[float] = None,
    save_path: Optional[Path] = None,
    prov_ids: Optional[np.ndarray] = None,
    domain_adv: bool = False,
    lambda_domain: float = 0.25,
    grl_lambda: float = 1.0,
    focal_alpha: float = 0.55,
    focal_gamma: float = 1.0,
    label_smoothing: float = 0.02,
    selection_criterion: str = "hybrid",
    ece_lambda: float = 0.3,
    loss_type: str = config.DEFAULT_LOSS_TYPE,
    rank_weight: float = config.DEFAULT_RANK_WEIGHT,
    margin_weight: float = config.DEFAULT_MARGIN_WEIGHT,
    target_margin: float = config.DEFAULT_TARGET_MARGIN,
    pos_weight_power: float = config.DEFAULT_POS_WEIGHT_POWER,
    pos_weight_cap: float = config.DEFAULT_POS_WEIGHT_CAP,
    prior_match_weight: float = config.DEFAULT_PRIOR_MATCH_WEIGHT,
    overpredict_weight: float = config.DEFAULT_OVERPREDICT_WEIGHT,
    overpredict_tol: float = config.DEFAULT_OVERPREDICT_TOL,
    scale_reg_weight: float = config.DEFAULT_SCALE_REG_WEIGHT,
    scale_upper: float = config.DEFAULT_SCALE_UPPER,
    use_group_dro: bool = True,
    group_dro_eta: float = config.DEFAULT_GROUP_DRO_ETA,
    aux_prob_weight: float = config.DEFAULT_AUX_PROB_WEIGHT,
    aux_calib_weight: float = config.DEFAULT_AUX_CALIB_WEIGHT,
    logit_adjust_tau: float = config.DEFAULT_LOGIT_ADJ_TAU,
    late_rank_mult: float = config.DEFAULT_LATE_RANK_MULT,
    worst_group_weight: float = config.DEFAULT_WORST_GROUP_WEIGHT,
    X_eng: Optional[np.ndarray] = None,
    y_aux: Optional[List[np.ndarray]] = None,
    aux_task_weights: Optional[List[float]] = None,
    use_aucm: bool = config.DEFAULT_USE_AUCM,
    aucm_margin: float = config.DEFAULT_AUCM_MARGIN,
    aucm_weight: float = config.DEFAULT_AUCM_WEIGHT,
    use_pcgrad: bool = config.DEFAULT_USE_PCGRAD,
    mono_weight: float = config.DEFAULT_MONO_WEIGHT,
    coral_weight: float = config.DEFAULT_CORAL_WEIGHT,
    f2_sen_floor: float = config.DEFAULT_F2_SEN_FLOOR,
    aucm_warmup: int = config.DEFAULT_AUCM_WARMUP,
    aucm_delay: int = config.DEFAULT_AUCM_DELAY,
    swa_start: float = config.DEFAULT_SWA_START,
    swa_freq: int = config.DEFAULT_SWA_FREQ,
) -> Dict[str, Any]:
    """Train the fused predictor with ranking, calibration, and domain losses."""
    if loss_type != "asl":
        raise ValueError("Only the ASL primary loss is supported.")
    model = model.to(device)
    p_pos = float(np.mean(y[idx_train]))
    p_pos = float(min(max(p_pos, 0.0001), 1.0 - 0.0001))
    prior_logit = math.log(p_pos / (1.0 - p_pos))
    _model_accepts_eng = hasattr(model, "forward_multitask") or (
        hasattr(model, "backbone")
        and hasattr(model.backbone, "n_engineered")
        and (getattr(model.backbone, "n_engineered", 0) > 0)
    )
    _use_eng = _model_accepts_eng and X_eng is not None
    _n_aux = 0
    _y_aux_arrays: List[np.ndarray] = []
    _aux_weights: List[float] = []
    if y_aux is not None and hasattr(model, "aux_heads") and (model.aux_heads is not None):
        _n_aux = min(len(y_aux), len(model.aux_heads))
        _y_aux_arrays = [np.asarray(y_aux[i], dtype=np.float32) for i in range(_n_aux)]
        if aux_task_weights is not None:
            _aux_weights = [
                float(aux_task_weights[i]) if i < len(aux_task_weights) else 0.15
                for i in range(_n_aux)
            ]
        else:
            _aux_weights = [0.2, 0.1][:_n_aux]
        print(f"[train_predictor] Multi-task: n_aux={_n_aux}, weights={_aux_weights}", flush=True)
    try:
        with torch.no_grad():
            if (
                isinstance(model.score_head[-1], nn.Linear)
                and model.score_head[-1].bias is not None
            ):
                model.score_head[-1].bias.fill_(float(np.clip(prior_logit * 0.2, -2.5, 2.5)))
            if (
                isinstance(model.score_shortcut, nn.Linear)
                and model.score_shortcut.bias is not None
            ):
                model.score_shortcut.bias.fill_(float(np.clip(prior_logit * 0.1, -2.0, 2.0)))
    except Exception:
        pass
    inferred_pos_weight = ((1.0 - p_pos) / max(p_pos, 1e-06)) ** float(pos_weight_power)
    inferred_pos_weight = float(np.clip(inferred_pos_weight, 1.0, float(pos_weight_cap)))
    pos_weight_eff = float(pos_weight) if pos_weight is not None else float(inferred_pos_weight)
    pw_tensor = torch.tensor(float(max(pos_weight_eff, 1.0)), dtype=torch.float32, device=device)
    num_groups = int(prov_ids.max() + 1) if prov_ids is not None and prov_ids.size > 0 else 0
    use_group_dro = bool(use_group_dro) and num_groups > 1
    group_q = np.ones(num_groups, dtype=np.float64) / float(num_groups) if use_group_dro else None

    def iter_batches(idx: np.ndarray, shuffle: bool):
        idx = np.array(idx, dtype=int)
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            bidx = idx[i : i + batch_size]
            xb = X[bidx]
            yb = y[bidx]
            pb = prov_ids[bidx] if prov_ids is not None else None
            eb = X_eng[bidx] if _use_eng and X_eng is not None else None
            yb_aux = [_y_aux_arrays[k][bidx] for k in range(_n_aux)] if _n_aux > 0 else []
            yield (xb, yb, pb, eb, yb_aux)

    def _smooth_targets(targets: torch.Tensor) -> torch.Tensor:
        eps = float(label_smoothing)
        return targets * (1.0 - eps) + 0.5 * eps

    def _score_bce_vec(
        score_logits: torch.Tensor, targets: torch.Tensor, do_adjust: bool = True
    ) -> torch.Tensor:
        y_s = _smooth_targets(targets)
        adj_logits = (
            score_logits - float(logit_adjust_tau) * float(prior_logit)
            if do_adjust
            else score_logits
        )
        return F.binary_cross_entropy_with_logits(
            adj_logits, y_s, pos_weight=pw_tensor, reduction="none"
        )

    def _pair_rank_loss(
        logits: torch.Tensor, targets: torch.Tensor, mult: float = 1.0, n_pairs: int = 512
    ) -> torch.Tensor:
        pos = logits[targets > 0.5]
        neg = logits[targets <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0 or float(rank_weight) <= 0.0:
            return logits.new_zeros(())
        n = int(min(n_pairs, pos.numel(), neg.numel()))
        pos_s = pos[torch.randint(low=0, high=pos.numel(), size=(n,), device=logits.device)]
        neg_s = neg[torch.randint(low=0, high=neg.numel(), size=(n,), device=logits.device)]
        margin = float(target_margin) * 0.35
        return float(rank_weight) * float(mult) * F.softplus(-(pos_s - neg_s - margin)).mean()

    def _auc_surrogate(
        logits: torch.Tensor, targets: torch.Tensor, mult: float = 1.0
    ) -> torch.Tensor:
        pos = logits[targets > 0.5]
        neg = logits[targets <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return logits.new_zeros(())
        n_pairs = int(min(512, pos.numel(), neg.numel()))
        pos_s = pos[torch.randint(low=0, high=pos.numel(), size=(n_pairs,), device=logits.device)]
        neg_s = neg[torch.randint(low=0, high=neg.numel(), size=(n_pairs,), device=logits.device)]
        return 0.06 * float(mult) * (1.0 - torch.sigmoid(pos_s - neg_s)).mean()

    def _group_auc(
        y_true: np.ndarray, probs: np.ndarray, prov_ref: Optional[np.ndarray], idx_ref: np.ndarray
    ) -> float:
        if prov_ref is None:
            return float("nan")
        vals = []
        gsub = prov_ref[idx_ref]
        for g in np.unique(gsub):
            mask = gsub == g
            yg = y_true[mask]
            if len(np.unique(yg)) < 2:
                continue
            vals.append(float(roc_auc_score(yg, probs[mask])))
        return float(min(vals)) if vals else float("nan")

    def _ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
        ece = 0.0
        n = len(y_true)
        for lo in np.linspace(0, 1, n_bins + 1)[:-1]:
            hi = lo + 1.0 / n_bins
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() == 0:
                continue
            ece += (
                mask.sum() / max(n, 1) * abs(float(y_true[mask].mean()) - float(probs[mask].mean()))
            )
        return float(ece)

    def _thr_metrics(y_true: np.ndarray, probs: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
        pred = (probs >= float(thr)).astype(np.int64)
        cm = confusion_matrix(y_true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel().tolist()
        f1 = float(f1_score(y_true, pred)) if len(np.unique(pred)) > 1 else 0.0
        sen = float(tp) / max(tp + fn, 1)
        spe = float(tn) / max(tn + fp, 1)
        return {"f1": f1, "sensitivity": sen, "specificity": spe}

    def _best_thr(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float]:
        return f2_score_at_sensitivity_floor(y_true, probs, sen_floor=float(f2_sen_floor))

    def _domain_invariance_penalty(h: torch.Tensor, pb: Optional[np.ndarray]) -> torch.Tensor:
        if pb is None or float(coral_weight) <= 0.0:
            return h.new_zeros(())
        groups = np.unique(pb)
        if len(groups) < 2:
            return h.new_zeros(())
        pbt = torch.as_tensor(pb, dtype=torch.long, device=h.device)
        group_feats = []
        for g in groups:
            mask = pbt == int(g)
            if int(mask.sum().item()) < 6:
                continue
            group_feats.append(h[mask])
        if len(group_feats) < 2:
            return h.new_zeros(())
        total_coral = h.new_zeros(())
        n_pairs = 0
        for i in range(len(group_feats)):
            for j in range(i + 1, len(group_feats)):
                total_coral = total_coral + coral_loss(group_feats[i], group_feats[j])
                n_pairs += 1
        return total_coral / max(n_pairs, 1)

    def _gate_regularizer(alpha: torch.Tensor) -> torch.Tensor:
        target = float(config.DEFAULT_FUSION_PROB_INIT)
        centered = (alpha.mean() - target).pow(2)
        entropy = -(
            alpha.clamp(1e-06, 1 - 1e-06) * torch.log(alpha.clamp(1e-06, 1 - 1e-06))
            + (1.0 - alpha).clamp(1e-06, 1 - 1e-06)
            * torch.log((1.0 - alpha).clamp(1e-06, 1 - 1e-06))
        ).mean()
        return centered - 0.01 * entropy

    def _apply_group_dro(base_vec: torch.Tensor, pb: Optional[np.ndarray]) -> torch.Tensor:
        nonlocal group_q
        if not use_group_dro or pb is None:
            return base_vec.mean()
        grp_losses = []
        for g in np.unique(pb):
            mask_np = pb == g
            mask_t = torch.as_tensor(mask_np, dtype=torch.bool, device=base_vec.device)
            if int(mask_t.sum().item()) == 0:
                continue
            grp_losses.append((int(g), base_vec[mask_t].mean()))
        if not grp_losses:
            return base_vec.mean()
        for g, loss_g in grp_losses:
            group_q[g] *= math.exp(float(group_dro_eta) * float(loss_g.detach().cpu().item()))
        group_q = group_q / np.clip(group_q.sum(), 1e-08, None)
        return torch.stack(
            [
                torch.as_tensor(float(group_q[g]), device=base_vec.device, dtype=base_vec.dtype)
                * lg
                for g, lg in grp_losses
            ]
        ).sum()

    for p in model.parameters():
        p.requires_grad_(True)
    _aucm_fn = AUCMLoss(
        margin=float(aucm_margin), p_hat=float(p_pos), ema_decay=0.95, lam_max=5.0
    ).to(device)
    _listnet_fn = ListNetRankLoss(pos_weight=8.0, temperature=1.0)
    _asl_fn_global = AsymmetricLoss(
        gamma_neg=config.DEFAULT_ASL_GAMMA_NEG,
        gamma_pos=config.DEFAULT_ASL_GAMMA_POS,
        clip=config.DEFAULT_ASL_CLIP,
    )
    _mtl_uncertainty: Optional[UncertaintyWeightedMTL] = None
    if _n_aux > 0 and hasattr(model, "uncertainty_mtl") and (model.uncertainty_mtl is not None):
        _mtl_uncertainty = model.uncertainty_mtl
        print(f"[Model] Uncertainty-weighted auxiliary tasks (n_aux={_n_aux})", flush=True)
    _has_ordinal_emb = (
        float(mono_weight) > 0.0
        and hasattr(model, "backbone")
        and hasattr(model.backbone, "emb")
        and isinstance(model.backbone.emb, OrdinalAwareEmbedding)
    )
    _swa_start_epoch = int(float(swa_start) * float(epochs))
    _swa_pool: List[Dict] = []
    opt = torch.optim.AdamW(
        list(model.parameters()),
        lr=float(lr),
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-08,
    )
    n_warmup = max(3, int(epochs * 0.1))

    def _lr_lambda(ep: int) -> float:
        if ep < n_warmup:
            return float(ep + 1) / float(n_warmup)
        progress = float(ep - n_warmup) / max(float(epochs - n_warmup), 1.0)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
    best_state = None
    best_score = -1000000000.0
    bad = 0
    history: List[Dict[str, float]] = []
    pbar = tqdm(total=int(epochs), desc="Stage2 Predictor", ncols=160)
    for epoch in range(int(epochs)):
        model.train()
        losses = []
        progress = float(epoch + 1) / max(float(epochs), 1.0)
        rank_mult = 1.0 if progress < 0.7 else float(late_rank_mult)
        if epoch < int(aucm_delay):
            _aucm_w_eff = 0.0
        elif epoch < int(aucm_delay) + int(aucm_warmup):
            _aucm_w_eff = (
                float(aucm_weight) * (epoch - int(aucm_delay)) / max(float(aucm_warmup), 1.0)
            )
        else:
            _aucm_w_eff = float(aucm_weight)
        for xb, yb, pb, eb, yb_aux in iter_batches(idx_train, shuffle=True):
            xb_t = torch.as_tensor(xb, dtype=torch.long, device=device)
            yb_t = torch.tensor(yb, dtype=torch.float32, device=device)
            eb_t = (
                torch.as_tensor(eb, dtype=torch.float32, device=device) if eb is not None else None
            )
            score_logit, prob_logit, fused_logit, h, alpha = model.forward_all(
                xb_t, X_eng=eb_t, detach_prob_features=False, return_details=True
            )
            aux_logits = (
                model.aux_logits_from_features(h)
                if _n_aux > 0 and hasattr(model, "aux_logits_from_features")
                else []
            )
            score_probs = torch.sigmoid(score_logit)
            prob_probs = torch.sigmoid(prob_logit)
            torch.sigmoid(fused_logit)
            inv_pen = _domain_invariance_penalty(h, pb)
            asl_loss = _asl_fn_global(score_logit, yb_t)
            score_bce_vec = _score_bce_vec(score_logit, yb_t, do_adjust=True)
            dro_loss = _apply_group_dro(score_bce_vec, pb)
            if bool(use_aucm) and _aucm_w_eff > 1e-06:
                aucm_loss_val = _aucm_fn(score_logit, yb_t)
                listnet_w = float(rank_weight) * rank_mult * 0.5
            else:
                aucm_loss_val = score_logit.new_zeros(())
                listnet_w = float(rank_weight) * rank_mult
            listnet_loss = _listnet_fn(score_logit, yb_t)
            calib_reg = (score_probs.mean() - float(p_pos)).pow(2)
            overpredict_reg = F.relu(
                score_probs.mean() - (float(p_pos) + float(overpredict_tol))
            ).pow(2)
            prob_bce = F.binary_cross_entropy_with_logits(prob_logit, yb_t)
            fused_bce = F.binary_cross_entropy_with_logits(fused_logit, yb_t)
            prob_calib_reg = (prob_probs.mean() - float(p_pos)).pow(2)
            prob_over_reg = F.relu(prob_probs.mean() - (float(p_pos) + float(overpredict_tol))).pow(
                2
            )
            branch_consistency = F.smooth_l1_loss(prob_probs, score_probs.detach())
            gate_reg = _gate_regularizer(alpha)
            margin_reg = (
                F.relu(float(target_margin) * 0.3 - (2.0 * yb_t - 1.0) * score_logit).mean()
                if float(margin_weight) > 0
                else score_logit.new_zeros(())
            )
            if float(scale_reg_weight) > 0.0 and hasattr(model, "logit_scale_log"):
                scale_t = torch.exp(model.logit_scale_log.clamp(math.log(0.7), math.log(2.5)))
                scale_reg = float(scale_reg_weight) * F.relu(scale_t - float(scale_upper)).pow(2)
            else:
                scale_reg = score_logit.new_zeros(())
            if _has_ordinal_emb and epoch % 3 == 0 and (_aucm_w_eff >= float(aucm_weight)):
                mono_reg = ordinal_monotonicity_loss(
                    model.backbone.emb, device=device, sample_vars=24
                )
            else:
                mono_reg = score_logit.new_zeros(())
            loss = (
                0.48 * dro_loss
                + 0.18 * asl_loss
                + float(_aucm_w_eff) * aucm_loss_val
                + listnet_w * listnet_loss
                + float(prior_match_weight) * calib_reg
                + float(overpredict_weight) * overpredict_reg
                + float(aux_prob_weight) * prob_bce
                + 0.35 * float(aux_prob_weight) * fused_bce
                + float(aux_calib_weight)
                * (prob_calib_reg + 0.5 * prob_over_reg + 0.35 * branch_consistency)
                + 0.02 * gate_reg
                + float(coral_weight) * inv_pen
                + float(margin_weight) * margin_reg
                + scale_reg
                + float(mono_weight) * mono_reg
            )
            if not torch.isfinite(loss) or loss.abs().item() > 1000000.0:
                opt.zero_grad(set_to_none=True)
                losses.append(float(np.nan))
                continue
            if _n_aux > 0 and aux_logits:
                aux_losses_list: List[torch.Tensor] = []
                for k, (aux_logit, aux_label_np) in enumerate(zip(aux_logits, yb_aux)):
                    aux_mask = np.isfinite(aux_label_np) & (aux_label_np >= 0)
                    if int(aux_mask.sum()) < 4:
                        continue
                    aux_lbl_t = torch.as_tensor(
                        aux_label_np[aux_mask], dtype=torch.float32, device=device
                    )
                    aux_log_t = aux_logit[torch.from_numpy(np.where(aux_mask)[0]).long().to(device)]
                    aux_losses_list.append(F.binary_cross_entropy_with_logits(aux_log_t, aux_lbl_t))
                if aux_losses_list:
                    if _mtl_uncertainty is not None:
                        combined = _mtl_uncertainty([loss] + aux_losses_list)
                    else:
                        combined = loss
                        for k, al in enumerate(aux_losses_list):
                            combined = (
                                combined + (_aux_weights[k] if k < len(_aux_weights) else 0.15) * al
                            )
                    opt.zero_grad(set_to_none=True)
                    combined.backward()
                else:
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
            else:
                opt.zero_grad(set_to_none=True)
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        scheduler.step()
        current_lr = float(scheduler.get_last_lr()[0])
        model.eval()
        with torch.no_grad():
            flogits_val, ys_val = ([], [])
            for xb, yb, _pb, eb, _yaux in iter_batches(idx_val, shuffle=False):
                xb_t = torch.as_tensor(xb, dtype=torch.long, device=device)
                eb_t = (
                    torch.as_tensor(eb, dtype=torch.float32, device=device)
                    if eb is not None
                    else None
                )
                _sl, _pl, flogit, _h, _alpha = model.forward_all(
                    xb_t, X_eng=eb_t, detach_prob_features=False, return_details=True
                )
                flogits_val.append(flogit.detach().cpu().numpy())
                ys_val.append(yb)
        slogits_val = np.concatenate(flogits_val, axis=0)
        ys_val = np.concatenate(ys_val, axis=0).astype(np.int64)
        score_probs_val = 1.0 / (1.0 + np.exp(-np.clip(slogits_val, -30, 30)))
        val_auc = (
            float(roc_auc_score(ys_val, score_probs_val))
            if len(np.unique(ys_val)) > 1
            else float("nan")
        )
        val_ap = (
            float(average_precision_score(ys_val, score_probs_val))
            if len(np.unique(ys_val)) > 1
            else float("nan")
        )
        val_ece = _ece(ys_val, score_probs_val)
        val_brier = float(np.mean((ys_val.astype(float) - score_probs_val) ** 2))
        thr_opt, val_f2_opt = _best_thr(ys_val, score_probs_val)
        val_f1_opt = val_f2_opt
        thr05 = _thr_metrics(ys_val, score_probs_val, thr=0.5)
        prev_gap = abs(float(np.mean(score_probs_val)) - float(p_pos))
        worst_auc = _group_auc(ys_val, score_probs_val, prov_ids, idx_val)
        if not np.isfinite(worst_auc):
            worst_auc = val_auc
        sel = str(selection_criterion).lower().strip()
        if sel == "auc_ece":
            score = val_auc - float(ece_lambda) * val_ece
        elif sel == "brier":
            score = -val_brier
        elif sel == "auc":
            score = val_auc
        else:
            score = (
                0.42 * val_auc
                + 0.28 * val_f2_opt
                + 0.15 * val_ap
                + float(worst_group_weight) * float(worst_auc)
                - 0.05 * val_ece
                - 0.02 * min(prev_gap, 0.3)
            )
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(losses)),
                "val_auc": val_auc,
                "val_ap": val_ap,
                "val_ece": val_ece,
                "val_brier": val_brier,
                "val_prev_gap": float(prev_gap),
                "val_score_mean": float(np.mean(score_probs_val)),
                "val_f1_05": float(thr05["f1"]),
                "val_sensitivity_05": float(thr05["sensitivity"]),
                "val_specificity_05": float(thr05["specificity"]),
                "val_f2_opt": float(val_f2_opt),
                "val_f1_opt": float(val_f1_opt),
                "val_best_thr": float(thr_opt),
                "val_worst_group_auc": float(worst_auc),
                "lr": current_lr,
                "stage": "single",
            }
        )
        if score > best_score + 1e-05:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch >= _swa_start_epoch and epoch % max(int(swa_freq), 1) == 0:
            _swa_pool.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        pbar.set_postfix(
            loss=f"{float(np.nanmean(losses)):.4f}",
            auc=f"{val_auc:.4f}",
            ap=f"{val_ap:.4f}",
            wgauc=f"{worst_auc:.4f}",
            sen=f"{thr05['sensitivity']:.3f}",
            best=f"{best_score:.4f}",
            lr=f"{current_lr:.2e}",
            aucm_w=f"{_aucm_w_eff:.3f}",
        )
        pbar.update(1)
        if bad >= early_stop_patience:
            break
    pbar.close()
    if len(_swa_pool) >= 3:
        print(
            f"[Model] Averaging {len(_swa_pool)} checkpoints (epochs {_swa_start_epoch}+)...",
            flush=True,
        )
        swa_state = {}
        for k in _swa_pool[0]:
            try:
                swa_state[k] = (
                    torch.stack([c[k].float() for c in _swa_pool]).mean(0).to(_swa_pool[0][k].dtype)
                )
            except Exception:
                swa_state[k] = _swa_pool[-1][k]
        model.load_state_dict(swa_state)
        model.eval()
        with torch.no_grad():
            swa_logits, swa_ys = ([], [])
            for xb, yb, _pb, eb, _yaux in iter_batches(idx_val, shuffle=False):
                xb_t = torch.as_tensor(xb, dtype=torch.long, device=device)
                eb_t = (
                    torch.as_tensor(eb, dtype=torch.float32, device=device)
                    if eb is not None
                    else None
                )
                slogit = (
                    model.forward_score(xb_t, X_eng=eb_t)
                    if hasattr(model, "forward_score")
                    else model.forward_all(xb_t, X_eng=eb_t)[0]
                )
                swa_logits.append(slogit.detach().cpu().numpy())
                swa_ys.append(yb)
        swa_probs = 1.0 / (1.0 + np.exp(-np.clip(np.concatenate(swa_logits), -30, 30)))
        swa_ys = np.concatenate(swa_ys).astype(np.int64)
        if len(np.unique(swa_ys)) > 1:
            swa_auc = float(roc_auc_score(swa_ys, swa_probs))
            print(
                f"[Model] val_AUC: best_checkpoint={best_score:.4f}  SWA_val_AUC={swa_auc:.4f}",
                flush=True,
            )
            if best_state is not None:
                model.load_state_dict(best_state)
                model.eval()
                with torch.no_grad():
                    bs_logits, bs_ys = ([], [])
                    for xb, yb, _pb, eb, _yaux in iter_batches(idx_val, shuffle=False):
                        xb_t = torch.as_tensor(xb, dtype=torch.long, device=device)
                        eb_t = (
                            torch.as_tensor(eb, dtype=torch.float32, device=device)
                            if eb is not None
                            else None
                        )
                        slogit = (
                            model.forward_score(xb_t, X_eng=eb_t)
                            if hasattr(model, "forward_score")
                            else model.forward_all(xb_t, X_eng=eb_t)[0]
                        )
                        bs_logits.append(slogit.detach().cpu().numpy())
                        bs_ys.append(yb)
                bs_probs = 1.0 / (1.0 + np.exp(-np.clip(np.concatenate(bs_logits), -30, 30)))
                if len(np.unique(np.concatenate(bs_ys))) > 1:
                    bs_auc = float(roc_auc_score(np.concatenate(bs_ys).astype(np.int64), bs_probs))
                else:
                    bs_auc = -1.0
                if swa_auc >= bs_auc - 0.005:
                    print(
                        f"[Model] Using SWA state (val_AUC {swa_auc:.4f} vs best {bs_auc:.4f})",
                        flush=True,
                    )
                    best_state = swa_state
                else:
                    print(
                        f"[Model] Keeping best_checkpoint (val_AUC {bs_auc:.4f} > SWA {swa_auc:.4f})",
                        flush=True,
                    )
    else:
        print(f"[Model] Skipped (only {len(_swa_pool)} checkpoints collected)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path is not None:
        torch.save(model.state_dict(), str(save_path))
    return {
        "best": float(best_score),
        "history": history,
        "loss_type": "asl+listnet",
        "pos_weight": float(pos_weight_eff),
        "train_prevalence": float(p_pos),
        "stage_schedule": {"single": int(epochs)},
    }


def extract_predictor_probabilities(
    model: DualHeadPredictor,
    X: np.ndarray,
    idx: np.ndarray,
    device: torch.device,
    batch_size: int,
    X_eng: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    model.eval()
    out = {"score": [], "prob": [], "fused": [], "alpha": []}
    _has_eng = X_eng is not None
    with torch.no_grad():
        for s in range(0, len(idx), int(batch_size)):
            b = idx[s : s + int(batch_size)]
            xb = torch.from_numpy(X[b]).long().to(device)
            eb = torch.from_numpy(X_eng[b].astype(np.float32)).to(device) if _has_eng else None
            sl, pl, fl, _h, alpha = model.forward_all(
                xb, X_eng=eb, detach_prob_features=False, return_details=True
            )
            out["score"].append(torch.sigmoid(sl).cpu().numpy())
            out["prob"].append(torch.sigmoid(pl).cpu().numpy())
            out["fused"].append(torch.sigmoid(fl).cpu().numpy())
            out["alpha"].append(alpha.cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}
