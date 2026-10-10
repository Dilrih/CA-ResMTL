"""Shared prediction stage for fresh and resumed experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from . import config
from .ablation import run_ablation_study
from .auxiliary import build_auxiliary_targets
from .baselines import _get_hgbdt_test_probs, baseline_extended, baseline_hgbdt, run_hybrid_stacking
from .calibration import apply_temperature_calibration, compute_calibration_analysis
from .features import CausalFeatureEngineer
from .graph import (
    build_attention_components,
    choose_threshold_by_target_degree,
    load_causal_focus_indices,
)
from .models.backbones import CausalAttentionBackbone, DeepResMLPBackbone, FTTransformerBackbone
from .models.predictor import CAResMTL, DualHeadPredictor
from .training import (
    eval_predictor,
    extract_predictor_probabilities,
    make_dataloader,
    train_predictor,
)


def run_prediction_stage(
    args,
    output_dir,
    X_xy,
    var_specs_xy,
    A_prob_full,
    idx_train,
    idx_val,
    idx_test,
    device,
    raw=None,
):
    OUTPUT_DIR = Path(output_dir)
    X = X_xy[:, :-1].astype(np.int64)
    y = (X_xy[:, -1] - 1).clip(0, 1).astype(np.int64)
    print("\n===== Stage 2: CA-ResMTL Predictor (OOD) =====", flush=True)
    focus_idx = load_causal_focus_indices(
        OUTPUT_DIR, var_specs_xy[:-1], A_prob_full=A_prob_full, topk=args.causal_topk
    )
    _stable_names: List[str] = []
    _stable_csv = OUTPUT_DIR / "y_parents_stable.csv"
    if _stable_csv.exists():
        try:
            _df_stab = pd.read_csv(_stable_csv)
            _stable_names = _df_stab["cause"].astype(str).tolist()
        except Exception:
            pass
    if not _stable_names:
        _stable_names = [var_specs_xy[i].name for i in focus_idx if i < len(var_specs_xy) - 1]
    X_eng_all: Optional[np.ndarray] = None
    causal_fe: Optional[CausalFeatureEngineer] = None
    if _stable_names and bool(getattr(args, "use_causal_features", True)):
        print(
            f"[Model] Building causal interaction features from {len(_stable_names)} stable parents...",
            flush=True,
        )
        causal_fe = CausalFeatureEngineer(
            stable_parent_names=_stable_names,
            col_names=[v.name for v in var_specs_xy[:-1]],
            n_interaction_pairs=int(getattr(args, "n_interaction_pairs", 60)),
        )
        try:
            X_eng_all = causal_fe.fit_transform(X, y, idx_train)
            n_eng = X_eng_all.shape[1]
            print(f"[Model] Engineered feature matrix: {X_eng_all.shape}", flush=True)
            np.save(OUTPUT_DIR / "X_eng.npy", X_eng_all)
        except Exception as _fe_err:
            print(f"[Model] Feature engineering failed: {_fe_err}", flush=True)
            X_eng_all = None
            n_eng = 0
    else:
        n_eng = 0
    y_aux = (
        build_auxiliary_targets(raw, args.target_mode)
        if args.use_multitask and raw is not None
        else None
    )
    if y_aux is None and args.use_multitask and (OUTPUT_DIR / "auxiliary_targets.npz").exists():
        with np.load(OUTPUT_DIR / "auxiliary_targets.npz", allow_pickle=False) as saved:
            y_aux = [saved[key] for key in sorted(saved.files)]
        if any(len(values) != len(y) for values in y_aux):
            raise ValueError("Saved auxiliary labels do not match the primary outcome rows.")
    n_aux = len(y_aux) if y_aux is not None else 0
    A_prob_x = A_prob_full[: X.shape[1], : X.shape[1]]
    thr_attn = choose_threshold_by_target_degree(
        A_prob_x, target_degree=int(args.graph_target_degree), min_thr=0.02, max_thr=0.6
    )
    (OUTPUT_DIR / "attention_threshold.json").write_text(
        json.dumps(
            {"thr_attn": thr_attn, "target_degree": int(args.graph_target_degree)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    attn_mask, attn_bias = build_attention_components(
        A_prob_x,
        threshold=thr_attn,
        mode=str(getattr(args, "attn_prior_mode", config.DEFAULT_ATTN_PRIOR_MODE)),
        prior_strength=float(
            getattr(args, "attn_prior_strength", config.DEFAULT_ATTN_PRIOR_STRENGTH)
        ),
        nonancestor_penalty=float(
            getattr(args, "attn_nonancestor_penalty", config.DEFAULT_ATTN_NONANCESTOR_PENALTY)
        ),
    )
    focus_idx = load_causal_focus_indices(
        OUTPUT_DIR,
        var_specs_xy[:-1],
        A_prob_full=A_prob_full,
        topk=int(getattr(args, "causal_topk", config.DEFAULT_CAUSAL_TOPK)),
    )
    try:
        focus_names = [var_specs_xy[:-1][i].name for i in focus_idx]
        (OUTPUT_DIR / "causal_focus_indices.json").write_text(
            json.dumps(
                {"focus_idx": focus_idx, "focus_names": focus_names}, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
    except Exception:
        pass
    backbone_choice = str(getattr(args, "backbone", "deep_resmlp")).lower()
    if backbone_choice == "ft_transformer":
        _ft_batch = min(int(args.batch_size), 256)
        _ft_ffn = int(getattr(args, "ffn_factor", 2))
        print(f"[FT-Transformer] batch={_ft_batch}, ffn_factor={_ft_ffn}", flush=True)
        backbone = FTTransformerBackbone(
            var_specs=var_specs_xy[:-1],
            emb_dim=args.emb_dim,
            n_layers=args.layers,
            n_heads=args.heads,
            ffn_factor=_ft_ffn,
            dropout=args.dropout,
            causal_attn_bias=attn_bias,
            causal_focus_idx=focus_idx,
        )
        args.__dict__["_ft_batch_size"] = _ft_batch
    elif backbone_choice == "causal_attention":
        backbone = CausalAttentionBackbone(
            var_specs=var_specs_xy[:-1],
            emb_dim=args.emb_dim,
            n_layers=args.layers,
            n_heads=args.heads,
            dropout=args.dropout,
            causal_attn_mask=attn_mask,
            causal_attn_bias=attn_bias,
            causal_focus_idx=focus_idx,
            causal_residual_dropout=float(
                getattr(args, "causal_residual_dropout", config.DEFAULT_CAUSAL_RESIDUAL_DROPOUT)
            ),
        )
    else:
        n_blk = int(getattr(args, "resmlp_blocks", 6))
        print(
            f"[DeepResMLP] n_blocks={n_blk}, emb_dim={args.emb_dim}, n_engineered={n_eng}, focus_idx={len(focus_idx)}",
            flush=True,
        )
        backbone = DeepResMLPBackbone(
            var_specs=var_specs_xy[:-1],
            emb_dim=args.emb_dim,
            n_blocks=n_blk,
            dropout=args.dropout,
            causal_focus_idx=focus_idx,
            variant="full",
            n_engineered=n_eng,
        )
    prov_ids = None
    prov_path = OUTPUT_DIR / "prov_ids.npy"
    if prov_path.exists():
        try:
            prov_ids = np.load(prov_path).astype(np.int64)
        except Exception:
            prov_ids = None
    domain_adv = bool(args.domain_adv) and prov_ids is not None
    num_domains = int(prov_ids.max() + 1) if prov_ids is not None and prov_ids.size > 0 else 1
    if isinstance(backbone, DeepResMLPBackbone):
        pred = CAResMTL(
            backbone=backbone,
            emb_dim=args.emb_dim,
            dropout=args.dropout,
            n_aux_tasks=n_aux,
            logit_scale_init=float(
                getattr(args, "logit_scale_init", config.DEFAULT_LOGIT_SCALE_INIT)
            ),
        )
        print(f"[Model] Using CAResMTL (n_aux={n_aux})", flush=True)
    else:
        pred = DualHeadPredictor(
            backbone=backbone,
            emb_dim=args.emb_dim,
            dropout=args.dropout,
            num_domains=num_domains,
            domain_adv=domain_adv,
            logit_scale_init=float(
                getattr(args, "logit_scale_init", config.DEFAULT_LOGIT_SCALE_INIT)
            ),
        )
    train_hist = train_predictor(
        model=pred,
        X=X,
        y=y,
        idx_train=idx_train,
        idx_val=idx_val,
        device=device,
        epochs=int(args.epochs_pred),
        lr=float(args.lr_pred),
        batch_size=int(getattr(args, "_ft_batch_size", args.batch_size)),
        weight_decay=float(getattr(args, "weight_decay", 0.0001)),
        early_stop_patience=int(
            getattr(args, "early_stop_patience", config.DEFAULT_EARLY_STOP_PATIENCE)
        ),
        save_path=OUTPUT_DIR / "resmtl_best_model.pth",
        prov_ids=prov_ids,
        domain_adv=domain_adv,
        lambda_domain=float(args.lambda_domain),
        grl_lambda=float(getattr(args, "grl_lambda", 1.0)),
        focal_alpha=float(getattr(args, "focal_alpha", config.DEFAULT_FOCAL_ALPHA)),
        focal_gamma=float(getattr(args, "focal_gamma", config.DEFAULT_FOCAL_GAMMA)),
        label_smoothing=float(getattr(args, "label_smoothing", config.DEFAULT_LABEL_SMOOTHING)),
        selection_criterion=str(getattr(args, "selection_criterion", "hybrid")),
        ece_lambda=float(getattr(args, "ece_lambda", 0.3)),
        loss_type=str(getattr(args, "loss_type", config.DEFAULT_LOSS_TYPE)),
        rank_weight=float(getattr(args, "rank_weight", config.DEFAULT_RANK_WEIGHT)),
        margin_weight=float(getattr(args, "margin_weight", config.DEFAULT_MARGIN_WEIGHT)),
        target_margin=float(getattr(args, "target_margin", config.DEFAULT_TARGET_MARGIN)),
        use_group_dro=prov_ids is not None
        and (not bool(getattr(args, "disable_group_dro", False))),
        group_dro_eta=float(getattr(args, "group_dro_eta", config.DEFAULT_GROUP_DRO_ETA)),
        aux_prob_weight=float(getattr(args, "aux_prob_weight", config.DEFAULT_AUX_PROB_WEIGHT)),
        aux_calib_weight=float(getattr(args, "aux_calib_weight", config.DEFAULT_AUX_CALIB_WEIGHT)),
        logit_adjust_tau=float(getattr(args, "logit_adjust_tau", config.DEFAULT_LOGIT_ADJ_TAU)),
        late_rank_mult=float(getattr(args, "late_rank_mult", config.DEFAULT_LATE_RANK_MULT)),
        worst_group_weight=float(
            getattr(args, "worst_group_weight", config.DEFAULT_WORST_GROUP_WEIGHT)
        ),
        X_eng=X_eng_all,
        y_aux=y_aux,
        aux_task_weights=[0.2, 0.1][:n_aux] if n_aux > 0 else None,
        use_aucm=bool(getattr(args, "use_aucm", config.DEFAULT_USE_AUCM)),
        aucm_margin=float(getattr(args, "aucm_margin", config.DEFAULT_AUCM_MARGIN)),
        aucm_weight=float(getattr(args, "aucm_weight", config.DEFAULT_AUCM_WEIGHT)),
        aucm_warmup=int(getattr(args, "aucm_warmup", config.DEFAULT_AUCM_WARMUP)),
        aucm_delay=int(getattr(args, "aucm_delay", config.DEFAULT_AUCM_DELAY)),
        use_pcgrad=bool(getattr(args, "use_pcgrad", config.DEFAULT_USE_PCGRAD)),
        mono_weight=float(getattr(args, "mono_weight", config.DEFAULT_MONO_WEIGHT)),
        coral_weight=float(getattr(args, "coral_weight", config.DEFAULT_CORAL_WEIGHT)),
        f2_sen_floor=float(getattr(args, "f2_sen_floor", config.DEFAULT_F2_SEN_FLOOR)),
        swa_start=float(getattr(args, "swa_start", config.DEFAULT_SWA_START)),
        swa_freq=int(getattr(args, "swa_freq", config.DEFAULT_SWA_FREQ)),
    )
    test_metrics = eval_predictor(
        pred,
        X,
        y,
        idx_test,
        device,
        args.batch_size,
        source=str(getattr(args, "eval_source", config.DEFAULT_EVAL_SOURCE)),
        X_eng=X_eng_all,
    )
    base = baseline_hgbdt(X, y, idx_train, idx_val, idx_test)
    out_metrics = {
        "val_best": train_hist["best"],
        "test": test_metrics,
        "baseline_histgbdt": base,
        "prediction_file": "pred_test_predictions.csv",
        "run_manifest": "run_manifest.json",
        "eval_source": str(getattr(args, "eval_source", config.DEFAULT_EVAL_SOURCE)),
    }
    print("\n=====  Affine Probability Calibration =====", flush=True)
    try:
        cal_probs, cal_T, best_thr = apply_temperature_calibration(
            model=pred,
            X=X,
            y=y,
            idx_val=idx_val,
            idx_test=idx_test,
            device=device,
            batch_size=int(getattr(args, "_ft_batch_size", args.batch_size)),
            output_dir=OUTPUT_DIR,
            prov_ids=prov_ids,
            focus_idx=focus_idx,
            seed=int(getattr(args, "seed", 42)),
            X_eng=X_eng_all,
        )
        cal_pred_opt = (cal_probs >= best_thr).astype(int)
        cal_f1_opt = (
            float(f1_score(y[idx_test], cal_pred_opt)) if len(np.unique(cal_pred_opt)) > 1 else 0.0
        )
        cal_cm_opt = confusion_matrix(y[idx_test], cal_pred_opt, labels=[0, 1]).tolist()
        tn_c, fp_c, fn_c, tp_c = (
            cal_cm_opt[0][0],
            cal_cm_opt[0][1],
            cal_cm_opt[1][0],
            cal_cm_opt[1][1],
        )
        out_metrics["calibrated"] = {
            "temperature": cal_T,
            "optimal_threshold": best_thr,
            "auc": float(roc_auc_score(y[idx_test], cal_probs)),
            "ap": float(average_precision_score(y[idx_test], cal_probs)),
            "brier": float(brier_score_loss(y[idx_test], cal_probs)),
            "f1_at_optimal_thr": cal_f1_opt,
            "sensitivity": float(tp_c) / max(tp_c + fn_c, 1),
            "specificity": float(tn_c) / max(tn_c + fp_c, 1),
            "cm_at_optimal_thr": cal_cm_opt,
        }
    except Exception as _cal_e:
        print(f"[Warn] Temperature Scaling 失败: {_cal_e}，跳过校准步骤。", flush=True)
    out_metrics["backbone"] = backbone_choice
    out_metrics["n_aux_tasks"] = n_aux if isinstance(pred, CAResMTL) else 0
    out_metrics["n_engineered_features"] = n_eng if isinstance(backbone, DeepResMLPBackbone) else 0
    (OUTPUT_DIR / "pred_metrics.json").write_text(
        json.dumps(out_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pred.eval()
    rows = []
    _test_probs: List[float] = []
    with torch.no_grad():
        for xb, yb in make_dataloader(X, y, idx_test, batch_size=args.batch_size, shuffle=False):
            xb_t = torch.from_numpy(xb).long().to(device)
            eb_t = (
                torch.from_numpy(
                    X_eng_all[idx_test][len(rows) : len(rows) + len(yb)].astype(np.float32)
                ).to(device)
                if X_eng_all is not None
                else None
            )
            slogit, plogit, flogit, _h, alpha = pred.forward_all(
                xb_t, X_eng=eb_t, detach_prob_features=False, return_details=True
            )
            p_score = torch.sigmoid(slogit).cpu().numpy()
            p_prob = torch.sigmoid(plogit).cpu().numpy()
            p_fused = torch.sigmoid(flogit).cpu().numpy()
            p_alpha = alpha.cpu().numpy()
            for ps, pp, pf, pa, yy in zip(
                p_score.tolist(), p_prob.tolist(), p_fused.tolist(), p_alpha.tolist(), yb.tolist()
            ):
                rows.append(
                    {
                        "risk_score_prob": ps,
                        "risk_prob_head": pp,
                        "risk_fused_prob": pf,
                        "fusion_alpha": pa,
                        "y_true": yy,
                    }
                )
                _test_probs.append(float(pf))
    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "pred_test_predictions.csv", index=False, encoding="utf-8-sig"
    )
    if bool(getattr(args, "run_extended_baselines", False)):
        print("\n===== [Stage2-only 补充实验B] Extended Baselines =====", flush=True)
        ext_results = baseline_extended(
            X=X,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            var_names=[vs.name for vs in var_specs_xy[:-1]],
            output_dir=OUTPUT_DIR,
            prov_ids=prov_ids,
            focus_idx=focus_idx,
        )
        for mname, mres in ext_results.items():
            if mname.startswith("_"):
                continue
            out_metrics[f"baseline_{mname}"] = {
                k: v for k, v in mres.items() if k != "top_features"
            }
        (OUTPUT_DIR / "pred_metrics.json").write_text(
            json.dumps(out_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if bool(getattr(args, "run_calibration", False)):
        print("\n===== [Stage2-only 补充实验F] Calibration Analysis =====", flush=True)
        y_test_arr = y[idx_test]
        calib_resmtl = compute_calibration_analysis(
            y_true=y_test_arr,
            y_prob=np.array(_test_probs),
            model_name="ca_resmtl",
            n_bins=10,
            output_dir=OUTPUT_DIR,
        )
        _hgbdt_probs = _get_hgbdt_test_probs(X, y, idx_train, idx_test)
        if _hgbdt_probs is not None:
            calib_hgbdt = compute_calibration_analysis(
                y_true=y_test_arr,
                y_prob=_hgbdt_probs,
                model_name="histgbdt",
                n_bins=10,
                output_dir=OUTPUT_DIR,
            )
            out_metrics["calibration_ca_resmtl"] = {
                k: v for k, v in calib_resmtl.items() if k != "calibration_curve"
            }
            out_metrics["calibration_histgbdt"] = {
                k: v for k, v in calib_hgbdt.items() if k != "calibration_curve"
            }
            (OUTPUT_DIR / "pred_metrics.json").write_text(
                json.dumps(out_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    if bool(getattr(args, "run_hybrid_stack", False)):
        print("\n===== Hybrid Stacking =====", flush=True)
        probs_train = extract_predictor_probabilities(
            pred, X, idx_train, device, int(args.batch_size), X_eng=X_eng_all
        )
        probs_val = extract_predictor_probabilities(
            pred, X, idx_val, device, int(args.batch_size), X_eng=X_eng_all
        )
        probs_test = extract_predictor_probabilities(
            pred, X, idx_test, device, int(args.batch_size), X_eng=X_eng_all
        )
        stack_res = run_hybrid_stacking(
            X=X,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            predictor_probs_train=probs_train,
            predictor_probs_val=probs_val,
            predictor_probs_test=probs_test,
            output_dir=OUTPUT_DIR,
            prov_ids=prov_ids,
            X_eng=X_eng_all,
        )
        out_metrics["hybrid_stack"] = {
            k: v for k, v in stack_res.items() if k not in {"meta_coefficients", "models"}
        }
        (OUTPUT_DIR / "pred_metrics.json").write_text(
            json.dumps(out_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if bool(getattr(args, "run_ablation", False)):
        print("\n===== [Stage2-only 补充实验A] Ablation Study =====", flush=True)
        _ = run_ablation_study(
            X=X,
            y=y,
            var_specs=var_specs_xy[:-1],
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            causal_attn_mask=attn_mask,
            causal_attn_bias=attn_bias,
            device=device,
            emb_dim=int(args.emb_dim),
            n_layers=int(args.layers),
            n_heads=int(args.heads),
            dropout=float(args.dropout),
            epochs=int(args.epochs_pred),
            lr=float(args.lr_pred),
            batch_size=int(getattr(args, "_ft_batch_size", args.batch_size)),
            output_dir=OUTPUT_DIR,
            X_eng=X_eng_all,
            y_aux=y_aux,
            causal_focus_idx=focus_idx,
            n_blocks=int(getattr(args, "resmlp_blocks", 6)),
        )
    pd.DataFrame(train_hist["history"]).to_csv(OUTPUT_DIR / "predictor_history.csv", index=False)
    return out_metrics
