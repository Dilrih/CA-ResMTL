"""Structural and representation ablation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from . import config
from .data import VarSpec
from .metrics import delong_auc_test
from .models.backbones import DeepResMLPBackbone
from .models.predictor import CAResMTL
from .training import eval_predictor, train_predictor


def run_ablation_study(
    X: np.ndarray,
    y: np.ndarray,
    var_specs: List[VarSpec],
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    causal_attn_mask: Optional[np.ndarray],
    causal_attn_bias: Optional[np.ndarray],
    device: torch.device,
    emb_dim: int = 64,
    n_layers: int = 3,
    n_heads: int = 4,
    dropout: float = 0.15,
    epochs: int = 25,
    lr: float = 0.002,
    batch_size: int = 256,
    output_dir: Optional[Path] = None,
    X_eng: Optional[np.ndarray] = None,
    y_aux: Optional[List[np.ndarray]] = None,
    causal_focus_idx: Optional[Sequence[int]] = None,
    n_blocks: int = 6,
) -> Dict[str, Any]:
    """Compare the full model with structural and embedding ablations."""
    VARIANTS = ["full", "no_eng", "no_multitask", "no_causal_branch", "std_emb"]
    DESC = {
        "full": "正式模型（ResMLP + causal features + multitask + causal branch + ordinal emb）",
        "no_eng": "w/o 因果驱动特征工程（X_eng）",
        "no_multitask": "w/o 多任务辅助监督",
        "no_causal_branch": "w/o causal focus branch",
        "std_emb": "w/o 有序建模（标准Embedding）",
    }
    all_results: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        print(f"\n[Ablation-New] ── 变体: {DESC[variant]} ──", flush=True)
        use_eng = X_eng is not None and variant != "no_eng"
        use_aux = y_aux is not None and len(y_aux) > 0 and (variant != "no_multitask")
        focus_idx_variant = (
            list(causal_focus_idx)
            if causal_focus_idx is not None and variant != "no_causal_branch"
            else []
        )
        n_eng_variant = int(X_eng.shape[1]) if use_eng else 0
        n_aux_variant = int(len(y_aux)) if use_aux else 0
        backbone = DeepResMLPBackbone(
            var_specs=var_specs,
            emb_dim=int(emb_dim),
            n_blocks=int(n_blocks),
            dropout=float(dropout),
            causal_focus_idx=focus_idx_variant,
            variant="std_emb" if variant == "std_emb" else "full",
            n_engineered=n_eng_variant,
        )
        pred = CAResMTL(
            backbone=backbone,
            emb_dim=int(emb_dim),
            dropout=float(dropout),
            n_aux_tasks=n_aux_variant,
            logit_scale_init=float(config.DEFAULT_LOGIT_SCALE_INIT),
        )
        save_path = output_dir / f"ablation_{variant}.pth" if output_dir else None
        train_hist = train_predictor(
            model=pred,
            X=X,
            y=y,
            idx_train=idx_train,
            idx_val=idx_val,
            device=device,
            epochs=int(epochs),
            lr=float(lr),
            batch_size=int(batch_size),
            weight_decay=0.0001,
            early_stop_patience=10,
            focal_alpha=config.DEFAULT_FOCAL_ALPHA,
            focal_gamma=config.DEFAULT_FOCAL_GAMMA,
            label_smoothing=config.DEFAULT_LABEL_SMOOTHING,
            loss_type=config.DEFAULT_LOSS_TYPE,
            rank_weight=config.DEFAULT_RANK_WEIGHT,
            margin_weight=config.DEFAULT_MARGIN_WEIGHT,
            target_margin=config.DEFAULT_TARGET_MARGIN,
            selection_criterion="hybrid",
            save_path=save_path,
            X_eng=X_eng if use_eng else None,
            y_aux=y_aux if use_aux else None,
            aux_task_weights=[0.2, 0.1][:n_aux_variant] if n_aux_variant > 0 else None,
            use_aucm=config.DEFAULT_USE_AUCM,
            aucm_margin=config.DEFAULT_AUCM_MARGIN,
            aucm_weight=config.DEFAULT_AUCM_WEIGHT,
            aucm_warmup=config.DEFAULT_AUCM_WARMUP,
            aucm_delay=config.DEFAULT_AUCM_DELAY,
            use_pcgrad=config.DEFAULT_USE_PCGRAD,
            mono_weight=config.DEFAULT_MONO_WEIGHT,
            coral_weight=config.DEFAULT_CORAL_WEIGHT,
            f2_sen_floor=config.DEFAULT_F2_SEN_FLOOR,
            swa_start=config.DEFAULT_SWA_START,
            swa_freq=config.DEFAULT_SWA_FREQ,
        )
        test_m = eval_predictor(
            pred, X, y, idx_test, device, batch_size, X_eng=X_eng if use_eng else None
        )
        test_probs = []
        test_y = []
        pred.eval()
        with torch.no_grad():
            for s in range(0, len(idx_test), int(batch_size)):
                b = idx_test[s : s + int(batch_size)]
                xb_t = torch.as_tensor(X[b], dtype=torch.long, device=device)
                eb_t = None
                if use_eng:
                    eb_t = torch.as_tensor(X_eng[b], dtype=torch.float32, device=device)
                fl_ab = pred.forward_score(xb_t, X_eng=eb_t)
                test_probs.extend(torch.sigmoid(fl_ab).cpu().numpy().tolist())
                test_y.extend(y[b].tolist())
        test_probs = np.asarray(test_probs, dtype=np.float32)
        test_y = np.asarray(test_y, dtype=np.int64)
        cm = test_m["cm"]
        tn, fp, fn, tp = (int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1]))
        sensitivity = float(tp) / max(tp + fn, 1)
        specificity = float(tn) / max(tn + fp, 1)
        all_results[variant] = {
            "description": DESC[variant],
            "val_best_auc": float(train_hist["best"]),
            "test": {**test_m, "sensitivity": sensitivity, "specificity": specificity},
            "test_probs": test_probs,
            "test_y": test_y,
            "uses_eng": bool(use_eng),
            "uses_multitask": bool(use_aux),
            "uses_causal_branch": bool(len(focus_idx_variant) > 0),
            "uses_ordinal_embedding": bool(variant != "std_emb"),
        }
        summary_rows.append(
            {
                "variant": variant,
                "description": DESC[variant],
                "val_best_auc": round(float(train_hist["best"]), 4),
                "test_auc": round(float(test_m["auc"]), 4),
                "test_ap": round(float(test_m["ap"]), 4),
                "test_f1": round(float(test_m["f1"]), 4),
                "test_acc": round(float(test_m["acc"]), 4),
                "test_sensitivity": round(sensitivity, 4),
                "test_specificity": round(specificity, 4),
                "test_fn": fn,
                "test_fp": fp,
                "test_tp": tp,
                "test_tn": tn,
                "use_eng": int(use_eng),
                "use_multitask": int(use_aux),
                "use_causal_branch": int(len(focus_idx_variant) > 0),
                "use_ordinal_embedding": int(variant != "std_emb"),
            }
        )
        print(
            f"  val_AUC={train_hist['best']:.4f}  test_AUC={test_m['auc']:.4f}  AP={test_m['ap']:.4f}  F1={test_m['f1']:.4f}  Sens={sensitivity:.4f}",
            flush=True,
        )
    df_summary = pd.DataFrame(summary_rows)
    if output_dir:
        df_summary.to_csv(output_dir / "ablation_summary.csv", index=False, encoding="utf-8-sig")
        _JSON_SKIP_KEYS = {"test_probs", "test_y"}
        ser = {
            k: {kk: vv for kk, vv in v.items() if kk not in _JSON_SKIP_KEYS}
            for k, v in all_results.items()
        }
        (output_dir / "ablation_full_results.json").write_text(
            json.dumps(ser, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n[Ablation-New] Results saved to {output_dir}/ablation_summary.csv", flush=True)
        print("\n[Ablation-New] DeLong AUC Significance Tests vs full model:", flush=True)
        delong_rows = []
        if "full" in all_results:
            y_test_ab = all_results["full"].get("test_y", None)
            preds_full = all_results["full"].get("test_probs", None)
            if y_test_ab is not None and preds_full is not None:
                for var in VARIANTS:
                    if var == "full":
                        continue
                    preds_var = all_results[var].get("test_probs", None)
                    if preds_var is None:
                        continue
                    try:
                        dl = delong_auc_test(y_test_ab, preds_full, preds_var)
                        delong_rows.append(
                            {
                                "variant": var,
                                "description": DESC[var],
                                "full_auc": dl["auc_a"],
                                "variant_auc": dl["auc_b"],
                                "delta_auc": dl["auc_diff"],
                                "z_stat": dl["z_stat"],
                                "p_value": dl["p_value"],
                                "significant_005": dl["significant"],
                            }
                        )
                        print(f"  {DESC[var]}: {dl.get('interpretation', '')}", flush=True)
                    except Exception as e:
                        print(f"  DeLong test failed for {var}: {e}", flush=True)
                if delong_rows:
                    pd.DataFrame(delong_rows).to_csv(
                        output_dir / "ablation_delong_tests.csv", index=False, encoding="utf-8-sig"
                    )
    print("\n[Ablation-New] Complete Summary:")
    show_cols = [
        "variant",
        "test_auc",
        "test_ap",
        "test_f1",
        "test_sensitivity",
        "test_fn",
        "test_fp",
    ]
    print(df_summary[show_cols].to_string(index=False))
    return {"summary": df_summary, "results": all_results}
