"""Structure fitting and parent stability selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
import torch

from .graph import extract_y_parents_adaptive, save_causal_graph
from .models.structure import DOCDModel
from .training import bootstrap_structure_y_parents, train_docd


def run_structure_stage(
    args,
    output_dir,
    X_xy,
    var_specs_xy,
    names_xy,
    y_index,
    idx_train,
    allowed_mask,
    y_in_prior,
    device,
):
    OUTPUT_DIR = Path(output_dir)
    print("\n===== Stage 1: Y-aware DOCD (Y-focused + Stable DAG Penalty) =====")
    docd = DOCDModel(
        var_specs=var_specs_xy,
        emb_dim=args.emb_dim,
        allowed_mask=allowed_mask,
        y_index=y_index,
        y_nll_weight=args.y_nll_weight,
        init_logit=args.init_logit,
        init_logit_y_in=args.init_logit_y_in,
        y_in_prior=y_in_prior,
        y_in_prior_lambda=args.mi_prior_lambda,
        mlp_hidden=192,
        mlp_depth=3,
    )
    docd_logs = train_docd(
        docd=docd,
        X_xy=X_xy,
        idx_train=idx_train,
        device=device,
        inner_steps=args.docd_inner,
        outer_loops=args.docd_outer,
        min_outer_loops=args.min_outer_loops,
        lr_docd=args.lr_docd,
        lr_struct_mult=args.lr_struct_mult,
        l1_lambda=args.l1_lambda,
        l1_y_in_mult=args.l1_y_in_mult,
        dag_tol=args.dag_tol,
        rho_max=args.rho_max,
        batch_size=args.batch_size,
        stop_patience=args.stop_patience,
        hutch_eval_samples=args.hutch_eval_samples,
    )
    A_prob_full = docd_logs["A_prob"]
    print(f"Saving trained DOCD model to {OUTPUT_DIR / 'docd_final.pth'}...")
    torch.save(
        {k: v.detach().cpu() for k, v in docd.state_dict().items()}, OUTPUT_DIR / "docd_final.pth"
    )
    save_causal_graph(
        A_prob_full, names_xy, edge_threshold=args.edge_threshold, output_dir=OUTPUT_DIR
    )
    y_fixed = [
        (names_xy[i], float(A_prob_full[i, y_index]))
        for i in range(A_prob_full.shape[0])
        if i != y_index and float(A_prob_full[i, y_index]) >= float(args.edge_threshold)
    ]
    y_fixed.sort(key=lambda x: -x[1])
    pd.DataFrame(y_fixed, columns=["cause", "edge_prob"]).to_csv(
        OUTPUT_DIR / "y_parents_fixed.csv", index=False, encoding="utf-8-sig"
    )
    res_adapt = extract_y_parents_adaptive(
        A_prob_full=A_prob_full,
        names_xy=names_xy,
        y_index=y_index,
        fixed_threshold=float(args.edge_threshold),
        min_k=int(args.y_parent_min_k),
        max_k=int(args.y_parent_max_k),
        min_edge=float(args.y_parent_min_edge),
    )
    res_adapt["parents_df"].to_csv(
        OUTPUT_DIR / "y_parents_adaptive.csv", index=False, encoding="utf-8-sig"
    )
    (OUTPUT_DIR / "y_parents_thresholding.json").write_text(
        json.dumps(
            {
                "method": res_adapt.get("method"),
                "threshold_used": res_adapt.get("threshold_used"),
                "topk_threshold": res_adapt.get("topk_threshold", None),
                "elbow_threshold": res_adapt.get("elbow_threshold", None),
                "elbow_k": res_adapt.get("elbow_k", None),
                "fixed_threshold": res_adapt.get("fixed_threshold", float(args.edge_threshold)),
                "y_in_stats": res_adapt.get("y_in_stats", {}),
                "n_fixed": int(len(y_fixed)),
                "n_adaptive": int(res_adapt["parents_df"].shape[0]),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Y parents (fixed thr={args.edge_threshold}): {len(y_fixed)}")
    print(
        f"Y parents (adaptive {res_adapt.get('method')} thr={res_adapt.get('threshold_used'):.4f}): {int(res_adapt['parents_df'].shape[0])}"
    )
    stable_causes: List[str] = []
    print("\n===== Stage 1b: Bootstrap Stability (X->Y, Top-N Rank Stability) =====")
    if int(args.bootstrap_struct) > 0:
        stab_df = bootstrap_structure_y_parents(
            X_xy=X_xy,
            var_specs_xy=var_specs_xy,
            names_xy=names_xy,
            y_index=y_index,
            allowed_mask=allowed_mask,
            idx_train=idx_train,
            device=device,
            n_bootstrap=int(args.bootstrap_struct),
            edge_threshold=float(args.edge_threshold),
            lr_docd=float(args.lr_docd),
            lr_struct_mult=float(args.lr_struct_mult),
            inner_steps=max(160, int(args.docd_inner) // 2),
            outer_loops=max(4, int(args.docd_outer)),
            min_outer_loops=max(3, int(args.min_outer_loops)),
            l1_lambda=float(args.l1_lambda),
            l1_y_in_mult=float(args.l1_y_in_mult),
            dag_tol=float(args.dag_tol),
            rho_max=float(args.rho_max),
            batch_size=int(getattr(args, "_ft_batch_size", args.batch_size)),
            emb_dim=int(args.emb_dim),
            y_nll_weight=float(args.y_nll_weight),
            init_logit=float(args.init_logit),
            init_logit_y_in=float(args.init_logit_y_in),
            y_in_prior=y_in_prior,
            y_in_prior_lambda=float(args.mi_prior_lambda),
            save_yin_path=OUTPUT_DIR / "y_in_prob_bootstrap.npy"
            if bool(getattr(args, "bootstrap_save_yin", False))
            else None,
            save_yin_summary_path=OUTPUT_DIR / "y_in_prob_bootstrap_summary.csv"
            if bool(getattr(args, "bootstrap_save_yin", False))
            else None,
            topn=int(args.stability_topn),
        )
        stab_df.to_csv(OUTPUT_DIR / "stability_y_parents.csv", index=False, encoding="utf-8-sig")
        stab_sel = stab_df[
            (stab_df["freq_rate_topn"] >= float(args.stability_freq))
            & (stab_df["mean_prob_topn"] >= float(args.y_parent_min_edge))
        ].copy()
        stab_sel = stab_sel.sort_values(
            ["freq_rate_topn", "mean_prob_topn"], ascending=[False, False]
        ).reset_index(drop=True)
        stab_sel.to_csv(OUTPUT_DIR / "y_parents_stable.csv", index=False, encoding="utf-8-sig")
        stable_causes = stab_sel["cause"].tolist()
        print(
            f"Stable causes (freq_topn>={args.stability_freq}, topn={args.stability_topn}): {len(stable_causes)}"
        )
    return (docd, A_prob_full)
