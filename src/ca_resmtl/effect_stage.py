"""Shared intervention analysis for fresh and resumed experiments."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from . import config
from .effects import ate_ranking_for_y_parents, bootstrap_ate_uncertainty, run_subgroup_ate_analysis
from .graph import choose_threshold_by_target_degree
from .metrics import compute_crosstask_spearman, summarize_threshold_sensitivity


def run_effect_stage(
    args, output_dir, docd, X_xy, var_specs_xy, names_xy, y_index, A_prob_full, idx_pool, device
):
    OUTPUT_DIR = Path(output_dir)
    topo_order = list(range(X_xy.shape[1]))
    idx_pool = np.asarray(idx_pool, dtype=np.int64)
    deg_ate = (
        int(args.graph_target_degree)
        if int(args.ate_graph_target_degree) <= 0
        else int(args.ate_graph_target_degree)
    )
    thr_ate = choose_threshold_by_target_degree(
        A_prob_full,
        target_degree=int(deg_ate),
        min_thr=float(config.DEFAULT_ATE_EDGE_THRESHOLD_MIN),
        max_thr=0.6,
    )
    (OUTPUT_DIR / "ate_graph_threshold.json").write_text(
        json.dumps(
            {"thr_ate": float(thr_ate), "target_degree": int(deg_ate)}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    thresholds: List[float] = []
    if str(getattr(args, "ate_thresholds", "")).strip():
        parts = re.split("[,\\s]+", str(args.ate_thresholds).strip())
        thresholds = [float(x) for x in parts if x]
    elif bool(getattr(args, "ate_robust", False)):
        t = float(thr_ate)
        thresholds = sorted(set([max(0.02, t * 0.85), t, min(0.6, t * 1.15)]))
    else:
        thresholds = [float(thr_ate)]
    (OUTPUT_DIR / "ate_thresholds_used.json").write_text(
        json.dumps(
            {"thresholds": thresholds, "thr_main": float(thr_ate)}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    per_thr_ranks: List[pd.DataFrame] = []
    rank_dfs_by_thr: Dict[float, pd.DataFrame] = {}
    cand: List[str] = []
    if getattr(args, "ate_candidates_file", ""):
        try:
            dfc = pd.read_csv(args.ate_candidates_file)
            if "cause" in dfc.columns:
                cand = dfc["cause"].astype(str).tolist()
            else:
                cand = dfc.iloc[:, 0].astype(str).tolist()
        except Exception as _e:
            print(f"[Warn] Failed to load ate_candidates_file: {_e}")
            cand = []
    if not cand:
        src = str(getattr(args, "ate_candidates_source", "stable"))
        if src == "stable" and (OUTPUT_DIR / "y_parents_stable.csv").exists():
            cand = pd.read_csv(OUTPUT_DIR / "y_parents_stable.csv")["cause"].astype(str).tolist()
        elif src == "adaptive" and (OUTPUT_DIR / "y_parents_adaptive.csv").exists():
            cand = pd.read_csv(OUTPUT_DIR / "y_parents_adaptive.csv")["cause"].astype(str).tolist()
        elif src == "fixed" and (OUTPUT_DIR / "y_parents_fixed.csv").exists():
            cand = pd.read_csv(OUTPUT_DIR / "y_parents_fixed.csv")["cause"].astype(str).tolist()
    if not cand:
        _M = A_prob_full.shape[0]
        _y_in = [(i, float(A_prob_full[i, y_index])) for i in range(_M) if i != y_index]
        _y_in.sort(key=lambda x: -x[1])
        _k = int(getattr(args, "y_parent_min_k", 20))
        cand = [names_xy[i] for i, _ in _y_in[: max(25, _k)]]
        print(f"[ATE] cand fallback to top-{len(cand)} graph parents of Y.")
    print(f"[ATE] candidate_causes count = {len(cand)}")
    for thr in thresholds:
        thr_tag = f"{thr:.3f}"
        rank_path = OUTPUT_DIR / (
            f"ate_ranking_thr_{thr_tag}.csv" if len(thresholds) > 1 else "ate_ranking.csv"
        )
        detail_path = OUTPUT_DIR / (
            f"ate_details_topk_thr_{thr_tag}.csv" if len(thresholds) > 1 else "ate_details_topk.csv"
        )
        df_rank, df_detail = ate_ranking_for_y_parents(
            docd=docd,
            A_prob=A_prob_full,
            X_xy=X_xy,
            var_specs_xy=var_specs_xy,
            names_xy=names_xy,
            y_index=y_index,
            idx_pool=idx_pool,
            edge_threshold=float(thr),
            device=device,
            topo_order=topo_order,
            base_samples=int(args.ate_base_samples),
            mc_samples=int(args.ate_mc_samples),
            topk_detail=int(args.ate_topk),
            min_candidate_edges=max(25, int(getattr(args, "top_nodes", 20))),
            candidate_causes=cand,
            ate_chunk=int(args.ate_chunk),
            resume_rank_path=OUTPUT_DIR / f"ate_ranking_thr_{thr_tag}.partial.csv",
            resume_detail_path=OUTPUT_DIR / f"ate_details_thr_{thr_tag}.partial.csv",
            resume=True,
            adaptive=bool(getattr(args, "ate_adaptive", False)),
            ci_eps=float(getattr(args, "ate_ci_eps", 0.01)),
            min_base=int(getattr(args, "ate_min_base", 300)),
            step_base=int(getattr(args, "ate_step_base", 300)),
            max_base=int(args.ate_base_samples),
        )
        df_rank.to_csv(rank_path, index=False, encoding="utf-8-sig")
        df_detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
        tmp = df_rank[["cause", "name", "ate_range"]].copy()
        tmp["threshold"] = float(thr)
        tmp["rank_at_thr"] = np.arange(1, len(tmp) + 1)
        per_thr_ranks.append(tmp)
        rank_dfs_by_thr[float(thr)] = df_rank
    if len(per_thr_ranks) > 1:
        df_all = pd.concat(per_thr_ranks, axis=0, ignore_index=True)
        agg = df_all.groupby(["cause", "name"], as_index=False).agg(
            mean_rank=("rank_at_thr", "mean"),
            median_rank=("rank_at_thr", "median"),
            std_rank=("rank_at_thr", "std"),
            mean_ate_range=("ate_range", "mean"),
            std_ate_range=("ate_range", "std"),
            n=("rank_at_thr", "count"),
        )
        agg = agg.sort_values(
            ["mean_rank", "median_rank", "mean_ate_range"], ascending=[True, True, False]
        ).reset_index(drop=True)
        agg.insert(0, "robust_rank", np.arange(1, len(agg) + 1))
        agg.to_csv(OUTPUT_DIR / "ate_ranking_robust.csv", index=False, encoding="utf-8-sig")
        final_cols = [
            "robust_rank",
            "cause",
            "name",
            "mean_ate_range",
            "mean_rank",
            "median_rank",
            "std_rank",
            "n",
        ]
        out = agg[final_cols].rename(columns={"robust_rank": "rank", "mean_ate_range": "ate_range"})
        out.to_csv(OUTPUT_DIR / "ate_ranking.csv", index=False, encoding="utf-8-sig")
    if len(per_thr_ranks) > 1:
        df_sens = summarize_threshold_sensitivity(
            per_thr_ranks, topk=int(getattr(args, "ate_sensitivity_topk", 10))
        )
        if df_sens is not None and (not df_sens.empty):
            df_sens.to_csv(
                OUTPUT_DIR / "ate_threshold_sensitivity.csv", index=False, encoding="utf-8-sig"
            )
    if int(getattr(args, "ate_bootstrap_reps", 0)) > 0:
        topk_bs = int(getattr(args, "ate_bootstrap_topk", 10))
        for thr in thresholds:
            rdf = rank_dfs_by_thr.get(float(thr))
            if rdf is None or rdf.empty:
                continue
            top_causes = rdf["cause"].head(topk_bs).astype(int).tolist()
            df_sum, df_samp = bootstrap_ate_uncertainty(
                docd=docd,
                A_prob=A_prob_full,
                X_xy=X_xy,
                var_specs_xy=var_specs_xy,
                names_xy=names_xy,
                y_index=y_index,
                idx_pool=idx_pool,
                edge_threshold=float(thr),
                candidate_causes=top_causes,
                topo_order=topo_order,
                device=device,
                reps=int(getattr(args, "ate_bootstrap_reps", 0)),
                base_samples=int(args.ate_base_samples),
                mc_samples=int(args.ate_mc_samples),
                replace=bool(getattr(args, "ate_bootstrap_replace", False)),
                ci=float(getattr(args, "ate_bootstrap_ci", 0.95)),
            )
            thr_tag = f"{float(thr):.3f}"
            df_sum.to_csv(
                OUTPUT_DIR / f"ate_bootstrap_summary_thr_{thr_tag}.csv",
                index=False,
                encoding="utf-8-sig",
            )
            df_samp.to_csv(
                OUTPUT_DIR / f"ate_bootstrap_samples_thr_{thr_tag}.csv",
                index=False,
                encoding="utf-8-sig",
            )
    if bool(getattr(args, "run_subgroup_ate", False)):
        print("\n===== [补充实验G] Subgroup Heterogeneity ATE =====", flush=True)
        name_to_idx_main = {n: i for i, n in enumerate(names_xy)}
        _ate_main_df_path = OUTPUT_DIR / "ate_ranking.csv"
        _main_ate_cand: List[int] = []
        if _ate_main_df_path.exists():
            _main_rank_df = pd.read_csv(_ate_main_df_path)
            _n_cand_sg = min(20, len(_main_rank_df))
            for _cn in _main_rank_df["cause"].head(_n_cand_sg).astype(str).tolist():
                _idx = name_to_idx_main.get(_cn)
                if _idx is None:
                    try:
                        _idx = int(_cn)
                    except Exception:
                        pass
                if _idx is not None and int(_idx) != int(y_index):
                    _main_ate_cand.append(int(_idx))
        if not _main_ate_cand:
            _yi = [
                (i, float(A_prob_full[i, y_index]))
                for i in range(A_prob_full.shape[0])
                if i != y_index
            ]
            _yi.sort(key=lambda x: -x[1])
            _main_ate_cand = [int(i) for i, _ in _yi[:20]]
        _sg_col_names: List[str] = list(getattr(args, "subgroup_col_names", []) or [])
        if not _sg_col_names:
            for _pfx in config.SUBGROUP_COL_PREFIXES.values():
                for _n in names_xy:
                    if str(_n).startswith(_pfx):
                        _sg_col_names.append(_n)
                        break
        print(f"[Subgroup ATE] 将对以下变量分组分析: {_sg_col_names}", flush=True)
        for _sg_col in _sg_col_names[:4]:
            run_subgroup_ate_analysis(
                docd=docd,
                A_prob=A_prob_full,
                X_xy=X_xy,
                var_specs_xy=var_specs_xy,
                names_xy=names_xy,
                y_index=y_index,
                idx_pool=idx_pool,
                thr_ate=float(thr_ate),
                device=device,
                topo_order=topo_order,
                candidate_causes=_main_ate_cand,
                subgroup_col_name=_sg_col,
                output_dir=OUTPUT_DIR,
                base_samples=int(getattr(args, "subgroup_ate_base_samples", 800)),
                mc_samples=int(args.ate_mc_samples),
            )
    _cross_path = str(getattr(args, "crosstask_ate_path", "")).strip()
    if _cross_path:
        _cross_p = Path(_cross_path)
        if _cross_p.exists():
            print("\n===== [补充实验I] Cross-task ATE Rank Correlation =====", flush=True)
            _current_ate_path = OUTPUT_DIR / "ate_ranking.csv"
            if _current_ate_path.exists():
                df_ate_current = pd.read_csv(_current_ate_path)
                df_ate_other = pd.read_csv(_cross_p)
                compute_crosstask_spearman(
                    ate_df1=df_ate_current,
                    ate_df2=df_ate_other,
                    task1_name=str(getattr(args, "crosstask_task1_name", "current")),
                    task2_name=str(getattr(args, "crosstask_task2_name", "other")),
                    output_dir=OUTPUT_DIR,
                )
            else:
                print("[Cross-task] 当前实验的 ate_ranking.csv 未找到，跳过。", flush=True)
        else:
            print(f"[Cross-task] 对照实验路径不存在：{_cross_path}，跳过。", flush=True)
    print("\nAll done. Outputs in:", OUTPUT_DIR.resolve())
