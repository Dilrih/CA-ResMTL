"""Model-based intervention effects and uncertainty estimates."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import VarSpec
from .graph import transitive_closure
from .models.structure import DOCDModel


@torch.inference_mode()
def docd_conditional_probs(
    docd: DOCDModel, x_batch: torch.Tensor, A_det: torch.Tensor, j: int
) -> torch.Tensor:
    E = docd.embed(x_batch)
    Aj = A_det[:, j].view(1, -1, 1)
    Pj = (E * Aj).sum(dim=1)
    zj = docd.f_list[j](Pj).squeeze(-1)
    zj = zj * docd.z_scale[j] + docd.z_bias[j]
    probs = docd.probit_heads[j](zj)
    return probs


@torch.inference_mode()
def estimate_p_y_under_do(
    docd: DOCDModel,
    A_bin: np.ndarray,
    reach: np.ndarray,
    X_xy: np.ndarray,
    var_specs_xy: List[VarSpec],
    y_index: int,
    do_var: int,
    do_value: float,
    idx_pool: np.ndarray,
    base_samples: int,
    mc_samples: int,
    device: torch.device,
    topo_order: List[int],
    rng: np.random.RandomState,
    resample_nodes: Optional[List[int]] = None,
    base_ids: Optional[np.ndarray] = None,
    return_stats: bool = False,
    chunk_size: int = 64,
) -> Any:
    """Sample descendants to estimate risk under a fixed intervention."""
    if base_ids is None:
        pool = np.array(idx_pool, dtype=int)
        n = min(int(base_samples), len(pool))
        base_ids = rng.choice(pool, size=n, replace=False)
    else:
        base_ids = np.array(base_ids, dtype=int)
        if base_samples is not None and int(base_samples) < len(base_ids):
            base_ids = base_ids[: int(base_samples)]
    mc = int(mc_samples)
    if resample_nodes is None:
        ancY = set(np.where(reach[:, y_index])[0].tolist())
        desc = set(np.where(reach[do_var, :])[0].tolist())
        nodes = sorted(list((desc & ancY) - {y_index} - {do_var}))
    else:
        nodes = list(resample_nodes)
    nodes_set = set(nodes)
    K_do = len(var_specs_xy[do_var].values)
    do_int = int(max(1, min(K_do - 1, round(float(do_value)))))
    A_det = torch.from_numpy(A_bin.astype(np.float32)).to(device)
    total_sum = 0.0
    total_sumsq = 0.0
    total_count = 0
    chunk_sz = int(chunk_size)
    if chunk_sz <= 0:
        raise ValueError("The intervention chunk size must be positive.")
    for s in range(0, len(base_ids), chunk_sz):
        bid = base_ids[s : s + chunk_sz]
        B = len(bid)
        bX = X_xy[bid].copy()
        bX[:, do_var] = do_int
        bX_rep = np.repeat(bX, mc, axis=0)
        xb_t = torch.from_numpy(bX_rep).long().to(device)
        for j in topo_order:
            if j == do_var or j == y_index:
                continue
            if j not in nodes_set:
                continue
            probs_j = docd_conditional_probs(docd, xb_t, A_det, j)
            N = probs_j.shape[0]
            K_real = probs_j.shape[1]
            cumprobs = probs_j.cumsum(dim=-1).clamp(max=1.0)
            u = torch.rand(N, 1, device=device)
            cat_real = (cumprobs < u).sum(dim=-1).clamp(0, K_real - 1)
            xb_t[:, j] = cat_real + 1
        probs_y = docd_conditional_probs(docd, xb_t, A_det, y_index)
        py_all = probs_y[:, -1].cpu().numpy()
        py_mean = py_all.reshape(B, mc).mean(axis=1)
        total_sum += float(py_mean.sum())
        total_sumsq += float((py_mean**2).sum())
        total_count += int(B)
    if return_stats:
        return (total_sum, total_sumsq, total_count)
    return float(total_sum / max(total_count, 1))


def run_subgroup_ate_analysis(
    docd: DOCDModel,
    A_prob: np.ndarray,
    X_xy: np.ndarray,
    var_specs_xy: List[VarSpec],
    names_xy: List[str],
    y_index: int,
    idx_pool: np.ndarray,
    thr_ate: float,
    device: torch.device,
    topo_order: List[int],
    candidate_causes: List[int],
    subgroup_col_name: str,
    output_dir: Path,
    base_samples: int = 1000,
    mc_samples: int = 64,
) -> Dict[str, pd.DataFrame]:
    name_to_idx = {n: i for i, n in enumerate(names_xy)}
    sg_idx = name_to_idx.get(subgroup_col_name)
    if sg_idx is None:
        print(f"[Subgroup ATE] '{subgroup_col_name}' 不在 names_xy 中，跳过。", flush=True)
        return {}
    sg_spec = var_specs_xy[sg_idx]
    sg_col = X_xy[idx_pool, sg_idx]
    unique_codes = sorted([int(c) for c in np.unique(sg_col) if c > 0])
    if len(unique_codes) < 2:
        print(
            f"[Subgroup ATE] '{subgroup_col_name}' 在样本池中只有 {len(unique_codes)} 个取值，跳过。",
            flush=True,
        )
        return {}
    results: Dict[str, pd.DataFrame] = {}
    all_rows: List[Dict] = []
    print(
        f"\n[Subgroup ATE] 分组变量: '{subgroup_col_name}'  取值数={len(unique_codes)}", flush=True
    )
    for code in unique_codes:
        group_mask = X_xy[idx_pool, sg_idx] == code
        group_idx = idx_pool[group_mask]
        n_group = len(group_idx)
        if n_group < 50:
            print(f"  code={code}: n={n_group} < 50，跳过。", flush=True)
            continue
        if 0 < int(code) < len(sg_spec.values):
            label = f"{sg_spec.values[int(code)]}"
        else:
            label = f"cat{int(code)}"
        safe_label = re.sub("[^\\w\\u4e00-\\u9fff]", "_", label)[:20]
        print(f"  [{safe_label}] n={n_group}，估计ATE...", flush=True)
        df_rank, _ = ate_ranking_for_y_parents(
            docd=docd,
            A_prob=A_prob,
            X_xy=X_xy,
            var_specs_xy=var_specs_xy,
            names_xy=names_xy,
            y_index=y_index,
            idx_pool=group_idx,
            edge_threshold=float(thr_ate),
            device=device,
            topo_order=topo_order,
            base_samples=int(min(base_samples, n_group)),
            mc_samples=mc_samples,
            topk_detail=15,
            min_candidate_edges=len(candidate_causes),
            candidate_causes=candidate_causes,
        )
        df_rank = df_rank.copy()
        df_rank["subgroup_var"] = subgroup_col_name
        df_rank["subgroup_group"] = label
        df_rank["n_group"] = n_group
        results[safe_label] = df_rank
        for _, row in df_rank.head(5).iterrows():
            all_rows.append(
                {
                    "subgroup_var": subgroup_col_name,
                    "subgroup_group": label,
                    "n_group": n_group,
                    "rank": int(row["rank"]),
                    "cause_name": str(row["name"]),
                    "ate_range": float(row["ate_range"]),
                }
            )
        safe_var = re.sub("[^\\w]", "_", subgroup_col_name)[:15]
        fname = f"subgroup_ate_{safe_var}_{safe_label}.csv"
        df_rank.to_csv(output_dir / fname, index=False, encoding="utf-8-sig")
    if all_rows:
        df_all = pd.DataFrame(all_rows)
        safe_var = re.sub("[^\\w]", "_", subgroup_col_name)[:15]
        df_all.to_csv(
            output_dir / f"subgroup_ate_summary_{safe_var}.csv", index=False, encoding="utf-8-sig"
        )
        print(f"[Subgroup ATE] 汇总保存至 subgroup_ate_summary_{safe_var}.csv", flush=True)
    return results


def ate_ranking_for_y_parents(
    docd: DOCDModel,
    A_prob: np.ndarray,
    X_xy: np.ndarray,
    var_specs_xy: List[VarSpec],
    names_xy: List[str],
    y_index: int,
    idx_pool: np.ndarray,
    edge_threshold: float,
    device: torch.device,
    topo_order: List[int],
    base_samples: int = 2000,
    mc_samples: int = 64,
    topk_detail: int = 30,
    min_candidate_edges: int = 25,
    candidate_causes: Optional[List[int]] = None,
    ate_chunk: int = 64,
    resume_rank_path: Optional[Path] = None,
    resume_detail_path: Optional[Path] = None,
    resume: bool = True,
    adaptive: bool = False,
    ci_eps: float = 0.01,
    min_base: int = 300,
    step_base: int = 300,
    max_base: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rank candidate causes by their modeled intervention risk range."""
    rng = np.random.RandomState(1234)
    A_bin = (A_prob >= float(edge_threshold)).astype(np.int32)
    reach = transitive_closure(A_bin)
    if candidate_causes is None or len(candidate_causes) == 0:
        cand = np.where(A_bin[:, y_index] > 0)[0].tolist()
        candidate_causes = [c for c in cand if c != y_index]
    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(names_xy)}
    resolved: List[int] = []
    for c in candidate_causes:
        try:
            resolved.append(int(c))
        except (ValueError, TypeError):
            idx = name_to_idx.get(str(c))
            if idx is not None:
                resolved.append(idx)
    candidate_causes = [c for c in resolved if int(c) != int(y_index)]
    if len(candidate_causes) == 0:
        return (pd.DataFrame(columns=["rank", "cause", "name", "ate_range"]), pd.DataFrame())
    max_base_eff = (
        int(max_base) if max_base is not None and int(max_base) > 0 else int(base_samples)
    )
    pool = np.array(idx_pool, dtype=int)
    max_base_eff = min(max_base_eff, len(pool))
    records = []
    details = []
    for do_var in tqdm(candidate_causes, desc="ATE vars", ncols=140):
        do_var = int(do_var)
        ancY = set(np.where(reach[:, y_index])[0].tolist())
        desc = set(np.where(reach[do_var, :])[0].tolist())
        resample_nodes = sorted(list((desc & ancY) - {y_index} - {do_var}))
        xcol = X_xy[:, do_var]
        if var_specs_xy[do_var].is_binary:
            levels = [1.0, float(var_specs_xy[do_var].n_real_cats)]
        else:
            valid = xcol[~np.isnan(xcol)]
            if len(valid) == 0:
                levels = [0.0, 0.0, 0.0]
            else:
                qs = np.quantile(valid, [0.1, 0.5, 0.9])
                levels = [float(qs[0]), float(qs[1]), float(qs[2])]
        base_master = rng.choice(pool, size=max_base_eff, replace=False)
        py_levels = []
        base_used_levels = []
        ci_hw_levels = []
        for v in levels:
            if not adaptive:
                py1 = estimate_p_y_under_do(
                    docd=docd,
                    A_bin=A_bin,
                    reach=reach,
                    X_xy=X_xy,
                    var_specs_xy=var_specs_xy,
                    y_index=y_index,
                    do_var=do_var,
                    do_value=float(v),
                    idx_pool=pool,
                    base_samples=int(base_samples),
                    mc_samples=int(mc_samples),
                    device=device,
                    topo_order=topo_order,
                    rng=rng,
                    resample_nodes=resample_nodes,
                    base_ids=base_master[: int(base_samples)],
                    return_stats=False,
                    chunk_size=ate_chunk,
                )
                py_levels.append(float(py1))
                base_used_levels.append(int(base_samples))
                ci_hw_levels.append(float("nan"))
            else:
                total_sum = 0.0
                total_sumsq = 0.0
                total_n = 0
                target = float(ci_eps)
                min_b = min(int(min_base), max_base_eff)
                step_b = max(1, int(step_base))
                ci_hw = float("inf")
                while True:
                    need = min(step_b, max_base_eff - total_n)
                    if need <= 0:
                        break
                    chunk_ids = base_master[total_n : total_n + need]
                    ssum, ssq, cnt = estimate_p_y_under_do(
                        docd=docd,
                        A_bin=A_bin,
                        reach=reach,
                        X_xy=X_xy,
                        var_specs_xy=var_specs_xy,
                        y_index=y_index,
                        do_var=do_var,
                        do_value=float(v),
                        idx_pool=pool,
                        base_samples=int(len(chunk_ids)),
                        mc_samples=int(mc_samples),
                        device=device,
                        topo_order=topo_order,
                        rng=rng,
                        resample_nodes=resample_nodes,
                        base_ids=chunk_ids,
                        return_stats=True,
                        chunk_size=ate_chunk,
                    )
                    total_sum += float(ssum)
                    total_sumsq += float(ssq)
                    total_n += int(cnt)
                    mean = total_sum / max(total_n, 1)
                    if total_n >= 2:
                        var = max(total_sumsq / total_n - mean * mean, 0.0) * (
                            total_n / max(total_n - 1, 1)
                        )
                        se = math.sqrt(var / total_n)
                        ci_hw = 1.96 * se
                    if total_n >= min_b and ci_hw <= target:
                        break
                py_levels.append(float(total_sum / max(total_n, 1)))
                base_used_levels.append(int(total_n))
                ci_hw_levels.append(float(ci_hw if np.isfinite(ci_hw) else float("nan")))
        ate_range = float(max(py_levels) - min(py_levels))
        records.append({"cause": do_var, "name": names_xy[do_var], "ate_range": ate_range})
        for v, py1, nb, ci in zip(levels, py_levels, base_used_levels, ci_hw_levels):
            details.append(
                {
                    "cause": do_var,
                    "name": names_xy[do_var],
                    "do_value": float(v),
                    "p_y": float(py1),
                    "n_base_used": int(nb),
                    "ci_halfwidth_95": float(ci),
                }
            )
    df_rank = pd.DataFrame(records).sort_values("ate_range", ascending=False).reset_index(drop=True)
    df_rank.insert(0, "rank", np.arange(1, len(df_rank) + 1))
    df_detail = pd.DataFrame(details)
    if topk_detail > 0 and len(df_rank) > 0:
        top_vars = set(df_rank["cause"].head(int(topk_detail)).tolist())
        df_detail = df_detail[df_detail["cause"].isin(top_vars)].reset_index(drop=True)
    return (df_rank, df_detail)


@torch.no_grad()
def bootstrap_ate_uncertainty(
    *,
    docd: DOCDModel,
    A_prob: np.ndarray,
    X_xy: np.ndarray,
    var_specs_xy: List[VarSpec],
    names_xy: List[str],
    y_index: int,
    idx_pool: np.ndarray,
    edge_threshold: float,
    candidate_causes: List[int],
    topo_order: List[int],
    device: torch.device,
    reps: int,
    base_samples: int,
    mc_samples: int,
    replace: bool,
    ci: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate uncertainty in the intervention risk range by resampling rows."""
    rng = np.random.RandomState(2025)
    pool = np.array(idx_pool, dtype=int)
    if pool.size == 0:
        return (pd.DataFrame(), pd.DataFrame())
    base_samples_eff = int(min(int(base_samples), int(pool.size)))
    A_bin = (A_prob >= float(edge_threshold)).astype(np.int32)
    reach = transitive_closure(A_bin)
    pre: List[Tuple[int, List[float], List[int]]] = []
    for do_var in candidate_causes:
        do_var = int(do_var)
        if do_var == int(y_index):
            continue
        ancY = set(np.where(reach[:, y_index])[0].tolist())
        desc = set(np.where(reach[do_var, :])[0].tolist())
        resample_nodes = sorted(list((desc & ancY) - {y_index} - {do_var}))
        xcol = X_xy[:, do_var].astype(float)
        if var_specs_xy[do_var].is_binary:
            levels = [1.0, float(var_specs_xy[do_var].n_real_cats)]
        else:
            valid = xcol[~np.isnan(xcol)]
            if len(valid) == 0:
                levels = [1.0, 1.0, 1.0]
            else:
                qs = np.quantile(valid, [0.1, 0.5, 0.9])
                levels = [float(qs[0]), float(qs[1]), float(qs[2])]
        pre.append((do_var, levels, resample_nodes))
    rows = []
    for r in tqdm(range(int(reps)), desc="ATE bootstrap reps", ncols=120):
        base_ids = rng.choice(pool, size=base_samples_eff, replace=bool(replace))
        for do_var, levels, res_nodes in pre:
            py_levels = []
            for v in levels:
                py = estimate_p_y_under_do(
                    docd=docd,
                    A_bin=A_bin,
                    reach=reach,
                    X_xy=X_xy,
                    var_specs_xy=var_specs_xy,
                    y_index=int(y_index),
                    do_var=int(do_var),
                    do_value=float(v),
                    idx_pool=pool,
                    base_samples=base_samples_eff,
                    mc_samples=int(mc_samples),
                    device=device,
                    topo_order=topo_order,
                    rng=rng,
                    resample_nodes=res_nodes,
                    base_ids=base_ids,
                    return_stats=False,
                )
                py_levels.append(float(py))
            rows.append(
                {
                    "rep": int(r + 1),
                    "cause": int(do_var),
                    "name": names_xy[int(do_var)],
                    "threshold": float(edge_threshold),
                    "ate_range": float(max(py_levels) - min(py_levels)),
                }
            )
    df_samples = pd.DataFrame(rows)
    if df_samples.empty:
        return (pd.DataFrame(), pd.DataFrame())
    alpha = 1.0 - float(ci)
    lo = alpha / 2.0
    hi = 1.0 - alpha / 2.0
    df_sum = (
        df_samples.groupby(["cause", "name", "threshold"], as_index=False)
        .agg(
            mean_ate_range=("ate_range", "mean"),
            std_ate_range=("ate_range", "std"),
            q_lo=("ate_range", lambda x: float(np.quantile(x, lo))),
            q_hi=("ate_range", lambda x: float(np.quantile(x, hi))),
            n=("ate_range", "count"),
        )
        .sort_values("mean_ate_range", ascending=False)
        .reset_index(drop=True)
    )
    df_sum.insert(0, "rank_by_mean", np.arange(1, len(df_sum) + 1))
    df_sum["ci_level"] = float(ci)
    df_sum["replace"] = bool(replace)
    df_sum["base_samples"] = int(base_samples_eff)
    df_sum["mc_samples"] = int(mc_samples)
    return (df_sum, df_samples)
