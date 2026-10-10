"""Graph constraints, reachability, and structure exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mutual_info_score
from tqdm import tqdm

from . import config
from .data import EncodedData, VarSpec


def transitive_closure(A_bin: np.ndarray) -> np.ndarray:
    """Return directed reachability, including each node itself."""
    A = np.asarray(A_bin)
    n = A.shape[0]
    reach = A != 0
    np.fill_diagonal(reach, True)
    for k in range(n):
        if reach[:, k].any():
            reach = reach | reach[:, [k]] & reach[[k], :]
    return reach


def gumbel_sigmoid(logits: torch.Tensor, tau: float, hard: bool = False) -> torch.Tensor:
    U = torch.rand_like(logits).clamp(1e-06, 1.0 - 1e-06)
    g = -torch.log(-torch.log(U))
    y = torch.sigmoid((logits + g) / max(1e-06, tau))
    if hard:
        y_hard = (y > 0.5).float()
        y = (y_hard - y).detach() + y
    return y


def factorial_float(k: int) -> float:
    out = 1.0
    for i in range(2, k + 1):
        out *= float(i)
    return out


def acyclicity_hutchinson_expm(
    W: torch.Tensor,
    n_trace: int = config.DEFAULT_HUTCH_SAMPLES,
    series_k: int = config.DEFAULT_HUTCH_SERIES_K,
    scale_by_dim: bool = True,
    clamp_nonneg: bool = True,
) -> torch.Tensor:
    """Estimate the matrix-exponential DAG constraint using random probes."""
    d = int(W.shape[0])
    M = W * W
    if scale_by_dim:
        M = M / float(max(1, d))
    M = M.double()
    facts = [factorial_float(k) for k in range(1, int(series_k) + 1)]
    acc = torch.zeros((), device=W.device, dtype=torch.double)
    for _ in range(int(n_trace)):
        v = torch.empty((d,), device=W.device, dtype=torch.double).bernoulli_(0.5)
        v = v * 2.0 - 1.0
        u = v
        for k in range(1, int(series_k) + 1):
            u = M @ u
            acc = acc + v @ u / float(facts[k - 1])
    tr_expm_est = float(d) + acc / float(max(1, int(n_trace)))
    h = (tr_expm_est - float(d)).float()
    if clamp_nonneg:
        h = h.clamp_min(0.0)
    return h


def build_allowed_mask_with_priors(
    var_specs: List[VarSpec], y_index: int, use_question_order_prior: bool = True
) -> np.ndarray:
    """Apply question-order constraints and prevent outgoing edges from Y."""
    M = len(var_specs)
    mask = np.ones((M, M), dtype=np.float32)
    np.fill_diagonal(mask, 0.0)
    mask[y_index, :] = 0.0
    mask[y_index, y_index] = 0.0
    mask[:, y_index] = 1.0
    mask[y_index, y_index] = 0.0
    if use_question_order_prior:
        keys = [vs.q_key for vs in var_specs]
        for i in range(M):
            for j in range(M):
                if i == j:
                    continue
                if i == y_index:
                    mask[i, j] = 0.0
                    continue
                if j == y_index:
                    continue
                if keys[i] == (10**9, 10**9) or keys[j] == (10**9, 10**9):
                    continue
                if keys[i] > keys[j]:
                    mask[i, j] = 0.0
    return mask


def build_reachability(A_bin: np.ndarray) -> np.ndarray:
    M = A_bin.shape[0]
    reach = A_bin.astype(bool).copy()
    np.fill_diagonal(reach, True)
    for k in range(M):
        reach = reach | reach[:, [k]] & reach[[k], :]
    return reach


def build_causal_attention_mask(
    A_prob_x: np.ndarray, threshold: float, transpose_for_query: bool = True
) -> np.ndarray:
    A_bin = (A_prob_x >= threshold).astype(np.int32)
    np.fill_diagonal(A_bin, 0)
    reach = build_reachability(A_bin)
    mask = reach.T if bool(transpose_for_query) else reach
    np.fill_diagonal(mask, True)
    return mask.astype(np.bool_)


def build_causal_attention_bias(
    A_prob_x: np.ndarray,
    threshold: float,
    prior_strength: float = config.DEFAULT_ATTN_PRIOR_STRENGTH,
    nonancestor_penalty: float = config.DEFAULT_ATTN_NONANCESTOR_PENALTY,
) -> np.ndarray:
    anc_mask = build_causal_attention_mask(A_prob_x, threshold=threshold, transpose_for_query=True)
    direct = np.asarray(A_prob_x, dtype=np.float32).copy().T
    np.fill_diagonal(direct, 0.0)
    bias = np.where(anc_mask, 0.0, float(nonancestor_penalty)).astype(np.float32)
    pos_vals = direct[direct > 0]
    if pos_vals.size > 0:
        scale = float(np.percentile(pos_vals, 95))
        scale = max(scale, 1e-06)
        bias = bias + float(prior_strength) * np.clip(direct / scale, 0.0, 1.5).astype(np.float32)
    np.fill_diagonal(bias, 0.0)
    return bias.astype(np.float32)


def build_attention_components(
    A_prob_x: np.ndarray,
    threshold: float,
    mode: str = config.DEFAULT_ATTN_PRIOR_MODE,
    prior_strength: float = config.DEFAULT_ATTN_PRIOR_STRENGTH,
    nonancestor_penalty: float = config.DEFAULT_ATTN_NONANCESTOR_PENALTY,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    anc_mask = build_causal_attention_mask(A_prob_x, threshold=threshold, transpose_for_query=True)
    mode = str(mode).lower().strip()
    if mode == "hard":
        return (anc_mask, None)
    if mode == "hybrid":
        return (
            anc_mask,
            build_causal_attention_bias(
                A_prob_x,
                threshold=threshold,
                prior_strength=prior_strength,
                nonancestor_penalty=0.0,
            ),
        )
    return (
        np.ones_like(anc_mask, dtype=np.bool_),
        build_causal_attention_bias(
            A_prob_x,
            threshold=threshold,
            prior_strength=prior_strength,
            nonancestor_penalty=nonancestor_penalty,
        ),
    )


def choose_threshold_by_target_degree(
    A_prob: np.ndarray, target_degree: int, min_thr: float = 0.0, max_thr: float = 0.95
) -> float:
    A_prob.shape[0]
    off = A_prob.copy()
    np.fill_diagonal(off, 0.0)
    vals = off.reshape(-1)
    vals = vals[(vals > 0) & np.isfinite(vals)]
    if vals.size == 0:
        return float(min_thr)
    lo, hi = (float(min_thr), float(max_thr))
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        A_bin = (off >= mid).astype(np.int32)
        avg_deg = float(A_bin.sum(axis=1).mean())
        if avg_deg > float(target_degree):
            lo = mid
        else:
            hi = mid
    thr = float(hi)
    thr = max(float(min_thr), min(float(max_thr), thr))
    return thr


def extract_y_parents_adaptive(
    A_prob_full: np.ndarray,
    names_xy: List[str],
    y_index: int,
    fixed_threshold: float,
    min_k: int = config.DEFAULT_Y_PARENT_MIN_K,
    max_k: int = config.DEFAULT_Y_PARENT_MAX_K,
    min_edge: float = config.DEFAULT_Y_PARENT_MIN_EDGE,
) -> Dict[str, Any]:
    y_in = []
    for i in range(A_prob_full.shape[0]):
        if i == y_index:
            continue
        y_in.append((i, float(A_prob_full[i, y_index])))
    y_in.sort(key=lambda x: -x[1])
    probs = np.array([w for _, w in y_in], dtype=np.float64)
    stats = {
        "y_in_max": float(np.max(probs)) if probs.size else 0.0,
        "y_in_mean": float(np.mean(probs)) if probs.size else 0.0,
        "y_in_q90": float(np.quantile(probs, 0.9)) if probs.size else 0.0,
        "y_in_q95": float(np.quantile(probs, 0.95)) if probs.size else 0.0,
        "y_in_q99": float(np.quantile(probs, 0.99)) if probs.size else 0.0,
    }

    def _parents_by_thr(thr: float) -> List[Tuple[str, float]]:
        out = []
        for i, w in y_in:
            if w >= thr:
                out.append((names_xy[i], w))
        out.sort(key=lambda x: -x[1])
        return out

    fixed = float(fixed_threshold)
    fixed_parents = _parents_by_thr(fixed)
    if len(fixed_parents) >= int(min_k):
        df = pd.DataFrame(fixed_parents, columns=["cause", "edge_prob"])
        return {
            "method": "fixed_threshold",
            "threshold_used": fixed,
            "parents_df": df,
            "y_in_stats": stats,
        }
    k_use = int(min_k)
    kth = (
        float(probs[k_use - 1]) if probs.size >= k_use else float(probs[-1]) if probs.size else 0.0
    )
    kth = max(float(min_edge), kth)
    topk_parents = _parents_by_thr(kth)
    elbow_thr = kth
    elbow_k = k_use
    if probs.size >= int(min_k) + 2:
        k_hi = min(int(max_k), int(probs.size) - 1)
        k_lo = int(min_k)
        gaps = []
        for k in range(k_lo, k_hi):
            gaps.append((k, float(probs[k - 1] - probs[k])))
        k_star, gap_star = max(gaps, key=lambda x: x[1])
        if gap_star > 0.5 * float(np.std(probs[:k_hi]) + 1e-08):
            elbow_k = int(k_star)
            elbow_thr = max(float(min_edge), float(probs[elbow_k - 1]))
    elbow_parents = _parents_by_thr(elbow_thr)
    cand = [
        ("topk", kth, topk_parents, abs(len(topk_parents) - int(min_k))),
        ("elbow", elbow_thr, elbow_parents, abs(len(elbow_parents) - int(min_k))),
    ]
    cand.sort(key=lambda x: (x[3], -x[1]))
    method, thr_used, parents, _ = cand[0]
    df = pd.DataFrame(parents, columns=["cause", "edge_prob"])
    return {
        "method": method,
        "threshold_used": float(thr_used),
        "parents_df": df,
        "y_in_stats": stats,
        "topk_threshold": float(kth),
        "elbow_threshold": float(elbow_thr),
        "elbow_k": int(elbow_k),
        "fixed_threshold": float(fixed_threshold),
    }


def build_xy_for_docd(
    enc_x: EncodedData, y: np.ndarray, y_name: str
) -> Tuple[np.ndarray, List[VarSpec], List[str], int]:
    y_disc = (y.astype(np.int64) + 1).reshape(-1, 1)
    X_xy = np.concatenate([enc_x.X, y_disc], axis=1)
    y_spec = VarSpec(
        name=f"Y::{y_name}",
        values=["<phi>", "0", "1"],
        is_numeric=True,
        is_ordinal=True,
        q_key=(10**9, 10**9),
    )
    var_specs_xy = enc_x.var_specs + [y_spec]
    names_xy = enc_x.col_names + [y_spec.name]
    y_index = len(names_xy) - 1
    return (X_xy, var_specs_xy, names_xy, y_index)


def mutual_info_screening(
    X: np.ndarray,
    y: np.ndarray,
    col_names: List[str],
    idx_train: np.ndarray,
    topk: int,
    min_mi: float,
) -> pd.DataFrame:
    ys = y[idx_train]
    rows = []
    for j in tqdm(range(X.shape[1]), desc="MI Screening"):
        xj = X[idx_train, j]
        mi = mutual_info_score(xj, ys)
        rows.append((col_names[j], float(mi)))
    df = (
        pd.DataFrame(rows, columns=["var", "mi"])
        .sort_values("mi", ascending=False)
        .reset_index(drop=True)
    )
    df["selected"] = False
    sel = df[df["mi"] >= float(min_mi)].head(int(topk))
    df.loc[sel.index, "selected"] = True
    return df


def save_causal_graph(
    A_prob: np.ndarray, names: List[str], edge_threshold: float, output_dir: Path
) -> None:
    """Write the adjacency matrix and thresholded edge list."""
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A_prob, index=names, columns=names).to_csv(
        OUTPUT_DIR / "causal_adj_full.csv", encoding="utf-8-sig"
    )
    edges = []
    M = A_prob.shape[0]
    thr = float(edge_threshold)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            w = float(A_prob[i, j])
            if w >= thr:
                edges.append((names[i], names[j], w))
    edges.sort(key=lambda x: -x[2])
    pd.DataFrame(edges, columns=["cause", "effect", "prob"]).to_csv(
        OUTPUT_DIR / "causal_edges_full_top.csv", index=False, encoding="utf-8-sig"
    )


def _infer_focus_from_graph(A_prob_x: Optional[np.ndarray], topk: int) -> List[int]:
    if A_prob_x is None:
        return []
    M = int(A_prob_x.shape[0])
    deg = A_prob_x.sum(axis=0) + A_prob_x.sum(axis=1)
    order = np.argsort(-deg)[: min(int(topk), M)]
    return sorted((int(i) for i in order.tolist()))


def load_causal_focus_indices(
    output_dir: Optional[Path],
    var_specs: List[VarSpec],
    A_prob_full: Optional[np.ndarray] = None,
    topk: int = config.DEFAULT_CAUSAL_TOPK,
) -> List[int]:
    name_to_idx = {str(v.name).strip(): i for i, v in enumerate(var_specs)}
    picked: List[int] = []
    if output_dir is not None:
        for fname in ["y_parents_stable.csv", "y_parents_adaptive.csv"]:
            fp = Path(output_dir) / fname
            if fp.exists():
                try:
                    dfp = pd.read_csv(fp)
                    if "cause" in dfp.columns:
                        cand_names = dfp["cause"].astype(str).tolist()
                    else:
                        cand_names = dfp.iloc[:, 0].astype(str).tolist()
                    for nm in cand_names:
                        key = str(nm).strip()
                        if key in name_to_idx:
                            picked.append(int(name_to_idx[key]))
                    if len(picked) >= int(topk):
                        break
                except Exception:
                    pass
    picked = list(dict.fromkeys(picked))
    if len(picked) < max(8, min(16, int(topk // 2))):
        A_prob_x = None
        if A_prob_full is not None:
            M = len(var_specs)
            A_prob_x = A_prob_full[:M, :M]
        picked.extend(_infer_focus_from_graph(A_prob_x, topk=int(topk)))
        picked = list(dict.fromkeys(picked))
    if len(picked) > int(topk):
        picked = picked[: int(topk)]
    return sorted((int(i) for i in picked))


def _build_random_dag_mask(ref_mask: np.ndarray, seed: int = 999) -> np.ndarray:
    M = ref_mask.shape[0]
    off_diag = ref_mask.copy()
    np.fill_diagonal(off_diag, 0)
    density = float(off_diag.sum()) / max(1.0, float(M * (M - 1)))
    rng = np.random.RandomState(seed)
    rand_adj = np.zeros((M, M), dtype=np.int32)
    for i in range(1, M):
        for j in range(i):
            if rng.random() < density:
                rand_adj[j, i] = 1
    rand_reach = build_reachability(rand_adj).T.astype(np.bool_)
    np.fill_diagonal(rand_reach, True)
    return rand_reach
