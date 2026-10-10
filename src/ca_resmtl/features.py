"""Training-fitted interactions and questionnaire aggregates."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import mutual_info_score


class CausalFeatureEngineer:
    """Fit interaction and aggregate features using training rows only."""

    GRADE_KEYWORDS = {
        "高中以上": "senior_hs",
        "初中以上": "junior_above",
        "大学生": "university",
        "女生回答": "female",
    }
    GRADE_PREFIX_GROUPS = {
        "senior_hs": [
            "23、",
            "24、",
            "25、",
            "26、",
            "27、",
            "28、",
            "29、",
            "30、",
            "31、",
            "64、",
            "66、",
            "67、",
            "85、",
            "90、",
        ],
        "junior_above": ["59、", "60、", "61、"],
        "university": ["32、", "88、", "89、"],
        "female": ["20、", "21、"],
    }
    DOMAIN_GROUPS = {
        "family_adversity": ["44、", "34、（5）", "34、(5）", "34、(（5）", "45、", "46、"],
        "bullying_exposure": [
            "40、（3）",
            "40、(3）",
            "40、(（3）",
            "40、（4）",
            "40、(4）",
            "40、(（4）",
            "42、",
        ],
        "chronic_health": ["59、", "54、"],
        "perceived_wellbeing": ["56、", "57、"],
        "sleep_lifestyle": [
            "63、（1）",
            "63、(1）",
            "63、(（1）",
            "63、（2）",
            "63、(2）",
            "63、(（2）",
            "18、",
        ],
    }

    def __init__(
        self,
        stable_parent_names: List[str],
        col_names: List[str],
        n_interaction_pairs: int = 80,
        seed: int = 42,
    ):
        self.stable_parent_names = [str(n) for n in stable_parent_names]
        self.col_names = [str(n) for n in col_names]
        self.n_interaction_pairs = int(n_interaction_pairs)
        self.seed = seed
        self.name2idx: Dict[str, int] = {n: i for i, n in enumerate(self.col_names)}
        self.parent_indices: List[int] = []
        for nm in self.stable_parent_names:
            if nm in self.name2idx:
                self.parent_indices.append(self.name2idx[nm])
        self.parent_indices = sorted(set(self.parent_indices))
        self.grade_indices: Dict[str, List[int]] = {k: [] for k in self.GRADE_KEYWORDS.values()}
        for gtype in ["senior_hs", "junior_above", "university", "female"]:
            if gtype not in self.grade_indices:
                self.grade_indices[gtype] = []
        for i, nm in enumerate(self.col_names):
            for kw, gtype in self.GRADE_KEYWORDS.items():
                if kw in nm:
                    self.grade_indices[gtype].append(i)
                    break
            else:
                for gtype, prefixes in self.GRADE_PREFIX_GROUPS.items():
                    if any((nm.startswith(pfx) for pfx in prefixes)):
                        self.grade_indices[gtype].append(i)
                        break
        for gtype in self.grade_indices:
            self.grade_indices[gtype] = sorted(set(self.grade_indices[gtype]))
        self.domain_indices: Dict[str, List[int]] = {}
        for dom, prefixes in self.DOMAIN_GROUPS.items():
            idxs = []
            for nm in self.col_names:
                for pfx in prefixes:
                    if nm.startswith(pfx):
                        if nm in self.name2idx:
                            idxs.append(self.name2idx[nm])
                        break
            if idxs:
                self.domain_indices[dom] = sorted(set(idxs))
        self.interaction_pairs: List[Tuple[int, int]] = []
        self.feature_names: List[str] = []
        self._fitted: bool = False
        for gtype, idxs in self.grade_indices.items():
            if idxs:
                print(
                    f"[CausalFE] {gtype}: {len(idxs)} grade/gender-restricted columns detected",
                    flush=True,
                )

    def _phi_rate(self, X: np.ndarray, col_indices: List[int]) -> np.ndarray:
        if not col_indices:
            return np.zeros(X.shape[0], dtype=np.float32)
        mat = X[:, col_indices].astype(float)
        return (mat == 0).mean(axis=1).astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, idx_train: np.ndarray) -> "CausalFeatureEngineer":
        X_tr = X[idx_train]
        y_tr = y[idx_train]
        if len(X_tr) == 0:
            raise ValueError("Feature engineering requires nonempty training rows.")
        self._medians = np.ones(X.shape[1], dtype=float)
        self._column_max = np.ones(X.shape[1], dtype=float)
        self._observed = np.zeros(X.shape[1], dtype=bool)
        for column in range(X.shape[1]):
            values = X_tr[:, column].astype(float)
            valid = values[(values > 0) & np.isfinite(values)]
            if len(valid):
                self._medians[column] = np.median(valid)
                self._column_max[column] = valid.max()
                self._observed[column] = True
        parents = self.parent_indices
        if len(parents) >= 2:
            pairs_mi: List[Tuple[Tuple[int, int], float]] = []
            for ii in range(len(parents)):
                for jj in range(ii + 1, len(parents)):
                    pi, pj = (parents[ii], parents[jj])
                    xi = X_tr[:, pi].astype(float)
                    xj = X_tr[:, pj].astype(float)
                    xi_f = np.where(
                        xi > 0, xi, float(np.median(xi[xi > 0])) if (xi > 0).any() else 1.0
                    )
                    xj_f = np.where(
                        xj > 0, xj, float(np.median(xj[xj > 0])) if (xj > 0).any() else 1.0
                    )
                    np.nan_to_num(xi_f, nan=1.0, copy=False)
                    np.nan_to_num(xj_f, nan=1.0, copy=False)
                    product = xi_f * xj_f
                    product_disc = np.digitize(
                        product, bins=np.percentile(product, [20, 40, 60, 80])
                    ).clip(0, 4)
                    try:
                        mi = float(mutual_info_score(y_tr, product_disc))
                    except Exception:
                        mi = 0.0
                    pairs_mi.append(((pi, pj), mi))
            pairs_mi.sort(key=lambda x: -x[1])
            self.interaction_pairs = [p for p, _ in pairs_mi[: self.n_interaction_pairs]]
        self._pair_bounds = {}
        for pi, pj in self.interaction_pairs:
            xi = np.where(X_tr[:, pi] > 0, X_tr[:, pi], self._medians[pi])
            xj = np.where(X_tr[:, pj] > 0, X_tr[:, pj], self._medians[pj])
            product = xi * xj
            ratio = xi / (xj + 1.0)
            self._pair_bounds[pi, pj] = (product.min(), product.max(), ratio.min(), ratio.max())
        names = []
        for gtype in sorted(self.grade_indices.keys()):
            if self.grade_indices[gtype]:
                names.append(f"GRADE_{gtype}_skip_rate")
                names.append(f"GRADE_{gtype}_answered")
        for i, j in self.interaction_pairs:
            ni = self.col_names[i][:16].rstrip("、").strip()
            nj = self.col_names[j][:16].rstrip("、").strip()
            names.append(f"PROD_{ni}×{nj}")
            names.append(f"RATIO_{ni}/{nj}")
        for dom in sorted(self.domain_indices.keys()):
            names.append(f"DOM_{dom}_mean")
            names.append(f"DOM_{dom}_miss")
        self.feature_names = names
        self._fitted = True
        n_grade = sum((2 for g in self.grade_indices.values() if g))
        n_pair = len(self.interaction_pairs) * 2
        n_dom = len(self.domain_indices) * 2
        total = n_grade + n_pair + n_dom
        print(
            f"[CausalFE] Fitted: {n_grade} grade/gender + {n_pair} pair (×{len(self.interaction_pairs)} pairs) + {n_dom} domain = {total} features",
            flush=True,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")
        n = X.shape[0]
        feats: List[np.ndarray] = []
        for gtype in sorted(self.grade_indices.keys()):
            idxs = self.grade_indices[gtype]
            if not idxs:
                continue
            skip_rate = self._phi_rate(X, idxs)
            answered = (1.0 - skip_rate).astype(np.float32)
            feats.append(skip_rate)
            feats.append(answered)
        for pi, pj in self.interaction_pairs:
            xi = X[:, pi].astype(float)
            xj = X[:, pj].astype(float)
            # Reuse training statistics so a response is independent of its evaluation batch.
            xi_f = np.where(xi > 0, xi, self._medians[pi])
            xj_f = np.where(xj > 0, xj, self._medians[pj])
            product = xi_f * xj_f
            p_min, p_max, r_min, r_max = self._pair_bounds[pi, pj]
            feats.append(((product - p_min) / max(p_max - p_min, 1.0)).astype(np.float32))
            ratio = xi_f / (xj_f + 1.0)
            feats.append(((ratio - r_min) / max(r_max - r_min, 1.0)).astype(np.float32))
        for dom in sorted(self.domain_indices.keys()):
            idxs = self.domain_indices[dom]
            if not idxs:
                feats.append(np.zeros(n, dtype=np.float32))
                feats.append(np.zeros(n, dtype=np.float32))
                continue
            group_vals, group_miss = ([], [])
            for idx in idxs:
                col = X[:, idx].astype(float)
                if not self._observed[idx]:
                    continue
                col_norm = np.where(
                    col > 0, (col - 1.0) / max(self._column_max[idx] - 1.0, 1.0), 0.0
                )
                group_vals.append(col_norm)
                group_miss.append((col == 0).astype(float))
            if group_vals:
                feats.append(np.stack(group_vals, axis=1).mean(axis=1).astype(np.float32))
                feats.append(np.stack(group_miss, axis=1).mean(axis=1).astype(np.float32))
            else:
                feats.append(np.zeros(n, dtype=np.float32))
                feats.append(np.zeros(n, dtype=np.float32))
        if not feats:
            return np.zeros((n, 1), dtype=np.float32)
        return np.stack(feats, axis=1).astype(np.float32)

    def fit_transform(self, X: np.ndarray, y: np.ndarray, idx_train: np.ndarray) -> np.ndarray:
        self.fit(X, y, idx_train)
        return self.transform(X)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)
