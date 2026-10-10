"""Questionnaire cleaning, encoding, and domain-level splitting."""

from __future__ import annotations

import glob
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config


def is_meta_col(col: str) -> bool:
    for pat in config.META_COL_PATTERNS:
        if re.search(pat, col):
            return True
    return False


def is_questionnaire_artifact_col(col: str) -> bool:
    s = str(col).strip()
    for pat in config.QUESTIONNAIRE_ARTIFACT_PATTERNS:
        if re.search(pat, s):
            return True
    return False


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize blank cells and questionnaire missing-value codes."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.replace("^\\s*$", np.nan, regex=True)
    for v in config.COMMON_NA_SENTINELS:
        df = df.replace(v, np.nan)
        df = df.replace(str(v), np.nan)
    return df


def load_all_excels(data_glob: str) -> pd.DataFrame:
    """Read one workbook per domain; the filename stem identifies the domain."""
    files = sorted([Path(p) for p in glob.glob(data_glob)])
    if not files:
        raise FileNotFoundError(f"没有找到任何xlsx文件：{data_glob}")
    dfs = []
    for fp in tqdm(files, desc="Loading Excel files"):
        df = pd.read_excel(fp)
        df["__province__"] = fp.stem
        df["__source_file__"] = str(fp)
        dfs.append(df)
    return pd.concat(dfs, axis=0, ignore_index=True)


def split_location_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    if col not in df.columns:
        return df
    s = df[col].astype(str).replace("nan", np.nan)
    s = s.str.replace("[—–－]+", "-", regex=True)
    parts = s.str.split("[-_/\\\\s]+", expand=True)
    if parts.shape[1] >= 1:
        df[col + "__省"] = parts[0].replace("nan", np.nan)
    if parts.shape[1] >= 2:
        df[col + "__市"] = parts[1].replace("nan", np.nan)
    if parts.shape[1] >= 3:
        df[col + "__区县"] = parts[2].replace("nan", np.nan)
    return df


def drop_meta_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = []
    hard_drop_names = {"总分", "量表总分", "得分"}
    for c in df.columns:
        sc = str(c).strip()
        if c in ["__province__", "__source_file__"]:
            keep_cols.append(c)
            continue
        if is_meta_col(sc) or is_questionnaire_artifact_col(sc) or sc in hard_drop_names:
            continue
        keep_cols.append(c)
    return df[keep_cols].copy()


def drop_group_leakage_location_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    dropped: List[str] = []
    for c in list(df.columns):
        s = str(c)
        if s in ["__province__", "__source_file__"]:
            continue
        if "目前居住" in s or s.endswith("__省") or s.endswith("__市") or s.endswith("__区县"):
            dropped.append(c)
    if dropped:
        df = df.drop(columns=sorted(set(dropped)), errors="ignore")
    return (df, {"dropped": sorted(set(map(str, dropped)))})


def apply_schema_preprocessing(
    df: pd.DataFrame, schema_rules: Optional[Dict[str, Any]]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    report: Dict[str, Any] = {"dropped_by_schema": [], "range_sanitized": {}}
    if not schema_rules:
        return (df, report)
    drop_cols: List[str] = []
    for c in list(df.columns):
        if c in ["__province__", "__source_file__"]:
            continue
        rule = resolve_variable_rule(str(c), schema_rules)
        if not rule:
            continue
        if bool(rule.get("drop", False)) or is_questionnaire_artifact_col(str(c)):
            drop_cols.append(c)
            continue
        rmin = rule.get("min", None)
        rmax = rule.get("max", None)
        rtype = str(rule.get("type", "")).strip().lower()
        if (rmin is None and rmax is None) and rtype not in {
            "ordinal",
            "continuous",
            "binary",
            "nominal",
        }:
            continue
        sn = pd.to_numeric(df[c], errors="coerce")
        if sn.notna().sum() == 0:
            continue
        mask = pd.Series(True, index=df.index)
        if rmin is not None:
            mask &= (sn >= float(rmin)) | sn.isna()
        if rmax is not None:
            mask &= (sn <= float(rmax)) | sn.isna()
        invalid = ~mask & sn.notna()
        n_bad = int(invalid.sum())
        if n_bad > 0:
            sn = sn.where(mask, np.nan)
            report["range_sanitized"][str(c)] = {
                "n_invalid": n_bad,
                "min": None if rmin is None else float(rmin),
                "max": None if rmax is None else float(rmax),
            }
        if rmin is not None or rmax is not None or rtype in {"ordinal", "continuous", "binary"}:
            df[c] = sn
    if drop_cols:
        df = df.drop(columns=sorted(set(drop_cols)), errors="ignore")
        report["dropped_by_schema"] = sorted(set(map(str, drop_cols)))
    return (df, report)


def drop_sparse_or_constant_columns(
    df: pd.DataFrame,
    min_nonmissing_per_feature: int = 15,
    max_missing_rate_per_feature: float = 0.985,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = df.copy()
    dropped_sparse: List[str] = []
    dropped_constant: List[str] = []
    keep = []
    for c in df.columns:
        if c in ["__province__", "__source_file__"]:
            keep.append(c)
            continue
        s = df[c]
        nonmissing = int(s.notna().sum())
        miss_rate = float(s.isna().mean())
        nunique = int(s.dropna().nunique())
        if nonmissing < int(min_nonmissing_per_feature) or miss_rate > float(
            max_missing_rate_per_feature
        ):
            dropped_sparse.append(str(c))
            continue
        if nunique <= 1:
            dropped_constant.append(str(c))
            continue
        keep.append(c)
    return (
        df[keep].copy(),
        {
            "dropped_sparse": sorted(set(dropped_sparse)),
            "dropped_constant": sorted(set(dropped_constant)),
        },
    )


def find_label_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    for c in df.columns:
        name = str(c)
        for k in keywords:
            if k in name:
                return c
    return None


def parse_question_key(col_name: str) -> Tuple[int, int]:
    s = str(col_name).strip()
    m = re.match("^\\s*(\\d+)\\s*[、\\.\\)]", s)
    if not m:
        return (10**9, 10**9)
    q = int(m.group(1))
    sub = 0
    m2 = re.search("[（(]\\s*[（(]?\\s*(\\d+)\\s*[）)]?\\s*[）)]", s)
    if m2:
        sub = int(m2.group(1))
    return (q, sub)


def hash_bucket(s: str, n_buckets: int) -> int:
    if n_buckets <= 1:
        return 1
    h = hashlib.blake2b(str(s).encode("utf-8"), digest_size=8).digest()
    v = int.from_bytes(h, "little", signed=False)
    return int(v % int(n_buckets)) + 1


def _normalize_schema_rule(rule: Any) -> Dict[str, Any]:
    if isinstance(rule, str):
        return {"type": str(rule).strip().lower()}
    if isinstance(rule, dict):
        out: Dict[str, Any] = {}
        if "type" in rule and rule.get("type") is not None:
            out["type"] = str(rule.get("type")).strip().lower()
        if "min" in rule and rule.get("min") is not None:
            out["min"] = float(rule.get("min"))
        if "max" in rule and rule.get("max") is not None:
            out["max"] = float(rule.get("max"))
        if "drop" in rule:
            out["drop"] = bool(rule.get("drop"))
        return out
    return {}


def load_variable_type_overrides(schema_json_path: Optional[str]) -> Dict[str, Any]:
    if not schema_json_path:
        return {}
    p = Path(schema_json_path)
    if not p.exists():
        raise FileNotFoundError(f"未找到变量类型schema文件：{schema_json_path}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    out: Dict[str, Any] = {}
    if isinstance(raw, dict) and all((isinstance(v, str) for v in raw.values())):
        for k, v in raw.items():
            out[str(k).strip()] = _normalize_schema_rule(v)
        return out
    if isinstance(raw, dict):
        by_name = raw.get("by_name", {})
        by_prefix = raw.get("by_prefix", {})
        for k, v in by_name.items():
            out[str(k).strip()] = _normalize_schema_rule(v)
        for k, v in by_prefix.items():
            out[f"prefix::{str(k).strip()}"] = _normalize_schema_rule(v)
    return out


def resolve_variable_rule(
    name: str, overrides: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    if not overrides:
        return None
    nm = str(name).strip()
    if nm in overrides:
        rule = dict(overrides[nm])
        rule["__matched_by__"] = nm
        return rule
    best = None
    best_len = -1
    best_key = None
    for k, v in overrides.items():
        if not str(k).startswith("prefix::"):
            continue
        pfx = str(k).split("::", 1)[1]
        if nm.startswith(pfx) and len(pfx) > best_len:
            best = dict(v)
            best_len = len(pfx)
            best_key = pfx
    if best is not None:
        best["__matched_by__"] = f"prefix::{best_key}"
    return best


def resolve_variable_type_override(name: str, overrides: Optional[Dict[str, Any]]) -> Optional[str]:
    rule = resolve_variable_rule(name, overrides)
    if not rule:
        return None
    tp = rule.get("type", None)
    return None if tp is None else str(tp).strip().lower()


def _numeric_unique_is_consecutive_small_integers(uniq: np.ndarray) -> bool:
    if uniq.size == 0 or uniq.size > 12:
        return False
    if not np.all(np.isfinite(uniq)):
        return False
    if np.max(np.abs(uniq - np.round(uniq))) > 1e-08:
        return False
    uniq_i = np.sort(np.unique(np.round(uniq).astype(int)))
    if uniq_i.size <= 1:
        return False
    return bool(np.all(np.diff(uniq_i) == 1))


def guess_numeric_variable_semantics(name: str, uniq: np.ndarray) -> Tuple[bool, bool]:
    nm = str(name)
    if uniq.size <= 2:
        return (True, False)
    ordinal_keywords = [
        "频率",
        "次数",
        "程度",
        "满意",
        "同意",
        "经常",
        "偶尔",
        "从不",
        "总是",
        "很少",
        "有时",
        "几乎",
        "严重",
        "轻度",
        "中度",
        "重度",
    ]
    nominal_keywords = [
        "性别",
        "民族",
        "户籍",
        "身份",
        "学校类型",
        "地区",
        "省",
        "市",
        "县",
        "类别",
        "类型",
        "编码",
    ]
    if any((k in nm for k in nominal_keywords)):
        return (True, False)
    if _numeric_unique_is_consecutive_small_integers(uniq):
        if any((k in nm for k in ordinal_keywords)):
            return (True, True)
        return (True, False)
    return (True, True)


def stratified_binary_split(
    indices: np.ndarray, y: np.ndarray, seed: int, frac_first: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=int)
    rng = np.random.RandomState(seed)
    parts_a, parts_b = ([], [])
    for cls in [0, 1]:
        sub = idx[y[idx] == cls].copy()
        rng.shuffle(sub)
        if len(sub) <= 1:
            cut = len(sub)
        else:
            cut = int(round(len(sub) * float(frac_first)))
            cut = min(max(cut, 1), len(sub) - 1)
        parts_a.append(sub[:cut])
        parts_b.append(sub[cut:])
    a = np.concatenate(parts_a) if parts_a else np.array([], dtype=int)
    b = np.concatenate(parts_b) if parts_b else np.array([], dtype=int)
    if len(a) == 0 or len(b) == 0:
        idx = idx.copy()
        rng.shuffle(idx)
        cut = max(1, min(len(idx) - 1, int(round(len(idx) * float(frac_first)))))
        return (np.sort(idx[:cut]), np.sort(idx[cut:]))
    return (np.sort(a), np.sort(b))


def split_val_for_nested_calibration(
    idx_val: np.ndarray, y: np.ndarray, prov_ids: Optional[np.ndarray], seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    idx_val = np.asarray(idx_val, dtype=int)
    if idx_val.size < 8:
        return (idx_val.copy(), idx_val.copy())
    if prov_ids is not None:
        pv = prov_ids[idx_val]
        uniq = np.unique(pv)
        if uniq.size >= 2:
            rng = np.random.RandomState(seed)
            uniq = uniq.copy()
            rng.shuffle(uniq)
            cut = max(1, uniq.size // 2)
            fit_prov = uniq[:cut]
            sel_prov = uniq[cut:]
            fit_idx = idx_val[np.isin(pv, fit_prov)]
            sel_idx = idx_val[np.isin(pv, sel_prov)]
            if fit_idx.size > 0 and sel_idx.size > 0:
                return (np.sort(fit_idx), np.sort(sel_idx))
    return stratified_binary_split(idx_val, y=y, seed=seed, frac_first=0.5)


def build_depression_score(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    dep_cols = [
        c for c in df.columns if str(c).strip().startswith(config.DEFAULT_DEPRESSION_ITEM_PREFIX)
    ]
    dep_cols2 = []
    for c in dep_cols:
        if "(" in str(c) or "（" in str(c) or len(dep_cols) >= 8:
            dep_cols2.append(c)
    dep_cols = dep_cols2
    if not dep_cols:
        dep_score = np.zeros((len(df),), dtype=np.float32)
        dep_level = np.zeros((len(df),), dtype=np.int64)
        return (dep_score, dep_level, dep_cols)
    X = df[dep_cols].copy()
    for c in dep_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c] - 1.0
        X[c] = X[c].clip(0, 3)
        if any((k in str(c) for k in config.DEPRESSION_REVERSE_KEYWORDS)):
            X[c] = 3.0 - X[c]
    med = X.median(axis=0, skipna=True)
    X = X.fillna(med)
    dep_score = X.sum(axis=1).astype(np.float32).to_numpy()
    q1, q2, q3 = np.quantile(dep_score, [0.5, 0.75, 0.9])
    dep_level = np.zeros_like(dep_score, dtype=np.int64)
    dep_level[dep_score >= q1] = 1
    dep_level[dep_score >= q2] = 2
    dep_level[dep_score >= q3] = 3
    return (dep_score, dep_level, dep_cols)


def build_target_y(
    raw: pd.DataFrame,
    target_mode: str,
    label_keywords: List[str],
    y_positive_value: int = 1,
    disable_auto_flip_y: bool = False,
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if target_mode == config.TARGET_SUICIDE:
        col = find_label_column(raw, label_keywords)
        if col is None:
            raise RuntimeError(f"未找到自杀意念标签列，keywords={label_keywords}")
        ser = raw[col]

        def _to_num(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return np.nan
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v)
            s = str(v).strip()
            if s == "":
                return np.nan
            if s in ["是", "有", "有过", "出现过", "经常", "偶尔"]:
                return 1.0
            if s in ["否", "无", "没有", "从不", "未出现"]:
                return 0.0
            try:
                return float(s)
            except Exception:
                return np.nan

        y_num = ser.map(_to_num).to_numpy()
        pos_val = float(y_positive_value)
        if not disable_auto_flip_y:
            uniq = set(pd.Series(y_num).dropna().unique().tolist())
            if uniq.issubset({1.0, 2.0}):
                rate_1 = float(np.mean(y_num == 1.0))
                rate_2 = float(np.mean(y_num == 2.0))
                if rate_1 > 0.5 and rate_2 < 0.5 and (int(y_positive_value) == 1):
                    pos_val = 2.0
                    meta["auto_flip_y"] = True
                    meta["auto_flip_reason"] = (
                        f"Detected high rate of value=1 (rate_1={rate_1:.3f}), assume 2=是"
                    )
        valid_mask = ~np.isnan(y_num)
        y = np.full((len(y_num),), -1, dtype=np.int64)
        y[valid_mask] = (y_num[valid_mask] == pos_val).astype(np.int64)
        meta["label_col"] = col
        meta["y_positive_value_used"] = int(pos_val)
        meta["y_positive_value_requested"] = int(y_positive_value)
        meta["n_total"] = int(len(y_num))
        meta["n_labeled"] = int(valid_mask.sum())
        meta["n_missing_label"] = int((~valid_mask).sum())
        meta["missing_label_rate"] = float((~valid_mask).mean())
        meta["valid_mask"] = valid_mask.astype(bool).tolist()
        meta["y_positive_rate_labeled_only"] = (
            float(np.mean(y[valid_mask] == 1)) if valid_mask.any() else float("nan")
        )
        return (y, col, meta)
    if target_mode == config.TARGET_DEPRESSION:
        dep_score, dep_level, dep_cols = build_depression_score(raw)
        y = (dep_level >= 2).astype(np.int64)
        meta["dep_cols"] = dep_cols
        meta["dep_score_quantiles"] = [float(x) for x in np.quantile(dep_score, [0.5, 0.75, 0.9])]
        meta["valid_mask"] = np.ones((len(y),), dtype=bool).tolist()
        meta["n_total"] = int(len(y))
        meta["n_labeled"] = int(len(y))
        meta["n_missing_label"] = 0
        meta["missing_label_rate"] = 0.0
        return (y, "__dep_abnormal__", meta)
    raise ValueError(f"Unknown target_mode: {target_mode}")


def get_same_question_family_cols(raw_cols: List[str], label_col: str) -> List[str]:
    q, _ = parse_question_key(label_col)
    fam = []
    for c in raw_cols:
        qq, _ = parse_question_key(c)
        if qq == q:
            fam.append(c)
    return fam


def leakage_filter_columns(
    raw: pd.DataFrame,
    target_mode: str,
    y_label_col: Optional[str],
    dep_cols: Optional[List[str]],
    aggressive_drop_post_outcome: bool = True,
    drop_symptom_items_for_suicide: bool = True,
    strict_construct_filter: bool = False,
    drop_state_proxy_items: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Exclude target items and configured overlapping outcome constructs."""
    df = drop_meta_columns(raw).copy()
    cols = [c for c in df.columns if c not in ["__province__", "__source_file__"]]
    dropped = {
        "explicit": [],
        "family": [],
        "post_outcome": [],
        "dep_items": [],
        "symptom_items": [],
        "score_like": [],
        "construct_overlap": [],
        "state_proxy": [],
        "final_dropped": [],
    }
    for c in cols:
        if any((k in str(c) for k in ["总分", "得分", "量表总分", "score", "Score"])):
            dropped["score_like"].append(c)
    if target_mode == config.TARGET_SUICIDE:
        if y_label_col is None:
            raise RuntimeError("target_mode=suicide_ideation 但 y_label_col=None")
        dropped["explicit"].append(y_label_col)
        fam = get_same_question_family_cols(cols, y_label_col)
        for c in fam:
            if c != y_label_col:
                dropped["family"].append(c)
        for c in cols:
            name = str(c)
            if any((k in name for k in config.DEFAULT_ATTEMPT_LABEL_KEYWORDS)):
                dropped["post_outcome"].append(c)
            if any((k in name for k in config.DEFAULT_SELFHARM_KEYWORDS)):
                dropped["post_outcome"].append(c)
            if "自杀计划" in name or ("计划" in name and "自杀" in name):
                dropped["post_outcome"].append(c)
        if aggressive_drop_post_outcome:
            for c in cols:
                name = str(c)
                if any((k in name for k in config.POST_OUTCOME_KEYWORDS)):
                    dropped["post_outcome"].append(c)
        if drop_symptom_items_for_suicide:
            for c in cols:
                name = str(c)
                if any((name.startswith(pfx) for pfx in config.STRICT_SUICIDE_OVERLAP_PREFIXES)):
                    dropped["symptom_items"].append(c)
    elif target_mode == config.TARGET_DEPRESSION:
        if dep_cols:
            dropped["dep_items"].extend(dep_cols)
        if aggressive_drop_post_outcome:
            for c in cols:
                name = str(c)
                if any((k in name for k in config.POST_OUTCOME_KEYWORDS)):
                    dropped["post_outcome"].append(c)
    if strict_construct_filter:
        for c in cols:
            name = str(c)
            if any((name.startswith(pfx) for pfx in config.CONSTRUCT_OVERLAP_PREFIXES)):
                dropped["construct_overlap"].append(c)
    if drop_state_proxy_items:
        for c in cols:
            name = str(c)
            if any((name.startswith(pfx) for pfx in config.FACTOR_STATE_PROXY_PREFIXES)):
                dropped["state_proxy"].append(c)
    drop_set = set(
        dropped["explicit"]
        + dropped["family"]
        + dropped["post_outcome"]
        + dropped["dep_items"]
        + dropped["symptom_items"]
        + dropped["score_like"]
        + dropped["construct_overlap"]
        + dropped["state_proxy"]
    )
    drop_set = {c for c in drop_set if c in df.columns}
    dropped["final_dropped"] = sorted(list(drop_set))
    df = df.drop(columns=dropped["final_dropped"], errors="ignore")
    report = {
        "target_mode": target_mode,
        "y_label_col": y_label_col,
        "n_before": len(cols),
        "n_after": len([c for c in df.columns if c not in ["__province__", "__source_file__"]]),
        "strict_construct_filter": strict_construct_filter,
        "dropped": {k: sorted(list(set(v))) for k, v in dropped.items()},
    }
    return (df, report)


@dataclass
class VarSpec:
    """Category vocabulary and ordering information for one variable."""

    name: str
    values: List[str]
    is_numeric: bool
    is_ordinal: bool
    q_key: Tuple[int, int]

    @property
    def is_binary(self) -> bool:
        return len(self.values) - 1 <= 2

    @property
    def n_real_cats(self) -> int:
        return max(1, len(self.values) - 1)


@dataclass
class EncodedData:
    """Integer feature matrix, schema, and domain membership."""

    X: np.ndarray
    col_names: List[str]
    var_specs: List[VarSpec]
    provinces: np.ndarray


def infer_var_specs(
    df: pd.DataFrame,
    max_unique: int,
    hash_buckets: int,
    schema_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[VarSpec], List[str]]:
    """Encode categories with zero reserved for missing values."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_names: List[str] = []
    var_specs: List[VarSpec] = []
    feature_cols = [c for c in df.columns if c not in ["__province__", "__source_file__"]]
    for c in feature_cols:
        s = df[c]
        name = str(c).strip()
        col_names.append(name)
        qkey = parse_question_key(name)
        resolve_variable_rule(name, schema_overrides)
        override = resolve_variable_type_override(name, schema_overrides)
        sn = pd.to_numeric(s, errors="coerce")
        numeric_ratio = float(sn.notna().mean())
        if override in {"continuous", "ordinal", "binary"}:
            numeric_ratio = 1.0
        if numeric_ratio > 0.85:
            vals = sn.dropna().to_numpy()
            uniq = np.unique(vals)
            is_numeric, is_ordinal = guess_numeric_variable_semantics(name, uniq)
            if override == "nominal":
                is_numeric, is_ordinal = (False, False)
            elif override == "binary":
                is_numeric, is_ordinal = (True, False)
            elif override == "ordinal":
                is_numeric, is_ordinal = (True, True)
            elif override == "continuous":
                is_numeric, is_ordinal = (True, True)
            if len(uniq) <= max_unique:
                uniq_sorted = np.sort(uniq)
                values = ["<phi>"] + [
                    str(int(u)) if float(u).is_integer() else str(u) for u in uniq_sorted.tolist()
                ]
                mapping = {u: i + 1 for i, u in enumerate(uniq_sorted.tolist())}
                out = np.zeros((len(df),), dtype=np.int64)
                arr = sn.to_numpy()
                for i, v in enumerate(arr):
                    if np.isnan(v):
                        out[i] = 0
                    else:
                        out[i] = mapping.get(v, 0)
                df[c] = out
                var_specs.append(
                    VarSpec(
                        name=name,
                        values=values,
                        is_numeric=is_numeric,
                        is_ordinal=is_ordinal,
                        q_key=qkey,
                    )
                )
            else:
                qs = np.quantile(vals, [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875])
                bins = [-np.inf] + qs.tolist() + [np.inf]
                out = np.zeros((len(df),), dtype=np.int64)
                arr = sn.to_numpy()
                for i, v in enumerate(arr):
                    if np.isnan(v):
                        out[i] = 0
                    else:
                        b = int(np.searchsorted(bins, v, side="right") - 1)
                        out[i] = max(1, min(b, 8))
                values = ["<phi>"] + [f"bin_{k}" for k in range(1, 9)]
                df[c] = out
                var_specs.append(
                    VarSpec(name=name, values=values, is_numeric=True, is_ordinal=True, q_key=qkey)
                )
        else:
            sv = s.astype(str).replace("nan", np.nan)
            uniq = sv.dropna().unique().tolist()
            if len(uniq) <= max_unique:
                uniq_sorted = sorted([str(u) for u in uniq])
                values = ["<phi>"] + uniq_sorted
                mapping = {u: i + 1 for i, u in enumerate(uniq_sorted)}
                out = np.zeros((len(df),), dtype=np.int64)
                arr = sv.to_numpy()
                for i, v in enumerate(arr):
                    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
                        out[i] = 0
                    else:
                        out[i] = mapping.get(str(v), 0)
                df[c] = out
                var_specs.append(
                    VarSpec(
                        name=name, values=values, is_numeric=False, is_ordinal=False, q_key=qkey
                    )
                )
            else:
                values = ["<phi>"] + [f"hb_{i}" for i in range(1, hash_buckets + 1)]
                out = np.zeros((len(df),), dtype=np.int64)
                arr = sv.to_numpy()
                for i, v in enumerate(arr):
                    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v):
                        out[i] = 0
                    else:
                        out[i] = hash_bucket(str(v), hash_buckets)
                df[c] = out
                var_specs.append(
                    VarSpec(
                        name=name, values=values, is_numeric=False, is_ordinal=False, q_key=qkey
                    )
                )
    return (df, var_specs, col_names)


def encode_dataframe(
    df: pd.DataFrame,
    max_unique: int,
    hash_buckets: int,
    schema_overrides: Optional[Dict[str, Any]] = None,
) -> EncodedData:
    df2, var_specs, col_names = infer_var_specs(
        df, max_unique=max_unique, hash_buckets=hash_buckets, schema_overrides=schema_overrides
    )
    X = df2[[c for c in df2.columns if c not in ["__province__", "__source_file__"]]].to_numpy(
        dtype=np.int64
    )
    provinces = df2["__province__"].astype(str).to_numpy()
    return EncodedData(X=X, col_names=col_names, var_specs=var_specs, provinces=provinces)


def build_splits(
    provinces: np.ndarray,
    seed: int,
    train_k: int = -1,
    val_provinces: int = 1,
    test_provinces: int = 1,
) -> Dict[str, List[str]]:
    """Partition entire domains into training, validation, and test groups."""
    unique = sorted(list(set(provinces.tolist())))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n_prov = len(unique)
    if n_prov < 3:
        raise ValueError(f"省份数量不足以OOD划分：{unique}")
    holdout_needed = int(val_provinces) + int(test_provinces)
    if n_prov <= holdout_needed:
        raise ValueError(
            f"省份数量不足：n_prov={n_prov}, val={val_provinces}, test={test_provinces}"
        )
    if int(train_k) <= 0:
        train_k = n_prov - holdout_needed
    train_k = min(int(train_k), n_prov - holdout_needed)
    train = unique[:train_k]
    val = unique[train_k : train_k + int(val_provinces)]
    test = unique[train_k + int(val_provinces) : train_k + int(val_provinces) + int(test_provinces)]
    return {"train": train, "val": val, "test": test, "all": unique}


def indices_by_provinces(provinces: np.ndarray, prov_list: List[str]) -> np.ndarray:
    mask = np.isin(provinces, prov_list)
    return np.where(mask)[0]


def generate_province_splits(
    prov_list: List[str], n_test: int = 2, n_val: int = 2, n_splits: int = 5, seed: int = 42
) -> List[Dict[str, List[str]]]:
    rng = np.random.RandomState(seed)
    n_prov = len(prov_list)
    splits = []
    seen: set = set()
    max_tries = 500
    for _ in range(max_tries):
        if len(splits) >= n_splits:
            break
        idx = rng.permutation(n_prov)
        test_p = tuple(sorted([prov_list[i] for i in idx[:n_test]]))
        val_p = tuple(sorted([prov_list[i] for i in idx[n_test : n_test + n_val]]))
        train_p = [prov_list[i] for i in idx[n_test + n_val :]]
        key = test_p + val_p
        if key not in seen:
            seen.add(key)
            splits.append({"train": train_p, "val": list(val_p), "test": list(test_p)})
    if not splits:
        splits = [
            {
                "train": prov_list[n_test + n_val :],
                "val": prov_list[n_test : n_test + n_val],
                "test": prov_list[:n_test],
            }
        ]
    return splits
