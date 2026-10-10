"""Auxiliary labels retained locally for consistent resumption."""

import numpy as np
import pandas as pd

from . import config
from .data import build_depression_score


def build_auxiliary_targets(raw, target_mode):
    """Build auxiliary outcomes from the filtered rows of the raw questionnaire."""
    targets = []
    if target_mode != config.TARGET_DEPRESSION:
        _, levels, columns = build_depression_score(raw)
        if columns:
            values = (levels >= 2).astype(np.float32)
            if 0 < values.mean() < 1:
                targets.append(values)
    for column in raw.columns:
        if any(word in str(column) for word in config.DEFAULT_SELFHARM_KEYWORDS):
            responses = pd.to_numeric(raw[column], errors="coerce").to_numpy()
            # In this questionnaire, 1 means never and 2 or above indicates self-harm.
            values = np.where(np.isfinite(responses), responses >= 2, np.nan).astype(np.float32)
            rate = np.nanmean(values)
            if 0.005 < rate < 0.80:
                targets.append(values)
            break
    return targets or None
