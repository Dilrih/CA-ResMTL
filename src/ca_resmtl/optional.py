"""Load optional tree libraries only when an optional experiment uses them."""

import importlib
import importlib.util

_HAS_CATBOOST = importlib.util.find_spec("catboost") is not None
_HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None


def __getattr__(name):
    if name == "CatBoostClassifier":
        value = importlib.import_module("catboost").CatBoostClassifier if _HAS_CATBOOST else None
    elif name in {"LGBMClassifier", "lgb"}:
        module = importlib.import_module("lightgbm") if _HAS_LIGHTGBM else None
        value = module.LGBMClassifier if module is not None and name == "LGBMClassifier" else module
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value
