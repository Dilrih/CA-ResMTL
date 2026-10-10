"""Experiment orchestration and artifact resumption."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import config
from .artifacts import save_run_manifest
from .data import VarSpec
from .effect_stage import run_effect_stage
from .models.structure import DOCDModel
from .prediction_stage import run_prediction_stage
from .preparation import prepare_experiment
from .structure_stage import run_structure_stage
from .utils import set_seed


def _public_args(args):
    return {key: value for key, value in vars(args).items() if not key.startswith("_")}


def _load_artifacts(output_dir):
    required = [
        "X_xy.npy",
        "var_specs_xy.json",
        "causal_adj_full.csv",
        "idx_train.npy",
        "idx_val.npy",
        "idx_test.npy",
        "run_args.json",
    ]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Complete Stage 1 before resuming. Missing: {', '.join(missing)}")
    matrix = np.load(output_dir / "X_xy.npy", allow_pickle=False).astype(np.int64)
    specifications = json.loads((output_dir / "var_specs_xy.json").read_text(encoding="utf-8"))
    variables = [
        VarSpec(
            name=item["name"],
            values=item["values"],
            is_numeric=item["is_numeric"],
            is_ordinal=item["is_ordinal"],
            q_key=tuple(item["q_key"]),
        )
        for item in specifications
    ]
    adjacency = pd.read_csv(output_dir / "causal_adj_full.csv", index_col=0).to_numpy(
        dtype=np.float32
    )
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(variables)
        or adjacency.shape != (len(variables), len(variables))
    ):
        raise ValueError("Saved matrix, variable schema, and adjacency dimensions do not match.")
    indices = [
        np.load(output_dir / f"idx_{part}.npy", allow_pickle=False).astype(np.int64)
        for part in ("train", "val", "test")
    ]
    if any(
        index.ndim != 1 or len(index) == 0 or index.min() < 0 or index.max() >= len(matrix)
        for index in indices
    ):
        raise ValueError("Saved split indices are invalid.")
    return matrix, variables, adjacency, indices


def _load_structure(args, variables, output_dir, device):
    path = output_dir / "docd_final.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved structure model: {path.name}")
    model = DOCDModel(
        var_specs=variables,
        emb_dim=args.emb_dim,
        allowed_mask=np.ones((len(variables), len(variables)), dtype=np.float32),
        y_index=len(variables) - 1,
        y_nll_weight=args.y_nll_weight,
        init_logit=args.init_logit,
        init_logit_y_in=args.init_logit_y_in,
        y_in_prior=None,
        y_in_prior_lambda=args.mi_prior_lambda,
        mlp_hidden=192,
        mlp_depth=3,
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()


def run_experiment(args):
    """Run a fresh experiment, the prediction stage, or intervention analysis."""
    output_dir = Path(args.output_dir)
    resumed = bool(args.ate_only or args.stage2_only)
    if resumed:
        matrix, variables, adjacency, indices = _load_artifacts(output_dir)
        saved_args = json.loads((output_dir / "run_args.json").read_text(encoding="utf-8"))
        explicit = set(getattr(args, "_explicit_options", []))
        mode_options = {
            "config",
            "output_dir",
            "ate_only",
            "stage2_only",
            "skip_stage3",
            "device",
            "num_threads",
        }
        for key, value in saved_args.items():
            if key not in explicit | mode_options and hasattr(args, key):
                setattr(args, key, value)
        if args.emb_dim != saved_args.get("emb_dim", args.emb_dim):
            raise ValueError(
                "Resume with the embedding dimension used in the saved structure model."
            )
    elif (output_dir / "X_xy.npy").exists() or (output_dir / "run_args.json").exists():
        raise FileExistsError(
            "This output directory already contains an experiment. Choose a new directory or use --stage2_only / --ate_only."
        )

    if args.pipeline_profile == "screening":
        args.selection_criterion = "hybrid"
        args.min_sensitivity_thr = max(args.min_sensitivity_thr, 0.60)
        args.strict_construct_filter = False
        args.drop_state_proxy_items = False
    elif args.pipeline_profile == "factor":
        args.selection_criterion = "auc"
        args.min_sensitivity_thr = 0.0
        args.strict_construct_filter = True
        args.drop_state_proxy_items = True
    config.DEFAULT_MIN_SENSITIVITY_THR = args.min_sensitivity_thr
    config.DEFAULT_ASL_GAMMA_NEG = args.asl_gamma_neg
    config.DEFAULT_ASL_GAMMA_POS = args.asl_gamma_pos
    config.DEFAULT_ASL_CLIP = args.asl_clip
    set_seed(args.seed)
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Use --device cpu or --device auto."
        )
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (
        "ate_run_args.json"
        if args.ate_only
        else "stage2_run_args.json"
        if args.stage2_only
        else "run_args.json"
    )
    (output_dir / suffix).write_text(
        json.dumps(_public_args(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Device: {device}; profile: {args.pipeline_profile}; backbone: {args.backbone}", flush=True
    )

    raw = None
    if resumed:
        idx_train, idx_val, idx_test = indices
        names = [variable.name for variable in variables]
        y_index = len(names) - 1
        structure = None
    else:
        raw, _, matrix, variables, names, y_index, idx_train, idx_val, idx_test, allowed, prior = (
            prepare_experiment(args, output_dir)
        )
        structure, adjacency = run_structure_stage(
            args, output_dir, matrix, variables, names, y_index, idx_train, allowed, prior, device
        )
    if not args.ate_only:
        run_prediction_stage(
            args,
            output_dir,
            matrix,
            variables,
            adjacency,
            idx_train,
            idx_val,
            idx_test,
            device,
            raw=raw,
        )
    if not args.skip_stage3:
        if structure is None:
            structure = _load_structure(args, variables, output_dir, device)
        run_effect_stage(
            args,
            output_dir,
            structure,
            matrix,
            variables,
            names,
            y_index,
            adjacency,
            np.concatenate([idx_train, idx_val]),
            device,
        )
    save_run_manifest(
        output_dir,
        args,
        {"n_train": len(idx_train), "n_val": len(idx_val), "n_test": len(idx_test)},
        extra={
            "mode": "ate_only" if args.ate_only else "stage2_only" if resumed else "full",
            "backbone": args.backbone,
        },
        filename="ate_manifest.json"
        if args.ate_only
        else "stage2_manifest.json"
        if resumed
        else "run_manifest.json",
    )
    print(f"Finished. Results are in {output_dir}", flush=True)
    return output_dir
