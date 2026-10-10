"""Command-line arguments for the research pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="CA-ResMTL research pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data_glob", type=str, default=config.DATA_GLOB)
    ap.add_argument("--output_dir", type=str, default=str(config.OUTPUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--target_mode",
        type=str,
        default=config.TARGET_SUICIDE,
        choices=[config.TARGET_SUICIDE, config.TARGET_DEPRESSION],
    )
    ap.add_argument(
        "--label_keywords", type=str, nargs="*", default=config.DEFAULT_PRIMARY_LABEL_KEYWORDS
    )
    ap.add_argument("--max_unique", type=int, default=config.DEFAULT_MAX_UNIQUE_CATEGORIES)
    ap.add_argument("--hash_buckets", type=int, default=config.DEFAULT_HASH_BUCKETS)
    ap.add_argument(
        "--var_schema_json",
        type=str,
        default="",
        help="Variable schema JSON with exact-name or prefix rules for type, min, max, and drop.",
    )
    ap.add_argument(
        "--min_nonmissing_per_feature",
        type=int,
        default=15,
        help="Minimum nonmissing responses required to retain a feature.",
    )
    ap.add_argument(
        "--max_missing_rate_per_feature",
        type=float,
        default=0.985,
        help="Maximum missing fraction allowed for a feature.",
    )
    ap.add_argument(
        "--keep_location_question",
        action="store_true",
        help="Retain location questions; these are otherwise excluded from domain prediction.",
    )
    ap.add_argument(
        "--train_k",
        type=int,
        default=-1,
        help="Number of training domains; -1 uses all remaining domains.",
    )
    ap.add_argument(
        "--val_provinces",
        type=int,
        default=2,
        help="Number of validation domains.",
    )
    ap.add_argument(
        "--test_provinces",
        type=int,
        default=2,
        help="Number of test domains.",
    )
    ap.add_argument("--emb_dim", type=int, default=config.DEFAULT_EMB_DIM)
    ap.add_argument("--docd_inner", type=int, default=config.DEFAULT_EPOCHS_DOCD_INNER)
    ap.add_argument("--docd_outer", type=int, default=config.DEFAULT_DOCD_OUTER_LOOPS)
    ap.add_argument("--min_outer_loops", type=int, default=config.DEFAULT_MIN_OUTER_LOOPS)
    ap.add_argument("--lr_docd", type=float, default=config.DEFAULT_LR_DOCD)
    ap.add_argument("--lr_struct_mult", type=float, default=config.DEFAULT_LR_STRUCT_MULT)
    ap.add_argument("--l1_lambda", type=float, default=config.DEFAULT_L1_LAMBDA)
    ap.add_argument("--l1_y_in_mult", type=float, default=config.DEFAULT_L1_Y_IN_MULT)
    ap.add_argument("--dag_tol", type=float, default=config.DEFAULT_DAG_TOL)
    ap.add_argument("--rho_max", type=float, default=config.DEFAULT_RHO_MAX)
    ap.add_argument("--edge_threshold", type=float, default=config.DEFAULT_EDGE_THRESHOLD)
    ap.add_argument("--y_nll_weight", type=float, default=config.DEFAULT_Y_NLL_WEIGHT)
    ap.add_argument("--init_logit", type=float, default=config.DEFAULT_INIT_LOGIT)
    ap.add_argument("--init_logit_y_in", type=float, default=config.DEFAULT_INIT_LOGIT_Y_IN)
    ap.add_argument("--mi_prior_lambda", type=float, default=config.DEFAULT_MI_PRIOR_LAMBDA)
    ap.add_argument("--stop_patience", type=int, default=config.DEFAULT_STOP_PATIENCE)
    ap.add_argument("--hutch_eval_samples", type=int, default=config.DEFAULT_HUTCH_EVAL_SAMPLES)
    ap.add_argument("--graph_target_degree", type=int, default=config.DEFAULT_GRAPH_TARGET_DEGREE)
    ap.add_argument("--y_parent_min_k", type=int, default=config.DEFAULT_Y_PARENT_MIN_K)
    ap.add_argument("--y_parent_max_k", type=int, default=config.DEFAULT_Y_PARENT_MAX_K)
    ap.add_argument("--y_parent_min_edge", type=float, default=config.DEFAULT_Y_PARENT_MIN_EDGE)
    ap.add_argument("--stability_topn", type=int, default=config.DEFAULT_STAB_TOPN)
    ap.add_argument("--stability_freq", type=float, default=config.DEFAULT_STAB_FREQ)
    ap.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    ap.add_argument("--epochs_pred", type=int, default=config.DEFAULT_EPOCHS_PRED)
    ap.add_argument("--layers", type=int, default=config.DEFAULT_TRANSFORMER_LAYERS)
    ap.add_argument("--heads", type=int, default=config.DEFAULT_HEADS)
    ap.add_argument("--dropout", type=float, default=config.DEFAULT_DROPOUT)
    ap.add_argument("--lr_pred", type=float, default=config.DEFAULT_LR_PRED)
    ap.add_argument(
        "--weight_decay", type=float, default=0.0001, help="AdamW weight decay for the predictor."
    )
    ap.add_argument(
        "--early_stop_patience",
        type=int,
        default=config.DEFAULT_EARLY_STOP_PATIENCE,
        help="Predictor epochs without a validation selection-score improvement before stopping.",
    )
    ap.add_argument(
        "--label_smoothing",
        type=float,
        default=config.DEFAULT_LABEL_SMOOTHING,
        help="Label smoothing coefficient.",
    )
    ap.add_argument(
        "--selection_criterion",
        type=str,
        default="hybrid",
        choices=["hybrid", "auc", "auc_ece", "brier"],
        help="Validation criterion used to select the predictor.",
    )
    ap.add_argument(
        "--ece_lambda",
        type=float,
        default=0.3,
        help="ECE penalty weight for auc_ece model selection.",
    )
    ap.add_argument(
        "--loss_type",
        type=str,
        default=config.DEFAULT_LOSS_TYPE,
        choices=["asl"],
        help="Primary prediction loss.",
    )
    ap.add_argument(
        "--rank_weight",
        type=float,
        default=config.DEFAULT_RANK_WEIGHT,
        help="ListNet ranking loss weight.",
    )
    ap.add_argument(
        "--margin_weight",
        type=float,
        default=config.DEFAULT_MARGIN_WEIGHT,
        help="Classification margin loss weight.",
    )
    ap.add_argument(
        "--target_margin",
        type=float,
        default=config.DEFAULT_TARGET_MARGIN,
        help="Target classification margin in logit space.",
    )
    ap.add_argument(
        "--logit_scale_init",
        type=float,
        default=config.DEFAULT_LOGIT_SCALE_INIT,
        help="Initial learnable logit scale.",
    )
    ap.add_argument(
        "--causal_topk",
        type=int,
        default=config.DEFAULT_CAUSAL_TOPK,
        help="Maximum number of outcome-related features used by the residual branch.",
    )
    ap.add_argument(
        "--causal_residual_dropout",
        type=float,
        default=config.DEFAULT_CAUSAL_RESIDUAL_DROPOUT,
        help="Dropout for the structure-guided residual branch.",
    )
    ap.add_argument(
        "--disable_group_dro",
        action="store_true",
        default=False,
        help="Disable domain-based group DRO.",
    )
    ap.add_argument(
        "--group_dro_eta",
        type=float,
        default=config.DEFAULT_GROUP_DRO_ETA,
        help="Group DRO weight update rate.",
    )
    ap.add_argument(
        "--aux_prob_weight",
        type=float,
        default=config.DEFAULT_AUX_PROB_WEIGHT,
        help="Probability-head auxiliary loss weight.",
    )
    ap.add_argument(
        "--aux_calib_weight",
        type=float,
        default=config.DEFAULT_AUX_CALIB_WEIGHT,
        help="Probability-head calibration and prevalence penalty weight.",
    )
    ap.add_argument(
        "--logit_adjust_tau",
        type=float,
        default=config.DEFAULT_LOGIT_ADJ_TAU,
        help="Training-time logit adjustment strength.",
    )
    ap.add_argument(
        "--late_rank_mult",
        type=float,
        default=config.DEFAULT_LATE_RANK_MULT,
        help="Ranking weight multiplier in the final training phase.",
    )
    ap.add_argument(
        "--worst_group_weight",
        type=float,
        default=config.DEFAULT_WORST_GROUP_WEIGHT,
        help="Worst-domain AUC weight in model selection.",
    )
    ap.add_argument(
        "--attn_prior_mode",
        type=str,
        default=config.DEFAULT_ATTN_PRIOR_MODE,
        choices=["soft", "hybrid", "hard"],
        help="Attention prior derived from the learned directed graph.",
    )
    ap.add_argument(
        "--attn_prior_strength",
        type=float,
        default=config.DEFAULT_ATTN_PRIOR_STRENGTH,
        help="Attention prior strength for ancestors and direct edges.",
    )
    ap.add_argument(
        "--attn_nonancestor_penalty",
        type=float,
        default=config.DEFAULT_ATTN_NONANCESTOR_PENALTY,
        help="Penalty for nonancestor connections in the soft attention prior.",
    )
    ap.add_argument("--bootstrap_struct", type=int, default=config.DEFAULT_BOOTSTRAP_STRUCT)
    ap.add_argument("--ate_base_samples", type=int, default=config.DEFAULT_ATE_BASE_SAMPLES)
    ap.add_argument(
        "--ate_adaptive",
        action="store_true",
        help="Increase intervention base samples until the interval-width target or sample limit is reached.",
    )
    ap.add_argument(
        "--ate_ci_eps",
        type=float,
        default=0.01,
        help="Target half-width of the 95%% interval for estimated intervention risk.",
    )
    ap.add_argument(
        "--ate_min_base",
        type=int,
        default=300,
        help="Minimum base samples for adaptive intervention analysis.",
    )
    ap.add_argument(
        "--ate_step_base",
        type=int,
        default=300,
        help="Base sample increment for adaptive intervention analysis.",
    )
    ap.add_argument(
        "--ate_robust",
        action="store_true",
        help="Aggregate intervention rankings across graph thresholds.",
    )
    ap.add_argument(
        "--ate_thresholds",
        type=str,
        default="",
        help="Comma-separated graph thresholds; an empty value selects thresholds automatically.",
    )
    ap.add_argument("--ate_mc_samples", type=int, default=config.DEFAULT_ATE_MC_SAMPLES)
    ap.add_argument("--ate_topk", type=int, default=config.DEFAULT_ATE_TOPK)
    ap.add_argument(
        "--bootstrap_save_yin",
        action="store_true",
        help="Save incoming-outcome edge probabilities for bootstrap diagnostics.",
    )
    ap.add_argument(
        "--ate_sensitivity_topk",
        type=int,
        default=10,
        help="Top variables used for graph-threshold ranking comparisons.",
    )
    ap.add_argument(
        "--ate_bootstrap_reps",
        type=int,
        default=50,
        help="Intervention bootstrap repetitions; zero disables bootstrapping.",
    )
    ap.add_argument(
        "--ate_bootstrap_topk",
        type=int,
        default=10,
        help="Top variables included in intervention bootstrap estimates.",
    )
    ap.add_argument(
        "--ate_bootstrap_replace",
        action="store_true",
        help="Sample bootstrap base rows with replacement.",
    )
    ap.add_argument(
        "--ate_bootstrap_ci",
        type=float,
        default=0.95,
        help="Intervention bootstrap confidence level.",
    )
    ap.add_argument(
        "--stage2_only",
        action="store_true",
        default=False,
        help="Train the predictor using structure and preprocessing artifacts from output_dir.",
    )
    ap.add_argument(
        "--skip_stage3",
        action="store_true",
        default=False,
        help="Skip intervention analysis.",
    )
    ap.add_argument(
        "--ate_only",
        action="store_true",
        default=False,
        help="Run intervention analysis using the saved structure model.",
    )
    ap.add_argument(
        "--ate_chunk",
        type=int,
        default=64,
        help="Intervention inference chunk size; larger values use more memory.",
    )
    ap.add_argument(
        "--ate_candidates_source",
        type=str,
        default="stable",
        choices=["stable", "adaptive", "fixed"],
        help="Saved candidate list used for intervention ranking.",
    )
    ap.add_argument(
        "--ate_candidates_file",
        type=str,
        default="",
        help="Candidate CSV with a cause column, or variable names in its first column.",
    )
    ap.add_argument(
        "--ate_graph_target_degree",
        type=int,
        default=-1,
        help="Target graph degree for intervention analysis; -1 reuses graph_target_degree.",
    )
    ap.add_argument(
        "--disable_auto_flip_y",
        action="store_true",
        default=False,
        help="Disable prevalence-based reversal of 1/2-coded outcome labels.",
    )
    ap.add_argument(
        "--y_positive_value",
        type=int,
        default=1,
        help="Positive response code for the primary binary outcome.",
    )
    ap.add_argument("--no_aggressive_post_outcome_drop", action="store_true", default=False)
    ap.add_argument("--keep_symptom_items_for_suicide", action="store_true", default=False)
    ap.add_argument(
        "--pipeline_profile",
        type=str,
        default="balanced",
        choices=config.PIPELINE_PROFILE_CHOICES,
        help="Balanced prediction, screening, or factor-analysis settings.",
    )
    ap.add_argument(
        "--drop_state_proxy_items",
        action="store_true",
        default=False,
        help="Exclude proximal state-proxy items Q54, Q56, and Q57.",
    )
    ap.add_argument(
        "--strict_construct_filter",
        action="store_true",
        default=False,
        help="Exclude subjective emotion items Q55, Q64, Q71, and Q73 for depression analysis.",
    )
    ap.add_argument(
        "--run_ablation",
        action="store_true",
        default=False,
        help="Run full, no_mask, random_mask, std_emb, and mlp_only prediction variants.",
    )
    ap.add_argument(
        "--run_extended_baselines",
        action="store_true",
        default=False,
        help="Run additional linear and tree baselines with feature-importance summaries.",
    )
    ap.add_argument(
        "--run_calibration",
        action="store_true",
        default=False,
        help="Compute predictor and HistGBDT Brier scores, ECE, and calibration curves.",
    )
    ap.add_argument(
        "--run_hybrid_stack",
        action="store_true",
        default=False,
        help="Combine tree OOF features with prefitted neural predictions; see docs/implementation.md.",
    )
    ap.add_argument(
        "--run_subgroup_ate",
        action="store_true",
        default=False,
        help="Run subgroup intervention analysis after the main intervention stage.",
    )
    ap.add_argument(
        "--subgroup_col_names",
        type=str,
        nargs="*",
        default=[],
        help="Subgroup column names; an empty list searches for configured demographic items.",
    )
    ap.add_argument(
        "--subgroup_ate_base_samples",
        type=int,
        default=800,
        help="Base samples for each subgroup intervention estimate.",
    )
    ap.add_argument(
        "--crosstask_ate_path",
        type=str,
        default="",
        help="Path to another experiment's ate_ranking.csv for rank comparisons.",
    )
    ap.add_argument(
        "--crosstask_task1_name",
        type=str,
        default="current",
        help="Current task label in the cross-task report.",
    )
    ap.add_argument(
        "--crosstask_task2_name",
        type=str,
        default="other",
        help="Comparison task label in the cross-task report.",
    )
    ap.add_argument("--disable_mi_screening", action="store_true", default=False)
    ap.add_argument("--y_cand_topk", type=int, default=config.DEFAULT_Y_CAND_TOPK)
    ap.add_argument("--y_cand_minmi", type=float, default=config.DEFAULT_Y_CAND_MINMI)
    ap.add_argument(
        "--backbone",
        type=str,
        default="deep_resmlp",
        choices=["causal_attention", "ft_transformer", "deep_resmlp"],
        help="Prediction backbone.",
    )
    ap.add_argument(
        "--resmlp_blocks",
        type=int,
        default=6,
        help="Number of residual blocks in the residual MLP.",
    )
    ap.add_argument(
        "--use_causal_features",
        action="store_true",
        default=True,
        help="Enable structure-guided interaction features for the residual MLP.",
    )
    ap.add_argument(
        "--no_causal_features",
        dest="use_causal_features",
        action="store_false",
        help="Disable engineered interaction features.",
    )
    ap.add_argument(
        "--n_interaction_pairs",
        type=int,
        default=80,
        help="Number of interaction pairs selected by mutual information.",
    )
    ap.add_argument(
        "--use_multitask",
        action="store_true",
        default=True,
        help="Enable available auxiliary tasks for the residual MLP.",
    )
    ap.add_argument(
        "--no_multitask",
        dest="use_multitask",
        action="store_false",
        help="Disable auxiliary task heads.",
    )
    ap.add_argument(
        "--min_sensitivity_thr",
        type=float,
        default=config.DEFAULT_MIN_SENSITIVITY_THR,
        help="Minimum sensitivity for prediction threshold selection.",
    )
    ap.add_argument(
        "--use_aucm",
        action="store_true",
        default=config.DEFAULT_USE_AUCM,
        help="Enable the auxiliary AUC maximization objective.",
    )
    ap.add_argument(
        "--no_aucm",
        dest="use_aucm",
        action="store_false",
        help="Disable the AUC maximization objective.",
    )
    ap.add_argument(
        "--aucm_margin",
        type=float,
        default=config.DEFAULT_AUCM_MARGIN,
        help="AUC maximization margin.",
    )
    ap.add_argument(
        "--aucm_weight",
        type=float,
        default=config.DEFAULT_AUCM_WEIGHT,
        help="AUC maximization loss weight.",
    )
    ap.add_argument(
        "--aucm_warmup",
        type=int,
        default=config.DEFAULT_AUCM_WARMUP,
        help="Epochs used to ramp up the AUC maximization loss weight.",
    )
    ap.add_argument(
        "--aucm_delay",
        type=int,
        default=config.DEFAULT_AUCM_DELAY,
        help="Epochs before enabling the AUC maximization objective.",
    )
    ap.add_argument(
        "--mono_weight",
        type=float,
        default=config.DEFAULT_MONO_WEIGHT,
        help="Ordinal monotonicity regularization weight.",
    )
    ap.add_argument(
        "--coral_weight",
        type=float,
        default=config.DEFAULT_CORAL_WEIGHT,
        help="CORAL domain alignment weight.",
    )
    ap.add_argument(
        "--f2_sen_floor",
        type=float,
        default=config.DEFAULT_F2_SEN_FLOOR,
        help="Sensitivity floor for F2-based model selection.",
    )
    ap.add_argument(
        "--swa_start",
        type=float,
        default=config.DEFAULT_SWA_START,
        help="Fraction of training epochs before starting stochastic weight averaging.",
    )
    ap.add_argument(
        "--swa_freq",
        type=int,
        default=config.DEFAULT_SWA_FREQ,
        help="Epoch interval for stochastic weight averaging updates.",
    )
    ap.add_argument(
        "--asl_gamma_neg",
        type=float,
        default=config.DEFAULT_ASL_GAMMA_NEG,
        help="ASL negative-class focusing parameter.",
    )
    ap.add_argument(
        "--asl_gamma_pos",
        type=float,
        default=config.DEFAULT_ASL_GAMMA_POS,
        help="ASL positive-class focusing parameter.",
    )
    ap.add_argument(
        "--asl_clip",
        type=float,
        default=config.DEFAULT_ASL_CLIP,
        help="ASL probability clipping offset.",
    )
    ap.add_argument(
        "--config",
        default="",
        help="JSON configuration; explicit command-line options take precedence.",
    )
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="PyTorch CPU threads; zero uses the runtime default.",
    )
    ap.set_defaults(
        domain_adv=False,
        lambda_domain=0.25,
        grl_lambda=1.0,
        focal_alpha=0.55,
        focal_gamma=1.0,
        use_pcgrad=False,
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="")
    preliminary, _ = config_parser.parse_known_args(argv)
    options = {
        action.dest: action for action in ap._actions if action.dest not in {"help", "config"}
    }
    settings = {}
    if preliminary.config:
        try:
            settings = json.loads(Path(preliminary.config).read_text(encoding="utf-8"))
            if not isinstance(settings, dict):
                raise ValueError("Configuration must be a JSON object.")
            for key, value in settings.items():
                if key not in options:
                    raise ValueError(f"Unknown configuration option: {key}")
                action = options[key]
                if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                    if not isinstance(value, bool):
                        raise ValueError(f"{key} must be a boolean.")
                elif action.nargs == "*":
                    if not isinstance(value, list) or not all(
                        (isinstance(item, str) for item in value)
                    ):
                        raise ValueError(f"{key} must be an array of strings.")
                elif action.type is not None:
                    if isinstance(value, bool) or value != action.type(value):
                        raise ValueError(f"Invalid type for {key}.")
                    settings[key] = action.type(value)
                elif not isinstance(value, str):
                    raise ValueError(f"{key} must be a string.")
                if action.choices is not None and value not in action.choices:
                    raise ValueError(f"Unsupported value for {key}: {value}")
            ap.set_defaults(**settings)
        except (OSError, ValueError, TypeError) as error:
            ap.error(str(error))
    args = ap.parse_args(argv)
    flag_destinations = {
        flag: action.dest for action in ap._actions for flag in action.option_strings
    }
    args._explicit_options = sorted(
        set(settings)
        | {
            flag_destinations[token.split("=", 1)[0]]
            for token in argv
            if token.split("=", 1)[0] in flag_destinations
        }
    )
    if args.ate_only and args.stage2_only:
        ap.error("--ate_only and --stage2_only are mutually exclusive.")
    if args.ate_only and args.skip_stage3:
        ap.error("--ate_only cannot be combined with --skip_stage3.")
    for name in (
        "emb_dim",
        "heads",
        "layers",
        "resmlp_blocks",
        "epochs_pred",
        "batch_size",
        "docd_inner",
        "docd_outer",
        "min_outer_loops",
        "ate_base_samples",
        "ate_mc_samples",
        "ate_chunk",
        "val_provinces",
        "test_provinces",
    ):
        if getattr(args, name) <= 0:
            ap.error(f"--{name} must be positive.")
    if args.emb_dim % args.heads:
        ap.error("--emb_dim must be divisible by --heads.")
    if args.num_threads < 0 or args.bootstrap_struct < 0 or args.ate_bootstrap_reps < 0:
        ap.error("Thread and bootstrap counts cannot be negative.")
    if not 0 <= args.min_sensitivity_thr <= 1 or not 0 <= args.f2_sen_floor <= 1:
        ap.error("Sensitivity floors must lie in [0, 1].")
    return args


def main(argv=None):
    args = parse_args(argv)
    from .pipeline import run_experiment

    run_experiment(args)
