# CA-ResMTL

CA-ResMTL combines directed questionnaire structure learning with a residual multilayer perceptron and auxiliary-task prediction. The pipeline includes data preprocessing, domain-based train/validation/test splits, probability calibration, and model-based intervention ranking.

The repository includes a synthetic example that runs without access to participant data. Study data and trained weights are not included.

## Installation

Use Python 3.12 in a virtual environment. Run these commands from the repository root:

```bash
python -m venv .venv
```

Activate the environment with `.venv/Scripts/Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on Linux/macOS. Then install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` pins the core dependency versions used for CPU verification and installs this package. `pyproject.toml` defines package metadata, supported dependency ranges, and optional extras. CatBoost and LightGBM can be installed with `python -m pip install -e ".[trees]"`; the standard workflow does not require them.

## Synthetic example

```bash
python -m ca_resmtl.demo --output_dir outputs/demo
```

The example generates six artificial questionnaire domains and runs structure learning, residual multitask prediction, calibration, and intervention ranking on CPU. It uses short training schedules and small sampling budgets. Choose a new output directory for each run.

## Questionnaire analysis

Provide one Excel workbook per domain. Each row represents one response. Column names, response codes, and outcome coding must match the questionnaire. See [Input data](docs/data_format.md).

```bash
python -m ca_resmtl --data_glob "data/private/*.xlsx" --output_dir outputs/study --device auto
python -m ca_resmtl --help
```

The default predictor uses the `deep_resmlp` backbone with engineered interaction features and auxiliary tasks when the required labels are available. The `causal_attention` and `ft_transformer` options provide alternative backbones without auxiliary task heads. See [Implementation](docs/implementation.md) for preprocessing, calibration, and evaluation details.

A JSON configuration is optional. Explicit command-line options override its values:

```bash
python -m ca_resmtl --config configs/example.json
```

`configs/example.json` illustrates the configuration format; it is not a complete specification of a published experiment. Full run settings and package versions are recorded in the output directory.

## Saved stages

Use an existing experiment directory to retrain its predictor or rerun intervention analysis:

```bash
python -m ca_resmtl --stage2_only --output_dir outputs/study --skip_stage3
python -m ca_resmtl --ate_only --output_dir outputs/study
```

Saved settings are restored unless explicitly overridden. Fresh experiments require a new output directory.

## Outputs

| File | Contents |
| --- | --- |
| `run_args.json`, `run_manifest.json` | Run settings, split sizes, source hashes, and package versions |
| `causal_adj_full.csv` | Learned directed adjacency |
| `docd_final.pth` | Structure model parameters |
| `resmtl_best_model.pth` | Selected predictor parameters |
| `pred_metrics.json` | Prediction and calibration metrics |
| `ate_ranking.csv` | Model-based intervention ranking |

Encoded responses, split indices, labels, and individual predictions are also saved locally. Input data and generated outputs are excluded by `.gitignore`.

## Tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
ruff check src tests
ruff format --check src tests
```

Tests use synthetic data and cover preprocessing, model gradients, calibration, all three backbone paths, stage resumption, and optional analyses. The GitHub workflow runs the same checks on Python 3.12.

## Repository structure

```text
src/ca_resmtl/   preprocessing, models, training, calibration, and analysis
configs/        example run settings and variable schema
tests/          synthetic unit and integration tests
docs/           input format and implementation details
data/           local input location
```
