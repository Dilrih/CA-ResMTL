# Input data

The pipeline reads Excel workbooks (`.xlsx`) with a header row and one response per row. Each workbook represents a province or another domain. Filename stems define domain identifiers used for splitting.

## Columns and outcomes

Default headers are numbered Chinese questionnaire items, such as `10、睡眠状况` and `35、很认真地考虑过自杀`. Outcome matching can be configured:

```bash
python -m ca_resmtl --data_glob "data/private/*.xlsx" --output_dir outputs/custom --label_keywords "outcome" --y_positive_value 1 --disable_auto_flip_y
```

Set the positive code from the questionnaire's data dictionary. The automatic check for 1/2-coded suicide labels can reverse the positive code based on prevalence; `--disable_auto_flip_y` preserves the explicit choice. Rows with missing primary outcomes are removed before splitting.

Negative missing/skip codes, including `-3`, are treated as missing. Encoded category `0` represents missingness (`<phi>`). High-cardinality numerical variables are discretized, and nominal variables may be hashed.

## Variable schema

A schema can define types and valid ranges using exact column names or prefixes:

```bash
python -m ca_resmtl --data_glob "data/private/*.xlsx" --output_dir outputs/custom --var_schema_json configs/schema.example.json
```

Supported types are `nominal`, `ordinal`, `binary`, and `continuous`. Optional `min`, `max`, and `drop` fields constrain response codes or exclude columns. Continuous variables still use a discrete representation in the model. The example schema demonstrates the format; dataset-specific rules must reflect the questionnaire.

## Domain splits

The default split requires at least five domains: two for validation, two for testing, and at least one for training. For three or four domains, use `--val_provinces 1 --test_provinces 1`. Every partition must contain both outcome classes.

Default filters exclude outcome-family columns, configured symptom constructs, platform metadata, and location questions. These filters depend on the questionnaire. Direct identifiers and personal free text should be removed from the input.

## Auxiliary targets

Depression targets are derived from the configured depression items and reverse-scored items. Self-harm auxiliary labels use `1 = never` and `2 or above = some self-harm`; missing responses are masked. A depression auxiliary outcome is omitted when depression is the primary task.

Auxiliary labels and domain membership are saved with the encoded responses and reused when retraining the predictor. These local outputs can contain participant-level information and are excluded from the repository.
