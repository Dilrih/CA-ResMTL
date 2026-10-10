import json

import pytest

from ca_resmtl.cli import main, parse_args


def test_config_and_explicit_option_precedence(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"epochs_pred": 7, "device": "cpu", "use_multitask": False}))
    args = parse_args(["--config", str(path), "--epochs_pred", "3"])
    assert args.epochs_pred == 3 and args.device == "cpu" and not args.use_multitask
    assert {"epochs_pred", "device", "use_multitask"} <= set(args._explicit_options)


@pytest.mark.parametrize(
    "flags",
    [
        ["--ate_only", "--stage2_only"],
        ["--ate_only", "--skip_stage3"],
        ["--epochs_pred", "0"],
        ["--loss_type", "focal"],
        ["--emb_dim", "15", "--heads", "4"],
    ],
)
def test_invalid_settings_fail_before_running(flags):
    with pytest.raises(SystemExit):
        parse_args(flags)


def test_unknown_config_option_is_rejected(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"epochs_predd": 2}))
    with pytest.raises(SystemExit):
        parse_args(["--config", str(path)])


def test_console_entrypoint_returns_success(monkeypatch, tmp_path):
    from ca_resmtl import pipeline

    monkeypatch.setattr(pipeline, "run_experiment", lambda args: tmp_path)
    assert main(["--device", "cpu"]) is None
