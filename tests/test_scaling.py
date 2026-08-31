"""Scaling-experiment result schema and report generation. Pure-function
tests — no training runs — that lock the machine-readable result format and
verify the report never invents missing data points."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def _load(module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, _ROOT / "scripts" / f"{module_name}.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


report = _load("scaling_report")
experiment = _load("scaling_experiment")


def _row(tokens, val, params=51_393_024):
    return {
        "model_parameters": params,
        "training_tokens_budget": tokens,
        "tokens_seen": tokens,
        "unique_tokens": 17_000_000,
        "epochs": round(tokens / 17_000_000, 4),
        "tokens_per_parameter": round(tokens / params, 4),
        "final_train_loss": 2.0,
        "best_val_loss": val,
        "final_val_loss": val + 0.05,
        "best_val_perplexity": round(2.718 ** val, 4),
        "training_time_sec": 100.0,
        "tokens_per_sec": tokens / 100.0,
        "effective_max_steps": tokens // 12288,
        "dataset_version": "aila_pretrain_v1",
        "dataset_sha256": "abc123",
        "language_mixture": None,
        "seed": 1337,
    }


def test_load_results_sorts_and_skips_malformed(tmp_path):
    (tmp_path / "tokens_5000000.json").write_text(json.dumps(_row(5_000_000, 2.4)))
    (tmp_path / "tokens_2000000.json").write_text(json.dumps(_row(2_000_000, 2.6)))
    (tmp_path / "tokens_bad.json").write_text("{ not json")
    rows = report.load_results(tmp_path)
    assert [r["tokens_seen"] for r in rows] == [2_000_000, 5_000_000]  # sorted, bad skipped


def test_markdown_table_has_all_columns():
    table = report.markdown_table([_row(2_000_000, 2.6)])
    for col in ("Tokens seen", "Epochs", "Tokens/param", "Best val loss",
                "Val PPL", "Final val loss", "Time (s)", "Tokens/sec"):
        assert col in table
    assert "2,000,000" in table


def test_empty_report_states_no_data_not_fabricated(tmp_path):
    rep = report.build_report([], tmp_path)
    assert "No result JSONs" in rep
    assert "*" not in rep  # no plotted points invented


def test_ascii_plot_handles_one_and_many_points():
    assert "no data" in report.ascii_plot([], "tokens_seen", "best_val_loss")
    one = report.ascii_plot([_row(2_000_000, 2.6)], "tokens_seen", "best_val_loss")
    assert "*" in one
    many = report.ascii_plot(
        [_row(2_000_000, 2.6), _row(5_000_000, 2.4), _row(10_000_000, 2.3)],
        "tokens_seen", "best_val_loss",
    )
    assert many.count("*") >= 1


def test_report_reports_measured_point_count(tmp_path):
    rows = [_row(2_000_000, 2.6), _row(5_000_000, 2.4)]
    rep = report.build_report(rows, tmp_path)
    assert "Data points: 2 (measured, not interpolated)" in rep
    assert "51,393,024" in rep


def test_language_mixture_none_without_manifest():
    # No manifest on disk for this version -> None, never fabricated.
    assert experiment._language_mixture("no_such_version_xyz") is None
    assert experiment._language_mixture(None) is None


def test_build_model_defaults_to_50m():
    class A:
        model_config = None
        preset = None
    cfg = experiment._build_model(A())
    assert cfg.d_model == 512 and cfg.n_layers == 12
