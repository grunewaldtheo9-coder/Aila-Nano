#!/usr/bin/env python3
"""Turn the JSON results from scripts/scaling_experiment.py into a scaling
report: a Markdown table (Tokens | Best Val Loss | PPL | Time | Tokens/sec,
plus tokens/parameter and epochs) and two plots — validation loss vs
training tokens, and validation loss vs tokens-per-parameter.

Reads only the result JSONs that exist; it never invents missing points.
If matplotlib is unavailable, it still writes the table and an ASCII plot.

Example:
    python scripts/scaling_report.py \
        --results-dir experiments/50m_data_scaling \
        --out experiments/50m_data_scaling/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(results_dir.glob("tokens_*.json")):
        try:
            rows.append(json.loads(f.read_text()))
        except Exception:
            continue
    rows.sort(key=lambda r: r.get("tokens_seen", 0))
    return rows


def markdown_table(rows: list[dict]) -> str:
    header = (
        "| Tokens seen | Epochs | Tokens/param | Best val loss | Val PPL | "
        "Final val loss | Time (s) | Tokens/sec |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r['tokens_seen']:,} | {r.get('epochs','?')} | "
            f"{r.get('tokens_per_parameter','?')} | {r['best_val_loss']} | "
            f"{r['best_val_perplexity']} | {r.get('final_val_loss','?')} | "
            f"{r.get('training_time_sec','?')} | {r.get('tokens_per_sec','?')} |"
        )
    return header + "\n".join(lines)


def ascii_plot(rows: list[dict], x_key: str, y_key: str, width: int = 50, height: int = 12) -> str:
    pts = [(r[x_key], r[y_key]) for r in rows if x_key in r and y_key in r]
    if len(pts) < 1:
        return "(no data points)"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    grid = [[" "] * width for _ in range(height)]
    for x, y in pts:
        cx = 0 if xmax == xmin else int((x - xmin) / (xmax - xmin) * (width - 1))
        cy = 0 if ymax == ymin else int((ymax - y) / (ymax - ymin) * (height - 1))
        grid[cy][cx] = "*"
    body = "\n".join("|" + "".join(r) for r in grid)
    return (
        f"  y: {y_key} [{ymin:.4f} .. {ymax:.4f}]   x: {x_key} [{xmin:,} .. {xmax:,}]\n"
        + body + "\n+" + "-" * width
    )


def try_matplotlib_plots(rows: list[dict], out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    written = []
    for x_key, fname, xlabel in (
        ("tokens_seen", "loss_vs_tokens.png", "training tokens"),
        ("tokens_per_parameter", "loss_vs_tokens_per_param.png", "tokens / parameter"),
    ):
        pts = [(r[x_key], r["best_val_loss"]) for r in rows if x_key in r]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, "o-")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("best validation loss")
        ax.set_title("Aila Nano data scaling")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(str(path))
    return written


def build_report(rows: list[dict], out_dir: Path) -> str:
    if not rows:
        return "# Aila Nano data-scaling report\n\n_No result JSONs found yet._\n"
    params = rows[0].get("model_parameters", "?")
    ver = rows[0].get("dataset_version") or "(unversioned)"
    plots = try_matplotlib_plots(rows, out_dir)
    parts = [
        "# Aila Nano data-scaling report",
        "",
        f"- Model parameters: **{params:,}**" if isinstance(params, int) else f"- Model parameters: {params}",
        f"- Dataset version: `{ver}`",
        f"- Data points: {len(rows)} (measured, not interpolated)",
        "",
        "## Results",
        "",
        markdown_table(rows),
        "",
        "## Validation loss vs training tokens",
        "",
        "```",
        ascii_plot(rows, "tokens_seen", "best_val_loss"),
        "```",
        "",
        "## Validation loss vs tokens per parameter",
        "",
        "```",
        ascii_plot(rows, "tokens_per_parameter", "best_val_loss"),
        "```",
    ]
    if plots:
        parts += ["", "## Plots", ""] + [f"- `{p}`" for p in plots]
    return "\n".join(parts) + "\n"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="experiments/50m_data_scaling")
    p.add_argument("--out", default=None, help="Markdown output path (default: <results-dir>/REPORT.md)")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    rows = load_results(results_dir)
    out = Path(args.out) if args.out else results_dir / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(rows, out.parent)
    out.write_text(report)
    print(report)
    print(f"\n[written {out}]")


if __name__ == "__main__":
    main()
