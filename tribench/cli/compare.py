"""Compare command: aggregate results across methods."""

import json
from pathlib import Path
from typing import Optional

import typer

from tribench.cli.config import load_cli_config, section

compare_app = typer.Typer(no_args_is_help=True)


@compare_app.callback(invoke_without_command=True)
def compare(
    inputs: Optional[list[str]] = typer.Option(None, "--inputs", "-i", help="Paths to JSON files (metrics/profile/stats)"),
    config: Optional[Path] = typer.Option(None, "--config", help="TriBench comparison config YAML"),
    output: str = typer.Option("report.md", "--output", "-o", help="Output report file path"),
):
    """Compare results across multiple methods.
    
    Aggregates JSON files from different runs and produces:
    - Comparison tables
    - Tradeoff analysis (quality vs memory vs speed vs primitive count)
    - Markdown report
    """
    from tribench.core.stats import ModelStats
    from tribench.core.reports import generate_markdown_report

    if config is not None:
        cfg = load_cli_config(config)
        compare_cfg = section(cfg, "compare")
        inputs = inputs or compare_cfg.get("inputs")
        if output == "report.md":
            output = str(compare_cfg.get("output") or section(cfg, "output").get("report_file", output))
    if not inputs:
        raise typer.BadParameter("Use --config, or provide at least one --inputs value.")

    typer.echo(f"Comparing {len(inputs)} input files...")

    all_stats = []
    all_profiles = []
    all_metrics = []

    for input_path in inputs:
        path = Path(input_path)
        if not path.exists():
            typer.echo(f"Warning: {path} not found, skipping.", err=True)
            continue

        data = json.loads(path.read_text())

        # Auto-detect file type by content
        if "primitive_count" in data:
            stats = ModelStats.from_dict(data)
            all_stats.append(stats)
        elif "forward" in data or "mean_ms" in data:
            all_profiles.append(data)
        elif "psnr_mean" in data or "aggregated" in data:
            all_metrics.append(data)

    # Generate comparison tables
    lines = ["# TriBench Comparison Report", ""]

    if all_stats:
        lines.extend(_generate_stats_table(all_stats))

    if all_metrics:
        lines.extend(_generate_metrics_table(all_metrics))

    if all_profiles:
        lines.extend(_generate_profile_table(all_profiles))

    report = "\n".join(lines)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)

    typer.echo(f"Comparison report saved to {output_path}")


def _generate_stats_table(stats_list: list) -> list[str]:
    """Generate a comparison table for model statistics."""
    lines = [
        "## Model Statistics Comparison",
        "",
        "| Method | Scene | Primitives | Vertices | Params | Checkpoint (MB) |",
        "|--------|-------|------------|----------|--------|-----------------|",
    ]
    for s in stats_list:
        lines.append(
            f"| {s.method} | {s.scene} | {s.primitive_count:,} | "
            f"{s.vertex_count:,} | {s.trainable_param_count:,} | "
            f"{s.checkpoint_size_mb:.1f} |"
        )
    lines.append("")
    return lines


def _generate_metrics_table(metrics_list: list) -> list[str]:
    """Generate a comparison table for evaluation metrics."""
    lines = [
        "## Quality Metrics Comparison",
        "",
        "| # | PSNR | SSIM | LPIPS | Views |",
        "|---|------|------|-------|-------|",
    ]
    for i, m in enumerate(metrics_list):
        agg = m.get("aggregated", m)
        psnr = agg.get("psnr_mean", "N/A")
        ssim = agg.get("ssim_mean", "N/A")
        lpips = agg.get("lpips_mean", "N/A")
        n = m.get("num_views", "N/A")
        psnr_str = f"{psnr:.2f}" if isinstance(psnr, float) else str(psnr)
        ssim_str = f"{ssim:.4f}" if isinstance(ssim, float) else str(ssim)
        lpips_str = f"{lpips:.4f}" if isinstance(lpips, float) else str(lpips)
        lines.append(f"| {i + 1} | {psnr_str} | {ssim_str} | {lpips_str} | {n} |")
    lines.append("")
    return lines


def _generate_profile_table(profile_list: list) -> list[str]:
    """Generate a comparison table for profiling results."""
    lines = [
        "## Performance Comparison",
        "",
        "| # | Forward (ms) | FPS | Backward (ms) |",
        "|---|-------------|-----|---------------|",
    ]
    for i, p in enumerate(profile_list):
        fwd = p.get("forward", p)
        bwd = p.get("backward", {})
        fwd_ms = fwd.get("mean_ms", "N/A")
        fps = fwd.get("fps", "N/A")
        bwd_ms = bwd.get("mean_ms", "N/A")
        fwd_str = f"{fwd_ms:.2f}" if isinstance(fwd_ms, float) else str(fwd_ms)
        fps_str = f"{fps:.1f}" if isinstance(fps, float) else str(fps)
        bwd_str = f"{bwd_ms:.2f}" if isinstance(bwd_ms, float) else str(bwd_ms)
        lines.append(f"| {i + 1} | {fwd_str} | {fps_str} | {bwd_str} |")
    lines.append("")
    return lines
