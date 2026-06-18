"""Report generation for TriangleBench evaluation runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from trianglebench.core.stats import ModelStats


def write_stats_json(stats: ModelStats, path: str | Path) -> None:
    """Write model statistics to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_json(path)


def write_profile_json(profile: dict[str, Any], path: str | Path) -> None:
    """Write profiling results to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile, indent=2))


def write_metrics_json(metrics: dict[str, Any], path: str | Path) -> None:
    """Write evaluation metrics to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))


def generate_markdown_report(
    stats: ModelStats | None = None,
    profile: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> str:
    """Generate a human-readable Markdown report.
    
    Args:
        stats: Model statistics.
        profile: Profiling results.
        metrics: Evaluation metrics.
        
    Returns:
        Markdown-formatted report string.
    """
    lines = [
        "# TriangleBench Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    if stats is not None:
        lines.extend([
            "## Model Statistics",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Method | {stats.method} |",
            f"| Scene | {stats.scene} |",
            f"| Primitive Count | {stats.primitive_count:,} |",
            f"| Vertex Count | {stats.vertex_count:,} |",
            f"| Face Count | {stats.face_count:,} |",
            f"| Trainable Params | {stats.trainable_param_count:,} |",
            f"| Checkpoint Size | {stats.checkpoint_size_mb:.1f} MB |",
            f"| Forward Time | {stats.forward_ms:.2f} ms |",
            f"| Backward Time | {stats.backward_ms:.2f} ms |",
            f"| Render FPS | {stats.render_fps:.1f} |",
            f"| Peak Memory (Forward) | {stats.peak_memory_forward_gb:.2f} GB |",
            f"| Peak Memory (Train) | {stats.peak_memory_train_gb:.2f} GB |",
            "",
        ])

        if stats.triangle_area_mean > 0:
            lines.extend([
                "### Triangle Diagnostics",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Area (mean) | {stats.triangle_area_mean:.6f} |",
                f"| Area (median) | {stats.triangle_area_median:.6f} |",
                f"| Area (p95) | {stats.triangle_area_p95:.6f} |",
                f"| Opacity (mean) | {stats.opacity_mean:.4f} |",
                f"| Opacity (median) | {stats.opacity_median:.4f} |",
                f"| Visible Ratio | {stats.visible_primitive_ratio:.4f} |",
                f"| Screen Radii (mean) | {stats.screen_radii_mean:.2f} |",
                f"| Screen Radii (p95) | {stats.screen_radii_p95:.2f} |",
                "",
            ])

        if stats.backend_extra:
            lines.extend([
                "### Backend-Specific Fields",
                "",
                f"| Field | Value |",
                f"|-------|-------|",
            ])
            for k, v in stats.backend_extra.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

    if profile is not None:
        lines.extend([
            "## Profiling Results",
            "",
        ])
        if "forward" in profile:
            fwd = profile["forward"]
            lines.extend([
                "### Forward Pass",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Mean | {fwd.get('mean_ms', 0):.2f} ms |",
                f"| Std | {fwd.get('std_ms', 0):.2f} ms |",
                f"| P50 | {fwd.get('p50_ms', 0):.2f} ms |",
                f"| P95 | {fwd.get('p95_ms', 0):.2f} ms |",
                f"| FPS | {fwd.get('fps', 0):.1f} |",
                "",
            ])
        if "backward" in profile:
            bwd = profile["backward"]
            lines.extend([
                "### Backward Pass",
                "",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Mean | {bwd.get('mean_ms', 0):.2f} ms |",
                f"| P95 | {bwd.get('p95_ms', 0):.2f} ms |",
                "",
            ])

    if metrics is not None:
        lines.extend([
            "## Evaluation Metrics",
            "",
        ])
        aggregated = metrics.get("aggregated", metrics)
        lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
        ])
        for key, value in aggregated.items():
            if isinstance(value, float):
                lines.append(f"| {key} | {value:.4f} |")
            else:
                lines.append(f"| {key} | {value} |")
        lines.append("")

        num_views = metrics.get("num_views", "N/A")
        lines.append(f"Number of views evaluated: {num_views}")
        lines.append("")

    return "\n".join(lines)


def save_run_report(
    run_dir: str | Path,
    stats: ModelStats | None = None,
    profile: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Save a complete run report to a directory.
    
    Creates:
        run_dir/
            config.yaml (if config provided)
            stats.json (if stats provided)
            profile.json (if profile provided)
            metrics.json (if metrics provided)
            report.md (always generated)
    
    Args:
        run_dir: Output directory.
        stats: Model statistics.
        profile: Profiling results.
        metrics: Evaluation metrics.
        config: Run configuration.
        
    Returns:
        Path to the generated report.md.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if config is not None:
        import yaml
        (run_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    if stats is not None:
        write_stats_json(stats, run_dir / "stats.json")

    if profile is not None:
        write_profile_json(profile, run_dir / "profile.json")

    if metrics is not None:
        write_metrics_json(metrics, run_dir / "metrics.json")

    report_path = run_dir / "report.md"
    report = generate_markdown_report(stats, profile, metrics)
    report_path.write_text(report)

    return report_path
