"""Render commands: split renders, videos, viewer, and mesh export."""

from pathlib import Path

import typer

render_app = typer.Typer(no_args_is_help=True)
RESOLUTION_HELP = (
    "1/2/4/8 downscale factor; other positive values are target width; "
    "-1 caps large images"
)


def _load(method: str, checkpoint: str):
    from trianglebench.core.rendering import load_adapter

    return load_adapter(method, checkpoint)


def _load_dataset(
    dataset: str,
    dataset_type: str,
    split: str,
    eval_every: int,
    image_dir: str,
    resolution: int,
):
    from trianglebench.core.datasets import load_dataset

    return load_dataset(
        dataset,
        dataset_type=dataset_type,
        split=split,
        eval_every=eval_every,
        image_dir=image_dir,
        resolution=resolution,
    )


@render_app.callback()
def render():
    """Render images, videos, meshes, and online viewer previews."""


@render_app.command("images")
def render_images(
    method: str = typer.Option(..., "--method", "-m", help="Method name"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to model checkpoint"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Path to dataset directory"),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split: train/test/all"),
    output_dir: str = typer.Option("renders/", "--output-dir", "-o", help="Output directory"),
    dataset_type: str = typer.Option("auto", "--dataset-type", help="auto/colmap/mipnerf360/tanks/dtu/blender"),
    image_dir: str = typer.Option("images", "--image-dir", help="COLMAP image directory"),
    resolution: int = typer.Option(
        1,
        "--resolution",
        "-r",
        help=RESOLUTION_HELP,
    ),
    eval_every: int = typer.Option(8, "--eval-every", help="COLMAP/DTU holdout stride"),
    save_gt: bool = typer.Option(False, "--save-gt", help="Also save ground-truth images"),
    save_aux: bool = typer.Option(False, "--save-aux", help="Save depth/alpha when available"),
):
    """Render images from a trained model for one dataset split."""
    from trianglebench.core.rendering import render_dataset_split

    adapter = _load(method, checkpoint)
    ds = _load_dataset(dataset, dataset_type, split, eval_every, image_dir, resolution)
    manifest = render_dataset_split(adapter, ds, output_dir, save_gt=save_gt, save_aux=save_aux)
    typer.echo(f"Rendered {manifest['num_frames']} frames to {output_dir}")


@render_app.command("split")
def render_split(
    method: str = typer.Option(..., "--method", "-m"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    dataset: str = typer.Option(..., "--dataset", "-d"),
    split: str = typer.Option("test", "--split", "-s"),
    output_dir: str = typer.Option("renders/", "--output-dir", "-o"),
    dataset_type: str = typer.Option("auto", "--dataset-type"),
    image_dir: str = typer.Option("images", "--image-dir"),
    resolution: int = typer.Option(1, "--resolution", "-r", help=RESOLUTION_HELP),
    eval_every: int = typer.Option(8, "--eval-every"),
    save_gt: bool = typer.Option(True, "--save-gt/--no-save-gt"),
    save_aux: bool = typer.Option(False, "--save-aux"),
    metrics: bool = typer.Option(False, "--metrics"),
):
    """Render train/test/all images, optionally computing image metrics."""
    from trianglebench.core.rendering import render_dataset_split

    adapter = _load(method, checkpoint)
    ds = _load_dataset(dataset, dataset_type, split, eval_every, image_dir, resolution)
    manifest = render_dataset_split(
        adapter,
        ds,
        output_dir,
        save_gt=save_gt,
        save_aux=save_aux,
        metrics=metrics,
    )
    typer.echo(f"Rendered {manifest['num_frames']} frames to {output_dir}")


@render_app.command("video")
def render_video_cmd(
    method: str = typer.Option(..., "--method", "-m"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    dataset: str = typer.Option(..., "--dataset", "-d"),
    split: str = typer.Option("train", "--split", "-s"),
    output_dir: str = typer.Option("video/", "--output-dir", "-o"),
    dataset_type: str = typer.Option("auto", "--dataset-type"),
    image_dir: str = typer.Option("images", "--image-dir"),
    resolution: int = typer.Option(1, "--resolution", "-r", help=RESOLUTION_HELP),
    eval_every: int = typer.Option(8, "--eval-every"),
    frames: int = typer.Option(240, "--frames"),
    fps: int = typer.Option(30, "--fps"),
    zoom: float = typer.Option(1.0, "--zoom"),
    z_variation: float = typer.Option(0.0, "--z-variation"),
    z_phase: float = typer.Option(0.0, "--z-phase", help="Phase offset for vertical oscillation [0, 1]"),
):
    """Render a PCA-aligned ellipse trajectory video from dataset cameras."""
    from trianglebench.core.rendering import generate_ellipse_cameras, render_video

    adapter = _load(method, checkpoint)
    ds = _load_dataset(dataset, dataset_type, split, eval_every, image_dir, resolution)
    cameras = generate_ellipse_cameras(
        ds.get_all_cameras(),
        n_frames=frames,
        zoom=zoom,
        z_variation=z_variation,
        z_phase=z_phase,
    )
    video_path = render_video(adapter, cameras, output_dir, fps=fps)
    typer.echo(f"Video saved to {video_path}")


@render_app.command("viewer")
def viewer(
    method: str = typer.Option(..., "--method", "-m"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    dataset: str = typer.Option(..., "--dataset", "-d"),
    split: str = typer.Option("test", "--split", "-s"),
    dataset_type: str = typer.Option("auto", "--dataset-type"),
    image_dir: str = typer.Option("images", "--image-dir"),
    resolution: int = typer.Option(1, "--resolution", "-r", help=RESOLUTION_HELP),
    eval_every: int = typer.Option(8, "--eval-every"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(7007, "--port"),
):
    """Start a lightweight online viewer for dataset cameras."""
    from trianglebench.core.viewer import serve_viewer

    adapter = _load(method, checkpoint)
    ds = _load_dataset(dataset, dataset_type, split, eval_every, image_dir, resolution)
    serve_viewer(adapter, ds, host=host, port=port)


@render_app.command("mesh")
def mesh(
    method: str = typer.Option(..., "--method", "-m"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c"),
    output: str = typer.Option("mesh.ply", "--output", "-o"),
):
    """Export the adapter primitive as a PLY mesh/triangle soup."""
    from trianglebench.core.mesh_eval import export_adapter_mesh

    adapter = _load(method, checkpoint)
    path = export_adapter_mesh(adapter, output)
    typer.echo(f"Mesh saved to {path}")
