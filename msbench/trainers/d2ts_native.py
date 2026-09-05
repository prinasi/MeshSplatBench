"""Native 2DTS training integration."""

from __future__ import annotations

from copy import deepcopy
import re
import time
from pathlib import Path
from typing import Any, Mapping

from msbench.core.config import Config, merge_dicts


_DEFAULT_SYNTHETIC_RANDOM_INIT = {
    "bbox_list": [[-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]],
    "point_num_list": [100000],
    "normal_list": ["random"],
}
_CHECKPOINT_RE = re.compile(r"^(\d+)\.ckpt$")
_POINT_CLOUD_RE = re.compile(r"^(\d+)\.ply$")


def run_d2ts_native_config(
    *,
    cfg: Mapping[str, Any],
    dataset_root: str,
    output_dir: Path,
    max_steps: int,
    quiet: bool = True,
) -> dict[str, Any]:
    """Run the bundled 2DTS VanillaTS trainer from a MeshSplatBench config."""
    from msbench.vendor.d2ts.diff_recon import VanillaTSTrainer

    native = _build_d2ts_native_config(
        cfg=cfg,
        dataset_root=dataset_root,
        output_dir=output_dir,
        max_steps=max_steps,
        quiet=quiet,
    )

    start_step = 0
    if getattr(native.trainer, "start_checkpoint", None) is not None:
        start_step = int(native.trainer.start_checkpoint)
        print(f"Loaded checkpoint: {output_dir / 'ckpt' / f'{start_step}.ckpt'}")
    elif getattr(native.trainer, "start_pointcloud", None) is not None:
        start_step = int(native.trainer.start_pointcloud)
        print(f"Loaded checkpoint: {output_dir / 'ckpt' / 'point_cloud' / f'{start_step}.ply'}")

    if start_step >= max_steps:
        return {
            "total_steps": int(max_steps),
            "start_step": start_step,
            "trained_steps": 0,
            "total_time_s": 0.0,
            "avg_step_time_ms": 0.0,
            "final_losses": {},
        }

    start = time.time()
    trainer = VanillaTSTrainer(native, exp_name=output_dir.name, device=None, log_file=not quiet)
    trainer.train()
    total_time = time.time() - start
    trained_steps = max(int(max_steps) - start_step, 0)
    return {
        "total_steps": int(max_steps),
        "start_step": start_step,
        "trained_steps": trained_steps,
        "total_time_s": total_time,
        "avg_step_time_ms": total_time * 1000.0 / max(trained_steps, 1),
        "final_losses": {},
    }


def _build_d2ts_native_config(
    *,
    cfg: Mapping[str, Any],
    dataset_root: str,
    output_dir: Path,
    max_steps: int,
    quiet: bool = True,
):
    """Build the native 2DTS config without launching training."""
    from msbench.vendor.d2ts.diff_recon.utils.config import configToDict, dictToConfig, loadConfig

    tri_cfg = Config(cfg).to_dict()
    d2ts_cfg = tri_cfg.get("d2ts", {}) or {}
    native_config = d2ts_cfg.get("native_config")
    if native_config is None:
        raise ValueError("2DTS configs must set d2ts.native_config.")

    native_path = Path(str(native_config)).expanduser()
    if not native_path.is_absolute():
        native_path = Path.cwd() / native_path
    native = loadConfig(str(native_path))
    native_dict = configToDict(native)

    overrides = d2ts_cfg.get("native_overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise TypeError("d2ts.native_overrides must be a mapping.")
    native_dict = merge_dicts(native_dict, overrides)
    native = dictToConfig(native_dict)
    native.__setattr__("_config_path", str(native_path))

    dataset_cfg = tri_cfg.get("dataset", {}) or {}
    scene = str(dataset_cfg.get("scene") or Path(dataset_root).name)
    dataset_path = Path(dataset_root).expanduser()
    if dataset_path.name == scene:
        native.dataset.local_dir = str(dataset_path.parent)
        native.dataset.scene_id = scene
    else:
        native.dataset.local_dir = str(dataset_path)
        native.dataset.scene_id = scene

    resolution = dataset_cfg.get("resolution")
    if resolution is not None:
        native.dataset.train_target_res = int(resolution)
        native.dataset.test_target_res = int(resolution)
    image_dir = dataset_cfg.get("image_dir")
    if image_dir is not None:
        native.dataset.image_dir = str(image_dir)
    dataset_type = str(dataset_cfg.get("type", dataset_cfg.get("dataset_type", ""))).lower()
    synthetic_dataset = dataset_type in {
        "synthetic",
        "blender",
        "nerf_synthetic",
        "nerf-synthetic",
    }
    if synthetic_dataset:
        background = getattr(native.dataset, "background", None)
        if background is None:
            background = "random"
        native.dataset.background = str(background)
        test_background = getattr(native.dataset, "test_background", None)
        if test_background is None:
            test_background = "white"
        native.dataset.test_background = str(test_background)
        native.dataset.dtu_use_alpha = bool(dataset_cfg.get("use_alpha", True))
        random_init = getattr(native.model, "random_init", None)
        if getattr(native.dataset, "pcd_path", None) is None and not _has_complete_random_init(random_init):
            native.model.random_init = dictToConfig(deepcopy(_DEFAULT_SYNTHETIC_RANDOM_INIT))
    else:
        dtu_eval_mode = str(dataset_cfg.get("dtu_eval_mode", "full")).lower()
        native.dataset.dtu_eval_mode = dtu_eval_mode
        native.dataset.dtu_use_alpha = dtu_eval_mode == "foreground"
        if dataset_type == "dtu" or str(dataset_cfg.get("name", "")).lower() == "dtu":
            # Keep the native renderer background aligned with the image
            # loading/evaluation protocol.  Foreground mode composites RGBA GT
            # against each camera's random training colour and uses white at
            # test time.  Full mode preserves the source RGB, so a random
            # renderer background would inject an unmatched loss signal; use
            # the benchmark's deterministic black background for both splits.
            if dtu_eval_mode == "foreground":
                native.dataset.background = "random"
                native.dataset.test_background = "white"
            else:
                native.dataset.background = "black"
                native.dataset.test_background = "black"

    native.trainer.output_dir = str(output_dir.parent)
    native.trainer.iterations = int(max_steps)
    _retarget_iteration_list(native.trainer, "save_iterations", max_steps)
    _retarget_iteration_list(native.trainer, "checkpoint_iterations", max_steps, force=True)
    _retarget_iteration_list(native.trainer, "save_mesh_iterations", max_steps)
    _retarget_iteration_list(native.trainer, "save_pcd_iterations", max_steps)

    clean_output_dir = d2ts_cfg.get("clean_output_dir")
    if clean_output_dir is not None:
        native.trainer.clean_output_dir = bool(clean_output_dir)
    use_tensorboard = d2ts_cfg.get("use_tensorboard")
    if use_tensorboard is not None:
        native.trainer.use_tensorboard = bool(use_tensorboard)
    if quiet:
        native.trainer.use_tensorboard = False
        native.trainer.save_train_img = False
        native.trainer.save_eval_img = False

    # Keep native 2DTS DTU semantics: foreground mode means RGBA loading and
    # white-background compositing, but the published DTU protocol trains and
    # evaluates on the full composited image. Users can still opt into masked
    # losses explicitly through d2ts.native_overrides.trainer.*_alpha_mask.

    target_point_num = d2ts_cfg.get("target_point_num")
    max_primitives = (tri_cfg.get("trainer", {}) or {}).get("max_primitives")
    if max_primitives is not None:
        cap = int(max_primitives)
        if cap <= 0:
            raise ValueError(f"trainer.max_primitives must be positive, got {cap!r}")
        target_point_num = cap if target_point_num is None else min(int(target_point_num), cap)
        if getattr(native.model, "model_update", None) is not None:
            native.model.model_update.max_primitives = cap
    if target_point_num is not None:
        densification = getattr(native.model.model_update, "densification", None)
        if densification is not None:
            densification.target_point_num = int(target_point_num)

    resume = _find_latest_d2ts_checkpoint(output_dir)
    if resume is not None:
        resume_kind, resume_step = resume
        if resume_step >= max_steps:
            return native
        if resume_kind == "checkpoint":
            native.trainer.start_checkpoint = int(resume_step)
            native.trainer.start_pointcloud = None
        else:
            native.trainer.start_pointcloud = int(resume_step)
            native.trainer.start_checkpoint = None

    return native


def _has_complete_random_init(config: Any) -> bool:
    if config is None:
        return False
    return all(getattr(config, name, None) for name in ("bbox_list", "point_num_list", "normal_list"))


def _retarget_iteration_list(obj: Any, name: str, max_steps: int, *, force: bool = False) -> None:
    value = getattr(obj, name, None)
    if force or value is not None:
        setattr(obj, name, [int(max_steps)])


def _find_latest_d2ts_checkpoint(output_dir: str | Path) -> tuple[str, int] | None:
    ckpt_root = Path(output_dir).expanduser() / "ckpt"
    candidates: list[tuple[int, str]] = []

    checkpoint_dir = ckpt_root
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.iterdir():
            if not path.is_file():
                continue
            match = _CHECKPOINT_RE.match(path.name)
            if match is None:
                continue
            candidates.append((int(match.group(1)), "checkpoint"))

    point_cloud_dir = ckpt_root / "point_cloud"
    if point_cloud_dir.is_dir():
        for path in point_cloud_dir.iterdir():
            if not path.is_file():
                continue
            match = _POINT_CLOUD_RE.match(path.name)
            if match is None:
                continue
            candidates.append((int(match.group(1)), "pointcloud"))

    if not candidates:
        return None
    step, kind = max(candidates, key=lambda item: item[0])
    return kind, step
