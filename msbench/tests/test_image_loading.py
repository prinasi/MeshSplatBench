"""Tests for image loading details that affect benchmark metrics."""

from __future__ import annotations

import json
import numpy as np
import torch
from PIL import Image
from types import SimpleNamespace

from msbench.core.builder import build_dataset
from msbench.core.datasets import (
    DTUDataset,
    NerfSyntheticDataset,
    _load_image,
    load_dataset,
)
from msbench.trainers.triangle_splatting_method import _pil_rgb_uint8
from msbench.vendor.dtu_utils import load_dtu_pil_image
from msbench.vendor.training_images import is_dtu_scene, load_rgba_for_training
from msbench.vendor.triangle_splatting.scene.dataset_readers import readDTUSceneInfo
from msbench.vendor.triangle_splatting.utils.general_utils import PILtoTorch


def _write_basic_ply(path):
    path.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 1\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
        "0 0 0 0 0 1 128 128 128\n",
        encoding="utf-8",
    )


def test_load_image_ignores_rgba_alpha_when_resizing(tmp_path):
    path = tmp_path / "rgba.png"
    image = Image.new("RGBA", (2, 2))
    image.putdata(
        [
            (200, 100, 50, 0),
            (200, 100, 50, 64),
            (200, 100, 50, 128),
            (200, 100, 50, 255),
        ]
    )
    image.save(path)

    loaded = _load_image(path, size=(1, 1))

    expected = torch.tensor([200 / 255.0, 100 / 255.0, 50 / 255.0])
    assert torch.allclose(loaded[0, 0], expected, atol=1 / 255)


def test_load_image_composites_rgba_when_background_is_given(tmp_path):
    path = tmp_path / "rgba.png"
    image = Image.new("RGBA", (1, 1), (20, 40, 80, 0))
    image.save(path)

    loaded = _load_image(path, bg_color=(1.0, 1.0, 1.0))

    assert torch.allclose(loaded[0, 0], torch.ones(3), atol=1 / 255)


def test_nerf_synthetic_dataset_composites_rgba_to_white_by_default(tmp_path):
    root = tmp_path / "lego"
    (root / "train").mkdir(parents=True)
    Image.new("RGBA", (1, 1), (20, 40, 80, 0)).save(root / "train" / "r_0.png")
    (root / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.6911112070083618,
                "frames": [
                    {
                        "file_path": "train/r_0",
                        "transform_matrix": np.eye(4, dtype=np.float32).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_dataset(root, dataset_type="nerf_synthetic", split="train")
    sample = dataset.sample(0)

    assert torch.allclose(sample.image[0, 0], torch.ones(3), atol=1 / 255)
    assert sample.mask is not None
    assert torch.allclose(sample.mask[0, 0], torch.zeros(1), atol=1 / 255)
    assert sample.metadata["eval_background_color"] == (1.0, 1.0, 1.0)


def test_nerf_synthetic_dataset_can_disable_alpha_compositing(tmp_path):
    root = tmp_path / "lego"
    (root / "train").mkdir(parents=True)
    Image.new("RGBA", (1, 1), (20, 40, 80, 0)).save(root / "train" / "r_0.png")
    (root / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.6911112070083618,
                "frames": [
                    {
                        "file_path": "train/r_0",
                        "transform_matrix": np.eye(4, dtype=np.float32).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_dataset(
        root,
        dataset_type="nerf_synthetic",
        split="train",
        background=None,
        use_alpha=False,
    )
    sample = dataset.sample(0)

    assert torch.allclose(sample.image[0, 0], torch.tensor([20, 40, 80]) / 255.0, atol=1 / 255)
    assert sample.mask is None


def test_build_dataset_accepts_synthetic_alias(tmp_path):
    root = tmp_path / "lego"
    (root / "train").mkdir(parents=True)
    Image.new("RGBA", (1, 1), (20, 40, 80, 255)).save(root / "train" / "r_0.png")
    (root / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.6911112070083618,
                "frames": [
                    {
                        "file_path": "train/r_0",
                        "transform_matrix": np.eye(4, dtype=np.float32).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = build_dataset(
        {
            "type": "synthetic",
            "root": str(root),
            "split": "train",
            "background": "white",
            "use_alpha": True,
        }
    )

    assert isinstance(dataset, NerfSyntheticDataset)
    assert len(dataset) == 1


def test_dtu_dataset_full_mode_ignores_alpha_and_external_mask(tmp_path):
    root = tmp_path / "scan24"
    (root / "images").mkdir(parents=True)
    (root / "mask").mkdir()

    image = Image.new("RGBA", (2, 1))
    image.putdata([(20, 40, 80, 0), (10, 30, 50, 255)])
    image.save(root / "images" / "0000.png")

    external_mask = Image.new("L", (2, 1), 0)
    external_mask.save(root / "mask" / "000.png")

    K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    extrinsic = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    world_mat = np.eye(4, dtype=np.float32)
    world_mat[:3, :4] = K @ extrinsic
    np.savez(
        root / "cameras.npz",
        world_mat_0=world_mat,
        scale_mat_0=np.eye(4, dtype=np.float32),
    )

    sample = DTUDataset(root, split="all", resolution=1, dtu_eval_mode="full").sample(0)

    assert torch.allclose(sample.image[0, 0], torch.tensor([20, 40, 80]) / 255.0, atol=1 / 255)
    assert torch.allclose(sample.image[0, 1], torch.tensor([10, 30, 50]) / 255.0, atol=1 / 255)
    assert sample.mask is None
    assert sample.metadata["dtu_eval_mode"] == "full"
    assert "eval_background_color" not in sample.metadata


def test_dtu_dataset_foreground_composites_mask_to_white(tmp_path):
    root = tmp_path / "scan24"
    (root / "images").mkdir(parents=True)
    (root / "mask").mkdir()

    image = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    image.putalpha(Image.new("L", (4, 4), 255))
    image.getchannel("A").save(root / "mask" / "000.png")
    alpha = Image.new("L", (4, 4), 0)
    alpha.putdata([255, 255, 0, 0] * 4)
    image.putalpha(alpha)
    image.save(root / "images" / "0000.png")

    K = np.array(
        [[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    extrinsic = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    world_mat = np.eye(4, dtype=np.float32)
    world_mat[:3, :4] = K @ extrinsic
    np.savez(
        root / "cameras.npz",
        world_mat_0=world_mat,
        scale_mat_0=np.eye(4, dtype=np.float32),
    )

    dataset = DTUDataset(root, split="all", resolution=2, dtu_eval_mode="foreground")
    sample = dataset.sample(0)
    expected_rgb, expected_mask = load_rgba_for_training(
        load_dtu_pil_image(root, root / "images" / "0000.png"),
        (2, 2),
        PILtoTorch,
        composite_white=True,
    )
    expected_image = expected_rgb.permute(1, 2, 0)
    expected_mask = expected_mask.permute(1, 2, 0)

    assert sample.image.shape == (2, 2, 3)
    assert torch.allclose(sample.image, expected_image, atol=1 / 255)
    assert sample.mask is not None
    assert torch.allclose(sample.mask, expected_mask, atol=1 / 255)


def test_dtu_dataset_prefers_rgba_alpha_without_double_compositing(tmp_path):
    root = tmp_path / "scan24"
    (root / "images").mkdir(parents=True)
    (root / "mask").mkdir()

    image = Image.new("RGBA", (2, 1))
    image.putdata([(255, 0, 0, 128), (0, 255, 0, 255)])
    image.save(root / "images" / "0000.png")

    K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
    extrinsic = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    world_mat = np.eye(4, dtype=np.float32)
    world_mat[:3, :4] = K @ extrinsic
    np.savez(
        root / "cameras.npz",
        world_mat_0=world_mat,
        scale_mat_0=np.eye(4, dtype=np.float32),
    )

    sample = DTUDataset(root, split="all", resolution=1, dtu_eval_mode="foreground").sample(0)

    expected_edge = torch.tensor([1.0, 127 / 255.0, 127 / 255.0])
    assert torch.allclose(sample.image[0, 0], expected_edge, atol=1 / 255)
    assert torch.allclose(sample.image[0, 1], torch.tensor([0.0, 1.0, 0.0]), atol=1 / 255)
    assert sample.mask is not None
    assert torch.allclose(sample.mask[0, :, 0], torch.tensor([128 / 255.0, 1.0]), atol=1 / 255)


def test_native_dtu_reader_matches_eval_camera_and_mask(tmp_path):
    root = tmp_path / "scan24"
    (root / "images").mkdir(parents=True)
    (root / "mask").mkdir()
    _write_basic_ply(root / "points3d_dtu.ply")

    image = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    alpha = Image.new("L", (4, 4), 0)
    alpha.putdata([255, 255, 0, 0] * 4)
    image.putalpha(alpha)
    image.save(root / "images" / "0000.png")

    mask = Image.new("L", (4, 4), 255)
    mask.save(root / "mask" / "000.png")

    K = np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    extrinsic = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0]],
        dtype=np.float32,
    )
    world_mat = np.eye(4, dtype=np.float32)
    world_mat[:3, :4] = K @ extrinsic
    np.savez(
        root / "cameras.npz",
        world_mat_0=world_mat,
        scale_mat_0=np.eye(4, dtype=np.float32),
    )

    eval_sample = DTUDataset(root, split="all", resolution=1, dtu_eval_mode="foreground").sample(0)
    scene_info = readDTUSceneInfo(str(root), "images", eval=False)
    native_cam = scene_info.train_cameras[0]

    native_w2c = np.eye(4, dtype=np.float32)
    native_w2c[:3, :3] = native_cam.R.T
    native_w2c[:3, 3] = native_cam.T

    assert np.allclose(
        eval_sample.camera.viewmats[0].numpy(),
        native_w2c,
        atol=1e-5,
    )
    assert native_cam.image.mode == "RGBA"
    assert native_cam.image.getchannel("A").getpixel((2, 0)) == 0
    assert torch.allclose(eval_sample.image[0, 2], torch.ones(3), atol=1 / 255)


def test_uint8_rgb_helper_extracts_rgb_without_compositing_alpha():
    image = Image.new("RGBA", (2, 2))
    image.putdata(
        [
            (20, 40, 80, 0),
            (20, 40, 80, 64),
            (20, 40, 80, 128),
            (20, 40, 80, 255),
        ]
    )

    arr = _pil_rgb_uint8(image, (1, 1), resample=Image.Resampling.LANCZOS)

    assert arr.shape == (1, 1, 3)
    assert arr[0, 0].tolist() == [20, 40, 80]


def test_native_rgba_loader_composites_dtu_alpha_to_white():
    image = Image.new("RGBA", (3, 1))
    image.putdata(
        [
            (20, 40, 80, 0),
            (20, 40, 80, 128),
            (20, 40, 80, 255),
        ]
    )

    rgb, alpha = load_rgba_for_training(
        image,
        (3, 1),
        PILtoTorch,
        composite_white=True,
    )

    assert torch.allclose(rgb[:, 0, 0], torch.ones(3), atol=1 / 255)
    assert torch.allclose(rgb[:, 0, 2], torch.tensor([20, 40, 80]) / 255.0, atol=1 / 255)
    assert torch.allclose(alpha[:, 0, 0], torch.zeros(1), atol=1 / 255)


def test_native_rgba_loader_preserves_non_dtu_alpha_rgb():
    image = Image.new("RGBA", (1, 1), (20, 40, 80, 0))

    rgb, alpha = load_rgba_for_training(
        image,
        (1, 1),
        PILtoTorch,
        composite_white=False,
    )

    assert torch.allclose(rgb[:, 0, 0], torch.tensor([20, 40, 80]) / 255.0, atol=1 / 255)
    assert torch.allclose(alpha[:, 0, 0], torch.zeros(1), atol=1 / 255)


def test_is_dtu_scene_tolerates_missing_source_path():
    assert is_dtu_scene(SimpleNamespace()) is False


def test_is_dtu_scene_detects_cameras_npz(tmp_path):
    (tmp_path / "cameras.npz").write_bytes(b"")

    assert is_dtu_scene(SimpleNamespace(source_path=str(tmp_path))) is True
