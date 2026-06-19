"""Tests for CameraBatch."""

import torch
import pytest

from tribench.core.cameras import CameraBatch


class TestCameraBatch:
    """Tests for CameraBatch creation and properties."""

    def test_creation(self, dummy_camera_batch):
        cam = dummy_camera_batch
        assert cam.batch_size == 2
        assert cam.width == 640
        assert cam.height == 480
        assert cam.near == 0.01
        assert cam.far == 100.0

    def test_tensor_shapes(self, dummy_camera_batch):
        cam = dummy_camera_batch
        assert cam.viewmats.shape == (2, 4, 4)
        assert cam.camtoworlds.shape == (2, 4, 4)
        assert cam.Ks.shape == (2, 3, 3)

    def test_device_property(self, dummy_camera_batch):
        assert dummy_camera_batch.device == torch.device("cpu")

    def test_to_device(self, dummy_camera_batch):
        cam = dummy_camera_batch.to("cpu")
        assert cam.device == torch.device("cpu")
        assert cam.batch_size == 2

    def test_metadata_default_none(self, dummy_viewmats, dummy_camtoworlds, dummy_Ks):
        cam = CameraBatch(
            viewmats=dummy_viewmats,
            camtoworlds=dummy_camtoworlds,
            Ks=dummy_Ks,
            width=640,
            height=480,
        )
        assert cam.metadata is None

    def test_metadata_custom(self, dummy_viewmats, dummy_camtoworlds, dummy_Ks):
        cam = CameraBatch(
            viewmats=dummy_viewmats,
            camtoworlds=dummy_camtoworlds,
            Ks=dummy_Ks,
            width=640,
            height=480,
            metadata={"scene": "garden"},
        )
        assert cam.metadata == {"scene": "garden"}
