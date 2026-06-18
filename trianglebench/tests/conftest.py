"""Shared test fixtures for TriangleBench."""

import torch
import pytest


@pytest.fixture
def dummy_viewmats():
    """Identity view matrices for a batch of 2 cameras."""
    return torch.eye(4).unsqueeze(0).repeat(2, 1, 1)


@pytest.fixture
def dummy_camtoworlds():
    """Identity camera-to-world matrices for a batch of 2 cameras."""
    return torch.eye(4).unsqueeze(0).repeat(2, 1, 1)


@pytest.fixture
def dummy_Ks():
    """Simple intrinsic matrices (fx=fy=500, cx=320, cy=240)."""
    K = torch.zeros(3, 3)
    K[0, 0] = 500.0
    K[1, 1] = 500.0
    K[0, 2] = 320.0
    K[1, 2] = 240.0
    K[2, 2] = 1.0
    return K.unsqueeze(0).repeat(2, 1, 1)


@pytest.fixture
def dummy_camera_batch(dummy_viewmats, dummy_camtoworlds, dummy_Ks):
    """A complete CameraBatch with 2 cameras."""
    from trianglebench.core.cameras import CameraBatch

    return CameraBatch(
        viewmats=dummy_viewmats,
        camtoworlds=dummy_camtoworlds,
        Ks=dummy_Ks,
        width=640,
        height=480,
    )


@pytest.fixture
def dummy_triangle_vertices():
    """Random triangle vertices [10, 3, 3]."""
    torch.manual_seed(42)
    return torch.randn(10, 3, 3)


@pytest.fixture
def dummy_opacity():
    """Random opacity values [10]."""
    torch.manual_seed(42)
    return torch.rand(10)


@pytest.fixture
def dummy_mesh_data():
    """Simple mesh: 4 vertices, 2 triangles (a quad split into 2 triangles)."""
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=torch.long)
    return vertices, faces


@pytest.fixture
def dummy_images():
    """Pair of identical dummy images [H, W, 3]."""
    torch.manual_seed(42)
    img = torch.rand(48, 64, 3)
    return img, img.clone()


@pytest.fixture
def tmp_run_dir(tmp_path):
    """Temporary directory for run outputs."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    return run_dir
