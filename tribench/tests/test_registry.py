"""Tests for method registry."""

import pytest

from tribench.core.registry import (
    register,
    get_adapter,
    list_methods,
    is_registered,
    _REGISTRY,
)


class TestRegistry:
    """Tests for the adapter registry."""

    def test_list_methods_includes_registered(self):
        # The adapters are auto-registered via __init__.py imports
        methods = list_methods()
        assert isinstance(methods, list)
        # Check that at least some methods are registered
        assert len(methods) >= 4  # 2dts, triangle-splatting, mesh-splatting, diffsoup

    def test_get_2dts(self):
        cls = get_adapter("2dts")
        assert cls.name.fget(None) == "2dts" or hasattr(cls, "name")

    def test_get_2dts_alias(self):
        assert get_adapter("d2ts") is get_adapter("2dts")

    def test_get_mesh_splatting_alias(self):
        assert get_adapter("mesh_splatting") is get_adapter("mesh-splatting")

    def test_get_triangle_splatting(self):
        cls = get_adapter("triangle-splatting")
        assert hasattr(cls, "render")

    def test_get_mesh_splatting(self):
        cls = get_adapter("mesh-splatting")
        assert hasattr(cls, "render")

    def test_get_diffsoup(self):
        cls = get_adapter("diffsoup")
        assert hasattr(cls, "render")

    def test_get_nonexistent_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_adapter("nonexistent_method")

    def test_is_registered(self):
        assert is_registered("2dts")
        assert not is_registered("nonexistent")

    def test_register_new_method(self):
        from tribench.renderers.base import RendererAdapter

        @register("test_method")
        class TestAdapter(RendererAdapter):
            @property
            def name(self):
                return "test_method"

            def load_checkpoint(self, path):
                pass

            def load_scene(self, dataset_path, split="test"):
                pass

            def model_stats(self):
                return {}

            def render(self, cameras, *, mode="eval"):
                pass

        assert is_registered("test_method")
        cls = get_adapter("test_method")
        assert cls is TestAdapter

        # Cleanup
        del _REGISTRY["test_method"]

    def test_register_as_function(self):
        from tribench.renderers.base import RendererAdapter

        class AnotherAdapter(RendererAdapter):
            @property
            def name(self):
                return "func_test"

            def load_checkpoint(self, path):
                pass

            def load_scene(self, dataset_path, split="test"):
                pass

            def model_stats(self):
                return {}

            def render(self, cameras, *, mode="eval"):
                pass

        register("func_test", AnotherAdapter)
        assert is_registered("func_test")

        # Cleanup
        del _REGISTRY["func_test"]
