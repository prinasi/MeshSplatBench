from .triangle_renderer import TriangleRenderer

_OPTIONAL_RASTERIZATION_MODULES = {
    "custom_gaussian_rasterization",
    "hybrid_rasterization",
}


def _optional_import(import_fn):
    try:
        return import_fn()
    except ModuleNotFoundError as exc:
        if exc.name not in _OPTIONAL_RASTERIZATION_MODULES:
            raise
        return None


GaussianRenderer = _optional_import(lambda: __import__(
    f"{__name__}.gaussian_renderer",
    fromlist=["GaussianRenderer"],
).GaussianRenderer)
HybridRenderer = _optional_import(lambda: __import__(
    f"{__name__}.hybrid_renderer",
    fromlist=["HybridRenderer"],
).HybridRenderer)

__all__ = [
    "TriangleRenderer",
    "GaussianRenderer",
    "HybridRenderer",
]
