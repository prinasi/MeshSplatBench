from .trainers.VanillaTS_trainer import VanillaTSTrainer

from .datasets.Colmap_dataset import ColmapDatasetFactory, BaseDatasetFactory
from .datasets.NerfSynthetic_dataset import NerfSyntheticDatasetFactory
from .datasets.MatrixCity_dataset import MatrixCityDatasetFactory

from .models.VanillaTS_model import VanillaTSModel
from .models.animated_triangle import AnimatedTriangle

from .models.raw_gaussian import RawGaussian
from .models.raw_triangle import RawTriangle

from .utils.config import loadConfig, Config
from .utils.pipeline_utils import run_exp_with_args, run_exp
from .utils.logger import stdout_logger

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


VanillaGSTrainer = _optional_import(lambda: __import__(
    f"{__name__}.trainers.VanillaGS_trainer",
    fromlist=["VanillaGSTrainer"],
).VanillaGSTrainer)
VanillaGSModel = _optional_import(lambda: __import__(
    f"{__name__}.models.VanillaGS_model",
    fromlist=["VanillaGSModel"],
).VanillaGSModel)
HybridGTModel = _optional_import(lambda: __import__(
    f"{__name__}.models.HybridGT_model",
    fromlist=["HybridGTModel"],
).HybridGTModel)
