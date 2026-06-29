"""Argument helpers vendored from triangle-splatting.

The original project exposes argparse parameter groups. Keeping the same
attribute names lets TriBench call the native training code while still
remaining a single installable package.
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none: bool = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            value_type = type(value)
            default = value if not fill_none else None
            if shorthand:
                if value_type == bool:
                    group.add_argument("--" + key, "-" + key[0:1], default=default, action="store_true")
                else:
                    group.add_argument("--" + key, "-" + key[0:1], default=default, type=value_type)
            elif value_type == bool:
                group.add_argument("--" + key, default=default, action="store_true")
            else:
                group.add_argument("--" + key, default=default, type=value_type)

    def extract(self, args):
        group = GroupParams()
        for key, value in vars(args).items():
            if key in vars(self) or "_" + key in vars(self):
                setattr(group, key, value)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser: ArgumentParser, sentinel: bool = False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.dtu_eval_mode = "full"
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        group = super().extract(args)
        group.source_path = os.path.abspath(group.source_path)
        return group


class PipelineParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 1.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser: ArgumentParser):
        self.iterations = 30_000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.014
        self.lambda_dssim = 0.2
        self.densification_interval = 500
        self.densify_from_iter = 500
        self.densify_until_iter = 25_000
        self.random_background = False
        self.foreground_training = False
        self.mask_threshold = 0.01
        self.lr_mask = 0.01
        self.nb_points = 3
        self.triangle_size = 2.23
        self.set_opacity = 0.28
        self.set_sigma = 1.16
        self.noise_lr = 5e5
        self.mask_dead = 0.08
        self.lambda_normals = 0.0001
        self.lambda_dist = 0.0
        self.lambda_opacity = 0.0055
        self.lambda_size = 0.00000001
        self.opacity_dead = 0.014
        self.importance_threshold = 0.022
        self.iteration_mesh = 5000
        self.cloning_sigma = 1.0
        self.cloning_opacity = 1.0
        self.lr_sigma = 0.0008
        self.lr_triangles_points_init = 0.0018
        self.proba_distr = 2
        self.split_size = 24.0
        self.start_lr_sigma = 0
        self.max_noise_factor = 1.5
        self.max_shapes = 3_000_000
        self.add_shape = 1.3
        self.p = 1.6
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdline_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdline_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print(f"Config file found: {cfgfilepath}")
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")

    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    for key, value in vars(args_cmdline).items():
        if value is not None:
            merged_dict[key] = value
    return Namespace(**merged_dict)
