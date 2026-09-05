"""Argument helpers vendored from MeshSplatting."""

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
                    group.add_argument(
                        "--" + key,
                        "-" + key[0:1],
                        default=default,
                        action="store_true",
                    )
                else:
                    group.add_argument(
                        "--" + key,
                        "-" + key[0:1],
                        default=default,
                        type=value_type,
                    )
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
    def __init__(self, parser, sentinel: bool = False):
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
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 1.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.lambda_dssim = 0.2
        self.densification_interval = 500
        self.densify_from_iter = 500
        self.densify_until_iter = 10_000
        self.run_restricted_delaunay = -1
        self.random_background = False
        self.foreground_training = False
        self.feature_lr = 0.0016
        self.max_points = 4_000_000
        self.max_primitives = -1
        self.set_weight = 0.28
        self.weight_lr = 0.03
        self.lambda_weight = 1.9e-06
        self.iteration_mesh = 5_000
        self.lambda_normals = 0.00005
        self.lambda_normals_super = 0.01
        self.add_percentage = 1.23
        self.set_sigma = 1.0
        self.intervall_add_triangles = 500
        self.prune_triangles_threshold = 0.235
        self.lr_triangles_points_init = 0.0015
        self.start_opacity_floor = 5_000
        self.start_pruning = 4_000
        self.sigma_until = 30_000
        self.final_opacity_iter = 24_000
        self.sigma_start = 0
        self.splitt_large_triangles = 100
        self.start_upsampling = 20_000
        self.upscaling_factor = 2
        self.size_probs_zero = 7.5e-05
        self.size_probs_zero_image_space = 0.0
        self.prune_size = 1400
        self.lambda_vertex = 0.00025
        self.max_diff_threshold = 0.5
        self.start_vertex_opt = 12_000
        self.lamba_depth = 0.05
        self.depth_lambda_init = 0.01
        self.depth_lambda_final = 0.001
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


def update_indoor(params):
    params.add_percentage = 1.27
    params.densify_from_iter = 1000
    params.densify_until_iter = 10000
    params.feature_lr = 0.004
    params.size_probs_zero = 0.0
    params.splitt_large_triangles = 500
    params.start_pruning = 3000
    params.weight_lr = 0.05
    params.lambda_weight = 0.0
    params.lambda_normals = 0.00001
    params.lambda_normals_super = 0.01
    params.prune_size = 1300
    params.lambda_vertex = 0.00025
    params.depth_lambda_init = 0.0
    params.depth_lambda_final = 0.0
    params.iteration_mesh = 12000
    return params
