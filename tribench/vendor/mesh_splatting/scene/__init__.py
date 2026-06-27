"""Scene loader vendored from MeshSplatting with package-local imports."""

from __future__ import annotations

import json
import os
import random

from tribench.vendor.mesh_splatting.scene.dataset_readers import sceneLoadTypeCallbacks
from tribench.vendor.mesh_splatting.scene.triangle_model import TriangleModel
from tribench.vendor.mesh_splatting.utils.camera_utils import (
    cameraList_from_camInfos,
    camera_to_JSON,
)
from tribench.vendor.mesh_splatting.utils.system_utils import searchForMaxIteration


class Scene:
    triangles: TriangleModel

    def __init__(
        self,
        args,
        triangles: TriangleModel,
        init_opacity,
        set_sigma,
        load_iteration=None,
        shuffle=True,
        resolution_scales=None,
        segment=False,
        ratio_threshold=0.75,
    ):
        if resolution_scales is None:
            resolution_scales = [1.0]

        self.model_path = args.model_path
        self.loaded_iter = None
        self.triangles = triangles

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print(f"Loading trained model at iteration {self.loaded_iter}")

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "cameras.npz")):
            scene_info = sceneLoadTypeCallbacks["DTU"](
                args.source_path, args.images, args.eval
            )
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            raise FileNotFoundError(
                f"Could not recognize scene type at {args.source_path}. "
                "Expected DTU cameras.npz, COLMAP sparse/, or transforms_train.json."
            )

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for idx, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(idx, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.seed(7)
            random.shuffle(scene_info.train_cameras)
            random.shuffle(scene_info.test_cameras)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.triangles.load_parameters(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                ),
                segment=segment,
                ratio_threshold=ratio_threshold,
            )
        else:
            self.triangles.create_from_pcd(scene_info.point_cloud, init_opacity, set_sigma)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, f"point_cloud/iteration_{iteration}"
        )
        print("Save model to: ", point_cloud_path)
        self.triangles.save_parameters(point_cloud_path)
        return os.path.join(point_cloud_path, "point_cloud_state_dict.pt")

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]


__all__ = ["Scene", "TriangleModel"]
