import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from argparse import Namespace

from .colmap_loader import read_points3D_binary, CameraInfo, readColmapCameras
from .Base_dataset import BaseDatasetFactory
from .dataset_utils import getCameraExtent
from ..models.point_cloud import PointCloud
from ..models.raw_gaussian import RawGaussian
from ..utils.config import Config
from ..utils.logger import Logger
from ..utils.file_handler import LocalHandler, BaseFileHandler
from ..utils.camera import Camera, getWorld2ViewMatrix
from ..utils.sh_utils import SH2RGB


def solve_target_res(target_res: int | list[int] | None, orig_w: int, orig_h: int) -> tuple[int, int]:
    w, h = orig_w, orig_h

    if target_res is None:
        if w >= h and w > 1600:
            w, h = 1600, 1600 * orig_h // orig_w
        elif w < h and h > 1600:
            w, h = 1600 * orig_w // orig_h, 1600
    elif isinstance(target_res, int):
        if target_res <= 0:
            target_res = 1
        w, h = orig_w // target_res, orig_h // target_res
    elif isinstance(target_res, list):
        w, h = target_res
    else:
        raise ValueError("target_res must be either an int of scale ratio or a list of [width, height]")

    return w, h


class ColmapDataset(Dataset):
    def __init__(
        self,
        file_handler: BaseFileHandler,
        cam_infos: list[CameraInfo],
        target_res: int | list[int] = None,
        background: str = None,
        znear: float = 1.0,
        prefetch: bool = False,
        neighbor_cams_args: Namespace = None,
    ):
        super().__init__()
        self.file_handler = file_handler
        self.cam_infos = cam_infos
        self.target_res = target_res
        self.znear = znear
        self.background = background

        if prefetch:
            self._prefetched_imgs = [self._get_image(cam_info.image_path) for cam_info in self.cam_infos]
        if neighbor_cams_args is not None:
            self._neighbor_cams = self._get_neighbor_graph(neighbor_cams_args)

    def _get_neighbor_graph(self, args) -> list[list[int]]:
        neighbor_cams = []
        centers = np.array([-cam_info.R @ cam_info.T for cam_info in self.cam_infos])
        dirs = np.array([cam_info.R[:, 2] for cam_info in self.cam_infos])
        for i in range(len(self.cam_infos)):
            dists = np.linalg.norm(centers[i] - centers, axis=1)
            angles = np.arccos((dirs[i] @ dirs.T).clip(-1, 1)) / np.pi * 180
            sorted_ids = np.lexsort((angles, dists))

            mask = (angles[sorted_ids] < args.max_angle) & (dists[sorted_ids] > args.min_dist) & (dists[sorted_ids] < args.max_dist)
            mask[i] = False  # exclude itself
            sorted_ids = sorted_ids[mask][: args.max_n]
            neighbor_cams.append(sorted_ids.tolist())
        return neighbor_cams

    def _get_bg_color(self) -> np.ndarray:
        if self.background is None:
            bg_color = None
        elif self.background == "white":
            bg_color = np.array([1, 1, 1])
        elif self.background == "black":
            bg_color = np.array([0, 0, 0])
        elif self.background == "random":
            bg_color = np.random.rand(3)
        else:
            raise ValueError("dataset background must be either 'white', 'black', 'random' or None")
        return bg_color

    def _get_image(self, image_path: str) -> np.ndarray:
        """
        Load, resize and normalize the image
        :param image_path: Path to the image
        :param img_size: Target size (W, H)
        :return: Normalized image as a numpy array of shape (3, H, W) or (4, H, W)
        """
        if image_path is None:
            return None

        # some dataset's images are stored with a different structure
        if not self.file_handler.hasFile(image_path):
            path_list = image_path.split("/")
            img_dir = "_".join(path_list[-1].split("_")[:2])
            path_list = path_list[:-1] + [img_dir, path_list[-1]]
            image_path = "/".join(path_list)

        image = Image.open(self.file_handler.getFilePath(image_path))
        img_size = solve_target_res(self.target_res, image.width, image.height)
        image = image.resize(img_size, Image.Resampling.BILINEAR)
        image_array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        image.close()
        return image_array

    def __len__(self):
        return len(self.cam_infos)

    def _get_item(self, idx: int) -> Camera:
        cam_info: CameraInfo = self.cam_infos[idx]
        if hasattr(self, "_prefetched_imgs"):
            gt_image_array = self._prefetched_imgs[idx]
        else:
            gt_image_array = self._get_image(cam_info.image_path)
        bg_color = self._get_bg_color()

        if gt_image_array is not None and gt_image_array.shape[0] == 4:
            gt_alpha_mask = gt_image_array[3]
            gt_image_array = gt_image_array[:3]
            if bg_color is not None:
                gt_image_array = gt_image_array * gt_alpha_mask + bg_color.reshape(3, 1, 1) * (1 - gt_alpha_mask)
        else:
            gt_alpha_mask = None

        camera = Camera(
            R=cam_info.R,
            T=cam_info.T,
            FoVx=cam_info.FovX,
            FoVy=cam_info.FovY,
            image_width=cam_info.width if gt_image_array is None else None,
            image_height=cam_info.height if gt_image_array is None else None,
            gt_image=gt_image_array,
            gt_alpha_mask=gt_alpha_mask,
            image_name=cam_info.image_name,
            camera_id=cam_info.camera_id,
            uid=idx,
            znear=self.znear,
            bg_color=bg_color,
        )
        return camera

    def __getitem__(self, idx) -> Camera:
        camera = self._get_item(idx)
        if hasattr(self, "_neighbor_cams") and len(self._neighbor_cams[idx]) > 0:
            neighbor_cam_id = np.random.choice(self._neighbor_cams[idx])
            camera.neighbor_cam = self._get_item(neighbor_cam_id)
            camera.neighbor_cam.bg_color = camera.bg_color  # ensure same bg color for neighbor cam when using random bg
        return camera


class ColmapDatasetFactory(BaseDatasetFactory):
    def __init__(self, config: Config = None, logger: Logger = None):
        super().__init__(config, logger)
        train_target_res = config.train_target_res
        test_target_res = config.test_target_res
        hold_test_set = config.hold_test_set
        background = config.background
        test_background = config.test_background if config.test_background is not None else background
        prefetch = config.prefetch if config.prefetch is not None else False
        neighbor_cams_args = config.neighbor_cams_args if config.neighbor_cams_args is not None else None

        self._file_handler = self._get_file_handler()

        train_cam_infos, test_cam_infos = self.getCameraInfos()
        if not hold_test_set:
            train_cam_infos += test_cam_infos
            self._logger.warning(f"hold_test_set not set, will merge test set into train set")
        self._logger.info(f"Train set size: {len(train_cam_infos)}, Test set size: {len(test_cam_infos)}")

        self.cameras_extent = getCameraExtent(train_cam_infos)
        self.znear = self.cameras_extent / 1000
        self._logger.info(f"Camera extent: {self.cameras_extent:.2f}, znear set to: {self.znear:.5f}")
        if prefetch:
            self._logger.info("Prefetching data into memory...")

        if neighbor_cams_args is not None:
            self._logger.info("Loading neighbor cameras")
        self._train_dataset = ColmapDataset(self._file_handler, train_cam_infos, train_target_res, background, self.znear, prefetch, neighbor_cams_args)
        self._test_dataset = ColmapDataset(self._file_handler, test_cam_infos, test_target_res, test_background, self.znear, prefetch)

        if len(self._train_dataset) > 0:
            train_cam = self._train_dataset[0]
            self._logger.info(f"Dataset contains alpha mask: {train_cam.alpha_mask is not None}.")
            self._logger.info(f"Train resolution: {train_cam.image_width}x{train_cam.image_height}")
        if len(self._test_dataset) > 0:
            test_cam = self._test_dataset[0]
            self._logger.info(f"Test resolution: {test_cam.image_width}x{test_cam.image_height}")

    def _get_file_handler(self) -> BaseFileHandler:
        if self._config.local_dir is None:
            raise ValueError("local_dir must be set in the config")
        dataset_path = os.path.join(self._config.local_dir, self._config.scene_id) if self._config.scene_id else self._config.local_dir
        return LocalHandler(dataset_path)

    def _getCameraInfos(self) -> tuple[list[CameraInfo], list[CameraInfo]]:
        fs = self._file_handler
        images_bin_path = "sparse/0/images.bin"
        images_txt_path = "sparse/0/images.txt"
        cameras_bin_path = "sparse/0/cameras.bin"
        cameras_txt_path = "sparse/0/cameras.txt"
        images_folder = "images"

        if fs.hasFile(images_bin_path):
            images_path = fs.getFilePath(images_bin_path)
            self._logger.info(f"Fetching extrinsics data from {images_bin_path}.")
        elif fs.hasFile(images_txt_path):
            images_path = fs.getFilePath(images_txt_path)
            self._logger.info(f"Fetching extrinsics data from {images_txt_path}.")
        else:
            raise FileNotFoundError(f"Cannot find {images_bin_path} or {images_txt_path}")

        if fs.hasFile(cameras_bin_path):
            cameras_path = fs.getFilePath(cameras_bin_path)
            self._logger.info(f"Fetching intrinsics data from {cameras_bin_path}.")
        elif fs.hasFile(cameras_txt_path):
            cameras_path = fs.getFilePath(cameras_txt_path)
            self._logger.info(f"Fetching intrinsics data from {cameras_txt_path}.")
        else:
            raise FileNotFoundError(f"Cannot find {cameras_bin_path} or {cameras_txt_path}")

        cam_infos = readColmapCameras(images_path, cameras_path, images_folder)
        cam_infos = sorted(cam_infos, key=lambda x: x.image_name)

        hold_interval = self._config.hold_interval if self._config.hold_interval is not None else 8
        train_cam_infos = [cam for i, cam in enumerate(cam_infos) if i % hold_interval != 0]
        test_cam_infos = [cam for i, cam in enumerate(cam_infos) if i % hold_interval == 0]
        return train_cam_infos, test_cam_infos

    def _getPointCloud(self) -> PointCloud:
        fs = self._file_handler
        pcd_path = self._config.pcd_path

        if pcd_path is None:
            return PointCloud()

        pcd_path = fs.getFilePath(pcd_path)
        self._logger.info(f"Fetching point cloud data from {pcd_path}.")

        if pcd_path.endswith(".bin"):
            xyz, rgb, _ = read_points3D_binary(pcd_path)
            pcd = PointCloud(xyz, rgb)
        elif pcd_path.endswith(".ply"):
            try:
                raw_gs = RawGaussian(ply_path=pcd_path)
                pcd = PointCloud()
                pcd.points = raw_gs.xyz
                pcd.colors = SH2RGB(raw_gs.shs[:, :3])
                pcd.normals = raw_gs.normals
            except Exception as e:
                pcd = PointCloud().fetchPly(pcd_path)
        else:
            raise ValueError(f"Unsupported point cloud file format: {pcd_path.split('.')[-1]}")

        return pcd

    def getCameraInfos(self) -> tuple[list[CameraInfo], list[CameraInfo]]:
        if not hasattr(self, "_cam_infos") or self._cam_infos is None:
            self._cam_infos = self._getCameraInfos()
        return self._cam_infos

    def getPointCloud(self) -> PointCloud:
        if not hasattr(self, "_point_cloud") or self._point_cloud is None:
            self._point_cloud = self._getPointCloud()
        return self._point_cloud

    def getSceneInfo(self) -> dict:
        return None
