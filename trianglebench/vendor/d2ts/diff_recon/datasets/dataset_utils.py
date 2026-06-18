import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.spatial.distance import cdist

from .colmap_loader import CameraInfo
from ..utils.camera import rotmat2qvec, qvec2rotmat
from ..models.point_cloud import PointCloud


def getCameraExtent(cam_infos: list[CameraInfo]) -> float:
    cam_centers = [-cam.R @ cam.T for cam in cam_infos]
    cam_centers = np.stack(cam_centers, axis=0)
    center = cam_centers.mean(axis=0, keepdims=True)
    extent = np.linalg.norm(cam_centers - center, axis=1).max() * 1.1
    return extent


def camInfosToColmap(cam_infos: list[CameraInfo], save_dir: str) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    cam_dict = {}
    with open(f"{save_dir}/images.txt", "w", encoding="utf-8") as f_images:
        for i, cam in enumerate(cam_infos):
            width, height = cam.width, cam.height
            f = 0.5 * height / np.tan(0.5 * cam.FovY)
            if cam.camera_id in cam_dict:
                if cam_dict[cam.camera_id] != (width, height, f):
                    raise ValueError(f"Camera ID {cam.camera_id} has inconsistent parameters: {cam_dict[cam.camera_id]} vs ({width}, {height}, {f})")
            else:
                cam_dict[cam.camera_id] = (width, height, f)

            quat = rotmat2qvec(cam.R.T)

            f_images.write(f"{i} {quat[0]} {quat[1]} {quat[2]} {quat[3]} {cam.T[0]} {cam.T[1]} {cam.T[2]} {cam.camera_id} {cam.image_name}\n\n")

    with open(f"{save_dir}/cameras.txt", "w", encoding="utf-8") as f_cameras:
        for cam_id, (width, height, f) in cam_dict.items():
            f_cameras.write(f"{cam_id} PINHOLE {width} {height} {f} {f} {width / 2} {height / 2}\n")


def pointCloudToColmap(pcd: PointCloud, save_dir: str) -> None:
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/points3D.txt", "w", encoding="utf-8") as f:
        for i, (point, color) in enumerate(zip(pcd.points, pcd.colors)):
            color = (color * 255).astype(np.uint8)
            f.write(f"{i} {point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]} {0}\n")


def jsonCamerasToCameraInfos(json_data: list[dict]) -> list[CameraInfo]:
    cam_infos = []
    for cam in json_data:
        R = np.array(cam["rotation"], dtype=np.float32) @ np.diag([1, -1, -1])
        T = -R.T @ np.array(cam["position"], dtype=np.float32)
        fovx = 2 * np.arctan(cam["width"] / (2 * cam["fx"]))
        fovy = 2 * np.arctan(cam["height"] / (2 * cam["fy"]))
        cam_info = CameraInfo(
            camera_id=cam["id"],
            R=R,
            T=T,
            FovY=fovy,
            FovX=fovx,
            image_path=None,
            image_name=None,
            width=cam["width"],
            height=cam["height"],
        )
        cam_infos.append(cam_info)
    return cam_infos


def cameraInfosToJsonCameras(cam_infos: list[CameraInfo]) -> list[dict]:
    json_data = []
    for cam in cam_infos:
        fx = 0.5 * cam.width / np.tan(0.5 * cam.FovX)
        fy = 0.5 * cam.height / np.tan(0.5 * cam.FovY)
        position = (-cam.R @ cam.T).tolist()
        rotation = (cam.R @ np.diag([1, -1, -1])).tolist()
        json_cam = dict(
            id=cam.camera_id,
            width=cam.width,
            height=cam.height,
            fx=fx,
            fy=fy,
            position=position,
            rotation=rotation,
        )
        json_data.append(json_cam)
    return json_data


def interpolateCameraInfos(
    source_cam_infos: list[CameraInfo],
    target_cam_infos: list[CameraInfo],
    num_interp: int,
    ground_z: float = None,
) -> list[CameraInfo]:
    """
    Find the closest source camera view for each target view and interpolate between these pairs.

    Args:
        source_cam_infos: List of source camera information
        target_cam_infos: List of target camera information
        num_interp: Number of interpolation steps between each source-target pair

    Returns:
        List of interpolated camera information
    """
    # Extract camera positions and rotations
    s_positions = np.array([-cam.R @ cam.T for cam in source_cam_infos])  # Camera centers in world coordinates
    t_positions = np.array([-cam.R @ cam.T for cam in target_cam_infos])

    s_Rs = np.array([cam.R for cam in source_cam_infos])
    t_Rs = np.array([cam.R for cam in target_cam_infos])

    # Convert rotation matrices to scipy Rotation objects for SLERP
    s_rotations = [Rotation.from_matrix(R) for R in s_Rs]  # Camera to world rotation
    t_rotations = [Rotation.from_matrix(R) for R in t_Rs]

    # Find closest source camera for each target camera
    if ground_z is not None:
        s_lookat = s_Rs[:, :, 2]
        t_lookat = t_Rs[:, :, 2]
        s_lookat_pos = s_positions + s_lookat * (ground_z - s_positions[:, 2:3]) / s_lookat[:, 2:3]
        t_lookat_pos = t_positions + t_lookat * (ground_z - t_positions[:, 2:3]) / t_lookat[:, 2:3]
        position_dist = cdist(t_lookat_pos, s_lookat_pos)
    else:
        position_dist = cdist(t_positions, s_positions)  # [num_targets, num_sources]
    rotation_dist = 1 - (t_Rs[:, None, :, 0] * s_Rs[None, :, :, 0]).sum(axis=-1)
    blend_dist = position_dist / 200 + rotation_dist
    closest_source_indices = np.argmin(blend_dist, axis=1)

    interpolated_cam_infos = []

    # Create interpolation weights
    alphas = np.sqrt(np.linspace(0, 1, num_interp + 1)[1:])

    for j, alpha in enumerate(alphas):
        for i, target_cam in enumerate(target_cam_infos):
            closest_source_idx = closest_source_indices[i]

            # Get positions and rotations for interpolation
            source_pos = s_positions[closest_source_idx]
            target_pos = t_positions[i]
            source_rot = s_rotations[closest_source_idx]
            target_rot = t_rotations[i]

            # Interpolate position
            interp_pos = (1 - alpha) * source_pos + alpha * target_pos

            # Interpolate rotation using SLERP
            slerp = Slerp([0, 1], Rotation.concatenate([source_rot, target_rot]))
            interp_rot = slerp(alpha)

            # Convert back to camera coordinate system
            R_cam_to_world = interp_rot.as_matrix()
            T_cam = -R_cam_to_world.T @ interp_pos

            # Create interpolated camera info
            interp_cam_info = CameraInfo(
                camera_id=0,
                R=R_cam_to_world,
                T=T_cam,
                FovY=target_cam.FovY,
                FovX=target_cam.FovX,
                image_path=None,
                image_name=f"interp_{j:02d}_{i:04d}",
                width=target_cam.width,
                height=target_cam.height,
            )
            interpolated_cam_infos.append(interp_cam_info)

    return interpolated_cam_infos
