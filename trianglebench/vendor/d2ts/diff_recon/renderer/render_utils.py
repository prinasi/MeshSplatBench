import torch
import numpy as np
import cv2
from tqdm import tqdm
import trimesh
import kaolin as kal

from ..utils.camera import Camera
from ..models.VanillaGS_model import VanillaGSModel


def image_tensor_to_cv(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().numpy().transpose(1, 2, 0)[..., ::-1]
    image = np.ascontiguousarray(image * 255).astype(np.uint8)
    return image


@torch.no_grad()
def render_BEV_image(model: VanillaGSModel, save_path: str = None, img_size=(2160, 1440), center=None, elevation=1200) -> torch.Tensor:
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    FoVx, FoVy = 0.610, 0.414

    center = model.get_xyz.mean(dim=0).detach().cpu().numpy() if center is None else center
    cam_pos = center + np.array([0, 0, elevation])
    T = R.T @ -cam_pos

    cam = Camera(R, T, FoVx, FoVy, img_size[0], img_size[1]).to(model.device)
    image = model.forward(cam, "black", False)["render"]

    if save_path is not None:
        image = image_tensor_to_cv(image)
        cv2.imwrite(save_path, image)

    return image


def pos_target_to_RT(pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = target - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    R = np.array([x, y, z]).T
    T = R.T @ -pos
    return R, T


@torch.no_grad()
def render_tour(model: VanillaGSModel, save_path: str, img_size=(2160, 1440)):
    outer_radius_x = 800
    outer_radius_y = 400
    inner_radius_x = 200
    inner_radius_y = 100
    fps = 30
    duration = 10
    num_cams = fps * duration
    FoVx, FoVy = 0.610, 0.414
    elevation = 400

    center = model.get_xyz.mean(dim=0).detach().cpu().numpy()
    theta = np.linspace(0, 2 * np.pi, num_cams, endpoint=False)
    coord = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).T
    outer_radius = np.array([outer_radius_x, outer_radius_y, 0])
    inner_radius = np.array([inner_radius_x, inner_radius_y, 0])
    cam_pos = coord * outer_radius + center + np.array([0, 0, elevation])
    target_pos = coord * inner_radius + center

    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, img_size)
    for i in tqdm(range(num_cams)):
        R, T = pos_target_to_RT(cam_pos[i], target_pos[i])
        cam = Camera(R, T, FoVx, FoVy, img_size[0], img_size[1]).to(model.device)
        image = model.forward(cam, False, "black")["render"]
        image = image_tensor_to_cv(image)
        video.write(image)
    video.release()


@torch.no_grad()
def render_tour_compare(model1: VanillaGSModel, model2: VanillaGSModel, save_path: str, img_size=(2160, 1440), name1="model1", name2="model2"):
    outer_radius_x = 800
    outer_radius_y = 400
    inner_radius_x = 200
    inner_radius_y = 100
    fps = 30
    duration = 10
    num_cams = fps * duration
    FoVx, FoVy = 0.610, 0.414
    elevation = 400

    center = model1.get_xyz.mean(dim=0).detach().cpu().numpy()
    theta = np.linspace(0, 2 * np.pi, num_cams, endpoint=False)
    coord = np.array([np.cos(theta), np.sin(theta), np.zeros_like(theta)]).T
    outer_radius = np.array([outer_radius_x, outer_radius_y, 0])
    inner_radius = np.array([inner_radius_x, inner_radius_y, 0])
    cam_pos = coord * outer_radius + center + np.array([0, 0, elevation])
    target_pos = coord * inner_radius + center

    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, img_size)
    for i in tqdm(range(num_cams)):
        R, T = pos_target_to_RT(cam_pos[i], target_pos[i])
        cam = Camera(R, T, FoVx, FoVy, img_size[0], img_size[1]).to(model1.device)
        image1 = model1.forward(cam, False, "black")["render"]
        image2 = model2.forward(cam, False, "black")["render"]

        image = torch.cat([image1[..., : img_size[0] // 2], image2[..., img_size[0] // 2 :]], dim=2)
        image = image_tensor_to_cv(image)

        # write name on image
        cv2.putText(image, name1, (img_size[0] // 20, img_size[1] // 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(image, name2, (img_size[0] // 2 + img_size[0] // 20, img_size[1] // 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # draw separation line
        cv2.line(image, (img_size[0] // 2, 0), (img_size[0] // 2, img_size[1]), (255, 255, 255), 2)
        video.write(image)
    video.release()


def _srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_srgb_to_rgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _srgb_to_rgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1] and out.shape[2] == f.shape[2]
    return out


def load_render_data_from_mesh(mesh, device: torch.device) -> dict[str, torch.Tensor]:
    if isinstance(mesh, trimesh.Trimesh):
        vertices = torch.tensor(mesh.vertices.reshape(-1, 3), dtype=torch.float32).to(device)  # (Nv, 3)
        faces = torch.tensor(mesh.faces, dtype=torch.int64).to(device)  # (Nf, 3)
        if hasattr(mesh.visual, "vertex_colors"):
            # If vertex colors are available, use them
            vertex_colors = torch.tensor(mesh.visual.vertex_colors / 255.0, dtype=torch.float32).to(device)  # (Nv, 4) or (Nv, 3)
            face_colors = vertex_colors[faces].unsqueeze(0)  # (1, Nf, 3, 4) or (1, Nf, 3, 3)
        elif hasattr(mesh.visual, "face_colors"):
            # If face colors are available, use them
            face_colors = torch.tensor(mesh.visual.face_colors / 255.0, dtype=torch.float32).to(device)  # (Nf, 4) or (Nf, 3)
            nc = face_colors.shape[1]
            face_colors = face_colors.view(1, -1, 1, nc).repeat(1, 1, 3, 1)  # (1, Nf, 3, 4) or (1, Nf, 3, 3)
        elif hasattr(mesh.visual, "material") and hasattr(mesh.visual.material, "baseColorTexture"):
            # If material is available, use it
            vertex_uvs = torch.tensor(mesh.visual.uv, dtype=torch.float32).to(device)  # (Nv, 2)
            face_uvs = vertex_uvs[faces].unsqueeze(0)  # (1, Nf, 3, 2)
            diffuse_texture = torch.tensor(np.array(mesh.visual.material.baseColorTexture))
            diffuse_texture = diffuse_texture.float().to(device).permute(2, 0, 1).unsqueeze(0) / 255.0  # (1, c, H, W)
        else:
            raise ValueError("Mesh does not have vertex/face colors or textures.")
    elif isinstance(mesh, kal.rep.SurfaceMesh):
        vertices = mesh.vertices.to(device)  # (Nv, 3)
        faces = mesh.faces.to(device)  # (Nf, 3)
        face_uvs = mesh.face_uvs.unsqueeze(0).to(device)  # (1, Nf, 3, 2)
        face_uvs[..., 1] = 1 - face_uvs[..., 1]  # Flip the V coordinate
        diffuse_texture = mesh.materials[0]["map_Kd"] / 255.0
        diffuse_texture = srgb_to_rgb(diffuse_texture)
        diffuse_texture = diffuse_texture.float().to(device).permute(2, 0, 1).unsqueeze(0)  # (1, c, H, W)

    return {
        "vertices": vertices,
        "faces": faces,
        "face_colors": face_colors if "face_colors" in locals() else None,
        "face_uvs": face_uvs if "face_uvs" in locals() else None,
        "diffuse_texture": diffuse_texture[:, :3] if "diffuse_texture" in locals() else None,  # alpha channel is not handled
    }


def load_render_data_from_file(file_path: str, device: torch.device, rotate: bool = False) -> dict[str, torch.Tensor]:
    if file_path.endswith(".glb") or file_path.endswith(".ply"):
        mesh = trimesh.load_mesh(file_path)
    elif file_path.endswith(".obj"):
        mesh = kal.io.obj.import_mesh(file_path, with_materials=True, with_normals=False)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    if rotate:
        R = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        if isinstance(mesh, trimesh.Trimesh):
            mesh.apply_transform(R)
        elif isinstance(mesh, kal.rep.SurfaceMesh):
            rot = torch.tensor(R[:3, :3], dtype=mesh.vertices.dtype, device=mesh.vertices.device)
            trans = torch.tensor(R[:3, 3], dtype=mesh.vertices.dtype, device=mesh.vertices.device)
            mesh.vertices = (mesh.vertices @ rot.T) + trans
        else:
            raise ValueError(f"Unsupported mesh type for rotation: {type(mesh)}")

    return load_render_data_from_mesh(mesh, device)
