import torch
import kaolin as kal
import trimesh

from ..utils.camera import Camera
from .render_utils import load_render_data_from_file, load_render_data_from_mesh


class KaolinRenderer:
    def __init__(
        self,
        cam: Camera,
        bg_color: list = [0, 0, 0],
    ):
        self.cam = cam
        self.bg_color = torch.tensor(bg_color).float().to(cam.device)

    def render(
        self,
        vertices: torch.Tensor = None,
        faces: torch.Tensor = None,
        face_colors: torch.Tensor = None,
        face_uvs: torch.Tensor = None,
        diffuse_texture: torch.Tensor = None,
        mesh: trimesh.Trimesh = None,
        mesh_path: str = None,
        mesh_data: dict[str, torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # Prepare camera parameters
        cam = self.cam
        device = cam.device
        image_width = cam.image_width
        image_height = cam.image_height
        znear = cam.znear
        camera_projection = torch.tensor([1.0 / cam.tan_fovx, 1.0 / cam.tan_fovy, -1]).float().to(device).view(3, 1)
        camera_transform = cam.world_view_transform[:, :3].unsqueeze(0) @ torch.diag(torch.tensor([1, -1, -1])).float().to(device)

        # If mesh or mesh_path is provided, load the mesh data
        if mesh is not None:
            mesh_data = load_render_data_from_mesh(mesh, device)
        elif mesh_path is not None:
            mesh_data = load_render_data_from_file(mesh_path, device)
        if mesh_data is not None:
            vertices = mesh_data["vertices"]
            faces = mesh_data["faces"]
            face_colors = mesh_data.get("face_colors", None)
            face_uvs = mesh_data.get("face_uvs", None)
            diffuse_texture = mesh_data.get("diffuse_texture", None)
        
        # Ensure all required parameters are provided
        assert vertices is not None and faces is not None, "Vertices and faces must be provided."
        assert face_colors is not None or (face_uvs is not None and diffuse_texture is not None), \
            "Either face_colors or (face_uvs and diffuse_texture) must be provided."

        # Kaolin rendering
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            vertices, faces, camera_proj=camera_projection, camera_transform=camera_transform
        )
        valid_faces = (-face_vertices_camera[:, :, :, -1] > znear).all(dim=-1)
        face_mask = torch.ones((1, faces.shape[0], 3, 1)).float().to(device)

        if face_colors is not None:
            face_attributes = [face_colors, face_mask]
            (image, mask), face_idx = kal.render.mesh.rasterize(
                image_height,
                image_width,
                face_vertices_camera[:, :, :, -1],
                face_vertices_image,
                face_attributes,
                valid_faces=valid_faces,
                backend="nvdiffrast_fwd",
            )
        elif face_uvs is not None and diffuse_texture is not None:
            face_attributes = [face_uvs, face_mask]
            (uv_map, mask), face_idx = kal.render.mesh.rasterize(
                image_height,
                image_width,
                face_vertices_camera[:, :, :, -1],
                face_vertices_image,
                face_attributes,
                valid_faces=valid_faces,
                backend="nvdiffrast_fwd",
            )
            image = kal.render.mesh.texture_mapping(uv_map, diffuse_texture, mode="bilinear")  # (1, H, W, 3)

        # apply background color
        bg_color = self.bg_color if image.shape[-1] == 3 else torch.cat([self.bg_color, torch.tensor([1.0], device=device)])
        image = image * mask + bg_color * (1 - mask)
        image = torch.clamp(image, 0.0, 1.0)

        image = image.permute(0, 3, 1, 2).squeeze(0)
        mask = mask.permute(0, 3, 1, 2).squeeze(0)

        return {
            "render": image,
            "mask": mask,
        }
