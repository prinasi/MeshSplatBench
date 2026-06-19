import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from .raw_triangle import RawTriangle
from ..utils.gltf_utils import build_texture_atlas, ensure_rgba, load_glb_chunks, load_image_from_gltf, read_accessor
from ..utils.sh_utils import RGB2SH


class AnimatedTriangle:
    """Load a glTF/GLB scene and sample it into a RawTriangle at time t.

    The sampled RawTriangle stores per-vertex colors converted from glTF materials,
    textures, vertex colors, and emissive terms. Static scenes are supported as a
    degenerate animation case.
    """

    def __init__(self, *, glb_path: str = None, animation_index: int = 0) -> None:
        self.glb_path: str | None = None
        self.active_animation_index = animation_index
        self._object_transform = np.eye(4, dtype=np.float32)

        self._gltf: dict | None = None
        self._bin_chunk: bytes = b""
        self._scene_index: int = 0
        self._scene_roots: list[int] = []
        self._nodes: list[dict] = []
        self._meshes: list[dict] = []
        self._skins: list[dict] = []
        self._animations: list[dict] = []
        self._textures: list[dict] = []
        self._materials: list[dict] = []
        self._texture_atlas: np.ndarray | None = None
        self._rest_triangle: RawTriangle | None = None

        if glb_path is not None:
            self.loadGLB(glb_path)

    def __len__(self):
        return sum(len(mesh["primitives"]) for mesh in self._meshes)

    @property
    def animation_names(self) -> list[str]:
        return [animation["name"] for animation in self._animations]

    def setAnimation(self, animation: int | str | None):
        if animation is None:
            self.active_animation_index = 0
            return self
        if isinstance(animation, int):
            if animation < 0 or animation >= len(self._animations):
                raise IndexError(f"Animation index out of range: {animation}")
            self.active_animation_index = animation
            return self
        for animation_idx, clip in enumerate(self._animations):
            if clip["name"] == animation:
                self.active_animation_index = animation_idx
                return self
        raise ValueError(f"Animation not found: {animation}")

    def getAnimationDuration(self, animation_index: int | None = None) -> float:
        if not self._animations:
            return 0.0
        clip = self._animations[self._resolve_animation_index(animation_index)]
        return float(clip["end_time"] - clip["start_time"])

    def getRestTriangle(self, copy: bool = True, apply_object_transform: bool = True) -> RawTriangle:
        if self._rest_triangle is None:
            raise RuntimeError("Call loadGLB before requesting the rest triangle")

        triangle = deepcopy(self._rest_triangle) if copy else self._rest_triangle
        if apply_object_transform and len(triangle) > 0:
            vertex_shape = triangle.vertex.shape
            triangle.vertex = np.ascontiguousarray(
                self._transform_points(self._object_transform, triangle.vertex.reshape(-1, 3)).reshape(vertex_shape)
            )
        elif len(triangle) > 0:
            triangle.vertex = np.ascontiguousarray(triangle.vertex)
        return triangle

    def center(self, center: np.ndarray) -> "AnimatedTriangle":
        center = self._validate_vec3(center, "center")
        translation = center - self._get_current_rest_center()
        self._object_transform = self._translation_matrix(translation) @ self._object_transform
        return self

    def reset_transform(self) -> "AnimatedTriangle":
        self._object_transform = np.eye(4, dtype=np.float32)
        return self

    def rotate(self, axis: np.ndarray, degree: np.ndarray | None = None) -> "AnimatedTriangle":
        rotation = self._rotation_matrix_from_axis_degree(axis, degree)

        center = self._get_current_rest_center()
        rotation_h = np.eye(4, dtype=np.float32)
        rotation_h[:3, :3] = rotation
        self._object_transform = (
            self._translation_matrix(center)
            @ rotation_h
            @ self._translation_matrix(-center)
            @ self._object_transform
        )
        return self

    def scale(self, scale: np.ndarray) -> "AnimatedTriangle":
        scale = np.asarray(scale, dtype=np.float32)
        if scale.shape == ():
            scale = np.full((3,), float(scale), dtype=np.float32)
        if scale.shape != (3,):
            raise ValueError(f"scale must have shape (3,), got {scale.shape}")

        center = self._get_current_rest_center()
        scale_h = np.eye(4, dtype=np.float32)
        scale_h[:3, :3] = np.diag(scale)
        self._object_transform = (
            self._translation_matrix(center)
            @ scale_h
            @ self._translation_matrix(-center)
            @ self._object_transform
        )
        return self

    def loadGLB(self, path: str) -> "AnimatedTriangle":
        glb_path = Path(path)
        if not glb_path.exists():
            raise FileNotFoundError(f"GLB file not found: {path}")

        self.glb_path = str(glb_path)
        self._object_transform = np.eye(4, dtype=np.float32)
        self._gltf, self._bin_chunk = load_glb_chunks(glb_path)
        self._scene_index = self._gltf.get("scene", 0)
        self._scene_roots = list(self._gltf.get("scenes", [{}])[self._scene_index].get("nodes", []))

        self._nodes = self._parse_nodes()
        self._textures = self._parse_textures(glb_path)
        self._materials = self._parse_materials()
        self._build_material_texture_atlas()
        self._meshes = self._parse_meshes()
        self._skins = self._parse_skins()
        self._animations = self._parse_animations()
        self._rest_triangle = self._sample_impl(0.0, animation_index=None, loop=False, apply_object_transform=False, use_active_animation=False)
        return self

    def sample(self, t: float, animation_index: int | str | None = None, loop: bool = False) -> RawTriangle:
        return self._sample_impl(t, animation_index=animation_index, loop=loop, apply_object_transform=True)

    def _sample_impl(
        self,
        t: float,
        animation_index: int | str | None = None,
        loop: bool = False,
        apply_object_transform: bool = True,
        use_active_animation: bool = True,
    ) -> RawTriangle:
        if self._gltf is None:
            raise RuntimeError("Call loadGLB before sampling")

        clip = self._get_animation(animation_index) if use_active_animation else None
        node_states = self._build_node_states(clip, t, loop)
        local_matrices = [self._compose_node_matrix(node, state) for node, state in zip(self._nodes, node_states)]
        world_matrices = self._compute_world_matrices(local_matrices)

        tri_vertex_list = []
        tri_opacity_list = []
        tri_sh_list = []
        tri_uv_list = []

        use_texture = self._can_sample_texture()

        for node_idx, node in enumerate(self._nodes):
            mesh_index = node["mesh"]
            if mesh_index is None:
                continue

            mesh = self._meshes[mesh_index]
            node_weights = node_states[node_idx]["weights"]

            for primitive in mesh["primitives"]:
                positions = self._apply_morph_targets(primitive, mesh, node_weights)
                if node["skin"] is not None and primitive["joints"] is not None and primitive["weights"] is not None:
                    positions_world = self._apply_skinning(node, primitive, positions, world_matrices)
                else:
                    positions_world = self._transform_points(world_matrices[node_idx], positions)
                if apply_object_transform:
                    positions_world = self._transform_points(self._object_transform, positions_world)

                vertex_rgba = self._sample_vertex_rgba(primitive)
                face_indices = primitive["faces"]
                tri_vertex = positions_world[face_indices]
                tri_rgba = vertex_rgba[face_indices]
                tri_alpha = tri_rgba[..., 3].mean(axis=1, keepdims=True)
                tri_rgb = np.clip(tri_rgba[..., :3], 0.0, 1.0)

                tri_vertex_list.append(tri_vertex.astype(np.float32))
                tri_opacity_list.append(self._alpha_to_logit(tri_alpha))
                tri_sh_list.append(RGB2SH(tri_rgb.astype(np.float32)).astype(np.float32))
                if use_texture:
                    tri_uv = self._get_atlas_uv(primitive)
                    tri_uv_list.append(tri_uv[face_indices].astype(np.float32))

        if not tri_vertex_list:
            return RawTriangle(
                vertex=np.empty((0, 3, 3), dtype=np.float32),
                opacity=np.empty((0, 1), dtype=np.float32),
                shs=np.empty((0, 3, 3), dtype=np.float32),
                uv=np.empty((0, 3, 2), dtype=np.float32),
                texture=self._texture_atlas.copy() if use_texture and self._texture_atlas is not None else None,
            )

        return RawTriangle(
            vertex=np.ascontiguousarray(np.concatenate(tri_vertex_list, axis=0)),
            opacity=np.ascontiguousarray(np.concatenate(tri_opacity_list, axis=0)),
            shs=np.ascontiguousarray(np.concatenate(tri_sh_list, axis=0)),
            uv=np.ascontiguousarray(np.concatenate(tri_uv_list, axis=0)) if use_texture else None,
            texture=np.ascontiguousarray(self._texture_atlas.copy()) if use_texture and self._texture_atlas is not None else None,
        )

    def _get_current_rest_center(self) -> np.ndarray:
        if self._rest_triangle is None or len(self._rest_triangle) == 0:
            return np.zeros((3,), dtype=np.float32)

        vertices = self._rest_triangle.vertex.reshape(-1, 3).astype(np.float32)
        vertices = self._transform_points(self._object_transform, vertices)
        return vertices.mean(axis=0).astype(np.float32)

    def _parse_nodes(self) -> list[dict]:
        nodes = []
        for node_def in self._gltf.get("nodes", []):
            matrix = None
            if "matrix" in node_def:
                matrix = np.asarray(node_def["matrix"], dtype=np.float32).reshape(4, 4).T

            translation = np.asarray(node_def.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
            rotation = self._normalize_quaternion(np.asarray(node_def.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32))
            scale = np.asarray(node_def.get("scale", [1.0, 1.0, 1.0]), dtype=np.float32)
            weights = np.asarray(node_def.get("weights", []), dtype=np.float32) if "weights" in node_def else None

            nodes.append(
                {
                    "name": node_def.get("name"),
                    "children": list(node_def.get("children", [])),
                    "mesh": node_def.get("mesh"),
                    "skin": node_def.get("skin"),
                    "matrix": matrix,
                    "translation": translation,
                    "rotation": rotation,
                    "scale": scale,
                    "weights": weights,
                }
            )
        return nodes

    def _parse_textures(self, glb_path: Path) -> list[dict]:
        images = [load_image_from_gltf(image_def, glb_path, self._bin_chunk, self._gltf) for image_def in self._gltf.get("images", [])]
        samplers = self._gltf.get("samplers", [])

        textures = []
        for texture_def in self._gltf.get("textures", []):
            sampler = samplers[texture_def["sampler"]] if "sampler" in texture_def and texture_def["sampler"] < len(samplers) else {}
            image = images[texture_def["source"]] if "source" in texture_def and texture_def["source"] < len(images) else None
            textures.append(
                {
                    "image": image,
                    "wrap_s": sampler.get("wrapS", 10497),
                    "wrap_t": sampler.get("wrapT", 10497),
                }
            )
        return textures

    def _parse_materials(self) -> list[dict]:
        materials = []
        for material_def in self._gltf.get("materials", []):
            pbr = material_def.get("pbrMetallicRoughness", {})
            emissive_strength = material_def.get("extensions", {}).get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0)
            base_color_texture = pbr.get("baseColorTexture")
            emissive_texture = material_def.get("emissiveTexture")
            base_texcoord_set = base_color_texture.get("texCoord", 0) if base_color_texture is not None else None
            emissive_texcoord_set = emissive_texture.get("texCoord", 0) if emissive_texture is not None else None
            texture_compatible = base_texcoord_set is None or emissive_texcoord_set is None or base_texcoord_set == emissive_texcoord_set
            texture_index = base_color_texture.get("index") if base_color_texture is not None else emissive_texture.get("index") if emissive_texture is not None else None
            wrap_s = self._textures[texture_index]["wrap_s"] if texture_index is not None else 10497
            wrap_t = self._textures[texture_index]["wrap_t"] if texture_index is not None else 10497
            materials.append(
                {
                    "base_color_factor": np.asarray(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]), dtype=np.float32),
                    "base_color_texture": base_color_texture,
                    "emissive_factor": np.asarray(material_def.get("emissiveFactor", [0.0, 0.0, 0.0]), dtype=np.float32),
                    "emissive_texture": emissive_texture,
                    "emissive_strength": float(emissive_strength),
                    "alpha_mode": material_def.get("alphaMode", "OPAQUE"),
                    "alpha_cutoff": float(material_def.get("alphaCutoff", 0.5)),
                    "uv_set": base_texcoord_set if base_texcoord_set is not None else emissive_texcoord_set,
                    "texture_compatible": texture_compatible,
                    "atlas_transform": None,
                    "resolved_texture": None,
                    "wrap_s": wrap_s,
                    "wrap_t": wrap_t,
                }
            )

        if not materials:
            materials.append(
                {
                    "base_color_factor": np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    "base_color_texture": None,
                    "emissive_factor": np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
                    "emissive_texture": None,
                    "emissive_strength": 1.0,
                    "alpha_mode": "OPAQUE",
                    "alpha_cutoff": 0.5,
                    "uv_set": None,
                    "texture_compatible": True,
                    "atlas_transform": None,
                    "resolved_texture": np.ones((1, 1, 4), dtype=np.float32),
                    "wrap_s": 10497,
                    "wrap_t": 10497,
                }
            )
        return materials

    def _build_material_texture_atlas(self):
        textures = []
        texture_material_indices = []
        for material_idx, material in enumerate(self._materials):
            material["resolved_texture"] = self._compose_material_texture(material)
            if material["resolved_texture"] is not None:
                textures.append(material["resolved_texture"])
                texture_material_indices.append(material_idx)

        if not textures:
            self._texture_atlas = None
            return

        atlas, transforms = build_texture_atlas(textures)
        self._texture_atlas = atlas[..., :3] if np.allclose(atlas[..., 3], 1.0) else atlas
        for material_idx, transform in zip(texture_material_indices, transforms):
            self._materials[material_idx]["atlas_transform"] = transform

    def _compose_material_texture(self, material: dict) -> np.ndarray | None:
        if not material["texture_compatible"]:
            return None

        base_texture = np.ones((1, 1, 4), dtype=np.float32)
        if material["base_color_texture"] is not None:
            texture_info = material["base_color_texture"]
            base_texture = ensure_rgba(self._textures[texture_info["index"]]["image"])
        base_texture = np.clip(base_texture * material["base_color_factor"].reshape(1, 1, 4), 0.0, 1.0)

        emissive_rgb = material["emissive_factor"] * material["emissive_strength"]
        if material["emissive_texture"] is not None:
            texture_info = material["emissive_texture"]
            emissive_texture = ensure_rgba(self._textures[texture_info["index"]]["image"])
            if emissive_texture.shape[:2] != base_texture.shape[:2]:
                return None
            base_texture[..., :3] = np.clip(base_texture[..., :3] + emissive_texture[..., :3] * emissive_rgb.reshape(1, 1, 3), 0.0, 1.0)
        else:
            base_texture[..., :3] = np.clip(base_texture[..., :3] + emissive_rgb.reshape(1, 1, 3), 0.0, 1.0)

        if material["alpha_mode"] == "OPAQUE":
            base_texture[..., 3] = 1.0
        elif material["alpha_mode"] == "MASK":
            base_texture[..., 3] = (base_texture[..., 3] >= material["alpha_cutoff"]).astype(np.float32)
        return np.clip(base_texture, 0.0, 1.0)

    def _parse_meshes(self) -> list[dict]:
        meshes = []
        for mesh_def in self._gltf.get("meshes", []):
            mesh_weights = np.asarray(mesh_def.get("weights", []), dtype=np.float32) if "weights" in mesh_def else None
            primitives = []
            for primitive_def in mesh_def.get("primitives", []):
                attributes = primitive_def["attributes"]
                positions = read_accessor(self._gltf, self._bin_chunk, attributes["POSITION"]).astype(np.float32)
                faces = self._indices_to_faces(primitive_def, len(positions))

                texcoords = {}
                for key, accessor_index in attributes.items():
                    if key.startswith("TEXCOORD_"):
                        texcoords[int(key.split("_")[1])] = read_accessor(self._gltf, self._bin_chunk, accessor_index).astype(np.float32)

                vertex_rgba = None
                if "COLOR_0" in attributes:
                    vertex_rgba = read_accessor(self._gltf, self._bin_chunk, attributes["COLOR_0"]).astype(np.float32)
                    if vertex_rgba.shape[1] == 3:
                        vertex_rgba = np.concatenate([vertex_rgba, np.ones((len(vertex_rgba), 1), dtype=np.float32)], axis=1)

                joints = self._read_joint_or_weight_sets(attributes, prefix="JOINTS_")
                weights = self._read_joint_or_weight_sets(attributes, prefix="WEIGHTS_")

                morph_positions = []
                for target in primitive_def.get("targets", []):
                    if "POSITION" in target:
                        morph_positions.append(read_accessor(self._gltf, self._bin_chunk, target["POSITION"]).astype(np.float32))
                    else:
                        morph_positions.append(None)

                primitives.append(
                    {
                        "positions": positions,
                        "faces": faces,
                        "texcoords": texcoords,
                        "vertex_rgba": vertex_rgba,
                        "joints": joints,
                        "weights": weights,
                        "material": primitive_def.get("material", 0),
                        "morph_positions": morph_positions,
                    }
                )

            meshes.append({"weights": mesh_weights, "primitives": primitives})
        return meshes

    def _parse_skins(self) -> list[dict]:
        skins = []
        for skin_def in self._gltf.get("skins", []):
            joint_indices = np.asarray(skin_def.get("joints", []), dtype=np.int32)
            if "inverseBindMatrices" in skin_def:
                inverse_bind = read_accessor(self._gltf, self._bin_chunk, skin_def["inverseBindMatrices"]).astype(np.float32)
            else:
                inverse_bind = np.tile(np.eye(4, dtype=np.float32), (len(joint_indices), 1, 1))
            skins.append({"name": skin_def.get("name"), "joints": joint_indices, "inverse_bind": inverse_bind})
        return skins

    def _parse_animations(self) -> list[dict]:
        animations = []
        for animation_def in self._gltf.get("animations", []):
            tracks = []
            start_time = math.inf
            end_time = -math.inf
            for channel in animation_def.get("channels", []):
                sampler = animation_def["samplers"][channel["sampler"]]
                times = read_accessor(self._gltf, self._bin_chunk, sampler["input"]).reshape(-1).astype(np.float32)
                values = read_accessor(self._gltf, self._bin_chunk, sampler["output"]).astype(np.float32)
                interpolation = sampler.get("interpolation", "LINEAR")
                if interpolation == "CUBICSPLINE":
                    values = values.reshape(len(times), 3, *values.shape[1:])

                tracks.append(
                    {
                        "node": channel["target"]["node"],
                        "path": channel["target"]["path"],
                        "times": times,
                        "values": values,
                        "interpolation": interpolation,
                    }
                )
                start_time = min(start_time, float(times[0]))
                end_time = max(end_time, float(times[-1]))

            if start_time is math.inf:
                start_time = 0.0
                end_time = 0.0

            animations.append(
                {
                    "name": animation_def.get("name", f"animation_{len(animations)}"),
                    "tracks": tracks,
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                }
            )
        return animations

    def _build_node_states(self, clip: dict | None, t: float, loop: bool) -> list[dict]:
        node_states = []
        for node in self._nodes:
            node_states.append(
                {
                    "translation": node["translation"].copy(),
                    "rotation": node["rotation"].copy(),
                    "scale": node["scale"].copy(),
                    "weights": None if node["weights"] is None else node["weights"].copy(),
                }
            )

        if clip is None:
            return node_states

        sample_time = self._resolve_sample_time(clip, t, loop)
        for track in clip["tracks"]:
            value = self._sample_track(track, sample_time)
            if track["path"] == "rotation":
                value = self._normalize_quaternion(value)
            node_states[track["node"]][track["path"]] = value.astype(np.float32)
        return node_states

    def _compose_node_matrix(self, node: dict, state: dict) -> np.ndarray:
        if node["matrix"] is not None:
            return node["matrix"].copy()
        return self._compose_trs(state["translation"], state["rotation"], state["scale"])

    def _compute_world_matrices(self, local_matrices: list[np.ndarray]) -> np.ndarray:
        world_matrices = [np.eye(4, dtype=np.float32) for _ in local_matrices]

        def visit(node_idx: int, parent_matrix: np.ndarray):
            world_matrices[node_idx] = parent_matrix @ local_matrices[node_idx]
            for child_idx in self._nodes[node_idx]["children"]:
                visit(child_idx, world_matrices[node_idx])

        if self._scene_roots:
            for root_idx in self._scene_roots:
                visit(root_idx, np.eye(4, dtype=np.float32))
        else:
            child_nodes = {child for node in self._nodes for child in node["children"]}
            roots = [idx for idx in range(len(self._nodes)) if idx not in child_nodes]
            for root_idx in roots:
                visit(root_idx, np.eye(4, dtype=np.float32))
        return np.stack(world_matrices, axis=0)

    def _apply_morph_targets(self, primitive: dict, mesh: dict, node_weights: np.ndarray | None) -> np.ndarray:
        positions = primitive["positions"].copy()
        if not primitive["morph_positions"]:
            return positions

        if node_weights is not None:
            weights = node_weights
        elif mesh["weights"] is not None:
            weights = mesh["weights"]
        else:
            weights = np.zeros(len(primitive["morph_positions"]), dtype=np.float32)

        weights = np.asarray(weights, dtype=np.float32)
        if len(weights) < len(primitive["morph_positions"]):
            weights = np.pad(weights, (0, len(primitive["morph_positions"]) - len(weights)))

        for target_idx, delta in enumerate(primitive["morph_positions"]):
            if delta is not None and target_idx < len(weights) and weights[target_idx] != 0:
                positions += delta * weights[target_idx]
        return positions

    def _apply_skinning(self, node: dict, primitive: dict, positions: np.ndarray, world_matrices: list[np.ndarray]) -> np.ndarray:
        skin = self._skins[node["skin"]]
        joint_matrices = world_matrices[skin["joints"]] @ skin["inverse_bind"]

        positions_h = np.concatenate([positions.astype(np.float32), np.ones((len(positions), 1), dtype=np.float32)], axis=1)
        joints = primitive["joints"].astype(np.int32)
        weights = primitive["weights"].astype(np.float32)

        total_weight = weights.sum(axis=1, keepdims=True)
        weights = np.divide(weights, np.maximum(total_weight, 1e-8), out=np.zeros_like(weights), where=total_weight > 0)

        skinned = np.zeros((len(positions), 4), dtype=np.float32)
        for influence_idx in range(joints.shape[1]):
            influence_weight = weights[:, influence_idx : influence_idx + 1]
            if not np.any(influence_weight > 0):
                continue
            influence_joint = joints[:, influence_idx]
            influence_mats = joint_matrices[influence_joint]
            skinned += influence_weight * np.einsum("nij,nj->ni", influence_mats, positions_h)

        return skinned[:, :3]

    def _sample_vertex_rgba(self, primitive: dict) -> np.ndarray:
        positions = primitive["positions"]
        rgba = np.ones((len(positions), 4), dtype=np.float32)

        if primitive["vertex_rgba"] is not None:
            rgba *= primitive["vertex_rgba"]

        material = self._materials[primitive["material"]] if primitive["material"] < len(self._materials) else self._materials[0]
        uv = self._get_material_uv(primitive)
        if material["resolved_texture"] is not None and uv is not None:
            rgba *= self._sample_texture({"image": material["resolved_texture"], "wrap_s": material["wrap_s"], "wrap_t": material["wrap_t"]}, uv)
            return np.clip(rgba, 0.0, 1.0)

        rgba *= material["base_color_factor"]
        rgba[:, :3] = np.clip(rgba[:, :3] + material["emissive_factor"] * material["emissive_strength"], 0.0, 1.0)

        if material["alpha_mode"] == "OPAQUE":
            rgba[:, 3] = 1.0
        elif material["alpha_mode"] == "MASK":
            rgba[:, 3] = (rgba[:, 3] >= material["alpha_cutoff"]).astype(np.float32)
        return np.clip(rgba, 0.0, 1.0)

    def _can_sample_texture(self) -> bool:
        if self._texture_atlas is None:
            return False
        for mesh in self._meshes:
            for primitive in mesh["primitives"]:
                if primitive["vertex_rgba"] is not None:
                    return False
                material = self._materials[primitive["material"]] if primitive["material"] < len(self._materials) else self._materials[0]
                if material["resolved_texture"] is None or material["atlas_transform"] is None:
                    return False
                if material["uv_set"] is not None and material["uv_set"] not in primitive["texcoords"]:
                    return False
        return True

    def _get_material_uv(self, primitive: dict) -> np.ndarray | None:
        material = self._materials[primitive["material"]] if primitive["material"] < len(self._materials) else self._materials[0]
        uv_set = material["uv_set"]
        if uv_set is None:
            return np.full((len(primitive["positions"]), 2), 0.5, dtype=np.float32)
        return primitive["texcoords"].get(uv_set)

    def _get_atlas_uv(self, primitive: dict) -> np.ndarray:
        material = self._materials[primitive["material"]] if primitive["material"] < len(self._materials) else self._materials[0]
        uv = self._get_material_uv(primitive)
        if uv is None:
            raise ValueError("Texture sampling requested for primitive without UV coordinates")
        uv = uv.copy().astype(np.float32)
        uv[:, 0] = self._wrap_uv(uv[:, 0], material["wrap_s"])
        uv[:, 1] = self._wrap_uv(uv[:, 1], material["wrap_t"])
        transform = material["atlas_transform"]
        remapped = uv
        remapped[..., 0] = transform[2] + remapped[..., 0] * transform[0]
        remapped[..., 1] = transform[3] + remapped[..., 1] * transform[1]
        return remapped

    def _sample_texture(self, texture: dict, uv: np.ndarray) -> np.ndarray:
        image = texture["image"]
        if image is None:
            return np.ones((len(uv), 4), dtype=np.float32)

        u = self._wrap_uv(uv[:, 0], texture["wrap_s"])
        v = self._wrap_uv(1.0 - uv[:, 1], texture["wrap_t"])

        height, width = image.shape[:2]
        x = np.clip(u * (width - 1), 0.0, width - 1)
        y = np.clip(v * (height - 1), 0.0, height - 1)

        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, width - 1)
        y1 = np.clip(y0 + 1, 0, height - 1)

        wx = (x - x0).astype(np.float32)[:, None]
        wy = (y - y0).astype(np.float32)[:, None]

        top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
        bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
        return (top * (1.0 - wy) + bottom * wy).astype(np.float32)

    def _resolve_animation_index(self, animation_index: int | str | None) -> int:
        if isinstance(animation_index, str):
            for idx, animation in enumerate(self._animations):
                if animation["name"] == animation_index:
                    return idx
            raise ValueError(f"Animation not found: {animation_index}")
        if animation_index is None:
            return self.active_animation_index
        if animation_index < 0 or animation_index >= len(self._animations):
            raise IndexError(f"Animation index out of range: {animation_index}")
        return animation_index

    def _get_animation(self, animation_index: int | str | None) -> dict | None:
        if not self._animations:
            return None
        return self._animations[self._resolve_animation_index(animation_index)]

    def _resolve_sample_time(self, clip: dict, t: float, loop: bool) -> float:
        start_time = clip["start_time"]
        end_time = clip["end_time"]
        if end_time <= start_time:
            return start_time
        if loop:
            return ((t - start_time) % (end_time - start_time)) + start_time
        return min(max(float(t), start_time), end_time)

    def _sample_track(self, track: dict, t: float) -> np.ndarray:
        times = track["times"]
        interpolation = track["interpolation"]
        values = track["values"]

        if len(times) == 1 or t <= times[0]:
            return values[0, 1] if interpolation == "CUBICSPLINE" else values[0]
        if t >= times[-1]:
            return values[-1, 1] if interpolation == "CUBICSPLINE" else values[-1]

        next_idx = int(np.searchsorted(times, t, side="right"))
        prev_idx = next_idx - 1
        t0, t1 = float(times[prev_idx]), float(times[next_idx])
        s = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)

        if interpolation == "STEP":
            return values[prev_idx]
        if interpolation == "LINEAR":
            value0 = values[prev_idx]
            value1 = values[next_idx]
            if track["path"] == "rotation":
                return self._slerp(value0, value1, s)
            return ((1.0 - s) * value0 + s * value1).astype(np.float32)
        if interpolation == "CUBICSPLINE":
            value0 = values[prev_idx, 1]
            value1 = values[next_idx, 1]
            out_tangent = values[prev_idx, 2]
            in_tangent = values[next_idx, 0]
            dt = t1 - t0
            s2 = s * s
            s3 = s2 * s
            result = (
                (2 * s3 - 3 * s2 + 1) * value0
                + (s3 - 2 * s2 + s) * dt * out_tangent
                + (-2 * s3 + 3 * s2) * value1
                + (s3 - s2) * dt * in_tangent
            )
            if track["path"] == "rotation":
                return self._normalize_quaternion(result)
            return result.astype(np.float32)
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")

    def _read_joint_or_weight_sets(self, attributes: dict, prefix: str) -> np.ndarray | None:
        matched_keys = sorted(key for key in attributes.keys() if key.startswith(prefix))
        if not matched_keys:
            return None
        arrays = [read_accessor(self._gltf, self._bin_chunk, attributes[key]) for key in matched_keys]
        return np.concatenate(arrays, axis=1)

    def _indices_to_faces(self, primitive_def: dict, vertex_count: int) -> np.ndarray:
        mode = primitive_def.get("mode", 4)
        if "indices" in primitive_def:
            indices = read_accessor(self._gltf, self._bin_chunk, primitive_def["indices"]).reshape(-1).astype(np.int32)
        else:
            indices = np.arange(vertex_count, dtype=np.int32)

        if mode == 4:
            if len(indices) % 3 != 0:
                raise ValueError("Triangle primitive index count must be divisible by 3")
            return indices.reshape(-1, 3)
        if mode == 5:
            faces = []
            for idx in range(len(indices) - 2):
                tri = [indices[idx], indices[idx + 1], indices[idx + 2]]
                if idx % 2 == 1:
                    tri[1], tri[2] = tri[2], tri[1]
                if tri[0] != tri[1] and tri[1] != tri[2] and tri[0] != tri[2]:
                    faces.append(tri)
            return np.asarray(faces, dtype=np.int32)
        if mode == 6:
            root = int(indices[0])
            faces = [[root, int(indices[idx]), int(indices[idx + 1])] for idx in range(1, len(indices) - 1)]
            return np.asarray(faces, dtype=np.int32)
        raise ValueError(f"Unsupported glTF primitive mode for triangles: {mode}")

    @staticmethod
    def _alpha_to_logit(alpha: np.ndarray) -> np.ndarray:
        alpha = np.clip(alpha, 1e-5, 1.0 - 1e-5)
        return np.log(alpha / (1.0 - alpha)).astype(np.float32)

    @staticmethod
    def _validate_scalar(value: np.ndarray, name: str) -> float:
        value = np.asarray(value, dtype=np.float32)
        if value.shape == ():
            return float(value)
        if value.shape == (1,):
            return float(value[0])
        raise ValueError(f"{name} must have shape () or (1,), got {value.shape}")

    @staticmethod
    def _validate_vec3(value: np.ndarray, name: str) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (3,):
            raise ValueError(f"{name} must have shape (3,), got {value.shape}")
        return value

    @staticmethod
    def _rotation_matrix_from_axis_degree(axis: np.ndarray, degree: np.ndarray | None) -> np.ndarray:
        axis = np.asarray(axis, dtype=np.float32)
        if degree is None:
            if axis.shape != (3, 3):
                raise ValueError(f"axis must have shape (3,) when degree is provided, or (3, 3) when used as a rotation matrix; got {axis.shape}")
            return axis

        axis = AnimatedTriangle._validate_vec3(axis, "axis")
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            raise ValueError("axis must be non-zero")
        axis = axis / norm
        angle = np.deg2rad(AnimatedTriangle._validate_scalar(degree, "degree"))

        x, y, z = axis.astype(np.float32)
        cos_theta = float(np.cos(angle))
        sin_theta = float(np.sin(angle))
        one_minus_cos = 1.0 - cos_theta
        return np.array(
            [
                [cos_theta + x * x * one_minus_cos, x * y * one_minus_cos - z * sin_theta, x * z * one_minus_cos + y * sin_theta],
                [y * x * one_minus_cos + z * sin_theta, cos_theta + y * y * one_minus_cos, y * z * one_minus_cos - x * sin_theta],
                [z * x * one_minus_cos - y * sin_theta, z * y * one_minus_cos + x * sin_theta, cos_theta + z * z * one_minus_cos],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _translation_matrix(offset: np.ndarray) -> np.ndarray:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = AnimatedTriangle._validate_vec3(offset, "offset")
        return matrix

    @staticmethod
    def _compose_trs(translation: np.ndarray, rotation: np.ndarray, scale: np.ndarray) -> np.ndarray:
        rotation_matrix = AnimatedTriangle._quat_to_matrix(rotation)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = rotation_matrix @ np.diag(scale.astype(np.float32))
        matrix[:3, 3] = translation.astype(np.float32)
        return matrix

    @staticmethod
    def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
        x, y, z, w = quaternion.astype(np.float32)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quaternion)
        if norm < 1e-8:
            return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return (quaternion / norm).astype(np.float32)

    @staticmethod
    def _slerp(q0: np.ndarray, q1: np.ndarray, s: float) -> np.ndarray:
        q0 = AnimatedTriangle._normalize_quaternion(q0)
        q1 = AnimatedTriangle._normalize_quaternion(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            return AnimatedTriangle._normalize_quaternion((1.0 - s) * q0 + s * q1)

        theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * s
        sin_theta = math.sin(theta)
        s0 = math.sin(theta_0 - theta) / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * q0 + s1 * q1).astype(np.float32)

    @staticmethod
    def _transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
        points_h = np.concatenate([points.astype(np.float32), np.ones((len(points), 1), dtype=np.float32)], axis=1)
        return (matrix @ points_h.T).T[:, :3].astype(np.float32)

    @staticmethod
    def _wrap_uv(coord: np.ndarray, wrap_mode: int) -> np.ndarray:
        if wrap_mode == 33071:
            return np.clip(coord, 0.0, 1.0)
        if wrap_mode == 33648:
            coord = np.mod(coord, 2.0)
            return np.where(coord <= 1.0, coord, 2.0 - coord)
        return np.mod(coord, 1.0)