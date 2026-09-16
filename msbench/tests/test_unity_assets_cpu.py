"""GPU-free regression checks for Unity export semantics.

This module intentionally uses only unittest and CPU PyTorch so it can run on
machines without CUDA, Unity, pytest, or the compiled rasterizer extensions.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from tools.validate_unity_triasset_cpu import validate_triasset
from tools.create_off import _load_mesh_splatting
from tools.patch_legacy_unity_vulkan import patch_unity_project
from tools.run_unity_triasset_eval import unity_graphics_arguments, validate_captured_images

from msbench.unity_assets import (
    _d2ts_gamma_vertex_rescale,
    _deployment_background_color,
    _general_purpose_contract,
    _mesh_splatting_opacity_floor,
    export_triasset,
    mesh_splatting_triangle_opacity,
)


class MeshOpacityTests(unittest.TestCase):
    def test_legacy_floor_recovery_matches_terminal_training_schedule(self) -> None:
        checkpoint = Path("run/point_cloud/iteration_30000/point_cloud_state_dict.pt")
        floor, source = _mesh_splatting_opacity_floor({}, checkpoint, override=None)
        self.assertEqual(source, "legacy_default_schedule")
        self.assertAlmostEqual(floor, 0.9999)

    def test_reduction_is_min_after_activation_not_mean(self) -> None:
        realized = torch.tensor([0.2, 0.8, 0.9, 0.6], dtype=torch.float32)
        logits = torch.logit(realized)
        faces = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)

        actual = mesh_splatting_triangle_opacity(
            logits, faces, opacity_floor=0.0, chunk_faces=1
        )

        torch.testing.assert_close(actual, torch.tensor([0.2, 0.6]))
        self.assertFalse(torch.allclose(actual, realized[faces].mean(dim=1)))

    def test_floor_is_applied_before_minimum(self) -> None:
        logits = torch.tensor([-3.0, 0.0, 3.0], dtype=torch.float32)
        floor = 0.75
        actual = mesh_splatting_triangle_opacity(
            logits, torch.tensor([[0, 1, 2]]), opacity_floor=floor
        )
        expected = floor + (1.0 - floor) * torch.sigmoid(logits[0])
        torch.testing.assert_close(actual, expected.reshape(1))

    def test_export_writes_auditable_per_triangle_buffer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "point_cloud_state_dict.pt"
            state = {
                "triangles_points": torch.arange(12, dtype=torch.float32).reshape(4, 3),
                "_triangle_indices": torch.tensor([[0, 1, 2], [0, 2, 3]]),
                "vertex_weight": torch.logit(torch.tensor([[0.2], [0.8], [0.9], [0.6]])),
                "opacity_floor": 0.0,
                "sigma": -9.0,
                "active_sh_degree": 0,
                "features_dc": torch.zeros(4, 1, 3),
                "features_rest": torch.zeros(4, 15, 3),
            }
            torch.save(state, checkpoint)

            package = export_triasset("mesh-splatting", checkpoint, root / "scene")
            manifest = json.loads(package.manifest_path.read_text())
            self.assertEqual(manifest["export_contract_revision"], 2)
            spec = manifest["buffers"]["triangle_opacity"]
            values = torch.from_file(
                str(package.path / spec["file"]), dtype=torch.float32, size=2
            ).clone()

            torch.testing.assert_close(values, torch.tensor([0.2, 0.2]))
            self.assertEqual(
                manifest["rendering"]["triangle_opacity_reduction"],
                "min_after_activation",
            )
            self.assertFalse(manifest["rendering"]["terminal_solid_eligible"])
            validation = validate_triasset(package.path, max_faces=1)
            self.assertEqual(validation["status"], "ok")
            self.assertEqual(validation["mesh_opacity"]["faces_checked"], 1)

    def test_export_topology_soup_materializes_per_corner_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "point_cloud_state_dict.pt"
            state = {
                "triangles_points": torch.arange(12, dtype=torch.float32).reshape(4, 3),
                "_triangle_indices": torch.tensor([[0, 1, 2], [0, 2, 3]]),
                "vertex_weight": torch.zeros(4, 1),
                "opacity_floor": 0.0,
                "sigma": -9.0,
                "active_sh_degree": 0,
                "features_dc": torch.zeros(4, 1, 3),
                "features_rest": torch.zeros(4, 15, 3),
            }
            torch.save(state, checkpoint)

            package = export_triasset(
                "mesh-splatting", checkpoint, root / "scene", export_topology="soup"
            )
            manifest = json.loads(package.manifest_path.read_text())

            self.assertEqual(manifest["rendering"]["primitive_topology"], "triangle-soup")
            self.assertEqual(manifest["rendering"]["export_topology"], "materialized-soup")
            self.assertEqual(manifest["rendering"]["source_vertex_count"], 4)
            self.assertEqual(manifest["rendering"]["exported_vertex_count"], 6)
            self.assertEqual(manifest["buffers"]["positions"]["shape"], [6, 3])
            self.assertEqual(manifest["buffers"]["vertex_weight_logits"]["shape"], [6, 1])
            self.assertEqual(manifest["buffers"]["sh_rest"]["shape"], [6, 15, 3])

    def test_generic_preview_uses_the_same_minimum_reduction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "point_cloud_state_dict.pt"
            state = {
                "triangles_points": torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                ),
                "_triangle_indices": torch.tensor([[0, 1, 2]]),
                "vertex_weight": torch.logit(torch.tensor([[0.2], [0.8], [0.9]])),
                "opacity_floor": 0.0,
                "features_dc": torch.zeros(3, 1, 3),
                "features_rest": torch.zeros(3, 15, 3),
                "active_sh_degree": 0,
            }
            torch.save(state, checkpoint)
            mesh = _load_mesh_splatting(
                checkpoint, color_mode="opacity", camera_center=torch.zeros(3)
            )
            torch.testing.assert_close(mesh.opacity, torch.tensor([0.2]))

    def test_failed_forced_export_preserves_previous_complete_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "point_cloud_state_dict.pt"
            valid_state = {
                "triangles_points": torch.zeros(3, 3),
                "_triangle_indices": torch.tensor([[0, 1, 2]]),
                "vertex_weight": torch.zeros(3, 1),
                "opacity_floor": 0.0,
                "sigma": -9.0,
                "features_dc": torch.zeros(3, 1, 3),
                "features_rest": torch.zeros(3, 15, 3),
            }
            torch.save(valid_state, checkpoint)
            original = export_triasset("mesh-splatting", checkpoint, root / "scene")
            original_manifest = original.manifest_path.read_bytes()
            sentinel = original.path / "keep-me.txt"
            sentinel.write_text("previous complete package")

            invalid_state = dict(valid_state)
            invalid_state.pop("sigma")
            torch.save(invalid_state, checkpoint)
            with self.assertRaises(KeyError):
                export_triasset(
                    "mesh-splatting", checkpoint, root / "scene", overwrite=True
                )

            self.assertEqual(original.manifest_path.read_bytes(), original_manifest)
            self.assertEqual(sentinel.read_text(), "previous complete package")
            self.assertEqual(list(root.glob(".scene.triasset.tmp-*")), [])


class D2TSRescaleTests(unittest.TestCase):
    def test_gamma_rescale_matches_native_formula(self) -> None:
        gamma = 4.0
        beta = 1.0 / gamma
        expected = 1.0 / math.sqrt((2.0**beta) * beta * math.gamma(beta))
        self.assertAlmostEqual(_d2ts_gamma_vertex_rescale(gamma), expected, places=12)

    def test_legacy_inference_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.ckpt"
            state = {
                "_vertex": torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
                "_opacity": torch.zeros(1, 1),
                "_f_dc": torch.zeros(1, 1, 3),
                "_f_rest": torch.zeros(1, 15, 3),
                "active_sh_degree": 0,
            }
            torch.save((state, {}, None, 4.0), checkpoint)
            package = export_triasset(
                "2dts", checkpoint, root / "scene", d2ts_gamma_rescale=False
            )
            rendering = json.loads(package.manifest_path.read_text())["rendering"]
            self.assertFalse(rendering["gamma_rescale"])
            self.assertEqual(rendering["gamma_rescale_source"], "export_override")
            self.assertEqual(rendering["gamma_vertex_rescale"], 1.0)

    def test_legacy_2dts_checkpoint_defaults_to_full_sh_degree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.ckpt"
            # Native 2DTS checkpoints are a (state, optimizer, bbox, gamma)
            # tuple and never serialise active_sh_degree; the exporter must
            # derive it from the stored coefficient count like the native
            # D2TSAdapter instead of defaulting to 0.
            state = {
                "_vertex": torch.arange(9, dtype=torch.float32).reshape(1, 3, 3),
                "_opacity": torch.zeros(1, 1),
                "_f_dc": torch.zeros(1, 1, 3),
                "_f_rest": torch.zeros(1, 15, 3),
            }
            torch.save((state, {}, None, 1.0), checkpoint)
            package = export_triasset("2dts", checkpoint, root / "scene")
            rendering = json.loads(package.manifest_path.read_text())["rendering"]
            self.assertEqual(rendering["active_sh_degree"], 3)


class DeploymentContractTests(unittest.TestCase):
    def test_dependency_light_export_entrypoint_runs_without_typer(self) -> None:
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            checkpoint = work / "point_cloud_state_dict.pt"
            torch.save({
                "triangles_points": torch.zeros(1, 3, 3),
                "opacity": torch.zeros(1, 1),
                "sigma": torch.zeros(1, 1),
                "features_dc": torch.zeros(1, 1, 3),
                "features_rest": torch.zeros(1, 0, 3),
            }, checkpoint)
            subprocess.run([
                sys.executable,
                str(root / "tools/export_unity_triasset.py"),
                "--method", "triangle-splatting",
                "--checkpoint", str(checkpoint),
                "--output", str(work / "scene"),
                "--background-color", "black",
            ], check=True, capture_output=True, text=True, cwd=work)
            manifest = json.loads(
                (work / "scene.triasset/manifest.json").read_text()
            )
            self.assertEqual(manifest["export_contract_revision"], 2)

    def test_background_defaults_and_override_are_deterministic(self) -> None:
        self.assertEqual(_deployment_background_color("2dts", None), "white")
        self.assertEqual(_deployment_background_color("mesh-splatting", None), "black")
        self.assertEqual(_deployment_background_color("triangle-splatting", "WHITE"), "white")
        with self.assertRaises(ValueError):
            _deployment_background_color("2dts", "random")

    def test_diffsoup_refuses_white_mesh_as_general_purpose_result(self) -> None:
        contract = _general_purpose_contract("diffsoup")
        self.assertFalse(contract["supported"])
        self.assertIn("texture", contract["reason"])
        self.assertTrue(_general_purpose_contract("mesh-splatting")["supported"])


class UnityTemplateTests(unittest.TestCase):
    def test_shader_contract_uses_flat_face_opacity_and_projected_incenter(self) -> None:
        root = Path(__file__).resolve().parents[2]
        shader = (root / "tools/unity_triasset_renderer/TriAssetSplat.shader").read_text()
        renderer = (
            root / "tools/unity_triasset_renderer/TriAssetSplatRenderer.cs"
        ).read_text()

        self.assertIn("nointerpolation float flatAlpha", shader)
        self.assertIn("_Opacity[face]", shader)
        self.assertNotIn("Sigmoid(_Opacity[entity])", shader)
        self.assertIn("_ScreenParams.xy", shader)
        self.assertIn('return "triangle_opacity"', renderer)
        self.assertIn('"MeshSplatBench/MeshSplatTerminalSolid"', renderer)
        self.assertIn("asset.Rendering.active_sh_degree", renderer)
        self.assertIn("_ShRestCoefficients", renderer)
        self.assertIn("e * _ShRestCoefficients * 3", shader)
        self.assertNotIn("o.alpha=_RenderMode == 1 ? _OpacityFloor", shader)

    def test_bootstrap_copies_the_versioned_templates_verbatim(self) -> None:
        root = Path(__file__).resolve().parents[2]
        script = root / "tools/create_unity_triasset_package.py"
        if not script.is_file():
            self.skipTest("tools/create_unity_triasset_package.py was removed")
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            (project / "Assets").mkdir()
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--unity-project",
                    str(project),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            installed = project / "Assets/MeshSplatBench"
            loader = (installed / "Scripts/TriAssetLoader.cs").read_text()
            self.assertIn("ApplyEvaluationBackground", loader)
            self.assertIn("manifest.export_contract_revision < 2", loader)
            self.assertEqual(
                (installed / "Scripts/TriAssetSplatRenderer.cs").read_text(),
                (root / "tools/unity_triasset_renderer/TriAssetSplatRenderer.cs").read_text(),
            )
            self.assertEqual(
                (installed / "Shaders/TriAssetSplat.shader").read_text(),
                (root / "tools/unity_triasset_renderer/TriAssetSplat.shader").read_text(),
            )
            self.assertEqual(
                (installed / "Shaders/MeshSplatTerminalSolid.shader").read_text(),
                (root / "tools/unity_triasset_renderer/MeshSplatTerminalSolid.shader").read_text(),
            )
            self.assertEqual(
                (project / "Assets/Resources/MeshSplatBench/TriAssetCull.compute").read_text(),
                (root / "tools/unity_triasset_renderer/TriAssetCull.compute").read_text(),
            )
            self.assertEqual(
                (installed / "Scripts/GeneralPurposeMeshTriAssetRenderer.cs").read_text(),
                (root / "tools/unity_triasset_renderer/GeneralPurposeMeshTriAssetRenderer.cs").read_text(),
            )
            self.assertEqual(
                (installed / "Shaders/GeneralPurposeVertexColor.shader").read_text(),
                (root / "tools/unity_triasset_renderer/GeneralPurposeVertexColor.shader").read_text(),
            )
            general_shader = (
                installed / "Shaders/GeneralPurposeVertexColor.shader"
            ).read_text()
            self.assertNotIn("noperspective float4 color", general_shader)
            general_renderer = (
                installed / "Scripts/GeneralPurposeMeshTriAssetRenderer.cs"
            ).read_text()
            self.assertIn("mesh.colors = colors", general_renderer)
            self.assertNotIn("mesh.colors32", general_renderer)


class LegacyUnityVulkanPatchTests(unittest.TestCase):
    def test_patch_accepts_current_msbench_camera_helper_names(self) -> None:
        from tools.patch_legacy_unity_vulkan import (
            _patch_method_specific_camera_state,
            _patch_triangle_splatting_camera_sort,
        )

        method_path = Path(
            "unity/Assets/MeshSplatBench/Scripts/MethodSpecificSplatRenderer.cs"
        )
        triangle_path = Path(
            "unity/Assets/MeshSplatBench/Scripts/TriangleSplattingTriAssetRenderer.cs"
        )
        method_source = method_path.read_text()
        prepare_start = method_source.index("        public override void PrepareCamera")
        prepare_end = method_source.index("        void OnRenderObject")
        duplicate_prepare = method_source[prepare_start:prepare_end]
        method_source = method_source.replace(
            "        void OnRenderObject()",
            duplicate_prepare + "        void OnRenderObject()",
            1,
        )

        patched_method = _patch_method_specific_camera_state(method_source)
        patched_triangle = _patch_triangle_splatting_camera_sort(triangle_path.read_text())

        self.assertEqual(
            patched_method.count("public override void PrepareCamera(Camera camera)"),
            1,
        )
        self.assertEqual(patched_method.count("void OnDestroy()"), 1)
        self.assertIn("InstallMsBenchCameraDraw", patched_method)
        self.assertNotIn("InstallMeshSplatBenchCameraDraw", patched_method)
        self.assertEqual(
            patched_triangle.count("public override void PrepareCamera(Camera camera)"),
            1,
        )

    def test_patch_is_idempotent_and_fixes_premultiplied_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            scripts = project / "Assets/MeshSplatBench/Scripts"
            shaders = project / "Assets/MeshSplatBench/Shaders"
            scripts.mkdir(parents=True)
            shaders.mkdir(parents=True)
            shader = shaders / "MethodSpecificSplat.shader"
            shader.write_text(
                "CGPROGRAM\n#pragma only_renderers metal\n#pragma vertex Vert\n"
                "ByteAddressBuffer _Positions;\nENDCG\n"
            )
            capture = scripts / "ColmapBatchCapture.cs"
            capture.write_text(
                "using System; using UnityEngine;\nnamespace T { class ColmapBatchCapture {\n"
                "void F() { var readback = new Texture2D(1,1,TextureFormat.RGB24,false); "
                "ApplyColmapPose(CaptureCamera, view, intr); CaptureCamera.Render(); "
                "readback.Apply(false, false); }\n"
                "Camera CaptureCamera;\n    }\n}\n"
            )
            renderer_base = scripts / "TriAssetRenderer.cs"
            renderer_base.write_text(
                "using UnityEngine;\nclass TriAssetRenderer\n{\n"
                "        public virtual void SetOutputRawCodeValues(bool enabled) { }\n}\n"
            )
            renderer = scripts / "MethodSpecificSplatRenderer.cs"
            renderer.write_text(
                "using System.Collections;\nusing System.IO;\nusing UnityEngine;\nclass R : TriAssetRenderer\n{\n"
                "        Camera TargetCamera;\n        Shader SplatShader;\n        ComputeBuffer positions, indices, opacity, sigma, dc, rest;\n        Material material;\n"
                "        int primitiveCount, mode, activeShDegree, meshAblation;\n        float gamma = 1f, opacityFloor;\n        bool ready, rawCode, deindexedSoup;\n        string status = \"Waiting to load method-specific renderer\";\n"
                "        IEnumerator Start()\n        {\n"
                "            string method = \"mesh-splatting\";\n            string b = \"buffers\";\n"
                "            int[] sourceIndices = ReadInt(Path.Combine(b, \"indices.bin\"));\n"
                "            if (sourceIndices.Length != primitiveCount * 3) { Fail(\"indices.bin does not match primitive_count.\"); yield break; }\n"
                "            if (deindexedSoup && mode == 1)\n            {\n"
                "                Debug.Log(\"[MeshSplatBench] MeshSplatting shader-level soup topology: retaining indexed buffers; procedural corners fetch via _Indices.\");\n            }\n"
                "            positions = Upload(ReadFloat(Path.Combine(b, \"positions.bin\")));\n"
                "            indices = Upload(sourceIndices);\n"
                "            opacity = Upload(ReadFloat(Path.Combine(b, mode == 1 ? \"vertex_weight_logits.bin\" : \"opacity_logits.bin\")));\n"
                "            dc = Upload(ReadFloat(Path.Combine(b, \"sh_dc.bin\")));\n"
                "            rest = Upload(ReadFloat(Path.Combine(b, \"sh_rest.bin\")));\n"
                "            if (mode == 1) sigma = Upload(ReadFloat(Path.Combine(b, \"sigma_logits.bin\")));\n"
                "            string ablation = \"full\";\n"
                "            string shaderName = mode == 1 && ablation == \"full\"\n"
                "                ? \"MeshSplatBench/MeshSplatTerminalSolid\"\n"
                "                : mode == 1 && ablation == \"alpha-test-depth\"\n"
                "                    ? \"MeshSplatBench/MeshSplatAlphaTestDepth\"\n"
                "                    : mode == 1 && ablation == \"opaque-depth\"\n"
                "                        ? \"MeshSplatBench/MeshSplatOpaqueDepth\"\n"
                "                        : \"MeshSplatBench/MethodSpecificSplat\";\n"
                "            SplatShader = SplatShader != null ? SplatShader : Shader.Find(shaderName);\n"
                "            material = new Material(SplatShader) { hideFlags = HideFlags.HideAndDontSave };\n"
                "            material.SetBuffer(\"_Positions\", positions); material.SetBuffer(\"_Indices\", indices);\n"
                "            material.SetBuffer(\"_Opacity\", opacity); material.SetBuffer(\"_ShDc\", dc); material.SetBuffer(\"_ShRest\", rest);\n"
                "            if (sigma != null) material.SetBuffer(\"_Sigma\", sigma);\n"
                "            material.SetInt(\"_RawCode\", 1);\n"
                "            ready = true; status = $\"Method-specific {method} renderer ready ({primitiveCount:N0} primitives; {(deindexedSoup ? \"shader-level triangle soup\" : \"indexed mesh\")})\";\n"
                "            yield break;\n        }\n"
                "        public override void PrepareCamera(Camera camera)\n        {\n"
                "            if (camera == null || material == null) return;\n"
                "            if (msBenchDrawCommands == null)\n            {\n"
                "                TargetCamera = camera;\n"
                "                if (!InstallMeshSplatBenchCameraDraw()) { enabled = false; return; }\n"
                "            }\n"
                "            material.SetVector(\"_CameraWorldPos\", camera.transform.position);\n"
                "            material.SetInt(\"_RawCode\", rawCode ? 1 : 0);\n"
                "            material.SetFloat(\"_OpacityFloor\", opacityFloor);\n"
                "            if (mode == 1)\n            {\n"
                "                material.SetInt(\"_MeshAblation\", meshAblation);\n"
                "                material.SetInt(\"_UseSh\", (meshAblation == 1 || meshAblation == 4) ? 0 : 1);\n"
                "            }\n        }\n"
                "        void OnRenderObject()\n        {\n"
                "            if (!ready || Camera.current != TargetCamera || material == null) return;\n"
                "            if (!material.SetPass(0)) { Debug.LogError(\"[MeshSplatBench] Shader pass is unsupported on \" + SystemInfo.graphicsDeviceType + \": \" + material.shader.name); enabled = false; return; }\n"
                "            Graphics.DrawProceduralNow(MeshTopology.Triangles, primitiveCount * 3, 1);\n        }\n"
                "        public override void SetOutputRawCodeValues(bool enabled) { rawCode = enabled; }\n"
                "        bool InstallMeshSplatBenchCameraDraw() { return true; }\n"
                "        void RemoveMeshSplatBenchCameraDraw() {}\n"
                "        static int[] ReadInt(string p) { return null; }\n"
                "        static float[] ReadFloat(string p) { return null; }\n"
                "        static ComputeBuffer Upload(System.Array values) { return null; }\n"
                "        void Fail(string message) {}\n"
                "        void OnDestroy() { RemoveMeshSplatBenchCameraDraw(); ready=false; positions?.Release(); indices?.Release(); opacity?.Release(); sigma?.Release(); dc?.Release(); rest?.Release(); if(material!=null) Destroy(material); }\n}\n"
            )

            first = patch_unity_project(project)
            second = patch_unity_project(project)

            self.assertTrue(first)
            self.assertEqual(second, [])
            self.assertIn("#pragma target 5.0", shader.read_text())
            self.assertIn("#pragma only_renderers metal vulkan", shader.read_text())
            capture_text = capture.read_text()
            self.assertIn("TextureFormat.RGBA32", capture_text)
            self.assertIn("PreparePremultipliedCaptureTarget", capture_text)
            self.assertIn("CompositePremultipliedBackground", capture_text)
            self.assertIn("Capture alpha coverage", capture_text)
            self.assertIn("Renderer.PrepareCamera(CaptureCamera)", capture_text)
            self.assertTrue(Path(str(capture) + ".msbench-vulkan.bak").is_file())
            renderer_text = renderer.read_text()
            self.assertIn("if (!material.SetPass(0))", renderer_text)
            self.assertIn("CommandBuffer msBenchDrawCommands", renderer_text)
            self.assertIn("AddCommandBuffer(CameraEvent.AfterForwardAlpha", renderer_text)
            self.assertIn('SetBuffer("_Sigma", sigma != null ? sigma : opacity)', renderer_text)
            self.assertIn("public override void PrepareCamera(Camera camera)", renderer_text)
            self.assertNotIn("if (!InstallMeshSplatBenchCameraDraw()) yield break", renderer_text)
            self.assertIn("if (msBenchDrawCommands == null)", renderer_text)
            self.assertIn("if (Application.isBatchMode) return", renderer_text)
            self.assertIn("public virtual void PrepareCamera(Camera camera)", renderer_base.read_text())
            self.assertIn("ComputeBuffer triangleOrderBuffer", renderer_text)
            self.assertIn("usesSortedTriangleOrder", renderer_text)
            self.assertIn("InitializeTriangleOrder(sourcePositions, sourceIndices)", renderer_text)
            self.assertIn("RefreshTriangleOrder(TargetCamera, force: true)", renderer_text)
            self.assertIn("RefreshTriangleOrder(camera)", renderer_text)
            self.assertIn("RefreshTriangleOrder(TargetCamera)", renderer_text)
            self.assertIn("triangleOrderBuffer.SetData(sortedTriangleIndices)", renderer_text)
            self.assertIn("sourceTriangleIndices", renderer_text)
            self.assertIn("sortedTriangleIndices", renderer_text)

            triangle = scripts / "TriangleSplattingTriAssetRenderer.cs"
            triangle.write_text(
                "using System;\nusing System.Collections;\nusing System.IO;\nusing System.Text.RegularExpressions;\nusing UnityEngine;\nusing UnityEngine.Rendering;\n"
                "class TriangleSplattingTriAssetRenderer : TriAssetRenderer\n{\n"
                "        Camera TargetCamera;\n        Shader SplatShader;\n        int PrimitiveCount = 4;\n        bool SortFrontToBack = true;\n        bool OutputRawCodeValues;\n"
                "        ComputeBuffer positionsBuffer;\n        ComputeBuffer indicesBuffer;\n        ComputeBuffer opacityBuffer;\n        ComputeBuffer sigmaBuffer;\n        ComputeBuffer shDcBuffer;\n        ComputeBuffer shRestBuffer;\n        ComputeBuffer orderBuffer;\n"
                "        Material material;\n        bool ready;\n        string status;\n        float fpsTimer;\n        int fpsFrames;\n        float displayedFps;\n"
                "        IEnumerator Start()\n        {\n"
                "            float[] positions = new float[12];\n"
                "            int[] indices = new int[12];\n"
                "            status = SortFrontToBack ? \"Sorting 4,517,295 primitives front-to-back\" : \"Creating primitive order\";\n"
                "            yield return null;\n"
                "            int[] order = BuildOrder(positions, indices);\n"
                "            orderBuffer = UploadRaw(order);\n"
                "            positions = null;\n"
                "            indices = null;\n"
                "            order = null;\n"
                "            GC.Collect();\n"
                "            yield break;\n        }\n"
                "        int[] BuildOrder(float[] positions, int[] indices)\n        {\n"
                "            int count = Mathf.Min(PrimitiveCount, indices.Length / 3);\n"
                "            PrimitiveCount = count;\n"
                "            int[] order = new int[count];\n"
                "            if (!SortFrontToBack)\n            {\n"
                "                for (int i = 0; i < count; ++i) order[i] = i;\n"
                "                return order;\n            }\n\n"
                "            float[] depths = new float[count];\n"
                "            Vector3 cam = TargetCamera.transform.position;\n"
                "            Vector3 forward = TargetCamera.transform.forward;\n"
                "            for (int t = 0; t < count; ++t)\n            {\n"
                "                int ib = t * 3;\n"
                "                int a = indices[ib] * 3;\n"
                "                int b = indices[ib + 1] * 3;\n"
                "                int c = indices[ib + 2] * 3;\n"
                "                float cx = (positions[a] + positions[b] + positions[c]) / 3f;\n"
                "                float cy = (positions[a + 1] + positions[b + 1] + positions[c + 1]) / 3f;\n"
                "                float cz = (positions[a + 2] + positions[b + 2] + positions[c + 2]) / 3f;\n"
                "                depths[t] = (cx - cam.x) * forward.x + (cy - cam.y) * forward.y + (cz - cam.z) * forward.z;\n"
                "                order[t] = t;\n            }\n"
                "            Array.Sort(depths, order);\n"
                "            return order;\n        }\n"
                "        public override void PrepareCamera(Camera camera)\n        {\n"
                "            if (camera == null || material == null) return;\n"
                "            if (msBenchDrawCommands == null)\n            {\n"
                "                TargetCamera = camera;\n"
                "                if (!InstallMeshSplatBenchCameraDraw()) enabled = false;\n"
                "            }\n"
                "        }\n"
                "        void OnRenderObject()\n        {\n"
                "            if (msBenchDrawCommands != null) return;\n"
                "            if (Application.isBatchMode) return;\n"
                "            if (!ready || Camera.current != TargetCamera || material == null) return;\n"
                "            if (!material.SetPass(0)) { Debug.LogError(\"[MeshSplatBench] Shader pass is unsupported on \" + SystemInfo.graphicsDeviceType + \": \" + material.shader.name); enabled = false; return; }\n"
                "            Graphics.DrawProceduralNow(MeshTopology.Triangles, PrimitiveCount * 3, 1);\n        }\n"
                "        public override void SetOutputRawCodeValues(bool enabled) { OutputRawCodeValues = enabled; }\n"
                "        bool InstallMeshSplatBenchCameraDraw() { return true; }\n"
                "        void RemoveMeshSplatBenchCameraDraw() {}\n"
                "        static ComputeBuffer UploadRaw(Array data) { return null; }\n"
                "        void Fail(string message) {}\n"
                "        void OnDestroy() { ready = false; }\n}\n"
            )

            first = patch_unity_project(project)
            self.assertTrue(first)
            triangle_text = triangle.read_text()
            self.assertIn("Vector3[] triangleCentroids", triangle_text)
            self.assertIn("InitializeTriangleOrder(positions, indices)", triangle_text)
            self.assertIn("RefreshTriangleOrder(TargetCamera, force: true)", triangle_text)
            self.assertIn("void RefreshTriangleOrder(Camera camera, bool force = false)", triangle_text)
            self.assertIn("orderBuffer.SetData(triangleOrder)", triangle_text)
            self.assertIn("if (TargetCamera != camera)", triangle_text)
            self.assertIn("RefreshTriangleOrder(camera)", triangle_text)
            self.assertIn("RefreshTriangleOrder(TargetCamera)", triangle_text)

    def test_capture_validation_rejects_solid_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "test/frame.png"
            path.parent.mkdir(parents=True)
            Image.new("RGB", (16, 16), 0).save(path)
            with self.assertRaisesRegex(RuntimeError, "solid-color"):
                validate_captured_images(root)

            image = Image.new("RGB", (16, 16), 0)
            image.putpixel((0, 0), (255, 255, 255))
            image.save(path)
            validate_captured_images(root)

    def test_linux_uses_vulkan(self) -> None:
        with patch.object(sys, "platform", "linux"):
            self.assertEqual(unity_graphics_arguments(), ["-force-vulkan"])


if __name__ == "__main__":
    unittest.main()
