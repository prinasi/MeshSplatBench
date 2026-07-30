#!/usr/bin/env python3
"""Patch the legacy TriBenchUnity Metal renderer for safe Vulkan capture.

The older standalone Unity project uses premultiplied front-to-back blending:
``Blend OneMinusDstAlpha One``.  That blend must start from transparent black;
clearing with ``Color.black`` or ``Color.white`` initializes destination alpha
to one and suppresses every splat.  This tool makes the legacy project
cross-platform, clears capture targets with alpha zero, composites the declared
black/white background after readback, and makes failed shader passes visible.

The legacy procedural renderers submitted geometry from ``OnRenderObject``.
That callback is not a reliable submission point for a disabled camera rendered
explicitly with ``Camera.Render`` in Linux Editor batch mode.  The patch moves
those draws into camera command buffers, which execute for both normal frames
and explicit capture renders.

The operation is idempotent and preserves one ``.tribench-vulkan.bak`` copy of
every modified source file.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


BACKUP_SUFFIX = ".tribench-vulkan.bak"
CAPTURE_PATCH_MARKER = "TriBench Vulkan premultiplied-capture patch"


def _write_if_changed(path: Path, original: str, updated: str) -> bool:
    if updated == original:
        return False
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(updated, encoding="utf-8")
    return True


def _patch_shader(source: str) -> str:
    updated = re.sub(
        r"(?m)^(\s*#pragma\s+only_renderers\s+)metal\s*$",
        r"\1metal vulkan",
        source,
    )
    if "ByteAddressBuffer" in updated and "#pragma target" not in updated:
        pragma = "#pragma only_renderers metal vulkan"
        if pragma in updated:
            updated = updated.replace(pragma, "#pragma target 5.0\n  " + pragma, 1)
        else:
            updated = updated.replace("#pragma vertex", "#pragma target 5.0\n  #pragma vertex", 1)
    return updated


def _patch_renderer(source: str) -> str:
    failure = (
        'if (!material.SetPass(0)) { Debug.LogError("[TriBench] Shader pass is '
        'unsupported on " + SystemInfo.graphicsDeviceType + ": " + '
        'material.shader.name); enabled = false; return; }'
    )
    updated = source.replace(
        "material.SetPass(0); Graphics.DrawProceduralNow",
        failure + " Graphics.DrawProceduralNow",
    )
    updated = updated.replace(
        "material.SetPass(0);\n            Graphics.DrawProceduralNow",
        failure + "\n            Graphics.DrawProceduralNow",
    )
    return updated


def _patch_renderer_command_buffer(source: str, primitive_count: str, name: str) -> str:
    """Submit a legacy procedural renderer through the target camera itself."""
    marker = "TriBench explicit camera command-buffer patch"
    if marker in source:
        # AfterEverything and BeforeImageEffects can both run after Unity has
        # ended the offscreen camera's Vulkan render pass.  Submitting with the
        # transparent queue keeps the colour attachment bound while retaining
        # the same camera matrices and depth buffer.
        updated = source.replace(
            "CameraEvent.AfterEverything",
            "CameraEvent.AfterForwardAlpha",
        )
        updated = updated.replace(
            "CameraEvent.BeforeImageEffects",
            "CameraEvent.AfterForwardAlpha",
        )
        # Material.SetPass applies graphics state immediately and requires an
        # active render pass.  PrepareCamera installs this command buffer before
        # camera rendering begins, so probing SetPass here can crash Vulkan with
        # a missing framebuffer attachment.  DrawProcedural selects the pass
        # later, when Unity executes the command buffer for the camera.
        updated = updated.replace(
            '''            if (!material.SetPass(0))
            {
                Fail("Shader pass is unsupported on " + SystemInfo.graphicsDeviceType + ": " + material.shader.name);
                return false;
            }
''',
            '''            if (material.shader == null || !material.shader.isSupported)
            {
                Fail("Shader is unsupported on " + SystemInfo.graphicsDeviceType + ": "
                    + (material.shader == null ? "<null>" : material.shader.name));
                return false;
            }
''',
            1,
        )
        updated = updated.replace(
            "            if (!InstallTriBenchCameraDraw()) yield break;\n",
            "",
            1,
        )
        callback_guard = "            if (triBenchDrawCommands != null) return;\n"
        batch_guard = callback_guard + "            if (Application.isBatchMode) return;\n"
        if batch_guard not in updated:
            updated = updated.replace(callback_guard, batch_guard, 1)
        return updated

    updated = source
    if "using UnityEngine.Rendering;" not in updated:
        updated = updated.replace(
            "using UnityEngine;",
            "using UnityEngine;\nusing UnityEngine.Rendering;",
            1,
        )

    field = "        Material material;"
    if field not in updated:
        raise ValueError(f"could not locate Material field in {name}")
    updated = updated.replace(
        field,
        field
        + "\n        CommandBuffer triBenchDrawCommands;"
        + "\n        Camera triBenchCommandCamera;",
        1,
    )

    helper = f'''        // TriBench explicit camera command-buffer patch. OnRenderObject is
        // not a reliable callback for Camera.Render() on Linux Editor batch mode.
        bool InstallTriBenchCameraDraw()
        {{
            if (TargetCamera == null) TargetCamera = Camera.main;
            if (TargetCamera == null || material == null)
            {{
                Fail("Cannot install procedural draw without a target camera and material.");
                return false;
            }}
            if (material.shader == null || !material.shader.isSupported)
            {{
                Fail("Shader is unsupported on " + SystemInfo.graphicsDeviceType + ": "
                    + (material.shader == null ? "<null>" : material.shader.name));
                return false;
            }}
            triBenchDrawCommands = new CommandBuffer {{ name = "TriBench/{name} procedural draw" }};
            triBenchDrawCommands.DrawProcedural(
                Matrix4x4.identity, material, 0, MeshTopology.Triangles,
                {primitive_count} * 3, 1);
            triBenchCommandCamera = TargetCamera;
            triBenchCommandCamera.AddCommandBuffer(CameraEvent.AfterForwardAlpha, triBenchDrawCommands);
            Debug.Log("[TriBench] Installed explicit camera draw for {name} on "
                + SystemInfo.graphicsDeviceType + ".");
            return true;
        }}

        void RemoveTriBenchCameraDraw()
        {{
            if (triBenchCommandCamera != null && triBenchDrawCommands != null)
                triBenchCommandCamera.RemoveCommandBuffer(CameraEvent.AfterForwardAlpha, triBenchDrawCommands);
            if (triBenchDrawCommands != null) triBenchDrawCommands.Release();
            triBenchDrawCommands = null;
            triBenchCommandCamera = null;
        }}

'''
    callback = "        void OnRenderObject()"
    if callback not in updated:
        raise ValueError(f"could not locate OnRenderObject in {name}")
    updated = updated.replace(callback, helper + callback, 1)

    # Retain OnRenderObject as a fallback for non-patched/custom cameras, but
    # never double-submit when the explicit target-camera command is installed.
    callback_open = "        void OnRenderObject()\n        {"
    updated = updated.replace(
        callback_open,
        callback_open
        + "\n            if (triBenchDrawCommands != null) return;"
        + "\n            if (Application.isBatchMode) return;",
        1,
    )

    # Older revisions installed the command during renderer Start(). In a
    # headless Editor frame this happens before ColmapBatchCapture has assigned
    # its offscreen RenderTexture, so Vulkan tries to draw into a nonexistent
    # screen framebuffer. Installation is now deferred to PrepareCamera().
    updated = updated.replace(
        "            if (!InstallTriBenchCameraDraw()) yield break;\n",
        "",
        1,
    )

    destroy = "void OnDestroy() {"
    if destroy in updated:
        updated = updated.replace(
            destroy,
            destroy + " RemoveTriBenchCameraDraw();",
            1,
        )
    else:
        destroy = "        void OnDestroy()\n        {"
        if destroy not in updated:
            raise ValueError(f"could not locate OnDestroy in {name}")
        updated = updated.replace(
            destroy,
            destroy + "\n            RemoveTriBenchCameraDraw();",
            1,
        )
    return updated


def _patch_renderer_base(source: str) -> str:
    marker = "Synchronizes per-camera shader state before rendering."
    if marker in source:
        return source
    anchor = "        public virtual void SetOutputRawCodeValues(bool enabled) { }"
    if anchor not in source:
        raise ValueError("could not locate TriAssetRenderer output-mode hook")
    return source.replace(
        anchor,
        anchor
        + "\n\n        /// <summary>Synchronizes per-camera shader state before rendering.</summary>"
        + "\n        public virtual void PrepareCamera(Camera camera) { }",
        1,
    )


def _patch_method_specific_camera_state(source: str) -> str:
    marker = "public override void PrepareCamera(Camera camera)"
    if marker in source:
        install_guard = '''            if (triBenchDrawCommands == null)
            {
                TargetCamera = camera;
                if (!InstallTriBenchCameraDraw()) { enabled = false; return; }
            }
'''
        anchor = "            if (camera == null || material == null) return;\n"
        if install_guard not in source:
            if anchor not in source:
                raise ValueError("could not migrate MethodSpecific PrepareCamera")
            source = source.replace(anchor, anchor + install_guard, 1)
        return source
    anchor = "        public override void SetOutputRawCodeValues(bool enabled) { rawCode = enabled; }"
    if anchor not in source:
        raise ValueError("could not locate MethodSpecific output-mode hook")
    replacement = '''        public override void PrepareCamera(Camera camera)
        {
            if (camera == null || material == null) return;
            if (triBenchDrawCommands == null)
            {
                TargetCamera = camera;
                if (!InstallTriBenchCameraDraw()) { enabled = false; return; }
            }
            material.SetVector("_CameraWorldPos", camera.transform.position);
            material.SetInt("_RawCode", rawCode ? 1 : 0);
            material.SetFloat("_OpacityFloor", opacityFloor);
            if (mode == 1)
            {
                material.SetInt("_MeshAblation", meshAblation);
                material.SetInt("_UseSh", (meshAblation == 1 || meshAblation == 4) ? 0 : 1);
            }
        }
        public override void SetOutputRawCodeValues(bool enabled)
        {
            rawCode = enabled;
            if (material != null) material.SetInt("_RawCode", enabled ? 1 : 0);
        }'''
    return source.replace(anchor, replacement, 1)


def _patch_generic_prepare_camera(source: str, name: str) -> str:
    """Install a procedural draw only after capture assigned its target RT."""
    marker = "public override void PrepareCamera(Camera camera)"
    if marker in source:
        return source
    callback = "        void OnRenderObject()"
    if callback not in source:
        raise ValueError(f"could not locate camera preparation anchor in {name}")
    method = '''        public override void PrepareCamera(Camera camera)
        {
            if (camera == null || material == null) return;
            if (triBenchDrawCommands == null)
            {
                TargetCamera = camera;
                if (!InstallTriBenchCameraDraw()) enabled = false;
            }
        }

'''
    return source.replace(callback, method + callback, 1)


def _patch_method_specific_vulkan_bindings(source: str) -> str:
    """Bind buffers that Vulkan requires even behind a runtime shader branch."""
    fixed = 'material.SetBuffer("_Sigma", sigma != null ? sigma : opacity);'
    if fixed in source:
        return source
    legacy = 'if (sigma != null) material.SetBuffer("_Sigma", sigma);'
    if legacy not in source:
        raise ValueError("could not locate MethodSpecific _Sigma binding")
    return source.replace(
        legacy,
        # 2DTS does not export sigma and the mode-2 shader branch never reads
        # it, but Vulkan descriptor validation still requires a bound buffer.
        fixed,
        1,
    )


def _capture_helpers() -> str:
    return r'''

        // TriBench Vulkan premultiplied-capture patch. The method-aware shaders
        // use front-to-back premultiplied accumulation. Destination alpha must
        // start at zero; Color.black/white both carry alpha one in Unity.
        void PreparePremultipliedCaptureTarget()
        {
            CaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            CaptureCamera.backgroundColor = new Color(0f, 0f, 0f, 0f);
        }

        void CompositePremultipliedBackground(Texture2D image)
        {
            bool white = String.Equals(
                TriAssetRuntimeOptions.Get("-background-color", "black"),
                "white",
                StringComparison.OrdinalIgnoreCase);
            int background = white ? 255 : 0;
            Color32[] pixels = image.GetPixels32();
            int coveredPixels = 0;
            for (int i = 0; i < pixels.Length; ++i)
            {
                Color32 p = pixels[i];
                if (p.a != 0) coveredPixels++;
                int remaining = 255 - p.a;
                p.r = (byte)Mathf.Clamp(p.r + (background * remaining + 127) / 255, 0, 255);
                p.g = (byte)Mathf.Clamp(p.g + (background * remaining + 127) / 255, 0, 255);
                p.b = (byte)Mathf.Clamp(p.b + (background * remaining + 127) / 255, 0, 255);
                p.a = 255;
                pixels[i] = p;
            }
            image.SetPixels32(pixels);
            image.Apply(false, false);
            if (coveredPixels == 0)
                Debug.LogError("[TriBench] Capture target has zero alpha coverage; no procedural geometry reached the camera.");
            else
                Debug.Log($"[TriBench] Capture alpha coverage: {coveredPixels}/{pixels.Length} pixels.");
        }
'''


def _patch_capture(source: str) -> str:
    updated = source.replace("TextureFormat.RGB24", "TextureFormat.RGBA32")
    if "PreparePremultipliedCaptureTarget();\n                CaptureCamera.Render();" not in updated:
        updated = updated.replace(
            "CaptureCamera.Render();",
            "PreparePremultipliedCaptureTarget();\n                CaptureCamera.Render();",
        )
    if "CompositePremultipliedBackground(readback);" not in updated:
        updated = updated.replace(
            "readback.Apply(false, false);",
            "readback.Apply(false, false);\n                CompositePremultipliedBackground(readback);",
        )
    if CAPTURE_PATCH_MARKER not in updated:
        class_end = updated.rfind("\n    }\n}")
        if class_end < 0:
            raise ValueError("could not locate ColmapBatchCapture class terminator")
        updated = updated[:class_end] + _capture_helpers() + updated[class_end:]
    elif "Capture alpha coverage" not in updated:
        # Upgrade projects that already received the first premultiplied-alpha
        # patch without duplicating its helper methods.
        updated = updated.replace(
            "            Color32[] pixels = image.GetPixels32();\n"
            "            for (int i = 0; i < pixels.Length; ++i)",
            "            Color32[] pixels = image.GetPixels32();\n"
            "            int coveredPixels = 0;\n"
            "            for (int i = 0; i < pixels.Length; ++i)",
            1,
        )
        updated = updated.replace(
            "                Color32 p = pixels[i];\n"
            "                int remaining = 255 - p.a;",
            "                Color32 p = pixels[i];\n"
            "                if (p.a != 0) coveredPixels++;\n"
            "                int remaining = 255 - p.a;",
            1,
        )
        updated = updated.replace(
            "            image.SetPixels32(pixels);\n"
            "            image.Apply(false, false);",
            "            image.SetPixels32(pixels);\n"
            "            image.Apply(false, false);\n"
            "            if (coveredPixels == 0)\n"
            "                Debug.LogError(\"[TriBench] Capture target has zero alpha coverage; no procedural geometry reached the camera.\");\n"
            "            else\n"
            "                Debug.Log($\"[TriBench] Capture alpha coverage: {coveredPixels}/{pixels.Length} pixels.\");",
            1,
        )
    # A camera command buffer has the correct view/projection globals during
    # Camera.Render(), but custom per-view uniforms must be synchronized after
    # each COLMAP pose change and before the camera starts rendering frames.
    pose = "ApplyColmapPose(CaptureCamera, view, intr);"
    prepare = pose + "\n                Renderer.PrepareCamera(CaptureCamera);"
    if prepare not in updated:
        updated = updated.replace(pose, prepare)
    return updated


def _patch_profile_player_build(source: str) -> str:
    marker = "TriBench Linux standalone profile player patch"
    if marker in source:
        return source
    if "target = BuildTarget.StandaloneOSX" not in source:
        raise ValueError("could not locate hard-coded standalone build target")
    updated = source
    updated = updated.replace(
        '''            BuildPlayerOptions options = new BuildPlayerOptions {
                scenes = new[] { "Assets/Scenes/TriBenchGarden.unity" },
                locationPathName = output,
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.Development,
            };''',
        '''            BuildTarget target = ParseTarget(GetArgument("-profile-player-target"));
            BuildPlayerOptions options = new BuildPlayerOptions {
                scenes = new[] { "Assets/Scenes/TriBenchGarden.unity" },
                locationPathName = output,
                target = target,
                // TriBench Linux standalone profile player patch.
                options = BuildOptions.Development,
            };''',
        1,
    )
    anchor = '''        static string GetArgument(string name)
        {'''
    if anchor not in updated:
        raise ValueError("could not locate GetArgument in profile player build source")
    helper = '''        static BuildTarget ParseTarget(string value)
        {
            string normalized = String.IsNullOrWhiteSpace(value) ? "linux64" : value.Trim().ToLowerInvariant();
            switch (normalized)
            {
                case "linux":
                case "linux64":
                case "standalonelinux64":
                    return BuildTarget.StandaloneLinux64;
                case "mac":
                case "macos":
                case "osx":
                case "standaloneosx":
                    return BuildTarget.StandaloneOSX;
                case "win":
                case "windows":
                case "windows64":
                case "standalonewindows64":
                    return BuildTarget.StandaloneWindows64;
                default:
                    throw new ArgumentException("Unsupported TriBench profile player target: " + value);
            }
        }

'''
    return updated.replace(anchor, helper + anchor, 1)


def patch_unity_project(project: Path) -> list[Path]:
    assets = project.expanduser().resolve() / "Assets" / "TriBench"
    if not assets.is_dir():
        raise FileNotFoundError(f"legacy TriBench Unity assets not found: {assets}")

    changed: list[Path] = []
    shader_root = assets / "Shaders"
    for path in sorted(shader_root.glob("*.shader")):
        original = path.read_text(encoding="utf-8")
        if _write_if_changed(path, original, _patch_shader(original)):
            changed.append(path)

    script_root = assets / "Scripts"
    renderer_base = script_root / "TriAssetRenderer.cs"
    if not renderer_base.is_file():
        raise FileNotFoundError(f"legacy renderer base source not found: {renderer_base}")
    original = renderer_base.read_text(encoding="utf-8")
    if _write_if_changed(renderer_base, original, _patch_renderer_base(original)):
        changed.append(renderer_base)

    capture = script_root / "ColmapBatchCapture.cs"
    if not capture.is_file():
        raise FileNotFoundError(f"legacy capture source not found: {capture}")
    original = capture.read_text(encoding="utf-8")
    if _write_if_changed(capture, original, _patch_capture(original)):
        changed.append(capture)

    renderer_counts = {
        "MethodSpecificSplatRenderer.cs": "primitiveCount",
        "DiffSoupTriAssetRenderer.cs": "primitiveCount",
        "TriangleSplattingTriAssetRenderer.cs": "PrimitiveCount",
    }
    for name, primitive_count in renderer_counts.items():
        path = script_root / name
        if not path.is_file():
            continue
        original = path.read_text(encoding="utf-8")
        updated = _patch_renderer(original)
        updated = _patch_renderer_command_buffer(updated, primitive_count, path.stem)
        if name == "MethodSpecificSplatRenderer.cs":
            updated = _patch_method_specific_vulkan_bindings(updated)
            updated = _patch_method_specific_camera_state(updated)
        else:
            updated = _patch_generic_prepare_camera(updated, path.stem)
        if _write_if_changed(path, original, updated):
            changed.append(path)

    build = assets / "Editor" / "TriBenchProfilePlayerBuild.cs"
    if build.is_file():
        original = build.read_text(encoding="utf-8")
        if _write_if_changed(build, original, _patch_profile_player_build(original)):
            changed.append(build)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch a legacy TriBenchUnity Metal project for Vulkan capture."
    )
    parser.add_argument("--unity-project", type=Path, required=True)
    args = parser.parse_args()
    changed = patch_unity_project(args.unity_project)
    if changed:
        print("Patched legacy TriBench Unity project:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Legacy TriBench Unity project is already patched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
