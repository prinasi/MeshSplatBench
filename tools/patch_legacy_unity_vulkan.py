#!/usr/bin/env python3
"""Patch the legacy MeshSplatBenchUnity Metal renderer for safe Vulkan capture.

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

The operation is idempotent and preserves one ``.msbench-vulkan.bak`` copy of
every modified source file.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


BACKUP_SUFFIX = ".msbench-vulkan.bak"
CAPTURE_PATCH_MARKER = "MeshSplatBench Vulkan premultiplied-capture patch"


def _write_if_changed(path: Path, original: str, updated: str) -> bool:
    if updated == original:
        return False
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
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
    if 'Shader "MeshSplatBench/MethodSpecificSplat"' in updated or 'Shader "TriBench/MethodSpecificSplat"' in updated:
        updated = updated.replace(
            "ByteAddressBuffer _Positions,_Indices,_Opacity,_Sigma,_ShDc,_ShRest;",
            "ByteAddressBuffer _Positions,_Indices,_TriangleOrder,_Opacity,_Sigma,_ShDc,_ShRest;",
            1,
        )
        updated = updated.replace(
            "struct V{float4 p:SV_POSITION;float3 b:TEXCOORD0;",
            "struct V{float4 p:SV_POSITION;noperspective float3 b:TEXCOORD0;",
            1,
        )
        updated = updated.replace(
            "uint t=id/3,corner=id-t*3,ib=t*3;",
            "uint t=id/3,corner=id-t*3,source=I(_TriangleOrder,t),ib=source*3;",
            1,
        )
        updated = updated.replace(
            "uint e=_Mode==1?vi:t;",
            "uint e=_Mode==1?vi:source;",
            1,
        )
    return updated


def _indexed_mesh_splat_shader() -> str:
    return r'''// True Unity indexed MeshRenderer path for MeshSplatting topology ablations.
Shader "MeshSplatBench/MeshSplatIndexedMesh"
{
 SubShader { Tags { "Queue"="Geometry" "RenderType"="Opaque" } Pass {
  Cull Off ZWrite On ZTest LEqual Blend Off
  CGPROGRAM
  #pragma target 5.0
  #pragma only_renderers metal vulkan
  #pragma vertex Vert
  #pragma fragment Frag
  #include "UnityCG.cginc"
  ByteAddressBuffer _ShDc,_ShRest;
  int _ShDegree,_RawCode; float3 _CameraWorldPos;
  float F(ByteAddressBuffer b,uint i){return asfloat(b.Load(i*4));}
  float3 Dc(uint e){uint k=e*3;return float3(F(_ShDc,k),F(_ShDc,k+1),F(_ShDc,k+2));}
  float3 Rest(uint e,uint c){uint k=e*45+c*3;return float3(F(_ShRest,k),F(_ShRest,k+1),F(_ShRest,k+2));}
  float3 Sh(uint e,float3 d){ float3 r=0.28209479177387814*Dc(e); if(_ShDegree<1)return max(r+.5,0); float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z; r+=-.4886025119*y*Rest(e,0)+.4886025119*z*Rest(e,1)-.4886025119*x*Rest(e,2); if(_ShDegree>1)r+=1.09254843*xy*Rest(e,3)-1.09254843*yz*Rest(e,4)+.315391565*(2*zz-xx-yy)*Rest(e,5)-1.09254843*xz*Rest(e,6)+.546274215*(xx-yy)*Rest(e,7); if(_ShDegree>2)r+=-.59004359*y*(3*xx-yy)*Rest(e,8)+2.89061144*xy*z*Rest(e,9)-.4570458*y*(4*zz-xx-yy)*Rest(e,10)+.37317633*z*(2*zz-3*xx-3*yy)*Rest(e,11)-.4570458*x*(4*zz-xx-yy)*Rest(e,12)+1.44530572*z*(xx-yy)*Rest(e,13)-.59004359*x*(xx-3*yy)*Rest(e,14); return max(r+.5,0); }
  struct V{float4 p:SV_POSITION;noperspective float3 c:TEXCOORD0;};
  V Vert(float4 vertex:POSITION,uint vid:SV_VertexID){V o;float3 world=mul(unity_ObjectToWorld,vertex).xyz;o.p=UnityObjectToClipPos(vertex);o.c=Sh(vid,normalize(world-_CameraWorldPos));return o;}
  float4 Frag(V i):SV_Target{float3 c=_RawCode!=0?i.c:GammaToLinearSpace(i.c);return float4(c,1);}
  ENDCG
 } }
}
'''


def _patch_renderer(source: str) -> str:
    failure = (
        'if (!material.SetPass(0)) { Debug.LogError("[MeshSplatBench] Shader pass is '
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


def _patch_standard_mesh_indexed_method_aware(source: str) -> str:
    marker = "indexed MeshSplatting method-aware MeshRenderer path"
    updated = source
    indexed_fields = (
        "        ComputeBuffer shDcBuffer, shRestBuffer;\n"
        "        bool indexedMeshMethodAware, rawCode = true;\n"
        "        int activeShDegree;\n"
    )
    indexed_awake = (
        '            string method = TriAssetRuntimeOptions.Get("-method", "").ToLowerInvariant();\n'
        '            indexedMeshMethodAware = method == "mesh-splatting" && TriAssetRuntimeOptions.Has("-indexed-mesh-method-aware");\n'
    )
    # Older development revisions of this patch were not idempotent.  Normalize
    # already-patched projects before applying any missing pieces below.
    updated = re.sub(r"(?:" + re.escape(indexed_fields) + r"){2,}", indexed_fields, updated)
    updated = re.sub(r"(?:" + re.escape(indexed_awake) + r"){2,}", indexed_awake, updated)
    updated = updated.replace(
        'Regex.Match(json,"\\\\\\""+key+"\\\\\\"\\\\s*:\\\\s*(\\\\\\\\d+)")',
        'Regex.Match(json,"\\\\\\""+key+"\\\\\\"\\\\s*:\\\\s*(\\\\d+)")',
    )
    updated = updated.replace(
        '''    /// Method-agnostic CG compatibility baseline: an indexed Unity Mesh with
    /// a conventional opaque vertex-color material.  SH DC is baked once at
    /// import to RGB; no SH evaluation, sigmoid/sigma activation, procedural
    /// draw, custom sorting, or splat coverage is performed at runtime.''',
        '''    /// Unity MeshRenderer deployment path: by default this is a method-agnostic
    /// indexed Mesh baseline with baked DC vertex color; with
    /// -indexed-mesh-method-aware for MeshSplatting it keeps the real Unity
    /// indexed Mesh draw while evaluating the learned SH appearance in shader.''',
    )
    updated = updated.replace(
        '''            string shaderName = indexedMeshMethodAware ? "MeshSplatBench/MeshSplatIndexedMesh" : "MeshSplatBench/StandardVertexColorRaw";
            VertexColorShader = VertexColorShader != null ? VertexColorShader : Shader.Find(shaderName);
            if (VertexColorShader == null) { Fail(shaderName + " shader was not found."); yield break; }''',
        '''            string shaderName = indexedMeshMethodAware ? "MeshSplatBench/MeshSplatIndexedMesh" : "MeshSplatBench/StandardVertexColorRaw";
            VertexColorShader = indexedMeshMethodAware ? (Resources.Load<Shader>("MeshSplatIndexedMesh") ?? Shader.Find(shaderName)) : (VertexColorShader != null ? VertexColorShader : Shader.Find(shaderName));
            if (VertexColorShader == null) { Fail(shaderName + " shader was not found."); yield break; }''',
    )
    updated = updated.replace(
        '''            string shaderName = indexedMeshMethodAware ? "MeshSplatBench/MeshSplatIndexedMesh" : "MeshSplatBench/StandardVertexColorRaw";
            VertexColorShader = indexedMeshMethodAware ? Shader.Find(shaderName) : (VertexColorShader != null ? VertexColorShader : Shader.Find(shaderName));
            if (VertexColorShader == null) { Fail(shaderName + " shader was not found."); yield break; }''',
        '''            string shaderName = indexedMeshMethodAware ? "MeshSplatBench/MeshSplatIndexedMesh" : "MeshSplatBench/StandardVertexColorRaw";
            VertexColorShader = indexedMeshMethodAware ? (Resources.Load<Shader>("MeshSplatIndexedMesh") ?? Shader.Find(shaderName)) : (VertexColorShader != null ? VertexColorShader : Shader.Find(shaderName));
            if (VertexColorShader == null) { Fail(shaderName + " shader was not found."); yield break; }''',
    )
    updated = updated.replace(
        "            Color32[] colors = new Color32[vertexCount];",
        "            Color32[] colors = indexedMeshMethodAware ? null : new Color32[vertexCount];",
    )
    updated = updated.replace(
        '            status = File.Exists(dcPath) ? "Baking SH DC to standard vertex RGB" : "No standard color field; using default white material";',
        '            status = indexedMeshMethodAware ? "Preparing method-aware indexed MeshSplatting mesh" : (File.Exists(dcPath) ? "Baking SH DC to standard vertex RGB" : "No standard color field; using default white material");',
    )
    updated = updated.replace(
        "            if (File.Exists(dcPath))\n            {",
        "            if (!indexedMeshMethodAware && File.Exists(dcPath))\n            {",
        1,
    )
    updated = updated.replace(
        "            else\n                for (int i = 0; i < vertexCount; ++i) colors[i] = new Color32(255, 255, 255, 255);",
        "            else if (!indexedMeshMethodAware)\n                for (int i = 0; i < vertexCount; ++i) colors[i] = new Color32(255, 255, 255, 255);",
    )
    updated = updated.replace(
        "                Color32[] soupColors = new Color32[indices.Length];",
        "                Color32[] soupColors = colors == null ? null : new Color32[indices.Length];",
    )
    updated = updated.replace(
        "                    soupColors[corner] = colors[source];",
        "                    if (soupColors != null) soupColors[corner] = colors[source];",
    )
    updated = updated.replace(
        "                colors = soupColors;",
        "                if (soupColors != null) colors = soupColors;",
    )
    updated = updated.replace(
        "            mesh.colors32 = colors;",
        "            if (colors != null) mesh.colors32 = colors;",
    )
    if "using System.Text.RegularExpressions;" not in updated:
        updated = updated.replace("using System.IO;", "using System.IO;\nusing System.Text.RegularExpressions;", 1)
    if "ComputeBuffer shDcBuffer, shRestBuffer;" not in updated:
        updated = updated.replace(
            "        Mesh mesh;\n        Material material;",
            "        Mesh mesh;\n        Material material;\n" + indexed_fields.rstrip("\n"),
            1,
        )
    if "-indexed-mesh-method-aware" not in updated:
        updated = updated.replace(
            '''            string topology = TriAssetRuntimeOptions.Get("-topology", "indexed").ToLowerInvariant();
            DeindexedSoup = topology == "soup" || TriAssetRuntimeOptions.Has("-deindexed-soup");''',
            '''            string topology = TriAssetRuntimeOptions.Get("-topology", "indexed").ToLowerInvariant();
            DeindexedSoup = topology == "soup" || TriAssetRuntimeOptions.Has("-deindexed-soup");
            string method = TriAssetRuntimeOptions.Get("-method", "").ToLowerInvariant();
            indexedMeshMethodAware = method == "mesh-splatting" && TriAssetRuntimeOptions.Has("-indexed-mesh-method-aware");''',
            1,
        )
    if "string restPath = Path.Combine(buffers, \"sh_rest.bin\");" not in updated:
        updated = updated.replace(
            '''            string dcPath = Path.Combine(buffers, "sh_dc.bin");
            if (!File.Exists(positionsPath) || !File.Exists(indicesPath))''',
            '''            string dcPath = Path.Combine(buffers, "sh_dc.bin");
            string restPath = Path.Combine(buffers, "sh_rest.bin");
            string manifestPath = Path.Combine(AssetDirectory ?? "", "manifest.json");
            activeShDegree = File.Exists(manifestPath) ? ReadInt(File.ReadAllText(manifestPath), "active_sh_degree", 0) : 0;
            if (indexedMeshMethodAware && (!File.Exists(dcPath) || !File.Exists(restPath)))
            {
                Fail("Indexed MeshSplatting method-aware renderer requires sh_dc and sh_rest buffers.");
                yield break;
            }
            if (!File.Exists(positionsPath) || !File.Exists(indicesPath))''',
            1,
        )
    if "MeshSplatBench/MeshSplatIndexedMesh" not in updated:
        updated = updated.replace(
            '''            VertexColorShader = VertexColorShader != null ? VertexColorShader : Shader.Find("MeshSplatBench/StandardVertexColorRaw");
            if (VertexColorShader == null) { Fail("MeshSplatBench/StandardVertexColorRaw shader was not found."); yield break; }
            material = new Material(VertexColorShader) { hideFlags = HideFlags.HideAndDontSave };''',
            '''            string shaderName = indexedMeshMethodAware ? "MeshSplatBench/MeshSplatIndexedMesh" : "MeshSplatBench/StandardVertexColorRaw";
            VertexColorShader = indexedMeshMethodAware ? (Resources.Load<Shader>("MeshSplatIndexedMesh") ?? Shader.Find(shaderName)) : (VertexColorShader != null ? VertexColorShader : Shader.Find(shaderName));
            if (VertexColorShader == null) { Fail(shaderName + " shader was not found."); yield break; }
            material = new Material(VertexColorShader) { hideFlags = HideFlags.HideAndDontSave };
            if (indexedMeshMethodAware)
            {
                shDcBuffer = Upload(ReadFloatArray(dcPath));
                shRestBuffer = Upload(ReadFloatArray(restPath));
                material.SetBuffer("_ShDc", shDcBuffer);
                material.SetBuffer("_ShRest", shRestBuffer);
                material.SetInt("_ShDegree", activeShDegree);
                material.SetInt("_RawCode", rawCode ? 1 : 0);
                material.SetVector("_CameraWorldPos", TargetCamera != null ? TargetCamera.transform.position : Vector3.zero);
                Debug.Log("[MeshSplatBench] Using true indexed MeshSplatting method-aware MeshRenderer path.");
            }''',
            1,
        )
    status_old = 'status = $"Standard {(DeindexedSoup ? "triangle-soup" : "indexed Mesh")} ready: {vertexCount:N0} vertices, {triangleCount:N0} triangles" + (File.Exists(dcPath) ? "; DC vertex color" : "; default white (no standard appearance field)");'
    if status_old in updated:
        updated = updated.replace(
            status_old,
            'status = $"Standard {(DeindexedSoup ? "triangle-soup" : "indexed Mesh")} ready: {vertexCount:N0} vertices, {triangleCount:N0} triangles" + (indexedMeshMethodAware ? "; full SH indexed MeshSplatting shader" : (File.Exists(dcPath) ? "; DC vertex color" : "; default white (no standard appearance field)"));',
            1,
        )
    helper_anchor = "        static byte ToByte(float value)"
    if marker not in updated:
        helper = f'''        // MeshSplatBench {marker}.
        public override void PrepareCamera(Camera camera)
        {{
            if (indexedMeshMethodAware && material != null && camera != null)
                material.SetVector("_CameraWorldPos", camera.transform.position);
        }}

        public override void SetOutputRawCodeValues(bool enabled)
        {{
            rawCode = enabled;
            if (material != null) material.SetInt("_RawCode", enabled ? 1 : 0);
        }}

'''
        if helper_anchor not in updated:
            raise ValueError("could not locate StandardMesh helper anchor")
        updated = updated.replace(helper_anchor, helper + helper_anchor, 1)
    if "static int ReadInt(string json" not in updated:
        updated = updated.replace(
            "        static byte ToByte(float value)",
            r'''        static int ReadInt(string json, string key, int fallback) { Match m=Regex.Match(json,"\\\""+key+"\\\"\\s*:\\s*(\\d+)"); return m.Success && Int32.TryParse(m.Groups[1].Value,out int v) ? v : fallback; }
        static ComputeBuffer Upload(Array values) { ComputeBuffer b=new ComputeBuffer(values.Length,4,ComputeBufferType.Raw); b.SetData(values); return b; }
        static byte ToByte(float value)''',
            1,
        )
    if "static ComputeBuffer Upload(Array values)" not in updated:
        updated = updated.replace(
            "        static byte ToByte(float value)",
            "        static ComputeBuffer Upload(Array values) { ComputeBuffer b=new ComputeBuffer(values.Length,4,ComputeBufferType.Raw); b.SetData(values); return b; }\n"
            "        static byte ToByte(float value)",
            1,
        )
    if "shDcBuffer?.Release();" not in updated:
        updated = updated.replace(
            "            if (mesh != null) Destroy(mesh);\n            if (material != null) Destroy(material);",
            "            if (mesh != null) Destroy(mesh);\n            shDcBuffer?.Release();\n            shRestBuffer?.Release();\n            shDcBuffer = null;\n            shRestBuffer = null;\n            if (material != null) Destroy(material);",
            1,
        )
    return updated


def _patch_renderer_command_buffer(source: str, primitive_count: str, name: str) -> str:
    """Submit a legacy procedural renderer through the target camera itself."""
    marker = "MeshSplatBench explicit camera command-buffer patch"
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
            "            if (!InstallMeshSplatBenchCameraDraw()) yield break;\n",
            "",
            1,
        )
        callback_guard = "            if (msBenchDrawCommands != null) return;\n"
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
        + "\n        CommandBuffer msBenchDrawCommands;"
        + "\n        Camera msBenchCommandCamera;",
        1,
    )

    helper = f'''        // MeshSplatBench explicit camera command-buffer patch. OnRenderObject is
        // not a reliable callback for Camera.Render() on Linux Editor batch mode.
        bool InstallMeshSplatBenchCameraDraw()
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
            msBenchDrawCommands = new CommandBuffer {{ name = "MeshSplatBench/{name} procedural draw" }};
            msBenchDrawCommands.DrawProcedural(
                Matrix4x4.identity, material, 0, MeshTopology.Triangles,
                {primitive_count} * 3, 1);
            msBenchCommandCamera = TargetCamera;
            msBenchCommandCamera.AddCommandBuffer(CameraEvent.AfterForwardAlpha, msBenchDrawCommands);
            Debug.Log("[MeshSplatBench] Installed explicit camera draw for {name} on "
                + SystemInfo.graphicsDeviceType + ".");
            return true;
        }}

        void RemoveMeshSplatBenchCameraDraw()
        {{
            if (msBenchCommandCamera != null && msBenchDrawCommands != null)
                msBenchCommandCamera.RemoveCommandBuffer(CameraEvent.AfterForwardAlpha, msBenchDrawCommands);
            if (msBenchDrawCommands != null) msBenchDrawCommands.Release();
            msBenchDrawCommands = null;
            msBenchCommandCamera = null;
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
        + "\n            if (msBenchDrawCommands != null) return;"
        + "\n            if (Application.isBatchMode) return;",
        1,
    )

    # Older revisions installed the command during renderer Start(). In a
    # headless Editor frame this happens before ColmapBatchCapture has assigned
    # its offscreen RenderTexture, so Vulkan tries to draw into a nonexistent
    # screen framebuffer. Installation is now deferred to PrepareCamera().
    updated = updated.replace(
        "            if (!InstallMeshSplatBenchCameraDraw()) yield break;\n",
        "",
        1,
    )

    destroy = "void OnDestroy() {"
    if destroy in updated:
        updated = updated.replace(
            destroy,
            destroy + " RemoveMeshSplatBenchCameraDraw();",
            1,
        )
    else:
        destroy = "        void OnDestroy()\n        {"
        if destroy not in updated:
            raise ValueError(f"could not locate OnDestroy in {name}")
        updated = updated.replace(
            destroy,
            destroy + "\n            RemoveMeshSplatBenchCameraDraw();",
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
    # Older revisions sorted a flattened index buffer.  That detached 2DTS
    # per-triangle SH/opacity from geometry because the shader still indexed
    # attributes by draw order.  Keep one sorted primitive-ID buffer instead.
    source = source.replace("        int[] sourceTriangleIndices;\n", "")
    source = source.replace("        int[] sortedTriangleIndices;\n", "")
    field_anchor = "        string status = \"Waiting to load method-specific renderer\";\n"
    field_fallback_anchor = "        int primitiveCount; bool ready, rawCode; float opacityFloor; int mode, meshAblation;\n"
    field_block = (
        "        ComputeBuffer triangleOrderBuffer;\n"
        "        bool usesSortedTriangleOrder;\n"
        "        Vector3[] triangleCentroids;\n"
        "        int[] triangleOrder;\n"
        "        float[] triangleDepths;\n"
        "        Vector3 lastSortCameraPosition = new Vector3(float.NaN, float.NaN, float.NaN);\n"
        "        Vector3 lastSortCameraForward = new Vector3(float.NaN, float.NaN, float.NaN);\n"
    )
    field_tokens = (
        "triangleOrderBuffer",
        "usesSortedTriangleOrder",
        "triangleCentroids",
        "triangleOrder",
        "triangleDepths",
        "lastSortCameraPosition",
        "lastSortCameraForward",
    )
    if not all(token in source for token in field_tokens):
        if field_anchor in source:
            source = source.replace(field_anchor, field_anchor + field_block, 1)
        elif field_fallback_anchor in source:
            source = source.replace(field_fallback_anchor, field_fallback_anchor + field_block, 1)
        else:
            raise ValueError("could not locate MethodSpecific field anchor")

    shader_old = '''            string shaderName = mode == 1 && ablation == "full"
                ? "MeshSplatBench/MeshSplatTerminalSolid"
                : mode == 1 && ablation == "alpha-test-depth"
                    ? "MeshSplatBench/MeshSplatAlphaTestDepth"
                    : mode == 1 && ablation == "opaque-depth"
                        ? "MeshSplatBench/MeshSplatOpaqueDepth"
                        : "MeshSplatBench/MethodSpecificSplat";
            SplatShader = SplatShader != null ? SplatShader : Shader.Find(shaderName);
'''
    shader_new = '''            string shaderName = mode == 1 && ablation == "full"
                ? "MeshSplatBench/MeshSplatTerminalSolid"
                : mode == 1 && ablation == "alpha-test-depth"
                    ? "MeshSplatBench/MeshSplatAlphaTestDepth"
                    : mode == 1 && ablation == "opaque-depth"
                        ? "MeshSplatBench/MeshSplatOpaqueDepth"
                        : "MeshSplatBench/MethodSpecificSplat";
            usesSortedTriangleOrder = shaderName == "MeshSplatBench/MethodSpecificSplat";
            SplatShader = SplatShader != null ? SplatShader : Shader.Find(shaderName);
'''
    if shader_new not in source:
        if shader_old not in source:
            raise ValueError("could not locate MethodSpecific shader selection")
        source = source.replace(shader_old, shader_new, 1)

    source_indices_anchor = '''            int[] sourceIndices = ReadInt(Path.Combine(b, "indices.bin"));
            if (sourceIndices.Length != primitiveCount * 3) { Fail("indices.bin does not match primitive_count."); yield break; }
'''
    source_positions_block = '''            float[] sourcePositions = ReadFloat(Path.Combine(b, "positions.bin"));
            if (sourcePositions.Length % 3 != 0) { Fail("positions.bin is not float3 data."); yield break; }
            InitializeTriangleOrder(sourcePositions, sourceIndices);
'''
    if "InitializeTriangleOrder(sourcePositions, sourceIndices);" not in source:
        if source_indices_anchor not in source:
            raise ValueError("could not locate MethodSpecific positions/indices upload")
        source = source.replace(source_indices_anchor, source_indices_anchor + source_positions_block, 1)

    if 'positions = Upload(ReadFloat(Path.Combine(b, "positions.bin")));' in source:
        source = source.replace(
            'positions = Upload(ReadFloat(Path.Combine(b, "positions.bin")));',
            'positions = Upload(sourcePositions);',
            1,
        )
    elif 'positions = Upload(sourcePositions);' not in source:
        raise ValueError("could not locate MethodSpecific positions upload")

    source = source.replace(
        'triangleOrderBuffer = Upload(sourceIndices);',
        'triangleOrderBuffer = Upload(triangleOrder);',
        1,
    )
    if 'triangleOrderBuffer = Upload(triangleOrder);' not in source:
        if 'indices = Upload(sourceIndices);' not in source:
            raise ValueError("could not locate MethodSpecific index upload")
        source = source.replace(
            'indices = Upload(sourceIndices);',
            'indices = Upload(sourceIndices);\n            triangleOrderBuffer = Upload(triangleOrder);',
            1,
        )

    buffer_old = '            material.SetBuffer("_Positions", positions); material.SetBuffer("_Indices", indices);\n'
    buffer_buggy = '            material.SetBuffer("_Positions", positions); material.SetBuffer("_Indices", usesSortedTriangleOrder ? triangleOrderBuffer : indices);\n'
    buffer_new = buffer_old + '            if (usesSortedTriangleOrder) material.SetBuffer("_TriangleOrder", triangleOrderBuffer);\n'
    source = source.replace(buffer_buggy, buffer_new, 1)
    if buffer_new not in source:
        if buffer_old not in source:
            raise ValueError("could not locate MethodSpecific material index buffer binding")
        source = source.replace(buffer_old, buffer_new, 1)

    ready_old = '            material.SetInt("_RawCode", 1);\n            ready = true; status = $"Method-specific {method} renderer ready ({primitiveCount:N0} primitives; {(deindexedSoup ? "shader-level triangle soup" : "indexed mesh")})";\n'
    ready_new = '            material.SetInt("_RawCode", 1);\n            RefreshTriangleOrder(TargetCamera, force: true);\n            ready = true; status = $"Method-specific {method} renderer ready ({primitiveCount:N0} primitives; {(deindexedSoup ? "shader-level triangle soup" : "indexed mesh")})";\n'
    if ready_new not in source:
        if ready_old not in source:
            raise ValueError("could not locate MethodSpecific ready state")
        source = source.replace(ready_old, ready_new, 1)

    helper_anchor = '        // MeshSplatBench explicit camera command-buffer patch. OnRenderObject is\n'
    helper_block = '''        void InitializeTriangleOrder(float[] positionsData, int[] sourceIndices)
        {
            int count = Mathf.Min(primitiveCount, sourceIndices.Length / 3);
            primitiveCount = count;
            triangleCentroids = new Vector3[count];
            triangleOrder = new int[count];
            triangleDepths = new float[count];
            int vertexCount = positionsData.Length / 3;
            for (int t = 0; t < count; ++t)
            {
                int ib = t * 3;
                int ia = sourceIndices[ib];
                int ibv = sourceIndices[ib + 1];
                int ic = sourceIndices[ib + 2];
                if ((uint)ia >= (uint)vertexCount || (uint)ibv >= (uint)vertexCount || (uint)ic >= (uint)vertexCount)
                    throw new InvalidDataException("indices.bin references an invalid triangle vertex.");
                int a = ia * 3;
                int b = ibv * 3;
                int c = ic * 3;
                float cx = (positionsData[a] + positionsData[b] + positionsData[c]) / 3f;
                float cy = (positionsData[a + 1] + positionsData[b + 1] + positionsData[c + 1]) / 3f;
                float cz = (positionsData[a + 2] + positionsData[b + 2] + positionsData[c + 2]) / 3f;
                triangleCentroids[t] = new Vector3(cx, cy, cz);
                triangleOrder[t] = t;
            }
        }

        void RefreshTriangleOrder(Camera camera, bool force = false)
        {
            if (!usesSortedTriangleOrder) return;
            if (camera == null || triangleCentroids == null || triangleOrder == null || triangleDepths == null || triangleOrderBuffer == null) return;
            Vector3 cam = camera.transform.position;
            Vector3 forward = camera.transform.forward;
            if (!force && cam == lastSortCameraPosition && forward == lastSortCameraForward) return;
            for (int t = 0; t < triangleOrder.Length; ++t)
            {
                Vector3 center = triangleCentroids[t];
                triangleDepths[t] = Vector3.Dot(center - cam, forward);
                triangleOrder[t] = t;
            }
            Array.Sort(triangleDepths, triangleOrder);
            triangleOrderBuffer.SetData(triangleOrder);
            lastSortCameraPosition = cam;
            lastSortCameraForward = forward;
        }

'''
    helper_start = source.find("        void InitializeTriangleOrder(float[] positionsData, int[] sourceIndices)\n")
    helper_end = source.find(helper_anchor)
    if helper_start >= 0 and helper_end > helper_start:
        source = source[:helper_start] + helper_block + source[helper_end:]
    elif helper_block not in source:
        if helper_end < 0:
            raise ValueError("could not locate MethodSpecific helper anchor")
        source = source[:helper_end] + helper_block + source[helper_end:]

    render_anchor = "            if (!ready || Camera.current != TargetCamera || material == null) return;\n"
    render_refresh = render_anchor + "            RefreshTriangleOrder(TargetCamera);\n"
    if "RefreshTriangleOrder(TargetCamera);" not in source:
        if render_anchor in source:
            source = source.replace(render_anchor, render_refresh, 1)
        else:
            callback_anchor = "        void OnRenderObject()\n        {\n"
            if callback_anchor not in source:
                raise ValueError("could not locate MethodSpecific OnRenderObject anchor")
            source = source.replace(callback_anchor, callback_anchor + "            RefreshTriangleOrder(TargetCamera);\n", 1)

    old_prepare = '''        public override void PrepareCamera(Camera camera)
        {
            if (camera == null || material == null) return;
            if (msBenchDrawCommands == null)
            {
                TargetCamera = camera;
                if (!InstallMeshSplatBenchCameraDraw()) { enabled = false; return; }
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
'''
    new_prepare = '''        public override void PrepareCamera(Camera camera)
        {
            if (camera == null || material == null) return;
            if (TargetCamera != camera)
            {
                RemoveMeshSplatBenchCameraDraw();
                TargetCamera = camera;
            }
            if (msBenchDrawCommands == null)
            {
                if (!InstallMeshSplatBenchCameraDraw()) { enabled = false; return; }
            }
            RefreshTriangleOrder(camera);
            material.SetVector("_CameraWorldPos", camera.transform.position);
            material.SetInt("_RawCode", rawCode ? 1 : 0);
            material.SetFloat("_OpacityFloor", opacityFloor);
            if (mode == 1)
            {
                material.SetInt("_MeshAblation", meshAblation);
                material.SetInt("_UseSh", (meshAblation == 1 || meshAblation == 4) ? 0 : 1);
            }
        }
'''
    current_prepare = new_prepare.replace("MeshSplatBenchCameraDraw", "MsBenchCameraDraw")
    if new_prepare not in source and current_prepare not in source:
        if old_prepare in source:
            source = source.replace(old_prepare, new_prepare, 1)
        else:
            callback_anchor = "        void OnRenderObject()\n"
            if callback_anchor not in source:
                raise ValueError("could not locate MethodSpecific PrepareCamera")
            prepare_block = '''        public override void PrepareCamera(Camera camera)
        {
            if (camera == null || material == null) return;
            if (TargetCamera != camera)
            {
                RemoveMeshSplatBenchCameraDraw();
                TargetCamera = camera;
            }
            if (msBenchDrawCommands == null)
            {
                if (!InstallMeshSplatBenchCameraDraw()) { enabled = false; return; }
            }
            RefreshTriangleOrder(camera);
            material.SetVector("_CameraWorldPos", camera.transform.position);
            material.SetInt("_RawCode", rawCode ? 1 : 0);
            material.SetFloat("_OpacityFloor", opacityFloor);
            if (mode == 1)
            {
                material.SetInt("_MeshAblation", meshAblation);
                material.SetInt("_UseSh", (meshAblation == 1 || meshAblation == 4) ? 0 : 1);
            }
        }
'''
            source = source.replace(callback_anchor, prepare_block + callback_anchor, 1)

    destroy_match = re.search(r"(?m)^        void OnDestroy\(\) \{[^\n]*\}\n", source)
    if destroy_match is None:
        raise ValueError("could not locate MethodSpecific OnDestroy")
    destroy_line = destroy_match.group(0)
    if "triangleOrderBuffer?.Release();" not in destroy_line:
        if "indices?.Release();" not in destroy_line:
            raise ValueError("could not locate MethodSpecific index cleanup")
        updated_destroy_line = destroy_line.replace(
            "indices?.Release();",
            "indices?.Release(); triangleOrderBuffer?.Release();",
            1,
        )
        source = (
            source[: destroy_match.start()]
            + updated_destroy_line
            + source[destroy_match.end() :]
        )

    return source


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
            if (msBenchDrawCommands == null)
            {
                TargetCamera = camera;
                if (!InstallMeshSplatBenchCameraDraw()) enabled = false;
            }
        }

'''
    return source.replace(callback, method + callback, 1)


def _patch_triangle_splatting_camera_sort(source: str) -> str:
    """Refresh legacy triangle-splatting order buffers for every replay camera."""
    updated = source
    field_anchor = "        float displayedFps;\n"
    field_block = (
        "        Vector3[] triangleCentroids;\n"
        "        int[] triangleOrder;\n"
        "        float[] triangleDepths;\n"
        "        Vector3 lastSortCameraPosition = new Vector3(float.NaN, float.NaN, float.NaN);\n"
        "        Vector3 lastSortCameraForward = new Vector3(float.NaN, float.NaN, float.NaN);\n"
    )
    if field_block not in updated:
        if field_anchor not in updated:
            raise ValueError("could not locate TriangleSplatting fps fields")
        updated = updated.replace(field_anchor, field_anchor + field_block, 1)

    old_start = '''            status = SortFrontToBack ? "Sorting 4,517,295 primitives front-to-back" : "Creating primitive order";
            yield return null;
            int[] order = BuildOrder(positions, indices);
            orderBuffer = UploadRaw(order);
            positions = null;
            indices = null;
            order = null;
            GC.Collect();
'''
    new_start = '''            status = SortFrontToBack ? "Sorting 4,517,295 primitives front-to-back" : "Creating primitive order";
            yield return null;
            InitializeTriangleOrder(positions, indices);
            RefreshTriangleOrder(TargetCamera, force: true);
            positions = null;
            indices = null;
            GC.Collect();
'''
    if new_start not in updated:
        if old_start not in updated:
            raise ValueError("could not locate TriangleSplatting startup order build")
        updated = updated.replace(old_start, new_start, 1)

    old_order = '''        int[] BuildOrder(float[] positions, int[] indices)
        {
            int count = Mathf.Min(PrimitiveCount, indices.Length / 3);
            PrimitiveCount = count;
            int[] order = new int[count];
            if (!SortFrontToBack)
            {
                for (int i = 0; i < count; ++i) order[i] = i;
                return order;
            }

            float[] depths = new float[count];
            Vector3 cam = TargetCamera.transform.position;
            Vector3 forward = TargetCamera.transform.forward;
            for (int t = 0; t < count; ++t)
            {
                int ib = t * 3;
                int a = indices[ib] * 3;
                int b = indices[ib + 1] * 3;
                int c = indices[ib + 2] * 3;
                float cx = (positions[a] + positions[b] + positions[c]) / 3f;
                float cy = (positions[a + 1] + positions[b + 1] + positions[c + 1]) / 3f;
                float cz = (positions[a + 2] + positions[b + 2] + positions[c + 2]) / 3f;
                depths[t] = (cx - cam.x) * forward.x + (cy - cam.y) * forward.y + (cz - cam.z) * forward.z;
                order[t] = t;
            }
            Array.Sort(depths, order);
            return order;
        }
'''
    new_order = '''        void InitializeTriangleOrder(float[] positions, int[] indices)
        {
            int count = Mathf.Min(PrimitiveCount, indices.Length / 3);
            PrimitiveCount = count;
            triangleCentroids = new Vector3[count];
            triangleOrder = new int[count];
            triangleDepths = new float[count];
            for (int t = 0; t < count; ++t)
            {
                int ib = t * 3;
                int a = indices[ib] * 3;
                int b = indices[ib + 1] * 3;
                int c = indices[ib + 2] * 3;
                float cx = (positions[a] + positions[b] + positions[c]) / 3f;
                float cy = (positions[a + 1] + positions[b + 1] + positions[c + 1]) / 3f;
                float cz = (positions[a + 2] + positions[b + 2] + positions[c + 2]) / 3f;
                triangleCentroids[t] = new Vector3(cx, cy, cz);
                triangleOrder[t] = t;
            }
            orderBuffer = UploadRaw(triangleOrder);
        }

        void RefreshTriangleOrder(Camera camera, bool force = false)
        {
            if (camera == null || orderBuffer == null || triangleOrder == null || triangleCentroids == null) return;
            Vector3 cam = camera.transform.position;
            Vector3 forward = camera.transform.forward;
            if (!force && cam == lastSortCameraPosition && forward == lastSortCameraForward) return;

            if (!SortFrontToBack)
            {
                for (int i = 0; i < triangleOrder.Length; ++i) triangleOrder[i] = i;
            }
            else
            {
                for (int t = 0; t < triangleOrder.Length; ++t)
                {
                    Vector3 center = triangleCentroids[t];
                    triangleDepths[t] = Vector3.Dot(center - cam, forward);
                    triangleOrder[t] = t;
                }
                Array.Sort(triangleDepths, triangleOrder);
            }
            orderBuffer.SetData(triangleOrder);
            lastSortCameraPosition = cam;
            lastSortCameraForward = forward;
        }
'''
    if new_order not in updated:
        if old_order not in updated:
            raise ValueError("could not locate TriangleSplatting BuildOrder")
        updated = updated.replace(old_order, new_order, 1)

    old_prepare = '''        public override void PrepareCamera(Camera camera)
        {
            if (camera == null || material == null) return;
            if (msBenchDrawCommands == null)
            {
                TargetCamera = camera;
                if (!InstallMeshSplatBenchCameraDraw()) enabled = false;
            }
        }
'''
    new_prepare = '''        public override void PrepareCamera(Camera camera)
        {
            if (camera == null || material == null) return;
            if (TargetCamera != camera)
            {
                RemoveMeshSplatBenchCameraDraw();
                TargetCamera = camera;
            }
            if (msBenchDrawCommands == null)
            {
                if (!InstallMeshSplatBenchCameraDraw()) { enabled = false; return; }
            }
            RefreshTriangleOrder(camera);
        }
'''
    current_prepare = new_prepare.replace("MeshSplatBenchCameraDraw", "MsBenchCameraDraw")
    if new_prepare not in updated and current_prepare not in updated:
        if old_prepare not in updated:
            raise ValueError("could not locate TriangleSplatting PrepareCamera")
        updated = updated.replace(old_prepare, new_prepare, 1)

    render_anchor = "            if (!ready || Camera.current != TargetCamera || material == null) return;\n"
    render_refresh = render_anchor + "            RefreshTriangleOrder(TargetCamera);\n"
    if "RefreshTriangleOrder(TargetCamera);" not in updated:
        if render_anchor not in updated:
            raise ValueError("could not locate TriangleSplatting OnRenderObject")
        updated = updated.replace(render_anchor, render_refresh, 1)

    return updated


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


def _patch_method_specific_shader_level_soup(source: str) -> str:
    """Avoid CPU-side MeshSplatting de-indexing for the soup topology ablation.

    MethodSpecificSplat* shaders already draw procedurally with one vertex
    invocation per triangle corner and fetch the source vertex attributes via
    ``_Indices``.  Duplicating positions/SH buffers in C# therefore only adds
    multi-GiB upload pressure without changing the corner-level draw topology.
    """

    marker = "MeshSplatting shader-level soup topology"
    updated = source
    old = '''            if (deindexedSoup)
            {
                // Same-asset layout intervention: preserve every triangle
                // corner's position and learned attributes, while replacing
                // shared vertex indices with sequential soup indices.
                status = "De-indexing learned MeshSplatting attributes into triangle soup";
                yield return null;
                positions = Upload(Deindex(ReadFloat(Path.Combine(b, "positions.bin")), sourceIndices, 3, "positions"));
                opacity = Upload(Deindex(ReadFloat(Path.Combine(b, mode == 1 ? "vertex_weight_logits.bin" : "opacity_logits.bin")), sourceIndices, 1, "opacity"));
                dc = Upload(Deindex(ReadFloat(Path.Combine(b, "sh_dc.bin")), sourceIndices, 3, "sh_dc"));
                rest = Upload(Deindex(ReadFloat(Path.Combine(b, "sh_rest.bin")), sourceIndices, 45, "sh_rest"));
                int[] soupIndices = new int[sourceIndices.Length];
                for (int i = 0; i < soupIndices.Length; ++i) soupIndices[i] = i;
                indices = Upload(soupIndices);
            }
            else
            {
                positions = Upload(ReadFloat(Path.Combine(b, "positions.bin")));
                indices = Upload(sourceIndices);
                opacity = Upload(ReadFloat(Path.Combine(b, mode == 1 ? "vertex_weight_logits.bin" : "opacity_logits.bin")));
                dc = Upload(ReadFloat(Path.Combine(b, "sh_dc.bin")));
                rest = Upload(ReadFloat(Path.Combine(b, "sh_rest.bin")));
            }'''
    new = '''            if (deindexedSoup && mode == 1)
            {
                // MeshSplatting shader-level soup topology: keep the original
                // indexed buffers and let each procedural triangle corner fetch
                // source vertex attributes through _Indices in the shader.  This
                // disables hardware vertex sharing in the draw path without
                // materializing multi-GiB duplicate SH buffers on the CPU/GPU.
                Debug.Log("[MeshSplatBench] MeshSplatting shader-level soup topology: retaining indexed buffers; procedural corners fetch via _Indices.");
            }
            positions = Upload(ReadFloat(Path.Combine(b, "positions.bin")));
            indices = Upload(sourceIndices);
            opacity = Upload(ReadFloat(Path.Combine(b, mode == 1 ? "vertex_weight_logits.bin" : "opacity_logits.bin")));
            dc = Upload(ReadFloat(Path.Combine(b, "sh_dc.bin")));
            rest = Upload(ReadFloat(Path.Combine(b, "sh_rest.bin")));'''
    if old in updated:
        updated = updated.replace(old, new, 1)
    elif marker not in updated and "positions = Upload(Deindex(" in updated:
        raise ValueError("could not migrate MethodSpecific soup de-index branch")

    updated = updated.replace(
        'deindexedSoup ? "triangle soup" : "indexed mesh"',
        'deindexedSoup ? "shader-level triangle soup" : "indexed mesh"',
    )
    updated = updated.replace(
        'deindexedSoup ? "soup" : "indexed"',
        'deindexedSoup ? "soup(shader-indexed)" : "indexed"',
    )
    return updated


def _capture_helpers() -> str:
    return r'''

        // MeshSplatBench Vulkan premultiplied-capture patch. The method-aware shaders
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
                Debug.LogError("[MeshSplatBench] Capture target has zero alpha coverage; no procedural geometry reached the camera.");
            else
                Debug.Log($"[MeshSplatBench] Capture alpha coverage: {coveredPixels}/{pixels.Length} pixels.");
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
            "                Debug.LogError(\"[MeshSplatBench] Capture target has zero alpha coverage; no procedural geometry reached the camera.\");\n"
            "            else\n"
            "                Debug.Log($\"[MeshSplatBench] Capture alpha coverage: {coveredPixels}/{pixels.Length} pixels.\");",
            1,
        )
    # A camera command buffer has the correct view/projection globals during
    # Camera.Render(), but custom per-view uniforms must be synchronized after
    # each COLMAP pose change and before the camera starts rendering frames.
    pose = "ApplyColmapPose(CaptureCamera, view, intr);"
    prepare = pose + "\n                Renderer.PrepareCamera(CaptureCamera);"
    if prepare not in updated:
        updated = updated.replace(pose, prepare)
    old_timing = '''            FrameTiming[] timing = new FrameTiming[1];
            for (int i = 0; i < ProfileTimedFrames; ++i)
            {
                FrameTimingManager.CaptureFrameTimings();
                yield return null;
                engine.Add(Time.unscaledDeltaTime * 1000.0f);
                if (FrameTimingManager.GetLatestTimings(1, timing) > 0)
                {
                    if (timing[0].cpuFrameTime > 0.0) cpu.Add((float)timing[0].cpuFrameTime);
                    if (timing[0].gpuFrameTime > 0.0) gpu.Add((float)timing[0].gpuFrameTime);
                }
                if (drawCalls.Valid && drawCalls.LastValue > 0) draws.Add(drawCalls.LastValue);
            }
'''
    new_timing = '''            FrameTiming[] timing = new FrameTiming[8];
            for (int i = 0; i < ProfileTimedFrames; ++i)
            {
                FrameTimingManager.CaptureFrameTimings();
                yield return null;
                engine.Add(Time.unscaledDeltaTime * 1000.0f);
                // The GPU timestamp readback for a captured frame lands
                // asynchronously.  Scan a window of recent captures and retry
                // briefly until one of them has landed; this keeps the GPU
                // sample set complete even when the GPU is saturated.
                uint count = 0;
                float gpuMs = 0.0f;
                for (int attempt = 0; attempt < 16 && gpuMs <= 0.0f; ++attempt)
                {
                    count = FrameTimingManager.GetLatestTimings((uint)timing.Length, timing);
                    for (int k = 0; k < count; ++k)
                    {
                        if (timing[k].gpuFrameTime > 0.0)
                        {
                            gpuMs = (float)timing[k].gpuFrameTime;
                            break;
                        }
                    }
                    if (gpuMs <= 0.0f && attempt < 15) System.Threading.Thread.Sleep(1);
                }
                if (gpuMs > 0.0) gpu.Add(gpuMs);
                if (count > 0 && timing[0].cpuFrameTime > 0.0) cpu.Add((float)timing[0].cpuFrameTime);
                if (drawCalls.Valid && drawCalls.LastValue > 0) draws.Add(drawCalls.LastValue);
            }
'''
    if new_timing not in updated:
        if old_timing not in updated:
            # Optional: only projects that contain the profile FrameTiming
            # sampling loop (the legacy ColmapBatchCapture profile machinery)
            # receive this robustness patch.  Minimal synthetic projects used
            # by the CPU-side patch tests omit the loop entirely.
            pass
        else:
            updated = updated.replace(old_timing, new_timing, 1)
    return updated


def _patch_profile_player_build(source: str) -> str:
    marker = "MeshSplatBench Linux standalone profile player patch"
    if marker in source:
        return source
    if "target = BuildTarget.StandaloneOSX" not in source:
        raise ValueError("could not locate hard-coded standalone build target")
    updated = source
    updated = updated.replace(
        '''            BuildPlayerOptions options = new BuildPlayerOptions {
                scenes = new[] { "Assets/Scenes/MeshSplatBenchGarden.unity" },
                locationPathName = output,
                target = BuildTarget.StandaloneOSX,
                options = BuildOptions.Development,
            };''',
        '''            BuildTarget target = ParseTarget(GetArgument("-profile-player-target"));
            BuildPlayerOptions options = new BuildPlayerOptions {
                scenes = new[] { "Assets/Scenes/MeshSplatBenchGarden.unity" },
                locationPathName = output,
                target = target,
                // MeshSplatBench Linux standalone profile player patch.
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
                    throw new ArgumentException("Unsupported MeshSplatBench profile player target: " + value);
            }
        }

'''
    return updated.replace(anchor, helper + anchor, 1)


def patch_unity_project(project: Path) -> list[Path]:
    assets = project.expanduser().resolve() / "Assets" / "MeshSplatBench"
    if not assets.is_dir():
        raise FileNotFoundError(f"legacy MeshSplatBench Unity assets not found: {assets}")

    changed: list[Path] = []
    shader_root = assets / "Shaders"
    for path in sorted(shader_root.glob("*.shader")):
        original = path.read_text(encoding="utf-8")
        if _write_if_changed(path, original, _patch_shader(original)):
            changed.append(path)
    indexed_mesh_shader = shader_root / "MeshSplatIndexedMesh.shader"
    original = indexed_mesh_shader.read_text(encoding="utf-8") if indexed_mesh_shader.is_file() else ""
    if _write_if_changed(indexed_mesh_shader, original, _indexed_mesh_splat_shader()):
        changed.append(indexed_mesh_shader)
    # Standalone players strip shaders that are not referenced by scenes,
    # materials, Resources, or Always Included Shaders.  Keep a Resources copy
    # and load it explicitly for the true indexed MeshRenderer topology run.
    resources_root = assets / "Resources"
    indexed_mesh_resource_shader = resources_root / "MeshSplatIndexedMesh.shader"
    original = (
        indexed_mesh_resource_shader.read_text(encoding="utf-8")
        if indexed_mesh_resource_shader.is_file()
        else ""
    )
    if _write_if_changed(indexed_mesh_resource_shader, original, _indexed_mesh_splat_shader()):
        changed.append(indexed_mesh_resource_shader)

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
        "StandardMeshTriAssetRenderer.cs": "0",
    }
    for name, primitive_count in renderer_counts.items():
        path = script_root / name
        if not path.is_file():
            continue
        original = path.read_text(encoding="utf-8")
        updated = _patch_renderer(original)
        if name == "StandardMeshTriAssetRenderer.cs":
            updated = _patch_standard_mesh_indexed_method_aware(updated)
        else:
            updated = _patch_renderer_command_buffer(updated, primitive_count, path.stem)
        if name == "MethodSpecificSplatRenderer.cs":
            updated = _patch_method_specific_vulkan_bindings(updated)
            updated = _patch_method_specific_shader_level_soup(updated)
            updated = _patch_method_specific_camera_state(updated)
        elif name == "TriangleSplattingTriAssetRenderer.cs":
            updated = _patch_generic_prepare_camera(updated, path.stem)
            updated = _patch_triangle_splatting_camera_sort(updated)
        elif name != "StandardMeshTriAssetRenderer.cs":
            updated = _patch_generic_prepare_camera(updated, path.stem)
        if _write_if_changed(path, original, updated):
            changed.append(path)

    build = assets / "Editor" / "MeshSplatBenchProfilePlayerBuild.cs"
    if build.is_file():
        original = build.read_text(encoding="utf-8")
        if _write_if_changed(build, original, _patch_profile_player_build(original)):
            changed.append(build)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch a legacy MeshSplatBenchUnity Metal project for Vulkan capture."
    )
    parser.add_argument("--unity-project", type=Path, required=True)
    args = parser.parse_args()
    changed = patch_unity_project(args.unity_project)
    if changed:
        print("Patched legacy MeshSplatBench Unity project:")
        for path in changed:
            print(f"  {path}")
    else:
        print("Legacy MeshSplatBench Unity project is already patched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
