using System;
using System.Collections;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Rendering;

namespace MeshSplatBench.UnityNative
{
    /// <summary>
    /// Unity Metal portability renderer for mesh-splatting and 2DTS.  It binds
    /// their exported learned buffers and evaluates their declared appearance
    /// semantics at draw time; it is not a claim of CUDA-rasterizer equivalence.
    /// </summary>
    public sealed class MethodSpecificSplatRenderer : TriAssetRenderer
    {
        public Camera TargetCamera;
        public Shader SplatShader;
        ComputeBuffer positions, indices, opacity, sigma, dc, rest;
        Material material;
        CommandBuffer msBenchDrawCommands;
        Camera msBenchCommandCamera;
        int primitiveCount, mode, activeShDegree, meshAblation;
        float gamma = 1f, opacityFloor;
        bool ready, rawCode, deindexedSoup;
        string status = "Waiting to load method-specific renderer";
        ComputeBuffer triangleOrderBuffer;
        bool usesSortedTriangleOrder;
        Vector3[] triangleCentroids;
        int[] triangleOrder;
        int[] sourceTriangleIndices;
        int[] sortedTriangleIndices;
        float[] triangleDepths;
        Vector3 lastSortCameraPosition = new Vector3(float.NaN, float.NaN, float.NaN);
        Vector3 lastSortCameraForward = new Vector3(float.NaN, float.NaN, float.NaN);
        public override bool IsReady => ready;
        public override string Status => status;

        void Awake() { AssetDirectory = TriAssetRuntimeOptions.Get("-triasset", AssetDirectory); }

        IEnumerator Start()
        {
            string method = TriAssetRuntimeOptions.Get("-method", "");
            if (method == "mesh-splatting") mode = 1;
            else if (method == "2dts") mode = 2;
            else { Fail("MethodSpecificSplatRenderer supports mesh-splatting and 2dts only."); yield break; }
            string manifest = Path.Combine(AssetDirectory ?? "", "manifest.json");
            if (!File.Exists(manifest)) { Fail("manifest.json is missing."); yield break; }
            string json = File.ReadAllText(manifest);
            primitiveCount = ReadInt(json, "primitive_count", 0);
            activeShDegree = ReadInt(json, "active_sh_degree", 0);
            gamma = ReadFloat(json, "gamma", 1f);
            opacityFloor = Mathf.Clamp01(ReadFloat(json, "opacity_floor", 0f));
            if (primitiveCount <= 0) { Fail("Invalid primitive_count."); yield break; }
            string b = Path.Combine(AssetDirectory, "buffers");
            deindexedSoup = TriAssetRuntimeOptions.Get("-topology", "indexed").ToLowerInvariant() == "soup"
                || TriAssetRuntimeOptions.Has("-deindexed-soup");
            status = "Loading learned render buffers";
            yield return null;
            int[] sourceIndices = ReadInt(Path.Combine(b, "indices.bin"));
            if (sourceIndices.Length != primitiveCount * 3) { Fail("indices.bin does not match primitive_count."); yield break; }
            float[] sourcePositions = ReadFloat(Path.Combine(b, "positions.bin"));
            if (sourcePositions.Length % 3 != 0) { Fail("positions.bin is not float3 data."); yield break; }
            InitializeTriangleOrder(sourcePositions, sourceIndices);
            if (deindexedSoup && mode == 1)
            {
                // MeshSplatting shader-level soup topology: keep the original
                // indexed buffers and let each procedural triangle corner fetch
                // source vertex attributes through _Indices in the shader.  This
                // disables hardware vertex sharing in the draw path without
                // materializing multi-GiB duplicate SH buffers on the CPU/GPU.
                Debug.Log("[MeshSplatBench] MeshSplatting shader-level soup topology: retaining indexed buffers; procedural corners fetch via _Indices.");
            }
            positions = Upload(sourcePositions);
            indices = Upload(sourceIndices);
            triangleOrderBuffer = Upload(sourceIndices);
            opacity = Upload(ReadFloat(Path.Combine(b, mode == 1 ? "vertex_weight_logits.bin" : "opacity_logits.bin")));
            dc = Upload(ReadFloat(Path.Combine(b, "sh_dc.bin")));
            rest = Upload(ReadFloat(Path.Combine(b, "sh_rest.bin")));
            if (mode == 1) sigma = Upload(ReadFloat(Path.Combine(b, "sigma_logits.bin")));
            string ablation = "full";
            if (mode == 1) ablation = TriAssetRuntimeOptions.Get("-mesh-ablation", "full").ToLowerInvariant();
            // Keep the renderer contract static: these two passes are ordinary
            // hardware-rasterized depth passes, not a runtime branch inside the
            // transparent splat shader.  This makes the alpha-compositing
            // intervention independently reproducible from the material state.
            // The released Mesh-Splatting checkpoint is in its terminal solid
            // regime (sigma -> 1e-4 and opacity floor -> 0.9999).  In that
            // regime a depth-writing hardware pass is the stable limiting form
            // of the native front-to-back compositor.  It still evaluates the
            // learned degree-3 SH field; only sub-0.1% residual transmission is
            // intentionally collapsed to hardware visibility.  Do not route
            // 2DTS here: its learned coverage remains genuinely transparent.
            string shaderName = mode == 1 && ablation == "full"
                ? "MeshSplatBench/MeshSplatTerminalSolid"
                : mode == 1 && ablation == "alpha-test-depth"
                    ? "MeshSplatBench/MeshSplatAlphaTestDepth"
                    : mode == 1 && ablation == "opaque-depth"
                        ? "MeshSplatBench/MeshSplatOpaqueDepth"
                        : "MeshSplatBench/MethodSpecificSplat";
            usesSortedTriangleOrder = shaderName == "MeshSplatBench/MethodSpecificSplat";
            SplatShader = SplatShader != null ? SplatShader : Shader.Find(shaderName);
            if (SplatShader == null) { Fail("MeshSplatBench/MethodSpecificSplat shader was not found."); yield break; }
            material = new Material(SplatShader) { hideFlags = HideFlags.HideAndDontSave };
            material.SetBuffer("_Positions", positions); material.SetBuffer("_Indices", usesSortedTriangleOrder ? triangleOrderBuffer : indices);
            material.SetBuffer("_Opacity", opacity); material.SetBuffer("_ShDc", dc); material.SetBuffer("_ShRest", rest);
            material.SetBuffer("_Sigma", sigma != null ? sigma : opacity);
            material.SetInt("_Mode", mode); material.SetInt("_PrimitiveCount", primitiveCount);
            material.SetInt("_ShDegree", activeShDegree); material.SetFloat("_Gamma", gamma);
            material.SetFloat("_OpacityFloor", opacityFloor);
            if (mode == 1)
            {
                meshAblation = ablation == "no-sh" ? 1 : ablation == "no-sigma" ? 2 : 0;
                material.SetInt("_MeshAblation", meshAblation);
                material.SetInt("_UseSh", (meshAblation == 1 || meshAblation == 4) ? 0 : 1);
            }
            material.SetInt("_RawCode", 1);
            RefreshTriangleOrder(TargetCamera, force: true);
            ready = true; status = $"Method-specific {method} renderer ready ({primitiveCount:N0} primitives; {(deindexedSoup ? "shader-level triangle soup" : "indexed mesh")})";
            if (mode == 1) Debug.Log($"[MeshSplatBench] MeshSplatting layout={(deindexedSoup ? "soup(shader-indexed)" : "indexed")}, ablation={ablation}, code={meshAblation}, opacityFloor={opacityFloor:F6}.");
            Debug.Log("[MeshSplatBench] " + status);
        }

        void InitializeTriangleOrder(float[] positionsData, int[] sourceIndices)
        {
            int count = Mathf.Min(primitiveCount, sourceIndices.Length / 3);
            primitiveCount = count;
            triangleCentroids = new Vector3[count];
            triangleOrder = new int[count];
            triangleDepths = new float[count];
            sourceTriangleIndices = sourceIndices;
            sortedTriangleIndices = new int[count * 3];
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
                triangleOrder[t] = t * 3;
            }
            Array.Sort(triangleDepths, triangleOrder);
            int write = 0;
            for (int t = 0; t < triangleOrder.Length; ++t)
            {
                int source = triangleOrder[t];
                sortedTriangleIndices[write++] = sourceTriangleIndices[source];
                sortedTriangleIndices[write++] = sourceTriangleIndices[source + 1];
                sortedTriangleIndices[write++] = sourceTriangleIndices[source + 2];
            }
            triangleOrderBuffer.SetData(sortedTriangleIndices);
            lastSortCameraPosition = cam;
            lastSortCameraForward = forward;
        }

        // MeshSplatBench explicit camera command-buffer patch. OnRenderObject is
        // not a reliable callback for Camera.Render() on Linux Editor batch mode.
        bool InstallMsBenchCameraDraw()
        {
            if (TargetCamera == null) TargetCamera = Camera.main;
            if (TargetCamera == null || material == null)
            {
                Fail("Cannot install procedural draw without a target camera and material.");
                return false;
            }
            if (material.shader == null || !material.shader.isSupported)
            {
                Fail("Shader is unsupported on " + SystemInfo.graphicsDeviceType + ": "
                    + (material.shader == null ? "<null>" : material.shader.name));
                return false;
            }
            msBenchDrawCommands = new CommandBuffer { name = "MeshSplatBench/MethodSpecificSplatRenderer procedural draw" };
            msBenchDrawCommands.DrawProcedural(
                Matrix4x4.identity, material, 0, MeshTopology.Triangles,
                primitiveCount * 3, 1);
            msBenchCommandCamera = TargetCamera;
            msBenchCommandCamera.AddCommandBuffer(CameraEvent.AfterForwardAlpha, msBenchDrawCommands);
            Debug.Log("[MeshSplatBench] Installed explicit camera draw for MethodSpecificSplatRenderer on "
                + SystemInfo.graphicsDeviceType + ".");
            return true;
        }

        void RemoveMsBenchCameraDraw()
        {
            if (msBenchCommandCamera != null && msBenchDrawCommands != null)
                msBenchCommandCamera.RemoveCommandBuffer(CameraEvent.AfterForwardAlpha, msBenchDrawCommands);
            if (msBenchDrawCommands != null) msBenchDrawCommands.Release();
            msBenchDrawCommands = null;
            msBenchCommandCamera = null;
        }

        public override void PrepareCamera(Camera camera)
        {
            if (camera == null || material == null) return;
            if (TargetCamera != camera)
            {
                RemoveMsBenchCameraDraw();
                TargetCamera = camera;
            }
            if (msBenchDrawCommands == null)
            {
                if (!InstallMsBenchCameraDraw()) { enabled = false; return; }
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
        public override void PrepareCamera(Camera camera)
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
        void OnRenderObject()
        {
            if (msBenchDrawCommands != null) return;
            if (Application.isBatchMode) return;
            if (!ready || Camera.current != TargetCamera || material == null) return;
            RefreshTriangleOrder(TargetCamera);
            material.SetVector("_CameraWorldPos", TargetCamera.transform.position);
            material.SetInt("_RawCode", rawCode ? 1 : 0);
            material.SetFloat("_OpacityFloor", opacityFloor);
            if (mode == 1)
            {
                material.SetInt("_MeshAblation", meshAblation);
                material.SetInt("_UseSh", (meshAblation == 1 || meshAblation == 4) ? 0 : 1);
            }
            if (!material.SetPass(0)) { Debug.LogError("[MeshSplatBench] Shader pass is unsupported on " + SystemInfo.graphicsDeviceType + ": " + material.shader.name); enabled = false; return; } Graphics.DrawProceduralNow(MeshTopology.Triangles, primitiveCount * 3, 1);
        }
        public override void SetOutputRawCodeValues(bool enabled) { rawCode = enabled; }

        static int ReadInt(string json, string key, int fallback) { Match m=Regex.Match(json,"\\\""+key+"\\\"\\s*:\\s*(\\d+)"); return m.Success && Int32.TryParse(m.Groups[1].Value,out int v) ? v : fallback; }
        static float ReadFloat(string json, string key, float fallback) { Match m=Regex.Match(json,"\\\""+key+"\\\"\\s*:\\s*([-+.0-9eE]+)"); return m.Success && Single.TryParse(m.Groups[1].Value,System.Globalization.NumberStyles.Float,System.Globalization.CultureInfo.InvariantCulture,out float v) ? v : fallback; }
        static float[] ReadFloat(string p) { byte[] raw=File.ReadAllBytes(p); if ((raw.Length&3)!=0) throw new InvalidDataException(p); float[] v=new float[raw.Length/4]; Buffer.BlockCopy(raw,0,v,0,raw.Length); return v; }
        static int[] ReadInt(string p) { byte[] raw=File.ReadAllBytes(p); if ((raw.Length&3)!=0) throw new InvalidDataException(p); int[] v=new int[raw.Length/4]; Buffer.BlockCopy(raw,0,v,0,raw.Length); return v; }
        static float[] Deindex(float[] source, int[] sourceIndices, int stride, string name)
        {
            if (source.Length % stride != 0) throw new InvalidDataException(name + " has invalid stride.");
            int vertexCount = source.Length / stride;
            float[] output = new float[sourceIndices.Length * stride];
            for (int corner = 0; corner < sourceIndices.Length; ++corner)
            {
                int sourceVertex = sourceIndices[corner];
                if ((uint)sourceVertex >= (uint)vertexCount) throw new InvalidDataException("indices.bin references an invalid " + name + " vertex.");
                // Array.Copy avoids a Unity/Mono Buffer.BlockCopy failure for
                // large managed float arrays during the soup conversion.
                Array.Copy(source, sourceVertex * stride, output, corner * stride, stride);
            }
            return output;
        }
        static ComputeBuffer Upload(Array values) { ComputeBuffer b=new ComputeBuffer(values.Length,4,ComputeBufferType.Raw); b.SetData(values); return b; }
        void Fail(string msg) { status="ERROR: "+msg; Debug.LogError("[MeshSplatBench] "+msg); }
        void OnDestroy() { RemoveMeshSplatBenchCameraDraw(); ready=false; positions?.Release(); indices?.Release(); triangleOrderBuffer?.Release(); opacity?.Release(); sigma?.Release(); dc?.Release(); rest?.Release(); if(material!=null) Destroy(material); }

        // original OnDestroy replaced by MeshSplatBench sort patch
        void OnDestroy_disabled { RemoveMsBenchCameraDraw(); ready=false; positions?.Release(); indices?.Release(); triangleOrderBuffer?.Release(); opacity?.Release(); sigma?.Release(); dc?.Release(); rest?.Release(); if(material!=null) Destroy(material); }
    }
}
