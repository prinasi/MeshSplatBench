using System;
using System.Collections;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Rendering;

namespace MeshSplatBench.UnityNative
{
    /// <summary>
    /// Faithful Unity/Metal evaluation path for exported DiffSoup assets.  The
    /// shader evaluates the accumulated level-Rmax alpha/features, the SH2
    /// view-direction encoding, and the exported ColorMLP.  Visibility follows
    /// DiffSoup's deterministic 0.5 alpha test plus hardware depth testing.
    /// </summary>
    public sealed class DiffSoupTriAssetRenderer : TriAssetRenderer
    {
        public Camera TargetCamera;
        public Shader SplatShader;
        ComputeBuffer positions, indices, features, alpha, w0, b0, w2, b2, w4, b4;
        Material material;
        CommandBuffer msBenchDrawCommands;
        Camera msBenchCommandCamera;
        bool ready;
        string status = "Waiting to load DiffSoup renderer";
        int primitiveCount, featureSamples, featureDim, rmax;

        public override bool IsReady => ready;
        public override string Status => status;

        void Awake() { AssetDirectory = TriAssetRuntimeOptions.Get("-triasset", AssetDirectory); }

        IEnumerator Start()
        {
            if (TargetCamera == null) TargetCamera = Camera.main;
            string manifestPath = Path.Combine(AssetDirectory ?? "", "manifest.json");
            if (TargetCamera == null || !File.Exists(manifestPath)) { Fail("Target camera or manifest.json is missing."); yield break; }
            string manifest = File.ReadAllText(manifestPath);
            primitiveCount = ReadInt(manifest, "primitive_count", 0);
            featureDim = ReadInt(manifest, "feature_dim", 0);
            rmax = ReadInt(manifest, "rmax", 0);
            featureSamples = ReadShapeDimension(manifest, "features", 1);
            if (primitiveCount <= 0 || featureDim != 7 || featureSamples <= 0 || rmax <= 0)
            { Fail("Unsupported DiffSoup feature layout; expected valid 7D Rmax features."); yield break; }

            string buffers = Path.Combine(AssetDirectory, "buffers");
            status = "Loading DiffSoup accumulated features and ColorMLP";
            yield return null;
            positions = Upload(ReadFloat(Path.Combine(buffers, "positions.bin")));
            indices = Upload(ReadIntArray(Path.Combine(buffers, "indices.bin")));
            features = Upload(ReadFloat(Path.Combine(buffers, "features.bin")));
            alpha = Upload(ReadFloat(Path.Combine(buffers, "alpha.bin")));
            w0 = Upload(ReadFloat(Path.Combine(buffers, "color_mlp__mlp_0_weight.bin")));
            b0 = Upload(ReadFloat(Path.Combine(buffers, "color_mlp__mlp_0_bias.bin")));
            w2 = Upload(ReadFloat(Path.Combine(buffers, "color_mlp__mlp_2_weight.bin")));
            b2 = Upload(ReadFloat(Path.Combine(buffers, "color_mlp__mlp_2_bias.bin")));
            w4 = Upload(ReadFloat(Path.Combine(buffers, "color_mlp__mlp_4_weight.bin")));
            b4 = Upload(ReadFloat(Path.Combine(buffers, "color_mlp__mlp_4_bias.bin")));
            SplatShader = SplatShader != null ? SplatShader : Shader.Find("MeshSplatBench/DiffSoupMetal");
            if (SplatShader == null) { Fail("MeshSplatBench/DiffSoupMetal shader was not found."); yield break; }
            material = new Material(SplatShader) { hideFlags = HideFlags.HideAndDontSave };
            material.SetBuffer("_Positions", positions); material.SetBuffer("_Indices", indices);
            material.SetBuffer("_Features", features); material.SetBuffer("_Alpha", alpha);
            material.SetBuffer("_W0", w0); material.SetBuffer("_B0", b0); material.SetBuffer("_W2", w2); material.SetBuffer("_B2", b2); material.SetBuffer("_W4", w4); material.SetBuffer("_B4", b4);
            material.SetInt("_PrimitiveCount", primitiveCount); material.SetInt("_FeatureSamples", featureSamples); material.SetInt("_FeatureDim", featureDim); material.SetInt("_Rmax", rmax);
            ready = true;
            status = $"DiffSoup renderer ready ({primitiveCount:N0} triangles, R{rmax}, {featureDim}D)";
            Debug.Log("[MeshSplatBench] " + status);
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
            msBenchDrawCommands = new CommandBuffer { name = "MeshSplatBench/DiffSoupTriAssetRenderer procedural draw" };
            msBenchDrawCommands.DrawProcedural(
                Matrix4x4.identity, material, 0, MeshTopology.Triangles,
                primitiveCount * 3, 1);
            msBenchCommandCamera = TargetCamera;
            msBenchCommandCamera.AddCommandBuffer(CameraEvent.AfterForwardAlpha, msBenchDrawCommands);
            Debug.Log("[MeshSplatBench] Installed explicit camera draw for DiffSoupTriAssetRenderer on "
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
            if (msBenchDrawCommands == null)
            {
                TargetCamera = camera;
                if (!InstallMsBenchCameraDraw()) enabled = false;
            }
        }

        void OnRenderObject()
        {
            if (msBenchDrawCommands != null) return;
            if (Application.isBatchMode) return;
            if (!ready || material == null || Camera.current != TargetCamera) return;
            if (!material.SetPass(0)) { Debug.LogError("[MeshSplatBench] Shader pass is unsupported on " + SystemInfo.graphicsDeviceType + ": " + material.shader.name); enabled = false; return; }
            Graphics.DrawProceduralNow(MeshTopology.Triangles, primitiveCount * 3, 1);
        }

        static int ReadInt(string json, string key, int fallback) { Match m = Regex.Match(json, "\\\"" + key + "\\\"\\s*:\\s*(\\d+)"); return m.Success && Int32.TryParse(m.Groups[1].Value, out int v) ? v : fallback; }
        static int ReadShapeDimension(string json, string name, int dimension)
        {
            Match m = Regex.Match(json, "\\\"name\\\"\\s*:\\s*\\\"" + name + "\\\"[\\s\\S]*?\\\"shape\\\"\\s*:\\s*\\[\\s*\\d+\\s*,\\s*(\\d+)");
            return m.Success && Int32.TryParse(m.Groups[1].Value, out int v) ? v : 0;
        }
        static float[] ReadFloat(string p) { byte[] raw = File.ReadAllBytes(p); if ((raw.Length & 3) != 0) throw new InvalidDataException(p); float[] data = new float[raw.Length / 4]; Buffer.BlockCopy(raw, 0, data, 0, raw.Length); return data; }
        static int[] ReadIntArray(string p) { byte[] raw = File.ReadAllBytes(p); if ((raw.Length & 3) != 0) throw new InvalidDataException(p); int[] data = new int[raw.Length / 4]; Buffer.BlockCopy(raw, 0, data, 0, raw.Length); return data; }
        static ComputeBuffer Upload(Array data) { ComputeBuffer b = new ComputeBuffer(data.Length, 4, ComputeBufferType.Raw); b.SetData(data); return b; }
        void Fail(string message) { status = "ERROR: " + message; Debug.LogError("[MeshSplatBench] " + message); }
        void OnDestroy() { RemoveMsBenchCameraDraw(); ready = false; positions?.Release(); indices?.Release(); features?.Release(); alpha?.Release(); w0?.Release(); b0?.Release(); w2?.Release(); b2?.Release(); w4?.Release(); b4?.Release(); if (material != null) Destroy(material); }
    }
}
