using System;
using System.Collections;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Rendering;

namespace MeshSplatBench.UnityNative
{
    /// <summary>
    /// Metal/Unity baseline for the MeshSplatBench triangle-splatting export contract.
    /// It consumes every exported buffer, evaluates degree-3 SH, applies the
    /// exported opacity/sigma activations, sorts triangle centroids front-to-back,
    /// and uses premultiplied front-to-back blending. The CUDA renderer still has
    /// a different tile-binning/rasterization path, so this is intentionally
    /// labelled a baseline rather than a native-equivalence implementation.
    /// </summary>
    public sealed class TriangleSplattingTriAssetRenderer : TriAssetRenderer
    {
        public Camera TargetCamera;
        public Shader SplatShader;
        public int PrimitiveCount = 4517295;
        public bool SortFrontToBack = true;
        [Tooltip("Write the native renderer's raw RGB code values for numerical evaluation. Keep false for the Unity display path.")]
        public bool OutputRawCodeValues;

        public override bool IsReady => ready;
        public override string Status => status;

        ComputeBuffer positionsBuffer;
        ComputeBuffer indicesBuffer;
        ComputeBuffer opacityBuffer;
        ComputeBuffer sigmaBuffer;
        ComputeBuffer shDcBuffer;
        ComputeBuffer shRestBuffer;
        ComputeBuffer orderBuffer;
        Material material;
        CommandBuffer msBenchDrawCommands;
        Camera msBenchCommandCamera;
        bool ready;
        string status = "Waiting to load";
        float fpsTimer;
        int fpsFrames;
        float displayedFps;
        Vector3[] triangleCentroids;
        int[] triangleOrder;
        float[] triangleDepths;
        Vector3 lastSortCameraPosition = new Vector3(float.NaN, float.NaN, float.NaN);
        Vector3 lastSortCameraForward = new Vector3(float.NaN, float.NaN, float.NaN);

        void Awake()
        {
            AssetDirectory = TriAssetRuntimeOptions.Get("-triasset", AssetDirectory);
        }

        IEnumerator Start()
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;

            if (TargetCamera == null) TargetCamera = Camera.main;
            if (TargetCamera == null)
            {
                Fail("No target camera is assigned.");
                yield break;
            }

            if (string.IsNullOrWhiteSpace(AssetDirectory) || !Directory.Exists(AssetDirectory))
            {
                Fail("Asset directory does not exist: " + AssetDirectory);
                yield break;
            }

            string manifest = Path.Combine(AssetDirectory, "manifest.json");
            if (!File.Exists(manifest))
            {
                Fail("manifest.json is missing: " + manifest);
                yield break;
            }
            PrimitiveCount = ReadPrimitiveCount(manifest, PrimitiveCount);

            SplatShader = SplatShader != null ? SplatShader : Shader.Find("MeshSplatBench/TriangleSplattingMetalBaseline");
            if (SplatShader == null)
            {
                Fail("MeshSplatBench/TriangleSplattingMetalBaseline shader was not found.");
                yield break;
            }
            material = new Material(SplatShader) { hideFlags = HideFlags.HideAndDontSave };

            string buffers = Path.Combine(AssetDirectory, "buffers");
            status = "Loading positions (161 MiB)";
            yield return null;
            float[] positions = ReadFloatArray(Path.Combine(buffers, "positions.bin"));
            positionsBuffer = UploadRaw(positions);

            status = "Loading indices (52 MiB)";
            yield return null;
            int[] indices = ReadIntArray(Path.Combine(buffers, "indices.bin"));
            indicesBuffer = UploadRaw(indices);

            status = SortFrontToBack ? "Sorting 4,517,295 primitives front-to-back" : "Creating primitive order";
            yield return null;
            InitializeTriangleOrder(positions, indices);
            RefreshTriangleOrder(TargetCamera, force: true);
            positions = null;
            indices = null;
            GC.Collect();

            status = "Loading opacity logits (18 MiB)";
            yield return null;
            opacityBuffer = UploadRaw(ReadFloatArray(Path.Combine(buffers, "opacity_logits.bin")));
            GC.Collect();

            status = "Loading sigma logits (18 MiB)";
            yield return null;
            sigmaBuffer = UploadRaw(ReadFloatArray(Path.Combine(buffers, "sigma_logits.bin")));
            GC.Collect();

            status = "Loading SH DC (53 MiB)";
            yield return null;
            shDcBuffer = UploadRaw(ReadFloatArray(Path.Combine(buffers, "sh_dc.bin")));
            GC.Collect();

            status = "Loading SH degree 1-3 (785 MiB)";
            yield return null;
            shRestBuffer = UploadRaw(ReadFloatArray(Path.Combine(buffers, "sh_rest.bin")));
            GC.Collect();

            material.SetBuffer("_Positions", positionsBuffer);
            material.SetBuffer("_Indices", indicesBuffer);
            material.SetBuffer("_OpacityLogits", opacityBuffer);
            material.SetBuffer("_SigmaLogits", sigmaBuffer);
            material.SetBuffer("_ShDc", shDcBuffer);
            material.SetBuffer("_ShRest", shRestBuffer);
            material.SetBuffer("_TriangleOrder", orderBuffer);
            material.SetInt("_PrimitiveCount", PrimitiveCount);
            material.SetInt("_OutputRawCodeValues", OutputRawCodeValues ? 1 : 0);

            ready = true;
            status = "Rendering all 4,517,295 primitives (Unity 6 Metal baseline)";
            Debug.Log("[MeshSplatBench] " + status);
        }

        static int ReadPrimitiveCount(string manifestPath, int fallback)
        {
            Match match = Regex.Match(File.ReadAllText(manifestPath), "\\\"primitive_count\\\"\\s*:\\s*(\\d+)");
            return match.Success && int.TryParse(match.Groups[1].Value, out int count) && count > 0 ? count : fallback;
        }

        void InitializeTriangleOrder(float[] positions, int[] indices)
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
            msBenchDrawCommands = new CommandBuffer { name = "MeshSplatBench/TriangleSplattingTriAssetRenderer procedural draw" };
            msBenchDrawCommands.DrawProcedural(
                Matrix4x4.identity, material, 0, MeshTopology.Triangles,
                PrimitiveCount * 3, 1);
            msBenchCommandCamera = TargetCamera;
            msBenchCommandCamera.AddCommandBuffer(CameraEvent.AfterForwardAlpha, msBenchDrawCommands);
            Debug.Log("[MeshSplatBench] Installed explicit camera draw for TriangleSplattingTriAssetRenderer on "
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
        }

        void OnRenderObject()
        {
            if (msBenchDrawCommands != null) return;
            if (Application.isBatchMode) return;
            if (!ready || Camera.current != TargetCamera || material == null) return;
            RefreshTriangleOrder(TargetCamera);
            if (!material.SetPass(0)) { Debug.LogError("[MeshSplatBench] Shader pass is unsupported on " + SystemInfo.graphicsDeviceType + ": " + material.shader.name); enabled = false; return; }
            Graphics.DrawProceduralNow(MeshTopology.Triangles, PrimitiveCount * 3, 1);
        }

        /// <summary>Switches between a display-linear output and the native raw RGB evaluation domain.</summary>
        public override void SetOutputRawCodeValues(bool enabled)
        {
            OutputRawCodeValues = enabled;
            if (material != null) material.SetInt("_OutputRawCodeValues", enabled ? 1 : 0);
        }

        void Update()
        {
            if (!ready) return;
            fpsTimer += Time.unscaledDeltaTime;
            fpsFrames++;
            if (fpsTimer >= 1f)
            {
                displayedFps = fpsFrames / fpsTimer;
                fpsTimer = 0f;
                fpsFrames = 0;
                Debug.Log($"[MeshSplatBench] Metal baseline FPS: {displayedFps:F1}");
            }
        }

        static float[] ReadFloatArray(string path)
        {
            long byteLength = new FileInfo(path).Length;
            if ((byteLength & 3) != 0 || byteLength / 4 > int.MaxValue)
                throw new InvalidDataException("Invalid float32 buffer: " + path);
            float[] values = new float[(int)(byteLength / 4)];
            ReadIntoArray(path, values, (int)byteLength);
            return values;
        }

        static int[] ReadIntArray(string path)
        {
            long byteLength = new FileInfo(path).Length;
            if ((byteLength & 3) != 0 || byteLength / 4 > int.MaxValue)
                throw new InvalidDataException("Invalid int32 buffer: " + path);
            int[] values = new int[(int)(byteLength / 4)];
            ReadIntoArray(path, values, (int)byteLength);
            return values;
        }

        static void ReadIntoArray(string path, Array destination, int byteLength)
        {
            const int blockSize = 4 * 1024 * 1024;
            byte[] block = new byte[blockSize];
            using FileStream stream = File.OpenRead(path);
            int destinationOffset = 0;
            while (destinationOffset < byteLength)
            {
                int requested = Math.Min(block.Length, byteLength - destinationOffset);
                int read = stream.Read(block, 0, requested);
                if (read <= 0) throw new EndOfStreamException(path);
                Buffer.BlockCopy(block, 0, destination, destinationOffset, read);
                destinationOffset += read;
            }
        }

        static ComputeBuffer UploadRaw(Array data)
        {
            ComputeBuffer buffer = new ComputeBuffer(data.Length, 4, ComputeBufferType.Raw);
            buffer.SetData(data);
            return buffer;
        }

        void Fail(string message)
        {
            status = "ERROR: " + message;
            Debug.LogError("[MeshSplatBench] " + message);
        }

        void OnDestroy()
        {
            RemoveMsBenchCameraDraw();
            ready = false;
            positionsBuffer?.Release();
            indicesBuffer?.Release();
            opacityBuffer?.Release();
            sigmaBuffer?.Release();
            shDcBuffer?.Release();
            shRestBuffer?.Release();
            orderBuffer?.Release();
            if (material != null) Destroy(material);
        }
    }
}
