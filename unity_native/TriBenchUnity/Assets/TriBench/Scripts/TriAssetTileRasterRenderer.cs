// TriBench Tile-based Compute Shader Renderer for feature-preserving .triasset packages.
// Directly reproduces native CUDA tile-based binning, radix sorting, and front-to-back
// software rasterization inside Unity Compute Shaders, eliminating transparency sorting artifacts.
using System;
using UnityEngine;
using UnityEngine.Rendering;

public abstract class TriAssetTileRasterRenderer : TriAssetRenderer
{
    [Tooltip("Optional; defaults to Camera.main.")] public Camera targetCamera;
    [Tooltip("Reject rather than silently render an incomplete package.")] public bool strictValidation = true;

    protected TriAssetLoader asset;
    private ComputeShader rasterizerCS;
    private RenderTexture outputTarget;

    // GPU Buffers
    private ComputeBuffer tileCounts;
    private ComputeBuffer triTileRectMin;
    private ComputeBuffer triTileRectMax;
    private ComputeBuffer triDepths;
    private ComputeBuffer preprocessedGeom;

    private ComputeBuffer scanOutput;
    private ComputeBuffer blockSums;
    private ComputeBuffer scannedBlockSums;
    private ComputeBuffer scanOffsets;

    private ComputeBuffer keysTileA;
    private ComputeBuffer keysDepthA;
    private ComputeBuffer valuesA;
    private ComputeBuffer keysTileB;
    private ComputeBuffer keysDepthB;
    private ComputeBuffer valuesB;

    private ComputeBuffer blockHistograms;
    private ComputeBuffer globalBucketOffsets;
    private ComputeBuffer tileRanges;
    private ComputeBuffer dummySigma;
    private ComputeBuffer dummyFeatures;
    private ComputeBuffer dummyShDc;
    private ComputeBuffer dummyShRest;
    private ComputeBuffer colorMlpParams;
    private ComputeBuffer dummyColorMlpParams;
    private ComputeBuffer totalKeys;      // [0] = validated key count, [1] = required count on overflow

    // The whole compute pipeline runs inside a per-camera CommandBuffer bound
    // at CameraEvent.AfterForwardAlpha.  Explicit Camera.Render() calls on a
    // disabled capture camera are the norm in Linux Editor batch mode, and
    // camera command buffers execute reliably for those renders while precull
    // submissions do not (see tools/patch_legacy_unity_vulkan.py).
    private CommandBuffer pipelineCommands;
    private Camera pipelineCamera;
    private bool outputTargetDirty = true;
    private int pendingOverflowChecks;

    private int primitiveCount;
    private int currentKeyCapacity;
    private bool configured;
    private int lastScreenWidth;
    private int lastScreenHeight;

    protected abstract int RenderMode { get; }
    protected virtual string OpacityBuffer { get { return "opacity_logits"; } }
    protected virtual bool UsesSigma { get { return true; } }
    protected virtual bool UsesSh { get { return true; } }

    protected virtual void Validate(TriAssetLoader source)
    {
        source.Require("positions");
        source.Require("indices");
        if (UsesSh) { source.Require("sh_dc"); source.Require("sh_rest"); }
        if (!String.IsNullOrEmpty(OpacityBuffer)) source.Require(OpacityBuffer);
        if (UsesSigma) source.Require("sigma_logits");
    }

    private void OnEnable()
    {
        Camera.onPreCull += OnCameraPreCull;
        RenderPipelineManager.beginCameraRendering += OnBeginCameraRendering;
    }

    private void OnDisable()
    {
        Camera.onPreCull -= OnCameraPreCull;
        RenderPipelineManager.beginCameraRendering -= OnBeginCameraRendering;
    }

    private void OnDestroy()
    {
        ReleaseRuntime();
    }

    public override void Configure(TriAssetLoader source)
    {
        ReleaseRuntime();
        asset = source;
        Validate(source);
        primitiveCount = source.PrimitiveCount;
        if (primitiveCount <= 0) throw new InvalidOperationException("TriAsset has no triangle primitives.");
        if (!SystemInfo.supportsComputeShaders) throw new NotSupportedException("TriBench requires ComputeShader support.");

        rasterizerCS = Resources.Load<ComputeShader>("TriBench/TriAssetTileRaster");
        if (rasterizerCS == null)
            throw new InvalidOperationException("TriBench TriAssetTileRaster compute shader is missing. Re-run create_unity_triasset_package.py.");

        AllocateBuffers();
        configured = true;
        Debug.Log($"TriBench {MethodName}: configured {primitiveCount} primitives for Tile-Based Compute Rasterization.");
    }

    private void AllocateBuffers()
    {
        tileCounts = new ComputeBuffer(primitiveCount, sizeof(uint), ComputeBufferType.Structured);
        triTileRectMin = new ComputeBuffer(primitiveCount, sizeof(uint) * 2, ComputeBufferType.Structured);
        triTileRectMax = new ComputeBuffer(primitiveCount, sizeof(uint) * 2, ComputeBufferType.Structured);
        triDepths = new ComputeBuffer(primitiveCount, sizeof(float), ComputeBufferType.Structured);
        
        // PreprocessedGeom struct: 28 floats (112 bytes). View depth/clip w are kept so
        // DiffSoup can reproduce native perspective-correct feature sampling
        // and per-pixel depth testing.
        preprocessedGeom = new ComputeBuffer(primitiveCount, 112, ComputeBufferType.Structured);

        // Scan buffers
        int numBlocks = Mathf.CeilToInt(primitiveCount / 256.0f);
        scanOutput = new ComputeBuffer(primitiveCount, sizeof(uint), ComputeBufferType.Structured);
        blockSums = new ComputeBuffer(Mathf.Max(numBlocks, 1), sizeof(uint), ComputeBufferType.Structured);
        scannedBlockSums = new ComputeBuffer(Mathf.Max(numBlocks, 1), sizeof(uint), ComputeBufferType.Structured);
        scanOffsets = new ComputeBuffer(primitiveCount, sizeof(uint), ComputeBufferType.Structured);
        totalKeys = new ComputeBuffer(2, sizeof(uint), ComputeBufferType.Structured);

        // Initial key capacity: capped at 65535 * 256 to ensure radixNumBlocks never exceeds the 65535 GPU dispatch limit
        currentKeyCapacity = Mathf.Clamp(Mathf.Max((int)(primitiveCount * 2.5f), 1024), 1024, 65535 * 256);
        AllocateKeyBuffers(currentKeyCapacity);
    }

    private void AllocateKeyBuffers(int capacity)
    {
        ReleaseKeyBuffers();
        currentKeyCapacity = capacity;

        keysTileA = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
        keysDepthA = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
        valuesA = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);

        keysTileB = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
        keysDepthB = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
        valuesB = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);

        int numBlocks = Mathf.CeilToInt(capacity / 256.0f);
        blockHistograms = new ComputeBuffer(numBlocks * 256, sizeof(uint), ComputeBufferType.Structured);
        globalBucketOffsets = new ComputeBuffer(256, sizeof(uint), ComputeBufferType.Structured);
    }

    private void ReleaseKeyBuffers()
    {
        if (keysTileA != null) { keysTileA.Release(); keysTileA = null; }
        if (keysDepthA != null) { keysDepthA.Release(); keysDepthA = null; }
        if (valuesA != null) { valuesA.Release(); valuesA = null; }
        if (keysTileB != null) { keysTileB.Release(); keysTileB = null; }
        if (keysDepthB != null) { keysDepthB.Release(); keysDepthB = null; }
        if (valuesB != null) { valuesB.Release(); valuesB = null; }
        if (blockHistograms != null) { blockHistograms.Release(); blockHistograms = null; }
        if (globalBucketOffsets != null) { globalBucketOffsets.Release(); globalBucketOffsets = null; }
    }

    private void EnsureScreenResources(int width, int height)
    {
        if (outputTarget == null || lastScreenWidth != width || lastScreenHeight != height)
        {
            if (outputTarget != null) { outputTarget.Release(); Destroy(outputTarget); }

            outputTarget = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBHalf);
            outputTarget.enableRandomWrite = true;
            outputTarget.filterMode = FilterMode.Point;
            outputTarget.Create();

            lastScreenWidth = width;
            lastScreenHeight = height;

            int gridX = Mathf.CeilToInt(width / 16.0f);
            int gridY = Mathf.CeilToInt(height / 16.0f);
            int totalTiles = gridX * gridY;

            if (tileRanges != null) tileRanges.Release();
            tileRanges = new ComputeBuffer(totalTiles, sizeof(uint) * 2, ComputeBufferType.Structured);
            outputTargetDirty = true;
        }
    }

    private bool Accept(Camera camera)
    {
        if (!configured || camera == null) return false;
        if (targetCamera == null) targetCamera = Camera.main;
        return targetCamera == null ? camera.cameraType == CameraType.Game : camera == targetCamera;
    }

    private void OnCameraPreCull(Camera camera)
    {
        if (GraphicsSettings.currentRenderPipeline == null && Accept(camera)) OnCameraFrame(camera);
    }

    private void OnBeginCameraRendering(ScriptableRenderContext context, Camera camera)
    {
        if (GraphicsSettings.currentRenderPipeline != null && Accept(camera)) OnCameraFrame(camera);
    }

    public void OnCameraFrame(Camera camera)
    {
        if (!configured) return;
        EnsureScreenResources(camera.pixelWidth, camera.pixelHeight);
        AttachPipelineCommands(camera);
        // The key-count readback must happen after a frame of the current view
        // has executed, so skip the first frame and read on the second.
        if (pendingOverflowChecks == 1) { pendingOverflowChecks = 2; return; }
        if (pendingOverflowChecks == 2)
        {
            pendingOverflowChecks = 0;
            ReportKeyOverflow();
        }
    }

    public override void PrepareCamera(Camera camera)
    {
        if (!configured || camera == null) return;
        EnsureScreenResources(camera.pixelWidth, camera.pixelHeight);
        AttachPipelineCommands(camera);
        pendingOverflowChecks = 1;
    }

    private void AttachPipelineCommands(Camera camera)
    {
        if (outputTarget == null) return;
        // Clean up any previously attached buffer on the old camera
        if (pipelineCamera != null && pipelineCamera != camera)
        {
            pipelineCamera.RemoveCommandBuffers(CameraEvent.AfterEverything);
            pipelineCamera = null;
        }
        if (pipelineCommands == null) pipelineCommands = new CommandBuffer { name = "TriBench TileRaster" };
        else pipelineCommands.Clear();

        RecordPipeline(camera, pipelineCommands);

        pipelineCommands.Blit(outputTarget, BuiltinRenderTextureType.CameraTarget);

        if (pipelineCamera != camera)
        {
            camera.RemoveCommandBuffers(CameraEvent.AfterEverything);
            camera.AddCommandBuffer(CameraEvent.AfterEverything, pipelineCommands);
            pipelineCamera = camera;
        }
        outputTargetDirty = false;
    }

    private void ReportKeyOverflow()
    {
        if (totalKeys == null) return;
        uint[] counts = new uint[2];
        totalKeys.GetData(counts);
        if (counts[1] != 0u)
        {
            const int maxCapacity = 65535 * 256;
            long required = counts[1];
            if (required > maxCapacity)
            {
                Debug.LogError(
                    $"[TriBench] Tile raster requires {required} keys, above the supported limit {maxCapacity}; "
                    + "the current view cannot be rendered without dropping primitives.");
                return;
            }

            int doubled = currentKeyCapacity <= maxCapacity / 2 ? currentKeyCapacity * 2 : maxCapacity;
            int target = Mathf.Max((int)required, doubled);
            int nextPower = Mathf.NextPowerOfTwo(target);
            int newCapacity = Mathf.Min(nextPower > 0 ? nextPower : maxCapacity, maxCapacity);
            Debug.LogWarning(
                $"[TriBench] Growing tile raster key capacity from {currentKeyCapacity} to {newCapacity} "
                + $"for {required} required keys; re-recording the current camera pipeline.");
            AllocateKeyBuffers(newCapacity);
            pendingOverflowChecks = 1;
            if (pipelineCamera != null) AttachPipelineCommands(pipelineCamera);
        }
        else if (counts[0] > 0u)
            Debug.Log($"[TriBench] Tile raster generated {counts[0]} (tile, primitive) keys for {primitiveCount} primitives.");
    }

    private void RecordPipeline(Camera camera, CommandBuffer commands)
    {
        int width = lastScreenWidth;
        int height = lastScreenHeight;
        int gridX = Mathf.CeilToInt(width / 16.0f);
        int gridY = Mathf.CeilToInt(height / 16.0f);
        int totalTiles = gridX * gridY;
        int numBlocks = Mathf.CeilToInt(primitiveCount / 256.0f);
        int radixNumBlocks = Mathf.CeilToInt(currentKeyCapacity / 256.0f);

        // -----------------------------------------------------------------
        // Shared uniforms (read by the kernels when the buffer executes)
        // -----------------------------------------------------------------
        commands.SetComputeIntParam(rasterizerCS, "_PrimitiveCount", primitiveCount);
        commands.SetComputeIntParam(rasterizerCS, "_RenderMode", RenderMode);
        commands.SetComputeIntParam(rasterizerCS, "_ScreenWidth", width);
        commands.SetComputeIntParam(rasterizerCS, "_ScreenHeight", height);
        commands.SetComputeIntParam(rasterizerCS, "_TileGridWidth", gridX);
        commands.SetComputeIntParam(rasterizerCS, "_TileGridHeight", gridY);
        commands.SetComputeFloatParam(rasterizerCS, "_OpacityFloor", asset.Rendering == null ? 0.0f : Mathf.Clamp01(asset.Rendering.opacity_floor));
        commands.SetComputeFloatParam(rasterizerCS, "_GammaVertexRescale", asset.Rendering == null ? 1.0f : Mathf.Max(asset.Rendering.gamma_vertex_rescale, 1e-6f));
        commands.SetComputeVectorParam(rasterizerCS, "_CameraWorldPos", camera.transform.position);

        Matrix4x4 gpuProj = GL.GetGPUProjectionMatrix(camera.projectionMatrix, false);
        Matrix4x4 viewProj = gpuProj * camera.worldToCameraMatrix;
        commands.SetComputeMatrixParam(rasterizerCS, "_ViewProj", viewProj);
        commands.SetComputeMatrixParam(rasterizerCS, "_InvViewProj", viewProj.inverse);
        commands.SetComputeMatrixParam(rasterizerCS, "_WorldToCamera", camera.worldToCameraMatrix);
        commands.SetComputeIntParam(rasterizerCS, "_KeyCapacity", currentKeyCapacity);
        commands.SetComputeIntParam(rasterizerCS, "_TileRangeCount", totalTiles);

        // -----------------------------------------------------------------
        // 1. Preprocess Triangles Kernel
        // -----------------------------------------------------------------
        int kPreprocess = rasterizerCS.FindKernel("PreprocessTriangles");
        commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_Positions", asset.GetBuffer("positions"));
        commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_Indices", asset.GetBuffer("indices"));
        commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_TileCounts", tileCounts);
        commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_TriTileRectMin", triTileRectMin);
        commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_TriTileRectMax", triTileRectMax);
        commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_TriDepths", triDepths);
        commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_PreprocessedGeom", preprocessedGeom);

        if (!String.IsNullOrEmpty(OpacityBuffer))
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_Opacity", asset.GetBuffer(OpacityBuffer));
        if (UsesSigma)
        {
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_Sigma", asset.GetBuffer("sigma_logits"));
            TriAssetBufferEntry sigma = asset.GetEntry("sigma_logits");
            commands.SetComputeIntParam(rasterizerCS, "_SigmaPerVertex", sigma.shape.Length > 0 && sigma.shape[0] == asset.GetEntry("positions").shape[0] ? 1 : 0);
        }
        else
        {
            // Unity requires all declared buffers to be bound, even if unused
            if (dummySigma == null) dummySigma = new ComputeBuffer(1, sizeof(float), ComputeBufferType.Structured);
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_Sigma", dummySigma);
            commands.SetComputeIntParam(rasterizerCS, "_SigmaPerVertex", 0);
        }
        if (UsesSh)
        {
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_ShDc", asset.GetBuffer("sh_dc"));
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_ShRest", asset.GetBuffer("sh_rest"));
            TriAssetBufferEntry rest = asset.GetEntry("sh_rest");
            commands.SetComputeIntParam(rasterizerCS, "_ShRestCoefficients", rest.shape[rest.shape.Length - 2]);
            commands.SetComputeIntParam(rasterizerCS, "_ShDegree", asset.Rendering == null ? 0 : Mathf.Clamp(asset.Rendering.active_sh_degree, 0, 3));
        }
        else
        {
            if (dummyShDc == null) dummyShDc = new ComputeBuffer(3, sizeof(float), ComputeBufferType.Structured);
            if (dummyShRest == null) dummyShRest = new ComputeBuffer(3, sizeof(float), ComputeBufferType.Structured);
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_ShDc", dummyShDc);
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_ShRest", dummyShRest);
            commands.SetComputeIntParam(rasterizerCS, "_ShRestCoefficients", 0);
            commands.SetComputeIntParam(rasterizerCS, "_ShDegree", 0);
        }
        commands.SetComputeIntParam(rasterizerCS, "_VertexColor", 0);
        commands.SetComputeFloatParam(rasterizerCS, "_Gamma", 1.0f);
        commands.SetComputeIntParam(rasterizerCS, "_FeatureLevel", 0);
        if (RenderMode == 2)
        {
            commands.SetComputeFloatParam(rasterizerCS, "_Gamma", asset.Rendering == null ? 1.0f : Mathf.Max(asset.Rendering.gamma, 1e-4f));
            TriAssetBufferEntry dc = asset.GetEntry("sh_dc");
            commands.SetComputeIntParam(rasterizerCS, "_VertexColor", dc.shape.Length == 4 ? 1 : 0);
        }
        if (RenderMode == 3)
        {
            TriAssetBufferEntry features = asset.GetEntry("features");
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_Features", asset.GetBuffer("features"));
            commands.SetComputeIntParam(rasterizerCS, "_FeatureSamples", features.shape[1]);
            commands.SetComputeIntParam(rasterizerCS, "_FeatureDim", features.shape[2]);
            commands.SetComputeIntParam(rasterizerCS, "_FeatureLevel", asset.Rendering.rmax);

            if (colorMlpParams == null)
            {
                float[] packedParams = new float[595];
                int offset = 0;
                string[] mlpKeys = new string[] {
                    "color_mlp__mlp_0_weight",
                    "color_mlp__mlp_0_bias",
                    "color_mlp__mlp_2_weight",
                    "color_mlp__mlp_2_bias",
                    "color_mlp__mlp_4_weight",
                    "color_mlp__mlp_4_bias"
                };
                foreach (string key in mlpKeys)
                {
                    ComputeBuffer b = asset.GetBuffer(key);
                    int count = b.count;
                    if (offset + count > packedParams.Length)
                        throw new InvalidOperationException($"DiffSoup ColorMLP parameter {key} exceeds the expected 595 scalars.");
                    b.GetData(packedParams, offset, 0, count);
                    offset += count;
                }
                if (offset != packedParams.Length)
                    throw new InvalidOperationException($"DiffSoup ColorMLP has {offset} scalars; expected {packedParams.Length}.");
                colorMlpParams = new ComputeBuffer(595, sizeof(float), ComputeBufferType.Structured);
                colorMlpParams.SetData(packedParams);
            }
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_ColorMlpParams", colorMlpParams);
        }
        else
        {
            // Bind dummy buffer for unused RenderMode 3 parameters
            if (dummyFeatures == null) dummyFeatures = new ComputeBuffer(1, sizeof(float), ComputeBufferType.Structured);
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_Features", dummyFeatures);
            commands.SetComputeIntParam(rasterizerCS, "_FeatureSamples", 0);
            commands.SetComputeIntParam(rasterizerCS, "_FeatureDim", 0);

            if (dummyColorMlpParams == null) dummyColorMlpParams = new ComputeBuffer(1, sizeof(float), ComputeBufferType.Structured);
            commands.SetComputeBufferParam(rasterizerCS, kPreprocess, "_ColorMlpParams", dummyColorMlpParams);
        }

        commands.DispatchCompute(rasterizerCS, kPreprocess, Mathf.CeilToInt(primitiveCount / 256.0f), 1, 1);

        // -----------------------------------------------------------------
        // 2. Clear Tile Ranges (must precede IdentifyTileRanges every frame)
        // -----------------------------------------------------------------
        int kClearRanges = rasterizerCS.FindKernel("ClearTileRanges");
        commands.SetComputeBufferParam(rasterizerCS, kClearRanges, "_TileRanges", tileRanges);
        commands.DispatchCompute(rasterizerCS, kClearRanges, Mathf.CeilToInt(totalTiles / 256.0f), 1, 1);

        // -----------------------------------------------------------------
        // 3. Parallel Prefix Sum (Scan) for Tile Offsets
        // -----------------------------------------------------------------
        int kBlockScan = rasterizerCS.FindKernel("BlockScan");
        int kBlockScanSums = rasterizerCS.FindKernel("BlockScanSums");
        int kAddBlockSums = rasterizerCS.FindKernel("AddBlockSums");

        commands.SetComputeIntParam(rasterizerCS, "_TotalElements", primitiveCount);
        commands.SetComputeIntParam(rasterizerCS, "_NumBlocks", numBlocks);
        commands.SetComputeIntParam(rasterizerCS, "_RadixNumBlocks", radixNumBlocks);

        commands.SetComputeBufferParam(rasterizerCS, kBlockScan, "_ScanInput", tileCounts);
        commands.SetComputeBufferParam(rasterizerCS, kBlockScan, "_ScanOutput", scanOutput);
        commands.SetComputeBufferParam(rasterizerCS, kBlockScan, "_BlockSums", blockSums);
        commands.DispatchCompute(rasterizerCS, kBlockScan, numBlocks, 1, 1);

        commands.SetComputeBufferParam(rasterizerCS, kBlockScanSums, "_BlockSums", blockSums);
        commands.SetComputeBufferParam(rasterizerCS, kBlockScanSums, "_ScannedBlockSums", scannedBlockSums);
        commands.DispatchCompute(rasterizerCS, kBlockScanSums, 1, 1, 1);

        commands.SetComputeBufferParam(rasterizerCS, kAddBlockSums, "_ScanInput", tileCounts);
        commands.SetComputeBufferParam(rasterizerCS, kAddBlockSums, "_ScanOutput", scanOutput);
        commands.SetComputeBufferParam(rasterizerCS, kAddBlockSums, "_ScannedBlockSums", scannedBlockSums);
        commands.SetComputeBufferParam(rasterizerCS, kAddBlockSums, "_ScanOffsetsRW", scanOffsets);
        commands.DispatchCompute(rasterizerCS, kAddBlockSums, numBlocks, 1, 1);

        // -----------------------------------------------------------------
        // 4. DuplicateWithKeys Kernel
        // -----------------------------------------------------------------
        int kDuplicate = rasterizerCS.FindKernel("DuplicateWithKeys");
        commands.SetComputeBufferParam(rasterizerCS, kDuplicate, "_TileCounts", tileCounts);
        commands.SetComputeBufferParam(rasterizerCS, kDuplicate, "_ScanOffsets", scanOffsets);
        commands.SetComputeBufferParam(rasterizerCS, kDuplicate, "_TriTileRectMin", triTileRectMin);
        commands.SetComputeBufferParam(rasterizerCS, kDuplicate, "_TriTileRectMax", triTileRectMax);
        commands.SetComputeBufferParam(rasterizerCS, kDuplicate, "_TriDepths", triDepths);
        commands.SetComputeBufferParam(rasterizerCS, kDuplicate, "_DstKeysTile", keysTileA);
        commands.SetComputeBufferParam(rasterizerCS, kDuplicate, "_DstKeysDepth", keysDepthA);
        commands.SetComputeBufferParam(rasterizerCS, kDuplicate, "_DstValues", valuesA);
        commands.DispatchCompute(rasterizerCS, kDuplicate, Mathf.CeilToInt(primitiveCount / 256.0f), 1, 1);

        // -----------------------------------------------------------------
        // 5. Finalize the validated (tile, primitive) key count
        // -----------------------------------------------------------------
        int kFinalize = rasterizerCS.FindKernel("FinalizeTotalKeys");
        commands.SetComputeBufferParam(rasterizerCS, kFinalize, "_ScanOffsetsRW", scanOffsets);
        commands.SetComputeBufferParam(rasterizerCS, kFinalize, "_ScanInput", tileCounts);
        commands.SetComputeBufferParam(rasterizerCS, kFinalize, "_TotalKeysRW", totalKeys);
        commands.DispatchCompute(rasterizerCS, kFinalize, 1, 1, 1);

        // -----------------------------------------------------------------
        // 6. LSD Radix Sort: four 8-bit depth passes, then four 8-bit tile
        //    passes.  Stability within a pass preserves the component that was
        //    sorted first, so the final order is tile-major, depth-minor
        //    (front-to-back) as required by the transmittance compositor.
        //    Eight passes leave the sorted keys in the A buffers.
        // -----------------------------------------------------------------
        int kRadixCount = rasterizerCS.FindKernel("RadixCount");
        int kRadixScanBuckets = rasterizerCS.FindKernel("RadixScanBuckets");
        int kRadixScatter = rasterizerCS.FindKernel("RadixScatter");
        for (int pass = 0; pass < 8; ++pass)
        {
            bool srcIsA = pass % 2 == 0;
            ComputeBuffer srcTile = srcIsA ? keysTileA : keysTileB;
            ComputeBuffer srcDepth = srcIsA ? keysDepthA : keysDepthB;
            ComputeBuffer srcValues = srcIsA ? valuesA : valuesB;
            ComputeBuffer dstTile = srcIsA ? keysTileB : keysTileA;
            ComputeBuffer dstDepth = srcIsA ? keysDepthB : keysDepthA;
            ComputeBuffer dstValues = srcIsA ? valuesB : valuesA;

            commands.SetComputeIntParam(rasterizerCS, "_RadixShift", (pass & 3) * 8);
            commands.SetComputeIntParam(rasterizerCS, "_RadixBitPass", pass < 4 ? 0 : 1);

            commands.SetComputeBufferParam(rasterizerCS, kRadixCount, "_TotalKeys", totalKeys);
            commands.SetComputeBufferParam(rasterizerCS, kRadixCount, "_SrcKeysDepth", srcDepth);
            commands.SetComputeBufferParam(rasterizerCS, kRadixCount, "_SrcKeysTile", srcTile);
            commands.SetComputeBufferParam(rasterizerCS, kRadixCount, "_BlockHistograms", blockHistograms);
            commands.DispatchCompute(rasterizerCS, kRadixCount, radixNumBlocks, 1, 1);

            commands.SetComputeBufferParam(rasterizerCS, kRadixScanBuckets, "_BlockHistograms", blockHistograms);
            commands.SetComputeBufferParam(rasterizerCS, kRadixScanBuckets, "_GlobalBucketOffsets", globalBucketOffsets);
            commands.DispatchCompute(rasterizerCS, kRadixScanBuckets, 1, 1, 1);

            commands.SetComputeBufferParam(rasterizerCS, kRadixScatter, "_TotalKeys", totalKeys);
            commands.SetComputeBufferParam(rasterizerCS, kRadixScatter, "_SrcKeysTile", srcTile);
            commands.SetComputeBufferParam(rasterizerCS, kRadixScatter, "_SrcKeysDepth", srcDepth);
            commands.SetComputeBufferParam(rasterizerCS, kRadixScatter, "_SrcValues", srcValues);
            commands.SetComputeBufferParam(rasterizerCS, kRadixScatter, "_BlockHistograms", blockHistograms);
            commands.SetComputeBufferParam(rasterizerCS, kRadixScatter, "_GlobalBucketOffsets", globalBucketOffsets);
            commands.SetComputeBufferParam(rasterizerCS, kRadixScatter, "_DstKeysTile", dstTile);
            commands.SetComputeBufferParam(rasterizerCS, kRadixScatter, "_DstKeysDepth", dstDepth);
            commands.SetComputeBufferParam(rasterizerCS, kRadixScatter, "_DstValues", dstValues);
            commands.DispatchCompute(rasterizerCS, kRadixScatter, radixNumBlocks, 1, 1);
        }

        // -----------------------------------------------------------------
        // 7. IdentifyTileRanges Kernel
        // -----------------------------------------------------------------
        int kIdentify = rasterizerCS.FindKernel("IdentifyTileRanges");
        commands.SetComputeBufferParam(rasterizerCS, kIdentify, "_SrcKeysTile", keysTileA);
        commands.SetComputeBufferParam(rasterizerCS, kIdentify, "_TileRanges", tileRanges);
        commands.SetComputeBufferParam(rasterizerCS, kIdentify, "_TotalKeys", totalKeys);
        commands.DispatchCompute(rasterizerCS, kIdentify, Mathf.CeilToInt(currentKeyCapacity / 256.0f), 1, 1);

        // -----------------------------------------------------------------
        // 8. TileRasterize Kernel (Software Rasterizer & Compositor)
        // -----------------------------------------------------------------
        int kRasterize = rasterizerCS.FindKernel("TileRasterize");
        commands.SetComputeBufferParam(rasterizerCS, kRasterize, "_TileRanges", tileRanges);
        commands.SetComputeBufferParam(rasterizerCS, kRasterize, "_SrcValues", valuesA);
        commands.SetComputeBufferParam(rasterizerCS, kRasterize, "_PreprocessedGeom", preprocessedGeom);
        commands.SetComputeBufferParam(rasterizerCS, kRasterize, "_TotalKeys", totalKeys);
        commands.SetComputeBufferParam(rasterizerCS, kRasterize, "_Opacity", asset.GetBuffer(OpacityBuffer));
        commands.SetComputeBufferParam(
            rasterizerCS, kRasterize, "_Features",
            RenderMode == 3 ? asset.GetBuffer("features") : dummyFeatures);
        commands.SetComputeBufferParam(
            rasterizerCS, kRasterize, "_ColorMlpParams",
            RenderMode == 3 ? colorMlpParams : dummyColorMlpParams);
        commands.SetComputeTextureParam(rasterizerCS, kRasterize, "_OutputTarget", outputTarget);
        commands.DispatchCompute(rasterizerCS, kRasterize, gridX, gridY, 1);
    }

    private void ReleaseRuntime()
    {
        configured = false;
        if (pipelineCamera != null)
        {
            pipelineCamera.RemoveCommandBuffers(CameraEvent.AfterEverything);
            pipelineCamera = null;
        }
        if (pipelineCommands != null) { pipelineCommands.Dispose(); pipelineCommands = null; }
        if (tileCounts != null) { tileCounts.Release(); tileCounts = null; }
        if (triTileRectMin != null) { triTileRectMin.Release(); triTileRectMin = null; }
        if (triTileRectMax != null) { triTileRectMax.Release(); triTileRectMax = null; }
        if (triDepths != null) { triDepths.Release(); triDepths = null; }
        if (preprocessedGeom != null) { preprocessedGeom.Release(); preprocessedGeom = null; }

        if (scanOutput != null) { scanOutput.Release(); scanOutput = null; }
        if (blockSums != null) { blockSums.Release(); blockSums = null; }
        if (scannedBlockSums != null) { scannedBlockSums.Release(); scannedBlockSums = null; }
        if (scanOffsets != null) { scanOffsets.Release(); scanOffsets = null; }

        ReleaseKeyBuffers();
        if (tileRanges != null) { tileRanges.Release(); tileRanges = null; }
        if (totalKeys != null) { totalKeys.Release(); totalKeys = null; }
        if (outputTarget != null) { outputTarget.Release(); Destroy(outputTarget); outputTarget = null; }
        if (dummySigma != null) { dummySigma.Release(); dummySigma = null; }
        if (dummyFeatures != null) { dummyFeatures.Release(); dummyFeatures = null; }
        if (dummyShDc != null) { dummyShDc.Release(); dummyShDc = null; }
        if (dummyShRest != null) { dummyShRest.Release(); dummyShRest = null; }
        if (colorMlpParams != null) { colorMlpParams.Release(); colorMlpParams = null; }
        if (dummyColorMlpParams != null) { dummyColorMlpParams.Release(); dummyColorMlpParams = null; }
    }
}

public sealed class TriangleSplattingTileRenderer : TriAssetTileRasterRenderer
{
    public override string MethodName { get { return "triangle-splatting"; } }
    protected override int RenderMode { get { return 0; } }
}

public sealed class MeshSplattingTileRenderer : TriAssetTileRasterRenderer
{
    public override string MethodName { get { return "mesh-splatting"; } }
    protected override int RenderMode { get { return 1; } }
    protected override string OpacityBuffer { get { return "triangle_opacity"; } }
}

public sealed class D2TSTileRenderer : TriAssetTileRasterRenderer
{
    public override string MethodName { get { return "2dts"; } }
    protected override int RenderMode { get { return 2; } }
    protected override bool UsesSigma { get { return false; } }
}

public sealed class DiffSoupTileRenderer : TriAssetTileRasterRenderer
{
    public override string MethodName { get { return "diffsoup"; } }
    protected override int RenderMode { get { return 3; } }
    protected override string OpacityBuffer { get { return "alpha"; } }
    protected override bool UsesSigma { get { return false; } }
    protected override bool UsesSh { get { return false; } }
    protected override void Validate(TriAssetLoader source)
    {
        base.Validate(source);
        source.Require("features");
        string[] mlpKeys = new string[] {
            "color_mlp__mlp_0_weight", "color_mlp__mlp_0_bias",
            "color_mlp__mlp_2_weight", "color_mlp__mlp_2_bias",
            "color_mlp__mlp_4_weight", "color_mlp__mlp_4_bias"
        };
        foreach (string key in mlpKeys) source.Require(key);

        TriAssetBufferEntry features = source.GetEntry("features");
        TriAssetBufferEntry alpha = source.GetEntry("alpha");
        int level = source.Rendering == null ? -1 : source.Rendering.rmax;
        int expectedSamples = level == 0 ? 3
            : level > 0 && level <= 15 ? ((1 << (level - 1)) + 1) * ((1 << level) + 1)
            : -1;
        bool validFeatures = features.shape != null && features.shape.Length == 3
            && features.shape[0] == source.PrimitiveCount && features.shape[1] == expectedSamples && features.shape[2] == 7;
        bool validAlpha = alpha.shape != null && alpha.shape.Length == 3
            && alpha.shape[0] == source.PrimitiveCount && alpha.shape[1] == expectedSamples && alpha.shape[2] == 1;
        if (!validFeatures || !validAlpha)
            throw new InvalidOperationException(
                $"DiffSoup requires accumulated Rmax feature/alpha lattices [T,{expectedSamples},7/1]; "
                + $"received feature level {level} with incompatible buffer shapes.");
    }
}
