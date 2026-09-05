// MeshSplatBench procedural renderer for feature-preserving .triasset packages.
// It uses a ComputeShader to produce an indirect visible-primitive list and a
// Metal-compatible DrawProcedural raster pass.  It deliberately does not turn
// TriAssets into Unity Meshes: all learned attributes stay in ComputeBuffers.
using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace MeshSplatBench.Unity
{
    public abstract class TriAssetSplatRenderer : TriAssetRenderer
    {
        [Tooltip("Optional; defaults to Camera.main.")] public Camera targetCamera;
        [Tooltip("Reject rather than silently render an incomplete package.")] public bool strictValidation = true;

        protected TriAssetLoader asset;
        private ComputeShader culler;
        protected Material material;
        private ComputeBuffer visible;
        private ComputeBuffer indirectArgs;
        private int primitiveCount;
        private bool configured;
        private readonly Bounds drawBounds = new Bounds(Vector3.zero, Vector3.one * 1000000.0f);

        protected abstract int RenderMode { get; }
        protected virtual string OpacityBuffer { get { return "opacity_logits"; } }
        protected virtual bool UsesSigma { get { return true; } }
        protected virtual bool UsesSh { get { return true; } }
        protected virtual string ResolveShaderName(TriAssetLoader source) { return "MeshSplatBench/TriAssetSplat"; }

        protected virtual void Validate(TriAssetLoader source)
        {
            source.Require("positions"); source.Require("indices");
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

        private void OnDestroy() { ReleaseRuntime(); }

        public override void Configure(TriAssetLoader source)
        {
            ReleaseRuntime();
            asset = source;
            Validate(source);
            primitiveCount = source.PrimitiveCount;
            if (primitiveCount <= 0) throw new InvalidOperationException("TriAsset has no triangle primitives.");
            if (!SystemInfo.supportsComputeShaders) throw new NotSupportedException("MeshSplatBench requires ComputeShader support (Metal on Apple Silicon).");

            culler = Resources.Load<ComputeShader>("MeshSplatBench/TriAssetCull");
            Shader shader = Shader.Find(ResolveShaderName(source));
            if (culler == null || shader == null)
                throw new InvalidOperationException("MeshSplatBench shaders are missing. Re-run create_unity_triasset_package.py.");
            material = new Material(shader) { hideFlags = HideFlags.DontSave };
            visible = new ComputeBuffer(primitiveCount, sizeof(uint), ComputeBufferType.Structured);
            indirectArgs = new ComputeBuffer(1, sizeof(uint) * 4, ComputeBufferType.IndirectArguments);
            indirectArgs.SetData(new uint[] { 3, (uint)primitiveCount, 0, 0 });
            BindBuffers();
            configured = true;
            Debug.Log($"MeshSplatBench {MethodName}: configured {primitiveCount} primitives for procedural Metal rendering.");
        }

        protected virtual void BindBuffers()
        {
            material.SetInt("_RenderMode", RenderMode);
            material.SetInt("_PrimitiveCount", primitiveCount);
            material.SetFloat("_OpacityFloor", asset.Rendering == null ? 0.0f : Mathf.Clamp01(asset.Rendering.opacity_floor));
            material.SetFloat("_GammaVertexRescale", asset.Rendering == null ? 1.0f : Mathf.Max(asset.Rendering.gamma_vertex_rescale, 1e-6f));
            material.SetBuffer("_Positions", asset.GetBuffer("positions"));
            material.SetBuffer("_Indices", asset.GetBuffer("indices"));
            material.SetBuffer("_Visible", visible);
            if (!String.IsNullOrEmpty(OpacityBuffer)) material.SetBuffer("_Opacity", asset.GetBuffer(OpacityBuffer));
            if (UsesSigma) material.SetBuffer("_Sigma", asset.GetBuffer("sigma_logits"));
            if (UsesSigma)
            {
                TriAssetBufferEntry sigma = asset.GetEntry("sigma_logits");
                material.SetInt("_SigmaPerVertex", sigma.shape.Length > 0 && sigma.shape[0] == asset.GetEntry("positions").shape[0] ? 1 : 0);
            }
            if (UsesSh)
            {
                material.SetBuffer("_ShDc", asset.GetBuffer("sh_dc"));
                material.SetBuffer("_ShRest", asset.GetBuffer("sh_rest"));
                TriAssetBufferEntry rest = asset.GetEntry("sh_rest");
                material.SetInt("_ShRestCoefficients", rest.shape[rest.shape.Length - 2]);
                // Checkpoints allocate the maximum SH tensor before all degrees
                // become active. Tensor shape therefore cannot replace the
                // serialized active_sh_degree contract.
                material.SetInt("_ShDegree", asset.Rendering == null
                    ? 0
                    : Mathf.Clamp(asset.Rendering.active_sh_degree, 0, 3));
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
            if (GraphicsSettings.currentRenderPipeline == null && Accept(camera)) Render(camera);
        }

        private void OnBeginCameraRendering(ScriptableRenderContext context, Camera camera)
        {
            if (GraphicsSettings.currentRenderPipeline != null && Accept(camera)) Render(camera);
        }

        private void Render(Camera camera)
        {
            int kernel = culler.FindKernel("CullTriangles");
            culler.SetInt("_PrimitiveCount", primitiveCount);
            culler.SetMatrix("_ViewProj", GL.GetGPUProjectionMatrix(camera.projectionMatrix, true) * camera.worldToCameraMatrix);
            culler.SetBuffer(kernel, "_Positions", asset.GetBuffer("positions"));
            culler.SetBuffer(kernel, "_Indices", asset.GetBuffer("indices"));
            culler.SetBuffer(kernel, "_Visible", visible);
            culler.Dispatch(kernel, Mathf.CeilToInt(primitiveCount / 128.0f), 1, 1);
            material.SetVector("_CameraWorldPos", camera.transform.position);
            Graphics.DrawProceduralIndirect(material, drawBounds, MeshTopology.Triangles, indirectArgs, 0, camera, null,
                ShadowCastingMode.Off, false, gameObject.layer);
        }

        private void ReleaseRuntime()
        {
            configured = false;
            if (visible != null) { visible.Release(); visible = null; }
            if (indirectArgs != null) { indirectArgs.Release(); indirectArgs = null; }
            if (material != null) { Destroy(material); material = null; }
        }
    }

    public sealed class TriangleSplattingTriAssetRenderer : TriAssetSplatRenderer
    {
        public override string MethodName { get { return "triangle-splatting"; } }
        protected override int RenderMode { get { return 0; } }
    }

    public sealed class MeshSplattingTriAssetRenderer : TriAssetSplatRenderer
    {
        public override string MethodName { get { return "mesh-splatting"; } }
        protected override int RenderMode { get { return 1; } }
        // The exporter has already applied the opacity floor and reduced the
        // three activated vertex weights with min(), exactly as native CUDA.
        protected override string OpacityBuffer { get { return "triangle_opacity"; } }
        protected override string ResolveShaderName(TriAssetLoader source)
        {
            return source.Rendering != null && source.Rendering.terminal_solid_eligible
                ? "MeshSplatBench/MeshSplatTerminalSolid"
                : base.ResolveShaderName(source);
        }
    }

    public sealed class D2TSTriAssetRenderer : TriAssetSplatRenderer
    {
        public override string MethodName { get { return "2dts"; } }
        protected override int RenderMode { get { return 2; } }
        protected override bool UsesSigma { get { return false; } }
        protected override void BindBuffers()
        {
            base.BindBuffers();
            TriAssetBufferEntry dc = asset.GetEntry("sh_dc");
            material.SetInt("_VertexColor", dc.shape.Length == 4 ? 1 : 0);
            material.SetFloat("_Gamma", asset.Rendering == null ? 1.0f : Mathf.Max(asset.Rendering.gamma, 1e-4f));
        }
    }

    public sealed class DiffSoupTriAssetRenderer : TriAssetSplatRenderer
    {
        public override string MethodName { get { return "diffsoup"; } }
        protected override int RenderMode { get { return 3; } }
        protected override string OpacityBuffer { get { return "alpha"; } }
        protected override bool UsesSigma { get { return false; } }
        protected override bool UsesSh { get { return false; } }
        protected override void Validate(TriAssetLoader source)
        {
            base.Validate(source); source.Require("features");
        }
        protected override void BindBuffers()
        {
            base.BindBuffers();
            TriAssetBufferEntry features = asset.GetEntry("features");
            material.SetBuffer("_Features", asset.GetBuffer("features"));
            material.SetInt("_FeatureSamples", features.shape[1]);
            material.SetInt("_FeatureDim", features.shape[2]);
        }
    }
}
