#!/usr/bin/env python3
"""Install the lightweight TriBench Unity-native asset loader into a project.

This installs the package contract and GPU-buffer loader, not a fake flat-color
implementation of the four renderers. A scene must attach a matching method
renderer before a ``.triasset`` can be benchmarked as native-equivalent.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TRI_ASSET_RENDERER = r'''using UnityEngine;

namespace TriBench.Unity
{
    // Implementations must consume raw TriAsset buffers, never baked colours.
    public abstract class TriAssetRenderer : MonoBehaviour
    {
        public abstract string MethodName { get; }
        public virtual bool SupportsMethod(string method) { return MethodName == method; }
        public abstract void Configure(TriAssetLoader asset);
    }
}
'''


TRI_ASSET_LOADER = r'''using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace TriBench.Unity
{
    [Serializable]
    public sealed class TriAssetBufferEntry
    {
        public string name;
        public string file;
        public string dtype;
        public int[] shape;
        public string semantic;
    }

    [Serializable]
    public sealed class TriAssetManifest
    {
        public string schema_version;
        public int export_contract_revision;
        public string method;
        public TriAssetBufferEntry[] buffer_list;
        public TriAssetRendering rendering;
    }

    [Serializable]
    public sealed class TriAssetRendering
    {
        public int active_sh_degree;
        public float gamma = 1.0f;
        public float opacity_floor = 0.0f;
        public bool gamma_rescale;
        public float gamma_vertex_rescale = 1.0f;
        public bool requires_per_camera_depth_sort;
        public bool terminal_solid_eligible;
        public bool vertex_color;
        public int rmin;
        public int rmax;
        public int feature_dim;
        public string background_color;
        public string unity_profile_name;
        public bool cuda_equivalent;
        public bool unity_method_aware_requires_per_camera_depth_sort;
        public string unity_method_aware_status;
        public TriAssetGeneralPurpose general_purpose;
    }

    [Serializable]
    public sealed class TriAssetGeneralPurpose
    {
        public bool supported;
        public string appearance;
        public string reason;
        public bool view_dependent_appearance;
        public bool opaque_depth_tested;
    }

    // Reads `.triasset` tensors and exposes them as ComputeBuffers. It refuses
    // native-equivalent rendering unless a matching renderer is attached.
    public sealed class TriAssetLoader : MonoBehaviour
    {
        [Header("Input")]
        [Tooltip("Absolute path to a .triasset directory.")]
        public string assetDirectory = "";
        [Header("Validation")]
        public bool loadOnStart = true;
        [Tooltip("Topology debugging only; never use this for quality/FPS results.")]
        public bool allowMeshPreview = false;

        private string root;
        private TriAssetManifest manifest;
        private readonly Dictionary<string, TriAssetBufferEntry> entries = new Dictionary<string, TriAssetBufferEntry>();
        private readonly Dictionary<string, ComputeBuffer> gpuBuffers = new Dictionary<string, ComputeBuffer>();
        public string Method { get { return manifest == null ? "" : manifest.method; } }
        public TriAssetRendering Rendering { get { return manifest == null ? null : manifest.rendering; } }
        public int PrimitiveCount {
            get {
                TriAssetBufferEntry indices;
                return entries.TryGetValue("indices", out indices) && indices.shape != null && indices.shape.Length > 0
                    ? indices.shape[0] : 0;
            }
        }

        private void Start() { if (loadOnStart) Load(); }

        public void Load()
        {
            ReleaseBuffers();
            root = Path.GetFullPath(assetDirectory);
            string manifestPath = Path.Combine(root, "manifest.json");
            if (!File.Exists(manifestPath)) throw new FileNotFoundException("TriAsset manifest not found", manifestPath);
            manifest = JsonUtility.FromJson<TriAssetManifest>(File.ReadAllText(manifestPath));
            if (manifest == null || manifest.schema_version != "1.0" || manifest.export_contract_revision < 2 || manifest.buffer_list == null)
                throw new InvalidDataException("Unsupported or stale TriAsset manifest; expected schema 1.0 contract revision 2.");
            ApplyEvaluationBackground();
            entries.Clear();
            foreach (TriAssetBufferEntry entry in manifest.buffer_list)
            {
                if (entry == null || String.IsNullOrEmpty(entry.name) || String.IsNullOrEmpty(entry.file))
                    throw new InvalidDataException("TriAsset has an invalid buffer entry.");
                entries.Add(entry.name, entry);
            }
            Require("positions");
            Require("indices");

            TriAssetRenderer renderer = GetComponent<TriAssetRenderer>();
            if (renderer == null)
            {
                if (!allowMeshPreview)
                    throw new InvalidOperationException("Attach a method-specific TriAssetRenderer before native-equivalent rendering.");
                CreateTopologyPreview();
                return;
            }
            if (!renderer.SupportsMethod(manifest.method))
                throw new InvalidOperationException($"Renderer {renderer.MethodName} cannot render asset method {manifest.method}.");
            renderer.Configure(this);
        }

        private void ApplyEvaluationBackground()
        {
            Camera camera = Camera.main;
            if (camera == null || manifest.rendering == null) return;
            string value = manifest.rendering.background_color ?? "";
            if (value != "black" && value != "white")
                throw new InvalidDataException("TriAsset must declare background_color as black or white.");
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = value == "white" ? Color.white : Color.black;
        }

        public ComputeBuffer GetBuffer(string name)
        {
            if (gpuBuffers.TryGetValue(name, out ComputeBuffer existing)) return existing;
            TriAssetBufferEntry entry = Require(name);
            int scalarBytes = ScalarBytes(entry.dtype);
            byte[] raw = File.ReadAllBytes(Path.Combine(root, entry.file));
            int expectedBytes = scalarBytes;
            for (int i = 0; i < entry.shape.Length; ++i) expectedBytes *= entry.shape[i];
            if (raw.Length != expectedBytes)
                throw new InvalidDataException($"Buffer {name} has {raw.Length} bytes, expected {expectedBytes}.");
            // Store scalars with stride 4. This avoids Unity's StructuredBuffer
            // stride limit for DiffSoup's high-resolution per-face tensors.
            ComputeBuffer buffer = new ComputeBuffer(raw.Length / scalarBytes, scalarBytes, ComputeBufferType.Structured);
            if (entry.dtype == "float32")
            {
                float[] values = new float[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
                buffer.SetData(values);
            }
            else if (entry.dtype == "int32")
            {
                int[] values = new int[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
                buffer.SetData(values);
            }
            else if (entry.dtype == "uint32")
            {
                uint[] values = new uint[raw.Length / 4];
                Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
                buffer.SetData(values);
            }
            else throw new InvalidDataException($"Unsupported TriAsset dtype {entry.dtype}.");
            gpuBuffers.Add(name, buffer);
            return buffer;
        }

        public TriAssetBufferEntry Require(string name)
        {
            if (!entries.TryGetValue(name, out TriAssetBufferEntry entry))
                throw new KeyNotFoundException($"TriAsset {Method} is missing required buffer {name}.");
            return entry;
        }

        public bool HasBuffer(string name) { return entries.ContainsKey(name); }

        public TriAssetBufferEntry GetEntry(string name) { return Require(name); }

        private void CreateTopologyPreview()
        {
            TriAssetBufferEntry position = Require("positions");
            TriAssetBufferEntry index = Require("indices");
            if (position.dtype != "float32" || position.shape.Length != 2 || position.shape[1] != 3)
                throw new InvalidDataException("Topology preview requires float32 positions [N,3].");
            if (index.dtype != "int32" || index.shape.Length != 2 || index.shape[1] != 3)
                throw new InvalidDataException("Topology preview requires int32 indices [F,3].");
            byte[] positionRaw = File.ReadAllBytes(Path.Combine(root, position.file));
            byte[] indexRaw = File.ReadAllBytes(Path.Combine(root, index.file));
            float[] p = new float[positionRaw.Length / 4];
            int[] i = new int[indexRaw.Length / 4];
            Buffer.BlockCopy(positionRaw, 0, p, 0, positionRaw.Length);
            Buffer.BlockCopy(indexRaw, 0, i, 0, indexRaw.Length);
            Vector3[] vertices = new Vector3[position.shape[0]];
            for (int v = 0; v < vertices.Length; ++v) vertices[v] = new Vector3(p[3 * v], p[3 * v + 1], p[3 * v + 2]);
            Mesh mesh = new Mesh { indexFormat = vertices.Length > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16 };
            mesh.vertices = vertices;
            mesh.triangles = i;
            mesh.RecalculateBounds();
            MeshFilter filter = GetComponent<MeshFilter>() ?? gameObject.AddComponent<MeshFilter>();
            filter.sharedMesh = mesh;
            if (GetComponent<MeshRenderer>() == null) gameObject.AddComponent<MeshRenderer>();
            Debug.LogWarning("TriAsset topology preview is active. Do not use it for quality/FPS benchmarking.");
        }

        private static int ScalarBytes(string dtype)
        {
            if (dtype == "float32" || dtype == "int32" || dtype == "uint32") return 4;
            throw new InvalidDataException($"Unsupported TriAsset dtype {dtype}.");
        }
        private void OnDestroy() { ReleaseBuffers(); }
        private void ReleaseBuffers()
        {
            foreach (ComputeBuffer buffer in gpuBuffers.Values) buffer.Release();
            gpuBuffers.Clear();
        }
    }
}
'''


TRI_BENCH_BENCHMARK = r'''using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace TriBench.Unity
{
    [Serializable]
    public sealed class TriBenchBenchmarkResult
    {
        public string method;
        public int width;
        public int height;
        public int warmupFrames;
        public int measuredFrames;
        public float cpuFrameMsMean;
        public float cpuFrameMsP50;
        public float cpuFrameMsP95;
        public float gpuFrameMsMean;
        public float gpuFrameMsP50;
        public float gpuFrameMsP95;
        public float fpsMean;
        public bool frameTimingAvailable;
    }

    // Attach after the method renderer is validated. It renders to a fixed
    // RenderTexture and writes one summary only after timing has completed.
    public sealed class TriBenchFrameBenchmark : MonoBehaviour
    {
        public Camera targetCamera;
        public TriAssetLoader asset;
        public int width = 1920;
        public int height = 1080;
        public int warmupFrames = 120;
        public int measuredFrames = 600;
        public string outputFile = "tribench_unity_benchmark.json";

        private RenderTexture target;
        private int frames;
        private readonly List<float> cpuMs = new List<float>();
        private readonly List<float> gpuMs = new List<float>();

        private void Start()
        {
            if (targetCamera == null) targetCamera = Camera.main;
            if (targetCamera == null) throw new InvalidOperationException("TriBench benchmark needs a camera.");
            if (asset == null) asset = FindObjectOfType<TriAssetLoader>();
            if (asset == null || String.IsNullOrEmpty(asset.Method)) throw new InvalidOperationException("Load a TriAsset before benchmarking.");
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
            target = new RenderTexture(width, height, 24, RenderTextureFormat.ARGBHalf);
            target.Create();
            targetCamera.targetTexture = target;
            FrameTimingManager.CaptureFrameTimings();
        }

        private void LateUpdate()
        {
            frames++;
            if (frames <= warmupFrames) return;
            if (frames <= warmupFrames + measuredFrames)
            {
                cpuMs.Add(Time.unscaledDeltaTime * 1000.0f);
                FrameTiming[] timing = new FrameTiming[1];
                if (FrameTimingManager.GetLatestTimings(1, timing) > 0 && timing[0].gpuFrameTime > 0.0)
                    gpuMs.Add((float)timing[0].gpuFrameTime);
                FrameTimingManager.CaptureFrameTimings();
                return;
            }
            WriteResult();
            enabled = false;
        }

        private void WriteResult()
        {
            float cpuMean = Mean(cpuMs);
            float gpuMean = Mean(gpuMs);
            TriBenchBenchmarkResult result = new TriBenchBenchmarkResult {
                method = asset.Method, width = width, height = height,
                warmupFrames = warmupFrames, measuredFrames = cpuMs.Count,
                cpuFrameMsMean = cpuMean, cpuFrameMsP50 = Quantile(cpuMs, 0.50f), cpuFrameMsP95 = Quantile(cpuMs, 0.95f),
                gpuFrameMsMean = gpuMean, gpuFrameMsP50 = Quantile(gpuMs, 0.50f), gpuFrameMsP95 = Quantile(gpuMs, 0.95f),
                fpsMean = cpuMean > 0 ? 1000.0f / cpuMean : 0,
                frameTimingAvailable = gpuMs.Count > 0,
            };
            string path = Path.IsPathRooted(outputFile) ? outputFile : Path.Combine(Application.persistentDataPath, outputFile);
            File.WriteAllText(path, JsonUtility.ToJson(result, true));
            Debug.Log("TriBench benchmark saved: " + path);
        }

        private static float Mean(List<float> values)
        {
            if (values.Count == 0) return 0;
            float sum = 0; foreach (float value in values) sum += value;
            return sum / values.Count;
        }

        private static float Quantile(List<float> values, float quantile)
        {
            if (values.Count == 0) return 0;
            List<float> sorted = new List<float>(values);
            sorted.Sort();
            int index = Mathf.Clamp(Mathf.CeilToInt(quantile * sorted.Count) - 1, 0, sorted.Count - 1);
            return sorted[index];
        }

        private void OnDestroy()
        {
            if (target != null) { target.Release(); Destroy(target); }
        }
    }
}
'''


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Install the TriBench TriAsset Unity loader.")
    parser.add_argument("--unity-project", type=Path, required=True, help="Unity project root containing Assets/.")
    args = parser.parse_args()
    assets = args.unity_project.expanduser().resolve() / "Assets"
    if not assets.is_dir():
        parser.error(f"Unity Assets directory not found: {assets}")
    root = assets / "TriBench" / "Scripts"
    _write(root / "TriAssetRenderer.cs", TRI_ASSET_RENDERER)
    _write(root / "TriAssetLoader.cs", TRI_ASSET_LOADER)
    _write(root / "TriBenchFrameBenchmark.cs", TRI_BENCH_BENCHMARK)
    template = Path(__file__).with_name("unity_triasset_renderer")
    if not template.is_dir():
        raise RuntimeError(f"Unity renderer template directory is missing: {template}")
    _write(root / "TriAssetSplatRenderer.cs", (template / "TriAssetSplatRenderer.cs").read_text(encoding="utf-8"))
    _write(root / "GeneralPurposeMeshTriAssetRenderer.cs", (template / "GeneralPurposeMeshTriAssetRenderer.cs").read_text(encoding="utf-8"))
    shader_root = assets / "TriBench" / "Shaders"
    _write(shader_root / "TriAssetSplat.shader", (template / "TriAssetSplat.shader").read_text(encoding="utf-8"))
    _write(shader_root / "MeshSplatTerminalSolid.shader", (template / "MeshSplatTerminalSolid.shader").read_text(encoding="utf-8"))
    _write(shader_root / "GeneralPurposeVertexColor.shader", (template / "GeneralPurposeVertexColor.shader").read_text(encoding="utf-8"))
    resource_root = assets / "Resources" / "TriBench"
    _write(resource_root / "TriAssetCull.compute", (template / "TriAssetCull.compute").read_text(encoding="utf-8"))
    print(f"Installed TriBench TriAsset loader and Metal procedural renderer under {assets / 'TriBench'}")
    print("Attach TriAssetLoader and one method-aware renderer, or GeneralPurposeMeshTriAssetRenderer when manifest.rendering.general_purpose.supported is true.")
    print("The DiffSoup component is an alpha-tested neural-feature preview; it is not native-equivalent until the exported ColorMLP and multires interpolation are validated.")


if __name__ == "__main__":
    main()
