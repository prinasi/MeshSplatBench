using System;
using UnityEngine;

namespace MeshSplatBench.UnityNative
{
    /// <summary>
    /// Configures the fixed evaluation scene when it is launched as a Metal
    /// standalone Player. Editor batch runs use TriAssetBatchRunner instead.
    /// </summary>
    [DefaultExecutionOrder(-20000)]
    public sealed class TriAssetRuntimeBatchBootstrap : MonoBehaviour
    {
        void Awake()
        {
            if (Application.isEditor || !TriAssetRuntimeOptions.Has("-profile-only")) return;
            string method = TriAssetRuntimeOptions.Get("-method", "triangle-splatting").ToLowerInvariant();
            string asset = TriAssetRuntimeOptions.Get("-triasset", "");
            bool standard = TriAssetRuntimeOptions.Has("-standard-mesh");
            bool methodSpecific = TriAssetRuntimeOptions.Has("-method-specific");
            Camera camera = Camera.main;
            TriAssetRenderer active;

            TriangleSplattingTriAssetRenderer triangle = GetComponent<TriangleSplattingTriAssetRenderer>();
            if (standard)
            {
                if (triangle != null) triangle.enabled = false;
                StandardMeshTriAssetRenderer mesh = GetComponent<StandardMeshTriAssetRenderer>() ?? gameObject.AddComponent<StandardMeshTriAssetRenderer>();
                mesh.AssetDirectory = asset; mesh.TargetCamera = camera; active = mesh;
            }
            else if (methodSpecific && method == "diffsoup")
            {
                if (triangle != null) triangle.enabled = false;
                DiffSoupTriAssetRenderer diff = GetComponent<DiffSoupTriAssetRenderer>() ?? gameObject.AddComponent<DiffSoupTriAssetRenderer>();
                diff.AssetDirectory = asset; diff.TargetCamera = camera; active = diff;
            }
            else if (methodSpecific && (method == "mesh-splatting" || method == "2dts"))
            {
                if (triangle != null) triangle.enabled = false;
                MethodSpecificSplatRenderer splat = GetComponent<MethodSpecificSplatRenderer>() ?? gameObject.AddComponent<MethodSpecificSplatRenderer>();
                splat.AssetDirectory = asset; splat.TargetCamera = camera; active = splat;
            }
            else
            {
                if (triangle == null) triangle = gameObject.AddComponent<TriangleSplattingTriAssetRenderer>();
                triangle.AssetDirectory = asset; triangle.TargetCamera = camera; triangle.enabled = true; active = triangle;
            }

            ColmapBatchCapture capture = GetComponent<ColmapBatchCapture>();
            if (capture == null) capture = gameObject.AddComponent<ColmapBatchCapture>();
            capture.Renderer = active;
            capture.CaptureCamera = camera;
            capture.AutoStart = true;
        }
    }
}
