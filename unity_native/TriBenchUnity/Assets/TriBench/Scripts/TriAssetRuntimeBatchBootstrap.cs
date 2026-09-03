using System;
using UnityEngine;

namespace TriBench.UnityNative
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
            Debug.Log($"[TriBench] Standalone profile bootstrap: method={method} asset={asset} standard={standard} methodSpecific={methodSpecific}");
            // Headless X servers (Xvfb) report a 0 Hz refresh rate; a vsync'd
            // first present can block forever before any coroutine runs.
            // Disable vsync before the first frame instead of waiting for the
            // profile path to do it.
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
            Camera camera = Camera.main;
            if (camera != null)
            {
                Debug.Log("[TriBench] Standalone profile bootstrap: disabling legacy scene camera component before runtime control");
                camera.enabled = false;
            }

            // Remove all old renderers
            foreach (TriAssetRenderer old in GetComponents<TriAssetRenderer>())
                Destroy(old);

            // Setup TriAssetLoader
            TriAssetLoader loader = GetComponent<TriAssetLoader>();
            if (loader == null) loader = gameObject.AddComponent<TriAssetLoader>();
            loader.assetDirectory = asset;
            loader.loadOnStart = false;

            // Add appropriate renderer
            TriAssetRenderer active;
            if (standard)
            {
                GeneralPurposeMeshTriAssetRenderer mesh = gameObject.AddComponent<GeneralPurposeMeshTriAssetRenderer>();
                active = mesh;
            }
            else if (methodSpecific)
            {
                active = AddMethodSpecificTileRenderer(method);
                if (active != null && active is TriAssetTileRasterRenderer tileRenderer)
                {
                    tileRenderer.targetCamera = camera;
                }
            }
            else
            {
                Debug.LogError("[TriBench] No renderer condition specified.");
                return;
            }

            if (active == null)
            {
                Debug.LogError("[TriBench] Failed to create renderer for method: " + method);
                return;
            }

            // Load the asset (this will call Configure on the renderer)
            loader.Load();

            Debug.Log($"[TriBench] Standalone profile renderer selected: type={active.GetType().Name} camera={(camera == null ? "<null>" : camera.name)}");
            ColmapBatchCapture capture = GetComponent<ColmapBatchCapture>();
            if (capture == null) capture = gameObject.AddComponent<ColmapBatchCapture>();
            capture.Loader = loader;
            capture.Renderer = active;
            capture.CaptureCamera = camera;
            capture.AutoStart = true;
            Debug.Log($"[TriBench] Standalone profile capture configured: renderer={capture.Renderer.GetType().Name} autoStart={capture.AutoStart} output={capture.OutputRoot}");
        }

        TriAssetRenderer AddMethodSpecificTileRenderer(string method)
        {
            string methodLower = method.ToLowerInvariant();
            if (methodLower == "triangle-splatting")
                return gameObject.AddComponent<TriangleSplattingTileRenderer>();
            else if (methodLower == "mesh-splatting")
                return gameObject.AddComponent<MeshSplattingTileRenderer>();
            else if (methodLower == "2dts")
                return gameObject.AddComponent<D2TSTileRenderer>();
            else if (methodLower == "diffsoup")
                return gameObject.AddComponent<DiffSoupTileRenderer>();
            else
            {
                Debug.LogError("[TriBench] Unknown method for tile renderer: " + method);
                return null;
            }
        }
    }
}
