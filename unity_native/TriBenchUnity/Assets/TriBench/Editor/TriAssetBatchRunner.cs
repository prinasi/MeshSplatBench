using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace TriBench.UnityNative.Editor
{
    /// <summary>
    /// Headless entry point used by tools/run_unity_triasset_eval.py.  It keeps
    /// Unity's Metal device enabled (no -nographics), enters Play mode, then
    /// exits only after ColmapBatchCapture writes its completion marker.
    /// </summary>
    public static class TriAssetBatchRunner
    {
        const string ScenePath = "Assets/Scenes/TriBenchGarden.unity";
        static string completionPath;
        static double started;
        static bool playRequested;

        public static void Run()
        {
            string output = GetArgument("-output");
            if (String.IsNullOrWhiteSpace(output))
            {
                Debug.LogError("[TriBench] -output is required for batch evaluation.");
                EditorApplication.Exit(2);
                return;
            }
            completionPath = Path.Combine(output, ".unity_capture_complete");
            if (File.Exists(completionPath)) File.Delete(completionPath);
            EditorSceneManager.OpenScene(ScenePath);
            ConfigureRenderer(GetArgument("-method"));
            started = EditorApplication.timeSinceStartup;
            EditorApplication.update += Tick;
        }

        static void ConfigureRenderer(string method)
        {
            GameObject root = GameObject.Find("Garden - Triangle Splatting");
            if (root == null)
            {
                Debug.LogError("[TriBench] Scene renderer root is missing.");
                return;
            }

            // Remove all old renderers
            foreach (TriAssetRenderer old in root.GetComponents<TriAssetRenderer>())
                UnityEngine.Object.DestroyImmediate(old);

            bool methodSpecific = GetArgument("-method-specific") != null;
            bool standardMesh = GetArgument("-standard-mesh") != null;
            string triassetPath = GetArgument("-triasset");

            // Setup TriAssetLoader
            TriAssetLoader loader = root.GetComponent<TriAssetLoader>();
            if (loader == null) loader = root.AddComponent<TriAssetLoader>();
            loader.assetDirectory = triassetPath;
            loader.loadOnStart = true;  // Reload in Play mode (needed for no-domain-reload workflow)

            // Add appropriate renderer
            TriAssetRenderer renderer;
            if (standardMesh)
            {
                GeneralPurposeMeshTriAssetRenderer mesh = root.AddComponent<GeneralPurposeMeshTriAssetRenderer>();
                renderer = mesh;
                Debug.Log("[TriBench] Configured general-purpose indexed-Mesh baseline for " + method + ".");
            }
            else if (methodSpecific)
            {
                // Use tile-based renderer for high-fidelity method-aware rendering
                renderer = AddMethodSpecificTileRenderer(root, method);
                if (renderer != null && renderer is TriAssetTileRasterRenderer tileRenderer)
                {
                    tileRenderer.targetCamera = Camera.main;
                }
                Debug.Log("[TriBench] Configured method-aware tile-based renderer for " + method + ".");
            }
            else
            {
                Debug.LogError("[TriBench] No renderer condition specified (-standard-mesh or -method-specific).");
                return;
            }

            if (renderer == null)
            {
                Debug.LogError("[TriBench] Failed to create renderer for method: " + method);
                return;
            }

            // Load the asset (this will call Configure on the renderer)
            loader.Load();

            // Setup ColmapBatchCapture
            ColmapBatchCapture capture = root.GetComponent<ColmapBatchCapture>();
            if (capture == null) capture = root.AddComponent<ColmapBatchCapture>();
            capture.Loader = loader;
            capture.Renderer = renderer;
            capture.CaptureCamera = Camera.main;
            capture.AutoStart = true;

            // DON'T save the scene - let components persist only in memory for this play mode session
            // Saving triggers asset reimport which breaks the play mode transition
        }

        static void Tick()
        {
            if (!playRequested)
            {
                playRequested = true;
                Debug.Log("[TriBench] Requesting play mode...");
                EditorApplication.isPlaying = true;
                return;
            }
            if (File.Exists(completionPath))
            {
                if (EditorApplication.isPlaying)
                {
                    EditorApplication.isPlaying = false;
                    return;
                }
                Finish(0, "complete");
                return;
            }
            if (EditorApplication.timeSinceStartup - started > 3600.0)
                Finish(3, "timed out waiting for Unity capture");
        }

        static void Finish(int code, string message)
        {
            EditorApplication.update -= Tick;
            Debug.Log($"[TriBench] Batch evaluation {message}; exiting with {code}.");
            EditorApplication.Exit(code);
        }

        static string GetArgument(string name)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; ++i)
                if (String.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
            return null;
        }

        static TriAssetRenderer AddMethodSpecificTileRenderer(GameObject root, string method)
        {
            string methodLower = method.ToLowerInvariant();
            if (methodLower == "triangle-splatting")
                return root.AddComponent<TriangleSplattingTileRenderer>();
            else if (methodLower == "mesh-splatting")
                return root.AddComponent<MeshSplattingTileRenderer>();
            else if (methodLower == "2dts")
                return root.AddComponent<D2TSTileRenderer>();
            else if (methodLower == "diffsoup")
                return root.AddComponent<DiffSoupTileRenderer>();
            else
            {
                Debug.LogError("[TriBench] Unknown method for tile renderer: " + method);
                return null;
            }
        }
    }
}
