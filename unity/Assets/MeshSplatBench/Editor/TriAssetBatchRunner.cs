using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace MeshSplatBench.UnityNative.Editor
{
    /// <summary>
    /// Headless entry point used by tools/run_unity_triasset_eval.py.  It keeps
    /// Unity's Metal device enabled (no -nographics), enters Play mode, then
    /// exits only after ColmapBatchCapture writes its completion marker.
    /// </summary>
    public static class TriAssetBatchRunner
    {
        const string ScenePath = "Assets/Scenes/MeshSplatBenchGarden.unity";
        static string completionPath;
        static double started;
        static bool playRequested;

        public static void Run()
        {
            string output = GetArgument("-output");
            if (String.IsNullOrWhiteSpace(output))
            {
                Debug.LogError("[MeshSplatBench] -output is required for batch evaluation.");
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
            bool methodSpecific = GetArgument("-method-specific") != null;
            if (methodSpecific && (String.Equals(method, "mesh-splatting", StringComparison.OrdinalIgnoreCase)
                || String.Equals(method, "2dts", StringComparison.OrdinalIgnoreCase)
                || String.Equals(method, "diffsoup", StringComparison.OrdinalIgnoreCase)))
            {
                ConfigureMethodSpecific(method);
                return;
            }
            bool standardMeshMethod = GetArgument("-standard-mesh") != null
                || String.Equals(method, "mesh-splatting", StringComparison.OrdinalIgnoreCase)
                || String.Equals(method, "2dts", StringComparison.OrdinalIgnoreCase)
                || String.Equals(method, "diffsoup", StringComparison.OrdinalIgnoreCase);
            if (!standardMeshMethod) return;
            GameObject root = GameObject.Find("Garden - Triangle Splatting");
            if (root == null)
            {
                Debug.LogError("[MeshSplatBench] Scene renderer root is missing.");
                return;
            }
            TriangleSplattingTriAssetRenderer triangle = root.GetComponent<TriangleSplattingTriAssetRenderer>();
            if (triangle != null) UnityEngine.Object.DestroyImmediate(triangle);
            StandardMeshTriAssetRenderer mesh = root.GetComponent<StandardMeshTriAssetRenderer>();
            if (mesh == null) mesh = root.AddComponent<StandardMeshTriAssetRenderer>();
            mesh.AssetDirectory = GetArgument("-triasset");
            mesh.TargetCamera = Camera.main;
            ColmapBatchCapture capture = root.GetComponent<ColmapBatchCapture>();
            if (capture == null) capture = root.AddComponent<ColmapBatchCapture>();
            capture.Renderer = mesh;
            capture.CaptureCamera = Camera.main;
            capture.AutoStart = true;
            Debug.Log("[MeshSplatBench] Configured standard indexed-Mesh baseline for " + method + ".");
        }

        static void ConfigureMethodSpecific(string method)
        {
            GameObject root = GameObject.Find("Garden - Triangle Splatting");
            if (root == null) { Debug.LogError("[MeshSplatBench] Scene renderer root is missing."); return; }
            TriangleSplattingTriAssetRenderer triangle = root.GetComponent<TriangleSplattingTriAssetRenderer>();
            if (triangle != null) UnityEngine.Object.DestroyImmediate(triangle);
            StandardMeshTriAssetRenderer mesh = root.GetComponent<StandardMeshTriAssetRenderer>();
            if (mesh != null) UnityEngine.Object.DestroyImmediate(mesh);
            TriAssetRenderer renderer;
            if (String.Equals(method, "diffsoup", StringComparison.OrdinalIgnoreCase))
            {
                MethodSpecificSplatRenderer old = root.GetComponent<MethodSpecificSplatRenderer>();
                if (old != null) UnityEngine.Object.DestroyImmediate(old);
                DiffSoupTriAssetRenderer diff = root.GetComponent<DiffSoupTriAssetRenderer>();
                if (diff == null) diff = root.AddComponent<DiffSoupTriAssetRenderer>();
                diff.AssetDirectory = GetArgument("-triasset"); diff.TargetCamera = Camera.main; renderer = diff;
            }
            else
            {
                DiffSoupTriAssetRenderer old = root.GetComponent<DiffSoupTriAssetRenderer>();
                if (old != null) UnityEngine.Object.DestroyImmediate(old);
                MethodSpecificSplatRenderer splat = root.GetComponent<MethodSpecificSplatRenderer>();
                if (splat == null) splat = root.AddComponent<MethodSpecificSplatRenderer>();
                splat.AssetDirectory = GetArgument("-triasset"); splat.TargetCamera = Camera.main; renderer = splat;
            }
            ColmapBatchCapture capture = root.GetComponent<ColmapBatchCapture>();
            if (capture == null) capture = root.AddComponent<ColmapBatchCapture>();
            capture.Renderer = renderer; capture.CaptureCamera = Camera.main; capture.AutoStart = true;
            Debug.Log("[MeshSplatBench] Configured method-specific Unity portability renderer for " + method + ".");
        }

        static void Tick()
        {
            if (!playRequested)
            {
                playRequested = true;
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
            Debug.Log($"[MeshSplatBench] Batch evaluation {message}; exiting with {code}.");
            EditorApplication.Exit(code);
        }

        static string GetArgument(string name)
        {
            string[] args = Environment.GetCommandLineArgs();
            for (int i = 0; i + 1 < args.Length; ++i)
                if (String.Equals(args[i], name, StringComparison.OrdinalIgnoreCase)) return args[i + 1];
            return null;
        }
    }
}
