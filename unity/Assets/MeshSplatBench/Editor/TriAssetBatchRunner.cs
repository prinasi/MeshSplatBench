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
            string trajectory = GetArgument("-video-trajectory") ?? GetArgument("-trajectory");
            bool isVideo = !string.IsNullOrEmpty(trajectory);
            completionPath = Path.Combine(output, isVideo ? ".unity_video_complete" : ".unity_capture_complete");
            if (File.Exists(completionPath)) File.Delete(completionPath);
            string altMarker = Path.Combine(output, isVideo ? ".unity_capture_complete" : ".unity_video_complete");
            if (File.Exists(altMarker)) File.Delete(altMarker);
            EditorSceneManager.OpenScene(ScenePath);
            ConfigureRenderer(GetArgument("-method"), isVideo, trajectory, output);
            started = EditorApplication.timeSinceStartup;
            EditorApplication.update += Tick;
        }

        public static void RunVideo()
        {
            Run();
        }

        static void ConfigureRenderer(string method, bool isVideo, string trajectory, string output)
        {
            bool methodSpecific = GetArgument("-method-specific") != null;
            if (methodSpecific && (String.Equals(method, "mesh-splatting", StringComparison.OrdinalIgnoreCase)
                || String.Equals(method, "2dts", StringComparison.OrdinalIgnoreCase)
                || String.Equals(method, "diffsoup", StringComparison.OrdinalIgnoreCase)))
            {
                ConfigureMethodSpecific(method, isVideo, trajectory, output);
                return;
            }
            bool standardMeshMethod = GetArgument("-standard-mesh") != null
                || String.Equals(method, "mesh-splatting", StringComparison.OrdinalIgnoreCase)
                || String.Equals(method, "2dts", StringComparison.OrdinalIgnoreCase)
                || String.Equals(method, "diffsoup", StringComparison.OrdinalIgnoreCase);
            if (!standardMeshMethod)
            {
                GameObject root = GameObject.Find("Garden - Triangle Splatting");
                if (root == null)
                {
                    Debug.LogError("[MeshSplatBench] Scene renderer root is missing.");
                    return;
                }
                TriangleSplattingTriAssetRenderer triangle = root.GetComponent<TriangleSplattingTriAssetRenderer>();
                if (triangle == null) triangle = root.AddComponent<TriangleSplattingTriAssetRenderer>();
                triangle.AssetDirectory = GetArgument("-triasset");
                triangle.TargetCamera = Camera.main;
                AttachCapture(root, triangle, isVideo, trajectory, output);
                Debug.Log("[MeshSplatBench] Configured Triangle Splatting renderer for " + method + (isVideo ? " (video trajectory)" : "") + ".");
                return;
            }
            GameObject meshRoot = GameObject.Find("Garden - Triangle Splatting");
            if (meshRoot == null)
            {
                Debug.LogError("[MeshSplatBench] Scene renderer root is missing.");
                return;
            }
            TriangleSplattingTriAssetRenderer oldTriangle = meshRoot.GetComponent<TriangleSplattingTriAssetRenderer>();
            if (oldTriangle != null) UnityEngine.Object.DestroyImmediate(oldTriangle);
            StandardMeshTriAssetRenderer mesh = meshRoot.GetComponent<StandardMeshTriAssetRenderer>();
            if (mesh == null) mesh = meshRoot.AddComponent<StandardMeshTriAssetRenderer>();
            mesh.AssetDirectory = GetArgument("-triasset");
            mesh.TargetCamera = Camera.main;
            AttachCapture(meshRoot, mesh, isVideo, trajectory, output);
            Debug.Log("[MeshSplatBench] Configured standard indexed-Mesh baseline for " + method + (isVideo ? " (video trajectory)" : "") + ".");
        }

        static void ConfigureMethodSpecific(string method, bool isVideo, string trajectory, string output)
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
            AttachCapture(root, renderer, isVideo, trajectory, output);
            Debug.Log("[MeshSplatBench] Configured method-specific Unity portability renderer for " + method + (isVideo ? " (video trajectory)" : "") + ".");
        }

        static void AttachCapture(GameObject root, TriAssetRenderer renderer, bool isVideo, string trajectory, string output)
        {
            if (isVideo)
            {
                ColmapBatchCapture oldCapture = root.GetComponent<ColmapBatchCapture>();
                if (oldCapture != null) UnityEngine.Object.DestroyImmediate(oldCapture);
                TriAssetVideoBatchCapture videoCapture = root.GetComponent<TriAssetVideoBatchCapture>();
                if (videoCapture == null) videoCapture = root.AddComponent<TriAssetVideoBatchCapture>();
                videoCapture.Renderer = renderer;
                videoCapture.CaptureCamera = Camera.main;
                videoCapture.TrajectoryPath = trajectory;
                videoCapture.OutputRoot = output;
                videoCapture.AutoStart = true;
            }
            else
            {
                TriAssetVideoBatchCapture oldVideo = root.GetComponent<TriAssetVideoBatchCapture>();
                if (oldVideo != null) UnityEngine.Object.DestroyImmediate(oldVideo);
                ColmapBatchCapture capture = root.GetComponent<ColmapBatchCapture>();
                if (capture == null) capture = root.AddComponent<ColmapBatchCapture>();
                capture.Renderer = renderer;
                capture.CaptureCamera = Camera.main;
                capture.AutoStart = true;
            }
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
