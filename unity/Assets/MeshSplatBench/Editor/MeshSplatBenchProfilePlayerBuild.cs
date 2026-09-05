using System;
using UnityEditor;
using UnityEditor.Build.Reporting;

namespace MeshSplatBench.UnityNative.Editor
{
    public static class MeshSplatBenchProfilePlayerBuild
    {
        public static void Build()
        {
            string output = GetArgument("-profile-player-output");
            if (String.IsNullOrWhiteSpace(output))
            {
                UnityEngine.Debug.LogError("[MeshSplatBench] -profile-player-output is required.");
                EditorApplication.Exit(2); return;
            }
            BuildTarget target = ParseTarget(GetArgument("-profile-player-target"));
            BuildPlayerOptions options = new BuildPlayerOptions {
                scenes = new[] { "Assets/Scenes/MeshSplatBenchGarden.unity" },
                locationPathName = output,
                target = target,
                // MeshSplatBench Linux standalone profile player patch.
                options = BuildOptions.Development,
            };
            BuildReport report = BuildPipeline.BuildPlayer(options);
            int code = report.summary.result == BuildResult.Succeeded ? 0 : 1;
            UnityEngine.Debug.Log($"[MeshSplatBench] Profile Player build {report.summary.result}: {output}");
            EditorApplication.Exit(code);
        }

        static BuildTarget ParseTarget(string value)
        {
            string normalized = String.IsNullOrWhiteSpace(value) ? "linux64" : value.Trim().ToLowerInvariant();
            switch (normalized)
            {
                case "linux":
                case "linux64":
                case "standalonelinux64":
                    return BuildTarget.StandaloneLinux64;
                case "mac":
                case "macos":
                case "osx":
                case "standaloneosx":
                    return BuildTarget.StandaloneOSX;
                case "win":
                case "windows":
                case "windows64":
                case "standalonewindows64":
                    return BuildTarget.StandaloneWindows64;
                default:
                    throw new ArgumentException("Unsupported MeshSplatBench profile player target: " + value);
            }
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
