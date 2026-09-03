using System;
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
