using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Profiling;
using Unity.Profiling;

namespace TriBench.UnityNative
{
    /// <summary>
    /// Captures COLMAP views using the same ordering and LLFF holdout
    /// convention as the upstream triangle-splatting loader (every eighth sorted
    /// image is a test view). PNG output is for quality evaluation, not FPS timing.
    /// </summary>
    [DefaultExecutionOrder(-10000)]
    public sealed class ColmapBatchCapture : MonoBehaviour
    {
        public TriAssetLoader Loader;
        public TriAssetRenderer Renderer;
        public Camera CaptureCamera;
        public bool AutoStart = true;
        public string DatasetRoot = "/Volumes/GLOWAY/Datasets/MipNeRF360/garden";
        public string OutputRoot = "/Volumes/GLOWAY/Unity/unity_native/TriBenchUnity/renders/garden_colmap_images_4";
        public int OutputWidth = 1297;
        public int OutputHeight = 840;
        public int Holdout = 8;
        [Tooltip("Capture only held-out views. FPS is always sampled on held-out views.")]
        public bool TestOnly;
        [Tooltip("Warmup engine frames per held-out view before timing. PNG readback is outside this interval.")]
        public int FpsWarmupFrames = 3;
        [Tooltip("Timed engine frames per held-out view. The benchmark uses no ReadPixels, EncodeToPNG, or disk writes.")]
        public int FpsTimedFrames = 12;
        [Tooltip("Run a no-I/O deployment profile instead of writing PNGs.")]
        public bool ProfileOnly;
        [Tooltip("Number of evenly spaced held-out cameras used by a profiling run.")]
        public int ProfileViewCount = 3;
        [Tooltip("Warmup frames per profiled camera.")]
        public int ProfileWarmupFrames = 60;
        [Tooltip("Timed frames per profiled camera.")]
        public int ProfileTimedFrames = 180;
        [Tooltip("Independent process-run identifier supplied by the supervisor.")]
        public int ProfileRunId;

        sealed class Intrinsics
        {
            public ulong Width, Height;
            public double Fx, Fy, Cx, Cy;
        }

        sealed class View
        {
            public string Name;
            public double[] R;
            public double[] T;
            public int CameraId;
        }

        bool running;

        [Serializable]
        sealed class ViewProfile
        {
            public string image;
            public int colmap_index;
            public int width, height;
            public int warmup_frames, timed_frames;
            public string cpu_source;
            public int cpu_samples, gpu_samples, draw_call_samples;
            public string draw_call_source;
            public float cpu_mean_ms, cpu_p50_ms, cpu_p95_ms;
            public float engine_mean_ms, engine_p50_ms, engine_p95_ms;
            public float gpu_mean_ms, gpu_p50_ms, gpu_p95_ms;
            public float draw_calls_mean, draw_calls_p50, draw_calls_p95;
            // Kept for exact pooled quantiles and run-to-run uncertainty in
            // the offline report; no readback or file I/O occurs while timing.
            public List<float> cpu_frame_samples_ms;
            public List<float> engine_frame_samples_ms;
            public List<float> gpu_frame_samples_ms;
            public List<float> draw_call_samples_values;
        }

        [Serializable]
        sealed class RuntimeProfile
        {
            public string schema_version = "1.0";
            public int run_id;
            public string renderer_type;
            public string asset_directory;
            public long asset_bytes;
            public float load_to_ready_ms;
            public long graphics_driver_allocated_bytes;
            public long unity_total_allocated_bytes;
            public long unity_total_reserved_bytes;
            public bool frame_timing_available;
            public bool draw_call_counter_available;
            public List<ViewProfile> views = new List<ViewProfile>();
        }

        void Awake()
        {
            DatasetRoot = TriAssetRuntimeOptions.Get("-dataset", DatasetRoot);
            OutputRoot = TriAssetRuntimeOptions.Get("-output", OutputRoot);
            OutputWidth = TriAssetRuntimeOptions.GetInt("-triasset-width", OutputWidth);
            OutputHeight = TriAssetRuntimeOptions.GetInt("-triasset-height", OutputHeight);
            FpsWarmupFrames = TriAssetRuntimeOptions.GetInt("-fps-warmup", FpsWarmupFrames);
            FpsTimedFrames = TriAssetRuntimeOptions.GetInt("-fps-frames", FpsTimedFrames);
            TestOnly = TriAssetRuntimeOptions.Has("-test-only");
            ProfileOnly = TriAssetRuntimeOptions.Has("-profile-only");
            ProfileViewCount = TriAssetRuntimeOptions.GetInt("-profile-views", ProfileViewCount);
            ProfileWarmupFrames = TriAssetRuntimeOptions.GetInt("-profile-warmup", ProfileWarmupFrames);
            ProfileTimedFrames = TriAssetRuntimeOptions.GetInt("-profile-frames", ProfileTimedFrames);
            ProfileRunId = TriAssetRuntimeOptions.GetInt("-profile-run", ProfileRunId);
        }

        IEnumerator Start()
        {
            if (AutoStart) yield return CaptureAll();
        }

        [ContextMenu("Capture all COLMAP views")]
        public void CaptureAllFromContextMenu()
        {
            if (!running) StartCoroutine(CaptureAll());
        }

        IEnumerator CaptureAll()
        {
            running = true;
            double loadStart = Time.realtimeSinceStartupAsDouble;
            string completionPath = Path.Combine(OutputRoot, ".unity_capture_complete");
            if (File.Exists(completionPath)) File.Delete(completionPath);
            if (Loader == null) Loader = FindFirstObjectByType<TriAssetLoader>();
            if (Renderer == null) Renderer = FindFirstObjectByType<TriAssetRenderer>();
            if (CaptureCamera == null) CaptureCamera = Camera.main;
            if (Loader == null || Renderer == null || CaptureCamera == null)
            {
                Debug.LogError("[TriBench] Batch capture requires the loader, renderer and target camera.");
                yield break;
            }

            // Wait for loader to be ready (it has completed loading)
            if (Loader.PrimitiveCount <= 0) Loader.Load();
            while (Loader.PrimitiveCount <= 0) yield return null;
            float loadToReadyMs = (float)((Time.realtimeSinceStartupAsDouble - loadStart) * 1000.0);
            Dictionary<int, Intrinsics> intrinsics = ReadCameras(Path.Combine(DatasetRoot, "sparse/0/cameras.bin"));
            List<View> views = ReadImages(Path.Combine(DatasetRoot, "sparse/0/images.bin"));
            views.Sort((a, b) => StringComparer.Ordinal.Compare(Path.GetFileNameWithoutExtension(a.Name), Path.GetFileNameWithoutExtension(b.Name)));

            if (ProfileOnly)
            {
                yield return ProfileRuntime(views, intrinsics, loadToReadyMs);
                File.WriteAllText(completionPath, "complete\n");
                running = false;
                // A standalone Player must exit itself.  Killing it as soon
                // as its marker appears can leave a stale macOS application
                // registration that makes a subsequent independent run abort.
                if (!Application.isEditor) Application.Quit(0);
                yield break;
            }

            Directory.CreateDirectory(Path.Combine(OutputRoot, "train"));
            Directory.CreateDirectory(Path.Combine(OutputRoot, "test"));
            StringBuilder manifest = new StringBuilder("split,index,image,render\n");
            StringBuilder fpsCsv = new StringBuilder("split,index,image,render_width,render_height,warmup_frames,timed_frames,mean_frame_ms,p50_frame_ms,p95_frame_ms,fps\n");

            // Native triangle-splatting trains and evaluates in raw PIL RGB/255
            // code values.  An sRGB target would encode the shader output a
            // second time in a Linear Unity project and invalidate PSNR.
            RenderTexture target = new RenderTexture(OutputWidth, OutputHeight, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            target.Create();
            Texture2D readback = new Texture2D(OutputWidth, OutputHeight, TextureFormat.RGBA32, false, true);
            bool wasEnabled = CaptureCamera.enabled;
            RenderTexture previousTarget = CaptureCamera.targetTexture;
            CaptureCamera.enabled = false;
            CaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            CaptureCamera.backgroundColor = new Color(0f, 0f, 0f, 0f);
            CaptureCamera.targetTexture = target;
            Renderer.SetOutputRawCodeValues(true);
            RenderTexture fpsTarget = new RenderTexture(OutputWidth, OutputHeight, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            fpsTarget.Create();

            for (int i = 0; i < views.Count; ++i)
            {
                View view = views[i];
                if (!intrinsics.TryGetValue(view.CameraId, out Intrinsics intr))
                {
                    Debug.LogError("[TriBench] Missing intrinsics for camera " + view.CameraId);
                    continue;
                }
                ApplyColmapPose(CaptureCamera, view, intr);
                Renderer.PrepareCamera(CaptureCamera);
                string split = i % Holdout == 0 ? "test" : "train";
                if (TestOnly && split != "test") continue;
                string stem = Path.GetFileNameWithoutExtension(view.Name);
                string renderName = stem + ".png";
                if (split == "test")
                    yield return BenchmarkView(view, intr, fpsTarget, fpsCsv, i, split, renderName);
                CaptureCamera.targetTexture = target;
                PreparePremultipliedCaptureTarget();
                CaptureCamera.Render();
                RenderTexture.active = target;
                readback.ReadPixels(new Rect(0, 0, OutputWidth, OutputHeight), 0, 0, false);
                readback.Apply(false, false);
                CompositePremultipliedBackground(readback);

                File.WriteAllBytes(Path.Combine(OutputRoot, split, renderName), ImageConversion.EncodeToPNG(readback));
                manifest.Append(split).Append(',').Append(i).Append(',').Append(view.Name).Append(',').Append(split).Append('/').Append(renderName).Append('\n');
                Debug.Log($"[TriBench] Captured {i + 1}/{views.Count}: {split}/{renderName}");
                yield return null;
            }

            RenderTexture.active = null;
            CaptureCamera.targetTexture = previousTarget;
            CaptureCamera.enabled = wasEnabled;
            Renderer.SetOutputRawCodeValues(false);
            target.Release();
            Destroy(target);
            fpsTarget.Release();
            Destroy(fpsTarget);
            Destroy(readback);
            File.WriteAllText(Path.Combine(OutputRoot, "manifest.csv"), manifest.ToString());
            File.WriteAllText(Path.Combine(OutputRoot, "fps_per_test_view.csv"), fpsCsv.ToString());
            File.WriteAllText(completionPath, "complete\n");
            Debug.Log($"[TriBench] COLMAP capture complete: {views.Count} views at {OutputWidth}x{OutputHeight}; output={OutputRoot}");
            running = false;
            if (!Application.isEditor) Application.Quit(0);
        }

        IEnumerator ProfileRuntime(List<View> views, Dictionary<int, Intrinsics> intrinsics, float loadToReadyMs)
        {
            List<View> heldout = new List<View>();
            for (int i = 0; i < views.Count; ++i)
                if (i % Holdout == 0) heldout.Add(views[i]);
            if (heldout.Count == 0) throw new InvalidOperationException("No held-out views available for profiling.");

            int sampleCount = Mathf.Clamp(ProfileViewCount, 1, heldout.Count);
            List<View> selected = new List<View>();
            for (int i = 0; i < sampleCount; ++i)
            {
                int index = sampleCount == 1 ? heldout.Count / 2 : Mathf.RoundToInt(i * (heldout.Count - 1f) / (sampleCount - 1f));
                selected.Add(heldout[index]);
            }

            RuntimeProfile report = new RuntimeProfile {
                run_id = ProfileRunId,
                renderer_type = Renderer.GetType().Name,
                asset_directory = Loader.assetDirectory,
                asset_bytes = DirectoryBytes(Loader.assetDirectory),
                load_to_ready_ms = loadToReadyMs,
            };
            // On Apple Silicon this is a graphics-driver allocation on unified
            // memory, not a physically separate VRAM reading.
            report.graphics_driver_allocated_bytes = Profiler.GetAllocatedMemoryForGraphicsDriver();
            report.unity_total_allocated_bytes = Profiler.GetTotalAllocatedMemoryLong();
            report.unity_total_reserved_bytes = Profiler.GetTotalReservedMemoryLong();

            RenderTexture previousTarget = CaptureCamera.targetTexture;
            bool wasEnabled = CaptureCamera.enabled;
            CaptureCamera.enabled = false;
            CaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            CaptureCamera.backgroundColor = new Color(0f, 0f, 0f, 0f);
            RenderTexture target = new RenderTexture(OutputWidth, OutputHeight, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            target.Create();
            CaptureCamera.targetTexture = target;
            Renderer.SetOutputRawCodeValues(true);
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;

            ProfilerRecorder drawCalls = ProfilerRecorder.StartNew(ProfilerCategory.Render, "Draw Calls Count", 1);
            FrameTimingManager.CaptureFrameTimings();
            for (int i = 0; i < selected.Count; ++i)
            {
                View view = selected[i];
                if (!intrinsics.TryGetValue(view.CameraId, out Intrinsics intr)) continue;
                int colmapIndex = views.IndexOf(view);
                ViewProfile result = null;
                yield return ProfileView(view, intr, colmapIndex, target, drawCalls, value => result = value);
                if (result != null)
                {
                    report.views.Add(result);
                    report.frame_timing_available |= result.gpu_samples > 0;
                    report.draw_call_counter_available |= result.draw_call_source == "ProfilerRecorder.Draw Calls Count";
                }
            }
            drawCalls.Dispose();
            CaptureCamera.enabled = false;
            CaptureCamera.targetTexture = previousTarget;
            CaptureCamera.enabled = wasEnabled;
            target.Release();
            Destroy(target);
            Renderer.SetOutputRawCodeValues(false);
            string path = Path.Combine(OutputRoot, $"runtime_profile_run_{ProfileRunId:D2}.json");
            File.WriteAllText(path, JsonUtility.ToJson(report, true));
            Debug.Log($"[TriBench] Runtime profile saved: {path}; GPU FrameTiming={report.frame_timing_available}.");
        }

        IEnumerator ProfileView(View view, Intrinsics intr, int colmapIndex, RenderTexture target, ProfilerRecorder drawCalls, Action<ViewProfile> done)
        {
            ApplyColmapPose(CaptureCamera, view, intr);
                Renderer.PrepareCamera(CaptureCamera);
            CaptureCamera.targetTexture = target;
            CaptureCamera.enabled = true;
            for (int i = 0; i < ProfileWarmupFrames; ++i)
            {
                FrameTimingManager.CaptureFrameTimings();
                yield return null;
            }
            List<float> engine = new List<float>();
            List<float> cpu = new List<float>();
            List<float> gpu = new List<float>();
            List<float> draws = new List<float>();
            FrameTiming[] timing = new FrameTiming[8];
            for (int i = 0; i < ProfileTimedFrames; ++i)
            {
                FrameTimingManager.CaptureFrameTimings();
                yield return null;
                engine.Add(Time.unscaledDeltaTime * 1000.0f);
                // The GPU timestamp readback for a captured frame lands
                // asynchronously.  Scan a window of recent captures and retry
                // briefly until one of them has landed; this keeps the GPU
                // sample set complete even when the GPU is saturated.
                uint count = 0;
                float gpuMs = 0.0f;
                for (int attempt = 0; attempt < 16 && gpuMs <= 0.0f; ++attempt)
                {
                    count = FrameTimingManager.GetLatestTimings((uint)timing.Length, timing);
                    for (int k = 0; k < count; ++k)
                    {
                        if (timing[k].gpuFrameTime > 0.0)
                        {
                            gpuMs = (float)timing[k].gpuFrameTime;
                            break;
                        }
                    }
                    if (gpuMs <= 0.0f && attempt < 15) System.Threading.Thread.Sleep(1);
                }
                if (gpuMs > 0.0) gpu.Add(gpuMs);
                if (count > 0 && timing[0].cpuFrameTime > 0.0) cpu.Add((float)timing[0].cpuFrameTime);
                if (drawCalls.Valid && drawCalls.LastValue > 0) draws.Add(drawCalls.LastValue);
            }
            CaptureCamera.enabled = false;
            List<float> selectedCpu = cpu.Count > 0 ? cpu : engine;
            bool profiledDrawCalls = draws.Count > 0;
            // Unity's draw-call counter is unavailable for macOS Metal. All
            // TriBench evaluation renderers submit exactly one material/pass
            // draw per frame (MeshRenderer or DrawProceduralNow); retain this
            // separately labelled deterministic count rather than pretending
            // it was a GPU-profiler sample.
            if (!profiledDrawCalls)
                for (int i = 0; i < ProfileTimedFrames; ++i) draws.Add(1f);
            done(new ViewProfile {
                image = view.Name,
                colmap_index = colmapIndex,
                width = OutputWidth, height = OutputHeight,
                warmup_frames = ProfileWarmupFrames, timed_frames = ProfileTimedFrames,
                cpu_source = cpu.Count > 0 ? "FrameTiming.cpuFrameTime" : "Time.unscaledDeltaTime_fallback",
                draw_call_source = profiledDrawCalls ? "ProfilerRecorder.Draw Calls Count" : "configured_one_pass_submission",
                cpu_samples = selectedCpu.Count, gpu_samples = gpu.Count, draw_call_samples = draws.Count,
                cpu_mean_ms = Mean(selectedCpu), cpu_p50_ms = Quantile(selectedCpu, 0.5f), cpu_p95_ms = Quantile(selectedCpu, 0.95f),
                engine_mean_ms = Mean(engine), engine_p50_ms = Quantile(engine, 0.5f), engine_p95_ms = Quantile(engine, 0.95f),
                gpu_mean_ms = Mean(gpu), gpu_p50_ms = Quantile(gpu, 0.5f), gpu_p95_ms = Quantile(gpu, 0.95f),
                draw_calls_mean = Mean(draws), draw_calls_p50 = Quantile(draws, 0.5f), draw_calls_p95 = Quantile(draws, 0.95f),
                cpu_frame_samples_ms = selectedCpu,
                engine_frame_samples_ms = engine,
                gpu_frame_samples_ms = gpu,
                draw_call_samples_values = draws,
            });
        }

        IEnumerator BenchmarkView(View view, Intrinsics intr, RenderTexture target, StringBuilder csv, int index, string split, string renderName)
        {
            ApplyColmapPose(CaptureCamera, view, intr);
                Renderer.PrepareCamera(CaptureCamera);
            CaptureCamera.targetTexture = target;
            CaptureCamera.enabled = true;
            // WaitForEndOfFrame is not resumed in Unity batch mode.  A normal
            // frame yield still renders the enabled camera and works both in
            // the Editor and in the headless evaluator.
            for (int i = 0; i < FpsWarmupFrames; ++i) yield return null;
            float[] samples = new float[FpsTimedFrames];
            for (int i = 0; i < FpsTimedFrames; ++i)
            {
                yield return null;
                samples[i] = Time.unscaledDeltaTime * 1000.0f;
            }
            CaptureCamera.enabled = false;
            CaptureCamera.targetTexture = target;
            Array.Sort(samples);
            float sum = 0;
            for (int i = 0; i < samples.Length; ++i) sum += samples[i];
            float mean = sum / samples.Length;
            float p50 = Percentile(samples, 0.50f);
            float p95 = Percentile(samples, 0.95f);
            float fps = mean > 0 ? 1000.0f / mean : 0;
            csv.Append(split).Append(',').Append(index).Append(',').Append(view.Name).Append(',')
                .Append(OutputWidth).Append(',').Append(OutputHeight).Append(',').Append(FpsWarmupFrames).Append(',').Append(FpsTimedFrames).Append(',')
                .Append(mean.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)).Append(',')
                .Append(p50.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)).Append(',')
                .Append(p95.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)).Append(',')
                .Append(fps.ToString("F6", System.Globalization.CultureInfo.InvariantCulture)).Append('\n');
            Debug.Log($"[TriBench] FPS sample {view.Name}: {fps:F2} FPS, {mean:F2} ms; capture IO excluded.");
        }

        static float Percentile(float[] sorted, float percentile)
        {
            if (sorted.Length == 0) return 0;
            float pos = (sorted.Length - 1) * percentile;
            int lo = Mathf.FloorToInt(pos), hi = Mathf.CeilToInt(pos);
            return Mathf.Lerp(sorted[lo], sorted[hi], pos - lo);
        }

        static float Mean(List<float> values)
        {
            if (values == null || values.Count == 0) return 0f;
            double sum = 0; for (int i = 0; i < values.Count; ++i) sum += values[i];
            return (float)(sum / values.Count);
        }

        static float Quantile(List<float> values, float percentile)
        {
            if (values == null || values.Count == 0) return 0f;
            List<float> sorted = new List<float>(values); sorted.Sort();
            float pos = (sorted.Count - 1) * percentile;
            int lo = Mathf.FloorToInt(pos), hi = Mathf.CeilToInt(pos);
            return Mathf.Lerp(sorted[lo], sorted[hi], pos - lo);
        }

        static long DirectoryBytes(string root)
        {
            if (String.IsNullOrWhiteSpace(root) || !Directory.Exists(root)) return 0;
            long total = 0;
            foreach (string file in Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories))
                total += new FileInfo(file).Length;
            return total;
        }

        void ApplyColmapPose(Camera camera, View view, Intrinsics intr)
        {
            double[] r = view.R;
            double[] t = view.T;
            Vector3 position = new Vector3(
                (float)-(r[0] * t[0] + r[3] * t[1] + r[6] * t[2]),
                (float)-(r[1] * t[0] + r[4] * t[1] + r[7] * t[2]),
                (float)-(r[2] * t[0] + r[5] * t[1] + r[8] * t[2]));
            Vector3 right = new Vector3((float)r[0], (float)r[1], (float)r[2]).normalized;
            Vector3 down = new Vector3((float)r[3], (float)r[4], (float)r[5]).normalized;
            Vector3 forward = new Vector3((float)r[6], (float)r[7], (float)r[8]).normalized;
            camera.transform.SetPositionAndRotation(position, Quaternion.LookRotation(forward, -down));
            camera.ResetProjectionMatrix();
            camera.aspect = (float)OutputWidth / OutputHeight;
            camera.fieldOfView = (float)(2.0 * Math.Atan(intr.Height / (2.0 * intr.Fy)) * Mathf.Rad2Deg);
            camera.nearClipPlane = 0.01f;
            camera.farClipPlane = 300f;
            // COLMAP uses a +X-right, +Y-down image basis.  The Unity camera
            // transform above has the correct pose but its view basis is a
            // reflection of the COLMAP raster basis, so carry that reflection
            // explicitly in the projection instead of mirroring pixels after
            // rendering.
            Matrix4x4 projection = camera.projectionMatrix;
            projection.m00 = -projection.m00;
            camera.projectionMatrix = projection;
        }

        static Dictionary<int, Intrinsics> ReadCameras(string path)
        {
            using BinaryReader reader = new BinaryReader(File.OpenRead(path));
            ulong count = reader.ReadUInt64();
            Dictionary<int, Intrinsics> result = new Dictionary<int, Intrinsics>();
            for (ulong i = 0; i < count; ++i)
            {
                int id = reader.ReadInt32();
                int model = reader.ReadInt32();
                ulong width = reader.ReadUInt64();
                ulong height = reader.ReadUInt64();
                int parameterCount = CameraParameterCount(model);
                double[] p = new double[parameterCount];
                for (int k = 0; k < parameterCount; ++k) p[k] = reader.ReadDouble();
                if (model == 0) result[id] = new Intrinsics { Width = width, Height = height, Fx = p[0], Fy = p[0], Cx = p[1], Cy = p[2] };
                else if (model == 1) result[id] = new Intrinsics { Width = width, Height = height, Fx = p[0], Fy = p[1], Cx = p[2], Cy = p[3] };
                else throw new InvalidDataException("Unsupported COLMAP camera model " + model + ". Use undistorted PINHOLE inputs.");
            }
            return result;
        }

        static List<View> ReadImages(string path)
        {
            using BinaryReader reader = new BinaryReader(File.OpenRead(path));
            ulong count = reader.ReadUInt64();
            List<View> result = new List<View>((int)count);
            for (ulong i = 0; i < count; ++i)
            {
                reader.ReadInt32();
                double qw = reader.ReadDouble(), qx = reader.ReadDouble(), qy = reader.ReadDouble(), qz = reader.ReadDouble();
                double tx = reader.ReadDouble(), ty = reader.ReadDouble(), tz = reader.ReadDouble();
                int cameraId = reader.ReadInt32();
                List<byte> bytes = new List<byte>();
                byte value;
                while ((value = reader.ReadByte()) != 0) bytes.Add(value);
                ulong points = reader.ReadUInt64();
                reader.BaseStream.Seek((long)points * 24L, SeekOrigin.Current);
                result.Add(new View { Name = Encoding.UTF8.GetString(bytes.ToArray()), R = Rotation(qw, qx, qy, qz), T = new[] { tx, ty, tz }, CameraId = cameraId });
            }
            return result;
        }

        static double[] Rotation(double w, double x, double y, double z)
        {
            return new[]
            {
                1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
                2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
                2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y
            };
        }

        static int CameraParameterCount(int model)
        {
            int[] counts = { 3, 4, 4, 5, 8, 8, 12, 5, 4, 5, 12 };
            if (model < 0 || model >= counts.Length) throw new InvalidDataException("Unknown COLMAP model " + model);
            return counts[model];
        }

        // TriBench Vulkan premultiplied-capture patch. The method-aware shaders
        // use front-to-back premultiplied accumulation. Destination alpha must
        // start at zero; Color.black/white both carry alpha one in Unity.
        void PreparePremultipliedCaptureTarget()
        {
            CaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            CaptureCamera.backgroundColor = new Color(0f, 0f, 0f, 0f);
        }

        void CompositePremultipliedBackground(Texture2D image)
        {
            bool white = String.Equals(
                TriAssetRuntimeOptions.Get("-background-color", "black"),
                "white",
                StringComparison.OrdinalIgnoreCase);
            int background = white ? 255 : 0;
            Color32[] pixels = image.GetPixels32();
            int coveredPixels = 0;
            for (int i = 0; i < pixels.Length; ++i)
            {
                Color32 p = pixels[i];
                if (p.a != 0) coveredPixels++;
                int remaining = 255 - p.a;
                p.r = (byte)Mathf.Clamp(p.r + (background * remaining + 127) / 255, 0, 255);
                p.g = (byte)Mathf.Clamp(p.g + (background * remaining + 127) / 255, 0, 255);
                p.b = (byte)Mathf.Clamp(p.b + (background * remaining + 127) / 255, 0, 255);
                p.a = 255;
                pixels[i] = p;
            }
            image.SetPixels32(pixels);
            image.Apply(false, false);
            if (coveredPixels == 0)
                Debug.LogError("[TriBench] Capture target has zero alpha coverage; no procedural geometry reached the camera.");
            else
                Debug.Log($"[TriBench] Capture alpha coverage: {coveredPixels}/{pixels.Length} pixels.");
        }

    }
}
