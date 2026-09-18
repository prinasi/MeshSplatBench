using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using Unity.Collections;
using UnityEngine;

namespace MeshSplatBench.UnityNative
{
    /// <summary>
    /// Renders video frames along a camera trajectory directly into video synthesis
    /// without saving intermediate PNG images to disk. Streams raw RGB frames
    /// directly to the video encoder or raw binary stream.
    /// </summary>
    [DefaultExecutionOrder(-10000)]
    public sealed class TriAssetVideoBatchCapture : MonoBehaviour
    {
        public TriAssetRenderer Renderer;
        public Camera CaptureCamera;
        public bool AutoStart = true;
        public string TrajectoryPath;
        public string OutputRoot;
        public int OutputWidth = 0;
        public int OutputHeight = 0;
        public string BackgroundColor = "black";
        public int StreamPort = 0;
        public bool WritePngFrames = false;

        const uint StreamMagic = 0x4D534256; // "MSBV" (MeshSplatBench Video)

        [Serializable]
        public sealed class TrajectoryFrame
        {
            public List<float> pose;
            public List<float> intrinsics;
            public List<int> image_size;
            public string file_path;
        }

        [Serializable]
        public sealed class TrajectoryData
        {
            public string format;
            public string camera_model;
            public List<int> image_size;
            public int fps = 30;
            public List<TrajectoryFrame> frames = new List<TrajectoryFrame>();
        }

        bool running;

        void Awake()
        {
            TrajectoryPath = TriAssetRuntimeOptions.Get("-video-trajectory", TrajectoryPath);
            if (string.IsNullOrEmpty(TrajectoryPath))
                TrajectoryPath = TriAssetRuntimeOptions.Get("-trajectory", TrajectoryPath);

            OutputRoot = TriAssetRuntimeOptions.Get("-output", OutputRoot);
            OutputWidth = TriAssetRuntimeOptions.GetInt("-triasset-width", OutputWidth);
            OutputHeight = TriAssetRuntimeOptions.GetInt("-triasset-height", OutputHeight);
            BackgroundColor = TriAssetRuntimeOptions.Get("-background-color", BackgroundColor ?? "black");

            StreamPort = TriAssetRuntimeOptions.GetInt("-video-stream-port", StreamPort);
            if (StreamPort <= 0)
                StreamPort = TriAssetRuntimeOptions.GetInt("-stream-port", StreamPort);

            WritePngFrames = TriAssetRuntimeOptions.Has("-write-frames") || WritePngFrames;

            // Uncap rendering speed for offline video capture
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = -1;
        }

        IEnumerator Start()
        {
            if (AutoStart) yield return RenderTrajectory();
        }

        [ContextMenu("Render Video Trajectory")]
        public void RenderTrajectoryFromContextMenu()
        {
            if (!running) StartCoroutine(RenderTrajectory());
        }

        public IEnumerator RenderTrajectory()
        {
            running = true;
            if (string.IsNullOrEmpty(TrajectoryPath) || !File.Exists(TrajectoryPath))
            {
                Debug.LogError($"[MeshSplatBench] Trajectory file not found: {TrajectoryPath}");
                running = false;
                yield break;
            }

            if (string.IsNullOrEmpty(OutputRoot))
            {
                Debug.LogError("[MeshSplatBench] -output root directory is required for video rendering.");
                running = false;
                yield break;
            }

            string completionPath = Path.Combine(OutputRoot, ".unity_video_complete");
            string legacyCompletionPath = Path.Combine(OutputRoot, ".unity_capture_complete");
            if (File.Exists(completionPath)) File.Delete(completionPath);
            if (File.Exists(legacyCompletionPath)) File.Delete(legacyCompletionPath);

            if (Renderer == null) Renderer = FindFirstObjectByType<TriAssetRenderer>();
            if (CaptureCamera == null) CaptureCamera = Camera.main;
            if (Renderer == null || CaptureCamera == null)
            {
                Debug.LogError("[MeshSplatBench] Video capture requires the renderer and its target camera.");
                running = false;
                yield break;
            }

            while (!Renderer.IsReady) yield return null;

            string jsonContent = File.ReadAllText(TrajectoryPath);
            TrajectoryData trajectory = JsonUtility.FromJson<TrajectoryData>(jsonContent);
            if (trajectory == null || trajectory.frames == null || trajectory.frames.Count == 0)
            {
                trajectory = ParseTrajectoryManually(jsonContent);
            }

            if (trajectory == null || trajectory.frames == null || trajectory.frames.Count == 0)
            {
                Debug.LogError($"[MeshSplatBench] Failed to parse any frames from trajectory: {TrajectoryPath}");
                running = false;
                yield break;
            }

            int defaultWidth = OutputWidth > 0 ? OutputWidth : (trajectory.image_size != null && trajectory.image_size.Count >= 2 ? trajectory.image_size[0] : 1920);
            int defaultHeight = OutputHeight > 0 ? OutputHeight : (trajectory.image_size != null && trajectory.image_size.Count >= 2 ? trajectory.image_size[1] : 1080);
            int fps = trajectory.fps > 0 ? trajectory.fps : 30;
            int frameCount = trajectory.frames.Count;

            // Establish direct stream connection to video encoder if stream port provided
            TcpClient tcpClient = null;
            NetworkStream netStream = null;
            FileStream rawFileStream = null;

            if (StreamPort > 0)
            {
                try
                {
                    tcpClient = new TcpClient();
                    tcpClient.NoDelay = true;
                    tcpClient.Connect("127.0.0.1", StreamPort);
                    netStream = tcpClient.GetStream();
                    Debug.Log($"[MeshSplatBench] Connected to video synthesizer on 127.0.0.1:{StreamPort}");
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[MeshSplatBench] Could not connect to socket on port {StreamPort}: {ex.Message}. Falling back to raw stream file.");
                }
            }

            if (netStream == null)
            {
                string rawPath = Path.Combine(OutputRoot, "render_traj.raw");
                rawFileStream = new FileStream(rawPath, FileMode.Create, FileAccess.Write, FileShare.Read, 65536);
                Debug.Log($"[MeshSplatBench] Streaming raw frames directly to {rawPath}");
            }

            // Send 20-byte stream header: magic (4), width (4), height (4), total_frames (4), fps (4)
            byte[] header = new byte[20];
            Buffer.BlockCopy(BitConverter.GetBytes(StreamMagic), 0, header, 0, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(defaultWidth), 0, header, 4, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(defaultHeight), 0, header, 8, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(frameCount), 0, header, 12, 4);
            Buffer.BlockCopy(BitConverter.GetBytes(fps), 0, header, 16, 4);

            if (netStream != null)
            {
                netStream.Write(header, 0, 20);
                netStream.Flush();
            }
            else if (rawFileStream != null)
            {
                rawFileStream.Write(header, 0, 20);
            }

            RenderTexture target = new RenderTexture(defaultWidth, defaultHeight, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            target.Create();
            Texture2D readback = new Texture2D(defaultWidth, defaultHeight, TextureFormat.RGBA32, false, true);

            bool wasEnabled = CaptureCamera.enabled;
            RenderTexture previousTarget = CaptureCamera.targetTexture;
            CaptureCamera.enabled = false;
            CaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            CaptureCamera.backgroundColor = new Color(0f, 0f, 0f, 0f);
            CaptureCamera.targetTexture = target;

            bool previousRawOutput = Renderer is TriangleSplattingTriAssetRenderer triangle && triangle.OutputRawCodeValues;
            Renderer.SetOutputRawCodeValues(true);

            byte[] rgbBuffer = new byte[defaultWidth * defaultHeight * 3];
            bool whiteBackground = String.Equals(BackgroundColor, "white", StringComparison.OrdinalIgnoreCase);

            Debug.Log($"[MeshSplatBench] Direct Unity video rendering: {frameCount} frames at {defaultWidth}x{defaultHeight}, {fps} FPS (no PNGs).");

            try
            {
                for (int i = 0; i < frameCount; ++i)
                {
                    TrajectoryFrame frame = trajectory.frames[i];
                    int frameWidth = (frame.image_size != null && frame.image_size.Count >= 2) ? frame.image_size[0] : defaultWidth;
                    int frameHeight = (frame.image_size != null && frame.image_size.Count >= 2) ? frame.image_size[1] : defaultHeight;

                    if (frameWidth != defaultWidth || frameHeight != defaultHeight)
                    {
                        defaultWidth = frameWidth;
                        defaultHeight = frameHeight;
                        target.Release();
                        Destroy(target);
                        Destroy(readback);
                        target = new RenderTexture(defaultWidth, defaultHeight, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
                        target.Create();
                        readback = new Texture2D(defaultWidth, defaultHeight, TextureFormat.RGBA32, false, true);
                        CaptureCamera.targetTexture = target;
                        rgbBuffer = new byte[defaultWidth * defaultHeight * 3];
                    }

                    ApplyTrajectoryPose(CaptureCamera, frame, defaultWidth, defaultHeight);
                    Renderer.PrepareCamera(CaptureCamera);

                    CaptureCamera.targetTexture = target;
                    PreparePremultipliedCaptureTarget();
                    CaptureCamera.Render();

                    RenderTexture.active = target;
                    readback.ReadPixels(new Rect(0, 0, defaultWidth, defaultHeight), 0, 0, false);

                    // Use GetRawTextureData instead of GetPixels32 to avoid per-frame managed heap allocation
                    NativeArray<Color32> rawPixels = readback.GetRawTextureData<Color32>();
                    ExtractRgbBytes(rawPixels, defaultWidth, defaultHeight, rgbBuffer, whiteBackground);

                    byte[] frameHeader = BitConverter.GetBytes(i);
                    if (netStream != null)
                    {
                        netStream.Write(frameHeader, 0, 4);
                        netStream.Write(rgbBuffer, 0, rgbBuffer.Length);
                        netStream.Flush();
                    }
                    else if (rawFileStream != null)
                    {
                        rawFileStream.Write(frameHeader, 0, 4);
                        rawFileStream.Write(rgbBuffer, 0, rgbBuffer.Length);
                    }

                    if ((i + 1) % 15 == 0 || i == frameCount - 1)
                    {
                        Debug.Log($"[MeshSplatBench] Rendered & streamed video frame {i + 1}/{frameCount} ({100f * (i + 1) / frameCount:F1}%)");
                    }

                    yield return null;
                }
            }
            finally
            {
                if (netStream != null)
                {
                    try { netStream.Flush(); netStream.Close(); } catch { }
                }
                if (tcpClient != null)
                {
                    try { tcpClient.Close(); } catch { }
                }
                if (rawFileStream != null)
                {
                    try { rawFileStream.Flush(); rawFileStream.Close(); } catch { }
                }
            }

            RenderTexture.active = null;
            CaptureCamera.targetTexture = previousTarget;
            CaptureCamera.enabled = wasEnabled;
            Renderer.SetOutputRawCodeValues(previousRawOutput);

            target.Release();
            Destroy(target);
            Destroy(readback);

            File.WriteAllText(completionPath, "complete\n");
            File.WriteAllText(legacyCompletionPath, "complete\n");

            Debug.Log($"[MeshSplatBench] Unity video trajectory streaming complete: {frameCount} frames directly synthesized.");
            running = false;
            if (!Application.isEditor) Application.Quit(0);
        }

        static void ExtractRgbBytes(NativeArray<Color32> pixels, int width, int height, byte[] rgbBuffer, bool whiteBackground)
        {
            // Unity textures have row 0 at the bottom. Flip vertically so row 0 is at the top for video.
            if (whiteBackground)
            {
                // White background: premultiplied alpha composite with (255,255,255)
                for (int r = 0; r < height; ++r)
                {
                    int srcY = height - 1 - r;
                    int srcRowOffset = srcY * width;
                    int dstRowOffset = r * width * 3;

                    for (int x = 0; x < width; ++x)
                    {
                        Color32 p = pixels[srcRowOffset + x];
                        int remaining = 255 - p.a;
                        int dstIndex = dstRowOffset + x * 3;
                        rgbBuffer[dstIndex + 0] = (byte)Math.Min(p.r + (255 * remaining + 127) / 255, 255);
                        rgbBuffer[dstIndex + 1] = (byte)Math.Min(p.g + (255 * remaining + 127) / 255, 255);
                        rgbBuffer[dstIndex + 2] = (byte)Math.Min(p.b + (255 * remaining + 127) / 255, 255);
                    }
                }
            }
            else
            {
                // Fast path for black background: alpha composite simplifies to identity
                // (background=0, so 0*remaining/255 == 0), just copy RGB channels with Y-flip.
                for (int r = 0; r < height; ++r)
                {
                    int srcY = height - 1 - r;
                    int srcRowOffset = srcY * width;
                    int dstRowOffset = r * width * 3;

                    for (int x = 0; x < width; ++x)
                    {
                        Color32 p = pixels[srcRowOffset + x];
                        int dstIndex = dstRowOffset + x * 3;
                        rgbBuffer[dstIndex + 0] = p.r;
                        rgbBuffer[dstIndex + 1] = p.g;
                        rgbBuffer[dstIndex + 2] = p.b;
                    }
                }
            }
        }

        void ApplyTrajectoryPose(Camera camera, TrajectoryFrame frame, int width, int height)
        {
            List<float> pose = frame.pose;
            if (pose == null || (pose.Count != 12 && pose.Count != 16))
            {
                Debug.LogError("[MeshSplatBench] Invalid pose matrix in trajectory frame; must be 12 or 16 floats.");
                return;
            }

            Vector3 position = new Vector3(pose[3], pose[7], pose[11]);
            Vector3 right = new Vector3(pose[0], pose[4], pose[8]).normalized;
            Vector3 down = new Vector3(pose[1], pose[5], pose[9]).normalized;
            Vector3 forward = new Vector3(pose[2], pose[6], pose[10]).normalized;

            camera.transform.SetPositionAndRotation(position, Quaternion.LookRotation(forward, -down));
            camera.ResetProjectionMatrix();
            camera.aspect = (float)width / height;

            float fy = 0f;
            float cx = width * 0.5f;
            float cy = height * 0.5f;

            if (frame.intrinsics != null && frame.intrinsics.Count >= 2)
            {
                fy = frame.intrinsics[1];
                if (frame.intrinsics.Count >= 4)
                {
                    cx = frame.intrinsics[2];
                    cy = frame.intrinsics[3];
                }
            }

            if (fy <= 0f)
            {
                camera.fieldOfView = 50f;
            }
            else
            {
                camera.fieldOfView = (float)(2.0 * Math.Atan(height / (2.0 * fy)) * Mathf.Rad2Deg);
            }

            camera.nearClipPlane = 0.01f;
            camera.farClipPlane = 300f;

            Matrix4x4 projection = camera.projectionMatrix;
            projection.m00 = -projection.m00;

            if (Math.Abs(cx - width * 0.5f) > 0.01f || Math.Abs(cy - height * 0.5f) > 0.01f)
            {
                projection.m02 = -((2.0f * cx / width) - 1.0f);
                projection.m12 = 1.0f - (2.0f * cy / height);
            }

            camera.projectionMatrix = projection;
        }

        void PreparePremultipliedCaptureTarget()
        {
            CaptureCamera.clearFlags = CameraClearFlags.SolidColor;
            CaptureCamera.backgroundColor = new Color(0f, 0f, 0f, 0f);
        }

        TrajectoryData ParseTrajectoryManually(string json)
        {
            TrajectoryData data = new TrajectoryData();
            try
            {
                int fpsIdx = json.IndexOf("\"fps\":", StringComparison.OrdinalIgnoreCase);
                if (fpsIdx >= 0)
                {
                    int comma = json.IndexOfAny(new[] { ',', '}', '\n' }, fpsIdx + 6);
                    string num = json.Substring(fpsIdx + 6, comma - (fpsIdx + 6)).Trim();
                    if (int.TryParse(num, out int parsedFps)) data.fps = parsedFps;
                }

                int framesIdx = json.IndexOf("\"frames\"", StringComparison.OrdinalIgnoreCase);
                if (framesIdx < 0) return null;

                int arrStart = json.IndexOf('[', framesIdx);
                if (arrStart < 0) return null;

                int depth = 0;
                int objStart = -1;
                for (int i = arrStart; i < json.Length; ++i)
                {
                    char c = json[i];
                    if (c == '{')
                    {
                        if (depth == 0) objStart = i;
                        depth++;
                    }
                    else if (c == '}')
                    {
                        depth--;
                        if (depth == 0 && objStart >= 0)
                        {
                            string frameObj = json.Substring(objStart, i - objStart + 1);
                            TrajectoryFrame frame = ParseFrameObject(frameObj);
                            if (frame != null) data.frames.Add(frame);
                            objStart = -1;
                        }
                    }
                    else if (c == ']' && depth == 0)
                    {
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogWarning($"[MeshSplatBench] Manual trajectory parse warning: {ex.Message}");
            }
            return data;
        }

        TrajectoryFrame ParseFrameObject(string objJson)
        {
            TrajectoryFrame frame = new TrajectoryFrame();
            frame.pose = ExtractFloatList(objJson, "\"pose\"");
            frame.intrinsics = ExtractFloatList(objJson, "\"intrinsics\"");
            List<float> dims = ExtractFloatList(objJson, "\"image_size\"");
            if (dims != null && dims.Count >= 2)
            {
                frame.image_size = new List<int> { (int)dims[0], (int)dims[1] };
            }
            return (frame.pose != null && (frame.pose.Count == 12 || frame.pose.Count == 16)) ? frame : null;
        }

        List<float> ExtractFloatList(string json, string key)
        {
            int idx = json.IndexOf(key, StringComparison.OrdinalIgnoreCase);
            if (idx < 0) return null;
            int start = json.IndexOf('[', idx);
            if (start < 0) return null;
            int end = json.IndexOf(']', start);
            if (end < 0) return null;
            string raw = json.Substring(start + 1, end - start - 1);
            string[] parts = raw.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            List<float> list = new List<float>(parts.Length);
            foreach (string p in parts)
            {
                if (float.TryParse(p.Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float val))
                    list.Add(val);
            }
            return list;
        }
    }
}
