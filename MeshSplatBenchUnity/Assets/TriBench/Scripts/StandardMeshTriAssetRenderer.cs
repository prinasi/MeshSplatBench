using System;
using System.Collections;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;
using UnityEngine.Rendering;

namespace TriBench.UnityNative
{
    /// <summary>
    /// Unity MeshRenderer deployment path: by default this is a method-agnostic
    /// indexed Mesh baseline with baked DC vertex color; with
    /// -indexed-mesh-method-aware for MeshSplatting it keeps the real Unity
    /// indexed Mesh draw while evaluating the learned SH appearance in shader.
    /// </summary>
    public sealed class StandardMeshTriAssetRenderer : TriAssetRenderer
    {
        public Camera TargetCamera;
        public Shader VertexColorShader;
        [Tooltip("Controlled layout intervention: duplicate every indexed corner into a triangle soup.")]
        public bool DeindexedSoup;

        Mesh mesh;
        Material material;
        ComputeBuffer shDcBuffer, shRestBuffer;
        bool indexedMeshMethodAware, rawCode = true;
        int activeShDegree;
        MeshFilter meshFilter;
        MeshRenderer meshRenderer;
        bool ready;
        string status = "Waiting to load standard Mesh";

        public override bool IsReady => ready;
        public override string Status => status;

        void Awake()
        {
            AssetDirectory = TriAssetRuntimeOptions.Get("-triasset", AssetDirectory);
            string topology = TriAssetRuntimeOptions.Get("-topology", "indexed").ToLowerInvariant();
            DeindexedSoup = topology == "soup" || TriAssetRuntimeOptions.Has("-deindexed-soup");
        }

        IEnumerator Start()
        {
            if (String.IsNullOrWhiteSpace(AssetDirectory) || !Directory.Exists(AssetDirectory))
            {
                Fail("Asset directory does not exist: " + AssetDirectory);
                yield break;
            }
            string buffers = Path.Combine(AssetDirectory, "buffers");
            string positionsPath = Path.Combine(buffers, "positions.bin");
            string indicesPath = Path.Combine(buffers, "indices.bin");
            string dcPath = Path.Combine(buffers, "sh_dc.bin");
            string restPath = Path.Combine(buffers, "sh_rest.bin");
            string manifestPath = Path.Combine(AssetDirectory ?? "", "manifest.json");
            activeShDegree = File.Exists(manifestPath) ? ReadInt(File.ReadAllText(manifestPath), "active_sh_degree", 0) : 0;
            if (indexedMeshMethodAware && (!File.Exists(dcPath) || !File.Exists(restPath)))
            {
                Fail("Indexed MeshSplatting method-aware renderer requires sh_dc and sh_rest buffers.");
                yield break;
            }
            if (!File.Exists(positionsPath) || !File.Exists(indicesPath))
            {
                Fail("Standard Mesh baseline requires positions and indices buffers.");
                yield break;
            }

            status = "Loading indexed positions";
            yield return null;
            float[] positions = ReadFloatArray(positionsPath);
            if (positions.Length % 3 != 0) { Fail("positions.bin is not float3 data."); yield break; }
            int vertexCount = positions.Length / 3;

            status = "Loading indexed triangles";
            yield return null;
            int[] indices = ReadIntArray(indicesPath);
            if (indices.Length % 3 != 0) { Fail("indices.bin is not triangle data."); yield break; }
            int triangleCount = indices.Length / 3;

            status = indexedMeshMethodAware ? "Preparing method-aware indexed MeshSplatting mesh" : (File.Exists(dcPath) ? "Baking SH DC to standard vertex RGB" : "No standard color field; using default white material");
            yield return null;
            Vector3[] vertices = new Vector3[vertexCount];
            Color32[] colors = indexedMeshMethodAware ? null : new Color32[vertexCount];
            const float c0 = 0.28209479177387814f;
            for (int i = 0, p = 0; i < vertexCount; ++i, p += 3)
                vertices[i] = new Vector3(positions[p], positions[p + 1], positions[p + 2]);
            if (!indexedMeshMethodAware && File.Exists(dcPath))
            {
                float[] dc = ReadFloatArray(dcPath);
                if (dc.Length == vertexCount * 3)
                    for (int i = 0, p = 0; i < vertexCount; ++i, p += 3)
                        colors[i] = DcColor(dc, p, c0);
                else if (dc.Length == triangleCount * 3)
                    for (int tri = 0, p = 0; tri < triangleCount; ++tri, p += 3)
                    {
                        Color32 color = DcColor(dc, p, c0);
                        int ib = tri * 3;
                        for (int corner = 0; corner < 3; ++corner)
                        {
                            int vertex = indices[ib + corner];
                            if ((uint)vertex >= (uint)vertexCount) { Fail("indices.bin references an invalid vertex."); yield break; }
                            colors[vertex] = color;
                        }
                    }
                else { Fail("sh_dc.bin is neither per-vertex nor per-triangle RGB data."); yield break; }
                dc = null;
            }
            else if (!indexedMeshMethodAware)
                for (int i = 0; i < vertexCount; ++i) colors[i] = new Color32(255, 255, 255, 255);
            positions = null;
            GC.Collect();

            if (DeindexedSoup)
            {
                status = "De-indexing mesh corners into triangle soup";
                yield return null;
                Vector3[] soupVertices = new Vector3[indices.Length];
                Color32[] soupColors = colors == null ? null : new Color32[indices.Length];
                int[] soupIndices = new int[indices.Length];
                for (int corner = 0; corner < indices.Length; ++corner)
                {
                    int source = indices[corner];
                    if ((uint)source >= (uint)vertexCount) { Fail("indices.bin references an invalid vertex."); yield break; }
                    soupVertices[corner] = vertices[source];
                    if (soupColors != null) soupColors[corner] = colors[source];
                    soupIndices[corner] = corner;
                }
                vertices = soupVertices;
                if (soupColors != null) colors = soupColors;
                indices = soupIndices;
                vertexCount = vertices.Length;
            }

            status = "Uploading standard " + (DeindexedSoup ? "triangle-soup" : "indexed") + " Unity Mesh";
            yield return null;
            mesh = new Mesh { name = "TriBench standard mesh compatibility baseline", indexFormat = IndexFormat.UInt32 };
            mesh.vertices = vertices;
            if (colors != null) mesh.colors32 = colors;
            mesh.SetIndices(indices, MeshTopology.Triangles, 0, false);
            mesh.RecalculateBounds();
            vertices = null;
            colors = null;
            indices = null;
            GC.Collect();

            string shaderName = indexedMeshMethodAware ? "TriBench/MeshSplatIndexedMesh" : "TriBench/StandardVertexColorRaw";
            VertexColorShader = indexedMeshMethodAware ? (Resources.Load<Shader>("MeshSplatIndexedMesh") ?? Shader.Find(shaderName)) : (VertexColorShader != null ? VertexColorShader : Shader.Find(shaderName));
            if (VertexColorShader == null) { Fail(shaderName + " shader was not found."); yield break; }
            material = new Material(VertexColorShader) { hideFlags = HideFlags.HideAndDontSave };
            if (indexedMeshMethodAware)
            {
                shDcBuffer = Upload(ReadFloatArray(dcPath));
                shRestBuffer = Upload(ReadFloatArray(restPath));
                material.SetBuffer("_ShDc", shDcBuffer);
                material.SetBuffer("_ShRest", shRestBuffer);
                material.SetInt("_ShDegree", activeShDegree);
                material.SetInt("_RawCode", rawCode ? 1 : 0);
                material.SetVector("_CameraWorldPos", TargetCamera != null ? TargetCamera.transform.position : Vector3.zero);
                Debug.Log("[TriBench] Using true indexed MeshSplatting method-aware MeshRenderer path.");
            }
            meshFilter = GetComponent<MeshFilter>();
            if (meshFilter == null) meshFilter = gameObject.AddComponent<MeshFilter>();
            meshRenderer = GetComponent<MeshRenderer>();
            if (meshRenderer == null) meshRenderer = gameObject.AddComponent<MeshRenderer>();
            meshFilter.sharedMesh = mesh;
            meshRenderer.sharedMaterial = material;
            meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
            meshRenderer.receiveShadows = false;
            meshRenderer.lightProbeUsage = LightProbeUsage.Off;
            meshRenderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
            ready = true;
            status = $"Standard {(DeindexedSoup ? "triangle-soup" : "indexed Mesh")} ready: {vertexCount:N0} vertices, {triangleCount:N0} triangles" + (indexedMeshMethodAware ? "; full SH indexed MeshSplatting shader" : (File.Exists(dcPath) ? "; DC vertex color" : "; default white (no standard appearance field)"));
            Debug.Log("[TriBench] " + status);
        }

        static int ReadInt(string json, string key, int fallback) { Match m=Regex.Match(json,"\\\""+key+"\\\"\\s*:\\s*(\\d+)"); return m.Success && Int32.TryParse(m.Groups[1].Value,out int v) ? v : fallback; }
        static ComputeBuffer Upload(Array values) { ComputeBuffer b=new ComputeBuffer(values.Length,4,ComputeBufferType.Raw); b.SetData(values); return b; }
        static byte ToByte(float value) => (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Clamp01(value) * 255f), 0, 255);
        static Color32 DcColor(float[] dc, int offset, float c0) => new Color32(ToByte(c0 * dc[offset] + 0.5f), ToByte(c0 * dc[offset + 1] + 0.5f), ToByte(c0 * dc[offset + 2] + 0.5f), 255);

        static float[] ReadFloatArray(string path)
        {
            long bytes = new FileInfo(path).Length;
            if ((bytes & 3) != 0 || bytes / 4 > Int32.MaxValue) throw new InvalidDataException("Invalid float32 buffer: " + path);
            float[] values = new float[(int)(bytes / 4)];
            ReadInto(path, values, (int)bytes);
            return values;
        }

        static int[] ReadIntArray(string path)
        {
            long bytes = new FileInfo(path).Length;
            if ((bytes & 3) != 0 || bytes / 4 > Int32.MaxValue) throw new InvalidDataException("Invalid int32 buffer: " + path);
            int[] values = new int[(int)(bytes / 4)];
            ReadInto(path, values, (int)bytes);
            return values;
        }

        static void ReadInto(string path, Array destination, int byteLength)
        {
            const int blockSize = 4 * 1024 * 1024;
            byte[] block = new byte[blockSize];
            using FileStream stream = File.OpenRead(path);
            int offset = 0;
            while (offset < byteLength)
            {
                int requested = Math.Min(block.Length, byteLength - offset);
                int count = stream.Read(block, 0, requested);
                if (count <= 0) throw new EndOfStreamException(path);
                Buffer.BlockCopy(block, 0, destination, offset, count);
                offset += count;
            }
        }

        void Fail(string message)
        {
            status = "ERROR: " + message;
            Debug.LogError("[TriBench] " + message);
        }

        void OnDestroy()
        {
            ready = false;
            if (meshFilter != null) meshFilter.sharedMesh = null;
            if (meshRenderer != null) meshRenderer.sharedMaterial = null;
            if (mesh != null) Destroy(mesh);
            shDcBuffer?.Release();
            shRestBuffer?.Release();
            shDcBuffer = null;
            shRestBuffer = null;
            if (material != null) Destroy(material);
        }
    }
}
