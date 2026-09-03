// Method-independent Unity Mesh baseline for revision-2 TriAssets.
// It deliberately consumes SH-DC only and refuses assets without a comparable
// colour field; rendering DiffSoup as a white mesh is not a benchmark condition.
using System;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

public sealed class GeneralPurposeMeshTriAssetRenderer : TriAssetRenderer
{
    public Shader vertexColorShader;
    private Mesh mesh;
    private Material material;
    private MeshFilter filter;
    private MeshRenderer meshRenderer;

    public override string MethodName { get { return "general-purpose"; } }

    public override bool SupportsMethod(string method)
    {
        TriAssetLoader loader = GetComponent<TriAssetLoader>();
        return loader != null && loader.Rendering != null
            && loader.Rendering.general_purpose != null
            && loader.Rendering.general_purpose.supported;
    }

    public override void Configure(TriAssetLoader asset)
    {
        ReleaseRuntime();
        if (asset.Rendering == null || asset.Rendering.general_purpose == null
            || !asset.Rendering.general_purpose.supported)
        {
            string reason = asset.Rendering != null && asset.Rendering.general_purpose != null
                ? asset.Rendering.general_purpose.reason : "manifest does not declare support";
            throw new InvalidOperationException("General-purpose Unity renderer is unavailable: " + reason);
        }
        TriAssetBufferEntry positionEntry = asset.Require("positions");
        TriAssetBufferEntry indexEntry = asset.Require("indices");
        TriAssetBufferEntry dcEntry = asset.Require("sh_dc");
        float[] positions = ReadFloat(asset.assetDirectory, positionEntry);
        int[] indices = ReadInt(asset.assetDirectory, indexEntry);
        float[] dc = ReadFloat(asset.assetDirectory, dcEntry);
        int vertexCount = positions.Length / 3;
        int triangleCount = indices.Length / 3;
        if (positions.Length % 3 != 0 || indices.Length % 3 != 0)
            throw new InvalidDataException("positions/indices do not contain float3/int3 records.");

        Vector3[] vertices = new Vector3[vertexCount];
        for (int v = 0; v < vertexCount; ++v)
            vertices[v] = new Vector3(positions[3*v], positions[3*v+1], positions[3*v+2]);
        Color[] colors;
        if (dc.Length == vertexCount * 3)
        {
            colors = new Color[vertexCount];
            for (int v = 0; v < vertexCount; ++v) colors[v] = DcColor(dc, 3*v);
        }
        else if (dc.Length == triangleCount * 3)
        {
            // Per-face colour cannot be stored on shared vertices without
            // overwriting incident faces. De-index exactly once so every
            // triangle retains its declared SH-DC colour.
            Vector3[] soupVertices = new Vector3[indices.Length];
            Color[] soupColors = new Color[indices.Length];
            int[] soupIndices = new int[indices.Length];
            for (int corner = 0; corner < indices.Length; ++corner)
            {
                int source = indices[corner];
                if ((uint)source >= (uint)vertexCount)
                    throw new InvalidDataException("indices reference an invalid vertex.");
                int face = corner / 3;
                soupVertices[corner] = vertices[source];
                soupColors[corner] = DcColor(dc, 3*face);
                soupIndices[corner] = corner;
            }
            vertices = soupVertices; colors = soupColors; indices = soupIndices;
        }
        else
        {
            throw new InvalidDataException("sh_dc is neither per-vertex nor per-face RGB; a fixed-budget appearance bake is required.");
        }

        mesh = new Mesh { name = "TriBench general-purpose opaque mesh", indexFormat = IndexFormat.UInt32 };
        mesh.vertices = vertices; mesh.colors = colors;
        mesh.SetIndices(indices, MeshTopology.Triangles, 0, false);
        mesh.RecalculateBounds();
        vertexColorShader = vertexColorShader != null ? vertexColorShader : Shader.Find("TriBench/GeneralPurposeVertexColor");
        if (vertexColorShader == null) throw new InvalidOperationException("TriBench/GeneralPurposeVertexColor shader is missing.");
        material = new Material(vertexColorShader) { hideFlags = HideFlags.DontSave };
        filter = GetComponent<MeshFilter>() ?? gameObject.AddComponent<MeshFilter>();
        meshRenderer = GetComponent<MeshRenderer>() ?? gameObject.AddComponent<MeshRenderer>();
        filter.sharedMesh = mesh; meshRenderer.sharedMaterial = material;
        meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
        meshRenderer.receiveShadows = false;
        meshRenderer.lightProbeUsage = LightProbeUsage.Off;
        meshRenderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
    }

    private static Color DcColor(float[] dc, int offset)
    {
        const float c0 = 0.28209479177387814f;
        return new Color(
            Mathf.Clamp01(c0*dc[offset]+0.5f),
            Mathf.Clamp01(c0*dc[offset+1]+0.5f),
            Mathf.Clamp01(c0*dc[offset+2]+0.5f),
            1.0f
        );
    }
    private static float[] ReadFloat(string root, TriAssetBufferEntry entry)
    {
        byte[] raw = File.ReadAllBytes(Path.Combine(root, entry.file));
        if ((raw.Length & 3) != 0) throw new InvalidDataException(entry.name);
        float[] values = new float[raw.Length/4]; Buffer.BlockCopy(raw, 0, values, 0, raw.Length); return values;
    }
    private static int[] ReadInt(string root, TriAssetBufferEntry entry)
    {
        byte[] raw = File.ReadAllBytes(Path.Combine(root, entry.file));
        if ((raw.Length & 3) != 0) throw new InvalidDataException(entry.name);
        int[] values = new int[raw.Length/4]; Buffer.BlockCopy(raw, 0, values, 0, raw.Length); return values;
    }
    private void OnDestroy() { ReleaseRuntime(); }
    private void ReleaseRuntime()
    {
        if (filter != null) filter.sharedMesh = null;
        if (meshRenderer != null) meshRenderer.sharedMaterial = null;
        if (mesh != null) Destroy(mesh); if (material != null) Destroy(material);
        mesh = null; material = null;
    }
}
