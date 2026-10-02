Shader "MeshSplatBench/StandardVertexColorRaw"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            // COLMAP-to-Unity camera conversion applies a raster-basis reflection.
            // Keep both winding orders visible for imported TSDF meshes so the
            // projection reflection cannot turn valid surfaces into black holes.
            Cull Off
            ZWrite On
            ZTest LEqual
            Blend Off
            CGPROGRAM
            #pragma target 3.5
            #pragma vertex Vert
            #pragma fragment Frag
            #pragma only_renderers metal vulkan
            #include "UnityCG.cginc"
            struct AppData { float4 vertex : POSITION; fixed4 color : COLOR; };
            struct Varyings { float4 position : SV_POSITION; fixed3 color : COLOR; };
            Varyings Vert(AppData v) { Varyings o; o.position = UnityObjectToClipPos(v.vertex); o.color = v.color.rgb; return o; }
            fixed4 Frag(Varyings i) : SV_Target { return fixed4(i.color, 1); }
            ENDCG
        }
    }
}
