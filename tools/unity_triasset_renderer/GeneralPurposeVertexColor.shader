Shader "MeshSplatBench/GeneralPurposeVertexColor"
{
    SubShader
    {
        Tags { "Queue"="Geometry" "RenderType"="Opaque" }
        Pass
        {
            Cull Off
            ZWrite On
            ZTest LEqual
            Blend Off
            HLSLPROGRAM
            #pragma vertex Vert
            #pragma fragment Frag
            #include "UnityCG.cginc"
            struct Input { float4 vertex : POSITION; float4 color : COLOR; };
            // Keep Unity's ordinary perspective-correct vertex interpolation.
            // Per-face SH-DC is already de-indexed to three identical colours.
            struct Varyings { float4 position : SV_POSITION; float4 color : COLOR; };
            Varyings Vert(Input input) { Varyings o; o.position=UnityObjectToClipPos(input.vertex); o.color=input.color; return o; }
            float4 Frag(Varyings input) : SV_Target { return float4(input.color.rgb, 1.0); }
            ENDHLSL
        }
    }
}
