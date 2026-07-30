// MeshSplatting terminal deployment approximation.  It is selected only when
// the manifest proves opacity_floor >= 0.999 and exp(sigma_logits) <= 0.001.
Shader "TriBench/MeshSplatTerminalSolid"
{
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            Cull Off
            ZWrite On
            ZTest LEqual
            Blend Off
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag
            #include "UnityCG.cginc"

            StructuredBuffer<float> _Positions, _ShDc, _ShRest;
            StructuredBuffer<int> _Indices;
            StructuredBuffer<uint> _Visible;
            int _ShDegree, _ShRestCoefficients;
            float3 _CameraWorldPos;

            float3 Pos(int i) { return float3(_Positions[i*3],_Positions[i*3+1],_Positions[i*3+2]); }
            float3 Dc(int e) { return float3(_ShDc[e*3],_ShDc[e*3+1],_ShDc[e*3+2]); }
            float3 Rest(int e,int c) { int k=(e*_ShRestCoefficients+c)*3; return float3(_ShRest[k],_ShRest[k+1],_ShRest[k+2]); }
            float3 Sh(int e,float3 d) {
                float3 r=.28209479177387814*Dc(e); if(_ShDegree<1)return max(r+.5,0);
                float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z;
                r+=-.4886025119*y*Rest(e,0)+.4886025119*z*Rest(e,1)-.4886025119*x*Rest(e,2);
                if(_ShDegree>1)r+=1.09254843*xy*Rest(e,3)-1.09254843*yz*Rest(e,4)+.315391565*(2*zz-xx-yy)*Rest(e,5)-1.09254843*xz*Rest(e,6)+.546274215*(xx-yy)*Rest(e,7);
                if(_ShDegree>2)r+=-.59004359*y*(3*xx-yy)*Rest(e,8)+2.89061144*xy*z*Rest(e,9)-.4570458*y*(4*zz-xx-yy)*Rest(e,10)+.37317633*z*(2*zz-3*xx-3*yy)*Rest(e,11)-.4570458*x*(4*zz-xx-yy)*Rest(e,12)+1.44530572*z*(xx-yy)*Rest(e,13)-.59004359*x*(xx-3*yy)*Rest(e,14);
                return max(r+.5,0);
            }
            struct V { float4 p:SV_POSITION; noperspective float3 c:TEXCOORD0; };
            V Vert(uint vertexID:SV_VertexID,uint instanceID:SV_InstanceID) {
                V o; uint face=_Visible[instanceID],corner=vertexID; int vi=_Indices[face*3+corner];
                float3 p=Pos(vi); o.p=mul(UNITY_MATRIX_VP,float4(p,1));
                o.c=Sh(vi,normalize(p-_CameraWorldPos)); return o;
            }
            float4 Frag(V i):SV_Target { return float4(i.c,1); }
            ENDHLSL
        }
    }
}
