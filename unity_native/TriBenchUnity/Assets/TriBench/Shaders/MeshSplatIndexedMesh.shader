// True Unity indexed MeshRenderer path for MeshSplatting topology ablations.
Shader "TriBench/MeshSplatIndexedMesh"
{
 SubShader { Tags { "Queue"="Geometry" "RenderType"="Opaque" } Pass {
  Cull Off ZWrite On ZTest LEqual Blend Off
  CGPROGRAM
  #pragma target 5.0
  #pragma only_renderers metal vulkan
  #pragma vertex Vert
  #pragma fragment Frag
  #include "UnityCG.cginc"
  ByteAddressBuffer _ShDc,_ShRest;
  int _ShDegree,_RawCode; float3 _CameraWorldPos;
  float F(ByteAddressBuffer b,uint i){return asfloat(b.Load(i*4));}
  float3 Dc(uint e){uint k=e*3;return float3(F(_ShDc,k),F(_ShDc,k+1),F(_ShDc,k+2));}
  float3 Rest(uint e,uint c){uint k=e*45+c*3;return float3(F(_ShRest,k),F(_ShRest,k+1),F(_ShRest,k+2));}
  float3 Sh(uint e,float3 d){ float3 r=0.28209479177387814*Dc(e); if(_ShDegree<1)return max(r+.5,0); float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z; r+=-.4886025119*y*Rest(e,0)+.4886025119*z*Rest(e,1)-.4886025119*x*Rest(e,2); if(_ShDegree>1)r+=1.09254843*xy*Rest(e,3)-1.09254843*yz*Rest(e,4)+.315391565*(2*zz-xx-yy)*Rest(e,5)-1.09254843*xz*Rest(e,6)+.546274215*(xx-yy)*Rest(e,7); if(_ShDegree>2)r+=-.59004359*y*(3*xx-yy)*Rest(e,8)+2.89061144*xy*z*Rest(e,9)-.4570458*y*(4*zz-xx-yy)*Rest(e,10)+.37317633*z*(2*zz-3*xx-3*yy)*Rest(e,11)-.4570458*x*(4*zz-xx-yy)*Rest(e,12)+1.44530572*z*(xx-yy)*Rest(e,13)-.59004359*x*(xx-3*yy)*Rest(e,14); return max(r+.5,0); }
  struct V{float4 p:SV_POSITION;noperspective float3 c:TEXCOORD0;};
  V Vert(float4 vertex:POSITION,uint vid:SV_VertexID){V o;float3 world=mul(unity_ObjectToWorld,vertex).xyz;o.p=UnityObjectToClipPos(vertex);o.c=Sh(vid,normalize(world-_CameraWorldPos));return o;}
  float4 Frag(V i):SV_Target{float3 c=_RawCode!=0?i.c:GammaToLinearSpace(i.c);return float4(c,1);}
  ENDCG
 } }
}
