Shader "TriBench/MethodSpecificSplat"
{
 Properties { _MeshAblation ("Mesh ablation", Int) = 0 }
 SubShader { Tags { "Queue"="Transparent" "RenderType"="Transparent" } Pass {
  Cull Off ZWrite Off ZTest LEqual Blend OneMinusDstAlpha One
  CGPROGRAM
  #pragma target 5.0
  #pragma only_renderers metal vulkan
  #pragma vertex Vert
  #pragma fragment Frag
  #include "UnityCG.cginc"
  ByteAddressBuffer _Positions,_Indices,_Opacity,_Sigma,_ShDc,_ShRest;
  int _Mode,_PrimitiveCount,_ShDegree,_RawCode,_MeshAblation; float _Gamma,_OpacityFloor; float3 _CameraWorldPos;
  float F(ByteAddressBuffer b,uint i){return asfloat(b.Load(i*4));} uint I(ByteAddressBuffer b,uint i){return b.Load(i*4);}
  float3 Pos(uint i){uint k=i*3;return float3(F(_Positions,k),F(_Positions,k+1),F(_Positions,k+2));}
  float3 Dc(uint e){uint k=e*3;return float3(F(_ShDc,k),F(_ShDc,k+1),F(_ShDc,k+2));}
  float3 Rest(uint e,uint c){uint k=e*45+c*3;return float3(F(_ShRest,k),F(_ShRest,k+1),F(_ShRest,k+2));}
  float3 Sh(uint e,float3 d){ float3 r=0.28209479177387814*Dc(e); if(_ShDegree<1)return max(r+.5,0); float x=d.x,y=d.y,z=d.z,xx=x*x,yy=y*y,zz=z*z,xy=x*y,yz=y*z,xz=x*z; r+=-.4886025119*y*Rest(e,0)+.4886025119*z*Rest(e,1)-.4886025119*x*Rest(e,2); if(_ShDegree>1)r+=1.09254843*xy*Rest(e,3)-1.09254843*yz*Rest(e,4)+.315391565*(2*zz-xx-yy)*Rest(e,5)-1.09254843*xz*Rest(e,6)+.546274215*(xx-yy)*Rest(e,7); if(_ShDegree>2)r+=-.59004359*y*(3*xx-yy)*Rest(e,8)+2.89061144*xy*z*Rest(e,9)-.4570458*y*(4*zz-xx-yy)*Rest(e,10)+.37317633*z*(2*zz-3*xx-3*yy)*Rest(e,11)-.4570458*x*(4*zz-xx-yy)*Rest(e,12)+1.44530572*z*(xx-yy)*Rest(e,13)-.59004359*x*(xx-3*yy)*Rest(e,14); return max(r+.5,0); }
  struct V{float4 p:SV_POSITION;float3 b:TEXCOORD0;nointerpolation float3 c:TEXCOORD1;nointerpolation float a:TEXCOORD2;};
  V Vert(uint id:SV_VertexID){V o;uint t=id/3,corner=id-t*3,ib=t*3;uint i0=I(_Indices,ib),i1=I(_Indices,ib+1),i2=I(_Indices,ib+2),vi=corner==0?i0:(corner==1?i1:i2);float3 p0=Pos(i0),p1=Pos(i1),p2=Pos(i2),p=Pos(vi);o.p=mul(UNITY_MATRIX_VP,float4(p,1));o.b=corner==0?float3(1,0,0):(corner==1?float3(0,1,0):float3(0,0,1));uint e=_Mode==1?vi:t; o.c=(_Mode==1&&_MeshAblation==1)?max(.28209479177387814*Dc(e)+.5,0):Sh(e,normalize((_Mode==1?p:(p0+p1+p2)/3)-_CameraWorldPos));float raw=1/(1+exp(-F(_Opacity,e)));o.a=_Mode==1?_OpacityFloor+(1-_OpacityFloor)*raw:raw;return o;}
  float4 Frag(V i):SV_Target{float cov=1; if(_Mode==2){float ecc=1-3*min(i.b.x,min(i.b.y,i.b.z));cov=exp(-.5*pow(max(ecc,0),2*max(_Gamma,1e-4)));}else if(_Mode==1&&_MeshAblation!=2){float s=exp(clamp(F(_Sigma,0),-20,20));cov=pow(max(3*min(i.b.x,min(i.b.y,i.b.z)),1e-6),s);}float a=min(.99,i.a*cov);clip(a-1.0/255.0);float3 c=_RawCode!=0?i.c:GammaToLinearSpace(i.c);return float4(c*a,a);}
  ENDCG
 } }
}
