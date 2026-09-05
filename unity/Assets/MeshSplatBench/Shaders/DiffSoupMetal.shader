Shader "MeshSplatBench/DiffSoupMetal"
{
 SubShader { Tags { "Queue"="Geometry" "RenderType"="Opaque" } Pass {
  Cull Off ZWrite On ZTest LEqual
  CGPROGRAM
  #pragma target 5.0
  #pragma only_renderers metal vulkan
  #pragma vertex Vert
  #pragma fragment Frag
  #include "UnityCG.cginc"
  ByteAddressBuffer _Positions,_Indices,_Features,_Alpha,_W0,_B0,_W2,_B2,_W4,_B4;
  int _PrimitiveCount,_FeatureSamples,_FeatureDim,_Rmax;
  float F(ByteAddressBuffer b,uint i){return asfloat(b.Load(i*4));} uint I(ByteAddressBuffer b,uint i){return b.Load(i*4);}
  float3 Pos(uint i){uint k=i*3;return float3(F(_Positions,k),F(_Positions,k+1),F(_Positions,k+2));}
  float Feature(uint tri,uint sample,uint channel){return F(_Features,(tri*_FeatureSamples+sample)*_FeatureDim+channel);}
  float Alpha(uint tri,uint sample){return F(_Alpha,tri*_FeatureSamples+sample);}
  struct V { float4 p:SV_POSITION; noperspective float3 bary:TEXCOORD0; nointerpolation uint tri:TEXCOORD1; float3 world:TEXCOORD2; };
  V Vert(uint id:SV_VertexID){ V o; uint tri=id/3, c=id-tri*3, k=tri*3; uint vi=I(_Indices,k+c); float3 world=Pos(vi); o.p=mul(UNITY_MATRIX_VP,float4(world,1)); o.world=world; o.bary=c==0?float3(1,0,0):(c==1?float3(0,1,0):float3(0,0,1)); o.tri=tri; return o; }
  // Exact lattice interpolation used in diffsoup_core/cuda/multires.cu at a single accumulated Rmax level.
  void Lattice(uint tri,float b0,float b1,out float feat[7],out float a){
   [unroll] for(int c=0;c<7;c++) feat[c]=0; a=0; uint res=1u<<_Rmax; float rb0=b0*res,rb1=b1*res; uint x=min((uint)floor(rb0),res-1),y=min((uint)floor(rb1),res-1-x); rb0-=x;rb1-=y; bool flip=rb0+rb1>1; uint f=flip?1:0; uint x0=x+1,y0=y,x1=x,y1=y+1,x2=x+f,y2=min(y+f,res-x2); uint n0=x0+y0,n1=x1+y1,n2=x2+y2; uint q0=n0*(n0+1)/2+y0,q1=n1*(n1+1)/2+y1,q2=n2*(n2+1)/2+y2; float w0=flip?1-rb1:rb0,w1=flip?1-rb0:rb1,w2=1-w0-w1; a=w0*Alpha(tri,q0)+w1*Alpha(tri,q1)+w2*Alpha(tri,q2); [unroll] for(int c=0;c<7;c++)feat[c]=w0*Feature(tri,q0,c)+w1*Feature(tri,q1,c)+w2*Feature(tri,q2,c);
  }
  float W(ByteAddressBuffer b,uint row,uint col,uint cols){return F(b,row*cols+col);} 
  float3 Color(float x[16]){ float h0[16],h1[16]; [unroll] for(uint r=0;r<16;r++){float s=F(_B0,r);[unroll]for(uint c=0;c<16;c++)s+=W(_W0,r,c,16)*x[c];h0[r]=max(s,0);} [unroll]for(uint r=0;r<16;r++){float s=F(_B2,r);[unroll]for(uint c=0;c<16;c++)s+=W(_W2,r,c,16)*h0[c];h1[r]=max(s,0);} float3 o;[unroll]for(uint r=0;r<3;r++){float s=F(_B4,r);[unroll]for(uint c=0;c<16;c++)s+=W(_W4,r,c,16)*h1[c];o[r]=1/(1+exp(-s));} return lerp(float3(x[0],x[1],x[2]),o,x[3]); }
  float4 Frag(V i):SV_Target{ float feat[7],a;Lattice(i.tri,i.bary.x,i.bary.y,feat,a);clip(a-.5); float x[16];[unroll]for(int k=0;k<7;k++)x[k]=feat[k]; float3 dir=normalize(_WorldSpaceCameraPos-i.world); float dx=dir.x,dy=dir.y,dz=dir.z,xx=dx*dx,yy=dy*dy,zz=dz*dz; x[7]=.2820947918;x[8]=-.4886025119*dy;x[9]=.4886025119*dz;x[10]=-.4886025119*dx;x[11]=1.09254843*dx*dy;x[12]=-1.09254843*dy*dz;x[13]=.315391565*(2*zz-xx-yy);x[14]=-1.09254843*dx*dz;x[15]=.546274215*(xx-yy);return float4(saturate(Color(x)),1); }
  ENDCG
 } }
}
