Shader "TriBench/TriAssetSplat"
{
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        Pass
        {
            Cull Off
            ZWrite Off
            ZTest LEqual
            Blend SrcAlpha OneMinusSrcAlpha
            HLSLPROGRAM
            #pragma target 4.5
            #pragma vertex Vert
            #pragma fragment Frag
            #include "UnityCG.cginc"

            StructuredBuffer<float> _Positions, _Opacity, _Sigma, _ShDc, _ShRest, _Features;
            StructuredBuffer<int> _Indices;
            StructuredBuffer<uint> _Visible;
            int _RenderMode, _PrimitiveCount, _ShDegree, _ShRestCoefficients, _VertexColor, _FeatureSamples, _FeatureDim, _SigmaPerVertex;
            float3 _CameraWorldPos;
            float _Gamma, _OpacityFloor, _GammaVertexRescale;

            struct Varyings {
                float4 position : SV_POSITION;
                noperspective float3 bary : TEXCOORD0;
                nointerpolation float3 incenterBary : TEXCOORD1;
                noperspective float3 color : TEXCOORD2;
                noperspective float alpha : TEXCOORD3;
                nointerpolation float flatAlpha : TEXCOORD4;
                nointerpolation float sigma : TEXCOORD5;
            };

            float3 Pos(int i) { return float3(_Positions[i * 3], _Positions[i * 3 + 1], _Positions[i * 3 + 2]); }
            float3 ShDc(int offset) { return float3(_ShDc[offset], _ShDc[offset + 1], _ShDc[offset + 2]); }
            float3 ShRest(int offset) { return float3(_ShRest[offset], _ShRest[offset + 1], _ShRest[offset + 2]); }
            float3 Feature(int offset) { return float3(_Features[offset], _Features[offset + 1], _Features[offset + 2]); }
            float Sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

            float3 EvalSh(int entity, float3 p, bool vertexLayout)
            {
                int stride = vertexLayout ? 3 : 1;
                int e = entity;
                float3 r = 0.28209479177387814 * ShDc(e * 3);
                if (_ShDegree == 0) return max(r + 0.5, 0.0);
                float3 d = normalize(p - _CameraWorldPos);
                float x=d.x, y=d.y, z=d.z, xx=x*x, yy=y*y, zz=z*z, xy=x*y, yz=y*z, xz=x*z;
                int baseRest = e * _ShRestCoefficients * 3;
                r += -0.4886025119029199*y*ShRest(baseRest)
                    + 0.4886025119029199*z*ShRest(baseRest+3)
                    - 0.4886025119029199*x*ShRest(baseRest+6);
                if (_ShDegree > 1) {
                    r += 1.0925484305920792*xy*ShRest(baseRest+9)
                       - 1.0925484305920792*yz*ShRest(baseRest+12)
                       + 0.31539156525252005*(2*zz-xx-yy)*ShRest(baseRest+15)
                       - 1.0925484305920792*xz*ShRest(baseRest+18)
                       + 0.5462742152960396*(xx-yy)*ShRest(baseRest+21);
                }
                if (_ShDegree > 2) {
                    r += -0.5900435899266435*y*(3*xx-yy)*ShRest(baseRest+24)
                       + 2.890611442640554*xy*z*ShRest(baseRest+27)
                       - 0.4570457994644658*y*(4*zz-xx-yy)*ShRest(baseRest+30)
                       + 0.3731763325901154*z*(2*zz-3*xx-3*yy)*ShRest(baseRest+33)
                       - 0.4570457994644658*x*(4*zz-xx-yy)*ShRest(baseRest+36)
                       + 1.445305721320277*z*(xx-yy)*ShRest(baseRest+39)
                       - 0.5900435899266435*x*(xx-3*yy)*ShRest(baseRest+42);
                }
                return max(r + 0.5, 0.0);
            }

            Varyings Vert(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
            {
                Varyings o;
                uint face = _Visible[instanceID];
                uint corner = vertexID;
                int i0=_Indices[face*3], i1=_Indices[face*3+1], i2=_Indices[face*3+2];
                int vi = corner == 0 ? i0 : (corner == 1 ? i1 : i2);
                float3 p0=Pos(i0), p1=Pos(i1), p2=Pos(i2);
                if (_RenderMode == 2 && abs(_GammaVertexRescale - 1.0) > 1e-6) {
                    float3 center=(p0+p1+p2)/3.0;
                    p0=center+(p0-center)*_GammaVertexRescale;
                    p1=center+(p1-center)*_GammaVertexRescale;
                    p2=center+(p2-center)*_GammaVertexRescale;
                }
                float3 p=corner == 0 ? p0 : (corner == 1 ? p1 : p2);
                // TriAsset coordinates are world-space by contract.  Keeping this
                // independent of the GameObject transform also matches the culler.
                float4 clip0=mul(UNITY_MATRIX_VP,float4(p0,1));
                float4 clip1=mul(UNITY_MATRIX_VP,float4(p1,1));
                float4 clip2=mul(UNITY_MATRIX_VP,float4(p2,1));
                o.position = corner == 0 ? clip0 : (corner == 1 ? clip1 : clip2);
                o.bary = corner == 0 ? float3(1,0,0) : (corner == 1 ? float3(0,1,0) : float3(0,0,1));
                // Native CUDA defines the incenter after projection.  World-space
                // side lengths are incorrect under perspective and aspect ratio.
                float2 s0=(clip0.xy/max(abs(clip0.w),1e-6))*_ScreenParams.xy;
                float2 s1=(clip1.xy/max(abs(clip1.w),1e-6))*_ScreenParams.xy;
                float2 s2=(clip2.xy/max(abs(clip2.w),1e-6))*_ScreenParams.xy;
                float a=length(s1-s2), b=length(s0-s2), c=length(s0-s1);
                o.incenterBary=float3(a,b,c)/max(a+b+c, 1e-6);
                if (_RenderMode == 3) {
                    int sample = min((int)corner, max(_FeatureSamples-1, 0));
                    int offset=(face*_FeatureSamples+sample)*_FeatureDim;
                    o.color = _FeatureDim >= 3 ? saturate(Feature(offset)) : float3(0.5,0.5,0.5);
                    o.alpha = _Opacity[face*_FeatureSamples+sample];
                    o.sigma = 1;
                } else {
                    bool perVertex = _RenderMode == 1 || (_RenderMode == 2 && _VertexColor != 0);
                    int entity = perVertex ? vi : (int)face;
                    o.color=EvalSh(entity, perVertex ? p : (p0+p1+p2)/3.0, perVertex);
                    // All triangle-family opacity buffers are indexed by face.
                    // MeshSplatting's value is already activated and min-reduced;
                    // Triangle Splatting and 2DTS retain per-face logits.  Do not
                    // reuse the vertex-colour entity for 2DTS opacity.
                    o.alpha=_RenderMode == 1 ? _Opacity[face] : Sigmoid(_Opacity[face]);
                    if (_RenderMode == 0) o.sigma=0.01+exp(_Sigma[face]);
                    else if (_RenderMode == 1) o.sigma=exp(_Sigma[_SigmaPerVertex != 0 ? entity : 0]);
                    else o.sigma=1;
                }
                o.flatAlpha=o.alpha;
                return o;
            }

            float4 Frag(Varyings i) : SV_Target
            {
                if (_RenderMode == 3 && i.alpha < 0.5) discard; // DiffSoup's deterministic benchmark alpha test.
                // Native triangle/mesh rasterizers use an incenter-normalised soft coverage.
                float phi=min(i.bary.x/max(i.incenterBary.x,1e-6), min(i.bary.y/max(i.incenterBary.y,1e-6), i.bary.z/max(i.incenterBary.z,1e-6)));
                float coverage=pow(saturate(phi), max(i.sigma, 1e-4));
                if (_RenderMode == 2) {
                    // 2DTS: exp(-0.5 * ecc^(2*gamma)), ecc = 1-3*min(bary).
                    float ecc = 1.0 - 3.0 * min(i.bary.x, min(i.bary.y, i.bary.z));
                    coverage = exp(-0.5 * pow(max(ecc, 0.0), 2.0 * max(_Gamma, 1e-4)));
                }
                float baseAlpha=_RenderMode == 1 ? i.flatAlpha : i.alpha;
                float alpha=saturate(baseAlpha * coverage);
                if (alpha < 1.0/255.0) discard;
                return float4(i.color, alpha);
            }
            ENDHLSL
        }
    }
}
