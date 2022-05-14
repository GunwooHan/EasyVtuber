uniform float4x4 ViewProj<
	bool automatic = true;
	string name = "View Projection Matrix";
>;
uniform texture2d InputA<
	bool automatic = true;
>;

uniform int mode;

sampler_state textureSampler {
	Filter   = Linear;
	AddressU = Clamp;
	AddressV = Clamp;
};

struct VertData {
	float4 pos : POSITION;
	float2 uv : TEXCOORD0;
};

VertData VSDefault(VertData vtx)
{
  vtx.pos = mul(float4(vtx.pos.xyz, 1.0), ViewProj);
	return vtx;
}

float4 PSAlphaFilter(VertData vtx) : TARGET
{
  float2 uv = vtx.uv;
  float3 pixel = InputA.Sample(textureSampler, float2(vtx.uv.x, vtx.uv.y)).rgb;
  float alpha = InputA.Sample(textureSampler, float2(vtx.uv.x + 0.5, vtx.uv.y)).g;
  return float4(clamp(pixel.rgb / alpha, 0.0, 1.0), alpha);
}

technique Draw
{
	pass
	{
		vertex_shader = VSDefault(vtx);
		pixel_shader = PSAlphaFilter(vtx);
	}
}